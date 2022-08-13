"""Demo code showing how to estimate human head pose.

There are three major steps:
1. Detect human face in the video frame.
2. Run facial landmark detection on the face image.
3. Estimate the pose by solving a PnP problem.

To find more details, please refer to:
https://github.com/yinguobing/head-pose-estimation
"""
from argparse import ArgumentParser

import cv2

from mark_detector import MarkDetector
from pose_estimator import PoseEstimator

import math
import shutil
import os 
import datetime
import signal
from dateutil.tz import tzlocal
from pathlib import Path
import json
import time
from playsound import playsound
import sys

print(__doc__)
print("OpenCV version: {}".format(cv2.__version__))

# Parse arguments from user input.
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=None,
                    help="The webcam index.")
parser.add_argument("--outdir", type=str, default=None,
                    help="The output directory - not a path to a specific file.")
parser.add_argument("--preview", action="store_true", default=False,
                    help="Whether to show a preview. Implies --sample-all")
parser.add_argument("--sample-all", action="store_true", default=False,
                    help="Whether to sample all frames")
parser.add_argument("--record", action="store_true", default=False,
                    help="Whether to record frames and video to disk.")
args = parser.parse_args()

class JsonLogger:
    def __init__(self, out_path):
        self.out_path = out_path
        self.fd = open(self.out_path, 'a', 4096, encoding='utf-8')

    def log(self, msg_obj):
        json.dump(msg_obj, self.fd, ensure_ascii=False, indent=4)

    def __exit__(self, exc_type, exc_value, traceback):
        self.fd.close()

def normalize_vec3(vec):
    mag = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2])
    vec[0] = vec[0] / mag
    vec[1] = vec[1] / mag
    vec[2] = vec[2] / mag

def format_number(num):
    num = num * 1000
    num = round(num)
    num = num / 1000
    return num

class SIGINT_handler():
    def __init__(self):
        self.SIGINT = False

    def signal_handler(self, signal, frame):
        print('Caught sigint, exiting')
        self.SIGINT = True

class PlaySoundInterval:
    def __init__(self, interval_seconds, audio_path):
        self.interval_seconds = interval_seconds
        self.audio_path = audio_path
        self.time_since_last = self.interval_seconds

    def update(self, seconds_since_last_frame):
        self.time_since_last += seconds_since_last_frame

    def play(self):
        if self.time_since_last >= self.interval_seconds:
            playsound(self.audio_path)
            self.time_since_last = 0.0

class AxisTracker():
    def __init__(self, num_seconds_warn, beep_interval_seconds):
        self.num_seconds_warn = num_seconds_warn
        self.in_bad_state = False
        self.time_in_bad_state = 0.0
        self.beeper = PlaySoundInterval(beep_interval_seconds, os.path.join(Path.cwd(), "assets", "beep2.wav"))

    def update(self, axis, seconds_since_last_frame, frame_idx, log):
        self.beeper.update(seconds_since_last_frame)
        log["time_in_bad_state"] = self.time_in_bad_state
        if abs(axis[0]) > 0.1:
        # if abs(axis[0]) > 0.05:
            self.time_in_bad_state += seconds_since_last_frame
            self.time_in_bad_state = min(self.num_seconds_warn, self.time_in_bad_state)
        else:
            # One could still be in a bad posture for long periods of time
            # with occasional resets to a good posture. An extreme
            # example would be 99% of the interval in a bad posture,
            # then 1% in a good posture.
            self.time_in_bad_state -= seconds_since_last_frame
            self.time_in_bad_state = max(0, self.time_in_bad_state)

        if self.time_in_bad_state >= self.num_seconds_warn:
            log["in_bad_state"] = True
            log["time_in_bad_state"] = self.time_in_bad_state 
            self.beeper.play()

if __name__ == '__main__':
    handler = SIGINT_handler()
    axis_tracker = AxisTracker(5, 2)
    signal.signal(signal.SIGINT, handler.signal_handler)

    # Before estimation started, there are some startup works to do.

    # 1. Setup the video source from webcam or video file.
    video_src = args.cam if args.cam is not None else args.video
    if video_src is None:
        print("Video source not assigned, default webcam will be used.")
        video_src = 0

    out_base_dir = args.outdir
    if out_base_dir is None:
        out_base_dir = os.path.join(Path.cwd(), "video")
        if not os.path.isdir(out_base_dir):
            os.mkdir(out_base_dir)

    now = datetime.datetime.now(tzlocal())
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S") 
    out_dir = os.path.join(out_base_dir, timestamp_str)
    os.mkdir(out_dir)

    out_log_path = os.path.join(out_dir, "log.json")
    logger = JsonLogger(out_log_path)

    out_video_path = os.path.join(out_dir, "recording.mp4")
    print(out_video_path)

    frames_out_base_dir = os.path.join(out_dir, "frames")
    os.mkdir(frames_out_base_dir)

    cap = cv2.VideoCapture(video_src)

    # Get the frame size. This will be used by the pose estimator.
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # 2. Introduce a pose estimator to solve pose.
    pose_estimator = PoseEstimator(img_size=(height, width))

    # 3. Introduce a mark detector to detect landmarks.
    mark_detector = MarkDetector()

    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps=%d width=%f height=%f" % (fps, width, height))
    out_size = (int(width), int(height))
    if args.record:
        out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), int(fps), out_size)

    frame_idx = 0
    skip_frame_count = 0
    num_skip_frames = 0
    # This is kind of a rough estimate since in preview mode it doesnt really
    # run at the same framerate of the webcam.
    seconds_per_frame = 1.0 / 30.0
    seconds_per_sample = seconds_per_frame
    if not (args.preview or args.sample_all):
        # My webcam is 30 fps, so this is 2 samples per second
        num_skip_frames = 5
        seconds_per_sample = seconds_per_frame * num_skip_frames
    font_color = (0, 0, 255)
    # font_color = (57, 143, 247)

    prev_time = time.perf_counter()

    # Now, let the frames flow.
    while not handler.SIGINT:
        log = {}
        # if frame_idx == 100:
        #     break

        # Read a frame.
        frame_got, frame = cap.read()
        if frame_got is False:
            break

        if skip_frame_count < num_skip_frames:
            skip_frame_count += 1
            continue
        else:
            skip_frame_count = 0

        # If the frame comes from webcam, flip it so it looks like a mirror.
        if video_src == 0:
            frame = cv2.flip(frame, 2)

        # Step 1: Get a face from current frame.
        facebox = mark_detector.extract_cnn_facebox(frame)

        # Any face found?
        if facebox is not None:

            # Step 2: Detect landmarks. Crop and feed the face area into the
            # mark detector.
            x1, y1, x2, y2 = facebox
            face_img = frame[y1: y2, x1: x2]

            # Run the detection.
            marks = mark_detector.detect_marks(face_img)

            # Convert the locations from local face area to the global image.
            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1

            # Try pose estimation with 68 points.
            pose = pose_estimator.solve_pose_by_68_points(marks)

            # All done. The best way to show the result would be drawing the
            # pose on the frame in realtime.

            # Do you want to see the pose annotation?
            pose_estimator.draw_annotation_box(
                frame, pose[0], pose[1], color=(0, 255, 0))

            # The first entry is a rotation vector, the second is translation
            axis = [pose[0][0][0], pose[0][1][0], pose[0][2][0]]
            normalize_vec3(axis)
            annotation_str = "axis=%f, %f, %f" % (format_number(axis[0]), format_number(axis[1]), format_number(axis[2]))
            pose_estimator.draw_axes(frame, pose[0], pose[1])

            cv2.putText(frame, annotation_str, (50, int(height)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, font_color, 2, cv2.LINE_AA)

            log["axis"] = axis

            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            axis_tracker.update(axis, dt, frame_idx, log)

            logger.log(log)

            # Do you want to see the marks?
            # mark_detector.draw_marks(frame, marks, color=(0, 255, 0))

            # Do you want to see the facebox?
            # mark_detector.draw_box(frame, [facebox])

        if args.preview:
            cv2.imshow("Preview", frame)
            cv2.waitKey(1)

        if args.record:
            out_frame_path = os.path.join(frames_out_base_dir, "frame_%d.png" % frame_idx) 
            cv2.imwrite(out_frame_path, frame)
            out_video.write(cv2.resize(frame, out_size))

        frame_idx += 1

    if args.record:
        out_video.release()
    cap.release()
