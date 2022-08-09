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
from playsound import playsound
# import simpleaudio

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
args = parser.parse_args()

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

# beep = simpleaudio.WaveObject.from_wave_file("path/to/file.wav")

if __name__ == '__main__':
    handler = SIGINT_handler()
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
    out_video_path = os.path.join(out_dir, timestamp_str + ".mp4")
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

    # 4. Measure the performance with a tick meter.
    # tm = cv2.TickMeter()

    # out_video_path = "/mnt/c/Users/ian/Documents/ergonomics/7_17_22/test/temp.mp4"

    # video_out_base_dir = os.path.splitext(video_src)[0] + "_frames"
    # print(video_out_base_dir)
    # try:
    #     shutil.rmtree(video_out_base_dir)
    # except OSError as e:
    #     pass
    # os.mkdir(video_out_base_dir)

    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps=%d width=%f height=%f" % (fps, width, height))
    out_size = (int(width), int(height))
    # out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"MJPG"), int(fps), out_size)
    out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), int(fps), out_size)
    # out_video = cv2.VideoWriter(out_video_path, -1, int(fps), out_size)

    frame_idx = 0
    skip_frame_count = 0
    num_skip_frames = 0
    if not (args.preview or args.sample_all):
        # My webcam is 30 fps, so this is 2 samples per second
        num_skip_frames = 15

    font_color = (0, 0, 255)
    # font_color = (57, 143, 247)

    logs = []

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
            # tm.start()
            marks = mark_detector.detect_marks(face_img)
            # tm.stop()

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

            # Do you want to see the head axes?
            # I dont know why they're stored this way
            # They're also not normalized, so we'll have to fix that
            axis = [pose[0][0][0], pose[0][1][0], pose[0][2][0]]
            normalize_vec3(axis)
            # annotation_str = "axis=" + str(format_number(axis[0])) + ",\n" + str(format_number(axis[1])) + ",\n" + str(format_number(axis[2]))
            annotation_str = "axis=%f, %f, %f" % (format_number(axis[0]), format_number(axis[1]), format_number(axis[2]))
            pose_estimator.draw_axes(frame, pose[0], pose[1])

            cv2.putText(frame, annotation_str, (50, int(height)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, font_color, 2, cv2.LINE_AA)

            log["axis"] = axis
            logs.append(log)

            # playsound(os.path.join(Path.cwd(), "assets", "beep2.wav"))

            # Do you want to see the marks?
            # mark_detector.draw_marks(frame, marks, color=(0, 255, 0))

            # Do you want to see the facebox?
            # mark_detector.draw_box(frame, [facebox])

        if args.preview:
            cv2.imshow("Preview", frame)
            cv2.waitKey(1)

        out_frame_path = os.path.join(frames_out_base_dir, "frame_%d.png" % frame_idx) 
        cv2.imwrite(out_frame_path, frame)
        out_video.write(cv2.resize(frame, out_size))

        frame_idx += 1

    out_video.release()
    cap.release()
    out_log_path = os.path.join(out_dir, timestamp_str + ".json")
    with open(out_log_path, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)
