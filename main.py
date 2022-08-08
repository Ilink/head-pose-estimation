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

print(__doc__)
print("OpenCV version: {}".format(cv2.__version__))

# Parse arguments from user input.
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=None,
                    help="The webcam index.")
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

if __name__ == '__main__':
    # Before estimation started, there are some startup works to do.

    # 1. Setup the video source from webcam or video file.
    video_src = args.cam if args.cam is not None else args.video
    if video_src is None:
        print("Video source not assigned, default webcam will be used.")
        video_src = 0

    cap = cv2.VideoCapture(video_src)

    # Get the frame size. This will be used by the pose estimator.
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # 2. Introduce a pose estimator to solve pose.
    pose_estimator = PoseEstimator(img_size=(height, width))

    # 3. Introduce a mark detector to detect landmarks.
    mark_detector = MarkDetector()

    # 4. Measure the performance with a tick meter.
    tm = cv2.TickMeter()

    out_video_path = "/mnt/c/Users/ian/Documents/ergonomics/7_17_22/test/temp.mp4"

    video_out_dir = os.path.splitext(video_src)[0] + "_frames"
    print(video_out_dir)
    try:
        shutil.rmtree(video_out_dir)
    except OSError as e:
        pass
    os.mkdir(video_out_dir)

    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps=%d width=%f height=%f" % (fps, width, height))
    out_size = (int(width), int(height))
    # out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"MJPG"), int(fps), out_size)
    out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"XVID"), int(fps), out_size)

    frame_idx = 0

    font_color = (0, 0, 255)
    # font_color = (57, 143, 247)

    # Now, let the frames flow.
    while True:
        # if frame_idx == 100:
        #     break

        # Read a frame.
        frame_got, frame = cap.read()
        if frame_got is False:
            break

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
            tm.start()
            marks = mark_detector.detect_marks(face_img)
            tm.stop()

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
            annotation_str = "axis=%f,%f,%f" % (format_number(axis[0]), format_number(axis[1]), format_number(axis[2]))
            pose_estimator.draw_axes(frame, pose[0], pose[1])

            cv2.putText(frame, annotation_str, (50, int(height)-100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, font_color, 2, cv2.LINE_AA)

            # Do you want to see the marks?
            # mark_detector.draw_marks(frame, marks, color=(0, 255, 0))

            # Do you want to see the facebox?
            # mark_detector.draw_box(frame, [facebox])

        # Show preview.
        # cv2.imshow("Preview", frame)
        # cv2.imwrite("/mnt/c/Users/ian/Documents/ergonomics/7_17_22/test/frame.png", frame)
        out_frame_path = os.path.join(video_out_dir, "frame_%d.png" % frame_idx) 
        cv2.imwrite(out_frame_path, frame)
        out_video.write(cv2.resize(frame, out_size))
        if cv2.waitKey(1) == 27:
            break

        frame_idx += 1

    out_video.release()
    cap.release()
