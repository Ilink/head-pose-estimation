import json
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--file-path", type=str, default=None,
                    help="Log file to be processed.")
args = parser.parse_args()

head_x_axis_vals = []
head_y_axis_vals = []
head_z_axis_vals = []
time_in_bad_state_vals = []

with open(args.file_path) as f:
    # logs = json.load(f)
    logs = []
    for line in f:
        if len(line) > 0:
            try:
                json_line = json.loads(line)
                logs.append(json_line)
            except:
                pass

    axis_sum = [0, 0, 0]
    axis_sum_size = 0
    head_forward_diff_sum = 0
    head_forward_diff_sum_size = 0
    back_diff_sum = 0
    back_diff_sum_size = 0

    for log in logs:
        if "head_forward_diff" in log:
            head_forward_diff_sum += log["head_forward_diff"]
            head_forward_diff_sum_size += 1

        if "PoseLandmark.LEFT_HIP" in log and "PoseLandmark.LEFT_SHOULDER" in log:
            diff = log["PoseLandmark.LEFT_SHOULDER"]["pos"][0] - log["PoseLandmark.LEFT_HIP"]["pos"][0]
            back_diff_sum += diff
            back_diff_sum_size += 1

        if "axis" in log:
            axis = log["axis"]
            axis_sum[0] += axis[0]
            head_x_axis_vals.append(axis[0])
            head_y_axis_vals.append(axis[1])
            head_z_axis_vals.append(axis[2])
            axis_sum[1] += axis[1]
            axis_sum[2] += axis[2]
            axis_sum_size += 1

        if "time_in_bad_state" in log:
            time_in_bad_state_vals.append(log["time_in_bad_state"])

    axis_avg = [0, 0, 0]
    if axis_sum_size > 0:
        axis_avg[0] = axis_sum[0] / axis_sum_size
        axis_avg[1] = axis_sum[1] / axis_sum_size
        axis_avg[2] = axis_sum[2] / axis_sum_size
        print(axis_avg)

    if head_forward_diff_sum_size > 0:
        head_forward_diff_avg = head_forward_diff_sum / head_forward_diff_sum_size 
        print("head_forward_diff_avg=", head_forward_diff_avg)

    if back_diff_sum_size > 0:
        back_diff_avg = back_diff_sum / back_diff_sum_size 
        print("back_diff_avg=", back_diff_avg)




# plt.plot(time_in_bad_state_vals, label='time in bad state')
plt.plot(head_x_axis_vals, label='xaxis')
plt.plot(head_y_axis_vals, label='yaxis')
# plt.plot(head_z_axis_vals, label='zaxis')
plt.legend()
out_dir = os.path.dirname(args.file_path)
plt.savefig(os.path.join(out_dir, 'out.png'))
