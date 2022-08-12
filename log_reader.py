import json
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--file-path", type=str, default=None,
                    help="Log file to be processed.")
args = parser.parse_args()

head_x_axis_vals = []
head_y_axis_vals = []
head_z_axis_vals = []

with open(args.file_path) as f:
    logs = json.load(f)
    axis_sum = [0, 0, 0]
    axis_sum_size = 0
    for log in logs:
        if "axis" in log:
            axis = log["axis"]
            axis_sum[0] += axis[0]
            head_x_axis_vals.append(axis[0])
            head_y_axis_vals.append(axis[1])
            head_z_axis_vals.append(axis[2])
            axis_sum[1] += axis[1]
            axis_sum[2] += axis[2]
            axis_sum_size += 1

    axis_avg = [0, 0, 0]
    axis_avg[0] = axis_sum[0] / axis_sum_size
    axis_avg[1] = axis_sum[1] / axis_sum_size
    axis_avg[2] = axis_sum[2] / axis_sum_size


    print(axis_avg)


plt.plot(head_x_axis_vals, label='xaxis')
plt.plot(head_y_axis_vals, label='yaxis')
# plt.plot(head_z_axis_vals, label='zaxis')
plt.legend()
plt.savefig('out.png')
