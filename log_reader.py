import json

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--file-path", type=str, default=None,
                    help="Log file to be processed.")
args = parser.parse_args()

with open(args.file_path) as f:
    logs = json.load(f)
    axis_sum = [0, 0, 0]
    axis_sum_size = 0
    for log in logs:
        if "axis" in log:
            axis = log["axis"]
            axis_sum[0] += axis[0]
            axis_sum[1] += axis[1]
            axis_sum[2] += axis[2]
            axis_sum_size += 1

    axis_avg = [0, 0, 0]
    axis_avg[0] = axis_sum[0] / axis_sum_size
    axis_avg[1] = axis_sum[1] / axis_sum_size
    axis_avg[2] = axis_sum[2] / axis_sum_size

    print(axis_avg)
