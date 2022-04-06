import json
import math


def main():
  input_filename = "mp_asknav_train.json" # CHANGE HERE
  f = open(input_filename)
  json_data = json.load(f)
  total_length = 0
  total_num = 0
  traj_lengths = [0] * 6
  for task in json_data:
    for i in range(len(task['trajectories'])):
      traj_len = len(task['trajectories'][i])
      total_length += traj_len
      total_num += 1
      traj_lengths[math.floor(traj_len/float(10))] += 1

  ave_length = float(total_length) / float(total_num)
  print(f'number of data points: {len(json_data)}')
  print(f'average trajectory length: {ave_length}')
  print('distribution of trajectory lengths: ', traj_lengths)

  f.close()

if __name__ == "__main__":
  main()

