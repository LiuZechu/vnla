import json
import random

'''
This script combines the Multi-Priority Dataset and Original VNLA Dataset
according to a specified proportion to generate a new *training* dataset.
Note that val and test sets are still the same since the mixture is for
the purpose of diversifying training only. 
'''

def print_stats(dataset):
  total_num = 0
  original_num = 0
  mp_num = 0
  for task in dataset:
    total_num += 1
    if 'first_goal_viewpoints' in task:
      mp_num += 1
    else:
      original_num += 1
  
  print(f'There are {total_num} tasks in total.')
  print(f'There are {original_num} original tasks.')
  print(f'There are {mp_num} multi-priority tasks.')

def combine(dataset1, dataset2, proportion, total_num):
  num_from_dataset1 = total_num * proportion
  num_from_dataset2 = total_num - num_from_dataset1

  new_dataset = []
  new_dataset.extend(random.sample(dataset1, num_from_dataset1))
  new_dataset.extend(random.sample(dataset2, num_from_dataset2))
  
  random.shuffle(new_dataset)

  return new_dataset

def main():
  original_filename = "./ori_asknav_train.json"
  mp_filename = "./mp_asknav_train.json"
  ori_proportion = 0.5 # CHANGE HERE (proportion of original tasks out of 1.0)
  total_num = 100000 # CHANGE HERE

  # Reading Original dataset
  f1 = open(original_filename) 
  original_data = json.load(f1)
  f1.close()

  # Reading MP dataset
  f2 = open(mp_filename)
  mp_data = json.load(f2)
  f2.close()

  new_dataset = combine(original_data, mp_data, ori_proportion, total_num)
  print_stats(new_dataset) # For sanity check

  # Writing new dataset to file
  output_filename = "./asknav_train.json"
  with open(output_filename, 'w') as result_file: 
    json.dump(new_dataset, result_file, indent=4, sort_keys=True)


if __name__ == "__main__":
  main()

