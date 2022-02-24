import json
import random

import os
import math
import networkx as nx
import functools
import scipy.stats
import random
import sys
import copy
import numpy as np

import torch

import utils
sys.path.append('../../build')
import MatterSim

'''
This file generates a multi-priority dataset from the original VNLA dataset.
V2: Now the natural-language priority descriptions will be more varied.
Also, path length checking is included.
'''

class PathCalculator(object):
  ''' Generate paths for tasks '''

  def __init__(self):
      self.scans = set()
      self.graph = {}
      self.paths = {}
      self.distances = {}

  def add_scans(self, scans, path=None):
      new_scans = set.difference(scans, self.scans)
      if new_scans:
          print('Loading navigation graphs for %d scans' % len(new_scans))
          for scan in new_scans:
              graph, paths, distances = self._compute_shortest_paths(scan, path=path)
              self.graph[scan] = graph
              self.paths[scan] = paths
              self.distances[scan] = distances
          self.scans.update(new_scans)  

  def _compute_shortest_paths(self, scan, path=None):
      ''' Load connectivity graph for each scan, useful for reasoning about shortest paths '''
      graph = utils.load_nav_graphs(scan, path=path)
      paths = dict(nx.all_pairs_dijkstra_path(graph))
      distances = dict(nx.all_pairs_dijkstra_path_length(graph))
      return graph, paths, distances

def group_tasks_by_house(filename):
  f = open(filename) 
  json_data = json.load(f)

  tasks_by_house = dict() # `scan` -> list of tasks
  for task in json_data:
    scan = task['scan']
    if scan not in tasks_by_house:
      tasks_by_house[scan] = []
    tasks_by_house[scan].append(task)
  
  f.close()
  return tasks_by_house

def combine_instructions(instruction1, instruction2):
  # Randomly generate different descriptions of priorities
  version_1 = "first " + instruction1 + " then " + instruction2
  version_2 = instruction2 + " but you should urgently " + instruction1
  version_3 = "it is urgent to " + instruction1 + " afterwards " + instruction2
  version_4 = "before you " + instruction2 + " please first " + instruction1
  version_5 = "hurry up and " + instruction1 + " while you are at it , " + instruction2
  version_6 = "there is no rush to " + instruction2 + " but please urgently " + instruction1
  version_7 = "it is very important to " + instruction1 + " but not so important to " + instruction2
  version_8 = "please " + instruction2 + " but more urgently , please first " + instruction1
  version_9 = "you should first " + instruction1 + " afterwards " + instruction2
  version_10 = "take your time to " + instruction2 + " but first , please " + instruction1

  result = random.choice([version_1, version_2, version_3, version_4, version_5, \
    version_6, version_7, version_8, version_9, version_10])

  return result

# Combine two tasks in the same house from original VNLA into a new task with two priorities
def combine_two_tasks(task1, task2, path_calculator):
  assert task1['scan'] == task2['scan'], "Tasks 1 and 2 are from different houses."
  
  new_task = {}
  new_task['scan'] = task1['scan']
  new_task['start_region'] = task1['start_region']
  new_task['start_region_name'] = task1['start_region_name']
  new_task['first_end_regions'] = task1['end_regions']
  new_task['first_end_region_name'] = task1['end_region_name']
  new_task['second_end_region'] = task2['end_regions']
  new_task['second_end_region_name'] = task2['end_region_name']
  new_task['initial_heading'] = task1['heading']
  new_task['instruction'] = combine_instructions(task1['instructions'][0], task2['instructions'][0])
  new_task['first_object_indices'] = task1['object_indices']
  new_task['first_object_name'] = task1['object_name']
  new_task['second_object_indices'] = task2['object_indices']
  new_task['second_object_name'] = task2['object_name']
  new_task['start_viewpoint'] = task1['paths'][0][0]
  new_task['instr_id'] = int(str(task1['path_id']) + str(task2['path_id']))

  first_goal_viewpoints = []
  for path in task1['paths']:
    first_goal_viewpoints.append(path[-1])
  new_task['first_goal_viewpoints'] = first_goal_viewpoints

  second_goal_viewpoints = []
  for path in task2['paths']:
    second_goal_viewpoints.append(path[-1])
  new_task['second_goal_viewpoints'] = second_goal_viewpoints

  # Add paths
  paths = []
  scan = new_task['scan']
  start = new_task['start_viewpoint']
  for first_goal in first_goal_viewpoints:
    first_leg = path_calculator.paths[scan][start][first_goal]
    for second_goal in second_goal_viewpoints:
      second_leg = path_calculator.paths[scan][first_goal][second_goal]
      path = first_leg + second_leg[1:]
      paths.append(path)
  new_task['paths'] = paths

  return new_task

# Take in original tasks from the same house and output a list of new tasks
# `limit` indicates the max number of resultant data points. Default is 3k but can be increased to ~4k.
def generate_tasks_from_same_house(tasks, path_calculator, limit=3000):
  results = []
  counter = 0
  for i in range(len(tasks) - 1):
    for j in range(i + 1, len(tasks)):
      new_task = combine_two_tasks(tasks[i], tasks[j], path_calculator)
      results.append(new_task)
      counter += 1
      if counter >= limit:
        return results
  return results

# Print statistics about this dataset of tasks
def print_tasks_stats(all_tasks):
  print(f'There are {len(all_tasks)} tasks in total.')

  # Calculate stats for paths
  paths = []
  for task in all_tasks:
    paths.extend(task['paths'])
  path_lengths = [len(path) for path in paths]
  max_path_length = max(path_lengths)
  min_path_length = min(path_lengths)
  average_path_length = float(sum(path_lengths))/float(len(path_lengths))
  print(f'Maximum path length: {max_path_length}')
  print(f'Minimum path length: {min_path_length}')
  print(f'Average path length: {average_path_length}')

# Return a PathCalculator to generate paths for tasks
def setup(filename):

  # Get a set of scans in the dataset
  f = open(filename) 
  json_data = json.load(f)

  scans = set()
  for task in json_data:
    scan = task['scan']
    scans.add(scan)
  
  f.close()

  path_calculator = PathCalculator()
  path_calculator.add_scans(scans)

  return path_calculator

def main():

  # Step One: group tasks in the same house (identified by `scan`)
  json_filename = './ori_asknav_test_unseen.json' # CHANGE HERE
  tasks_by_house = group_tasks_by_house(json_filename) # `scan` -> list of tasks

  # Step 1.5: set up PathCalculator
  path_calculator = setup(json_filename)

  # Step Two: construct a task from every pair of tasks from the same house/scan
  all_new_tasks = []
  for house, tasks in tasks_by_house.items():
    print(f'Now processing {house}')
    new_tasks = generate_tasks_from_same_house(tasks, path_calculator)
    all_new_tasks += new_tasks

  # Step Three: randomise ordering of new tasks and write to new file
  random.shuffle(all_new_tasks)
  all_new_tasks = all_new_tasks[:5000] # NOTE: CHANGE HERE; added this line to restrict size of val/test sets
  print_tasks_stats(all_new_tasks)
  with open('multipri_asknav_test_unseen.json', 'w') as result_file: # CHANGE HERE
    json.dump(all_new_tasks, result_file, indent=4, sort_keys=True)


if __name__ == "__main__":
  main()
