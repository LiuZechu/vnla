import json
import random

import os
import math
import networkx as nx
import functools
import scipy.stats
import sys
import numpy as np

import utils
sys.path.append('../../build')
import MatterSim

import utils
sys.path.append('../../build')

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

  def init_sim(self):
    self.sim = MatterSim.Simulator()
    self.sim.setRenderingEnabled(False)
    self.sim.setDiscretizedViewingAngles(True)
    self.sim.setCameraResolution(640, 480)
    self.sim.setCameraVFOV(math.radians(60))
    self.sim.setNavGraphPath(
        os.path.join(os.getenv('PT_DATA_DIR', '../../../data'), 'connectivity'))
    self.sim.init()

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

  def _find_nearest_point(self, scan, start_point, end_points):
      best_d = 1e9
      best_point = None

      for end_point in end_points:
          d = self.distances[scan][start_point][end_point]
          if d < best_d:
              best_d = d
              best_point = end_point
      return best_d, best_point

  def _shortest_path_action(self, ob):
      ''' Determine next action on the shortest path to goals. '''

      scan = ob['scan']
      start_point = ob['viewpoint']

      # Find nearest goal
      _, goal_point = self._find_nearest_point(scan, start_point, ob['goal_viewpoints'])

      # Stop if a goal is reached
      if start_point == goal_point:
          return (0, 0, 0)

      path = self.paths[scan][start_point][goal_point]
      next_point = path[1]

      # Can we see the next viewpoint?
      for i, loc in enumerate(ob['navigableLocations']):
          if loc.viewpointId == next_point:
              # Look directly at the viewpoint before moving
              if loc.rel_heading > math.pi/6.0:
                    return (0, 1, 0) # Turn right
              elif loc.rel_heading < -math.pi/6.0:
                    return (0,-1, 0) # Turn left
              elif loc.rel_elevation > math.pi/6.0 and ob['viewIndex'] // 12 < 2:
                    return (0, 0, 1) # Look up
              elif loc.rel_elevation < -math.pi/6.0 and ob['viewIndex'] // 12 > 0:
                    return (0, 0,-1) # Look down
              else:
                    return (i, 0, 0) # Move

      # Can't see it - first neutralize camera elevation
      if ob['viewIndex'] // 12 == 0:
          return (0, 0, 1) # Look up
      elif ob['viewIndex'] // 12 == 2:
          return (0, 0,-1) # Look down

      # Otherwise decide which way to turn
      target_rel = self.graph[ob['scan']].nodes[next_point]['position'] - ob['point']
      target_heading = math.pi / 2.0 - math.atan2(target_rel[1], target_rel[0])
      if target_heading < 0:
          target_heading += 2.0 * math.pi
      if ob['heading'] > target_heading and ob['heading'] - target_heading < math.pi:
          return (0, -1, 0) # Turn left
      if target_heading > ob['heading'] and target_heading - ob['heading'] > math.pi:
          return (0, -1, 0) # Turn left

      return (0, 1, 0) # Turn right

  def _shortest_path_actions(self, ob):
    actions = []
    
    assert not ob['ended']

    counter = 0 # For debugging
    while not ob['ended']:
      # For debugging
      counter += 1
      if counter == 500:
        print("500 iterations reached without getting to the final goal.")
        break

      # Query oracle for next action
      action = self._shortest_path_action(ob)
      # Take action
      self.sim.makeAction(*action)
      # Record action
      actions.append(list(action))

      reached_first_goal = ob['reached_first_goal']
      if action == (0, 0, 0) and not ob['reached_first_goal']:
          ob['reached_first_goal'] = True
          ob['goal_viewpoints'] = ob['second_goal_viewpoints']
      elif action == (0, 0, 0) and ob['reached_first_goal']:
          break

      state = self.sim.getState()

      ob['viewpoint'] = state.location.viewpointId
      ob['viewIndex'] = state.viewIndex
      ob['heading'] = state.heading
      ob['elevation'] = state.elevation
      ob['navigableLocations'] = state.navigableLocations
      ob['point'] = state.location.point
      ob['ended'] = ob['ended'] or (action == (0, 0, 0) and reached_first_goal) # Problem is here!

    return actions

  def simulate(self, datapoint):
    # Start simulator
    scanId = datapoint['scan']
    viewpointId = datapoint['start_viewpoint']
    heading = datapoint['initial_heading']

    self.init_sim()
    self.sim.newEpisode(scanId, viewpointId, heading, 0)

    # Get obs
    state = self.sim.getState()
    ob = {
      'instr_id' : datapoint['instr_id'],
      'scan' : state.scanId,
      'point': state.location.point,
      'viewpoint' : state.location.viewpointId,
      'viewIndex' : state.viewIndex,
      'heading' : state.heading,
      'elevation' : state.elevation,
      # 'feature' : feature,
      'step' : state.step,
      'navigableLocations' : state.navigableLocations,
      # 'instruction' : self.instructions[i],
      'goal_viewpoints': datapoint['first_goal_viewpoints'],
      'first_goal_viewpoints' : datapoint['first_goal_viewpoints'],
      'second_goal_viewpoints' : datapoint['second_goal_viewpoints'],
      'init_viewpoint' : datapoint['start_viewpoint'],
      'reached_first_goal': False,
      'ended': False
    }

    # Start simulation
    actions = self._shortest_path_actions(ob)

    return actions

############################################################################

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
  # version_1 = "first " + instruction1 + " then " + instruction2
  # version_2 = instruction2 + " but you should urgently " + instruction1
  # version_3 = "it is urgent to " + instruction1 + " afterwards " + instruction2
  # version_4 = "before you " + instruction2 + " please first " + instruction1
  # version_5 = "hurry up and " + instruction1 + " while you are at it , " + instruction2
  # version_6 = "there is no rush to " + instruction2 + " but please urgently " + instruction1
  # version_7 = "it is very important to " + instruction1 + " but not so important to " + instruction2
  # version_8 = "please " + instruction2 + " but more urgently , please first " + instruction1
  # version_9 = "you should first " + instruction1 + " afterwards " + instruction2
  # version_10 = "take your time to " + instruction2 + " but first , please " + instruction1

  # result = random.choice([version_1, version_2, version_3, version_4, version_5, \
  #   version_6, version_7, version_8, version_9, version_10])

  # return result

  return "first " + instruction1 + " then " + instruction2

# Ensure the task has a trajectory whose length is within the specified limits
# `single_min` is the minimum length for a single task;
# `single_max` is the maximum length for a single task;
# The overall min/max for a task will be `single_min` * 2 and `single_max` * 2
def generate_task_trajectory(datapoint, first_goal, second_goal, path_calculator, single_min=5, single_max=25):
  # Limit to goal viewpoints to one so as to generate a trajectory
  # for every combination of first+second goals
  task = datapoint.copy()
  task['first_goal_viewpoints'] = [first_goal]
  task['second_goal_viewpoints'] = [second_goal]
  trajectory = path_calculator.simulate(task)

  is_valid = (len(trajectory) - 1 >= single_min * 2) and (len(trajectory) - 1 <= single_max * 2)

  return trajectory, is_valid

# Ensure path is above minimum; which is set at 3
def check_path_validity(path, single_min=3):
  return len(path) >= single_min

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

  # Add valid paths and trajectories whose lengths are within acceptable range
  paths = []
  trajectories = []
  scan = new_task['scan']
  start = new_task['start_viewpoint']
  is_valid = True
  for first_goal in first_goal_viewpoints:
    first_leg = path_calculator.paths[scan][start][first_goal]
    if not is_valid:
      break
    for second_goal in second_goal_viewpoints:
      second_leg = path_calculator.paths[scan][first_goal][second_goal]
      trajectory, is_traj_valid = generate_task_trajectory(new_task, first_goal, second_goal, path_calculator)
      is_path_valid = check_path_validity(first_leg) and check_path_validity(second_leg)
      if not (is_traj_valid and is_path_valid):
        is_valid = False
        break
      path = first_leg + second_leg[1:]
      paths.append(path)
      trajectories.append(trajectory)

  new_task['paths'] = paths
  new_task['trajectories'] = trajectories

  return new_task, is_valid

# Take in original tasks from the same house and output a list of new tasks
# `limit` indicates the max number of resultant data points. Default is 3k but can be increased to ~5k.
def generate_tasks_from_same_house(tasks, path_calculator, limit=3000):
  results = []
  counter = 0
  num_invalid = 0
  for i in range(len(tasks) - 1):
    for j in range(i + 1, len(tasks)):
      new_task, is_new_task_valid = combine_two_tasks(tasks[i], tasks[j], path_calculator)
      if counter % 500 == 0:
        print("counter: ", counter)
        print("invalid: ", num_invalid)
      if not is_new_task_valid:
        num_invalid += 1
      if is_new_task_valid:
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

  # Calculate stats for trajectories
  trajectories = []
  for task in all_tasks:
    trajectories.extend(task['trajectories'])
  traj_lengths = [len(traj) for traj in trajectories]
  max_traj_length = max(traj_lengths)
  min_traj_length = min(traj_lengths)
  average_traj_length = float(sum(traj_lengths))/float(len(traj_lengths))
  print(f'Maximum traj length: {max_traj_length}')
  print(f'Minimum traj length: {min_traj_length}')
  print(f'Average traj length: {average_traj_length}')

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
  json_filename = './original_datasets/ori_asknav_val_seen.json' # CHANGE HERE
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
  with open('./datasets/multipri_asknav_val_seen.json', 'w') as result_file: # CHANGE HERE
    json.dump(all_new_tasks, result_file, indent=4, sort_keys=True)


if __name__ == "__main__":
  main()
