from collections import defaultdict
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
V3: Now path length checking is included.
The main aim of this script is to generate tasks with no explicit ordering of priority,
so that the agent can order the tasks itself. The two goals in a task should ideally vary
significantly in total number of steps for different orderings.

Current implementation: one goal should be in the same room as the starting point.
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
  random_number = random.random()
  if random_number >= 0.5:
    return instruction1 + " " + instruction2
  else:
    return instruction2 + " " + instruction1

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

  is_valid = (len(trajectory) >= single_min * 2) and (len(trajectory) <= single_max * 2)

  return trajectory, is_valid

# Ensure path is above minimum; which is set at 3
def check_path_validity(path, single_min=3):
  return len(path) >= single_min

# Combine two tasks in the same house from original VNLA into a new task with two priorities
def combine_two_tasks(near_task, far_task, path_calculator, start_point, start_region, start_region_name, heading):
  assert near_task['scan'] == far_task['scan'], "Near and Far tasks are from different houses."
  
  new_task = {}

  new_task['scan'] = near_task['scan']
  new_task['start_region'] = start_region
  new_task['start_region_name'] = start_region_name
  new_task['first_end_regions'] = near_task['end_regions']
  new_task['first_end_region_name'] = near_task['end_region_name']
  new_task['second_end_region'] = far_task['end_regions']
  new_task['second_end_region_name'] = far_task['end_region_name']
  new_task['initial_heading'] = heading
  new_task['instruction'] = combine_instructions(near_task['instructions'][0], far_task['instructions'][0])
  new_task['first_object_indices'] = near_task['object_indices']
  new_task['first_object_name'] = near_task['object_name']
  new_task['second_object_indices'] = far_task['object_indices']
  new_task['second_object_name'] = far_task['object_name']
  new_task['start_viewpoint'] = start_point
  new_task['instr_id'] = int(str(near_task['path_id']) + str(far_task['path_id']))

  first_goal_viewpoints = []
  for path in near_task['paths']:
    first_goal_viewpoints.append(path[-1])
  new_task['first_goal_viewpoints'] = first_goal_viewpoints

  second_goal_viewpoints = []
  for path in far_task['paths']:
    second_goal_viewpoints.append(path[-1])
  new_task['second_goal_viewpoints'] = second_goal_viewpoints

  # Add valid paths and trajectories whose lengths are within acceptable range
  paths = []
  trajectories = []
  scan = new_task['scan']
  start = new_task['start_viewpoint']
  is_valid = True

  # If near_goal is on the agent's way from starting point to far_goal, then this task is invalid.
  assert len(first_goal_viewpoints) == 1, "first_goal_viewpoints should only have one element."
  assert len(second_goal_viewpoints) == 1, "second_goal_viewpoints should only have one element."
  far_goal_path = path_calculator.paths[scan][start][second_goal_viewpoints[0]]
  if first_goal_viewpoints[0] in far_goal_path:
    is_valid = False
    return new_task, is_valid

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

  # Add two more value items: short_path_length and long_path_length 
  # (short_path_length is the number of steps for start -> near_goal -> far_goal)
  # (long_path_length is the number of steps for start -> far_goal -> near_goal)
  if is_valid:
    near_goal = first_goal_viewpoints[0]
    far_goal = second_goal_viewpoints[0]
    far_trajectory, _ = generate_task_trajectory(new_task, far_goal, near_goal, path_calculator)
    assert len(new_task['trajectories']) == 1, "A valid task should only have one trajectory."
    new_task['short_path_length'] = len(new_task['trajectories'][0])
    new_task['long_path_length'] = len(far_trajectory)
    # If short_path_length is longer than or equal to long_path_length, then the task is invalid.
    if new_task['short_path_length'] >= new_task['long_path_length']:
      is_valid = False

  return new_task, is_valid

# For tasks within the same house, generate (1) a mapping of end_region -> task,
# (2) a mapping of starting_point -> start_region
def generate_mappings_in_house(tasks):
  end_region_mapping = defaultdict(list)
  starting_point_mapping = dict()
  region_name_mapping = dict()
  for task in tasks:
    starting_point = task['paths'][0][0]
    start_region_index = task['start_region']
    start_region_name = task['start_region_name']
    starting_point_mapping[starting_point] = start_region_index
    region_name_mapping[start_region_index] = start_region_name

    # skip task with multiple possible goals
    if len(task['paths']) > 1:
      continue
    assert len(task['end_regions']) == 1
    end_region_index = task['end_regions'][0]
    end_region_mapping[end_region_index].append(task)
  
  return end_region_mapping, starting_point_mapping, region_name_mapping

# Take in original tasks from the same house and output a list of new tasks
# `limit` indicates the max number of resultant data points. Default is 3k but can be increased to ~5k.
def generate_tasks_from_same_house(tasks, path_calculator, limit=3000):
  end_region_mapping, starting_point_mapping, region_name_mapping = generate_mappings_in_house(tasks)
  results = []
  counter = 0
  num_invalid = 0

  end_region_set = set(end_region_mapping.keys())

  for start_task in tasks:
    repeat_counter = 0 # Repeat inner loop for a few times in case most attempts are all invalid
    
    while repeat_counter < 5:
      starting_point = start_task['paths'][0][0]
      start_region = starting_point_mapping[starting_point]
      heading = start_task['heading']
      if start_region not in end_region_mapping or len(end_region_mapping[start_region]) == 0:
        break
      near_task = random.sample(end_region_mapping[start_region], 1)[0]
      random_far_region = random.sample(end_region_set, 1)[0]
      if random_far_region == start_region:
        repeat_counter += 1
        continue
      far_task = random.sample(end_region_mapping[random_far_region], 1)[0]

      new_task, is_new_task_valid = combine_two_tasks(near_task, far_task, path_calculator, \
        start_point=starting_point, start_region=start_region, \
        start_region_name=region_name_mapping[start_region], heading=heading)
      if counter % 500 == 0:
        print("counter: ", counter)
        print("invalid: ", num_invalid)
      if not is_new_task_valid:
        repeat_counter += 1
        num_invalid += 1
      if is_new_task_valid:
        results.append(new_task)
        counter += 1
        break # break out of inner loop and proceed to next task
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
  json_filename = './original_datasets/ori_asknav_train.json' # CHANGE HERE
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
  # all_new_tasks = all_new_tasks[:5000] # NOTE: CHANGE HERE; added this line to restrict size of val/test sets
  print_tasks_stats(all_new_tasks)
  with open('./v3_datasets/v3_asknav_train.json', 'w') as result_file: # CHANGE HERE
    json.dump(all_new_tasks, result_file, indent=4, sort_keys=True)


if __name__ == "__main__":
  main()
