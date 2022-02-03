import json
import random

'''
This file generates a multi-priority dataset from the original VNLA dataset
'''

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
  return "first " + instruction1 + " then " + instruction2

# Combine two tasks in the same house from original VNLA into a new task with two priorities
def combine_two_tasks(task1, task2):
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

  return new_task

# Take in original tasks from the same house and output a list of new tasks
# `limit` indicates the max number of resultant data points. Default is 2k but can be increased to ~4k.
def generate_tasks_from_same_house(tasks, limit=3000):
  results = []
  counter = 0
  for i in range(len(tasks) - 1):
    for j in range(i + 1, len(tasks)):
      new_task = combine_two_tasks(tasks[i], tasks[j])
      results.append(new_task)
      counter += 1
      if counter >= limit:
        return results
  return results

def main():
  # Step One: group tasks in the same house (identified by `scan`)
  json_filename = './asknav_train.json' # mock file instead of the actual file
  tasks_by_house = group_tasks_by_house(json_filename) # `scan` -> list of tasks

  # Step Two: construct a task from every pair of tasks from the same house/scan
  all_new_tasks = []
  for house, tasks in tasks_by_house.items():
    print(f'Now processing {house}')
    new_tasks = generate_tasks_from_same_house(tasks)
    all_new_tasks += new_tasks

  # Step Three: randomise ordering of new tasks and write to new file
  random.shuffle(all_new_tasks)
  print(f'There are {len(all_new_tasks)} tasks in total.')
  with open('multipri_asknav_train.json', 'w') as result_file:
    json.dump(all_new_tasks, result_file, indent=4, sort_keys=True)


if __name__ == "__main__":
  main()
