import json

'''
This script transforms the original VNLA dataset to suit the format
of the modified agent, but does not change the tasks themselves.
'''

def transform_dataset(input_filename):
  input_file = open(input_filename) 
  json_data = json.load(input_file)

  for task in json_data:
    task['initial_heading'] = task['heading']
    del task['heading']
    
    task['instruction'] = task['instructions'][0]
    if len(task['instructions']) != 1:
      print("instructions length not one! its length is ", len(task['instructions']))
    del task['instructions']
    
    task['instr_id'] = task['path_id']
    del task['path_id']

    task['goal_viewpoints'] = []
    for path in task['paths']:
      task['goal_viewpoints'].append(path[-1])
    
    task['start_viewpoint'] = task['paths'][0][0]
  
  input_file.close()

  return json_data

def main():
  input_filename = './ori_asknav_val_seen.json' # CHANGE HERE
  output_filename = './asknav_val_seen.json' # CHANGE HERE
  transformed_data = transform_dataset(input_filename)

  with open(output_filename, 'w') as result_file: 
    json.dump(transformed_data, result_file, indent=4, sort_keys=True)


if __name__ == "__main__":
  main()

