import json

'''
This script gives a unique ID to each task.
'''

def main():
  filename = "./asknav_train.json" # CHANGE HERE
  f = open(filename) 
  json_data = json.load(f)
  results = []
  counter = 1 # CHANGE HERE
  for task in json_data:
    task['instr_id'] = counter
    results.append(task)
    counter += 1
  f.close()

  with open('./rnum_asknav_train.json', 'w') as result_file: # CHANGE HERE
    json.dump(results, result_file, indent=4, sort_keys=True)


if __name__ == "__main__":
  main()