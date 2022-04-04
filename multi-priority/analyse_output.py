import json
import math


def main():
  input_filename = "./main_learned_nav_sample_ask_teacher_test_seen_for_eval.json" # CHANGE HERE
  f = open(input_filename)
  json_data = json.load(f)
  questions = [0] * 10
  total_question_num = 0
  for task in json_data:
    traj_length = len(task['trajectory'])
    for i, ask in enumerate(task['agent_ask']):
      if ask == 1:
        total_question_num += 1
        normalised_time = float(i + 1) / float(traj_length)
        questions[math.ceil(normalised_time / 0.1) - 1] += 1
  
  assert sum(questions) == total_question_num, "total_question_num is wrong."
  ave_num_questions = float(total_question_num)/float(len(json_data))
  print(f'Average questions asked per trajectory: {ave_num_questions}')
  for i, num in enumerate(questions):
    fraction = float(num)/float(total_question_num)
    print(f'For normalised time of {int(((i+1)*0.1)*10)}, {fraction} of total questions were asked.')

  f.close()

if __name__ == "__main__":
  main()