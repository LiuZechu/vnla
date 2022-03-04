import json
import matplotlib.pyplot as plt

'''
Instruction on running:
python metrics_analysis.py
Remember to change filename under main() if necessary.
Plots of all metrics will show one by one, where each graph plots the metric
for both seen and unseen val sets in the same plot.
'''

# Return (seen_scores, unseen_scores), where `un/seen_scores` is a list of JSON score.
def parse_scores(filename):
  f = open(filename, "r")
  content = f.read()
  content = content.replace("\'", "\"")
  f.close()

  seen_scores = []
  unseen_scores = []
  for idx, item in enumerate(content.split("{")[1:]):
    score = json.loads("{" + item)
    if idx % 2 == 0:
      seen_scores.append(score)
    else:
      unseen_scores.append(score)
  
  return seen_scores, unseen_scores

# Plot a graph of the metric for both seen and unseen.
def plot_graph(seen_scores, unseen_scores, metric_name):
  seen_data = []
  unseen_data = []
  x = []

  for idx in range(len(seen_scores)):
    seen_data.append(seen_scores[idx][metric_name])
    unseen_data.append(unseen_scores[idx][metric_name])
    x.append(idx + 1)
  
  plt.plot(x, seen_data, label = "seen")
  plt.plot(x, unseen_data, label = "unseen")
  plt.xlabel('Iteration')
  plt.ylabel(metric_name)
  plt.title(metric_name + " through iterations")
  plt.legend()

  plt.show()
  
def main():
  seen_scores, unseen_scores = parse_scores("./intermediate_scores.txt")
  for metric_name in seen_scores[0].keys():
    plot_graph(seen_scores=seen_scores, unseen_scores=unseen_scores, \
      metric_name=metric_name)


if __name__ == "__main__":
  main()

