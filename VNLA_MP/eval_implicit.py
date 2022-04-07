import json
import os
import sys
from collections import defaultdict
import networkx as nx
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)

from utils import load_datasets, load_nav_graphs, load_region_label_to_name, load_panos_to_region

'''
This script is a duplicate of eval.py, modified to evaluate tasks with no explicit ordering of priority.
'''

class Evaluation(object):

    def __init__(self, hparams, splits, data_path):
        self.success_radius = hparams.success_radius
        self.splits = splits

        self.scans = set()
        self.graphs = {}
        self.distances = {}

        self.no_room = hasattr(hparams, 'no_room') and hparams.no_room
        if splits:
            self.load_data(load_datasets(splits, data_path,
                prefix='noroom' if self.no_room else 'asknav'))

        self.region_label_to_name = load_region_label_to_name()
        self.panos_to_region = {}
        for scan in self.scans:
            self.panos_to_region[scan] = load_panos_to_region(scan, self.region_label_to_name)


    def load_data(self, data):
        self.gt = {}
        self.instr_ids = []
        scans = []
        for item in data:
            self.gt[str(item['instr_id'])] = item
            if isinstance(item['instr_id'], int):
                self.instr_ids.append(str(item['instr_id']))
            else:
                self.instr_ids.append(item['instr_id'])
            scans.append(item['scan'])
        self.instr_ids = set(self.instr_ids)
        scans = set(scans)

        new_scans = set.difference(scans, self.scans)
        if new_scans:
            for scan in new_scans:
                self.graphs[scan] = load_nav_graphs(scan)
                self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(self.graphs[scan]))
        self.scans.update(new_scans)

    def _get_nearest(self, scan, goal_id, path):
        near_id = path[0][0]
        near_d = self.distances[scan][near_id][goal_id]
        for item in path:
            d = self.distances[scan][item[0]][goal_id]
            if d < near_d:
                near_id = item[0]
                near_d = d
        return near_id

    def _score_item(self, instr_id, path):
        gt = self.gt[instr_id]
        scan = gt['scan']

        self.scores['instr_id'].append(instr_id)
        
        first_nav_errors = second_nav_errors = oracle_errors = 1e9

        # NOTE: for original VNLA tasks
        if 'first_goal_viewpoints' not in gt:
            original_nav_errors = 1e9
            for shortest_path in gt['paths']:
                start = shortest_path[0]
                assert start == path[0][0], 'Result trajectories should include the start position'
                goal = shortest_path[-1]
                final_pos = path[-1][0]
                original_nav_errors = min(original_nav_errors, self.distances[scan][final_pos][goal])
            
            self.scores['original_nav_errors'].append(original_nav_errors)
            self.scores['combined_nav_errors'].append(original_nav_errors) # for both tasks!

            distance = 0
            prev = path[0]
            for curr in path[1:]:
                distance += self.distances[scan][prev[0]][curr[0]]
                prev = curr
            self.scores['original_trajectory_lengths'].append(distance)
            self.scores['original_trajectory_steps'].append(len(path) - 1)

            if not self.no_room:
                goal_room = None
                for shortest_path in gt['paths']:
                    assert goal_room is None or goal_room == \
                        self.panos_to_region[scan][shortest_path[-1]]
                    goal_room = self.panos_to_region[scan][shortest_path[-1]]

                assert goal_room is not None
                final_room = self.panos_to_region[scan][path[-1][0]]
                self.scores['original_room_successes'].append(final_room == goal_room)
        
        else: # for multi-priority tasks (until #########)
            self.scores['trajectory_steps'].append(len(path) - 1)
            
            # Added this to calculate relevant metrics (for first and second goals)
            start = gt['start_viewpoint']
            assert start == path[0][0], 'Result trajectories should include the start position' 
            
            ## Assuming agent took the correct ordering (first -> second)
            # For first goal
            first_goal_pos = None
            for goal in gt['first_goal_viewpoints']:
                nearest_pos = self._get_nearest(scan, goal, path)
                d = self.distances[scan][nearest_pos][goal]
                if d < first_nav_errors:
                    first_nav_errors = d
                    first_goal_pos = nearest_pos

            # For second goal
            final_pos = path[-1][0]
            for goal in gt['second_goal_viewpoints']:
                nearest_pos = self._get_nearest(scan, goal, path)
                second_nav_errors = min(second_nav_errors, self.distances[scan][final_pos][goal])
                oracle_errors = min(oracle_errors, self.distances[scan][nearest_pos][goal])

            self.scores['first_nav_errors'].append(first_nav_errors) # assuming order is correctly executed by agent
            self.scores['second_nav_errors'].append(second_nav_errors) # assuming order is correctly executed by agent

            ## Assuming agent took the wrong ordering (second -> first)
            wrong_first_nav_errors = wrong_second_nav_errors = 1e9
            # For second goal (reached first)
            for goal in gt['second_goal_viewpoints']:
                nearest_pos = self._get_nearest(scan, goal, path)
                d = self.distances[scan][nearest_pos][goal]
                if d < wrong_second_nav_errors:
                    wrong_second_nav_errors = d

            # For first goal (reached later)
            final_pos = path[-1][0]
            for goal in gt['first_goal_viewpoints']:
                nearest_pos = self._get_nearest(scan, goal, path)
                wrong_first_nav_errors = min(wrong_first_nav_errors, self.distances[scan][final_pos][goal])

            self.scores['wrong_first_nav_errors'].append(wrong_first_nav_errors)
            self.scores['wrong_second_nav_errors'].append(wrong_second_nav_errors)     

            ## Record the short_path_length and long_path_length of each task
            self.scores['short_path_lengths'].append(gt['short_path_length'] - 1)
            self.scores['long_path_lengths'].append(gt['long_path_length'] - 1)

            ## Other original metrics
            self.scores['combined_nav_errors'].append(second_nav_errors) # for both tasks!
            self.scores['oracle_errors'].append(oracle_errors)
            distance = 0
            prev = path[0]
            for curr in path[1:]:
                distance += self.distances[scan][prev[0]][curr[0]]
                prev = curr
            self.scores['trajectory_lengths'].append(distance)

            if not self.no_room:
                first_goal_room = second_goal_room = None
                
                # For first goal
                for goal_viewpoint in gt['first_goal_viewpoints']:
                    assert first_goal_room is None or first_goal_room == \
                        self.panos_to_region[scan][goal_viewpoint]
                    first_goal_room = self.panos_to_region[scan][goal_viewpoint]

                assert first_goal_room is not None
                first_room = self.panos_to_region[scan][first_goal_pos]
                self.scores['first_room_successes'].append(first_room == first_goal_room)

                # For second goal
                for goal_viewpoint in gt['second_goal_viewpoints']:
                    assert second_goal_room is None or second_goal_room == \
                        self.panos_to_region[scan][goal_viewpoint]
                    second_goal_room = self.panos_to_region[scan][goal_viewpoint]

                assert second_goal_room is not None
                final_room = self.panos_to_region[scan][path[-1][0]]
                self.scores['second_room_successes'].append(final_room == second_goal_room)
                ############

    def check_success(self, d):
        return d <= self.success_radius

    def score(self, output_file):
        ''' Evaluate each agent trajectory based on how close it got to the goal location '''
        self.scores = defaultdict(list)
        instr_ids = set(self.instr_ids)
        with open(output_file) as f:
            for item in json.load(f):
                # Check against expected ids
                if str(item['instr_id']) in instr_ids:
                    instr_ids.remove(str(item['instr_id']))
                    self._score_item(str(item['instr_id']), item['trajectory'])
        assert len(instr_ids) == 0, 'Missing %d of %d instruction ids from %s - not in %s'\
                       % (len(instr_ids), len(self.instr_ids), ",".join(self.splits), output_file)
        
        zero_multi_priority_tasks = 'first_nav_errors' not in self.scores
        zero_original_tasks = 'original_nav_errors' not in self.scores

        if not zero_original_tasks:
            print("Number of original_nav_errors:", len(self.scores['original_nav_errors']))
        if not zero_multi_priority_tasks:
            print("Number of first_nav_errors: ", len(self.scores['first_nav_errors']))
            print("Number of second_nav_errors: ", len(self.scores['second_nav_errors']))
        
        score_summary = {}
        if not zero_original_tasks:
            score_summary['original_num'] = len(self.scores['original_nav_errors'])
            score_summary['original_nav_errors'] = np.average(self.scores['original_nav_errors'])
            score_summary['original_trajectory_lengths'] = np.average(self.scores['original_trajectory_lengths'])
            score_summary['original_trajectory_steps'] = np.average(self.scores['original_trajectory_steps'])
            num_original_successes = len([d for d in self.scores['original_nav_errors'] if self.check_success(d)])
            score_summary['original_success_rate'] = float(num_original_successes)/float(len(self.scores['original_nav_errors']))
            if not self.no_room:
                score_summary['original_room_success_rate'] = float(sum(self.scores['original_room_successes'])) / \
                    len(self.scores['original_room_successes'])
        if not zero_multi_priority_tasks:
            score_summary['multi_priority_num'] = len(self.scores['first_nav_errors'])
            score_summary['first_nav_error'] = np.average(self.scores['first_nav_errors'])
            score_summary['second_nav_error'] = np.average(self.scores['second_nav_errors'])
            score_summary['oracle_error'] = np.average(self.scores['oracle_errors'])
            score_summary['steps'] = np.average(self.scores['trajectory_steps'])
            score_summary['length'] = np.average(self.scores['trajectory_lengths'])
            # Added these to evaluate non-explicit multi-priority tasks
            score_summary['wrong_first_nav_error'] = np.average(self.scores['wrong_first_nav_errors'])
            score_summary['wrong_second_nav_error'] = np.average(self.scores['wrong_second_nav_errors'])

        is_success = [(instr_id, self.check_success(d)) for d, instr_id
            in zip(self.scores['combined_nav_errors'], self.scores['instr_id'])]

        if zero_multi_priority_tasks:
            return score_summary, self.scores, is_success

        num_first_successes = len([d for d in self.scores['first_nav_errors'] if self.check_success(d)])
        num_second_successes = len([d for d in self.scores['second_nav_errors'] if self.check_success(d)])
        score_summary['first_success_rate'] = float(num_first_successes)/float(len(self.scores['first_nav_errors']))
        score_summary['second_success_rate'] = float(num_second_successes)/float(len(self.scores['second_nav_errors']))
        
        ## The metrics below are to evaluate non-explicit multi-priority tasks
        num_end_first_goal = 0 # number of tasks that end at the first (wrong) goal, after passing through 2nd goal
        total_wrong_steps_actual = 0 # sum up the actual steps taken for those with the wrong order but succeed in both goals
        total_wrong_steps_expected = 0 # sum up the expected steps taken for those with the wrong order but succeed in both goals
        total_wrong_steps_shorter_path = 0 # sum up the steps taken for those with the wrong order but succeed in both goals, if they had taken shorter paths.
        num_end_second_goal = 0 # number of tasks that end at the second (correct) goal, after passing through 1st goal
        total_correct_steps_actual = 0 # sum up the actual steps taken for those with the correct order but succeed in both goals
        total_correct_steps_expected = 0 # sum up the expected steps taken for those with the correct order but succeed in both goals
        total_correct_steps_longer_path = 0 # sum up the steps taken for those with the correct order but succeed in both goals, if they had taken longer paths.
        total_num = len(self.scores['first_nav_errors'])
        for i in range(total_num):
            end_first_goal = self.check_success(self.scores['wrong_first_nav_errors'][i]) # reached in the wrong order
            end_second_goal = self.check_success(self.scores['second_nav_errors'][i]) # reached in the right order
            if end_second_goal: # right order
                pass_through_first_goal = self.check_success(self.scores['first_nav_errors'][i])
                if pass_through_first_goal: # success!
                    num_end_second_goal += 1
                    total_correct_steps_actual += self.scores['trajectory_steps'][i]
                    total_correct_steps_expected += 0.5 * self.scores['short_path_lengths'][i] + 0.5 * self.scores['long_path_lengths'][i]
                    total_correct_steps_longer_path += self.scores['long_path_lengths'][i]
            elif end_first_goal: # wrong order
                pass_through_second_goal = self.check_success(self.scores['wrong_second_nav_errors'][i])
                if pass_through_second_goal: # success!
                    num_end_first_goal += 1
                    total_wrong_steps_actual += self.scores['trajectory_steps'][i]
                    total_wrong_steps_expected += 0.5 * self.scores['short_path_lengths'][i] + 0.5 * self.scores['long_path_lengths'][i]
                    total_wrong_steps_shorter_path += self.scores['short_path_lengths'][i]

        score_summary['wrong_order_success_rate'] = float(num_end_first_goal)/float(total_num)
        score_summary['correct_order_success_rate'] = float(num_end_second_goal)/float(total_num)
        score_summary['both_succeed_rate'] = score_summary['correct_order_success_rate']
        score_summary['wrong_order_actual_steps'] = float(total_wrong_steps_actual)/float(num_end_first_goal)
        score_summary['wrong_order_expected_steps'] = float(total_wrong_steps_expected)/float(num_end_first_goal)
        score_summary['wrong_order_shorter_steps'] = float(total_wrong_steps_shorter_path)/float(num_end_first_goal)
        score_summary['correct_order_actual_steps'] = float(total_correct_steps_actual)/float(num_end_second_goal)
        score_summary['correct_order_expected_steps'] = float(total_correct_steps_expected)/float(num_end_second_goal)
        score_summary['correct_order_longer_steps'] = float(total_correct_steps_longer_path)/float(num_end_second_goal)
        score_summary['correct_and_wrong_order_success_rate'] = float(num_end_first_goal + num_end_second_goal)/float(total_num)

        oracle_successes = len([d for d in self.scores['oracle_errors'] if self.check_success(d)])
        score_summary['oracle_rate'] = float(oracle_successes)/float(len(self.scores['oracle_errors']))
        if not self.no_room:
            score_summary['first_room_success_rate'] = float(sum(self.scores['first_room_successes'])) / \
                len(self.scores['first_room_successes'])
            score_summary['second_room_success_rate'] = float(sum(self.scores['second_room_successes'])) / \
                len(self.scores['second_room_successes'])

        return score_summary, self.scores, is_success





