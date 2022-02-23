import json
import os
import sys
from collections import defaultdict
import networkx as nx
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)

from utils import load_datasets, load_nav_graphs, load_region_label_to_name, load_panos_to_region


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
                # self.instr_ids.extend(['%d_%d' % (item['path_id'],i)
                #     for i in range(len(item['instructions']))])
                self.instr_ids.append(str(item['instr_id']))
            else:
                # self.instr_ids.extend(['%s_%d' % (item['path_id'],i)
                #     for i in range(len(item['instructions']))])
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
        # gt = self.gt[instr_id[:instr_id.rfind('_')]]
        gt = self.gt[instr_id]
        scan = gt['scan']

        self.scores['instr_id'].append(instr_id)
        self.scores['trajectory_steps'].append(len(path) - 1)

        first_nav_errors = second_nav_errors = oracle_errors = 1e9
        # NOTE: commented this out since paths is no longer available in the new multipri dataset
        # for shortest_path in gt['paths']:
        #     start = shortest_path[0]
        #     assert start == path[0][0], 'Result trajectories should include the start position'
        #     goal = shortest_path[-1]
        #     final_pos = path[-1][0]
        #     nearest_pos = self._get_nearest(scan, goal, path)
        #     nav_errors = min(nav_errors, self.distances[scan][final_pos][goal])
        #     oracle_errors = min(oracle_errors, self.distances[scan][nearest_pos][goal])

        # Added this to calculate relevant metrics (for first and second goals)
        start = gt['start_viewpoint']
        assert start == path[0][0], 'Result trajectories should include the start position' 
        
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

        self.scores['first_nav_errors'].append(first_nav_errors)
        self.scores['second_nav_errors'].append(second_nav_errors)
        self.scores['oracle_errors'].append(oracle_errors)
        distance = 0
        prev = path[0]
        for curr in path[1:]:
            distance += self.distances[scan][prev[0]][curr[0]]
            prev = curr
        self.scores['trajectory_lengths'].append(distance)

        if not self.no_room:
            first_goal_room = second_goal_room = None
            # for shortest_path in gt['paths']:
            #     assert goal_room is None or goal_room == \
            #         self.panos_to_region[scan][shortest_path[-1]]
            #     goal_room = self.panos_to_region[scan][shortest_path[-1]]
            
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
        assert len(self.scores['first_nav_errors']) == len(self.instr_ids)
        assert len(self.scores['second_nav_errors']) == len(self.instr_ids)
        score_summary = {
            'first_nav_error': np.average(self.scores['first_nav_errors']),
            'second_nav_error': np.average(self.scores['second_nav_errors']),
            'oracle_error': np.average(self.scores['oracle_errors']),
            'steps': np.average(self.scores['trajectory_steps']),
            'length': np.average(self.scores['trajectory_lengths'])
        }
        # NOTE: haven't changed below to is_first_success for now
        is_success = [(instr_id, self.check_success(d)) for d, instr_id
            in zip(self.scores['second_nav_errors'], self.scores['instr_id'])]
        num_first_successes = len([d for d in self.scores['first_nav_errors'] if self.check_success(d)])
        num_second_successes = len([d for d in self.scores['second_nav_errors'] if self.check_success(d)])
        score_summary['first_success_rate'] = float(num_first_successes)/float(len(self.scores['first_nav_errors']))
        score_summary['second_success_rate'] = float(num_second_successes)/float(len(self.scores['second_nav_errors']))
        
        # Add in "first_succeed_second_fail_rate", "first_fail_second_succeed_rate", "both_succeed_rate",
        # and "both_fail_rate"
        first_succeed_second_fail_num = 0
        first_fail_second_succeed_num = 0
        both_succeed_num = 0
        both_succeed_steps = 0 # To calculate average steps for fully successful tasks
        both_fail_num = 0
        total_num = len(self.scores['first_nav_errors'])
        for i in range(total_num):
            is_first_succeed = self.check_success(self.scores['first_nav_errors'][i])
            is_second_succeed = self.check_success(self.scores['second_nav_errors'][i])
            if is_first_succeed and not is_second_succeed:
                first_succeed_second_fail_num += 1
            elif not is_first_succeed and is_second_succeed:
                first_fail_second_succeed_num += 1
            elif is_first_succeed and is_second_succeed:
                both_succeed_num += 1
                both_succeed_steps += self.scores['trajectory_steps'][i]
            elif not is_first_succeed and not is_second_succeed:
                both_fail_num += 1
        score_summary['first_succeed_second_fail_rate'] = float(first_succeed_second_fail_num)/float(total_num)
        score_summary['first_fail_second_succeed_rate'] = float(first_fail_second_succeed_num)/float(total_num)
        score_summary['both_succeed_rate'] = float(both_succeed_num)/float(total_num)
        score_summary['both_succeed_steps'] = float(both_succeed_steps)/float(both_succeed_num)
        score_summary['both_fail_rate'] = float(both_fail_num)/float(total_num)
        
        oracle_successes = len([d for d in self.scores['oracle_errors'] if self.check_success(d)])
        score_summary['oracle_rate'] = float(oracle_successes)/float(len(self.scores['oracle_errors']))
        if not self.no_room:
            score_summary['first_room_success_rate'] = float(sum(self.scores['first_room_successes'])) / \
                len(self.scores['first_room_successes'])
            score_summary['second_room_success_rate'] = float(sum(self.scores['second_room_successes'])) / \
                len(self.scores['second_room_successes'])
        return score_summary, self.scores, is_success





