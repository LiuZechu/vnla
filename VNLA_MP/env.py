from __future__ import division
import os
import sys
import csv
import numpy as np
import math
import base64
import json
import random
import networkx as nx
from collections import defaultdict
import scipy.stats

sys.path.append('../../build')
import MatterSim

from oracle import make_oracle
from utils import load_datasets, load_nav_graphs
import utils

csv.field_size_limit(sys.maxsize)


class EnvBatch():

    def __init__(self, from_train_env=None, img_features=None, batch_size=100):
        if from_train_env is not None:
            self.features = from_train_env.features
            self.image_h  = from_train_env.image_h
            self.image_w  = from_train_env.image_w
            self.vfov     = from_train_env.vfov
        elif img_features is not None:
            self.image_h, self.image_w, self.vfov, self.features = \
                utils.load_img_features(img_features)
        else:
            print('Image features not provided')
            self.features = None
            self.image_w = 640
            self.image_h = 480
            self.vfov = 60
        self.sims = []
        for i in range(batch_size):
            sim = MatterSim.Simulator()
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.setNavGraphPath(
                os.path.join(os.getenv('PT_DATA_DIR', '../../../data'), 'connectivity'))
            sim.init()
            self.sims.append(sim)

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            self.sims[i].newEpisode(scanId, viewpointId, heading, 0)

    def getStates(self):
        feature_states = []
        for sim in self.sims:
            state = sim.getState()
            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if self.features:
                feature = self.features[long_id][state.viewIndex,:]
                feature_states.append((feature, state))
            else:
                feature_states.append((None, state))
        return feature_states

    def makeActions(self, actions):
        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction(index, heading, elevation)


class VNLABatch():

    def __init__(self, hparams, split=None, tokenizer=None, from_train_env=None,
                 traj_len_estimates=None):
        self.env = EnvBatch(
            from_train_env=from_train_env.env if from_train_env is not None else None,
            img_features=hparams.img_features, batch_size=hparams.batch_size)

        self.random = random
        self.random.seed(hparams.seed)

        self.tokenizer = tokenizer
        self.split = split
        self.batch_size = hparams.batch_size
        self.max_episode_length = hparams.max_episode_length
        self.n_subgoal_steps = hparams.n_subgoal_steps

        self.traj_len_estimates = defaultdict(list)

        self.query_ratio = hparams.query_ratio

        self.no_room = hasattr(hparams, 'no_room') and hparams.no_room

        if self.split is not None:
            self.load_data(load_datasets([split], hparams.data_path,
                prefix='noroom' if self.no_room else 'asknav'))

        # Estimate time budget using the upper 95% confidence bound
        if traj_len_estimates is None:
            for k in self.traj_len_estimates:
                self.traj_len_estimates[k] = min(self.max_episode_length,
                    float(np.average(self.traj_len_estimates[k]) +
                    1.95 * scipy.stats.sem(self.traj_len_estimates[k])))
                assert not math.isnan(self.traj_len_estimates[k])
        else:
            for k in self.traj_len_estimates:
                if k in traj_len_estimates:
                    self.traj_len_estimates[k] = traj_len_estimates[k]
                else:
                    self.traj_len_estimates[k] = self.max_episode_length

    def make_traj_estimate_key(self, item):
        if self.no_room:
            # NOTE: differentiate between single vs multi-priority
            if 'first_end_region_name' in item:
                # multi-priority task
                key = (item['start_region_name'], item['first_object_name'], item['second_object_name'])
            else:
                # original task
                key = (item['start_region_name'], item['object_name'])
        else:
            # NOTE: differentiate between single vs multi-priority
            if 'first_end_region_name' in item:
                # multi-priority task
                key = (item['start_region_name'], item['first_end_region_name'], item['second_end_region_name'])
            else:
                # original task
                key = (item['start_region_name'], item['end_region_name'])
        return key

    def encode(self, instr):
        if self.tokenizer is None:
            sys.exit('No tokenizer!')
        return self.tokenizer.encode_sentence(instr)

    def load_data(self, data):
        self.data = []
        self.scans = set()
        for item in data:
            self.scans.add(item['scan'])

            key = self.make_traj_estimate_key(item)
            self.traj_len_estimates[key].extend(
                len(t) for t in item['trajectories'])
          
            new_item = dict(item)
            self.data.append(new_item)

        self.reset_epoch()

        if self.split is not None:
            print('VNLABatch loaded with %d instructions, using split: %s' % (
                len(self.data), self.split))

    def _next_minibatch(self):
        if self.ix == 0:
            self.random.shuffle(self.data)
        batch = self.data[self.ix:self.ix+self.batch_size]
        if len(batch) < self.batch_size:
            self.random.shuffle(self.data)
            self.ix = self.batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += self.batch_size
        self.batch = batch

    def set_data_and_scans(self, data, scans):
        self.data = data
        self.scans = scans

    def reset_epoch(self):
        self.ix = 0

    def _get_obs(self, prev_obs=None):
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]
            # NOTE: changed here
            goal_viewpoints = None 
            if 'first_goal_viewpoints' in item:
                goal_viewpoints = item['first_goal_viewpoints'] # multi-priority task
            else: # original task
                goal_viewpoints = item['goal_viewpoints'] 
            reached_first_goal = False
            if (prev_obs is not None) and ('first_goal_viewpoints' in item): # multi-priority task
                goal_viewpoints = prev_obs[i]['goal_viewpoints']
                reached_first_goal = prev_obs[i]['reached_first_goal']
            obs.append({
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'point': state.location.point,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature,
                'step' : state.step,
                'navigableLocations' : state.navigableLocations,
                'instruction' : self.instructions[i],
                'goal_viewpoints': goal_viewpoints, # NOTE: this will be changed to second_goal_viewpoints after reaching 
                'init_viewpoint' : item['start_viewpoint'] # NOTE: changed here
            })
            if 'first_goal_viewpoints' in item: # multi-priority task
                obs[-1]['first_goal_viewpoints'] = item['first_goal_viewpoints'] # NOTE: changed here
                obs[-1]['second_goal_viewpoints'] = item['second_goal_viewpoints'] # NOTE: changed here
                obs[-1]['reached_first_goal'] = reached_first_goal # NOTE: changed here
            obs[-1]['max_queries'] = self.max_queries_constraints[i]
            obs[-1]['traj_len'] = self.traj_lens[i]
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
        return obs

    def _calculate_max_queries(self, traj_len):
        ''' Sample a help-requesting budget given a time budget. '''

        max_queries = self.query_ratio * traj_len / self.n_subgoal_steps
        int_max_queries = int(max_queries)
        frac_max_queries = max_queries - int_max_queries
        return int_max_queries + (self.random.random() < frac_max_queries)

    def reset(self, is_eval):
        ''' Load a new minibatch / episodes. '''

        self._next_minibatch()

        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['start_viewpoint'] for item in self.batch]
        headings = [item['initial_heading'] for item in self.batch]
        self.instructions = [item['instruction'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)

        self.max_queries_constraints = [None] * self.batch_size
        self.traj_lens = [None] * self.batch_size

        for i, item in enumerate(self.batch):
            # Assign time budget
            if is_eval:
                # If eval use expected trajectory length between start_region, first_end_region and second_end_region
                key = self.make_traj_estimate_key(item)
                traj_len_estimate = self.traj_len_estimates[key]
            else:
                # If train use average oracle trajectory length
                traj_len_estimate = sum(len(t)
                    for t in item['trajectories']) / len(item['trajectories'])

            self.traj_lens[i] = min(self.max_episode_length, int(round(traj_len_estimate)))

            # Assign help-requesting budget
            self.max_queries_constraints[i] = self._calculate_max_queries(self.traj_lens[i])
            assert not math.isnan(self.max_queries_constraints[i])

        return self._get_obs()

    def step(self, actions, prev_obs):
        self.env.makeActions(actions)
        # NOTE: changed here
        obs = self._get_obs(prev_obs)

        # Change `goal_viewpoints` and `reached_first_goal` after reaching first goal
        for i in range(len(obs)):
            ob = obs[i]
            if 'first_goal_viewpoints' in ob: # for multi-priority task
                current_viewpoint = ob['viewpoint']
                first_goal_viewpoints = ob['first_goal_viewpoints']
                reached_first_goal = False
                for goal in first_goal_viewpoints:
                    if current_viewpoint == goal:
                        reached_first_goal = True
                        break
                if reached_first_goal and not ob['reached_first_goal']:
                    ob['reached_first_goal'] = True
                    ob['goal_viewpoints'] = ob['second_goal_viewpoints']

        return obs

    def prepend_instruction(self, idx, instr):
        ''' Prepend subgoal to end-goal. '''

        self.instructions[idx] = instr + ' . ' + self.batch[idx]['instruction']

    def get_obs(self, prev_obs):
        return self._get_obs(prev_obs)


