#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: James Whittington
"""

import numpy as np
import copy as cp
import itertools


class Environment:
    def __init__(self, params, width, height, n_states):
        super(Environment, self).__init__()
        self.par = params
        self.width = width
        self.height = height
        self.n_actions = self.par.env.n_actions
        self.rels = self.par.env.rels
        self.walk_len = None #rep_num * width*height or None
        self.reward_value = 1.0
        self.reward_pos_training = []
        self.start_state, self.adj, self.tran, self.states_mat = None, None, None, None

        if n_states > self.par.max_states:
            raise ValueError(
                ('Too many states in your world. {} is bigger than {}. Adjust by decreasing environment size, or' +
                 'increasing params.max_states').format(n_states, self.par.max_states))


class Rectangle(Environment):

    def __init__(self, params, width, height):
        self.n_states = width * height

        super().__init__(params, width, height, self.n_states)

    def world(self, torus=False):
        """
        #state number counts accross then down
        a = np.asarray(range(25))
        print(a)
        print(np.reshape(a,(5,5)))
        [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]
         [20 21 22 23 24]]
        """
        states = int(self.width * self.height)
        adj = np.zeros((states, states))

        for i in range(states):
            # stay still
            if self.par.env.stay_still:
                adj[i, i] = 1
            # up - down
            if i + self.width < states:
                adj[i, i + self.width] = 1
                adj[i + self.width, i] = 1
                # left - right
            if np.mod(i, self.width) != 0:
                adj[i, i - 1] = 1
                adj[i - 1, i] = 1

            if torus and np.mod(i, self.width) == 0:
                adj[i, i + self.width - 1] = 1
                adj[i + self.width - 1, i] = 1

            if torus and int(i / self.width) == 0:
                adj[i, i + states - self.width] = 1
                adj[i + states - self.width, i] = 1

        tran = np.zeros((states, states))
        for i in range(states):
            if sum(adj[i]) > 0:
                tran[i] = adj[i] / sum(adj[i])

        self.adj, self.tran = adj, tran
        allowed_states = np.where(np.sum(self.adj, 1) > 0)[0]
        self.start_state = np.random.choice(allowed_states)

    def relation(self, s1, s2):
        # consider square environment. if move +1=up, -1=down, +length=right, -length=left.
        diff = s2 - s1
        if diff == self.width or diff == -self.width * (self.height - 1):  # down
            rel_type = 'down'
        elif diff == -self.width or diff == self.width * (self.height - 1):  # up
            rel_type = 'up'
        elif diff == -1 or diff == (self.width - 1):  # left
            rel_type = 'left'
        elif diff == 1 or diff == -(self.width - 1):  # right
            rel_type = 'right'
        elif diff == 0:
            rel_type = 'stay still'
        else:
            raise ValueError('impossible action')

        rel_index = self.rels.index(rel_type)

        return rel_index, rel_type

    def state_data(self):
        states_vec = np.zeros(self.n_states)
        choices = np.arange(self.par.s_size)

        for i in range(self.n_states):
            # choose which sense goes where
            new_state = np.random.choice(choices)
            states_vec[i] = new_state

        self.states_mat = states_vec.astype(int)

    def walk(self):
        """
        #state number counts accross then down
        a = np.asarray(range(25))
        print(a)
        print(np.reshape(a,(5,5)))
        [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]
         [20 21 22 23 24]]
        """
        time_steps = self.walk_len
        position = np.zeros(time_steps, dtype=np.int16)
        direc = np.zeros((self.n_actions, time_steps))

        current_angle = np.random.uniform(-np.pi, np.pi)

        # consider rectangular environment. if move +1=up, -1=down, +length=right, -length=left.
        if self.height * self.width != len(self.adj):
            raise ValueError('incorrect height/width : height * width not equal to number of states')

        #position[0] = int(self.start_state)
        position[0] = 0
        # choose random action to have gotten to start-state - doesn't get used as g_prior is for first state
        direc[0, 0] = 1
        print("TIME Step",time_steps)
        #print("self.par.env.bias_type=",self.par.env.bias_type)
        #print("self.height * self.width",self.height , self.width)
        move_dir = 'south'
        rep_num = 10 #20
        rep_num_k = -1
        left2right = True
        for i in range(time_steps - 1):
            available = np.where(self.tran[int(position[i]), :] > 0)[0].astype(int)
            # head towards objects, or in straight lines
            if self.par.env.bias_type == 'angle':
                new_poss_pos, current_angle = self.move_straight_bias(current_angle, position[i], available)
            else:
                new_poss_pos = np.random.choice(available)
            
            """if position[i] % self.width == 0 and left2right == False:
                left2right = True
            if position[i] % self.width == self.width-1 and left2right == True:
                left2right = False


            if position[i] < self.width:
                move_dir = 'south'
            if position[i] >= int(self.width * self.height) - self.width:
                    move_dir = 'north'
            if left2right == False:
                if move_dir == 'south':
                    action_dir = np.random.choice(5, size=1, p=[0.05, 0.9, 0, 0.05, 0]) # stop, south, north, west, east
                if move_dir == 'north':
                    action_dir = np.random.choice(5, size=1, p=[0.05, 0.0, 0.9, 0.05, 0])
            else:
                if move_dir == 'south':
                    action_dir = np.random.choice(5, size=1, p=[0.05, 0.9, 0, 0.0, 0.05]) # stop, south, north, west, east
                if move_dir == 'north':
                    action_dir = np.random.choice(5, size=1, p=[0.05, 0.0, 0.9, 0.0, 0.05])"""


            """if rep_num_k < rep_num * self.width:
                if position[i] < self.width:
                    if move_dir != 'east':
                        rep_num_k += 1
                    move_dir = 'south'
                if position[i] >= int(self.width * self.height) - self.width:
                    move_dir = 'north'
                    rep_num_k += 1
                if rep_num_k % rep_num == 0 and rep_num_k > 0 and position[i] - position[i-1] == -self.height:
                    move_dir = 'east'
                    #print("NEXT_COLUMN")
            else:
                if position[i] % self.height == self.height-1:
                    if position[i] - position[i-1] == 1:
                        rep_num_k += 1
                    move_dir = 'west'
                if position[i] % self.height == 0:
                    move_dir = 'east'
                    rep_num_k += 1
                if rep_num_k % rep_num == 0 and position[i] - position[i-1] == 1 and rep_num_k < rep_num * (self.width + self.height):
                    move_dir = 'south
            
            if move_dir == 'north':
                new_poss_pos = int(position[i]) - self.height
            elif move_dir == 'south':
                new_poss_pos = int(position[i]) + self.height
            elif move_dir == 'west':
                new_poss_pos = int(position[i]) - 1
            elif move_dir == 'east':
                new_poss_pos = int(position[i]) + 1"""
                
            """if action_dir == 0:
                new_poss_pos = int(position[i])
            elif action_dir == 1:
                new_poss_pos = int(position[i]) + self.height
            elif action_dir == 2:
                new_poss_pos = int(position[i]) - self.height
            elif action_dir == 3:
                new_poss_pos = int(position[i]) - 1
            elif action_dir == 4:
                new_poss_pos = int(position[i]) + 1"""

            #print("new_poss_pos",new_poss_pos)
            
            #print("new_poss_pos",new_poss_pos)

            if self.adj[position[i], new_poss_pos] == 1:
                position[i + 1] = new_poss_pos
            else:
                position[i + 1] = int(cp.deepcopy(position[i]))

            relation_taken, relation_type = self.relation(position[i], position[i + 1])
            #print("relation_taken",relation_taken,relation_type)
            if relation_taken < self.n_actions:
                direc[relation_taken, i + 1] = 1
            # stay still is just a set of zeros
            #print("position, direc",position, direc)
        #print("position",position)
        return position, direc

    def move_straight_bias(self, current_angle, position, available):
        # angle is allo-centric
        # from available position - find distance and angle from current pos
        diff_angle_min = np.pi / 4
        angles = [self.angle_between_states_square(position, x) if x != position else 10000 for x in available]
        # find angle closest to current angle
        a_diffs = [np.abs(a - current_angle) for a in angles]
        a_diffs = [a if a < np.pi else np.abs(2 * np.pi - a) for a in a_diffs]

        angle_diff = np.min(a_diffs)

        if angle_diff < diff_angle_min:
            a_min_index = np.where(a_diffs == angle_diff)[0][0]
            angle = current_angle
        else:  # hit a wall - then do random non stationary choice
            p_angles = [1 if a < 100 else 0.000001 for a in angles]
            a_min_index = np.random.choice(np.arange(len(available)), p=np.asarray(p_angles) / sum(p_angles))
            angle = angles[a_min_index]

        new_poss_pos = int(available[a_min_index])

        angle += np.random.uniform(-self.par.env.angle_bias_change, self.par.env.angle_bias_change)
        angle = np.mod(angle + np.pi, 2 * np.pi) - np.pi  # keep between +- pi

        if np.random.rand() > self.par.env.direc_bias:
            p = self.tran[int(position), available]
            new_poss_pos = np.random.choice(available, p=p)

        return new_poss_pos, angle

    def angle_between_states_square(self, s1, s2):
        x1 = s1 % self.width
        x2 = s2 % self.width

        y1 = np.floor(s1 / self.width)
        y2 = np.floor(s2 / self.width)

        angle = np.arctan2(y1 - y2, x2 - x1)

        return angle

    def get_node_positions(self, cells=None, _plot_specs=None, _mask=None):
        print("width",self.width)
        xs, ys = np.meshgrid(np.arange(self.width), np.arange(self.height))
        xs = xs.flatten() - (self.width - 1) / 2
        ys = - (ys.flatten() - (self.height - 1) / 2)

        if cells is not None:
            cell_prepared = cp.deepcopy(cells).flatten()

            return xs, ys, cell_prepared
        else:
            return xs, ys


def get_new_data_diff_envs(position, pars, envs_class):
    b_s = int(pars.batch_size)
    n_walk = position.shape[-1]  # pars.seq_len
    s_size = pars.s_size

    data = np.zeros((b_s, s_size, n_walk))
    for batch in range(b_s):
        data[batch] = sample_data(position[batch, :], envs_class[batch].states_mat, s_size)

    return data


def sample_data(position, states_mat, s_size):
    # makes one-hot encoding of sensory at each time-step
    time_steps = np.shape(position)[0]
    sense_data = np.zeros((s_size, time_steps))
    for i, pos in enumerate(position):
        ind = int(pos)
        sense_data[states_mat[ind], i] = 1
    return sense_data
