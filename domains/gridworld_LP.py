import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import cvxpy as cp
import matplotlib.pyplot as plt
import time

class gridworld:

    #"""A class for making gridworlds"""

    def __init__(self, image, targetx, targety, n_dirc=8, turning_loss=0.01, p_sys=0.01, p_row=0.001,p_col=0.0001):
        self.image = image
        self.n_row = image.shape[0]
        self.n_col = image.shape[1]
        self.n_dirc = n_dirc
        self.obstacles = []
        self.freespace = []
        self.targetx = targetx
        self.targety = targety
        self.G = []                     # transition matrix (by bool)                            G (<784*8,<784*8)
        self.W = []                     # transition matrix (by distance)                        W (<784*8,<784*8)
        self.R = []                     # action reward matrix (by distance)(all 0 for goal)     R (<784*8,8)
        self.P = []                     # transition matrix w.r.t action (by bool)               P (<784*8,<784*8,8)
        self.A = []
        self.PP = []
        self.C = []                     # for LP constrain: Cx =<_K d , C of size (num_Wij!=0, <784*8)
        self.d = []
        self.p_pos = [] # penalty for position
        self.n_states = 0
        self.n_actions = 0
        self.num_freespace = 0
        self.state_map_col = []
        self.state_map_row = []
        self.non_obstacles = []
        self.p_turn = turning_loss # penalty for turning
        self.p_sys = p_sys # penalty for sysmtric
        self.p_row = p_row # penalty for latitude
        self.p_col = p_col # penalty for longitude
        self.set_vals()

    def set_vals(self):
        # Setup function to initialize all necessary
        #  data
        row_obs, col_obs = np.where(self.image == 0)
        row_free, col_free = np.where(self.image != 0)

        self.obstacles = [row_obs, col_obs]
        self.freespace = [row_free, col_free]

        n_states = self.n_row * self.n_col * self.n_dirc   # 28*28*8
        n_actions = 8
        n_dirction = self.n_dirc
        self.n_states = n_states
        self.n_actions = n_actions

        p_n = np.zeros((self.n_states, self.n_states))
        p_s = np.zeros((self.n_states, self.n_states))
        p_e = np.zeros((self.n_states, self.n_states))
        p_w = np.zeros((self.n_states, self.n_states))
        p_ne = np.zeros((self.n_states, self.n_states))
        p_nw = np.zeros((self.n_states, self.n_states))
        p_se = np.zeros((self.n_states, self.n_states))
        p_sw = np.zeros((self.n_states, self.n_states))

        # build action reward matrix of szie (#states, #actions), whose goal_state row is all zeros.
        R = -1 * np.ones((self.n_states, self.n_actions))
        R[:, 4:self.n_actions] = R[:, 4:self.n_actions] * np.sqrt(2)
        target = np.ravel_multi_index(
            [self.targetx, self.targety, range(0,self.n_dirc)], (self.n_row, self.n_col, self.n_dirc), order='F')
        R[target, :] = 0

        for row in range(0, self.n_row):
            for col in range(0, self.n_col):
                for dirc in range(0, self.n_dirc):
                
                    # a int: the state(row,col)'s index in the 28*28*8 vector
                    curpos = np.ravel_multi_index(
                        [row, col, dirc], (self.n_row, self.n_col, self.n_dirc), order='F')

                    # three (3,) array: all possible next state's row/column/dirction in 28*28 matrix
                    rows, cols, dircs = self.neighbors(row, col, dirc)

                    # (3,) array: all possible next state's indices in the 28*28*3 vectors 
                    neighbor_inds = np.ravel_multi_index(
                        [rows, cols, dircs], (self.n_row, self.n_col, self.n_dirc), order='F')

                    # eight (28^2*8, 28^2*8) arrays: 8 state_transition matrices w.r.t 8 different actions
                    # p_a(i,j) means s_i ==> s_j by action a
                    p_turn = self.p_turn
                    p_sys = self.p_sys
                    if dirc == 0:
                        p_n[curpos, neighbor_inds[0]] = p_n[curpos, neighbor_inds[0]] + 1
                        p_ne[curpos, neighbor_inds[1]] = p_ne[curpos, neighbor_inds[1]] + 1 + p_turn + p_sys
                        p_nw[curpos, neighbor_inds[2]] = p_nw[curpos, neighbor_inds[2]] + 1 + p_turn
                    if dirc == 1:
                        p_s[curpos, neighbor_inds[0]] = p_s[curpos, neighbor_inds[0]] + 1
                        p_se[curpos, neighbor_inds[1]] = p_se[curpos, neighbor_inds[1]] + 1 + p_turn
                        p_sw[curpos, neighbor_inds[2]] = p_sw[curpos, neighbor_inds[2]] + 1 + p_turn + p_sys
                    if dirc == 2:
                        p_e[curpos, neighbor_inds[0]] = p_e[curpos, neighbor_inds[0]] + 1
                        p_ne[curpos, neighbor_inds[1]] = p_ne[curpos, neighbor_inds[1]] + 1 + p_turn
                        p_se[curpos, neighbor_inds[2]] = p_se[curpos, neighbor_inds[2]] + 1 + p_turn + p_sys
                    if dirc == 3:
                        p_w[curpos, neighbor_inds[0]] = p_w[curpos, neighbor_inds[0]] + 1
                        p_nw[curpos, neighbor_inds[1]] = p_nw[curpos, neighbor_inds[1]] + 1 + p_turn + p_sys
                        p_sw[curpos, neighbor_inds[2]] = p_sw[curpos, neighbor_inds[2]] + 1 + p_turn                       
                    if dirc == 4: 
                        p_ne[curpos, neighbor_inds[0]] = p_ne[curpos, neighbor_inds[0]] + 1
                        p_n[curpos, neighbor_inds[1]] = p_n[curpos, neighbor_inds[1]] + 1 + p_turn
                        p_e[curpos, neighbor_inds[2]] = p_e[curpos, neighbor_inds[2]] + 1 + p_turn + p_sys
                    if dirc == 5: 
                        p_nw[curpos, neighbor_inds[0]] = p_nw[curpos, neighbor_inds[0]] + 1
                        p_n[curpos, neighbor_inds[1]] = p_n[curpos, neighbor_inds[1]] + 1 + p_turn + p_sys
                        p_w[curpos, neighbor_inds[2]] = p_w[curpos, neighbor_inds[2]] + 1 + p_turn
                    if dirc == 6: 
                        p_se[curpos, neighbor_inds[0]] = p_se[curpos, neighbor_inds[0]] + 1
                        p_s[curpos, neighbor_inds[1]] = p_s[curpos, neighbor_inds[1]] + 1 + p_turn + p_sys
                        p_e[curpos, neighbor_inds[2]] = p_e[curpos, neighbor_inds[2]] + 1 + p_turn
                    if dirc == 7: 
                        p_sw[curpos, neighbor_inds[0]] = p_sw[curpos, neighbor_inds[0]] + 1
                        p_s[curpos, neighbor_inds[1]] = p_s[curpos, neighbor_inds[1]] + 1 + p_turn
                        p_w[curpos, neighbor_inds[2]] = p_w[curpos, neighbor_inds[2]] + 1 + p_turn + p_sys

        # penalty for position
        p_pos = np.zeros((self.n_row, self.n_col, self.n_dirc))
        for i in range(0,self.n_row):
            for j in range(0,self.n_col):
                p_pos[i,j,:] = pow(i,1/1)*self.p_row + pow(j,1/1)*self.p_col

        p_pos = p_pos.flatten('F')       

        # (28^2*8, 28^2*8) array: state_transition matrix by bool
        G = np.logical_or.reduce((p_n, p_s, p_e, p_w, p_ne, p_nw, p_se, p_sw))

        # (28^2*8, 28^28*8) array: state_transition matrix by distance
        W = np.maximum(
                np.maximum(
                    np.maximum(
                        np.maximum(
                            np.maximum(np.maximum(np.maximum(p_n, p_s), p_e), p_w),
                                np.sqrt(2) * p_ne),
                        np.sqrt(2) * p_nw),
                    np.sqrt(2) * p_se),
                np.sqrt(2) * p_sw)

        # (<28^28*8,) array: free spaces's indices in 28*28*8 vector 
        self.num_freespace = np.size(self.freespace[0])
        non_obstacles = np.ravel_multi_index(
            [np.tile(self.freespace[0], n_dirction), np.tile(self.freespace[1], n_dirction), 
                np.repeat(np.array(range(0, self.n_dirc)), self.num_freespace)], 
            (self.n_row, self.n_col, self.n_dirc),order='F')
        non_obstacles = np.sort(non_obstacles)
        self.non_obstacles = non_obstacles

        p_n = p_n[non_obstacles, :]
        p_n = np.expand_dims(p_n[:, non_obstacles], axis=2) # of size (<784*8,<784*8,1)

        p_s = p_s[non_obstacles, :]
        p_s = np.expand_dims(p_s[:, non_obstacles], axis=2)
        p_e = p_e[non_obstacles, :]
        p_e = np.expand_dims(p_e[:, non_obstacles], axis=2)
        p_w = p_w[non_obstacles, :]
        p_w = np.expand_dims(p_w[:, non_obstacles], axis=2)
        p_ne = p_ne[non_obstacles, :]
        p_ne = np.expand_dims(p_ne[:, non_obstacles], axis=2)
        p_nw = p_nw[non_obstacles, :]
        p_nw = np.expand_dims(p_nw[:, non_obstacles], axis=2)
        p_se = p_se[non_obstacles, :]
        p_se = np.expand_dims(p_se[:, non_obstacles], axis=2)
        p_sw = p_sw[non_obstacles, :]
        p_sw = np.expand_dims(p_sw[:, non_obstacles], axis=2)
        G = G[non_obstacles, :]
        G = G[:, non_obstacles]
        W = W[non_obstacles, :]
        W = W[:, non_obstacles]
        R = R[non_obstacles, :]
        p_pos = p_pos[non_obstacles]

        # Compute matrix C and vector d
        num_states = np.size(non_obstacles)
        k = 0
        C = np.zeros((np.count_nonzero(W), num_states))
        d = np.zeros((np.count_nonzero(W), ))
        for i in range(0,num_states):
            for j in range(0,num_states):
                if W[i,j] != 0:
                    C[k, j] = 1
                    C[k, i] = -1
                    d[k] = W[i,j] + p_pos[i] + p_pos[j]
                    k=k+1

        P = np.concatenate(
            (p_n, p_s, p_e, p_w, p_ne, p_nw, p_se, p_sw), axis=2)

        self.G = G
        self.W = W
        self.P = P
        self.R = R
        self.C = C
        self.d = d
        
        # for test net
        n_states_test = self.n_row * self.n_col
        pp_n = np.zeros((n_states_test, n_states_test))
        pp_s = np.zeros((n_states_test, n_states_test))
        pp_e = np.zeros((n_states_test, n_states_test))
        pp_w = np.zeros((n_states_test, n_states_test))
        pp_ne = np.zeros((n_states_test, n_states_test))
        pp_nw = np.zeros((n_states_test, n_states_test))
        pp_se = np.zeros((n_states_test, n_states_test))
        pp_sw = np.zeros((n_states_test, n_states_test))      

        for row in range(0, self.n_row):
            for col in range(0, self.n_col):
                
                # a int: the state(row,col)'s index in the 28*28 vector
                curpos = np.ravel_multi_index(
                    [row, col], (self.n_row, self.n_col), order='F')

                # two (8,) array: all possible next state's row/column in 28*28 matrix
                rows, cols = self.neighbors_test(row, col)

                # (8,) array: all possible next state's indices in the 28*28 vectors 
                neighbor_inds = np.ravel_multi_index(
                    [rows, cols], (self.n_row, self.n_col), order='F')

                # eight (28^2, 28^2) array: 8 state_transition matrices w.r.t 8 different actions 
                pp_n[curpos, neighbor_inds[0]] = pp_n[curpos, neighbor_inds[0]] + 1
                pp_s[curpos, neighbor_inds[1]] = pp_s[curpos, neighbor_inds[1]] + 1
                pp_e[curpos, neighbor_inds[2]] = pp_e[curpos, neighbor_inds[2]] + 1
                pp_w[curpos, neighbor_inds[3]] = pp_w[curpos, neighbor_inds[3]] + 1
                pp_ne[curpos, neighbor_inds[4]] = pp_ne[curpos, neighbor_inds[4]] + 1
                pp_nw[curpos, neighbor_inds[5]] = pp_nw[curpos, neighbor_inds[5]] + 1
                pp_se[curpos, neighbor_inds[6]] = pp_se[curpos, neighbor_inds[6]] + 1
                pp_sw[curpos, neighbor_inds[7]] = pp_sw[curpos, neighbor_inds[7]] + 1

        # (<28^28,) array: free spaces's indices in 28*28 vector 
        non_obstacles_test = np.ravel_multi_index(
            [self.freespace[0], self.freespace[1]], (self.n_row, self.n_col),
            order='F')
        non_obstacles_test = np.sort(non_obstacles_test)

        pp_n = pp_n[non_obstacles_test, :]
        pp_n = np.expand_dims(pp_n[:, non_obstacles_test], axis=2) # of size (<784,<784,1)
        pp_s = pp_s[non_obstacles_test, :]
        pp_s = np.expand_dims(pp_s[:, non_obstacles_test], axis=2)
        pp_e = pp_e[non_obstacles_test, :]
        pp_e = np.expand_dims(pp_e[:, non_obstacles_test], axis=2)
        pp_w = pp_w[non_obstacles_test, :]
        pp_w = np.expand_dims(pp_w[:, non_obstacles_test], axis=2)
        pp_ne = pp_ne[non_obstacles_test, :]
        pp_ne = np.expand_dims(pp_ne[:, non_obstacles_test], axis=2)
        pp_nw = pp_nw[non_obstacles_test, :]
        pp_nw = np.expand_dims(pp_nw[:, non_obstacles_test], axis=2)
        pp_se = pp_se[non_obstacles_test, :]
        pp_se = np.expand_dims(pp_se[:, non_obstacles_test], axis=2)
        pp_sw = pp_sw[non_obstacles_test, :]
        pp_sw = np.expand_dims(pp_sw[:, non_obstacles_test], axis=2)

        PP = np.concatenate(
            (pp_n, pp_s, pp_e, pp_w, pp_ne, pp_nw, pp_se, pp_sw), axis=2)
        self.PP = PP

        # generate mesh grid coordinate: use two 28*28 matrix represent a 784*784 mesh
        state_map_col, state_map_row = np.meshgrid(np.arange(0, self.n_col), np.arange(0, self.n_row))
        # generate <784*<784 coordinate
        self.state_map_col = state_map_col.flatten('F')[non_obstacles_test]
        self.state_map_row = state_map_row.flatten('F')[non_obstacles_test]

    def get_coords(self, states):
        # Given a state or states, state is a int <784*8, returns
        #  [row,col,dir] pairs for the state(s)
        non_obstacles = self.non_obstacles
        states = states.astype(int)
        r, c, d = np.unravel_index(
            non_obstacles[states], (self.n_col, self.n_row, self.n_dirc), order='F')
        return r, c, d

    def north(self, row, col):
        # Returns new [row,col]
        #  if we take the action
        new_row = np.max([row - 1, 0])
        new_col = col
        if self.image[new_row, new_col] == 0:
            new_row = row
            new_col = col
        return new_row, new_col

    def northeast(self, row, col):
        # Returns new [row,col]
        #  if we take the action
        new_row = np.max([row - 1, 0])
        new_col = np.min([col + 1, self.n_col - 1])
        if self.image[new_row, new_col] == 0:
            new_row = row
            new_col = col
        return new_row, new_col

    def northwest(self, row, col):
        # Returns new [row,col]
        #  if we take the action
        new_row = np.max([row - 1, 0])
        new_col = np.max([col - 1, 0])
        if self.image[new_row, new_col] == 0:
            new_row = row
            new_col = col
        return new_row, new_col

    def south(self, row, col):
        # Returns new [row,col]
        #  if we take the action
        new_row = np.min([row + 1, self.n_row - 1])
        new_col = col
        if self.image[new_row, new_col] == 0:
            new_row = row
            new_col = col
        return new_row, new_col

    def southeast(self, row, col):
        # Returns new [row,col]
        #  if we take the action
        new_row = np.min([row + 1, self.n_row - 1])
        new_col = np.min([col + 1, self.n_col - 1])
        if self.image[new_row, new_col] == 0:
            new_row = row
            new_col = col
        return new_row, new_col

    def southwest(self, row, col):
        # Returns new [row,col]
        #  if we take the action
        new_row = np.min([row + 1, self.n_row - 1])
        new_col = np.max([col - 1, 0])
        if self.image[new_row, new_col] == 0:
            new_row = row
            new_col = col
        return new_row, new_col

    def east(self, row, col):
        # Returns new [row,col]
        #  if we take the action
        new_row = row
        new_col = np.min([col + 1, self.n_col - 1])
        if self.image[new_row, new_col] == 0:
            new_row = row
            new_col = col
        return new_row, new_col

    def west(self, row, col):
        # Returns new [row,col]
        #  if we take the action
        new_row = row
        new_col = np.max([col - 1, 0])
        if self.image[new_row, new_col] == 0:
            new_row = row
            new_col = col
        return new_row, new_col

    def get_reward_prior(self):
        # Returns reward prior for gridworld
        im = -1 * np.ones((self.n_row, self.n_col))
        im[self.targetx, self.targety] = 10
        return im

    def t_get_reward_prior(self):
        # Returns reward prior as needed for
        #  dataset generation
        im = np.zeros((self.n_row, self.n_col))
        im[self.targetx, self.targety] = 10
        return im

    def neighbors(self, row, col, dirc):
        # Get valid neighbors in all valid directions
        rows, cols, dircs = [], [], []

        # N == 0
        if (dirc == 0) or (dirc == 4) or (dirc == 5): 
            new_row, new_col = self.north(row, col)
            rows, cols, dircs = np.append(rows, new_row), np.append(cols, new_col), np.append(dircs,0)
        # S == 1
        if (dirc == 1) or (dirc == 6) or (dirc == 7): 
            new_row, new_col = self.south(row, col)
            rows, cols, dircs = np.append(rows, new_row), np.append(cols, new_col), np.append(dircs,1)
        # E == 2
        if (dirc == 2) or (dirc == 4) or (dirc == 6):
            new_row, new_col = self.east(row, col)
            rows, cols, dircs = np.append(rows, new_row), np.append(cols, new_col), np.append(dircs,2)
        # W == 3
        if (dirc == 3) or (dirc == 5) or (dirc == 7):
            new_row, new_col = self.west(row, col)
            rows, cols, dircs = np.append(rows, new_row), np.append(cols, new_col), np.append(dircs,3)
        # NE == 4
        if (dirc == 4) or (dirc == 0) or (dirc == 2):
            new_row, new_col = self.northeast(row, col)
            rows, cols, dircs = np.append(rows, new_row), np.append(cols, new_col), np.append(dircs,4)
        # NW == 5
        if (dirc == 5) or (dirc == 0) or (dirc == 3):
            new_row, new_col = self.northwest(row, col)
            rows, cols, dircs = np.append(rows, new_row), np.append(cols, new_col), np.append(dircs,5)
        # SE == 6
        if (dirc == 6) or (dirc == 1) or (dirc == 2):
            new_row, new_col = self.southeast(row, col)
            rows, cols, dircs = np.append(rows, new_row), np.append(cols, new_col), np.append(dircs,6)
        # SW == 7
        if (dirc == 7) or (dirc == 1) or (dirc == 3):
            new_row, new_col = self.southwest(row, col)
            rows, cols, dircs = np.append(rows, new_row), np.append(cols, new_col), np.append(dircs,7)

        rows = rows.astype(np.int64)
        cols = cols.astype(np.int64)
        dircs = dircs.astype(np.int64)

        return rows, cols, dircs

    # functions for test
    def get_coords_test(self, states):
        # Given a state or states, returns
        #  [row,col] pairs for the state(s)
        non_obstacles = np.ravel_multi_index(
            [self.freespace[0], self.freespace[1]], (self.n_row, self.n_col),
            order='F')
        non_obstacles = np.sort(non_obstacles)
        states = states.astype(int)
        r, c = np.unravel_index(
            non_obstacles[states], (self.n_col, self.n_row), order='F')
        return r, c

    def next_state_prob(self, s, a):
        # Gets next state probability for
        #  a given action (a)
        if hasattr(a, "__iter__"):
            p = np.squeeze(self.PP[s, :, a])
        else:
            p = np.squeeze(self.PP[s, :, a]).T
        return p

    def rand_choose(self, in_vec):
        # Samples
        if len(in_vec.shape) > 1:
            if in_vec.shape[1] == 1:
                in_vec = in_vec.T
        temp = np.hstack((np.zeros((1)), np.cumsum(in_vec))).astype('int')
        q = np.random.rand()
        x = np.where(q > temp[0:-1])
        y = np.where(q < temp[1:])
        choice = np.intersect1d(x, y)[0]
        return choice

    def sample_next_state(self, s, a):
        # Gets the next state given the
        #  current state (s) and an
        #  action (a)
        vec = self.next_state_prob(s, a)
        result = self.rand_choose(vec)
        return result

    def map_ind_to_state(self, row, col):
        # Takes [row, col] and maps to a state
        rw = np.where(self.state_map_row == row)
        cl = np.where(self.state_map_col == col)
        return np.intersect1d(rw, cl)[0]

    def neighbors_test(self, row, col):
        # Get valid neighbors in all valid directions
        rows, cols = self.north(row, col)
        new_row, new_col = self.south(row, col)
        rows, cols = np.append(rows, new_row), np.append(cols, new_col)
        new_row, new_col = self.east(row, col)
        rows, cols = np.append(rows, new_row), np.append(cols, new_col)
        new_row, new_col = self.west(row, col)
        rows, cols = np.append(rows, new_row), np.append(cols, new_col)
        new_row, new_col = self.northeast(row, col)
        rows, cols = np.append(rows, new_row), np.append(cols, new_col)
        new_row, new_col = self.northwest(row, col)
        rows, cols = np.append(rows, new_row), np.append(cols, new_col)
        new_row, new_col = self.southeast(row, col)
        rows, cols = np.append(rows, new_row), np.append(cols, new_col)
        new_row, new_col = self.southwest(row, col)
        rows, cols = np.append(rows, new_row), np.append(cols, new_col)
        return rows, cols

def LP(M, n_traj=7):

    num_states = M.C.shape[1]
    C = M.C
    d = M.d
    Lambda = []
    Distance = []

    # get free position(no obs nor target) at vector (<784-1,) 
    non_obs_pos = np.ravel_multi_index(
        [M.freespace[0], M.freespace[1]], (M.n_row, M.n_col),order='F')
    target_pos = np.ravel_multi_index(
        [M.targetx, M.targety], (M.n_row, M.n_col),order='F')
    non_obs_pos = np.delete(non_obs_pos, np.argwhere(non_obs_pos==target_pos))

    num_pos = np.size(non_obs_pos)

    # sample n_traj start point
    if  num_pos >= n_traj:
        rand_ind = np.random.permutation(num_pos)
    else:
        rand_ind = np.tile(np.random.permutation(num_pos), (1, 10))
    start_ind = rand_ind[0:n_traj].flatten()
    start_xy = non_obs_pos[start_ind]
    startx, starty = np.unravel_index(
        start_xy, (M.n_row, M.n_col), order='F')

    # solve n_traj times LP, with the help of warm_start
    for k in range(0,n_traj):

        # set start/target point: two (8,) arrays means 16 indices in 784*8 vector
        start = np.ravel_multi_index(
            [startx[k], starty[k], range(0,M.n_dirc)], (M.n_row, M.n_col, M.n_dirc), order='F')
        target = np.ravel_multi_index(
            [M.targetx, M.targety, range(0,M.n_dirc)], (M.n_row, M.n_col, M.n_dirc), order='F')
        x_start = []
        x_target = []
        for i in range(0, M.n_dirc):
            x_start.append(np.argwhere(M.non_obstacles == start[i]).reshape((1,)))
            x_target.append(np.argwhere(M.non_obstacles == target[i]).reshape((1,)))

        # compute q_ij
        q = []
        for i in range(0, M.n_dirc):
            for j in range(0, M.n_dirc):
                q_temp = np.zeros((1, num_states))
                q_temp[0, x_target[j]] = 1
                q_temp[0, x_start[i]] = -1
                q.append(q_temp)
        qq = np.array(q).reshape(M.n_dirc*M.n_dirc, num_states)

        # Linear Program(LP)
        x = cp.Variable(shape = num_states)

        constraints = [C*x <= d]

        f_0 = cp.min(qq*x)

        prob = cp.Problem(objective = cp.Maximize(cp.min(qq*x)),
                          constraints = constraints)

        try:
            if k == 0:
                prob.solve(solver=cp.SCS, verbose=False)
            else:
                prob.solve(solver=cp.SCS, warm_start=True, verbose=False)
        except:
            Lambda_i = 'unsolved'
            Distance_i = 'unsolved'
            #print('unsolved')

        if prob.status is not None:
            #print(prob.status)
            if prob.status == 'unbounded':
                    Lambda_i = 'unbounded'
                    Distance_i = 'unbounded'            
            else:
                Lambda_i = constraints[0].dual_value
                Distance_i = f_0.value
            
        Lambda.append(Lambda_i)
        Distance.append(Distance_i)

    return Lambda, Distance, startx, starty

def visualize(dom, states_xy):
    fig, ax = plt.subplots()
    implot = plt.imshow(dom.T, cmap="Greys_r")
    ax.plot(states_xy[:, 0], states_xy[:, 1], c='b', label='Optimal Path')
    ax.plot(states_xy[0, 0], states_xy[0, 1], '-o', label='Start')
    ax.plot(states_xy[-1, 0], states_xy[-1, 1], '-s', label='Goal')
    legend = ax.legend(loc='upper right', shadow=False)
    for label in legend.get_texts():
        label.set_fontsize('x-small')  # the legend text size
    for label in legend.get_lines():
        label.set_linewidth(0.5)  # the legend line width
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)

def get_opt_path(Lambda, Distance, M, startx, starty):

    W = M.W
    C = M.C
    d = M.d
    targetx = M.targetx
    targety = M.targety
    r_C = np.size(C, axis=0)

    # search for best epsilon : NEED Revise Later !!! THE GOST THAT HOVER !!! ####################################
    epsilon = 0.5
    n_search = 10 #200
    t = 2 # t must bigger than 1
    Lambda_n = np.zeros((r_C,))
    b_max = int(np.floor(Distance / np.sqrt(2)))
    a = []
    for b in range(0, b_max+1):
        aa = Distance - b*np.sqrt(2)
        aa = np.amin(np.array([np.ceil(aa)-aa, aa-np.floor(aa)]))
        a.append(aa)
    b_true = int(np.argmin(np.array(a)))
    a_true = int(np.round(Distance - b_true*np.sqrt(2)))


    for i in range(0, r_C):
        if np.absolute(Lambda)[i] > epsilon:
            Lambda_n[i] = 1
        else:
            Lambda_n[i] = 0
    path = np.argwhere(Lambda_n==1)
    n_step = np.size(path, 0)
    #print('before searching, the number of lambda > %f is %d' %(epsilon,n_step))

    for k in range(1, n_search):
        t = (2 + np.sqrt(k))/(np.sqrt(k))
        if n_step > a_true + b_true + 0.1:
            epsilon = epsilon * t
            for i in range(0, r_C):
                if np.abs(Lambda)[i] > epsilon:
                    Lambda_n[i] = 1
                else:
                    Lambda_n[i] = 0
            path = np.argwhere(Lambda_n==1)
            n_step = np.size(path, 0)
        if n_step < a_true + b_true - 0.1:
            epsilon = epsilon / t
            for i in range(0, r_C):
                if np.abs(Lambda)[i] > epsilon:
                    Lambda_n[i] = 1
                else:
                    Lambda_n[i] = 0
            path = np.argwhere(Lambda_n==1)
            n_step = np.size(path, 0)

    #print('Lambda for the constraints that works = ', Lambda[path])
    #print('after searching %d iteration, the number of lambda > %f is %d' %(n_search,epsilon,n_step))
    distance_check = np.dot(Lambda_n, d)
        
    #if np.abs(distance_check - Distance) < 0.05:
        #print('Distance + search_loss = %f'%Distance)
        #print('check Distance + search_loss = %f'%distance_check)
        #print('check pass')
    #else:
        #print('check failure')
        #print('epsilon = ',epsilon)
        #print('true step = %d'%(a_true + b_true))
        #print('search setp = %d'%n_step)
    ################################################################################################################

    # find the Difference Constraints that works: of size(n_step,)
    constraints_ind = np.argwhere(Lambda_n == 1).reshape(-1,)

    # get optimal path connection matrix: of size(n_step, <784*8)
    Conn_matr = C[constraints_ind, :]

    # get coordinate in the 28*28 map

    step_from = []
    step_to = []
    for i in range(0,n_step):
        one_step = Conn_matr[i,:].reshape(-1,)

        from_ind = np.argwhere(one_step == -1).reshape(1,) # a int <784*8
        to_ind = np.argwhere(one_step == 1).reshape(1,) # a int <784*8

        step_from.append(from_ind)
        step_to.append(to_ind)

    # three arrays of size(1, n_step)
    coords_from_r, coords_from_c, coords_from_d = M.get_coords(np.array(step_from)) 
    coords_to_r, coords_to_c, coords_to_d = M.get_coords(np.array(step_to)) 
    # two arrays of size(2, n_step)
    states_from = np.concatenate((coords_from_r, coords_from_c), axis=1)
    states_to = np.concatenate((coords_to_r, coords_to_c), axis=1)

    # get right sequeence
    states_xy = np.zeros((n_step+1, 2))
    states_xy[0, :] = [startx, starty]

    search_failure = 0
    
    for i in range(0,n_step):
        ind_x = np.argwhere(coords_from_r.reshape((-1,)) == states_xy[i, 0])
        ind_y = np.argwhere(coords_from_c.reshape((-1,)) == states_xy[i, 1])
        intersec = np.setdiff1d(ind_x, np.setdiff1d(ind_x, ind_y))
        if (np.size(intersec) != 0):
            ind = intersec[0]
        else:
            states_xy = None
            search_failure = 1
            return states_xy, search_failure
        
        states_xy[i+1, :] = states_to[ind, :]
    
    return states_xy, search_failure

    

 
# ----------- Help to Understand ------------ #

import sys
sys.path.append('.')
from generators.obstacle_gen import *
sys.path.remove('.')

def main1():

 ### ==> Step1: Build map. Add obstacles, border and goal.

    obs = obstacles(domsize=[28,28],    # domain size
                   mask=[23,24],        # goal
                   size_max=2,          # obstacl's max size
                   dom=None,            # must be None
                   obs_types='rect',    # obstacl's tpye, 'rect' only
                   num_types=1)         # number of types, 1 only

    # add random obstacles to dom,
    # if this obstacl doesn't mask goal then add, else skip to add next one
    n_obs = obs.add_n_rand_obs(n=120)

    # add border to dom, 
    # if border don't mask goal then add, else skip to add next border point
    border_res = obs.add_border()

    # print final dom
    #obs.show()
    
 ### ==> Step2:  Find optimal path.

    # get final map
    im = obs.get_final()

    # generate gridworld from obstacle map
    G = gridworld(image=im,
                  n_dirc=8, 
                  targetx=23,      # goal[0]
                  targety=24,      # goal[1]
                  turning_loss=0.053333,
                  p_sys=0.010225,
                  p_row=0.0002,
                  p_col=0.0001)   

    # set number of traj
    n_traj = 100

    # solve LP problem
    print('solve LP problem:')
    start_time = time.time()
    Lambda, Distance, startx, starty = LP(M=G, n_traj=n_traj)
    end_time = time.time()
    print('time for solving LP = ', (end_time-start_time))
    #print('Lambda = ',Lambda)
    #print('Distance + search_loss = ', Distance)
    print('end LP, now search the path:')

    # search optimal path
    start_time = time.time()
    states_xy = []
    n_solver_failure = 0
    n_search_failure = 0
    n_problem_infeasibly = 0
    for i in range(0,n_traj):
        if Lambda[i] is 'unsolved':
            n_solver_failure = n_solver_failure + 1
        elif Lambda[i] is 'unbounded':
            n_problem_infeasibly = n_problem_infeasibly + 1
        else:
            states_xy_one_traj, search_failure = get_opt_path(Lambda=Lambda[i], 
                                                            Distance=Distance[i], 
                                                            M=G, 
                                                            startx=startx[i], 
                                                            starty=starty[i])
            states_xy.append(states_xy_one_traj)
            n_search_failure = n_search_failure + search_failure
    end_time = time.time()
    print('time for searching = ', (end_time-start_time))
    print('all solver failure %d times of %d'%(n_solver_failure, n_traj))
    print('all search failure %d times of %d'%(n_search_failure, n_traj))
    print('all infeasibly failure %d times of %d'%(n_problem_infeasibly, n_traj))
    print('press 0 for next path visualization')

    # visualize optiaml path
    j = 0
    for i in range(0,n_traj):
        if states_xy[i] is not None:
            visualize(im, states_xy[i])
            j = j + 1
            if j == 100: 
                break

    print('End All! Thank for Runing! -- Zhun YIN')

if __name__ == '__main__':
    main1() 