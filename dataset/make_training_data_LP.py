import sys

import numpy as np
from dataset import *

sys.path.append('.')
from domains.gridworld_LP import *
from generators.obstacle_gen import *
sys.path.remove('.')


def extract_action(traj):
    # Given a trajectory, outputs a 1D vector of
    #  actions corresponding to the trajectory.

    n_actions = 8
    action_vecs = np.asarray([[-1., 0.], [1., 0.], [0., 1.], [0., -1.],
                              [-1., 1.], [-1., -1.], [1., 1.], [1., -1.]])
    action_vecs[4:] = 1 / np.sqrt(2) * action_vecs[4:]
    action_vecs = action_vecs.T

    state_diff = np.diff(traj, axis=0)
    norm_state_diff = state_diff * np.tile(
        1 / np.sqrt(np.sum(np.square(state_diff), axis=1)), (2, 1)).T
    prj_state_diff = np.dot(norm_state_diff, action_vecs)
    actions_one_hot = np.abs(prj_state_diff - 1) < 0.00001
    actions = np.dot(actions_one_hot, np.arange(n_actions).T)
    return actions


def make_data(dom_size, n_domains, max_obs, max_obs_size, n_traj,
              state_batch_size):

    X_l = []
    S1_l = []
    S2_l = []
    Labels_l = []

    dom = 0.0
    while dom <= n_domains:

        #"""Step1: Build map. Add obstacles, border and goal."""

        goal = [np.random.randint(dom_size[0]), np.random.randint(dom_size[1])]
        # Generate obstacle map
        obs = obstacles(domsize=[dom_size[0], dom_size[1]],    # domain size
                        mask=goal,        # goal
                        size_max=3,          # obstacl's max size
                        dom=None,            # must be None
                        obs_types='rect',    # obstacl's tpye, 'rect' only
                        num_types=1)         # number of types, 1 only

        # add random obstacles to dom,
        n_obs = obs.add_n_rand_obs(n=max_obs)

        # add border to dom, 
        border_res = obs.add_border()

        ### ==> Step2:  Find optimal path.

        # get final map
        im = obs.get_final()

        # generate gridworld from obstacle map
        G = gridworld(image=im,
                      n_dirc=8, 
                      targetx=goal[0],      
                      targety=goal[1],      
                      turning_loss=0.053333,
                      p_sys=0.010225,
                      p_row=0.0002,
                      p_col=0.0001)   

        # solve LP problem
        Lambda, Distance, startx, starty = LP(M=G, n_traj=n_traj)

        # search optimal path
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
                if states_xy_one_traj is None:
                    pass
                else:
                    states_xy.append(states_xy_one_traj)
                n_search_failure = n_search_failure + search_failure

        # get all failure times
        n_failure = n_solver_failure + n_problem_infeasibly + n_search_failure

        # Get value prior
        value_prior = G.t_get_reward_prior()

        for i in range(0, n_traj-n_failure):
            if np.size(states_xy[i], axis=0) > 1:

                # Get optimal actions for each state
                actions = extract_action(states_xy[i])
                ns = states_xy[i].shape[0] - 1

                # Get last states_xy
                last_states_xy = np.concatenate((states_xy[i][0,:].reshape((1,2)), states_xy[i][0:-2,:]), axis=0) # (n_step, 2)
                last_states = np.zeros((ns, 1, dom_size[0], dom_size[1]))
                for k in range(0,ns):
                    last_states[k, 0, np.int(last_states_xy[k, 0]), np.int(last_states_xy[k, 1])] = 10 

                # Invert domain image => 0 = free, 1 = obstacle
                image = 1 - im
                # Resize domain and goal images and concate
                image_data = np.resize(image, (1, 1, dom_size[0], dom_size[1]))
                value_data = np.resize(value_prior,(1, 1, dom_size[0], dom_size[1]))
                iv_mixed = np.concatenate((image_data, value_data), axis=1)
                map_info = np.tile(iv_mixed, (ns, 1, 1, 1)) #  of size (n_step, 2, dom_size[0], dom_size[1])
                X_current = np.concatenate((map_info, last_states), axis=1)  #  of size (n_step, 3, dom_size[0], dom_size[1])

                # Resize states
                S1_current = np.expand_dims(states_xy[i][0:ns, 0], axis=1) # of size(n_step, 1) means state_x
                S2_current = np.expand_dims(states_xy[i][0:ns, 1], axis=1) # of size(n_step, 1) means state_y
                # Resize labels
                Labels_current = np.expand_dims(actions, axis=1) # of size(n_step, 1) means action

                # Append to output list
                X_l.append(X_current)
                S1_l.append(S1_current)
                S2_l.append(S2_current)
                Labels_l.append(Labels_current)

        dom += 1
        sys.stdout.write("\r" + str(int((dom / n_domains) * 100)) + "%")
        sys.stdout.flush()
    sys.stdout.write("\n")
    # Concat all outputs
    X_f = np.concatenate(X_l) # of size(<7*5000 n_steps' sum, 2, 28, 28), information of map and goal
    S1_f = np.concatenate(S1_l) # of size(<7*5000 n_steps' sum, 1), information of states_x
    S2_f = np.concatenate(S2_l) # of size(<7*5000 n_steps' sum, 1), information of states_y
    Labels_f = np.concatenate(Labels_l) # of of size(<7*5000 n_steps' sum, 1), information of actions
    return X_f, S1_f, S2_f, Labels_f


def main(dom_size=[16, 16],
         n_domains=50000,
         max_obs=33,
         max_obs_size=2,
         n_traj=14,
         state_batch_size=1):
    # Get path to save dataset
    save_path = "dataset/gridworld_{0}x{1}_LP_History_50000".format(dom_size[0], dom_size[1])
    # Get training data
    print("Now making training data...")
    X_out_tr, S1_out_tr, S2_out_tr, Labels_out_tr = make_data(
        dom_size, n_domains, max_obs, max_obs_size, n_traj, state_batch_size)
    # Get testing data
    print("\nNow making  testing data...")
    X_out_ts, S1_out_ts, S2_out_ts, Labels_out_ts = make_data(
        dom_size, n_domains / 6, max_obs, max_obs_size, n_traj,
        state_batch_size)
    # Save dataset
    np.savez_compressed(save_path, X_out_tr, S1_out_tr, S2_out_tr,
                        Labels_out_tr, X_out_ts, S1_out_ts, S2_out_ts,
                        Labels_out_ts)


if __name__ == '__main__':
    main()
