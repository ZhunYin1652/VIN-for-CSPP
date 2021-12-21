import sys
import argparse

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

import torch
from torch.autograd import Variable

from dataset.dataset import *
from utility.utils import *
from model_History import *

from domains.gridworld_LP import *
from generators.obstacle_gen import *


def main(config,
         n_domains=10,
         max_obs=10,
         max_obs_size=None,
         n_traj=10,
         n_actions=8):

    # Correct vs total:
    correct, total = 0.0, 0.0

    # Automatic swith of GPU mode if available
    use_GPU = torch.cuda.is_available()

    # Instantiate a VIN model
    vin = VIN(config)

    # Load model parameters
    vin.load_state_dict(torch.load(config.weights))

    # Use GPU if available
    if use_GPU:
        vin = vin.cuda()
        print('use GPU')

    for dom in range(n_domains):
        #"""Step1: Build map. Add obstacles, border and goal."""

        goal = [np.random.randint(config.imsize), np.random.randint(config.imsize)]
        # Generate obstacle map
        obs = obstacles(domsize=[config.imsize, config.imsize],    # domain size
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

        for i in range(n_traj - n_failure):
            if len(states_xy[i]) > 1:

                # Get number of steps to goal
                L = len(states_xy[i]) * 2

                # Allocate space for predicted steps
                pred_traj = np.zeros((L, 2))

                # Set starting position
                pred_traj[0, :] = states_xy[i][0, :]

                for j in range(1, L):

                    # Transform current state data
                    state_data = pred_traj[j - 1, :]
                    state_data = state_data.astype(np.int)

                    # Transform domain to Networks expected input shape
                    im_data = G.image.astype(np.int)
                    im_data = 1 - im_data
                    im_data = im_data.reshape(1, 1, config.imsize,
                                              config.imsize)

                    # Transfrom value prior to Networks expected input shape
                    value_data = value_prior.astype(np.int)
                    value_data = value_data.reshape(1, 1, config.imsize,
                                                    config.imsize)



                    # Transfrom last state to Networks expected input shape
                    last_states = np.zeros((1, 1, config.imsize, config.imsize))
                    if j == 1:
                        last_states[0, 0 , state_data[0], state_data[1]] = 10
                    else:
                        last_states[0, 0 , np.int(pred_traj[j-2, 0]), np.int(pred_traj[j-2, 1])] = 10



                    # Get inputs as expected by network
                    X = torch.from_numpy(
                        np.concatenate((im_data, value_data, last_states), axis=1)).float()
                    S1_in = torch.from_numpy(state_data[0].reshape(
                        [1, 1])).float()
                    S2_in = torch.from_numpy(state_data[1].reshape(
                        [1, 1])).float()
                    X[:, 2, :, :] =  X[:, 2, :, :]*-1
                    S_current = torch.zeros(config.batch_size, 1, config.imsize, config.imsize)
                    for k in range(0, config.batch_size):
                        S_current[k, 0, state_data[0], state_data[1]] = -10
                        if torch.equal(S_current[k, 0, :, :], X[k, 2, :, :]):
                            X[k, 2, :, :] = torch.zeros(1, 1, config.imsize, config.imsize)
                    X_in = torch.cat([X, S_current], 1)

                    # Send Tensors to GPU if available
                    if use_GPU:
                        X_in = X_in.cuda()
                        S1_in = S1_in.cuda()
                        S2_in = S2_in.cuda()

                    # Wrap to autograd.Variable
                    X_in, S1_in, S2_in = Variable(X_in), Variable(
                        S1_in), Variable(S2_in)

                    # Forward pass in our neural net
                    _, predictions, vv = vin(X_in, S1_in, S2_in, config)
                    v = (torch.squeeze(vv)).detach().cpu().numpy().T 

                    _, indices = torch.max(predictions.cpu(), 1, keepdim=True)
                    a = indices.data.numpy()[0][0]

                    # r = visualize_reward(vin, X_in, S1_in, S2_in, config)
                    # plt.subplot(1, 2, 1)
                    # sns.heatmap(r, annot=False, square=True, vmin=-1, vmax=1)
                    # plt.plot(states_xy[i][-1, 0]+0.5, states_xy[i][-1, 1]+0.5, marker='o', markerfacecolor='blue', label='Goal')
                    # plt.plot(state_data[0]+0.5, state_data[1]+0.5, marker='o', markerfacecolor='orange', label='State')
                    # # plt.subplot(1, 2, 1)
                    # # sns.heatmap(v, annot=False, square=True, vmin=-1, vmax=1)
                    # # plt.plot(states_xy[i][-1, 0]+0.5, states_xy[i][-1, 1]+0.5, marker='o', markerfacecolor='blue', label='Goal')
                    # # plt.plot(state_data[0]+0.5, state_data[1]+0.5, marker='o', markerfacecolor='orange', label='State')
                    # plt.subplot(1, 2, 2)
                    # plt.imshow(G.image.T, cmap="Greys_r")
                    # plt.plot(states_xy[i][-1, 0], states_xy[i][-1, 1], marker='o', markerfacecolor='blue', label='Goal')
                    # plt.plot(state_data[0], state_data[1], marker='o', markerfacecolor='orange', label='State')

                    # plt.legend()
                    # plt.draw()
                    # plt.waitforbuttonpress(0)
                    # plt.close()

                    # Transform prediction to indices
                    s = G.map_ind_to_state(pred_traj[j - 1, 0],
                                           pred_traj[j - 1, 1])
                    ns = G.sample_next_state(s, a)
                    nr, nc = G.get_coords_test(ns)
                    pred_traj[j, 0] = nr
                    pred_traj[j, 1] = nc
                    if nr == goal[0] and nc == goal[1]:
                        # We hit goal so fill remaining steps
                        pred_traj[j + 1:, 0] = nr
                        pred_traj[j + 1:, 1] = nc
                        break

                # Plot optimal and predicted path (also start, end)
                if pred_traj[-1, 0] == goal[0] and pred_traj[-1, 1] == goal[1]:
                    correct += 1
                total += 1
                if config.plot == True:
                    visualize_test(G.image.T, states_xy[i], pred_traj)

        sys.stdout.write("\r" + str(int(
            (float(dom) / n_domains) * 100.0)) + "%")
        sys.stdout.flush()
    sys.stdout.write("\n")
    print('Rollout Accuracy: {:.2f}%'.format(100 * (correct / total)))


def visualize_test(dom, states_xy, pred_traj):
    fig, ax = plt.subplots()
    implot = plt.imshow(dom, cmap="Greys_r")
    ax.plot(states_xy[:, 0], states_xy[:, 1], c='b', label='Optimal Path')
    ax.plot(pred_traj[:, 0], pred_traj[:, 1], '-X', c='r', label='Predicted Path')
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


def visualize_reward(model, X_in, S1_in, S2_in, config):

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = (torch.squeeze(output)).detach().cpu().numpy().T
        return hook

    model.r.register_forward_hook(get_activation('r'))
    output = model(X_in, S1_in, S2_in, config)
    return activation['r']


if __name__ == '__main__':
    # Parsing training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weights',
        type=str,
        default='trained/vin_16x16_LP_History_3_2021.pth',
        help='Path to trained weights')
    parser.add_argument('--plot', action='store_true', default=True)
    parser.add_argument('--imsize', type=int, default=16, help='Size of image')
    parser.add_argument(
        '--k', type=int, default=20, help='Number of Value Iterations')
    parser.add_argument(
        '--l_i', type=int, default=4, help='Number of channels in input layer')
    parser.add_argument(
        '--l_h1',
        type=int,
        default=50,
        help='Number of channels in 1st hidden layer')
    # parser.add_argument(
    #     '--l_h2',
    #     type=int,
    #     default=50,
    #     help='Number of channels in 2nd hidden layer')
    # parser.add_argument(
    #     '--l_h3',
    #     type=int,
    #     default=50,
    #     help='Number of channels in 3rd hidden layer')
    parser.add_argument(
        '--l_q',
        type=int,
        default=10,
        help='Number of channels in q layer (~actions) in VI-module')
    parser.add_argument(
        '--batch_size', type=int, default=1, help='Batch size')
    config = parser.parse_args()
    # Compute Paths generated by network and plot
    main(config)
