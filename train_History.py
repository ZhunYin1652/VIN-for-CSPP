import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

# import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from dataset.dataset import *
from utility.utils import *
from model_History import *

def train(net, trainloader, config, criterion, optimizer, use_GPU):
    print_header()
    avg_errors = []
    # Loop over dataset multiple times
    for epoch in range(config.epochs):  
        avg_error, avg_loss, num_batches = 0.0, 0.0, 0.0
        start_time = time.time()
        # Loop over batches of data
        for j, data in enumerate(trainloader):  
            # Get input batch
            X, S1, S2, labels = data
            # Drop those data, if not enough for a batch
            if X.size()[0] != config.batch_size:
                continue 
            # Do some transform
            X[:, 2, :, :] =  X[:, 2, :, :]*-1
            S_current = torch.zeros(config.batch_size, 1, config.imsize, config.imsize)
            for i in range(0, config.batch_size):
                S_current[i, 0, S1[i].long(), S2[i].long()] = -10
                if torch.equal(S_current, X[i, 2, :, :]):
                    X[i, 2, :, :] = torch.zeros(1, 1, config.imsize, config.imsize)
            X_input = torch.from_numpy(np.concatenate((X, S_current), axis=1))
            # Send Tensors to GPU if available
            if use_GPU:
                S1 = S1.cuda()
                S2 = S2.cuda()
                labels = labels.cuda()
                X_input = X_input.cuda()
            # Wrap to autograd.Variable
            X_input, labels = Variable(X_input), Variable(labels)
            S1, S2 = Variable(S1), Variable(S1)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass0
            outputs, predictions, _ = net(X_input, S1, S2, config)
            # Loss(CrossEntropy)
            loss = criterion(outputs, labels)
            # Backward pass
            loss.backward()
            # Update params
            optimizer.step()
            # Calculate Loss and Error
            loss_batch, error_batch = get_stats(loss, predictions, labels)
            avg_loss += loss_batch
            avg_error += error_batch
            num_batches += 1
        time_duration = time.time() - start_time
        # Print epoch logs
        print_stats(epoch, avg_loss, avg_error, num_batches, time_duration)
        avg_errors.append(avg_error/num_batches)

    with open("avg_errors.txt", "w") as output:
        output.write(str(avg_errors))

    print('\nFinished training. \n')


def test(net, testloader, config):
    total, correct = 0.0, 0.0
    for i, data in enumerate(testloader):
        # Get inputs
        X, S1, S2, labels = data
        if X.size()[0] != config.batch_size:
            continue  # Drop those data, if not enough for a batch
        # Send Tensors to GPU if available
        if use_GPU:
            X = X.cuda()
            S1 = S1.cuda()
            S2 = S2.cuda()
            labels = labels.cuda()
        # Wrap to autograd.Variable
        X, S1, S2 = Variable(X), Variable(S1), Variable(S2)
        # Forward pass
        outputs, predictions, _ = net(X, S1, S2, config)
        # Select actions with max scores(logits)
        _, predicted = torch.max(outputs, dim=1, keepdim=True)
        # Unwrap autograd.Variable to Tensor
        predicted = predicted.data
        # Compute test accuracy
        correct += (torch.eq(torch.squeeze(predicted), labels)).sum()
        total += labels.size()[0]
    print('Test Accuracy: {:.2f}%'.format(100 * (correct / total)))


if __name__ == '__main__':
    # Automatic swith of GPU mode if available
    torch.backends.cudnn.benchmark = True
    use_GPU = True #torch.cuda.is_available()
    # Parsing training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datafile',
        type=str,
        default='dataset/gridworld_16x16_LP_History.npz',
        help='Path to data file')
    parser.add_argument('--imsize', type=int, default=16, help='Size of image')
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='Learning rate, [0.01, 0.005, 0.002, 0.001]')
    parser.add_argument(
        '--epochs', type=int, default=3, help='Number of epochs to train')
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
        '--batch_size', type=int, default=128, help='Batch size')
    config = parser.parse_args()
    # Get path to save trained model
    save_path = "trained/vin_{0}x{0}_LP_History_{1}_2021.pth".format(config.imsize, config.epochs)
    # Instantiate a VIN model
    net = VIN(config)
    # Use GPU if available
    if use_GPU:
        net = net.cuda()
        print('use GPU')
    # Loss
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, betas=(0.9, 0.99))
    # Dataset transformer: torchvision.transforms
    transform = None
    # Define Dataset
    trainset = GridworldData(
        config.datafile, imsize=config.imsize, train=True, transform=transform)
    testset = GridworldData(
        config.datafile,
        imsize=config.imsize,
        train=False,
        transform=transform)
    # Create Dataloader
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    # Train the model
    train(net, trainloader, config, criterion, optimizer, use_GPU)
    # Test accuracy
    #test(net, testloader, config)
    # Save the trained model parameters
    torch.save(net.state_dict(), save_path)
