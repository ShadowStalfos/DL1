################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-14
################################################################################

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models

from cifar100_utils import get_train_validation_set, get_test_set


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(num_classes=100):
    """
    Returns a pretrained ResNet18 on ImageNet with the last layer
    replaced by a linear layer with num_classes outputs.
    Args:
        num_classes: Number of classes for the final layer (for CIFAR100 by default 100)
    Returns:
        model: nn.Module object representing the model architecture.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Get the pretrained ResNet18 model on ImageNet from torchvision.models
    model = models.resnet18(weights="DEFAULT")

    for param in model.parameters():
        param.requires_grad = False
    # Randomly initialize and modify the model's last layer for CIFAR100.
    layer = nn.Linear(512, 100)
    nn.init.normal_(layer.weight, 0, 0.01)

    model.fc = layer

    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name=None):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation.
        device: Device to use.
        augmentation_name: Augmentation to use for training.
    Returns:
        model: Model that has performed best on the validation set.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Load the datasets
    train_set, valid_set = get_train_validation_set(data_dir, augmentation_name=augmentation_name) 
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size = batch_size, shuffle=True)

    # Initialize the optimizer (Adam) to train the last layer of the model.
    optimizer = torch.optim.Adam(model.parameters() , lr=lr)

    # Training loop with validation after each epoch. Save the best model.
    model.to(device)
    loss_module = nn.CrossEntropyLoss()
    best_acc = 0
    model.train()

    for epoch in range(epochs):
        print(epoch)
        train_iter = iter(train_loader)
        current_loss = 0
        batch_total = len(train_iter)
        for batch_i, (data, targets) in enumerate(train_iter):
            print("{:.2%}".format(batch_i/batch_total), end='\r')
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_module(outputs, targets)
            loss.backward()

            optimizer.step()

            current_loss += loss.item()

        acc = evaluate_model(model, valid_loader, device)
        if acc>best_acc:
            best_acc = acc
            torch.save(model, checkpoint_name)

    # Load the best model on val accuracy and return it.
    model = torch.load(checkpoint_name)
    
    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set model to evaluation mode (Remember to set it back to training mode in the training loop)
    model.eval()

    # Loop over the dataset and compute the accuracy. Return the accuracy
    # Remember to use torch.no_grad().
    with torch.no_grad():
        eval_iter = iter(data_loader)
        correct = 0
        for data, targets in eval_iter:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            correct += np.sum(np.argmax(outputs, axis=1) == targets)
    accuracy = correct/len(data_loader)

    model.train()
    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def main(lr, batch_size, epochs, data_dir, seed, augmentation_name, test_noise):
    """
    Main function for training and testing the model.

    Args:
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        seed: Seed for reproducibility.
        augmentation_name: Name of the augmentation to use.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set the seed for reproducibility
    set_seed(seed)

    # Set the device to use for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = get_model()

    # Get the augmentation to use
    """weights = models.ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()"""
    "Add the augmentations here"

    # Train the model
    best_model = train_model(model, lr, batch_size, epochs, data_dir, "best_model", device, augmentation_name)

    # Evaluate the model on the test set
    test_set = get_test_set(data_dir, test_noise)
    accuracy = evaluate_model(best_model, test_set, device)
    print("Final accuracy: "+str(accuracy))
    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Feel free to add more arguments or change the setup

    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')
    parser.add_argument('--epochs', default=2, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=123, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR100 dataset.')
    parser.add_argument('--augmentation_name', default=None, type=str,
                        help='Augmentation to use.')
    parser.add_argument('--test_noise', default=False, action="store_true",
                        help='Whether to test the model on noisy images or not.')

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
