  ################################################################################
# MIT License
#
# Copyright (c) 2023 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2023
# Date Created: 2023-11-01
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


def confusion_matrix(predictions, targets):
    """
    Computes the confusion matrix, i.e. the number of true positives, false positives, true negatives and false negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size [n_classes, n_classes]
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    conf_mat = np.zeros((predictions.shape[1], predictions.shape[1]))
    for i, j in zip(np.argmax(predictions, axis=1), targets.astype(int)):
      conf_mat[i,j] += 1
    #######################
    # END OF YOUR CODE    #
    #######################
    return conf_mat


def confusion_matrix_to_metrics(confusion_matrix, beta=1.):
    """
    Converts a confusion matrix to accuracy, precision, recall and f1 scores.
    Args:
        confusion_matrix: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
    Returns: a dictionary with the following keys:
        accuracy: scalar float, the accuracy of the confusion matrix
        precision: 1D float array of size [n_classes], the precision for each class
        recall: 1D float array of size [n_classes], the recall for each clas
        f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    cm = confusion_matrix
    metrics = {}
    metrics["accuracy"] = np.trace(cm)/np.sum(cm, (0,1))
    metrics["precision"] = (cm.diagonal())/np.sum(cm, axis=0)
    metrics["recall"] = (cm.diagonal())/np.sum(cm, axis=1)
    metrics["f1_beta"] = ((1+beta**2)*(metrics["precision"]*metrics["recall"])/(beta**2*metrics["precision"]+metrics["recall"]))
    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def evaluate_model(model, data_loader, num_classes=10, beta=1):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    model.eval()
    with torch.no_grad():
      data_iter = iter(data_loader)
      preds = np.empty((0, num_classes))
      t = np.array([])
      for data, target in data_iter:
          preds = np.vstack((preds, model.forward(data.reshape(-1, 3072)).detach().numpy()))
          t = np.hstack((t,target))
      cm = confusion_matrix(preds, t)
      metrics = confusion_matrix_to_metrics(cm, beta)
      metrics["ConfMat"] = cm
    model.train()
    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics

def eval_f1_scores(model, dataloader):
    scores = []
    for beta in [0.1, 1, 10]:
        scores.append(evaluate_model(model, dataloader, 10, beta)["f1_beta"])
    return scores

def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation.
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # TODO: Initialize model and loss module
    model = MLP(3072, hidden_dims, 10, use_batch_norm)
    model.to(device)
    loss_module = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # TODO: Training loop including validation
    best_acc = 0
    logging_dict = {"Losses": []}
    val_accuracies = []
    model.train()
    evaluate_model(model, cifar10_loader["validation"], 10)
    for epoch in range(epochs):
        train_iter = iter(cifar10_loader["train"])
        current_loss = 0
        for data, targets in train_iter:
            data, targets = data.reshape(-1, 3072).to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_module(outputs, targets)
            loss.backward()

            optimizer.step()

            current_loss += loss.item()

        logging_dict["Losses"].append(current_loss)
        val_accuracies.append(evaluate_model(model, cifar10_loader["validation"], 10)["accuracy"])
        if val_accuracies[-1]>best_acc:
            best_acc = val_accuracies[-1]
            best_model = deepcopy(model)
        
    # TODO: Test best model
    test_eval = evaluate_model(best_model, cifar10_loader["test"], 10)
    logging_dict["Recall"] = test_eval["recall"]
    logging_dict["Precision"] = test_eval["precision"]
    logging_dict["ConfMat"] =  test_eval["ConfMat"]

    logging_dict["fscores"] = eval_f1_scores(model, cifar10_loader["test"])
    test_accuracy = test_eval["accuracy"]
    # TODO: Add any information you might want to save for plotting

    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    model, val_accuracies, test_accuracy, logging_dict = train(**kwargs)
    print("Accuracy: "+str(test_accuracy))
    print("Recall: "+str(logging_dict["Recall"]))
    print("Precision: "+str(logging_dict["Precision"]))
    print()
    print("F_0.1 score: "+str(logging_dict["fscores"][0]))
    print("F_1 score: "+str(logging_dict["fscores"][1]))
    print("F_10 score: "+str(logging_dict["fscores"][2]))
    plt.plot(logging_dict["Losses"])
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("Total loss of training data of model per epoch")
    plt.show()

    conf_mat = logging_dict["ConfMat"]
    plt.matshow(conf_mat, cmap=plt.cm.hot, alpha=0.7)

    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            plt.text(x=j, y=i,s=int(conf_mat[i, j]), va='center', ha='center')
    plt.xlabel('True Class')
    plt.ylabel('Predicted Class')
    plt.title('Confusion Matrix')
    plt.show()
    # Feel free to add any additional functions, such as plotting of the loss curve here
    