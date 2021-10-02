#!/usr/bin/env python3

"""Trains a new neural network to identify different flower categories.

This script accepts a set of images of different flowers, which are used to
train a neural network to identify different flower categories.  The script
uses transfer learning to speed the training process and reduce the amount of 
training data required.

Usage:
    train.py <data_path> --save_dir <path> --arch <name> --learning_rate <value> --hidden_units <value> --epochs <value> --gpu

Args:
    <data_path>:  The directory where the training and validation data is 
        located.  The data must be structured as per the generic data loader for
        the torchvision.datasets.ImageFolder class.
    --save_dir <path>:  The directory to save the model checkpoint files.
    --arch <name>:  The feature model to use for transfer learning.  The
        <name> of the feature model must be one of the following:
            densenet: Densenet-121 (default)
            resnet: ResNet-101
            vgg: VGG-11
    --learning_rate <value>:  The learning rate used to train the network.
        Default is 0.03.
    --hidden_units <value>:  The number of nodes in the hidden layer of the
        classifier.  Default is 256.
    --epochs <value>:  The number of epochs to run the training.  
        Default is 15.
    --gpu:  Use the GPU for training, if available.

Example:
    $ python train.py ./flowers --save_dir ./checkpoints --arch "resnet" --learning_rate 0.005 --hidden_units 256 --epochs 15 --gpu
    Epoch: 1/15..  Training Loss: 0.06112..  Validation Loss: 0.04535..  Accuracy: 0.463..  Duration: 55.8s
    Epoch: 2/15..  Training Loss: 0.03860..  Validation Loss: 0.02492..  Accuracy: 0.682..  Duration: 54.9s
    Epoch: 3/15..  Training Loss: 0.02582..  Validation Loss: 0.01547..  Accuracy: 0.836..  Duration: 54.7s
    ...
    Epoch: 14/15..  Training Loss: 0.00825..  Validation Loss: 0.00412..  Accuracy: 0.936..  Duration: 54.9s
    Epoch: 15/15..  Training Loss: 0.00763..  Validation Loss: 0.00402..  Accuracy: 0.938..  Duration: 54.8s
"""


import time
import argparse
import pathlib
import torch
from torch import optim
from torch import nn
from torchvision import datasets
import flower_net as fn


def get_input_args():
    args = argparse.ArgumentParser(
        description='Trains a new neural network to identify different flower categories.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    args.add_argument('data_path', type=pathlib.Path,
                      help='The directory path where the training and validation data is located.')
    args.add_argument('--save_dir', type=pathlib.Path, default='.',
                      help='The directory to save the model checkpoint files.')
    args.add_argument('--arch', choices=['densenet', 'resnet', 'vgg'], default='densenet',
                      help='The feature model to use for transfer learning.')
    args.add_argument('--learning_rate', type=float, default=0.03,
                      help='The learning rate used to train the network.')
    args.add_argument('--hidden_units', type=int, default=256,
                      help='The number of nodes in the hidden layer of the classifier.')
    args.add_argument('--epochs', type=int, default=15,
                      help='The number of epochs to run the training.')
    args.add_argument('--gpu', action="store_true",
                      help='Use the GPU for training, if available.')

    return args.parse_args()


def create_idx_to_cat(cat_to_idx):
    return {val: key for key, val in cat_to_idx.items()}


def create_dataloaders(data_dir):

    trng_data_dir = str(data_dir) + '/train'
    valn_data_dir = str(data_dir) + '/valid'

    # Load the datasets with ImageFolder
    trng_dataset = datasets.ImageFolder(trng_data_dir,
                                        transform=fn.TRNG_TRANSFORM)

    valn_dataset = datasets.ImageFolder(valn_data_dir,
                                        transform=fn.PRED_TRANSFORM)

    # Define the dataloaders using the image datasets and trainforms
    trng_dataloader = torch.utils.data.DataLoader(trng_dataset,
                                                  batch_size=64,
                                                  shuffle=True)

    valn_dataloader = torch.utils.data.DataLoader(valn_dataset,
                                                  batch_size=64,
                                                  shuffle=True)

    return trng_dataloader, valn_dataloader


def train(fnm, trng_dataloader, valn_dataloader, max_epochs, learning_rate,
          on_gpu):

    device = fn.get_device(on_gpu)

    optimiser = optim.Adam(fnm.classifier.parameters(), learning_rate)
    criterion = nn.NLLLoss()

    fnm.model.to(device)

    trng_losses, valn_losses = [], []
    for epoch in range(max_epochs):

        epoch_start_time = time.time()

        total_trng_loss = 0
        fnm.model.train()
        for images, exp_results in trng_dataloader:
            images, exp_results = images.to(device), exp_results.to(device)
            optimiser.zero_grad()

            log_pred_results = fnm.model(images)
            loss = criterion(log_pred_results, exp_results)
            total_trng_loss += loss.item()

            loss.backward()
            optimiser.step()

        total_valn_loss = 0
        total_correct = 0
        fnm.model.eval()
        with torch.no_grad():
            for images, exp_results in valn_dataloader:
                images, exp_results = images.to(device), exp_results.to(device)

                log_pred_results = fnm.model(images)

                loss = criterion(log_pred_results, exp_results)
                total_valn_loss += loss.item()

                predicted_results = torch.exp(log_pred_results)
                top_probs, top_results = predicted_results.topk(1, dim=1)
                correct_results = top_results == exp_results.view(*top_results.shape)
                total_correct += correct_results.sum().item()

        mean_trng_loss = total_trng_loss / len(trng_dataloader.dataset)
        mean_valn_loss = total_valn_loss / len(valn_dataloader.dataset)
        mean_correct = total_correct / len(valn_dataloader.dataset)

        trng_losses.append(mean_trng_loss)
        valn_losses.append(mean_valn_loss)

        epoch_duration = time.time() - epoch_start_time

        print("Epoch: {}/{}.. ".format(epoch + 1, max_epochs),
              "Training Loss: {:.5f}.. ".format(mean_trng_loss),
              "Validation Loss: {:.5f}.. ".format(mean_valn_loss),
              "Accuracy: {:.3f}.. ".format(mean_correct),
              "Duration: {:.1f}s".format(epoch_duration))


def main():
    args = get_input_args()

    trng_dataloader, valn_dataloader = create_dataloaders(args.data_path)
    
    idx_to_cat = create_idx_to_cat(trng_dataloader.dataset.class_to_idx)
    
    fnm = fn.create(args.arch, args.hidden_units, idx_to_cat)
    
    train(fnm, trng_dataloader, valn_dataloader, args.epochs, 
          args.learning_rate, args.gpu)

    checkpoint_file_name = str(args.save_dir) + "/fnm.pth"

    fn.save(fnm, checkpoint_file_name)


if __name__ == '__main__':
    main()
