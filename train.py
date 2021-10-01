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
    $ python train.py ./flowers --save-dir ./checkpoints --arch "resnet" --learning_rate 0.005 --hidden_units 256 --epochs 15 --gpu
    Epoch: 1/15..  Training Loss: 0.06112..  Validation Loss: 0.04535..  Accuracy: 0.463..  Duration: 55.8s
    Epoch: 2/15..  Training Loss: 0.03860..  Validation Loss: 0.02492..  Accuracy: 0.682..  Duration: 54.9s
    Epoch: 3/15..  Training Loss: 0.02582..  Validation Loss: 0.01547..  Accuracy: 0.836..  Duration: 54.7s
    ...
    Epoch: 14/15..  Training Loss: 0.00825..  Validation Loss: 0.00412..  Accuracy: 0.936..  Duration: 54.9s
    Epoch: 15/15..  Training Loss: 0.00763..  Validation Loss: 0.00402..  Accuracy: 0.938..  Duration: 54.8s
"""

import argparse
import pathlib
from flower_net import FlowerNet
from torchvision import datasets, transforms, models


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


def main():
#    args = get_input_args()
#    flwr_net = FlowerNet()
#    flwr_net.create_model(args.arch, args.hidden_units)
#    flwr_net.train(args.data_path, args.epochs, args.learning_rate, on_gpu=True)
    flwr_net = FlowerNet()
    flwr_net.create_model('densenet', 256)
    flwr_net.train('flowers', 15, 0.005, on_gpu=True)


if __name__ == '__main__':
    main()
