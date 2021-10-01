#!/usr/bin/env python3

"""Trains a new neural network to identify different flower categories.

This script accepts a set of images of different flowers, which are used to
train a neural network to identify different flower categories.  The script
uses transfer learning to speed the training process and reduce the amount of 
training data required.

Usage:
    train.py <data_path> --save-dir <path> --arch <name> --learning_rate <value> --hidden_units <value> --epochs <value> --gpu

Args:
    <data_path>:  The directory where the training and validation data is 
        located.  The data must be structured as per the generic data loader for
        the torchvision.datasets.ImageFolder class.
    --save-dir <path>:  The directory to save the model checkpoint file.
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
    $ python train.py ./flowers --save-dir ./checkpoints --arch "vgg11" --learning_rate 0.03 --hidden_units 256 --epochs 15 --gpu
    Epoch: 1/15..  Training Loss: 0.06112..  Validation Loss: 0.04535..  Accuracy: 0.463..  Duration: 55.8s
    Epoch: 2/15..  Training Loss: 0.03860..  Validation Loss: 0.02492..  Accuracy: 0.682..  Duration: 54.9s
    Epoch: 3/15..  Training Loss: 0.02582..  Validation Loss: 0.01547..  Accuracy: 0.836..  Duration: 54.7s
    ...
    Epoch: 14/15..  Training Loss: 0.00825..  Validation Loss: 0.00412..  Accuracy: 0.936..  Duration: 54.9s
    Epoch: 15/15..  Training Loss: 0.00763..  Validation Loss: 0.00402..  Accuracy: 0.938..  Duration: 54.8s
"""