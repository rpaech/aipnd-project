#!/usr/bin/env python3

"""Trains a new neural network to identify different flower categories.

This script accepts a set of images of different flowers, which are used to
train a neural network to identify different flower categories.  The script
uses transfer learning to speed the training process and reduce the amount of
training data required.

Usage:
    train.py <data_path> --save_dir <path> --arch <name> --learning_rate <value>
             --hidden_units <value> --epochs <value> --gpu

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
    $ python train.py flowers --save_dir checkpoints --arch "densenet"
    --learning_rate 0.0005 --hidden_units 256 --epochs 15 --gpu

    Parameters for training...
    Architecture: densenet
    Hidden units: 256
    Learning rate: 0.0005
    Epochs: 15

    Training model...
    Epoch: 1/15  Trng loss: 0.06174  Valn loss: 0.04596  Acc: 0.468  Dur: 53.9s
    Epoch: 2/15  Trng loss: 0.03889  Valn loss: 0.02523  Acc: 0.707  Dur: 53.1s
    Epoch: 3/15  Trng loss: 0.02599  Valn loss: 0.01608  Acc: 0.809  Dur: 53.2s
    ...
    Epoch: 15/15  Trng loss: 0.00788  Valn loss: 0.00412  Acc: 0.935  Dur: 53.3s
    
    Training complete.
    Model saved to 'checkpoints\densenet_h256_e15_a94.pth'.
"""


import time
import argparse
import torch
from torch import optim
from torch import nn
from torchvision import datasets
from pathlib import Path
import flowernet


TRNG_FOLDER = 'train'
VALN_FOLDER = 'valid'
CHKP_FILE_EXT = 'pth'

DEFAULT_ARCH = 'densenet'
DEFAULT_LEARNING_RATE = 0.0005
DEFAULT_HIDDEN_UNITS = 256
DEFAULT_EPOCHS = 15
DEFAULT_SAVE_DIR = '.'


def get_input_args():
    """Parse and return the input arguments to the script.
    
    Checks the paths provided to the input arguments and, if they don't exist an
    exception will be raised.
    
    Returns:
        argparse.Namespace
    """
    
    argp = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""Trains a new neural network to identify different
                       flower categories.""")

    argp.add_argument('data_path', type=Path,
                      help="""The directory path where the training and
                              validation data is located.""")
    argp.add_argument('--save_dir', type=Path, 
                      default=DEFAULT_SAVE_DIR,
                      help="""The directory to save the model checkpoint
                              files.""")
    argp.add_argument('--arch', choices=['densenet', 'resnet', 'vgg'],
                      default=DEFAULT_ARCH,
                      help="""The feature model to use for transfer
                              learning.""")
    argp.add_argument('--learning_rate', type=float, 
                      default=DEFAULT_LEARNING_RATE,
                      help="""The learning rate used to train the network.""")
    argp.add_argument('--hidden_units', type=int, 
                      default=DEFAULT_HIDDEN_UNITS,
                      help="""The number of nodes in the hidden layer of the
                              classifier.""")
    argp.add_argument('--epochs', type=int, 
                      default=DEFAULT_EPOCHS,
                      help="""The number of epochs to run the training.""")
    argp.add_argument('--gpu', action="store_true",
                      help="""Use the GPU for training, if available.""")

    args = argp.parse_args()

    if not args.data_path.exists():
        raise FileNotFoundError(
            "Data path '{}' doesn't exist.".format(args.data_path))
    if not args.save_dir.exists():
        raise FileNotFoundError(
            "Save path '{}' doesn't exist.".format(args.save_dir))

    return args


def create_idx_to_cat(cat_to_idx):
    """Reverse the mapping of a cat_to_idx dictionary so that the internal index
    numbers provided by the classifier can be mapped to the actual flower 
    category labels.
    
    Args:
        cat_to_idx:  Dictionary mapping flower categories to classifier indexes.
    returns
        Dictionary mapping classifier indexes to flower categories.
    """

    return {val: key for key, val in cat_to_idx.items()}


def create_dataloaders(data_dir):
    """Create the training and validation dataloaders.
    
    The data files must be structured as per the generic data loader for the 
    torchvision.datasets.ImageFolder class.
    
    Args:
        data_dir:  A pathlib.Path object containing the location of the 
            data files.
    Returns
        trng_dataloader:  A torch.utils.data.DataLoader
        valn_dataloader:  A torch.utils.data.DataLoader
    """

    trng_dataset = datasets.ImageFolder(data_dir / TRNG_FOLDER,
                                        transform=flowernet.trng_transform)
    trng_dataloader = torch.utils.data.DataLoader(trng_dataset,
                                                  batch_size=64,
                                                  shuffle=True)

    valn_dataset = datasets.ImageFolder(data_dir / VALN_FOLDER,
                                        transform=flowernet.pred_transform)
    valn_dataloader = torch.utils.data.DataLoader(valn_dataset,
                                                  batch_size=64,
                                                  shuffle=True)

    return trng_dataloader, valn_dataloader


def train(fnm, trng_dataloader, valn_dataloader, max_epochs, learning_rate,
          on_gpu):
    """Train a flowernet model.
    
    Args:
        fnm:  A FlowerNetModule to train.
        trng_dataloader:  A torch.utils.data.DataLoader containing the 
            training dataset.
        valn_dataloader:  A torch.utils.data.DataLoader containing the
            validation dataset.
        max_epochs:  The number of epochs to run the training.
        learning_rate:  The learning rate used to train the network.
        on_gpu:  If True, use an available GPU for training.
    Returns:
        accuracy:  The final accuracy of model, based on evaluation against the
            validation dataset.
    """

    device = flowernet.get_device(on_gpu)

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
                _, top_results = predicted_results.topk(1, dim=1)
                correct_results = top_results == exp_results.view(
                    *top_results.shape)
                total_correct += correct_results.sum().item()

        mean_trng_loss = total_trng_loss / len(trng_dataloader.dataset)
        mean_valn_loss = total_valn_loss / len(valn_dataloader.dataset)
        accuracy = total_correct / len(valn_dataloader.dataset)

        trng_losses.append(mean_trng_loss)
        valn_losses.append(mean_valn_loss)

        epoch_duration = time.time() - epoch_start_time

        print('Epoch: {}/{} '.format(epoch + 1, max_epochs),
              'Trng loss: {:.5f} '.format(mean_trng_loss),
              'Valn loss: {:.5f} '.format(mean_valn_loss),
              'Acc: {:.3f} '.format(accuracy),
              'Dur: {:.1f}s'.format(epoch_duration))

    return accuracy


def main():
    """Main function.
    
    Performs the following steps:
    1. Load the input arguments
    2. Create the dataloaders and a new untrained flowernet model.
    3. Trains the model
    4. Saves the model to a checkpoint file.
    """

    args = get_input_args()

    trng_dataloader, valn_dataloader = create_dataloaders(args.data_path)
    idx_to_cat = create_idx_to_cat(trng_dataloader.dataset.class_to_idx)
    fnm = flowernet.create(args.arch, args.hidden_units, idx_to_cat)

    print()
    print("Parameters for training...")
    print("Architecture: {}".format(args.arch))
    print("Hidden units: {}".format(args.hidden_units))
    print("Learning rate: {}".format(args.learning_rate))
    print("Epochs: {}".format(args.epochs))
    print()
    print("Training model...")

    accuracy = train(fnm, trng_dataloader, valn_dataloader, args.epochs,
                     args.learning_rate, args.gpu)

    chkp_file_name = '{}_h{}_e{}_a{:02.0f}.{}'.format(args.arch,
                                                      args.hidden_units,
                                                      args.epochs,
                                                      accuracy * 100,
                                                      CHKP_FILE_EXT)
    chkp_file_path = args.save_dir / chkp_file_name
    flowernet.save(fnm, chkp_file_path)
    
    print()
    print("Training complete.")
    print("Model saved to '{}'.".format(chkp_file_path))
    print()


if __name__ == '__main__':
    main()
