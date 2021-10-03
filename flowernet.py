"""The flowernet module.

Provides a set of helper classes, functions and constants to train and use
a neural network to predict categories of flower shown in an image.
"""

import torch
from torch import nn
from torchvision import transforms, models
from pathlib import Path

# String constants to identify the CPU and GPU devices.
CPU_DEVICE = 'cpu'
GPU_DEVICE = 'cuda'

# Dictionary keys for the checkpoint file
CPL_ARCH = 'arch'
CPL_MODEL_STATE = 'model_state'
CPL_IDX_TO_CAT = 'idx_to_cat'
CPL_HIDDEN_UNITS = 'hidden_units'

# Number of flower categories
FLOWER_CAT_COUNT = 102

# Parameters for the architecture models
ARCH_DENSNET = 'densenet'
ARCH_DENSNET_SIZE = 1024
ARCH_RESNET = 'resnet'
ARCH_RESNET_SIZE = 2048
ARCH_VGG = 'vgg'
ARCH_VGG_SIZE = 4096

# Drop rate to use when training the classifier.
TRNG_DROP_RATE = 0.2

# Standard transforms to use with training, validation and predication.
# The pred_transform is used for both validation and predication.
normalise_transform = transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])
trng_transform = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     normalise_transform])
pred_transform = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     normalise_transform])

class FlowerNetModule():
    """An adapter class to allow the flower network to support the different
    architectures and hold properties for saving and loading the model from
    a checkpoint file.

    Properties:
        arch:  The name of the feature model.  Required to rebuild the model
            when loaded from a checkpoint file.
        model:  The underlying model, including the flowernet classifier.
        classifier:  A reference to the flowernet classifier.  While many of the
            properties and methods are common across architectures, the
            reference to the classifier isn't one of them (eg, 'model.fc' for
            resnet and 'model.classifier[-1]' for vgg).  The classifier
            property simplifies passing the parameters to the optimiser.
        hidden_units:  The number of hidden units in the flowernet classifier.
            Required to rebuilt the model when loaded from a checkpoint file.
        idx_to_cat:  A dictionary mapping the internal index numbers provided by
            the classifier to the actual flower category labels.
    """

    def __init__(self, arch, model, classifier, hidden_units, idx_to_cat):
        self.arch = arch
        self.model = model
        self.classifier = classifier
        self.hidden_units = hidden_units
        self.idx_to_cat = idx_to_cat


def get_device(on_gpu=False):
    """Sets the device to use for model training and evaluation.

    By default, the device will be set to the CPU.  If on_gpu is True and a
    compatible GPU is available, the device will be set to the GPU.

    Args:
        on_gpu: Set to True to use the GPU, if available.
    Returns:
        torch.device: The device to use for model training and evaluation.
    """

    if on_gpu and torch.cuda.is_available():
        device = torch.device(GPU_DEVICE)
    else:
        device = torch.device(CPU_DEVICE)

    return device


def create(arch, hidden_units, idx_to_cat):
    """Create a flowernet model.

    The supported architectures are 'densenet', 'resnet' and 'vgg'.  If an
    unsupported architecture is provided, an exception will be raised.

    Args:
        arch: The name of the feature model.
        hidden_units:  The number of hidden units in the flowernet classifier.
        idx_to_cat:  A dictionary mapping the internal index numbers provided by
            the classifier to the actual flower category labels.
    Returns:
        FlowerNetModule: A new flowernet model.
    """

    create_funcs = {ARCH_DENSNET: create_densenet,
                    ARCH_RESNET: create_resnet,
                    ARCH_VGG: create_vgg}

    if arch not in create_funcs:
        raise Exception("Unknown architecture '{}'.".format(arch))

    return create_funcs[arch](hidden_units, idx_to_cat)


def load(path):
    """Load a flowernet model from a checkpoint file.

    Args:
        path: A pathlib.Path object containing the location of the checkpoint
            file.
    Returns:
        FlowerNetModule: A flowernet model initialised from the checkpoint file.
    """

    checkpoint = torch.load(path, map_location=CPU_DEVICE)

    arch = checkpoint[CPL_ARCH]

    fnm = create(arch, checkpoint[CPL_HIDDEN_UNITS],
                 checkpoint[CPL_IDX_TO_CAT])

    fnm.model.load_state_dict(checkpoint[CPL_MODEL_STATE])

    return fnm


def save(fnm, path):
    """Save a flowernet model to a checkpoint file.

    Args:
        fnm: A FlowerNetModule object to save to file.
        path: A pathlib.Path object containing the location to save the model.
    """

    checkpoint = {CPL_ARCH: fnm.arch,
                  CPL_HIDDEN_UNITS: fnm.hidden_units,
                  CPL_MODEL_STATE: fnm.model.state_dict(),
                  CPL_IDX_TO_CAT: fnm.idx_to_cat}

    torch.save(checkpoint, path)


def create_densenet(hidden_units, idx_to_cat):
    """Create a flowernet model based on the Densenet-121 architecture.

    Args:
        hidden_units:  The number of hidden units in the flowernet classifier.
        idx_to_cat:  A dictionary mapping the internal index numbers provided by
            the classifier to the actual flower category labels.
    Returns:
        FlowerNetModule: A new flowernet model.
    """

    model = models.densenet121(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    classifier = create_classifier(ARCH_DENSNET_SIZE, hidden_units)
    model.classifier = classifier

    return FlowerNetModule(ARCH_DENSNET, model, classifier, hidden_units,
                           idx_to_cat)


def create_resnet(hidden_units, idx_to_cat):
    """Create a flowernet model based on the ResNet-101 architecture.

    Args:
        hidden_units:  The number of hidden units in the flowernet classifier.
        idx_to_cat:  A dictionary mapping the internal index numbers provided by
            the classifier to the actual flower category labels.
    Returns:
        FlowerNetModule: A new flowernet model.
    """

    model = models.resnet101(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    classifier = create_classifier(ARCH_RESNET_SIZE, hidden_units)
    model.fc = classifier

    return FlowerNetModule(ARCH_RESNET, model, classifier, hidden_units,
                           idx_to_cat)


def create_vgg(hidden_units, idx_to_cat):
    """Create a flowernet model based on the VGG-11 architecture.

    Args:
        hidden_units:  The number of hidden units in the flowernet classifier.
        idx_to_cat:  A dictionary mapping the internal index numbers provided by
            the classifier to the actual flower category labels.
    Returns:
        FlowerNetModule: A new flowernet model.
    """

    model = models.vgg11(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    classifier = create_classifier(ARCH_VGG_SIZE, hidden_units)
    model.classifier[-1] = classifier

    return FlowerNetModule(ARCH_VGG, model, classifier, hidden_units,
                           idx_to_cat)


def create_classifier(arch_units, hidden_units):
    """Create the flowernet model classifier.

    Args:
        hidden_units:  The number of hidden units in the flowernet classifier.
    Returns:
        torch.nn.Module: The classifier.
    """

    return nn.Sequential(nn.Linear(arch_units, hidden_units),
                         nn.ReLU(),
                         nn.Dropout(TRNG_DROP_RATE),
                         nn.Linear(hidden_units, FLOWER_CAT_COUNT),
                         nn.LogSoftmax(dim=1))
