import torch
from torch import nn
from torchvision import transforms, models
from pathlib import Path


CPU_DEVICE = 'cpu'
GPU_DEVICE = 'cuda'

CPL_ARCH = 'arch'
CPL_MODEL_STATE = 'model_state'
CPL_IDX_TO_CAT = 'idx_to_cat'
CPL_HIDDEN_UNITS = 'hidden_units'

FLOWER_CAT_COUNT = 102

ARCH_DENSNET = 'densenet'
ARCH_DENSNET_SIZE = 1024
ARCH_RESNET = 'resnet'
ARCH_RESNET_SIZE = 2048
ARCH_VGG = 'vgg'
ARCH_VGG_SIZE = 4096

TRNG_DROP_RATE = 0.2


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

    def __init__(self, arch, model, classifier, hidden_units, idx_to_cat):
        self.arch = arch
        self.model = model
        self.classifier = classifier
        self.hidden_units = hidden_units
        self.idx_to_cat = idx_to_cat


def get_device(on_gpu=False):

    if on_gpu and torch.cuda.is_available():
        device = torch.device(GPU_DEVICE)
    else:
        device = torch.device(CPU_DEVICE)

    return device


def create(arch, hidden_units, idx_to_cat):

    create_funcs = {ARCH_DENSNET: create_densenet,
                    ARCH_RESNET: create_resnet,
                    ARCH_VGG: create_vgg}

    if arch not in create_funcs:
        raise Exception("Unknown architecture '{}'.".format(arch))

    return create_funcs[arch](hidden_units, idx_to_cat)


def load(path):

    checkpoint = torch.load(path, map_location=CPU_DEVICE)

    arch = checkpoint[CPL_ARCH]

    fnm = create(arch, checkpoint[CPL_HIDDEN_UNITS],
                 checkpoint[CPL_IDX_TO_CAT])

    fnm.model.load_state_dict(checkpoint[CPL_MODEL_STATE])

    return fnm


def save(fnm, path):

    checkpoint = {CPL_ARCH: fnm.arch,
                  CPL_HIDDEN_UNITS: fnm.hidden_units,
                  CPL_MODEL_STATE: fnm.model.state_dict(),
                  CPL_IDX_TO_CAT: fnm.idx_to_cat}

    torch.save(checkpoint, path)


def create_densenet(hidden_units, idx_to_cat):

    model = models.densenet121(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    classifier = create_classifier(ARCH_DENSNET_SIZE, hidden_units)
    model.classifier = classifier

    return FlowerNetModule(ARCH_DENSNET, model, classifier, hidden_units,
                           idx_to_cat)


def create_resnet(hidden_units, idx_to_cat):

    model = models.resnet101(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    classifier = create_classifier(ARCH_RESNET_SIZE, hidden_units)
    model.fc = classifier

    return FlowerNetModule(ARCH_RESNET, model, classifier, hidden_units,
                           idx_to_cat)


def create_vgg(hidden_units, idx_to_cat):

    model = models.vgg11(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    classifier = create_classifier(ARCH_VGG_SIZE, hidden_units)
    model.classifier[-1] = classifier

    return FlowerNetModule(ARCH_VGG, model, classifier, hidden_units,
                           idx_to_cat)


def create_classifier(arch_units, hidden_units):

    return nn.Sequential(nn.Linear(arch_units, hidden_units),
                         nn.ReLU(),
                         nn.Dropout(TRNG_DROP_RATE),
                         nn.Linear(hidden_units, FLOWER_CAT_COUNT),
                         nn.LogSoftmax(dim=1))
