import torch
from torch import nn
from torchvision import transforms, models


CPU_DEVICE = 'cpu'
GPU_DEVICE = 'cuda'

CPL_ARCH = 'arch'
CPL_MODEL_STATE = 'model_state'
CPL_IDX_TO_CAT = 'idx_to_cat'
CPL_HIDDEN_UNITS = 'hidden_units'

FLOWER_CAT_COUNT = 102

ARCH_DENSNET = 'densenet'
ARCH_RESNET = 'resnet'
ARCH_VGG = 'vgg'

ARCH_DENSNET_SIZE = 1024
ARCH_RESNET_SIZE = 2048
ARCH_VGG_SIZE = 4096


NORMALISE_TRANSFORM = transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])

TRNG_TRANSFORM = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     NORMALISE_TRANSFORM])

PRED_TRANSFORM = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     NORMALISE_TRANSFORM])


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
    if arch == ARCH_DENSNET:
        create_func = create_densenet
    elif arch == ARCH_RESNET:
        create_func = create_resnet
    elif arch == ARCH_VGG:
        create_func = create_vgg
    else:
        raise Exception("Unknown architecture '{}'.".format(arch))

    return create_func(hidden_units, idx_to_cat)


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

    model.classifier = nn.Sequential(nn.Linear(ARCH_DENSNET_SIZE,
                                               hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_units,
                                               FLOWER_CAT_COUNT),
                                     nn.LogSoftmax(dim=1))

    return FlowerNetModule(ARCH_DENSNET, model, model.classifier,
                           hidden_units, idx_to_cat)


def create_resnet(hidden_units, idx_to_cat):
    model = models.resnet101(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(nn.Linear(ARCH_RESNET_SIZE,
                                       hidden_units),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(hidden_units,
                                       FLOWER_CAT_COUNT),
                             nn.LogSoftmax(dim=1))

    return FlowerNetModule(ARCH_RESNET, model, model.fc,
                           hidden_units, idx_to_cat)


def create_vgg(hidden_units, idx_to_cat):
    model = models.vgg11(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(ARCH_VGG_SIZE,
                                               hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_units,
                                               FLOWER_CAT_COUNT),
                                     nn.LogSoftmax(dim=1))

    return FlowerNetModule(ARCH_VGG, model, model.classifier,
                           hidden_units, idx_to_cat)
