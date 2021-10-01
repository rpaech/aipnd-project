import time

from PIL import Image

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

CPU_DEVICE = 'cpu'
GPU_DEVICE = 'cuda'

CPL_FEATURE_MODEL = 'feature_model_name'
CPL_MODEL_STATE = 'model_state'
CPL_IDX_TO_CAT = 'idx_to_cat'
CPL_HIDDEN_LAYER_SIZE = 'hidden_layer_size'

FLOWER_CAT_COUNT = 102

FEATURE_MODEL_DENSNET = 'densenet'
FEATURE_MODEL_RESNET = 'resnet'
FEATURE_MODEL_VGG = 'vgg'

FEATURE_MODEL_DENSNET_SIZE = 1024
FEATURE_MODEL_RESNET_SIZE = 2048
FEATURE_MODEL_VGG_SIZE = 4096


class FlowerNet():

    def __init__(self):
        self.model = None
        self.idx_to_cat = None
        self.feature_model_name = None
        self.hidden_layer_size = None

        normalised_transform = transforms.Normalize([0.485, 0.456, 0.406],
                                                    [0.229, 0.224, 0.225])

        self.trng_transforms = transforms.Compose([transforms.RandomRotation(30),
                                                  transforms.RandomResizedCrop(224),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  normalised_transform])

        self.pred_transforms = transforms.Compose([transforms.Resize(255),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  normalised_transform])


    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=CPU_DEVICE)
        self.model = self.create_model(checkpoint[CPL_FEATURE_MODEL], 
                                       checkpoint[CPL_HIDDEN_LAYER_SIZE])
        self.model.load_state_dict(checkpoint[CPL_MODEL_STATE])
        self.idx_to_cat = checkpoint[CPL_IDX_TO_CAT]


    def save_checkpoint(self, path):
        checkpoint = {CPL_FEATURE_MODEL: self.feature_model_name,
                      CPL_HIDDEN_LAYER_SIZE: self.hidden_layer_size,
                      CPL_MODEL_STATE: self.model.state_dict(),
                      CPL_IDX_TO_CAT: self.idx_to_cat}
        torch.save(checkpoint, path)


    def create_model(self, feature_model_name, hidden_layer_size):
        self.feature_model_name = feature_model_name
        self.init_feature_model()

        for param in self.model.parameters():
            param.requires_grad = False
            
        classifier = self.classifier()
        self.model.classifier = nn.Sequential(nn.Linear(self.feature_layer_size(), 
                                            hidden_layer_size),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(hidden_layer_size, 
                                            FLOWER_CAT_COUNT),
                                    nn.LogSoftmax(dim=1))


    def init_feature_model(self):

        if self.feature_model_name == FEATURE_MODEL_DENSNET:
            self.model = models.densenet121(pretrained=True)
        elif self.feature_model_name == FEATURE_MODEL_RESNET:
            self.model = models.resnet101(pretrained=True)
        elif self.feature_model_name == FEATURE_MODEL_VGG:
            self.model = models.vgg11(pretrained=True)
        else:
            raise Exception


    def feature_layer_size(self):

        if self.feature_model_name == FEATURE_MODEL_DENSNET:
            result = FEATURE_MODEL_DENSNET_SIZE
        elif self.feature_model_name == FEATURE_MODEL_RESNET:
            result = FEATURE_MODEL_RESNET_SIZE
        elif self.feature_model_name == FEATURE_MODEL_VGG:
            result = FEATURE_MODEL_VGG_SIZE
        else:
            raise Exception

        return result


    def classifier(self):

        if self.feature_model_name == FEATURE_MODEL_DENSNET:
            result = self.model.classifier
        elif self.feature_model_name == FEATURE_MODEL_RESNET:
            result = self.model.fc
        elif self.feature_model_name == FEATURE_MODEL_VGG:
            result = self.model.classifier[-1]
        else:
            raise Exception

        return result


    def classifier_parameters(self):

        if self.feature_model_name == FEATURE_MODEL_DENSNET:
            result = self.model.classifier.parameters()
        elif self.feature_model_name == FEATURE_MODEL_RESNET:
            result = self.model.fc.parameters()
        elif self.feature_model_name == FEATURE_MODEL_VGG:
            result = self.model.classifier.parameters()
        else:
            raise Exception

        return result


    def get_device(self, on_gpu=False):

        if on_gpu and torch.cuda.is_available():
            device = torch.device(GPU_DEVICE)
        else:
            device = torch.device(CPU_DEVICE)

        return device


    def create_idx_to_cat(self, cat_to_idx):
        return {val: key for key, val in cat_to_idx.items()}


    def get_dataloaders(self, data_dir):

        trng_data_dir = str(data_dir) + '/train'
        valn_data_dir = str(data_dir) + '/valid'

        # Load the datasets with ImageFolder
        trng_dataset = datasets.ImageFolder(trng_data_dir, 
                                            transform=self.trng_transforms)

        valn_dataset = datasets.ImageFolder(valn_data_dir, 
                                            transform=self.pred_transforms)

        # Define the dataloaders using the image datasets and trainforms
        trng_dataloader = torch.utils.data.DataLoader(trng_dataset, 
                                                    batch_size=64, 
                                                    shuffle=True)

        valn_dataloader = torch.utils.data.DataLoader(valn_dataset,
                                                    batch_size=64,
                                                    shuffle=True)

        return trng_dataloader, valn_dataloader


    def train(self, data_dir, max_epochs, learning_rate, on_gpu=False):
        trng_dataloader, valn_dataloader = self.get_dataloaders(data_dir)
        self.idx_to_cat = self.create_idx_to_cat(trng_dataloader.dataset.class_to_idx)
        device = self.get_device(on_gpu)

        optimiser = optim.Adam(self.classifier_parameters(), learning_rate)
        criterion = nn.NLLLoss()
        self.model.to(device)

        trng_losses, valn_losses = [], []
        for epoch in range(max_epochs):

            epoch_start_time = time.time()
            
            total_trng_loss = 0
            self.model.train()
            for images, exp_results in trng_dataloader:
                images, exp_results = images.to(device), exp_results.to(device)
                optimiser.zero_grad()
                
                log_pred_results = self.model(images)
                loss = criterion(log_pred_results, exp_results)
                total_trng_loss += loss.item()

                loss.backward()
                optimiser.step()
                
            total_valn_loss = 0
            total_correct = 0
            self.model.eval()
            with torch.no_grad():
                for images, exp_results in valn_dataloader:
                    images, exp_results = images.to(device), exp_results.to(device)

                    log_pred_results = self.model(images)

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


        def get_image(self, image_path):
            with Image.open(image_path) as im:
                return self.pred_transforms(im)


        def predict(self, image_path, topk=1, on_gpu=False):
            image = self.get_image(image_path)
            device = self.get_device(on_gpu)
            self.model.eval()
            self.model.to(device)

            with torch.no_grad():
                pred_results = torch.exp(self.model(image.unsqueeze(dim=0).to(device)))

            top_probs, top_results = pred_results.topk(topk, dim=1)

            top_probs = top_probs.squeeze_().tolist()
            top_cats = [self.idx_to_cat[x.item()] for x in top_results.squeeze_()]

            return top_probs, top_cats