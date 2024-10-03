from __future__ import print_function, division
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from matplotlib import pyplot as plt
import time
import os
import copy
from torchmetrics import F1Score
from torchmetrics import ConfusionMatrix
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

input_dim = 224 # The input dimension for ResNet is 224

accuracy_res = None

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_dim),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_dim),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

with open("hyper_params.json") as hp:
    data = json.load(hp)
    # root_dir = data["root_directory"]
    root_dir = 'dataset/market'
    num_classes = data["num_classes"]
    num_epochs = data["num_epochs"]
    batch_size = data["batch_size"]
    num_workers = data["num_workers"]
    lr = data["learning_rate"]
    optim_name = data["optimizer"]
    momentum = data["momentum"]
    step_size = data["step_size"]
    gamma = data["gamma"]


image_datasets = {x: datasets.ImageFolder(os.path.join(root_dir, x),
                                          data_transforms[x])
            for x in ['train', 'val']}



dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)
            for x in ['train', 'val']}

#train_loaders = torch.utils.data.DataLoader(image_datasets['train'], batch_size = batch_size, sampler = sampler, shuffle = False, num_workers = num_workers)
#test_loaders = torch.utils.data.DataLoader(image_datasets['val'], batch_size = batch_size, shuffle = False, num_workers = num_workers)

#dataloaders = {'train':train_loaders,'val':test_loaders}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# Riordinamento delle classi secondo ordine numerico crescente --
class_names_sorted = []
class_names_idx = []

class_names = image_datasets['train'].classes
for cls in class_names:
    split = cls.split('_')
    class_names_idx.append(int(split[1]))

class_names_idx.sort()

for idx in class_names_idx:
    for name in class_names:
        name_split = name.split('_')
        if str(idx) == name_split[1]:
            class_names_sorted.append(name)

class_names = class_names_sorted
# --

class_map={}
for x in range (0,len(class_names)):
    class_map[x]=class_names[x]


with open('class_mapping.json', 'w') as outfile:  
    json.dump(class_map, outfile, indent = 4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print (device)
#device = torch.device("cpu")

def save_models(epochs, model):
    print()
    # torch.save(model.state_dict(), "./models/trained.model")
    torch.save(model, "./models/trained.model")
    print("****----Checkpoint Saved----****")
    print()

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('_' * 15)

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train' and epoch_acc > best_acc:
                save_models(epoch,model)
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print('Best val Acc: {:4f}'.format(best_acc))
    print("Accuracy result: {}".format(accuracy_res))

    model.load_state_dict(best_model_wts)
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    count = count_parameters(model_ft)
    print(count)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=momentum)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=num_epochs)


