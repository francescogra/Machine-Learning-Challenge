import torch
from torchvision.models import resnet18
from torchvision.transforms import transforms
from torchvision import datasets
import numpy as np
from torch.autograd import Variable
from PIL import Image
import argparse
import json
import io
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from IPython.display import display

with open("hyper_params.json") as hp:
    data = json.load(hp)
    num_class = data["num_classes"]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

device = torch.device("cpu")
model_path = 'D:/uni/magistrale/materie/Machine learning/progetto ml/Progetto_Grasso_Francesco_ML/models/trained.model'
# model = resnet18(num_classes)
model = torch.load(model_path, map_location=torch.device('cpu'))
count = count_parameters(model)
print(count)
# model.load_state_dict(checkpoint)
model.to(device)
print(model.eval())
data_transforms = {
    'predict': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    }

def calculate_prevision(model, dataloader):
    # Calcola le previsioni del modello sull'intero set di dati di test
    y_pred = []
    y_true = []
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(labels.numpy())
    # Restituisce le previsioni del modello e le etichette di classe effettive per l'intero set di dati di test
    return y_pred, y_true

if __name__ == '__main__':
    dataset = {'predict' : datasets.ImageFolder("./dataset/market/val", data_transforms['predict'])}
    dataloader = {'predict': torch.utils.data.DataLoader(dataset['predict'], batch_size = 1, shuffle=True, num_workers=4)}

    device = torch.device("cpu")
    outputs = list()
    since = time.time()
    NumberCircle = 0
    NumberMatch = 0
    for inputs, labels in dataloader['predict']:
        inputs = inputs.to(device)
        output = model(inputs)
        output = output.to(torch.device('cpu'))
        index = output.data.numpy().argmax()

        print("Input is {}, predicted class: {}".format(labels.numpy(), index))
        NumberCircle = NumberCircle + 1

        if labels.numpy() == index:
            NumberMatch = NumberMatch + 1

    print(since-time.time())
    print("Numero di giri {}, Numero di predizioni esatte: {}".format(NumberCircle, NumberMatch))

    #Calcolo la confusion matrix
    y_pred, y_true = calculate_prevision(model, dataloader['predict'])

    matrix = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)

    print('Confusion matrix:', matrix)
    print('f1_score:', f1)
    print('precision_score:', precision)
    print('recall_score:', recall)
    print('accuracy_score:', accuracy)