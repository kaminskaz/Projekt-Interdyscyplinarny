import torch
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, FashionMNIST, Food101
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from Augmentor import Augmentor
from dataset_wrapper import DatasetWrapper

import random
import os
import copy
import pandas as pd
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
import albumentations as A

#get parameters
import sys
args = dict(arg.split('=') for arg in sys.argv[1:])
aug= args.get('aug', 'none')
xy = args.get('xy', '00')
n_horizontal_splits = int(xy[0])
n_vertical_splits = int(xy[1])
dataset = args.get('dataset', 'cifar10')
mode = args.get('mode', 'same')


dir = f'./{aug}_{xy}'

if not os.path.exists(dir):
    os.makedirs(dir)
os.chdir(dir)

try:
    os.system(f"chmod -R 777 {dir}")
except:
    print("Error changing permissions")



def seed_everything(seed: int=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#preprocessing
def get_transform(is_grayscale=False):
    transform_list = [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),  # Resize to 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Pretrained normalization
        ]
    if is_grayscale:
        transform_list = [
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
        ] + transform_list  # Keep grayscale conversion before augmentation & normalization
    
    return transforms.Compose(transform_list)

if dataset == "cifar10":
    cifar10_train = CIFAR10(root="../data", train=True, transform=get_transform(), download=True)
    cifar10_test = CIFAR10(root="../data", train=False, transform=get_transform(), download=True)
    dataset = (cifar10_train, cifar10_test)
    cifar10_classes = cifar10_train.classes

elif dataset == "fashion":
    fashion_mnist_train = FashionMNIST(root="../data", train=True, transform=get_transform(is_grayscale=True), download=True)
    fashion_mnist_test = FashionMNIST(root="../data", train=False, transform=get_transform(is_grayscale=True), download=True)
    dataset = (fashion_mnist_train, fashion_mnist_test)
    fashion_mnist_classes = fashion_mnist_train.classes
else:
    raise ValueError("Dataset not supported. Please choose 'cifar10' or 'fashion'.")

augmentor = Augmentor()


# Load EfficientNet models (both pretrained and not pretrained)
mobilenet = models.mobilenet_v2(pretrained=False)


# Hyperparameters
batch_size = 64
epochs = 20 # You can adjust the number of epochs
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, seed, train_loader, optimizer, criterion, device, epochs=5, mode=None, learning_rate = 0.001):
    model.train()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    hist = pd.DataFrame(columns=["epoch", "loss", "accuracy", "recall", "precision", "f1"])
    res = pd.DataFrame(columns=["epoch", "loss", "accuracy", "recall", "precision", "f1"])

    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch + 1}/{epochs}")
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        recall = metrics.recall_score(labels.cpu(), predicted.cpu(), average='macro', zero_division=0)
        precision = metrics.precision_score(labels.cpu(), predicted.cpu(), average='macro', zero_division=0)
        f1 = metrics.f1_score(labels.cpu(), predicted.cpu(), average='macro', zero_division=0)
        new_row = pd.DataFrame([{
            "epoch": epoch + 1,
            "loss": running_loss / len(train_loader),
            "accuracy": 100 * correct / total,
            "recall": recall,
            "precision": precision,
            "f1": f1
        }])
        hist = pd.concat([hist, new_row], ignore_index=True)

        # evaluate on validation set
        res, dataset_wrapper = evaluate(model, dataloader_val, criterion, device, res, epoch)

    res.to_csv(f"{aug}_{xy}_{seed}_val.csv", index=False)    

    hist.to_csv(f"{aug}_{xy}_{seed}.csv", index=False)
    print(f"saved {aug}_{xy}_{seed}.csv")
    torch.save(model.state_dict(), f"{aug}_{xy}_{seed}.pth")


def evaluate(model, test_loader, criterion, device, res, epoch=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # Calculate metrics
    recall = metrics.recall_score(labels.cpu(), predicted.cpu(), average='macro', zero_division=0)
    precision = metrics.precision_score(labels.cpu(), predicted.cpu(), average='macro', zero_division=0)
    f1 = metrics.f1_score(labels.cpu(), predicted.cpu(), average='macro', zero_division=0)
    if epoch is not None:
      new_row = pd.DataFrame([{
        "epoch": epoch + 1,
        "loss": running_loss/len(test_loader),
        "accuracy": 100 * correct/total,
        "recall" : recall,
        "precision" : precision,
        "f1" : f1
      }])
    else:
      new_row = pd.DataFrame([{
        "loss": running_loss/len(test_loader),
        "accuracy": 100 * correct/total,
        "recall" : recall,
        "precision" : precision,
        "f1" : f1
      }])

    res = pd.concat([res, new_row], ignore_index=True)
    if isinstance(test_loader.dataset, Subset):
        dataset_wrapper = test_loader.dataset.dataset
    else:
        dataset_wrapper = test_loader.dataset

    return res, dataset_wrapper

    
    
dataset_wrapped = DatasetWrapper(dataset[0], augmentor,  x_splits_number=n_vertical_splits, y_splits_number=n_horizontal_splits, aug = aug)
dataset_wrapped.mode = mode

indices = list(range(len(dataset_wrapped)))
labels = [dataset_wrapped[idx][1] for idx in indices]
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=1, stratify=labels)
train_subset = Subset(dataset_wrapped, train_idx)
val_subset = Subset(dataset_wrapped, val_idx)





for seed in range(1, 7):
    seed_everything(seed)
    dataloader_train = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    model = copy.deepcopy(mobilenet)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=in_features, out_features=dataset_wrapped.num_classes(), bias=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, seed, dataloader_train, optimizer, criterion, device, epochs = epochs, mode=mode)

    # Evaluate on test set
    dataloader_test = DataLoader(DatasetWrapper(dataset[1]), batch_size=batch_size, shuffle=False)
    res = pd.DataFrame(columns=["loss", "accuracy", "recall", "precision", "f1"])
    res, dataset_wrapper = evaluate(model, dataloader_test, criterion, device, res)
    res.to_csv(f"{aug}_{xy}_{seed}_test.csv", index=False)