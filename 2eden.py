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
from STESAugmentor import STESAugmentor
import random
import os
import copy
import pandas as pd
import sklearn.metrics as metrics

with open("device_info.txt", "w") as f:
    f.write(f"CUDA available: {torch.cuda.is_available()}\n")
    f.write(f"CUDA device count: {torch.cuda.device_count()}\n")
    
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_index)
        f.write(f"Current device index: {device_index}\n")
        f.write(f"Device name: {device_name}\n")
    else:
        f.write("Running on CPU")




SEED = 42

def seed_everything(seed: int=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)



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

# Load training datasets with no special augumentation
cifar10_train = CIFAR10(root="./data", train=True, transform=get_transform(), download=True)
fashion_mnist_train = FashionMNIST(root="./data", train=True, transform=get_transform(is_grayscale=True), download=True)

# Load test datasets with no special augumentation
cifar10_test = CIFAR10(root="./data", train=False, transform=get_transform(), download=True)
fashion_mnist_test = FashionMNIST(root="./data", train=False, transform=get_transform(is_grayscale=True), download=True)

datasets = [(cifar10_train, cifar10_test), (fashion_mnist_train, fashion_mnist_test)]

augmentor = Augmentor()
stes_augmentor = STESAugmentor()

augmentors = [augmentor, stes_augmentor, None]

cifar_10_train_wrapped_aug = DatasetWrapper(cifar10_train, augmentor, "different")
cifar_10_train_wrapped_stes = DatasetWrapper(cifar10_train, stes_augmentor)

fashion_mnist_train_wrapped_aug = DatasetWrapper(fashion_mnist_train, augmentor, "different")
fashion_mnist_train_wrapped_stes = DatasetWrapper(fashion_mnist_train, stes_augmentor)

# Create DataLoaders




# Define class names for both datasets
cifar10_classes = cifar10_train.classes
fashion_mnist_classes = fashion_mnist_train.classes

# Load EfficientNet models (both pretrained and not pretrained)
# EfficientNet-B0
efficientnet_b0_pretrained = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
efficientnet_b0_non_pretrained = models.efficientnet_b0(weights=None)

# EfficientNet-B1 (you can change to B2, B3, etc. similarly)
efficientnet_b7_pretrained = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
efficientnet_b7_non_pretrained = models.efficientnet_b7(weights=None)

models = [efficientnet_b0_pretrained, efficientnet_b0_non_pretrained,
         efficientnet_b7_pretrained, efficientnet_b7_non_pretrained]

model_names = [
    "EfficientNet-B0 Pretrained",
    "EfficientNet-B0 Non-Pretrained",
    "EfficientNet-B7 Pretrained",
    "EfficientNet-B7 Non-Pretrained"
]

modes = ["same", "different", "combine"]

# Hyperparameters
batch_size = 64
epochs = 20  # You can adjust the number of epochs
learning_rate = 0.001
optimizer = optim.Adam(efficientnet_b0_pretrained.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, model_name, train_loader, optimizer, criterion, device, epochs=5):
    model.train()
    hist = pd.DataFrame(columns=["epoch", "loss", "accuracy", "recall", "precision", "f1"])
    for epoch in range(epochs):
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
        recall = metrics.recall_score(labels.cpu(), predicted.cpu(), average='macro')
        precision = metrics.precision_score(labels.cpu(), predicted.cpu(), average='macro')
        f1 = metrics.f1_score(labels.cpu(), predicted.cpu(), average='macro')
        new_row = pd.DataFrame([{
            "epoch": epoch + 1,
            "loss": running_loss / len(train_loader),
            "accuracy": 100 * correct / total,
            "recall": recall,
            "precision": precision,
            "f1": f1
        }])
        hist = pd.concat([hist, new_row], ignore_index=True)
    hist.to_csv(f"{model_name}_{train_loader.dataset.get_name()}_{train_loader.dataset.augmentor.__class__.__name__}_{train_loader.dataset.mode}.csv", index=False)
    print(f"saved {model_name}_{train_loader.dataset.get_name()}_{train_loader.dataset.augmentor.__class__.__name__}_{train_loader.dataset.mode}.csv")
    torch.save(model.state_dict(), f"{model_name}_{train_loader.dataset.get_name()}_{train_loader.dataset.augmentor.__class__.__name__}_{train_loader.dataset.mode}.pth")
        

def evaluate(model,model_name, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    res = pd.DataFrame(columns=["loss", "accuracy", "recall", "precision", "f1"])
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
    recall = metrics.recall_score(labels.cpu(), predicted.cpu(), average='macro')
    precision = metrics.precision_score(labels.cpu(), predicted.cpu(), average='macro')
    f1 = metrics.f1_score(labels.cpu(), predicted.cpu(), average='macro')
    new_row = pd.DataFrame([{
        "loss": running_loss/len(test_loader),
        "accuracy": 100 * correct/total,
        "recall" : recall,
        "precision" : precision,
        "f1" : f1
    }])
    res = pd.concat([res, new_row], ignore_index=True)
    res.to_csv(f"{model_name}_{test_loader.dataset.get_name()}_{test_loader.dataset.augmentor.__class__.__name__}_{test_loader.dataset.mode}_test.csv", index=False)
    



for i in range(len(models)):
    for dataset in datasets:
        dataset_wrapped = DatasetWrapper(dataset[0], augmentor)
        for augmentor in augmentors:
            if isinstance(augmentor, Augmentor):
                for mode in modes:
                    dataset_wrapped.mode = mode
                    dataloader_train = DataLoader(dataset_wrapped, batch_size=batch_size, shuffle=True)
                    
                    model = copy.deepcopy(models[i])
                    model_name = model_names[i]
                    model.classifier[1] = nn.Linear(in_features=1280, out_features=dataset_wrapped.num_classes(), bias=True)

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model.to(device)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(), lr=0.001)

                    train(model, model_name, dataloader_train, optimizer, criterion, device, epochs = epochs)

                    dataloader_test = DataLoader(DatasetWrapper(dataset[1]), batch_size=batch_size, shuffle=False)
                    evaluate(model, model_name, dataloader_test, criterion, device)

            else:
                dataloader_train = DataLoader(dataset_wrapped, batch_size=batch_size, shuffle=True)
                
                model = copy.deepcopy(models[i])
                model_name = model_names[i]
                model.classifier[1] = nn.Linear(in_features=1280, out_features=dataset_wrapped.num_classes(), bias=True)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                train(model, model_name, dataloader_train, optimizer, criterion, device, epochs = epochs)

                dataloader_test = DataLoader(DatasetWrapper(dataset[1]), batch_size=batch_size, shuffle=False)
                evaluate(model, model_name, dataloader_test, criterion, device)
                


