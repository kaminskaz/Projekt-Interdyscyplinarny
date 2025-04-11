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

from Augmentor import Augumentor
from dataset_wrapper import DatasetWrapper
from STESAugmentor import STESAugmentor
import random
import os
import copy


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

augmentor = Augumentor()
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
efficientnet_b1_pretrained = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
efficientnet_b1_non_pretrained = models.efficientnet_b1(weights=None)

models = [efficientnet_b0_pretrained, efficientnet_b0_non_pretrained,
         efficientnet_b1_pretrained, efficientnet_b1_non_pretrained]

modes = ["same", "different", "combine"]

# Hyperparameters
batch_size = 64
epochs = 50  # You can adjust the number of epochs
learning_rate = 0.001


def train(model, train_loader, optimizer, criterion, device, epochs=5):
    model.train()
    for epoch in range(epochs):
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
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct/total:.2f}%")

def evaluate(model, test_loader, criterion, device):
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
    
    print(f"Test Loss: {running_loss/len(test_loader):.4f}, Test Accuracy: {100 * correct/total:.2f}%")



for model in models:
    for dataset in datasets:
        for augmentor in augmentors:
            if isinstance(augmentor, Augumentor):
                for mode in modes:
                    pass
            elif isinstance(augmentor, STESAugmentor):
                pass


model = copy.deepcopy(model)  # Create a copy of the model
model.classifier[1] = nn.Linear(in_features=1280, out_features=len(dataset.num_classes()), bias=True)  # Adjust the classifier for the number of classes

# Set up device (CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop


# Start training
train(model, cifar10_train_loader, optimizer, criterion, device)

# Evaluate the model


# Test the model
evaluate(model, cifar10_test_loader, criterion, device)


torch.save(model.state_dict(), "efficientnet_cifar10_empty.pth")
print("Model saved as efficientnet_cifar10_empty.pth")