import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, FashionMNIST, Food101
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from Augumentor import Augumentor


# Initialize the augmentor
augmentor = Augumentor()

# Custom augmentation function for PyTorch transforms
def custom_augmentation(image):
    # Convert PIL Image to a NumPy array (since OpenCV works with NumPy)
    image_np = np.array(image)  
    
    # Apply your custom augmentation ('different' mode can be changed to 'same' or 'combine')
    augmented_np = augmentor.augment_image(image_np, x_splits_number=3, y_splits_number=2, 
                                           min_space_between_splits=5, mode='different')
    
    # Convert the augmented image back to PIL format
    return Image.fromarray(cv2.cvtColor(augmented_np, cv2.COLOR_BGR2RGB))

def get_transform(is_grayscale=False, is_train_dataset=True):
    if is_train_dataset:
        transform_list = [
            transforms.Lambda(custom_augmentation),  # Apply custom augmentation first
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),  # Resize to 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Pretrained normalization
        ]
    else:
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

# Load training datasets
cifar10_train = CIFAR10(root="./data", train=True, transform=get_transform(is_train_dataset=True), download=False)
fashion_mnist_train = FashionMNIST(root="./data", train=True, transform=get_transform(is_grayscale=True, is_train_dataset=True), download=False)

# Load test datasets
cifar10_test = CIFAR10(root="./data", train=False, transform=get_transform(is_train_dataset=False), download=False)
fashion_mnist_test = FashionMNIST(root="./data", train=False, transform=get_transform(is_grayscale=True, is_train_dataset=False), download=False)

# Create DataLoaders
batch_size = 64

# DataLoader for training datasets
cifar10_train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
fashion_mnist_train_loader = DataLoader(fashion_mnist_train, batch_size=batch_size, shuffle=True)

# DataLoader for test datasets
cifar10_test_loader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False)
fashion_mnist_test_loader = DataLoader(fashion_mnist_test, batch_size=batch_size, shuffle=False)


# Define class names for both datasets
cifar10_classes = cifar10_train.classes
fashion_mnist_classes = fashion_mnist_train.classes

cifar10_images, cifar10_labels = next(iter(cifar10_train_loader))
fashion_mnist_images, fashion_mnist_labels = next(iter(fashion_mnist_train_loader))
# Plotting CIFAR10 images
def plot_images(images, labels, classes, title):
    plt.figure(figsize=(12, 6))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].permute(1, 2, 0).numpy())
        plt.title(classes[labels[i]])
        plt.axis('off')
    plt.suptitle(title)
    plt.show()
# Plot CIFAR10 images
plot_images(cifar10_images, cifar10_labels, cifar10_classes, "CIFAR10 Images with Custom Augmentation")
# Plot FashionMNIST images
plot_images(fashion_mnist_images, fashion_mnist_labels, fashion_mnist_classes, "FashionMNIST Images with Custom Augmentation")
