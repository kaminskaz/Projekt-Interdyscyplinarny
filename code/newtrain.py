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
from STESAugmentor import STESAugmentor
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
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Pretrained normalization
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

# Load untransformed dataset for visualizing original images
if dataset == (cifar10_train, cifar10_test):
    base_dataset_raw = CIFAR10(root="../data", train=True, transform=None, download=False)
elif dataset == (fashion_mnist_train, fashion_mnist_test):
    base_dataset_raw = FashionMNIST(root="../data", train=True, transform=None, download=False)
else:
    raise ValueError("Unsupported dataset for visualization")


if aug == "stes":
    augmentor = STESAugmentor()
else:
    augmentor = Augmentor(p=.25)


# Load EfficientNet models (both pretrained and not pretrained)
mobilenet = models.mobilenet_v2(pretrained=False)


# Hyperparameters
batch_size = 64
epochs = 20 # You can adjust the number of epochs
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, seed, train_loader, optimizer, criterion, device, epochs=5, mode=None, learning_rate=0.001):
    model.train()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    hist = pd.DataFrame(columns=["epoch", "loss", "accuracy", "recall", "precision", "f1"])
    res = pd.DataFrame(columns=["epoch", "loss", "accuracy", "recall", "precision", "f1"])

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch + 1}/{epochs}")
        running_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Compute training metrics
        recall = metrics.recall_score(all_labels, all_preds, average='macro', zero_division=0)
        precision = metrics.precision_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = metrics.f1_score(all_labels, all_preds, average='macro', zero_division=0)
        train_accuracy = 100 * correct / total

        hist = pd.concat([hist, pd.DataFrame([{
            "epoch": epoch + 1,
            "loss": running_loss / len(train_loader),
            "accuracy": train_accuracy,
            "recall": recall,
            "precision": precision,
            "f1": f1
        }])], ignore_index=True)

        # Evaluate on validation set
        res, dataset_wrapper = evaluate(model, dataloader_val, criterion, device, res, epoch)
        val_accuracy = res.iloc[-1]["accuracy"]

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"New best validation accuracy: {val_accuracy:.2f}%")

    # Save best model
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), f"{aug}_{xy}_{seed}.pth")
    print(f"Saved best model for seed {seed} with accuracy {best_val_accuracy:.2f}%")

    # Save metrics
    hist.to_csv(f"{aug}_{xy}_{seed}.csv", index=False)
    res.to_csv(f"{aug}_{xy}_{seed}_val.csv", index=False)
    print(f"Saved metrics: {aug}_{xy}_{seed}.csv and {aug}_{xy}_{seed}_val.csv")

    return model



def evaluate(model, test_loader, criterion, device, res, epoch=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics across the entire test set
    recall = metrics.recall_score(all_labels, all_preds, average='macro', zero_division=0)
    precision = metrics.precision_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = metrics.f1_score(all_labels, all_preds, average='macro', zero_division=0)

    if epoch is not None:
        new_row = pd.DataFrame([{
            "epoch": epoch + 1,
            "loss": running_loss / len(test_loader),
            "accuracy": 100 * correct / total,
            "recall": recall,
            "precision": precision,
            "f1": f1
        }])
    else:
        new_row = pd.DataFrame([{
            "loss": running_loss / len(test_loader),
            "accuracy": 100 * correct / total,
            "recall": recall,
            "precision": precision,
            "f1": f1
        }])

    res = pd.concat([res, new_row], ignore_index=True)

    if isinstance(test_loader.dataset, Subset):
        dataset_wrapper = test_loader.dataset.dataset
    else:
        dataset_wrapper = test_loader.dataset

    return res, dataset_wrapper


    
    
dataset_wrapped = DatasetWrapper(dataset[0], augmentor,  x_splits_number=n_vertical_splits, y_splits_number=n_horizontal_splits, aug = aug)
dataset_wrapped.mode = mode

#Save sample images
# Create directory for samples
sample_dir = "./sample_images"
os.makedirs(sample_dir, exist_ok=True)

def tensor_to_numpy(tensor):
    """
    Convert a tensor in [0,1] to a displayable NumPy image in uint8.
    Assumes tensor is (C, H, W).
    """
    if isinstance(tensor, torch.Tensor):
        img = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        img = tensor
    else:
        raise TypeError("Input must be a Tensor or ndarray.")

    if img.shape[0] == 3 or img.shape[0] == 1:
        img = np.transpose(img, (1, 2, 0))  # (C, H, W) -> (H, W, C)

    img = np.clip(img, 0, 1) * 255
    return img.astype(np.uint8)


# is grayscale
dataset_name = args.get('dataset', 'cifar10')
is_grayscale = dataset_name == "fashion"

def print_image_stats(self,name, img):
    # if isinstance(img, torch.Tensor):
    #     arr = img.detach().cpu().numpy()
    #     print(f"{name}: Tensor - shape: {img.shape}, dtype: {img.dtype}, min: {arr.min():.3f}, max: {arr.max():.3f}")
    # elif isinstance(img, np.ndarray):
    #     print(f"{name}: ndarray - shape: {img.shape}, dtype: {img.dtype}, min: {img.min():.3f}, max: {img.max():.3f}")
    # elif isinstance(img, Image.Image):
    #      print(f"{name}: PIL - size: {img.size}, mode: {img.mode}")
    # else:
    #     print(f"{name}: Unknown type {type(img)}")
    pass

for i in range(5):
    # Original PIL image
    pil_img, label = base_dataset_raw[i]
    #pil_img.save(os.path.join(sample_dir, f"sample_{i}_label_{label}_original.png"))

    # Just resized + tensor (simulate what model sees without aug)
    tensor_img = get_transform(is_grayscale=is_grayscale)(pil_img)
    img_normalized = tensor_to_numpy(tensor_img)
    plt.imsave(os.path.join(sample_dir, f"sample_{i}_label_{label}_transformed.png"), img_normalized)

    # Full augmented version (from dataset wrapper)
    augmented_tensor, _ = dataset_wrapped[i]
    img_augmented = tensor_to_numpy(augmented_tensor)
    plt.imsave(os.path.join(sample_dir, f"sample_{i}_label_{label}_augmented.png"), img_augmented)


### end of sample images

indices = list(range(len(dataset_wrapped)))
labels = [dataset_wrapped[idx][1] for idx in indices]
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=1, stratify=labels)
train_subset = Subset(dataset_wrapped, train_idx)
val_subset = Subset(dataset_wrapped, val_idx)

for i in range(1,7):
    seed = i
    seed_everything(seed)
    dataloader_train = DataLoader(train_subset, batch_size=batch_size,num_workers = 8, shuffle=True, pin_memory=True)
    dataloader_val = DataLoader(val_subset, batch_size=batch_size,num_workers = 8, shuffle=False, pin_memory=True)

    model = copy.deepcopy(mobilenet)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=in_features, out_features=dataset_wrapped.num_classes(), bias=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = train(model, seed, dataloader_train, optimizer, criterion, device, epochs = epochs, mode=mode)

    # Evaluate on test set
    dataloader_test = DataLoader(DatasetWrapper(dataset[1]), batch_size=batch_size, shuffle=False)
    res = pd.DataFrame(columns=["loss", "accuracy", "recall", "precision", "f1"])
    res, dataset_wrapper = evaluate(model, dataloader_test, criterion, device, res)
    res.to_csv(f"{aug}_{xy}_{seed}_test.csv", index=False)

