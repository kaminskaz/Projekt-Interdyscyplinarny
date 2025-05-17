import random
import torch
from torch._C import parse_schema
from torch.utils.data import Dataset
import numpy as np
from Augmentor import Augmentor
from STESAugmentor import STESAugmentor
from PIL import Image
import albumentations as A


class DatasetWrapper(Dataset):
    def __init__(self, dataset, augmentor=None, mode=None, p = 0.5, x_splits_number=0, y_splits_number=0, aug = None, seed=42):
        """
        Args:
            dataset (Dataset): A PyTorch dataset object.
            augmentor (Augmentor): A custom Augmentor class obejct.
            mode (str): A string indicating the mode of augumentation (in Augmentor could be "same", "different", or "combine).
        """
        self.dataset = dataset
        self.augmentor = augmentor
        self.mode = mode
        self.p = p
        self.x_splits_number = x_splits_number
        self.y_splits_number = y_splits_number
        self.aug = aug
        
        #set all seeds
        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        #set augmentor seed
        if self.augmentor is not None:
            if isinstance(self.augmentor, Augmentor):
                self.augmentor.seed = self.seed
            elif isinstance(self.augmentor, STESAugmentor):
                self.augmentor.seed = self.seed
            else:
                raise ValueError("Invalid augmentor type. Use Augmentor or STESAugmentor.")
        random.seed(self.seed)


    def __len__(self):
        return len(self.dataset)
    
    def num_classes(self):
        return len(self.dataset.classes)

    def get_name(self):
        #get name and train or test
        return self.dataset.__class__.__name__ + self.dataset.train.__str__()

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image, label = sample

        print_image_stats(f"[{idx}] Original image (PIL or Tensor)", image)

        image_np = np.array(image)

        # Ensure pixel values are in [0, 255] for the augmentor
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
            print_image_stats(f"[{idx}] Converted to NumPy", image_np)

        if image_np.ndim == 3 and image_np.shape[0] in [1, 3, 4]:
            #print(f"[{idx}] Permuting from (C, H, W) to (H, W, C)")
            image_np = np.transpose(image_np, (1, 2, 0))
            print_image_stats(f"[{idx}] After permute", image_np)

        if self.augmentor:
            if isinstance(self.augmentor, Augmentor) and self.mode is not None:
                self.augmentor.p = self.p
                if self.mode not in ['same', 'different', 'combine']:
                    print("WARNING! Invalid mode. Choose 'same', 'different', or 'combine'. Changing to default 'different'.")
                    self.mode = 'different'
                self.augmentor.mode = self.mode

                #print(f"[{idx}] Applying Augmentor: mode={self.mode}, x={self.x_splits_number}, y={self.y_splits_number}")
                image_aug = self.augmentor.augment_image(
                    image_np,
                    mode=self.mode,
                    x_splits_number=self.x_splits_number,
                    y_splits_number=self.y_splits_number,
                    aug=self.aug
                )
                print_image_stats(f"[{idx}] After Augmentor", image_aug)
                image_aug = np.transpose(image_aug, (2, 0, 1))
                print_image_stats(f"[{idx}] After permute back to (C, H, W)", image_aug)
                image = image_aug

            elif isinstance(self.augmentor, STESAugmentor):
                print(f"[{idx}] Applying STESAugmentor")
                image_aug = self.augmentor.augment_image(image_np)
                print_image_stats(f"[{idx}] After STESAugmentor", image_aug)
                image_aug = np.transpose(image_aug, (2, 0, 1))
                print_image_stats(f"[{idx}] After permute back to (C, H, W)", image_aug)
                image = image_aug

        if isinstance(image, np.ndarray):
            image = torch.tensor(image / 255.0, dtype=torch.float32)
        elif isinstance(image, torch.Tensor):
            image = image.float()  # already normalized by ToTensor()
        else:
            raise TypeError("Unsupported image type")
        return image, label


def print_image_stats(name, img):
    # if isinstance(img, torch.Tensor):
    #     arr = img.detach().cpu().numpy()
    #     print(f"{name}: Tensor - shape: {img.shape}, dtype: {img.dtype}, min: {arr.min():.3f}, max: {arr.max():.3f}")
    # elif isinstance(img, np.ndarray):
    #     print(f"{name}: ndarray - shape: {img.shape}, dtype: {img.dtype}, min: {img.min():.3f}, max: {img.max():.3f}")
    # elif isinstance(img, Image.Image):
    #     print(f"{name}: PIL - size: {img.size}, mode: {img.mode}")
    # else:
    #     print(f"{name}: Unknown type {type(img)}")
    pass
