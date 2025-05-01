import torch
from torch.utils.data import Dataset
import numpy as np
from Augmentor import Augmentor
from STESAugmentor import STESAugmentor
from PIL import Image
import albumentations as A


class DatasetWrapper(Dataset):
    def __init__(self, dataset, augmentor=None, mode=None, p = 0.5):
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
        image_np = np.array(image)
        if image_np.ndim == 3 and image_np.shape[0] in [1, 3, 4]:
            #print("[DEBUG] Permuting (C, H, W) -> (H, W, C)")
            image_np = np.transpose(image_np, (1, 2, 0))
        if self.augmentor:
            if isinstance(self.augmentor, Augmentor) and self.mode is not None:
                self.augmentor.p = self.p
                if self.mode not in ['same', 'different', 'combine']:
                    print("WARNING! Invalid mode. Choose 'same', 'different', or 'combine'. Changing to default 'different'.")
                    self.mode = 'different'
                    self.augmentor.mode = self.mode
                image_aug = self.augmentor.augment_image(image_np, mode=self.mode)
                #permute back to (C, H, W) if needed
                image_aug = np.transpose(image_aug, (2, 0, 1))
                image = image_aug
                
            elif isinstance(self.augmentor, STESAugmentor):
                image_aug = self.augmentor.augment_image(image_np)
                #permute back to (C, H, W) if needed
                image_aug = np.transpose(image_aug, (2, 0, 1))
                image = image_aug #do dodania ewentualnie parametry alpha, beta

        return image, label