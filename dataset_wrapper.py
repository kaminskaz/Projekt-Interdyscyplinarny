import torch
from torch.utils.data import Dataset

class DatasetWrapper(Dataset):
    def __init__(self, dataset, augmentor=None, type=None):
        """
        Args:
            dataset (Dataset): A PyTorch dataset object.
            augmentor (Augumentor): A custom augumentor class obejct.
            type (str): A string indicating the type of augumentation (in Augumentor could be "same", "different", or "combine).
        """
        self.dataset = dataset
        self.augmentor = augmentor
        self.type = type

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image, label = sample

        if self.augmentor:
            if isinstance(self.augmentor, Augmentor) and self.type is not None:
                if self.type not in ['same', 'different', 'combine']:
                    raise ValueError("Invalid type. Choose 'same', 'different', or 'combine'.")
                image = self.augmentor.augument_image(image, type=self.type) #do dodania ewentualnie parametry num_splits itd, w zależności czy to ustalone będzie czy losowane gdzieś
            elif isinstance(self.augmentor, STESAugmentor):
                image = self.augmentor.augment_image(image) #do dodania ewentualnie parametry alpha, beta

        return image, label