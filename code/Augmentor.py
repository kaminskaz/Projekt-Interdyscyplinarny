
import cv2
import random
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import os

def rotate_segment(image, limit=15):
    transform = A.Compose([
        A.Rotate(limit=(-limit,limit), p=1.0) 
    ])
    augmented = transform(image=image)
    rotated = augmented['image']
    return rotated


def apply_color_space_transform(image):
    transform = A.Compose([
        A.HueSaturationValue(hue_shift_limit=(10, 350), sat_shift_limit=(0,0), val_shift_limit=(0,0), p=1.0)
    ])
    augmented = transform(image=image)
    transformed_segment = augmented['image']
    return transformed_segment



class Augmentor:
    def __init__(self, p=0.8, seed=123):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        self.p = p
    
        # self.aug_dictionary = {
        #     1: {  # flips and mirrors
        #         1: lambda p: A.HorizontalFlip(p=self.p), 
        #         2: lambda p: A.VerticalFlip(p=self.p),  
        #         3: lambda p: A.Compose([A.HorizontalFlip(p=self.p), A.VerticalFlip(p=self.p)])
        #     },
        #     2: {  # rotations
        #         1: lambda p: A.Rotate(limit=(5,5), p=self.p), 
        #         2: lambda p: A.Rotate(limit=(10,10), p=self.p), 
        #         3: lambda p: A.Rotate(limit=(15,15), p=self.p),  
        #         4: lambda p: A.Rotate(limit=(20,20), p=self.p)  
        #     },
        #     3: {  # blurs
        #         1: lambda p: A.GaussianBlur(blur_limit=(5, 5), p=self.p), 
        #         2: lambda p: A.GaussianBlur(blur_limit=(8, 8), p=self.p), 
        #         3: lambda p: A.GaussianBlur(blur_limit=(11, 11), p=self.p),
        #         4: lambda p: A.GaussianBlur(blur_limit=(14, 14), p=self.p)
        #     },
        #     4: {  # brightness and contrast
        #         1: lambda p: A.RandomBrightnessContrast(brightness_limit=(0.1, 0.1), contrast_limit=(0.1, 0.1), p=self.p), 
        #         2: lambda p: A.RandomBrightnessContrast(brightness_limit=(0.2, 0.2), contrast_limit=(0.2, 0.2), p=self.p),  
        #         3: lambda p: A.RandomBrightnessContrast(brightness_limit=(0.3, 0.3), contrast_limit=(0.3, 0.3), p=self.p), 
        #         4: lambda p: A.RandomBrightnessContrast(brightness_limit=(0.4, 0.4), contrast_limit=(0.4, 0.4), p=self.p) 
        #     },
        #     5: {  # noises
        #         1: lambda p: A.GaussNoise(std_range=(0.1,0.1), p=self.p),
        #         2: lambda p: A.GaussNoise(std_range=(0.2,0.2), p=self.p),  
        #         3: lambda p: A.GaussNoise(std_range=(0.3,0.3), p=self.p),  
        #         4: lambda p: A.GaussNoise(std_range=(0.4,0.4), p=self.p) 
        #     },
        #     6: {  # color adjustments
        #         1: lambda p: A.HueSaturationValue(hue_shift_limit=(10,10), sat_shift_limit=(10, 10), val_shift_limit=(10, 10), p=self.p),  
        #         2: lambda p: A.HueSaturationValue(hue_shift_limit=(20, 20), sat_shift_limit=(20, 20), val_shift_limit=(20, 20), p=self.p), 
        #         3: lambda p: A.HueSaturationValue(hue_shift_limit=(30, 30), sat_shift_limit=(30, 30), val_shift_limit=(30, 30), p=self.p), 
        #     },
        #     7: {  # negative & black and white
        #         1: lambda p: A.InvertImg(p=self.p),
        #         2: lambda p: A.ToGray(p=self.p),
        #     },
        # }
        self.aug_dictionary = {
            1: {  # rotations
                1: lambda p: A.Rotate(limit=(-5,5), p=self.p), 
                2: lambda p: A.Rotate(limit=(-10,10), p=self.p), 
                3: lambda p: A.Rotate(limit=(-12,12), p=self.p),  
                4: lambda p: A.Rotate(limit=(-15,15), p=self.p)  
            },
            2: {  # blurs
                1: lambda p: A.GaussianBlur(blur_limit=(5, 5), p=self.p), 
                2: lambda p: A.GaussianBlur(blur_limit=(8, 8), p=self.p), 
                3: lambda p: A.GaussianBlur(blur_limit=(11, 11), p=self.p),
                4: lambda p: A.GaussianBlur(blur_limit=(14, 14), p=self.p)
            },
            3: {  # brightness and contrast
                1: lambda p: A.RandomBrightnessContrast(brightness_limit=(0.03, 0.03), contrast_limit=(0.03, 0.03), p=self.p), 
                2: lambda p: A.RandomBrightnessContrast(brightness_limit=(0.05, 0.05), contrast_limit=(0.05, 0.05), p=self.p),  
                3: lambda p: A.RandomBrightnessContrast(brightness_limit=(0.08, 0.08), contrast_limit=(0.08, 0.08), p=self.p), 
                4: lambda p: A.RandomBrightnessContrast(brightness_limit=(0.1, 0.1), contrast_limit=(0.1, 0.1), p=self.p) 
            },
            4: {  # noises
                1: lambda p: A.GaussNoise(std_range=(0.01,0.01), p=self.p),
                2: lambda p: A.GaussNoise(std_range=(0.02,0.02), p=self.p),  
                3: lambda p: A.GaussNoise(std_range=(0.03,0.03), p=self.p),  
                4: lambda p: A.GaussNoise(std_range=(0.05,0.05), p=self.p) 
            },
            5: { #flips and mirrors
                1: lambda p: A.HorizontalFlip(p=self.p), 
                2: lambda p: A.VerticalFlip(p=self.p),  
                3: lambda p: A.Compose([A.HorizontalFlip(p=self.p), A.VerticalFlip(p=self.p)])
            },
            # 5: {  # color adjustments
            #     1: lambda p: A.HueSaturationValue(hue_shift_limit=(-1, 1), sat_shift_limit=(-1, 1), val_shift_limit=(-1, 1), p=self.p),  
            #     2: lambda p: A.HueSaturationValue(hue_shift_limit=(-2, 2), sat_shift_limit=(-2, 2), val_shift_limit=(-2, 2), p=self.p), 
            #     3: lambda p: A.HueSaturationValue(hue_shift_limit=(-3, 3), sat_shift_limit=(-3, 3), val_shift_limit=(-3, 3), p=self.p), 
            # },
        }

   
    
    def load_image(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Error: File '{path}' not found!")

        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # Load PNG with transparency support
        if image is None:
            raise ValueError(f"Error: Failed to load image from {path}. It may be corrupted or unsupported.")

        # print(f"Loaded image shape: {image.shape}, dtype: {image.dtype}")

        # Handle PNG with an alpha channel (convert to RGB)
        if image.shape[-1] == 4:
            print("Alpha channel detected, converting from BGRA to BGR")
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)  

        self.image = image
        return self.image

    def split_image(self, width, height, vertical_splits_number, horizontal_splits_number, min_space_between_splits):
        """
        Returns lists of x and y coordinates for splitting the image.
        Ensures the minimum space between splits is maintained.
        """
        def generate_splits(length, num_splits, min_space):
            if length < (num_splits + 1) * min_space:
                print(f"Warning: Length {length} is too small for {num_splits} splits with min space {min_space}. Adjusting.")
                num_splits = (length // min_space) - 1  # Adjust number of splits to fit the length
                if num_splits < 0:
                    raise ValueError("Cannot create splits with the given parameters.")


            max_possible_splits = (length // min_space) - 1
            if num_splits > max_possible_splits:
                print(f"Warning: Requested {num_splits} splits, but only {max_possible_splits} possible for length={length}. Adjusting.")
                num_splits = max_possible_splits

            splits = [0, length]  # Start and end boundaries
            attempts = 0
            max_attempts = 1000

            while len(splits) < num_splits + 1 and attempts < max_attempts:
                split = random.randint(min_space, length - min_space)
                if all(abs(split - s) >= min_space for s in splits):
                    splits.append(split)
                attempts += 1

            if len(splits) < num_splits + 1:
                raise RuntimeError(f"Could not find {num_splits} valid splits within {max_attempts} attempts. Got {len(splits)-2}.")

            return sorted(set(splits))
        
        x_splits = generate_splits(width, vertical_splits_number, min_space_between_splits)
        y_splits = generate_splits(height, horizontal_splits_number, min_space_between_splits)

        return x_splits, y_splits
    
    def extract_segment(self,image, x1, y1, x2, y2):
        return image[y1:y2, x1:x2]
    
    def iterate_over_image(self, image, x_splits, y_splits):
        for i in range(len(x_splits) - 1):
            for j in range(len(y_splits) - 1):
                x1, x2 = x_splits[i], x_splits[i + 1]
                y1, y2 = y_splits[j], y_splits[j + 1]

                # Ensure valid coordinates
                if x2 <= x1 or y2 <= y1:
                    print(f"Skipping invalid segment: ({x1}, {y1}, {x2}, {y2})")
                    continue

                segment = self.extract_segment(image, x1, y1, x2, y2)

                # Check for empty segment
                if segment.size == 0:
                    print(f"Warning: Extracted an empty segment at ({x1}, {y1}, {x2}, {y2})")
                    continue
                
                #print(f"Extracted segment shape: {segment.shape} at ({x1}, {y1}, {x2}, {y2})")
                yield x1, y1, x2, y2, segment

                
    def augment_image(self, image, mode, x_splits_number=0, y_splits_number=0, min_space_between_splits=10, aug=None):
        """Applies augmentation (vertical flip) to each segment and reconstructs the image."""
        if image.shape[-1] == 4:
            print("Alpha channel detected, converting from BGRA to BGR")
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)  
        augmented_image = np.copy(image)

        # in case we want to choose a random number of splits for further testing
        #x_splits_number = random.randint(0, x_splits_number)
        #y_splits_number = random.randint(0, y_splits_number)

        x_splits, y_splits = self.split_image(image.shape[1], image.shape[0], x_splits_number, y_splits_number, min_space_between_splits)

        if mode not in ['same', 'different', 'combine']:
            raise ValueError("Invalid mode. Choose 'same', 'different', or 'combine'.")
        
        if aug is None:
            if mode == 'same':
                aug_type = random.randint(1, len(self.aug_dictionary))
                for x1, y1, x2, y2, segment in self.iterate_over_image(image, x_splits, y_splits):
                    aug_power = random.randint(1, len(self.aug_dictionary[aug_type]))
                    augmented_segment = self.aug_dictionary[aug_type][aug_power](p=self.p)(image=segment)['image']
                    augmented_image[y1:y2, x1:x2] = augmented_segment

            if mode == 'different':
                for x1, y1, x2, y2, segment in self.iterate_over_image(image, x_splits, y_splits):
                    aug_type = random.randint(1, len(self.aug_dictionary))
                    aug_power = random.randint(1, len(self.aug_dictionary[aug_type]))
                    augmented_segment = self.aug_dictionary[aug_type][aug_power](p=self.p)(image=segment)['image']
                    augmented_image[y1:y2, x1:x2] = augmented_segment

            if mode == 'combine':
                for x1, y1, x2, y2, segment in self.iterate_over_image(image, x_splits, y_splits):
                    r = random.randint(1, 3)
                    chosen_augmentations = random.sample(list(self.aug_dictionary.keys()), r)
                    augmented_segment = segment.copy() 
                    for aug_type in chosen_augmentations:
                        aug_power = random.randint(1, len(self.aug_dictionary[aug_type]))
                        augment = self.aug_dictionary[aug_type][aug_power]
                        augmented_segment = augment(p=self.p)(image=augmented_segment)['image']
                    augmented_image[y1:y2, x1:x2] = augmented_segment
        else:
            if aug == 'rotate':
                for x1, y1, x2, y2, segment in self.iterate_over_image(image, x_splits, y_splits):
                    aug_power = random.randint(1, len(self.aug_dictionary[1]))
                    augmented_segment = self.aug_dictionary[1][aug_power](p=self.p)(image=segment)['image']
                    augmented_image[y1:y2, x1:x2] = augmented_segment
            elif aug == 'blur':
                for x1, y1, x2, y2, segment in self.iterate_over_image(image, x_splits, y_splits):
                    aug_power = random.randint(1, len(self.aug_dictionary[2]))
                    augmented_segment = self.aug_dictionary[2][aug_power](p=self.p)(image=segment)['image']
                    augmented_image[y1:y2, x1:x2] = augmented_segment
            elif aug == 'brightness':
                for x1, y1, x2, y2, segment in self.iterate_over_image(image, x_splits, y_splits):
                    aug_power = random.randint(1, len(self.aug_dictionary[3]))
                    augmented_segment = self.aug_dictionary[3][aug_power](p=self.p)(image=segment)['image']
                    augmented_image[y1:y2, x1:x2] = augmented_segment
            elif aug == 'noise':    
                for x1, y1, x2, y2, segment in self.iterate_over_image(image, x_splits, y_splits):
                    aug_power = random.randint(1, len(self.aug_dictionary[4]))
                    augmented_segment = self.aug_dictionary[4][aug_power](p=self.p)(image=segment)['image']
                    augmented_image[y1:y2, x1:x2] = augmented_segment
            elif aug == 'flip':
                for x1, y1, x2, y2, segment in self.iterate_over_image(image, x_splits, y_splits):
                    aug_power = random.randint(1, len(self.aug_dictionary[5]))
                    augmented_segment = self.aug_dictionary[5][aug_power](p=self.p)(image=segment)['image']
                    augmented_image[y1:y2, x1:x2] = augmented_segment
            elif aug == 'baseline':
                pass
            elif aug == 'combine':
                for x1, y1, x2, y2, segment in self.iterate_over_image(image, x_splits, y_splits):
                    r = random.random()
                    if r <= self.p:
                        colored_segment = apply_color_space_transform(segment)
                        augmented_segment = rotate_segment(image=colored_segment)
                        augmented_image[y1:y2, x1:x2] = augmented_segment
            elif aug in ['flip-rotate', 'flip-brightness', 'rotate-brightness','flip-rotate-brightness']:
                if aug == 'flip-rotate':
                    idxs = [5,1]
                elif aug == 'flip-brightness':
                    idxs = [5,3]
                elif aug == 'rotate-brightness':
                    idxs = [1,3]
                elif aug == 'flip-rotate-brightness':
                    idxs = [5,1,3]
                for x1, y1, x2, y2, segment in self.iterate_over_image(image, x_splits, y_splits):
                    aug_power = random.randint(1, len(self.aug_dictionary[5]))
                    augmented_segment = self.aug_dictionary[idxs[random.randint(0,len(idxs)-1)]][aug_power](p=self.p)(image=segment)['image']
                    augmented_image[y1:y2, x1:x2] = augmented_segment
                    
        return augmented_image

    
    def save_image(self, image, path):
        cv2.imwrite(path, image)
    
    def process_image(self, path = None, output_path = None, x_splits_number=0, y_splits_number=0, min_space_between_splits=0,  mode='different', image = None):
        
        if path is not None:
            image = self.load_image(path)

        augmented_image = self.augment_image(image=image, x_splits_number=x_splits_number, y_splits_number=y_splits_number, min_space_between_splits=min_space_between_splits, mode=mode)
        
        self.display_images(image, augmented_image)

        if output_path is not None:
            self.save_image(augmented_image, output_path)
            return
        else:
            return augmented_image

    def display_images(self, original_image, augmented_image):
        """Display original and augmented images side by side."""
        plt.figure(figsize=(10, 5)) 

        plt.subplot(1, 2, 1) 
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off') 

        plt.subplot(1, 2, 2)  
        plt.imshow(cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB))
        plt.title("Augmented Image")
        plt.axis('off')  

        plt.show()


