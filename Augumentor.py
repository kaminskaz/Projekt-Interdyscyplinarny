
import cv2
import random
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import os


class Augumentor:
    def __init__(self):
        self.aug_dictionary = {}

   
    
    def load_image(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Error: File '{path}' not found!")

        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # Load PNG with transparency support
        if image is None:
            raise ValueError(f"Error: Failed to load image from {path}. It may be corrupted or unsupported.")

        print(f"Loaded image shape: {image.shape}, dtype: {image.dtype}")

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
            splits = [0, length]  # Start and end boundaries

            while len(splits) < num_splits + 1:
                split = random.randint(min_space, length - min_space)  # Ensure within bounds
                if all(abs(split - s) >= min_space for s in splits):
                    splits.append(split)

            splits = sorted(set(splits))  # Ensure unique, sorted splits
            return splits
        
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


                
    def augment_image(self, image, x_splits_number, y_splits_number, min_space_between_splits):
        """Applies augmentation (vertical flip) to each segment and reconstructs the image."""
        augmented_image = np.copy(image)
        x_splits, y_splits = self.split_image(image.shape[1], image.shape[0], x_splits_number, y_splits_number, min_space_between_splits)
        for x1, y1, x2, y2, segment in self.iterate_over_image(image, x_splits, y_splits):
            augumented_segment = self.rotate(segment, 12)
            augmented_image[y1:y2, x1:x2] = augumented_segment
        return augmented_image
    
    def vertical_flip(self, image):
        if image.size == 0:
            raise ValueError("Received an empty image for flipping!")

        transform = A.VerticalFlip(p=1)
        transformed = transform(image=image)

        if transformed['image'].shape != image.shape:
            print(f"Shape mismatch: original {image.shape}, flipped {transformed['image'].shape}")

        return transformed['image']
    
    def rotate(self, image, angle):
        transform = A.Rotate((angle, angle), p=0.7)
        transformed = transform(image=image)
        return transformed['image']

    
    def save_image(self, image, path):
        cv2.imwrite(path, image)
    
    def process_image(self, path, output_path, x_splits_number, y_splits_number, min_space_between_splits):
        image = self.load_image(path)

        # Display image using Matplotlib
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # Hide axis
        plt.show()

        augmented_image = self.augment_image(image, x_splits_number, y_splits_number, min_space_between_splits)
        self.save_image(augmented_image, output_path)
        #return augmented_image