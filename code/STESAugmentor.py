import cv2
import random
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import os

class STESAugmentor:
    def __init__(self, seed=123):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def load_image(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Error: File '{path}' not found!")

        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Error: Failed to load image from {path}.")

        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        self.image = image
        return self.image

    def grid_partition(self, height, width, alpha=0.1, beta=0.3):
        L = min(height, width)
        l = random.randint(int(alpha * L), int(beta * L))
        Gh, Gw = l, l
        dh, dw = height % l, width % l
        return Gh, Gw, dh, dw



    def rotate_segment(self, segment, limit):
        transform = A.Compose([
            A.Rotate(limit=(limit,limit), p=1.0) 
        ])
        augmented = transform(image=segment)
        rotated = augmented['image']
        return rotated

    
    def apply_color_space_transform(self, segment):
        transform = A.Compose([
            A.HueSaturationValue(hue_shift_limit=(10, 350), sat_shift_limit=(0,0), val_shift_limit=(0,0), p=1.0)
        ])
        augmented = transform(image=segment)
        transformed_segment = augmented['image']
        return transformed_segment

    def augment_image(self, image, alpha=0.1, beta=0.3):
        if image.shape[-1] == 4:
            print("Alpha channel detected, converting from BGRA to BGR")
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)  
        augmented_image = np.copy(image)
        Gh, Gw, dh, dw = self.grid_partition(image.shape[0], image.shape[1], alpha, beta)
        chosen_segment_index = random.randint(0, 3)
        rotation_limit = random.randint(45, 315)

        for y in range(0, image.shape[0], Gh * 2):
            for x in range(0, image.shape[1], Gw * 2):
                segment_group = []
                for dy in range(0, Gh * 2, Gh):
                    for dx in range(0, Gw * 2, Gw):
                        segment = image[y+dy:y+dy+Gh, x+dx:x+dx+Gw]
                        if segment.size == 0:
                            continue
                        segment_group.append(segment)

                if len(segment_group) == 4:
                    chosen_segment = segment_group[chosen_segment_index]
                    
                    # Apply random color space transformation
                    chosen_segment = self.apply_color_space_transform(chosen_segment)

                    rotated_segment = self.rotate_segment(chosen_segment, rotation_limit)
                    rotated_segment = cv2.resize(rotated_segment, (Gw, Gh))

                    # Apply color adjustment to the rotated segment
                    color_adjustment = random.uniform(0.5, 1.5)
                    rotated_segment = cv2.convertScaleAbs(rotated_segment, alpha=color_adjustment, beta=0)

                    # Calculate correct offset based on the chosen segment
                    if chosen_segment_index == 0:
                        offset_x, offset_y = 0, 0
                    elif chosen_segment_index == 1:
                        offset_x, offset_y = Gw, 0
                    elif chosen_segment_index == 2:
                        offset_x, offset_y = 0, Gh
                    else:
                        offset_x, offset_y = Gw, Gh

                    # Handle edge cases: check if the segment fits within image dimensions
                    target_y_start = y + offset_y
                    target_x_start = x + offset_x
                    target_y_end = target_y_start + Gh
                    target_x_end = target_x_start + Gw

                    if target_y_end <= image.shape[0] and target_x_end <= image.shape[1]:
                        augmented_image[target_y_start:target_y_end, target_x_start:target_x_end] = rotated_segment

        return augmented_image




    def save_image(self, image, path):
        cv2.imwrite(path, image)

    def process_image(self, input_path, output_path, alpha=0.1, beta=0.3, image=None):
        if input_path:
            image = self.load_image(input_path)
        augmented_image = self.augment_image(image, alpha, beta)
        self.save_image(augmented_image, output_path)
        self.display_images(image, augmented_image)

    def display_images(self, original_image, augmented_image):
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
