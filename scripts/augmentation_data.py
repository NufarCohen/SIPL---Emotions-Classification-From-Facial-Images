import albumentations as A
import cv2
import os
import random
import shutil
from pathlib import Path


# define the augmentation transform
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomResizedCrop(height=48, width=48, scale=(0.95, 1.0), ratio=(1.0, 1.0), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.GaussNoise(var_limit=(0.1, 0.5), p=0.5)
])

def create_augmentation(input_root, output_root, target_count):
    data_root_dir = Path(output_root)

    if data_root_dir.exists():
        shutil.rmtree(data_root_dir)
    os.makedirs(output_root, exist_ok=True)

    for class_name in os.listdir(input_root):
        input_folder = os.path.join(input_root, class_name)
        output_folder = os.path.join(output_root, class_name)
        os.makedirs(output_folder, exist_ok=True)

        images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        num_existing = len(images)

        # copy original images
        for i, filename in enumerate(images):
            src_path = os.path.join(input_folder, filename)
            dst_path = os.path.join(output_folder, f"orig_{i}.jpg")
            img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(dst_path, img)

        # augment images 
        i = num_existing
        while i < target_count:
            filename = random.choice(images)
            path = os.path.join(input_folder, filename)
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            image = image[:, :, None]  
            augmented = transform(image=image)["image"]
            aug_path = os.path.join(output_folder, f"aug_{i}.jpg")
            cv2.imwrite(aug_path, augmented.squeeze())  
            i += 1
    print(f"finish augmentation for {target_count} img per class")

def main():
    input_root = "big_dataset_notUniform"
    output_root = "balancedAugm_big_dataset"
    target_count = 8000
    create_augmentation(input_root, output_root, target_count)