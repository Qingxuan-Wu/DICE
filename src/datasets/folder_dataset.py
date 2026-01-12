import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import cv2

class FolderDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
        self.transform = transforms.Compose([
            transforms.Lambda(self.center_crop_min_side),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_3x = transforms.Compose([
            transforms.Lambda(self.center_crop_min_side),
            transforms.Resize((672, 672)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')  # Ensure 3-channel RGB image
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = Image.fromarray(image)
        img_3x = self.transform_3x(image)
        image = self.transform(image)
        data = {
            "single_img_seqs": image,
            "img_3x": img_3x,
            "img_name": "_".join(img_name.split("/")[-2:])
        }
        return data

    def center_crop_min_side(self, img):
        width, height = img.size
        min_side = min(width, height)
        left = (width - min_side) // 2
        top = (height - min_side) // 2
        right = (width + min_side) // 2
        bottom = (height + min_side) // 2
        return img.crop((left, top, right, bottom))

# Example usage:
# dataset = CustomImageDataset('/path/to/your/image/folder')
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
