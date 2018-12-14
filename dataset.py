import os

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class AVADataset(Dataset):
    """AVA dataset."""

    def __init__(
        self,
        root_dir="/home/wangx/datasets/AVA/AVA_dataset/images/images",
        data_file="/home/wangx/datasets/AVA/AVA_dataset/AVA_filtered.txt",
        transform=None,
    ):
        self.root_dir = root_dir
        self.data_file = data_file
        self.transform = transform
        self.data = pd.read_csv(
            self.data_file,
            sep=" ",
            names=[
                "Index",
                "Image ID",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
                "Tag1",
                "Tag2",
                "Challenge",
            ],
        )

    def __getitem__(self, index):
        img_path = (
            os.path.join(self.root_dir, str(self.data.iloc[index]["Image ID"])) + ".jpg"
        )
        img = Image.open(img_path).convert("RGB")
        if self.transform is None:
            self.transform = torchvision.transforms.ToTensor()
        img = self.transform(img)
        label = np.array(self.data.iloc[index][2:12])
        label = label / label.sum()
        return img, label

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    img_size = 224
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((img_size, img_size)),
            torchvision.transforms.ToTensor(),
        ]
    )
    train_dataset = AVADataset(
        data_file="/home/wangx/datasets/AVA/AVA_dataset/AVA_filtered_train.txt",
        transform=transform,
    )
    test_dataset = AVADataset(
        data_file="/home/wangx/datasets/AVA/AVA_dataset/AVA_filtered_test.txt",
        transform=transform,
    )
    train_loader = DataLoader(train_dataset, 4, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, 4, shuffle=True, num_workers=4)
