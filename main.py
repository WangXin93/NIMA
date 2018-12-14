import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models, transforms

from dataset import AVADataset
from mobile_net_v2 import mobile_net_v2
from utils import AverageMeter, EMDLoss, ModelSaver

device = torch.device("cuda")
batch_size = 128
epochs = 30
log_step = 50
comment = sys.argv[1]

##############
# Dataloader #
##############
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)

train_dataset = AVADataset(
    root_dir="/export/home/wangx/datasets/AVA/AVA_dataset/images/",
    data_file="/export/home/wangx/datasets/AVA/AVA_dataset/AVA_filtered_train.txt",
    transform=torchvision.transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]
    ),
)
test_dataset = AVADataset(
    root_dir="/export/home/wangx/datasets/AVA/AVA_dataset/images/",
    data_file="/export/home/wangx/datasets/AVA/AVA_dataset/AVA_filtered_test.txt",
    transform=torchvision.transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]
    ),
)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, 128, shuffle=False, num_workers=4)

##########
# Logger #
##########
log_format = "%(asctime)s [%(levelname)-5.5s] %(message)s"
filename = "log_{}{}.log".format(__file__.split(".")[0], comment)
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[logging.FileHandler(filename), logging.StreamHandler()],
)

#########
# model #
#########
class NIMA(nn.Module):
    def __init__(self, pretrained_base_model=True):
        super(NIMA, self).__init__()
        base_model = mobile_net_v2(pretrained=pretrained_base_model)
        base_model = nn.Sequential(*list(base_model.children())[:-1])

        self.base_model = base_model

        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.75),
            nn.Linear(1280, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


model = NIMA()
model = model.to(device)

#########
# Train #
#########
parameters = [
    {"params": model.base_model.parameters()},
    {"params": model.head.parameters(), "lr": 3e-5},
]
optimizer = torch.optim.Adam(parameters, lr=3e-6)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
saver = ModelSaver()


def train(model, train_loader, scheduler, optimizer):
    criterion = EMDLoss()  # r=2 for train
    logging.info("Train Phase, Epoch: {}".format(epoch))
    scheduler.step()
    emd_losses = AverageMeter()

    model.train()
    for batch_num, batch in enumerate(train_loader, 1):
        imgs, labels = batch
        imgs, labels = imgs.to(device).float(), labels.to(device).float()

        scores = model(imgs)
        emd_loss = criterion(scores, labels)

        emd_losses.update(emd_loss.item(), imgs.shape[0])

        # Backward prop
        model.zero_grad()
        emd_loss.backward()
        optimizer.step()

        if batch_num % log_step == 0:
            logging.info(
                "Epoch:[{}/{}] #{} emd_loss: {:.4f}".format(
                    epoch, epochs, batch_num, emd_loss.item()
                )
            )
    logging.info("Train emd_loss: {:.4f}".format(emd_losses.avg))


def eval(model, test_loader):
    criterion = EMDLoss(r=2)  # r=1 for evaluate
    emd_losses = AverageMeter()
    scores_hist = []
    labels_hist = []

    model.eval()
    for batch_num, batch in enumerate(test_loader, 1):
        print("{}\r".format(batch_num), end="", flush=True)
        imgs, labels = batch
        imgs, labels = imgs.to(device).float(), labels.to(device).float()

        with torch.no_grad():
            scores = model(imgs)
            emd_loss = criterion(scores, labels)
            emd_losses.update(emd_loss.item(), imgs.shape[0])

        scores_hist.append(scores.cpu().data.numpy())
        labels_hist.append(labels.cpu().data.numpy())
    print()
    logging.info("Test emd_loss@r=1: {:.4f}".format(emd_losses.avg))

    scores_hist = np.concatenate(scores_hist)
    labels_hist = np.concatenate(labels_hist)
    scores_mean = np.dot(scores_hist, np.arange(1, 11))
    labels_mean = np.dot(labels_hist, np.arange(1, 11))
    SRCC_mean, _ = spearmanr(scores_mean, labels_mean)
    logging.info("Test SRCC_mean: {:.4f}".format(SRCC_mean))

    return emd_losses.avg, SRCC_mean


if __name__ == "__main__":
    for epoch in range(1, epochs + 1):
        # Train Phase
        train(model, train_loader, scheduler, optimizer)
        # Valid Phase
        emd_loss, SRCC_mean = eval(model, test_loader)
        # Save model
        fname = "model_{}{}.pth".format(__file__.split(".")[0], comment)
        saver.save(SRCC_mean, model.state_dict(), fname)
