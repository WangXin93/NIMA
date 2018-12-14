import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models, transforms

from dataset import AVADataset
from mobile_net_v2 import mobile_net_v2
from utils import AverageMeter, EMDLoss, ModelSaver

device = torch.device("cuda")
img_size = 224
batch_size = 64
epochs = 50
log_step = 50
learning_rate = 3e-6
comment = ""

##############
# Dataloader #
##############
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)

train_dataset = AVADataset(
    transform=torchvision.transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]
    )
)
test_dataset = AVADataset(
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
model_path = "model_main.pth"
model.load_state_dict(torch.load(model_path))
print("Load pretrained model from {}".format(model_path))


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
criterion = EMDLoss()


def train(model, train_loader, scheduler, optimizer):
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
            logging.info("#{} emb_loss: {:.4f}".format(batch_num, emd_loss.item()))
    logging.info("Train emd_loss: {:.4f}".format(emd_losses.avg))


def eval(model, test_loader):
    model.eval()
    emd_losses = AverageMeter()
    scores_hist = []
    labels_hist = []
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
    logging.info("Test emd_loss: {:.4f}".format(emd_losses.avg))

    scores_hist = np.concatenate(scores_hist)
    labels_hist = np.concatenate(labels_hist)
    scores_mean = np.dot(scores_hist, np.arange(1, 11))
    labels_mean = np.dot(labels_hist, np.arange(1, 11))
    SRCC_mean, _ = spearmanr(scores_mean, labels_mean)
    LCC_mean, _ = pearsonr(scores_mean, labels_mean)
    logging.info("Test SRCC_mean: {:.4f} LCC_mean: {:.4f}".format(SRCC_mean, LCC_mean))

    return emd_losses.avg, SRCC_mean, LCC_mean


def examples(model, test_loader):
    model.eval()

    imgs, labels = next(iter(test_loader))
    imgs, labels = imgs.to(device).float(), labels.to(device).float()

    with torch.no_grad():
        scores = model(imgs)

    imgs = imgs.cpu().data.numpy()
    imgs = imgs.transpose(0, 2, 3, 1)
    scores = scores.cpu().data.numpy()
    labels = labels.cpu().data.numpy()
    scores = np.dot(scores, np.arange(1, 11).reshape(10, 1))
    labels = np.dot(labels, np.arange(1, 11).reshape(10, 1))

    for idx, (img, score, label) in enumerate(zip(imgs, scores, labels)):
        if idx >= 10: break
        print("#{}\r".format(idx), end="", flush=True)
        img = (img * IMAGE_NET_STD + IMAGE_NET_MEAN) * 255
        plt.imshow(np.uint8(img))
        plt.axis("off")
        plt.title("Predict: {:.4f} Label: {:.4f}".format(score[0], label[0]))
        plt.savefig("{}.png".format(idx))


if __name__ == "__main__":
    # Valid Phase
    emd_loss, SRCC_mean = eval(model, test_loader)
    # Show 10 examples
    print("Show 10 examples...")
    examples(model, test_loader)
