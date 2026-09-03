from labC import smallcnn
import torch
import torch.nn as nn
from torchvision import models


def restnet18() :

    model = models.resnet18(weights=None)

    model.conv1 = nn.Conv2d(
        3, 64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )

    model.maxpool = nn.Identity()

    model.fc = nn.Linear(model.fc.in_features, 10)
    # CIFAR-10 32×32
    #     ↓
    # 3×3 Conv
    #     ↓
    # ResNet18
    #     ↓
    # FC
    #     ↓
    # 10 classes


def resnetPre() :
    # 



if __name__ == "__main__":
    print("<< Direct Call >> \n")
    print("<< Day 5 : LabC >> \n")

    print("Handmade SmallCNN ")
    smallcnn()

    print("Resnet - Scratch Version")
    #

    print("Resnet - Pretrained")
    #
