import csv
import os
import sys

# W&B 모듈이 import 될 때 sys.stdout을 가로채지 못하도록 원본 C-Stream(fd 1)으로 강제 고정
_orig_stdout = sys.stdout
import wandb
sys.stdout = _orig_stdout

from labC import smallcnn
import torch
import torch.nn as nn
from torchvision.models import models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
from data import get_hyperparameter
from data import get_name
from data import set_seed
from data import get_dataloaders
from train import train
from typing import Dict, Any
from plot_curves import plot


# 깨지거나 닫힌 표준 출력(stdout)을 OS 터미널 장치로 강제 재연결
try:
    sys.stdout.fileno()
except Exception:
    sys.stdout = open('/dev/stdout', 'w')


class ResNet18Scratch:
    def __init__(self, num_classes=10):
        # 1. 가중치 없이 ResNet18 생성 (Scratch 학습)
        self.model = models.resnet18(weights=None)
        
        # 2. 마지막 fc 레이어 교체 (CIFAR-10 클래스 수인 10개로)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def get_transform(self):
        # Scratch는 원본 크기(32x32) 유지 및 CIFAR 통계 사용
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], 
                std=[0.2023, 0.1994, 0.2010]
            )
        ])
    
    def get_model(self):
        return self.model

class ResNet18FineTune:
    def __init__(self, num_classes=10):
        # 1. ImageNet 사전학습 가중치 로드
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.model = models.resnet18(weights=weights)
        
        # 2. 마지막 fc 레이어 교체
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def get_transform(self):
        # Pretrained 모델은 224x224 리사이즈와 ImageNet 통계 필수
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def get_model(self):
        return self.model

def train_scratch(hyperparameter : Dict[str, Any], train_loader : DataLoader ) : 

    config = hyperparameter
    
    run_name = get_name(hyperparameter)


    wandbOn = True # wandb 사용 여부

    set_seed(hyperparameter['seed'])
    if wandbOn : 
        wandb.init(
            project="cifar10-onboarding", 
            name = run_name, 
            config=hyperparameter,
            settings=wandb.Settings(console="off", _service_wait= 300)
        )
    train_loader, val_loader = get_dataloaders()

    # train 시작 
    # ====== train 알고리즘 ==========

    # train(train_loader, val_loader,hyperparameter, wandbOn=wandbOn)  



    fileName = "SmallCNN_log.csv"
    plot(fileName) 




def train_finetune (num : int ) :
    print("not implemented yet")


if __name__ == "__main__":
    print("<< Direct Call >> ")
    print("<< Day 5 : LabC >> \n")

    # 1.데이터 불러오기
    # 1). 500장 이미지 불러오기
    
    # CIFAR-10 전체 중에서 딱 500장을 뽑을 인덱스를 만듭니다.
    indices = torch.randperm(50000)[:500]
    
    # Scratch 학습용 전처리 (32x32 + CIFAR 통계)
    transform_scratch = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    dataset_scratch = datasets.CIFAR10(root='./data', train=True, transform=transform_scratch)
    subset_scratch = Subset(dataset_scratch, indices) 

    # Pretrained 파인튜닝용 전처리 (224x224 리사이즈 + ImageNet 통계)
    transform_pretrained = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset_pretrained = datasets.CIFAR10(root='./data', train=True, transform=transform_pretrained)
    subset_pretrained = Subset(dataset_pretrained, indices)

    # 2.하이퍼 퍼라미터 불러오기
    hyperparameters = get_hyperparameter()




    # 3.SmallCNN 학습
    print("[Train] SmallCNN ")
    #train() 함수 사용

    # 3.Resnet - Scratch Version 학습 
    print("[Train] Resnet - Scratch Version")
    # 하이퍼 파라미터 값은 baseline.yaml 의 값을 사용한다. 

    # 4.Resnet - Pretrained 학습 # 
    print("[Train] Resnet - Pretrained")
    
