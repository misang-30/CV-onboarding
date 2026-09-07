# ruff: noqa
import csv
import os
import sys

# W&B 모듈이 import 될 때 sys.stdout을 가로채지 못하도록 원본 C-Stream(fd 1)으로 강제 고정
_orig_stdout = sys.stdout
import wandb
sys.stdout = _orig_stdout


import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
from data import get_hyperparameter
from train import train
from typing import Dict, Any
from plot_curves import plot
from train import train_type


# 깨지거나 닫힌 표준 출력(stdout)을 OS 터미널 장치로 강제 재연결
try:
    sys.stdout.fileno()
except Exception:
    sys.stdout = open('/dev/stdout', 'w')




def train_finetune (num : int ) :
    print("not implemented yet")


if __name__ == "__main__":
    print("<< Direct Call >> ")
    print("<< Day 5 : LabC >> \n")


    # 하이퍼 퍼라미터 불러오기
    hyperparameters = get_hyperparameter("configs/baseline_labD.yaml")

    # 1.데이터 불러오기
    # 1). 500장 이미지 불러오기
    
    # CIFAR-10 전체 중에서 딱 600장을 뽑을 인덱스를 만듭니다.
    indicesTotal = torch.randperm(50000)[:600]
    train_indices = indicesTotal[:500]  # 500장만 선택
    val_indices = indicesTotal[500:600]  # 나머지 100장은 검증용으로 선택

    # Scratch 학습용 전처리 (32x32 + CIFAR 통계)
    transform_scratch = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    # Pretrained 파인튜닝용 전처리 (224x224 리사이즈 + ImageNet 통계)
    transform_pretrained = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # [Scartch]데이터셋 객체를 각각 생성하여 Transform 분리
    dataset_train = datasets.CIFAR10(root='./data', train=True, transform=transform_scratch, download=True)
    dataset_val   = datasets.CIFAR10(root='./data', train=True, transform=transform_scratch, download=True)

    # [Scratch]Subset 생성
    train_subset = Subset(dataset_train, train_indices)
    val_subset   = Subset(dataset_val, val_indices)

    # [Scratch] DataLoader 생성 (Batch 단위 묶음 및 배치 텐서 자동 변환 수행)
    train_loader = DataLoader(train_subset, batch_size=hyperparameters["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_subset, batch_size=hyperparameters["batch_size"], shuffle=False)

    # [Pretrained]데이터셋 객체를 각각 생성하여 Transform 분리
    dataset_train_pretrained = datasets.CIFAR10(root='./data', train=True, transform=transform_pretrained, download=True)
    dataset_val_pretrained   = datasets.CIFAR10(root='./data', train=True, transform=transform_pretrained, download=True)

    # [Pretrained]Subset 생성
    train_pretrained = Subset(dataset_train_pretrained, train_indices)
    val_pretrained   = Subset(dataset_val_pretrained, val_indices)


    # [Pretrained] DataLoader 생성 (Batch 단위 묶음 및 배치 텐서 자동 변환 수행)
    train_loader_pretrained = DataLoader(train_pretrained, batch_size=int(hyperparameters["batch_size"]/8), shuffle=True) # 224 x 224 이미지를 쓰므로 Vram 사용량 줄이기위해 batch size를 1/8로 줄임
    val_loader_pretrained   = DataLoader(val_pretrained, batch_size=int(hyperparameters["batch_size"]/8), shuffle=False) 

    



    # 2.SmallCNN 학습
    print("[Train] SmallCNN ")
    train_type(0,train_loader, val_loader, hyperparameters, "labD_SmallCNN.csv", wandbOn=True)

    # 3.Resnet - Scratch Version 학습 
    print("[Train] Resnet - Scratch Version")
    train_type(1,train_loader, val_loader, hyperparameters, "labD_Resnet.csv", wandbOn=True)
    
    
    # 4.Resnet - Pretrained 학습 
    print("[Train] Resnet - Pretrained")
    train_type(2,train_loader, val_loader, hyperparameters, "labD_Resnet.csv", wandbOn=True)
