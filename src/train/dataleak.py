# ruff: noqa
import csv
import os
import sys

# W&B 모듈이 import 될 때 sys.stdout을 가로채지 못하도록 원본 C-Stream(fd 1)으로 강제 고정
_orig_stdout = sys.stdout
import wandb
sys.stdout = _orig_stdout

import torch as torch
from torch import nn
from torch import optim
from model import SmallCNN
from data  import get_dataloaders
from data  import get_hyperparameter
from data  import set_seed
from data  import get_dataloaders_experiment_B_proper
from data  import get_dataloaders_experiment_B
from train import train
from torch.utils.data import DataLoader
from typing import Dict, Any
from plot_curves import plot
from torch.optim.lr_scheduler import CosineAnnealingLR

# 깨지거나 닫힌 표준 출력(stdout)을 OS 터미널 장치로 강제 재연결
try:
    sys.stdout.fileno()
except Exception:
    sys.stdout = open('/dev/stdout', 'w')




if __name__ == "__main__" :
    print("<< Direct Call >> \n")
    print("<< Day 5 : Data Leakage>> \n")

    # 실험 이름과 하이퍼 파리미터 받는다.
    hyperparameter = get_hyperparameter("configs/baseline_dataleak.yaml") # depends on "baseline.yaml"
    config = hyperparameter
    
    if hyperparameter['concept'] == 'train' :
        run_name = (
            f"{hyperparameter['concept']}"
            f"_tr{hyperparameter['train_num']}"
            f"_w{hyperparameter['width']}"
            f"_lr{hyperparameter['lr']}"

        )

    elif hyperparameter['concept'] == 'augmentation' :
        run_name = (
            f"{hyperparameter['concept']}"
            f"_aug{hyperparameter['augmentation']}"
            f"_w{hyperparameter['width']}"
            f"_lr{hyperparameter['lr']}"

        )

    elif hyperparameter['concept'] == 'weight_decay' :
        run_name = (
            f"{hyperparameter['concept']}"
            f"{hyperparameter['weight_decay']}"
            f"_w{hyperparameter['width']}"
            f"_lr{hyperparameter['lr']}"

        )
    elif hyperparameter['concept'] == 'scheduler' :
        run_name = (
            f"{hyperparameter['concept']}"
            f"_{hyperparameter['scheduler']}"
            f"_w{hyperparameter['width']}"
            f"_lr{hyperparameter['lr']}"

        )
    elif hyperparameter['concept'] == 'dropout' :
        run_name = (
            f"{hyperparameter['concept']}"
            f"_{hyperparameter['dropout']}"
            f"_w{hyperparameter['width']}"
            f"_lr{hyperparameter['lr']}"

        )
    elif hyperparameter['concept'] == 'batch_size' :
        run_name = (
            f"{hyperparameter['concept']}"
            f"_{hyperparameter['batch_size']}"
            f"_w{hyperparameter['width']}"
            f"_lr{hyperparameter['lr']}"

        )
    elif hyperparameter['concept'] == 'seed' :
        run_name = (
            f"{hyperparameter['concept']}"
            f"_{hyperparameter['seed']}"
            f"_w{hyperparameter['width']}"
            f"_lr{hyperparameter['lr']}"

        )
    else :   
        run_name = (
            f"{hyperparameter['concept']}"
            f"_w{hyperparameter['width']}"
            f"_lr{hyperparameter['lr']}"
            f"_ep{hyperparameter['epochs']}"

        )

    # train 시작 
    wandbOn = True # wandb 사용 여부

    set_seed(hyperparameter['seed'])
    if wandbOn : 
        wandb.init(
            project="cifar10-onboarding", 
            name = run_name, 
            config=hyperparameter,
            settings=wandb.Settings(console="off", _service_wait= 300)
        )
    
    if hyperparameter['concept'] == 'DataLeak_Proper' :
        train_loader, val_loader = get_dataloaders_experiment_B_proper()
        train(train_loader, val_loader,hyperparameter,csv_path = "training_log_dataleak.csv" ,wandbOn=wandbOn)  

    elif hyperparameter['concept'] == 'DataLeak_Wrong' :
        train_loader, val_loader = get_dataloaders_experiment_B()
        train(train_loader, val_loader,hyperparameter,csv_path = "training_log_dataleak.csv" ,wandbOn=wandbOn)  
