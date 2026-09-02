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
from data  import getTest_dataloaders
from torch.utils.data import DataLoader
from typing import Dict, Any
from plot_curves import plot
from torch.optim.lr_scheduler import CosineAnnealingLR

# 깨지거나 닫힌 표준 출력(stdout)을 OS 터미널 장치로 강제 재연결
try:
    sys.stdout.fileno()
except Exception:
    sys.stdout = open('/dev/stdout', 'w')


def test_model(model, test_loader): # 수정 요망 wandb 기능 필요.
    print(f"Test Start ! ")

    # 1. 평가모드 전환
    device = torch.dev하ice("cuda" if torch.cuda.is_available() else "cpu") # GPU 세팅
    criterion = nn.CrossEntropyLoss() # 손실함수 정의 (분류 문제)


    model.eval() # 여기서는 epoch이 필요 없다.
    test_loss = 0.0
    correct = 0
    total = 0

    # 2. 기울기 계산 비활성화 (평가 시에는 필요 없음)
    with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                
                # 3. 순전파 (Forward)
                outputs = model(x)
                
                # 4. Loss 계산
                loss = criterion(outputs, y)
                
                # 5. Loss 및 Accuracy 누적
                # batch_size를 곱해서 전체 손실 합을 누적 (마지막 배치가 128개가 아닐 수 있으므로)
                total_loss += loss.item() * x.size(0)
                
                # 예측값 구하기 (가장 높은 확률/점수를 가진 클래스 인덱스)
                _, preds = torch.max(outputs, 1)
                correct += (preds == y).sum().item()
                total += y.size(0)

    # 6. 전체 데이터셋에 대한 평균 Loss와 Accuracy 계산
    test_loss = total_loss / total
    test_acc = (correct / total) * 100.0  # 백분율(%)

    return test_loss, test_acc



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
        model = train(train_loader, val_loader,hyperparameter,csv_path = "training_log_dataleak.csv" ,wandbOn=wandbOn)  
        test_data = getTest_dataloaders()
        test_model(model, test_data)

        
    elif hyperparameter['concept'] == 'DataLeak_Wrong' :
        train_loader, val_loader = get_dataloaders_experiment_B()
        model = train(train_loader, val_loader,hyperparameter,csv_path = "training_log_dataleak.csv" ,wandbOn=wandbOn)  
        test_data = getTest_dataloaders()
        test_model(model, test_data)

        