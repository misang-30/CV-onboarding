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

def test_model(test_loader, best_ckpt_path="checkpoints/best.pt"):
    print("\n==========================================")
    print("           Test Evaluation Start          ")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(best_ckpt_path):
        raise FileNotFoundError(f"[!] '{best_ckpt_path}' 체크포인트가 없습니다.")

    # 1. 가중치 파일 로드
    state_dict = torch.load(best_ckpt_path, map_location=device)

    # 2. 첫 번째 Conv 레이어에서 width 자동 추출
    first_key = list(state_dict.keys())[0]
    inferred_width = state_dict[first_key].shape[0]
    print(f"[*] Loaded checkpoint width: {inferred_width}")

    # 3. 모델 생성 및 가중치 주입
    model = SmallCNN(width=inferred_width).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    # 4. Test 평가 진행
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item() * x.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    test_loss = total_loss / total
    test_acc = (correct / total) * 100.0

    print(f"[Result] Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%\n")
    return test_loss, test_acc

def smallcnn() :
        # 하이퍼파라미터 불러오기
    hyperparameter = get_hyperparameter()

    # 시드 고정
    set_seed(hyperparameter["seed"])

    test_data = getTest_dataloaders()
    _, val_data = get_dataloaders()
    # 실제 best.pt가 존재하는 파일 경로 지정
    val_loss, val_acc = test_model(val_data, best_ckpt_path="checkpoints/best.pt")
    test_loss, test_acc = test_model(test_data, best_ckpt_path="checkpoints/best.pt")


if __name__ == "__main__":
    print("<< Direct Call >> \n")
    print("<< Day 5 : LabC >> \n")

    # 하이퍼파라미터 불러오기
    hyperparameter = get_hyperparameter()

    # 시드 고정
    set_seed(hyperparameter["seed"])

    test_data = getTest_dataloaders()
    _, val_data = get_dataloaders()
    # 실제 best.pt가 존재하는 파일 경로 지정
    val_loss, val_acc = test_model(val_data, best_ckpt_path="checkpoints/best.pt")
    test_loss, test_acc = test_model(test_data, best_ckpt_path="checkpoints/best.pt")