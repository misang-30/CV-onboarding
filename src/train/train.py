# ruff: noqa
import csv
import os

import wandb
import torch as torch
from torch import nn
from torch import optim
from model import SmallCNN
from data  import get_dataloaders
from data  import get_hyperparameter
from data  import set_seed
from torch.utils.data import DataLoader
from typing import Dict, Any
from plot_curves import plot
from torch.optim.lr_scheduler import CosineAnnealingLR



def train (train_loader :DataLoader,val_loader : DataLoader, hyperparameter : Dict[str, Any], csv_path : str = "training_log.csv", wandbOn: bool =True) :

    torch.use_deterministic_algorithms(True) # 결과 통제
    
    # ================== 모델 학습 ==================

    # nn.Module을 상속받아 만든 모델은 정의된 모든 레이어의 가중치를 자동으로 추적합니다. 
    # 따라서 직접 가중치를 일일이 건드리지 않아도 옵티마이저(Optimizer)에 
    # model.parameters()를 넘겨주기만 하면 옵티마이저가 알아서 모든 가중치에 접근해 업데이트를 수행합니다.
    # model.parameters() : 모델의 학습 가능한 모든 파라미터를 반환합니다.


    print(f"Train Start ! , Width = {hyperparameter['width']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU 세팅

    model = SmallCNN(width=hyperparameter["width"], dropout_p=hyperparameter["dropout"]).to(device) # 모델 생성
    criterion = nn.CrossEntropyLoss() # 손실함수 정의 (분류 문제)
    optimizer = torch.optim.SGD( # 옵티마이저 정의
        model.parameters(), 
        lr= hyperparameter["lr"], 
        momentum= hyperparameter["momentum"], 
        weight_decay=hyperparameter["weight_decay"]
    ) # 옵티마이저 정의

    # YAML의 scheduler 값에 따라 스케줄러 생성
    scheduler_type = hyperparameter['scheduler']

    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    elif scheduler_type == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler option: {scheduler_type}")

    # print(model) # 모델 구조 확인 가능



    # 저장할 파일 설정
    os.makedirs('checkpoints', exist_ok=True)

    with open(csv_path, mode = 'w', newline="", encoding="utf-8") as f :
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"])
            

    # 최고 epoch의 지표
    bestVal_loss = float('inf') # 양의 무한대 사용.
    patience = 0
    best_epoch = 0


    for epoch in range(1, hyperparameter["epochs"] + 1) :

        # ---------- loss, accuracy ----------
        train_loss = 0.0
        train_correct = 0.0

        val_loss = 0.0
        val_correct = 0.0        

        # ---------- train ----------

        # 1). 모델을 학습 모드로
        model.train() # 학습 모드로 전환, dropout, batchnorm 등 학습용 레이어 활성화
        # 2). train_loader 전체 순회
        for x,y in train_loader : # 한번에 batch_size 만큼 데이터 추출, (데이터셋 총개수 / batch_size) 만큼 반복 

            # a).모델을 device로 옮기기
            x = x.to(device) # 데이터를 CPU 메모리에서 GPU 메모리로 옮기기
            y = y.to(device)

            # b).순전파, 예측스칼라로 변환, x.size(0) : 배치 크기
            pred = model(x) # (n, 10), n은 배치 개수

            # c).loss 계산
            loss = criterion(pred, y)

            # d).이전 기울기 초기화
            optimizer.zero_grad()

            # e).역전파
            loss.backward() # 각 파라미터(가중치와 편향)이 어느 방향으로 움직여야 손실이 줄어드는지 메모리에 기록.

            # f).파라미터 갱신
            optimizer.step() # 역전파에서 구한 값을 바탕으로, 실제 가중치 값을 업데이트.
        
            # g).loss와 정확도를 "스칼라로" 누적
            preds = pred.argmax(dim=1) # 예측값 중 가장 큰 값의 인덱스를 가져온다. 
            train_correct += torch.sum(preds == y ).item()
            train_loss += loss.item() * x.size(0) # loss.item() : loss를 스칼라로 변환, x.size(0) : 배치 크기
        # 3). epoch 횟수별 평균 train_loss, train_acc 계산
        epoch_train_loss = train_loss / len(train_loader.dataset) # 평균 loss 계산
        epoch_train_acc = train_correct / len(train_loader.dataset) # 평균 정확도 계산     
        print(f"Epoch [{epoch}/{hyperparameter['epochs']}], Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")



        # ---------- validate ----------

        # 4) 모델을 평가 모드로
        model.eval() # 평가 모드로 전환, dropout, batchnorm 등 학습용 레이어 비활성화



        # 5) 기울기 계산을 끈 상태로 val_loader 전체 순회
        with torch.no_grad() : # 평가 모드에서는 기울기 계산을 끄고, 메모리 사용량을 줄인다.
            for x, y in val_loader : # 한번에 batch_size 만큼 데이터 추출, (데이터셋 총개수 / batch_size) 만큼 반복
                x = x.to(device) # 데이터를 CPU 메모리에서 GPU 메모리로 옮기기
                y = y.to(device)
                pred = model(x) 
                val_step_loss = criterion(pred, y)

                preds = pred.argmax(dim=1) # 예측값 중 가장 큰 값의 인덱스를 가져온다. 
                val_correct += torch.sum(preds == y ).item()
                val_loss += val_step_loss.item() * x.size(0) # loss.item() : loss를 스칼라로 변환, x.size(0) : 배치 크기
                                    
        # 6) val_loss, val_acc 계산
        epoch_val_loss = val_loss / len(val_loader.dataset) # 평균 loss 계산
        epoch_val_acc = val_correct / len(val_loader.dataset) # 평균 정확도 계산
        print(f"Epoch [{epoch}/{hyperparameter['epochs']}], Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")



        # ---------- 기록 ----------
        # 7) epoch, lr, train_loss, train_acc, val_loss, val_acc 를 CSV 한 줄로 append
        with open(csv_path, mode='a', newline= "", encoding="utf-8") as f :
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                hyperparameter["lr"],
                round(epoch_train_loss, 4),
                round(epoch_train_acc, 4),
                round(epoch_val_loss, 4),
                round(epoch_val_acc, 4),
            ])
        if wandbOn :
            wandb.log({"epoch": epoch, "train/loss": epoch_train_loss, "train/acc": epoch_train_acc,
            "val/loss": epoch_val_loss, "val/acc": epoch_val_acc, "lr": hyperparameter["lr"]})    

        # 8) 콘솔에도 한 줄 출력
        print(f"Epoch [{epoch}/{hyperparameter['epochs']}] 저장 완료!")

        # 9) Early Stooping & CheckPoint (Standard = Loss)
        # 함수로 만들 필요가 있다.
        if bestVal_loss > epoch_val_loss : 
            bestVal_loss = epoch_val_loss
            patience = 0 
            best_epoch = epoch
            torch.save(model.state_dict(), 'checkpoints/best.pt')


        else : patience += 1   

        if patience >= 10 :   # best_epoch, best_val_matric 출력, CSV/ 로그에 남기기
            print("[Over Patience, Training Over ! ]") 
            print(f"[ Best Epoch : {best_epoch} ]")
            print(f"[ Best Loss : {bestVal_loss} ]")

            # CSV / 로그 저장
            best_csv_path = "best_" + csv_path
            file_exists = os.path.exists(best_csv_path)

            with open(best_csv_path, 'a', newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["width", "best_epoch", "best_val_loss", "epoch_val_acc"])
                writer.writerow([hyperparameter["width"], best_epoch, round(bestVal_loss, 4), round(epoch_val_acc, 4)])
            if wandbOn :
                wandb.summary["best_val_acc"] = epoch_val_acc
                wandb.summary["best_epoch"] = best_epoch
                wandb.finish()
            break


if __name__ == "__main__" :
    print("<< Direct Call >> \n")
    print("<< Day 2 Lab : Train Loop >>\n")

    
    # 실험 이름과 하이퍼 파리미터 받는다.
    hyperparameter = get_hyperparameter() # depends on "baseline.yaml"
    config = hyperparameter

    run_name = (
        f"base_w{hyperparameter['width']}"
        f"_lr{hyperparameter['lr']}"
        f"_bat{hyperparameter['batch_size']}"
        f"_ep{hyperparameter['epochs']}"
    )

    # train 시작 
    set_seed(hyperparameter['seed'])

    wandb.init(project="cifar10-onboarding", name = run_name, config=hyperparameter)
    train_loader, val_loader = get_dataloaders()

    train(train_loader, val_loader,hyperparameter ) 
    fileName = "training_log.csv"
    plot(fileName) 