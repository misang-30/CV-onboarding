- 학습 파이프라인 구축 기록을 정리

## 1. 로컬 저장소
- .gitignore 파일에 아래 내용을 추가하라.
- .gitignore에 올린 파일은 Git이 변경사항으로 추적하지 않는다.

```gitignore
data/
checkpoints/
*.ckpt
wandb/
```

---

## 2. 데이터 불러오기 (Day1)
> test set은 Day 5에 최초 1회만 열었습니다.
> 그 이전의 모든 판단은 val set만으로 내렸습니다.

- data.py에서 실험에서 사용할 왜곡되거나 정상적인 데이터셋을 받는 함수를 정의


``` python
# data.py

# ruff: noqa
import yaml
import os 
import random
import numpy as np

import torch as torch
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset,Dataset, random_split, DataLoader
from typing import Dict, Any


hyperparameter = {} # 내부 전역 변수

def show_hyperparameter() : 
	print("<< Hyperparameter Status >> ")
	print("Concept, batch_size, epochs, lr, momentum, weight_decay,train_num,val_num, width,seed")
	print(
		hyperparameter ["concept"],
		hyperparameter ["batch_size"],  
		hyperparameter ["epochs"],  
		hyperparameter ["lr"],
		hyperparameter ["momentum"],
		hyperparameter ["weight_decay"],
		hyperparameter ["train_num"],
		hyperparameter ["val_num"],
		hyperparameter ["width"],
		hyperparameter ["seed"],
		hyperparameter ["dropout"],
		hyperparameter ["scheduler"],
		hyperparameter ["augmentation"]
		
	)



def get_hyperparameter (config_path : str = "configs/baseline.yaml"):
	# 여기서 하이퍼 파라미터 딕셔너리로 받아서 바로 사용.
	global hyperparameter
	if not os.path.exists(config_path) :
		raise FileNotFoundError(f"설정한 파일을 찾을 수 없습니다 : {config_path}")

	with open(config_path, "r", encoding="utf-8") as f :
		hyperparameter = yaml.safe_load(f)


	show_hyperparameter()
	return hyperparameter


# Seed 고정 함수
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_transforms(aug_type):
    transform_list = []
    
    # YAML의 augmentation 값에 따라 분기
    if aug_type == "crop":
        transform_list.append(transforms.RandomCrop(32, padding=4))
    elif aug_type == "crop_flip":
        transform_list.append(transforms.RandomCrop(32, padding=4))
        transform_list.append(transforms.RandomHorizontalFlip())
    elif aug_type == "none":
        pass
    else:
        raise ValueError(f"Unknown augmentation option: {aug_type}")


    # 기본 필수 변환
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return transforms.Compose(transform_list)



def get_dataloaders (): 
	# ================== 학습 환경 설정 ==================

	if not hyperparameter : 
		get_hyperparameter()

	# 1). 데이터셋 불러오기 (CIFAR10 :  텐서변환, 정규화)
	norm_transform = build_transforms(hyperparameter["augmentation"])
      
	train_full = datasets.CIFAR10(
		root="./data", 
		train=True,  
		download=True,
		transform=norm_transform
	)
	generator = torch.Generator().manual_seed(hyperparameter ["seed"])

	train_num = hyperparameter ["train_num"]
	val_num = hyperparameter ["val_num"]
	unused_num = len(train_full)-(train_num+val_num)

	# 2). train/validation 데이터셋 분리
	train_data, val_data, _ = random_split(
		train_full,
		[train_num, val_num, unused_num],
		generator = generator
	)

	# print("train data :", len(train_data))
	# print("validation data :", len(val_data))



	# 3). 데이터 로더 설정
	train_loader = DataLoader(
		train_data, 
		batch_size =hyperparameter ["batch_size"], 
		shuffle=True, # 편향을 막기 위해 필요하다.
		num_workers=4, # CPU 코어
		pin_memory = True, 
		drop_last = False
	)

	val_loader = DataLoader(
		val_data,
		batch_size = hyperparameter ["batch_size"],
		shuffle = False, # 검증용도라 셔플 필요가 없다.
		num_workers = 4,
		pin_memory = True,
		drop_last = False
	)

	return train_loader, val_loader

def getTest_dataloaders(config_path : str ) :

    # MEAN과 STD는 CIFAR-10 표준 값을 사용합니다.
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

    if not hyperparameter : 
        get_hyperparameter(config_path) 
                

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    # 2. CIFAR-10 Test 데이터셋 로드 (transform 필수 포함)
    test_set = datasets.CIFAR10(
        root="./data", 
        train=False, 
        download=True,
        transform=test_transform  # <- 전처리 전달 필수!
    )

    # 3. DataLoader 생성
    # Test 세트는 섞을 필요가 없으므로 shuffle=False
    test_loader = DataLoader(
        test_set, 
        batch_size=hyperparameter["batch_size"], 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )

    # 4. test_loader 반환
    return test_loader


# ==========  <day 5 : 7-b : 데이터 누수 실험>  ================


# Custom Dataset Wrapper (PIL Image -> Tensor 및 Transform 적용용)
class TransformDataset(Dataset):

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.dataset[index]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.dataset)


# 1. 원본을 받아서 3배로 먼저 증강하는 함수 
def augment_dataset(base_dataset, multiplier=3):
    # 원본용 전처리
    base_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),
        ]
    )

    # 증강용 전처리 (Random Crop, Flip, Rotation)
    aug_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),
        ]
    )

    datasets_list = []

    # 원본 1배 (500장)
    datasets_list.append(
        TransformDataset(base_dataset, transform=base_transform)
    )

    # 증강 2배 (500장 x 2 = 1000장)
    for _ in range(multiplier - 1):
        datasets_list.append(
            TransformDataset(base_dataset, transform=aug_transform)
        )

    # 총 1500장으로 병합
    return ConcatDataset(datasets_list)


# 2. 잘못된 방식 B의 DataLoader 구축 함수
def get_dataloaders_experiment_B(multiplier=3):
    
    if not hyperparameter:
        get_hyperparameter("configs/baseline_dataleak.yaml")
        
    # CIFAR-10 원본 로드 
    raw_train_full = datasets.CIFAR10(root="./data", train=True, download=True)
    generator = torch.Generator().manual_seed(hyperparameter["seed"])

    # 1) 원본에서 500장 추출
    target_500_dataset, _, _ = random_split(
        raw_train_full,
        [500, len(raw_train_full) - 500, 0],
        generator=generator,
    )

    # 2) [의도적 오류] Split 하기 전에 먼저 3배로 증강 (500장 -> 1500장)
    augmented_1500_dataset = augment_dataset(
        target_500_dataset, multiplier=multiplier
    )

    # 3) [의도적 오류] 데이터 누수가 존재하는 1500장을 1200장 / 300장으로 Split
    total_len = len(augmented_1500_dataset)  # 1500
    val_num = 300
    train_num = total_len - val_num  # 1200

    train_data, val_data = random_split(
        augmented_1500_dataset, [train_num, val_num], generator=generator
    )

    print(
        f"[실험 B - 잘못된 Split]  Train: {len(train_data)}장, Val: {len(val_data)}장"
    )

    # 4) DataLoader 반환
    train_loader = DataLoader(
        train_data,
        batch_size=hyperparameter["batch_size"],
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=hyperparameter["batch_size"],
        shuffle=False,
        num_workers=4,
    )

    return train_loader, val_loader

# 2. 올바른 방식 B의 DataLoader 구축 함수
def get_dataloaders_experiment_B_proper(multiplier=3):

    if not hyperparameter:
        get_hyperparameter("configs/baseline_dataleak.yaml")

    # CIFAR-10 원본 로드 (Transform 미적용 raw 이미지 상태)
    raw_train_full = datasets.CIFAR10(root="./data", train=True, download=True)
    generator = torch.Generator().manual_seed(hyperparameter["seed"])

    # 1) 원본에서 500장 추출
    target_500_dataset, _, _ = random_split(
        raw_train_full,
        [500, len(raw_train_full) - 500, 0],
        generator=generator,
    )

    # 2) 올바른 순서: 먼저 500장을 Train 400장 / Val 100장으로 Split
    train_raw, val_raw = random_split(
        target_500_dataset, [400, 100], generator=generator
    )

    # 3) Train 세트 400장을 k배로 증강 (400장 -> 1200장)
    augmented_train_dataset = augment_dataset(
        train_raw, multiplier=multiplier
    )

    # 4) Val 세트 증강 없이 기본 전처리(ToTensor, Normalize)만 입혀줌
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            ),
        ]
    )
    val_dataset = TransformDataset(val_raw, transform=val_transform)

    print(
        f"[실험 A - 올바른 Split] Train(증강후): {len(augmented_train_dataset)}장 , Val: {len(val_dataset)}장"
    )

    # 5) DataLoader 생성 (수정: augmented_train_dataset 과 val_dataset 사용)
    train_loader = DataLoader(
        augmented_train_dataset,  # 증강된 데이터셋 전달
        batch_size=hyperparameter["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,  # 기본 전처리가 적용된 데이터셋 전달
        batch_size=hyperparameter["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader
	

```

---


## 3. 모델 직접 구현 (Day 2)
- src/train/model.py 에 SmallCNN(nn.Module)을 만든다.



### 1). 모델 스펙

#### Block 1 

- `Conv3x3(pad=1) → BN → ReLU → Conv3x3(pad=1) → BN → ReLU → MaxPool2x2`
- 입출력 : `3×32×32` → `C×16×16`
- 컨볼루션 층을 거치면 C x 32 x 32 가 된다. (C는 필터 개수)  
- 3x32x32 이미지를 필터 3x3x3에 통과시키면 1x32x32 1개가 나온다.
- 필터가 C 개이면 결과물은 1x32x32xC 가 된다. (C개의 필터가 각각 1x32x32 결과물을 내놓는다.)
- 따라서 결과물은 Cx32x32 가 된다.


#### Block 2 
- `Conv3x3(pad=1) → BN → ReLU → Conv3x3(pad=1) → BN → ReLU → MaxPool2x2`
- 입출력 : `C×16×16` → `2C×8×8`

#### Block 3 
- `Conv3x3(pad=1) → BN → ReLU → Conv3x3(pad=1) → BN → ReLU → MaxPool2x2`
- 입출력 : `2C×8×8` → `4C×4×4`

#### Head 
- `AdaptiveAvgPool2d(1) → Flatten → Linear(4C, 10)`
- 입출력 : `4C×4×4` → `10`




### 2). 구현 코드


``` python
class SmallCNN ( nn.Module) :
    def __init__ (self, width: int =32, dropout_p : float = 0.5) :
        super().__init__()
        self.width = width
        self.dropout_p = dropout_p

        # nn.Conv2d(in_channels, out_channels(필터개수), kernel_size, stride=1, padding=0)
        # Sequential : PyTorch에서 여러 레이어를 순서대로 묶어 하나의 모델처럼 만드는 것
        self.block1 = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1), # 특징 추출, 채널 확장
            nn.BatchNorm2d(width), # width는 정규화할 채널 수
            nn.ReLU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1), # 특징 재조합
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(width, width*2, kernel_size=3, padding=1), # 특징 추출, 채널 확장
            nn.BatchNorm2d(width*2), # width*2는 정규화할 채널 수
            nn.ReLU(),
            nn.Conv2d(width*2, width*2, kernel_size=3, padding=1), # 특징 재조합
            nn.BatchNorm2d(width*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(width * 2, width *4, kernel_size=3, padding=1), # 특징 추출, 채널 확장
            nn.BatchNorm2d(width*4), # width*4는 정규화할 채널 수
            nn.ReLU(),
            nn.Conv2d(width*4, width*4, kernel_size=3, padding=1), # 특징 재조합
            nn.BatchNorm2d(width*4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.head = nn.Sequential(  
            nn.AdaptiveAvgPool2d(1), # 4C x 4 x 4 → 4C x 1 x 1 (평균값 뽑아낸다.)
            nn.Flatten(), # 4C x 1 x 1 → 4C (펼치기)
            nn.Linear(width*4, 10) # 4C → 10 , 요약한 4C개의 특징을 10개의 클래스에 대응

        )
 
    def forward(self, x: torch.Tensor) : 
        #forward : 순전파, 입력을 받아서 출력으로 변환하는 과정. predict()와 다르다.
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.head(x)
        return x

```



---

## 4. 학습 루프 구현 (Day 2)

- train.py에서 학습에 필요한 함수 구현 
``` python
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

from torch.utils.data import DataLoader
from typing import Dict, Any
from plot_curves import plot
from torch.optim.lr_scheduler import CosineAnnealingLR

# 깨지거나 닫힌 표준 출력(stdout)을 OS 터미널 장치로 강제 재연결
try:
    sys.stdout.fileno()
except Exception:
    sys.stdout = open('/dev/stdout', 'w')

def train (train_loader :DataLoader,val_loader : DataLoader, hyperparameter : Dict[str, Any], csv_path : str = "training_log.csv", wandbOn: bool =True) :



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
        lr = hyperparameter["lr"], 
        momentum = hyperparameter["momentum"], 
        weight_decay = float(hyperparameter["weight_decay"])
    ) # 옵티마이저 정의

    # YAML의 scheduler 값에 따라 스케줄러 생성
    scheduler_type = hyperparameter['scheduler']

    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=hyperparameter["epochs"])
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
        print(f"Epoch [{epoch}/{hyperparameter['epochs']}], Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}", flush=True)

        

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
        print(f"Epoch [{epoch}/{hyperparameter['epochs']}], Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}", flush=True)

        # ---------- 스케줄러 단계 업데이트 ----------
        if scheduler is not None:
            scheduler.step()

        # 현재 적용된 실제 학습률 가져오기
        current_lr = optimizer.param_groups[0]['lr']


        # ---------- 기록 ----------
        # 7) epoch, lr, train_loss, train_acc, val_loss, val_acc 를 CSV 한 줄로 append
        with open(csv_path, mode='a', newline= "", encoding="utf-8") as f :
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                current_lr,
                round(epoch_train_loss, 4),
                round(epoch_train_acc, 4),
                round(epoch_val_loss, 4),
                round(epoch_val_acc, 4),
            ])
        if wandbOn :
            wandb.log({"epoch": epoch, "train/loss": epoch_train_loss, "train/acc": epoch_train_acc,
            "val/loss": epoch_val_loss, "val/acc": epoch_val_acc, "lr": current_lr})    

        # 8) 콘솔에도 한 줄 출력
        print(f"Epoch [{epoch}/{hyperparameter['epochs']}] 저장 완료!", flush=True)

        # 9) Early Stooping & CheckPoint (Standard = Loss)
        # 함수로 만들 필요가 있다.
        if bestVal_loss > epoch_val_loss : 
            bestVal_loss = epoch_val_loss
            patience = 0 
            best_epoch = epoch
            torch.save(model.state_dict(), 'checkpoints/best.pt')


        else : patience += 1   

        if patience >= 10 :   # best_epoch, best_val_matric 출력, CSV/ 로그에 남기기
            print("[Over Patience, Training Over ! ]", flush=True) 
            print(f"[ Best Epoch : {best_epoch} ]", flush=True)
            print(f"[ Best Loss : {bestVal_loss} ]", flush=True)

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
                
            break
    wandb.finish()
    return model

```
