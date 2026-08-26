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

### 1). CIFAR10 다운로드


```bash
conda activate pytorch_env
pip install matplotlib pyyaml
pip install torch torchvision
```
### 2). 이미지 격자 

![](../img/cifar10_samples.png)

### 3).  텐서 변환, 정규화, split

- CIFAR-10 이미지는 transform 없이 가져오면 class 'PIL.Image.Image' 이런 식이다. 즉, 이미지 파일을 PIL이라는 이미지 객체로 가지고 있는 상태다. 
> 데이터를 텐서화한다  = 이미지 같은 데이터를 Pytorch가 계산할 수 있는 숫자 덩어리로 바꾼다는 것이다. 


> [결과]
> 유사한 값을 갖는다. 
> CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
> CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
> 계산한 Mean : 0.4917, 0.4823, 0.4467
> 계산한 Std : 0.2471, 0.2435, 0.2616



### 4). DataLoader

> [결과]
 x.shape : torch.Size([128, 3, 32, 32])
 x.dtype : torch.float32
 x.min   : -1.0000
 x.max   : 1.0000
 y[:10]  : tensor([4, 0, 7, 9, 7, 8, 5, 4, 8, 0])
 
### 5). 전체 코드 4


``` python
import os

import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

train_full = datasets.CIFAR10(
	root="./data", 
	train=True,  
	download=True
)
test_set = datasets.CIFAR10(
	root="./data", 
	train=False, 
	download=True
)

print(len(train_full), len(test_set))   # 50000 10000




## 1. CIFAR-10 클래스 이름

classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]
## 2. CIFAR-10 데이터 불러오기
train_full = datasets.CIFAR10(
	root="./data", 
	train=True,  
	download=True,
	
)




## 3. 클래스별 이미지 5장씩 가져오기
# 각 클래스에서 몇 장을 찾았는지 기록
class_counts = [0] * 10

# 클래스 별로 이미지를 저장할 공간 확보
selected_images = [[] for _ in range(10)]


for image, label in train_full:
	# 5개 못 넣었으면
	if class_counts[label] < 5:
		selected_images[label].append(image)
		class_counts[label] +=1
	
	# 5개 다 넣었으면 종료
	if all(count == 5 for count in class_counts ) : 
		break

## 4. 5x10 격자 만들기

fig, axes = plt.subplots(
	5,
	10,
	figsize = (32,32)
)


## 5. 이미지 출력
for class_idx in range(10):
	for image_idx in range (5):
		image = selected_images[class_idx][image_idx]
		
		ax = axes[image_idx, class_idx]
		
		ax.imshow(image)
		
		ax.set_title(
			classes[class_idx],
			fontsize = 8
		)
		ax.axis("off")
plt.tight_layout()

## 6. 이미지 저장
# 디렉토리 없으면 생성
os.makedirs("images", exist_ok=True)

# 저장
plt.savefig(
	"images/cifar10_samples.png",
	dpi = 150
)

plt.show()


## 7. 전체 클래스 개수 확인


total_counts = [0] * 10

for _, label in train_full:
	total_counts[label] +=1

print("\n클래스별 데이터 개수")

for i in range(10) : 
	print(
		f"{classes[i]:12s}"
		f"{total_counts[i]}"
		
	)

## 8.train_data의 mean,std 수동 계산
norm_transform = transforms.Compose([
	transforms.ToTensor()

])

train_full = datasets.CIFAR10(
	root="./data", 
	train=True,  
	download=True,
	transform=norm_transform
)
generator = torch.Generator().manual_seed(42)
train_data, val_data = random_split(
	train_full,
	[45000,5000],
	generator = generator
)




print("==Train_data Mean, Std==")
img = torch.stack([img for img, _ in train_data])

train_mean = img.mean(dim=[0,2,3])
train_std = img.std(dim=[0,2,3])

print(f"Calculated Mean : {train_mean[0]:.4f}, {train_mean[1]:.4f}, {train_mean[2]:.4f}") 
print(f"Calculated Std : {train_std[0]:.4f}, {train_std[1]:.4f}, {train_std[2]:.4f}")


## 9.텐서변환 및 정규화
norm_transform = transforms.Compose([
	transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_full = datasets.CIFAR10(
	root="./data", 
	train=True,  
	download=True,
	transform=norm_transform
)
## 10.데이터셋 나누기
generator = torch.Generator().manual_seed(42)

# train_data : 45,000 / 파라미터 학습용
# val_data : 5,000 / 하이퍼 파라미티 선택용
# test_data : 10,000 / 최종 성능 체크 ( 봉인, 나중에 씀)
train_data, val_data = random_split(
	train_full,
	[45000,5000],
	generator = generator
)

print("train data :", len(train_data))
print("validation data :", len(val_data))



## 11.배치 1개 추출
train_loader = DataLoader(
	train_data, 
	batch_size =128, 
	shuffle=True,
	num_workers=4, # CPU 코어
	pin_memory = True, 
	drop_last = False
)


images, labels = next(iter(train_loader)) 
# iter로 이터레이터 생성, next로 앞서 만든 이터레이터에서 하나 뽑음.
print(f"x.shape : {images.shape}")
print(f"x.dtype : {images.dtype}")
print(f"x.min   : {images.min():.4f}")
print(f"x.max   : {images.max():.4f}")
print(f"y[:10]  : {labels[:10]}")





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
# ruff: noqa
import torch as torch
from torch import nn

# ================== 모델 정의 ==================
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

if __name__ == "__main__" :
    print("Direct Call")


```


### 3). 체크

``` python



```



---
## 4. 학습 루프 구현 (Day 2)

- train.py
``` python

# ruff: noqa
import csv


import torch as torch
from torch import nn
from torch import optim
from model import SmallCNN
from data  import get_dataloaders
from data  import get_hyperparameter
from data  import set_hyperparameter


def train () :
    # ================== 학습 데이터셋 / 하이퍼파라미터==================

    train_loader, val_loader = get_dataloaders()
    hyperparameter = get_hyperparameter() 


    # ================== 모델 학습 ==================

    # nn.Module을 상속받아 만든 모델은 정의된 모든 레이어의 가중치를 자동으로 추적합니다. 
    # 따라서 직접 가중치를 일일이 건드리지 않아도 옵티마이저(Optimizer)에 
    # model.parameters()를 넘겨주기만 하면 옵티마이저가 알아서 모든 가중치에 접근해 업데이트를 수행합니다.
    # model.parameters() : 모델의 학습 가능한 모든 파라미터를 반환합니다.



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU 세팅

    model = SmallCNN(width=32, dropout_p=0.5).to(device) # 모델 생성
    criterion = nn.CrossEntropyLoss() # 손실함수 정의 (분류 문제)
    optimizer = torch.optim.SGD( # 옵티마이저 정의
        model.parameters(), 
        lr= hyperparameter["lr"], 
        momentum= hyperparameter["momentum"], 
        weight_decay=hyperparameter["weight_decay"]
    ) # 옵티마이저 정의

    # print(model) # 모델 구조 확인 가능



    # 저장할 파일 설정
    csv_path = "training_log.csv"
    with open(csv_path, mode = 'w', newline="", encoding="utf-8") as f :
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"])
            


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

                preds = pred.argmax(dim=1) # 예측값 중 가장 큰 값의 인덱스를 가져온다. 
                val_correct += torch.sum(preds == y ).item()
                val_loss += loss.item() * x.size(0) # loss.item() : loss를 스칼라로 변환, x.size(0) : 배치 크기
                                    
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


        # 8) 콘솔에도 한 줄 출력
        print(f"Epoch [{epoch}/{hyperparameter['epochs']}] 저장 완료!")

if __name__ == "__main__" :
    print("Direct Call")
    train() 
    
```
- plot_curves.py
```python


import matplotlib.pyplot as plt
import pandas as pd



def plot() : 
    # 1). Train 데이터 가져오기
    df =pd.read_csv("training_log.csv")
    epochs = df["epoch"]
    train_loss = df["train_loss"]
    val_loss = df["val_loss"]
    train_acc = df["train_acc"]
    val_acc = df["val_acc"]
    

    # 2). 그래프 생성
   
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 왼쪽: Loss 그래프 (2개 변수)
    ax1.plot(epochs, train_loss, label="Train Loss", color="blue", marker="o")
    ax1.plot(epochs, val_loss, label="Val Loss", color="orange", marker="o")
    ax1.set_title("Loss Trend")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # 오른쪽: Accuracy 그래프 (2개 변수)
    ax2.plot(epochs, train_acc, label="Train Acc", color="green", marker="s")
    ax2.plot(epochs, val_acc, label="Val Acc", color="red", marker="s")
    ax2.set_title("Accuracy Trend")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("../../img/train_plt.png")
    plt.show()
        

if __name__ == "__main__" :
    plot() 

```

----
## 5. 과적합 제조 (Day 3)





---
## 6. 실험관리도구 (Day 3)





---
## 7. 통제 실험(Day 4)



---
## 8. 함정 체험 (Day 5)


