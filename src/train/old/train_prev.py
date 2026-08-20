import csv
import torch as torch
from torch import nn
from torch import optim
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms

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
        #predict는 학습된 모델에서 순전파 계산하고 나온 값을 디코딩해서 최종 예측값을 반환하는 함수이다.
        #Pytorch에서는 forward() 함수를 자동 호출한다.
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.head(x)
        return x



# ================== 학습 환경 설정 ==================
# 1). 데이터셋 불러오기 (CIFAR10 :  텐서변환, 정규화)
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
generator = torch.Generator().manual_seed(42)

# 2). train/validation 데이터셋 분리
train_data, val_data = random_split(
	train_full,
	[45000,5000],
	generator = generator
)

# print("train data :", len(train_data))
# print("validation data :", len(val_data))

# 3). 하이퍼파라미터 설정
batch_size = 128
epochs = 30 # 에폭 수 , 총 데이터셋을 몇번 돌릴지.
learning_rate = 0.005 # 학습률, 가중치 업데이트 시 얼마나 크게 반영할지 결정하는 하이퍼파라미터
momentum = 0.9 # 모멘텀, 이전 기울기 정보를 얼마나 반영할지 결정하는 하이퍼파라미터
weight_decay = 0 # 가중치 감쇠(regularization) 정도를 결정하는 하이퍼파라미터

# 4). 데이터 로더 설정
train_loader = DataLoader(
	train_data, 
	batch_size =batch_size, 
	shuffle=True,
	num_workers=4, # CPU 코어
	pin_memory = True, 
	drop_last = False
)

val_loader = DataLoader(
    val_data,
    batch_size = batch_size,
    shuffle = False,
    num_workers = 4,
    pin_memory = True,
    drop_last = False
)

# 5). GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== 모델 학습 ==================

# nn.Module을 상속받아 만든 모델은 정의된 모든 레이어의 가중치를 자동으로 추적합니다. 
# 따라서 직접 가중치를 일일이 건드리지 않아도 옵티마이저(Optimizer)에 
# model.parameters()를 넘겨주기만 하면 옵티마이저가 알아서 모든 가중치에 접근해 업데이트를 수행합니다.
# model.parameters() : 모델의 학습 가능한 모든 파라미터를 반환합니다.

model = SmallCNN(width=32, dropout_p=0.5).to(device) # 모델 생성
criterion = nn.CrossEntropyLoss() # 손실함수 정의 (분류 문제)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay) # 옵티마이저 정의
# lr : learning rate
# momentum  : 이전 기울기 정보를 얼마나 반영할지 결정하는 하이퍼파라미터
# weight_decay  : 가중치 감쇠(regularization) 정도를 결정하는 하이퍼파라미터

# augmentation : 데이터 증강 
# Scheduler : 학습률 스케줄러, 처음에는 크게 움직이며 최적점 근처로 빠르게 가다가, 
# 학습 후반에는 보폭을 줄여 최적점에 미세하고 정밀하게 안착시키는 조절기입니다.


# print(model) # 모델 구조 확인 가능



# 저장할 파일 설정
csv_path = "training_log.csv"
with open(csv_path, mode = 'w', newline="", encoding="utf-8") as f :
    writer = csv.writer(f)
    writer.writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"])
        


for epoch in range(1, epochs + 1) :

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
    print(f"Epoch [{epoch}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")



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
    print(f"Epoch [{epoch}/{epochs}], Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")



    # ---------- 기록 ----------
    # 7) epoch, lr, train_loss, train_acc, val_loss, val_acc 를 CSV 한 줄로 append
    with open(csv_path, mode='a', newline= "", encoding="utf-8") as f :
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            learning_rate,
            round(epoch_train_loss, 4),
            round(epoch_train_acc, 4),
            round(epoch_val_loss, 4),
            round(epoch_val_acc, 4),
        ])


    # 8) 콘솔에도 한 줄 출력
    print(f"Epoch [{epoch}/{epochs}] 저장 완료!")

