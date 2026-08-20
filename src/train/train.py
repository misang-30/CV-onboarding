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
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.head(x)
        return x

    def predict(self, x : torch.Tensor) : 
        # not yet implemented
        print("not yet implemented")



# ================== 학습 환경 설정 ==================
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

train_data, val_data = random_split(
	train_full,
	[45000,5000],
	generator = generator
)

# print("train data :", len(train_data))
# print("validation data :", len(val_data))

batch_size = 128
epochs = 30 # 에폭 수

## 배치 1개 추출
train_loader = DataLoader(
	train_data, 
	batch_size =batch_size, 
	shuffle=True,
	num_workers=4, # CPU 코어
	pin_memory = True, 
	drop_last = False
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== 모델 학습 ==================

# nn.Module을 상속받아 만든 모델은 정의된 모든 레이어의 가중치를 자동으로 추적합니다. 
# 따라서 직접 가중치를 일일이 건드리지 않아도 옵티마이저(Optimizer)에 
# model.parameters()를 넘겨주기만 하면 옵티마이저가 알아서 모든 가중치에 접근해 업데이트를 수행합니다.
# model.parameters() : 모델의 학습 가능한 모든 파라미터를 반환합니다.

model = SmallCNN(width=32, dropout_p=0.5).to(device) # 모델 생성
criterion = nn.CrossEntropyLoss() # 손실함수 정의 (분류 문제)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0) # 옵티마이저 정의
# lr = learning rate
# momentum = 0.9 : 이전 기울기 정보를 얼마나 반영할지 결정하는 하이퍼파라미터
# weight_decay = 0 : 가중치 감쇠(regularization) 정도를 결정하는 하이퍼파라미터
# 
# print(model) # 모델 구조 확인 가능

# augmentation : 데이터 증강 
# Scheduler : 학습률 스케줄러, 처음에는 크게 움직이며 최적점 근처로 빠르게 가다가, 
# 학습 후반에는 보폭을 줄여 최적점에 미세하고 정밀하게 안착시키는 조절기입니다.


for epoch in range(1, 2) :
        
    # ---------- train ----------
    # 1) 모델을 학습 모드로
    # 2) for x, y in train_loader:
    #      a) 데이터를 device로 옮기기
    #      b) 순전파 → 예측
    #      c) loss 계산
    #      d) 이전 기울기 초기화
    #      e) 역전파
    #      f) 파라미터 갱신
    #      g) loss와 정확도를 "스칼라로" 누적
    # 3) epoch 평균 train_loss, train_acc 계산

    # ---------- train ----------


    for x,y in train_loader : # 한번에 batch_size 만큼 데이터 추출, (데이터셋 총개수 / batch_size) 만큼 반복 
        # 0). 모델을 학습 모드로 전환
        model.train() # 학습 모드로 전환, dropout, batchnorm 등 학습용 레이어 활성화


        # 1).모델을 device로 옮기기
        x = x.to(device) # 데이터를 CPU 메모리에서 GPU 메모리로 옮기기
        y = y.to(device)

        # 2).순전파, 예측
        pred = model(x) # (n, 10), n은 배치 개수

        # 3).loss 계산
        loss = criterion(pred, y)

        # 4).이전 기울기 초기화
        optimizer.zero_grad()

        # 5).역전파
        loss.backward()

        # 6).파라미터 갱신
        optimizer.step()    
    


        # # 7).loss와 정확도를 "스칼라로" 누적
        # train_acc = (pred.argmax(dim=1) == y).float().mean() # 정확도 계산
        # train_loss += loss.item() * x.size(0) # loss.item() : loss를 스칼라로 변환, x.size(0) : 배치 크기





    # ---------- validate ----------
    # 4) 모델을 평가 모드로
    # 5) 기울기 계산을 끈 상태로 val_loader 전체 순회
    # 6) val_loss, val_acc 계산

    # ---------- 기록 ----------
    # 7) epoch, lr, train_loss, train_acc, val_loss, val_acc 를 CSV 한 줄로 append
    # 8) 콘솔에도 한 줄 출력


