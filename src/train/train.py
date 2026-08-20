from model import SmallCNN
import torch as torch
from torch import nn
from torch import optim
from data import train_loader, val_loader



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== 모델 학습 ==================

# nn.Module을 상속받아 만든 모델은 정의된 모든 레이어의 가중치를 자동으로 추적합니다. 
# 따라서 직접 가중치를 일일이 건드리지 않아도 옵티마이저(Optimizer)에 
# model.parameters()를 넘겨주기만 하면 옵티마이저가 알아서 모든 가중치에 접근해 업데이트를 수행합니다.
# model.parameters() : 모델의 학습 가능한 모든 파라미터를 반환합니다.

model = SmallCNN(width=32, dropout_p=0.5) # 모델 생성
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

batch_size = 128
epochs = 30


for epoch in range(1,epochs +1) :
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
    for x,y in train_loader :
        # 1).모델을 device로 옮기기
        x = x.to(device)
        y = y.to(device)
        # 2).순전파, 예측
        




    # ---------- validate ----------
    # 4) 모델을 평가 모드로
    # 5) 기울기 계산을 끈 상태로 val_loader 전체 순회
    # 6) val_loss, val_acc 계산

    # ---------- 기록 ----------
    # 7) epoch, lr, train_loss, train_acc, val_loss, val_acc 를 CSV 한 줄로 append
    # 8) 콘솔에도 한 줄 출력


