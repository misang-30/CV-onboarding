# ruff: noqa
import torch as torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms

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


class ResNet18Scratch(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 1. 가중치 없이 기본 ResNet18 생성 (Stem: (7x7 conv, stride 2, padding 3 + 3x3 maxpool, stride 2, padding 1))
        # Stem : 모델 구조에서 입력 이미지를 가장 먼저 받아들이는 모델의 뿌리에 해당하는 극초반 레이어
        self.model = models.resnet18(weights=None)
        
        # 2. 마지막 FC 레이어만 CIFAR-10 클래스 수(10개)로 교체
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

class ResNet18FineTun(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 1. ImageNet 사전학습 가중치 로드
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.model = models.resnet18(weights=weights)
        
        # 2. 마지막 fc 레이어 교체
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)





if __name__ == "__main__" :
    print("<< Direct Call >> \n ")
    print("<< Day 2 Lab : Model Build >>\n")

    print(">> width 16")
    model = SmallCNN(width=16)
    x = torch.randn(2, 3, 32, 32)
    print(model(x).shape)                                              # (2, 10) 이어야 함
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    print(">> width 32")
    model = SmallCNN(width=32)
    x = torch.randn(2, 3, 32, 32)
    print(model(x).shape)                                              # (2, 10) 이어야 함
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    print(">> width 64")
    model = SmallCNN(width=64)
    x = torch.randn(2, 3, 32, 32)
    print(model(x).shape)                                              # (2, 10) 이어야 함
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))