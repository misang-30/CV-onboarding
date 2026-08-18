# ### 1). 모델 스펙
# #### Layer 1 

# - `Conv3x3(pad=1) → BN → ReLU → Conv3x3(pad=1) → BN → ReLU → MaxPool2x2`
# - 입출력 : `3×32×32` → `C×16×16`


# #### Layer 2 
# - `Conv3x3(pad=1) → BN → ReLU → Conv3x3(pad=1) → BN → ReLU → MaxPool2x2`
# - 입출력 : `C×16×16` → `2C×8×8`

# #### Layer 3 
# - `Conv3x3(pad=1) → BN → ReLU → Conv3x3(pad=1) → BN → ReLU → MaxPool2x2`
# - 입출력 : `2C×8×8` → `4C×4×4`

# #### Head 
# - `AdaptiveAvgPool2d(1) → Flatten → Linear(4C, 10)`
# - 입출력 : `4C×4×4` → `10`

# 3x32x32 이미지를 필터 3x3x3에 통과 시켠 1x32x32 1개가 나온다.

# 필터가 C 개이면 결과물은 1x32x32xC 가 된다. (C개의 필터가 각각 1x32x32 결과물을 내놓는다.)
# 따라서 결과물은 Cx32x32 가 된다.




# 1. BN : 배치 정규화
# - 딥러닝 학습시 데이터들의 숫자 범위를 일정하게 다듬어 주는 정돈 작업
# - 텐서의 형태(Shape)는 전혀 바꾸지 않고, 안에 들어있는 숫자 분포만 예쁘게 조정합니다.

# 2.torch.nn
# - pytorch에서 제공하는 신경망 모듈을 구현한 패키지
# - super().__init__()을 호출하고 nn.Module을 상속받는 순간, PyTorch가 모델 안의 모든 가중치(파라미터)들을 자동으로 추적하고 학습(미분)할 수 있게 만들어 줍니다.


import torch
import torch.nn as nn


class SmallCNN ( nn.Module) :
    def __init__ (self, width: int =32, dropout_p : float = 0.5) :
        super().__init__()
        self.width = width
        self.dropout_p = dropout_p

        # nn.Conv2d(in_channels, out_channels(필터개수), kernel_size, stride=1, padding=0)

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
 

    