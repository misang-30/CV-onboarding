## 0. 저장소
```
cv-onboarding/
├── docs/
│   ├── 01-ubuntu-install.md
│   ├── ...
│   ├── 05-training-basics.md     # 이번 주 문서 1: 학습 파이프라인 구축 기록
│   └── 06-experiments.md         # 이번 주 문서 2: 실험 노트 (핵심 산출물)
└── src/
    └── train/
        ├── data.py               # Dataset / DataLoader / split
        ├── model.py              # 직접 구현한 CNN
        ├── train.py              # 학습 루프
        ├── plot_curves.py        # CSV → 곡선 그리기 (Day 1~2용)
        └── configs/
            └── baseline.yaml
```
---

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

![](img/cifar10_samples.png)

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



```


### 3). 체크

``` python

```



---
## 4. 학습 루프 구현 (Day 2)




----
## 5. 과적합 제조 (Day 3)





---
## 6. 실험관리도구 (Day 3)





---
## 7. 통제 실험(Day 4)



---
## 8. 함정 체험 (Day 5)


