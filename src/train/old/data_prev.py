import os

import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader



## ============== 데이터셋을 정리 ============== 

train_full = datasets.CIFAR10(root="./data", train=True,  download=True)
test_set   = datasets.CIFAR10(root="./data", train=False, download=True)
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



