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
	
if __name__ == "__main__" :
    print("<< Direct Call >> \n")
    print("<< Day 1 Lab : Dataloading >> \n")

	