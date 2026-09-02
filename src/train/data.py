# ruff: noqa
import yaml
import os 
import random
import numpy as np

import torch as torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from typing import Dict, Any


hyperparameter = {} # 내부 전역 변수

def show_hyperparameter() : 
	print("<< Hyperparameter Status >> ")
	print("batch_size, epochs, lr, momentum, weight_decay,train_num,val_num, width,seed")
	print(
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

def getTest_dataloaders() :
	test_set = datasets.CIFAR10(
		root="./data", 
		train=False, 
		download=True
	)



if __name__ == "__main__" :
    print("<< Direct Call >> \n")
    print("<< Day 1 Lab : Dataloading >> \n")