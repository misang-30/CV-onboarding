# ruff: noqa
import torch as torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from typing import Dict, Any

batch_size = 128
epochs = 30 # 에폭 수 , 총 데이터셋을 몇번 돌릴지.
learning_rate = 0.005 # 학습률, 가중치 업데이트 시 얼마나 크게 반영할지 결정하는 하이퍼파라미터
momentum = 0.9 # 모멘텀, 이전 기울기 정보를 얼마나 반영할지 결정하는 하이퍼파라미터
weight_decay = 0.0 # 가중치 감쇠(regularization) 정도를 결정하는 하이퍼파라미터
train_num = 45000 # train data 수
val_num = 5000 # validation data 수

def show_hyperparameter() : 
	print("<< Hyperparameter Status >> ")
	print("batch_size, epochs, learning_rate, momentum, weight_decay,train_num,val_num")
	print(f"{batch_size} / {epochs} / {learning_rate} / {momentum} / {weight_decay} / {train_num} / {val_num} ")


def get_hyperparameter ():
	hyperparmeter = {
		"batch_size": batch_size,
		"epochs" : epochs,
		"lr" : learning_rate,
		"momentum" : momentum,
		"weight_decay" : weight_decay,
		"train_num" : train_num,
		"val_num" : val_num
	}
	show_hyperparameter()
	return hyperparmeter

def set_hyperparameter(config: Dict[str, Any]):
	global batch_size, epochs, learning_rate, momentum, weight_decay,train_num,val_num

	batch_size = config["batch_size"]
	epochs = config["epochs"]
	learning_rate = config["lr"]
	momentum = config["momentum"]
	weight_decay = config["weight_decay"]
	train_num = config["train_num"]
	val_num = config["val_num"]
	show_hyperparameter()

def get_dataloaders (): 
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
	f_, b_ = random_split(
		train_full,
		[train_num+val_num,len(train_full)-(train_num+val_num)],
		generator = generator
	)
	train_data, val_data = random_split(
		f_,
		[train_num, val_num],
		generator = generator
	)
	# print("train data :", len(train_data))
	# print("validation data :", len(val_data))



	# 3). 데이터 로더 설정
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
	show_hyperparameter()
	return train_loader, val_loader

def getTest_dataloaders() :
	test_set = datasets.CIFAR10(
		root="./data", 
		train=False, 
		download=True
	)



if __name__ == "__main__" :
    print("Direct Call")