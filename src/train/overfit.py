# ruff: noqa
import csv


import torch as torch
from torch import nn
from torch import optim
from model import SmallCNN
from data  import get_dataloaders
from data  import get_hyperparameter
from data  import set_hyperparameter
from train import train
from plot_curves import plot

hyperparameter = {
    "batch_size": 128,
    "epochs" : 100,
    "lr" : 0.005,
    "momentum" : 0.9,
    "weight_decay" : 0,
    "train_num" : 500,
    "val_num" : 5000
}
	
training_log = "training_log_overfit.csv"
plotAddr = "../../img/overfitting_plot.png"
width = 64

def overfitLab() : 
    set_hyperparameter(hyperparameter) 
    train_loader, val_loader = get_dataloaders()
    train(train_loader, val_loader, hyperparameter, training_log, width)
    plot(training_log, plotAddr )



    
if __name__ == "__main__" :
    print("<< Direct Call >>")
    print("<< Day 4 Lab : Overfitting >>")
    overfitLab() 