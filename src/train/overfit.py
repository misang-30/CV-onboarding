# ruff: noqa
import csv


import torch as torch
from torch import nn
from torch import optim
from model import SmallCNN
from data  import get_dataloaders
from data  import get_hyperparameter
from train import train
from plot_curves import plot


training_log = "training_log_overfit.csv"
plotAddr = "../../img/overfitting_plot.png"


def overfitLab() : 
    hyperparameter = get_hyperparameter("configs/baseline_overfit.yaml") 
    train_loader, val_loader = get_dataloaders()
    train(train_loader, val_loader, hyperparameter, training_log, wandbOn =False)
    plot(training_log, plotAddr )



    
if __name__ == "__main__" :
    print("<< Direct Call >>")
    print("<< Day 4 Lab : Overfitting >>")
    overfitLab() 