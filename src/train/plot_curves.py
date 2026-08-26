
# ruff: noqa
import matplotlib.pyplot as plt
import pandas as pd



def plot(fileName : str,saveAddr : str = "../../img/train_plt.png") : 
    # 1). Train 데이터 가져오기
    df =pd.read_csv(fileName)
    epochs = df["epoch"]
    train_loss = df["train_loss"]
    val_loss = df["val_loss"]
    train_acc = df["train_acc"]
    val_acc = df["val_acc"]
    

    # 2). 그래프 생성
   
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 왼쪽: Loss 그래프 (2개 변수)
    ax1.plot(epochs, train_loss, label="Train Loss", color="blue", marker="o")
    ax1.plot(epochs, val_loss, label="Val Loss", color="orange", marker="o")
    ax1.set_title("Loss Trend")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # 오른쪽: Accuracy 그래프 (2개 변수)
    ax2.plot(epochs, train_acc, label="Train Acc", color="green", marker="s")
    ax2.plot(epochs, val_acc, label="Val Acc", color="red", marker="s")
    ax2.set_title("Accuracy Trend")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(saveAddr)
    #plt.show()
    plt.close()
        

if __name__ == "__main__" :
    fileName = "training_log.csv"
    plot(fileName) 