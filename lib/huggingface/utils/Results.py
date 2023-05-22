import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def add_result(model, class_, acc, loss, f1):
    results = pd.read_csv("../results.csv", index_col=0)
    results = pd.concat([results, pd.DataFrame({
        "model": model,
        "class": class_,
        "loss": loss,
        "f1score": f1,
        "accuracy": acc,
        "train_time": datetime.now()
    }, index=[0])]).reset_index(drop=True)
    results.to_csv("../results.csv")

def plot_train_history(history, model_name):
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(history["train_acc"], label = "Train")
    ax1.plot(history["val_acc"], label="Validation")
    ax2.plot(history["train_loss"], label="Train")
    ax2.plot(history["val_loss"], label="Validation")
    ax1.legend()
    fig.suptitle(f'Precisión y pérdida {model_name}')