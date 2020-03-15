import numpy as np
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd


def epoch_loss(file_name):
    loss = list()
    val_loss = list()
    with open(file_name, "r") as f:
        tmp_loss = list()
        tmp_val_loss = list()
        for l in f.readlines():
            if l.startswith("Training finished"):
                loss.append(tmp_loss)
                val_loss.append(tmp_val_loss)
                tmp_loss = list()
                tmp_val_loss = list()
                continue
            if l.find("loss") != -1 and l.find("val_loss") != -1:
                tmp_loss.append(float(re.findall(" loss: (\d+.\d+)", l)[0]))
                tmp_val_loss.append(float(re.findall(" val_loss: (\d+.\d+)", l)[0]))
    return loss, val_loss


def plot(file_names, eta, clipnorm, markers=None, save_to_csv=False, save_fig=False):
    dict_ = defaultdict(list)

    for i, (file_name, clip) in enumerate(zip(file_names, clipnorm)):
        plt.figure(figsize=(9, 7))
        loss, val_loss = epoch_loss(file_name=file_name)
        plt.title(f"Loss (gradient clipnorm: {clip})", size=15)

        if save_to_csv:
            for eta_l, eta_vl, e in zip(loss, val_loss, eta):
                for l, vl in zip(eta_l, eta_vl):
                    dict_["clipnorm"].append(clip)
                    dict_["eta"].append(e)
                    dict_["training_loss"].append(l)
                    dict_["validation_loss"].append(vl)

        for i, (l, vl) in enumerate(zip(loss, val_loss)):
            x_range = range(1, len(l) + 1)
            plt.plot(x_range, l, marker=markers[2 * i])
            plt.plot(x_range, vl, marker=markers[2 * i + 1])
        plt.legend(
            [f"noise eta={e} ({s})" for e in eta for s in ["Train", "Validation"]]
        )
        plt.grid()
        plt.xlabel("Epoch", size=15)
        plt.ylabel("Loss (Categorical Cross Entropy)", size=15)
        plt.xticks(size=12)
        plt.yticks(size=12)
        if save_fig:
            plt.savefig(f"loss_clipnorm_{clip}.png", dpi=200)
        plt.show()

    if save_to_csv:
        df = pd.DataFrame.from_dict(dict_)
        df.to_csv("loss.csv")


if __name__ == "__main__":
    eta = [0.01, 0.1, 1, 10]
    clipnorm = [1.0, 0.7, 0.5, 0.3, 0.1]
    files = [f"variance_test_clipnorm_{c}.log.txt" for c in clipnorm]
    markers = ["o", "*", "v", "s", "8", "p", "P", "h"]
    plot(files, eta, clipnorm, markers, save_fig=True, save_to_csv=True)
