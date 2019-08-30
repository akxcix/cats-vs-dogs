import os
from PIL import Image
from sklearn import metrics
import numpy as np


def get_list(dir_path):
    files = [
        f for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f))
        ]
    return files


def load_img(img_path):
    img = Image.open(img_path)
    img = img.resize((200, 200))
    img = np.asarray(img, dtype=float)
    img = img/256
    img = np.moveaxis(img, -1, 0)
    return img


def label2num(x):
    if x == "dog":
        return 1
    return 0


def print_metrics(name, correct, pred):
    print("-----------", "name", "----------------")
    print("  Accuracy  : %.5f" % metrics.accuracy_score(correct, pred))
    print("  Precision : %.5f" % metrics.precision_score(correct, pred))
    print("  Recall    : %.5f" % metrics.recall_score(correct, pred))
    print("  F1 Score  : %.5f" % metrics.f1_score(correct, pred))
    print("  MCC       : %.5f" % metrics.matthews_corrcoef(correct, pred))
