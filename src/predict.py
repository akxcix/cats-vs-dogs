import torch
import numpy as np

# local files
from helpers import load_img
from models import VGG1


def predict(img):

    img = np.array([img])
    img = torch.from_numpy(img)
    img = img.float()

    with torch.no_grad():
        y = model(img)

    if y.item() > 0.5:
        category = "DOG"
    else:
        category = "CAT"

    return (y.item(), category)


# load model
model = VGG1()
model.load_state_dict(torch.load("./weights/vgg1-weights.pt"))

# predict
img = load_img("./data/test1/1001.jpg")
pred = predict(img)
print("Confidence: {:0.7f}\nClass: {:s}".format(pred[0], pred[1]))
