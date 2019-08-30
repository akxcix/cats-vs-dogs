from sklearn import model_selection, utils, metrics
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import os
import numpy as np

# local files
import helpers
import models

TRAIN_PATH = "./data/train"
VAL_TRAIN_RATIO = 0.25
images = helpers.get_list(TRAIN_PATH)

# split features and targets
x = images
y = [helpers.label2num(i.split(".")[0]) for i in images]

# shuffle datafrom sklearn import model_selection, utils, metrics

x, y = utils.shuffle(x, y)

# split into training and validation set
x_train, x_val, y_train, y_val = model_selection.train_test_split(
    x,
    y,
    test_size=VAL_TRAIN_RATIO,
    )
print("Training set length:", len(x_train))
print("Validation set length:", len(x_val))

# Hyperparameters
LEARNING_RATE = 5e-3
BATCH_SIZE = 128
EPOCHS = 20
MOMENTUM = 0.9

# set device. CUDA enabled GPU is preferred
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# tensorboard writer
writer = SummaryWriter("./runs/cats-vs-dogs")

# initialize pytorch classes
model = models.VGG1()
model.to(device)
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=MOMENTUM,
    )

# training loop
for epoch in range(EPOCHS):
    running_loss = 0  # sum losses in each batch

    # batches
    for i in range(0, len(x_train), BATCH_SIZE):
        optimizer.zero_grad()  # clear gradients

        # load X
        batch_x = [
            helpers.load_img(os.path.join(TRAIN_PATH, img))
            for img in x_train[i: BATCH_SIZE+i]
            ]
        batch_x = torch.from_numpy(np.array(batch_x))
        batch_x = batch_x.float()
        batch_x = batch_x.cuda()

        # load Y
        batch_y = [[y] for y in y_train[i: BATCH_SIZE+i]]
        batch_y = torch.from_numpy(np.array(batch_y))
        batch_y = batch_y.float()
        batch_y = batch_y.cuda()

        # forward pass
        y_pred = model(batch_x)
        loss = loss_fn(y_pred, batch_y)

        # backprop
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(epoch, i, loss.item())
        writer.add_scalar("loss/train/batch", loss.item(), (epoch+1)*(i+1))
    writer.add_scalar("loss/train/epoch", running_loss, epoch + 1)

    if epoch % 1 == 0:
        y_val_pred = []
        running_loss = 0
        with torch.no_grad():
            for i in range(0, len(x_val), BATCH_SIZE):

                # load X
                batch_x = [
                    helpers.load_img(os.path.join(TRAIN_PATH, img))
                    for img in x_val[i: BATCH_SIZE+i]
                    ]
                batch_x = torch.from_numpy(np.array(batch_x))
                batch_x = batch_x.float()
                batch_x = batch_x.cuda()

                # load Y
                batch_y = [[y] for y in y_val[i: BATCH_SIZE+i]]
                batch_y = torch.from_numpy(np.array(batch_y))
                batch_y = batch_y.float()
                batch_y = batch_y.cuda()

                y_pred = model(batch_x)
                loss = loss_fn(y_pred, batch_y)
                running_loss += loss.item()

                y_pred = y_pred.cpu().numpy()
                y_val_pred += [1 if i > 0.5 else 0 for i in y_pred]
        val_accuracy = metrics.accuracy_score(y_val, y_val_pred)
        val_mcc = metrics.matthews_corrcoef(y_val, y_val_pred)
        writer.add_scalar("loss/val/epoch", running_loss, epoch+1)
        writer.add_scalar("accuracy/val/epoch", val_accuracy, epoch+1)
        writer.add_scalar("mcc/val/epoch", val_mcc, epoch+1)
writer.close()

# save weights
model.to("cpu")
torch.save(model.state_dict(), "./weights/vgg1-weights.pt")
model.to(device)

y_train_pred = []
with torch.no_grad():
    for i in range(0, len(x_train), BATCH_SIZE):
        batch_x = [
            helpers.load_img(os.path.join(TRAIN_PATH, img))
            for img in x_train[i: BATCH_SIZE+i]
            ]
        batch_x = torch.from_numpy(np.array(batch_x))
        batch_x = batch_x.float()
        batch_x = batch_x.cuda()

        y_pred = model(batch_x)
        y_pred = y_pred.cpu().numpy()
        y_train_pred += [1 if i > 0.5 else 0 for i in y_pred]
helpers.print_metrics("Training", y_train, y_train_pred)

# testing
y_val_pred = []
with torch.no_grad():
    for i in range(0, len(x_val), BATCH_SIZE):
        batch_x = [
            helpers.load_img(os.path.join(TRAIN_PATH, img))
            for img in x_val[i: BATCH_SIZE+i]
            ]
        batch_x = torch.from_numpy(np.array(batch_x))
        batch_x = batch_x.float()
        batch_x = batch_x.cuda()

        y_pred = model(batch_x)
        y_pred = y_pred.cpu().numpy()
        y_val_pred += [1 if i > 0.5 else 0 for i in y_pred]
helpers.print_metrics("Validation", y_val, y_val_pred)
