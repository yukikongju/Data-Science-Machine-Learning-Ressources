import torch

from random import randrange
from model import CNN
from torch import optim, nn


### Dummy Data

num_imgs = 100
num_classes = 10
imgs = [torch.rand((1,3,32,32)) for _ in range(num_imgs)]
labels = [randrange(0,10) for _ in range(num_imgs)]


### Weight Decay with optimizer

n_epochs = 100
model = CNN(in_channels=3, num_classes=num_classes)
optimizer_wd = optim.SGD(model.parameters(), lr=1e-4, weight_decay=0.8, momentum=0.2)
loss_fn = nn.CrossEntropyLoss()

def training(n_epochs, model, loss_fn, optimizer, x_train, y_train):
    for epoch in range(n_epochs):
        for img, label in zip(x_train, y_train):
            t_p = model(img)
            loss = loss_fn(t_p, torch.tensor([label]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}, Loss {loss}")
    
training(n_epochs, model, loss_fn, optimizer_wd, imgs, labels)

### Weight Decay inside training loop

def training_with_decay(n_epochs, model, loss_fn, optimizer, x_train, y_train):
    for epoch in range(n_epochs):
        for img, label in zip(x_train, y_train):
            t_p = model(img)
            loss = loss_fn(t_p, torch.tensor([label]))

            l2_lambda = 1e-2
            l2_norm = sum(param.pow(2.0).sum() for param in model.parameters())
            loss = loss + l2_lambda * l2_norm


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}, Loss {loss}")

optimizer = optim.SGD(model.parameters(), lr=1e-4)
training_with_decay(n_epochs, model, loss_fn, optimizer, imgs, labels)

