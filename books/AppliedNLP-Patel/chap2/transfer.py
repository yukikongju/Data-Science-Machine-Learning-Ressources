# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# AlexNet input_size: [BxCxHxW] => [1x3xhxw]

import torch

from random import randrange
from torch import nn, optim
from torchvision import models, transforms

#  print(dir(models))

class CustomNet(nn.Module):

    def __init__(self, num_classes):
        super(CustomNet, self).__init__()
        self.alexnet = models.alexnet(pretrained = True)
        self.fct1 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.alexnet(x)
        x = torch.tanh(self.fct1(x))
        return x

#  t = torch.rand((1,3, 64, 64))
#  model = CustomNet(num_classes)
#  print(model(t))

### Create Dummy data

num_classes = 2
num_train_imgs = 50
x_train = [torch.rand((1,3,64,64)) for _ in range(num_train_imgs)]
y_train = [randrange(0,num_classes) for _ in range(num_train_imgs)]

### train model


n_epochs = 100

# Method 1: defining custom nn.Module class
#  model = CustomNet(num_classes)

# Method 2: add layer directly to model
model = models.vgg16(pretrained = True)
model.last_layer = nn.Linear(1000, 2)

optimizer = optim.SGD(model.parameters(), lr=1e-3)
loss_fn = nn.NLLLoss()


def training(n_epochs, model, loss_fn, optimizer, x_train, y_train):
    for epoch in range(n_epochs):
        for img, label in zip(x_train, y_train):
            out = model(img)
            loss = loss_fn(out, torch.tensor([label]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(f"Epoch {epoch}, Loss {loss}")

training(n_epochs, model, loss_fn, optimizer, x_train, y_train)
    
