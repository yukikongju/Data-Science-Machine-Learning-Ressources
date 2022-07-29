import torch

import torch.nn as nn
import torch.optim as optim

from torchvision import transforms, datasets


cifar10 = datasets.CIFAR10('data', train=False, download=True,
        transform=transforms.ToTensor())

model = nn.Sequential(
            nn.Linear(3072, 512),
            nn.Tanh(),
            nn.Linear(512, 10),
            nn.LogSoftmax(dim=1))

learning_rate = 1e-2

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

loss_fn = nn.NLLLoss()

n_epochs = 100

for epoch in range(n_epochs):
    for img, label in cifar10:
        t_u = img.view(-1).unsqueeze(0)
        out = model(t_u)
        print(t_u.shape)

        loss = loss_fn(out, torch.tensor([label]))
                
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss {loss}")

