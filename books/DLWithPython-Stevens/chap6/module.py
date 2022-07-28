import torch

import torch.nn as nn
from torch import optim

class CustomModel(nn.Module):

    def __init__(self):
        super(CustomModel, self).__init__()
        self.hidden_linear = nn.Linear(1, 8)
        self.hidden_activation = nn.Tanh()
        self.output_linear = nn.Linear(8, 1)

    def forward(self, x):
        out = self.hidden_linear(x)
        out = self.hidden_activation(out)
        out = self.output_linear(out)

        return out


def training_loop(n_epochs, model, loss_fn, optimizer, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        t_p = model(t_u)
        loss = loss_fn(t_p, t_c)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss {loss}")
    

t_c = torch.tensor([0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]).unsqueeze(1)
t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]).unsqueeze(1)
model = CustomModel()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.SoftMarginLoss()
n_epochs = 200
        
training_loop(n_epochs, model, loss_fn, optimizer, t_u, t_c)
