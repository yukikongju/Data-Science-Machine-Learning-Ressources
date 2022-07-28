import torch
import torch.nn as nn

from torch import optim

from collections import OrderedDict

def mean_squared_errors(t_p, t_c):
    sq_diffs = (t_p - t_c)**2
    return sq_diffs.mean()
    
def training(n_epochs, lr, optimizer, model, loss_fn, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        t_p = model(t_u)
        loss = loss_fn(t_p, t_c)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch <= 4: 
            print(f"Epoch {epoch}, Loss {loss}")
        elif epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss {loss}")


# initialize model  
t_c = torch.tensor([0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]).unsqueeze(1)
t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]).unsqueeze(1)
linear_model = nn.Linear(1, 1)
optimizer = optim.SGD(linear_model.parameters(), lr=1e-4)


# check model param
print([param.shape for param in linear_model.parameters()])

# train with our custum loss function
training(n_epochs=20000, lr=1e-4, optimizer=optimizer, model=linear_model, loss_fn=mean_squared_errors, t_u=t_u, t_c=t_c)


# train using pytorch loss function
training(n_epochs=20000, lr=1e-4, optimizer=optimizer, model=linear_model, loss_fn=nn.MSELoss(), t_u=t_u, t_c=t_c)

# train using sequential model
seq_model = nn.Sequential(OrderedDict(
    ('hidden_linear', nn.Linear(1, 8)),
    ('hidden_activation', nn.Tanh()),
    ('output_linear', nn.Linear(8, 1))
))
seq_optimizer = optim.Adam(seq_model.parameters(), lr=1e-2)
training(n_epochs=20000, lr=1e-4, optimizer=seq_optimizer, model=seq_model, loss_fn=nn.MSELoss(), t_u=t_u, t_c=t_c)

