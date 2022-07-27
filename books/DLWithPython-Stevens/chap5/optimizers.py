import torch 
from torch import optim

#  print(dir(optim))

def model(t_u, w, b):
    return w * t_u + b

def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()

def training(n_epochs, optimizer, lr, params, t_c, t_u):
    for epoch in range(1, n_epochs +1):

        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss {loss}")

    return params

def main():
    t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])
    t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])
    params = torch.tensor([1.0, 0.0], requires_grad=True)
    learning_rate = 1e-4
    n_epochs = 5000
    optimizer = optim.SGD([params], lr=learning_rate)

    training(n_epochs, optimizer, learning_rate, params, t_c, t_u)


if __name__ == "__main__":
    main()
    



