import torch


def model(t_u, w, b):
    return t_u * w + b

def loss_fn(t_p, t_c):
    squared_diff = (t_p - t_c)**2
    return squared_diff.mean()
    
def training(n_epochs, params, t_u, t_c, lr):
    for epoch in range(1, n_epochs + 1):
        if params.grad is not None: 
            params.grad.zero_()

        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        loss.backward()

        with torch.no_grad():
            params -= lr * params.grad

        print(f"Epoch {epoch}, Loss {loss}")

def main():
    t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])
    t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])
    params = torch.tensor([1.0, 0.0], requires_grad=True)

    training(100, params, t_u, t_c, lr=1e-4)

if __name__ == "__main__":
    main()
