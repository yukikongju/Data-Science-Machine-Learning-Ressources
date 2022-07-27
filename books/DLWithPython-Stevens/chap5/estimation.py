import torch
import matplotlib.pyplot as plt

def model(t_u, w, b):
    """ 
    linear function: t_c = w * t_u + b

    Parameters
    ----------
    t_u: tensor
        tensor
    w: float
        weight of the neuron
    b: bias
        bias of the neuron
    """
    return t_u * w + b
    
def loss_fn(t_p, t_c):
    """ 
    Compute residual mean

    Parameters
    ----------
    t_p: tensor
        predicted values
    t_c: tensor
        real values
    """
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()
    

def gradient_descent(t_u, t_c, w, b, delta=0.1, lr=1e-2):
    """ 
    Update parameters using gradient descent 

    Parameters
    ----------

    """
    loss_rate_of_change = \
            (loss_fn(model(t_u, w + delta, b), t_c) - 
            loss_fn(model(t_u, w - delta, b), t_c)) / (2.0 * delta)

    w -= lr * loss_rate_of_change
    b -= lr * loss_rate_of_change


def dloss(t_p, t_c):
    """ 
    compute loss function derivative

    Parameters
    ----------
    t_p: tensor
        predicted
    t_c: tensor
        real value
    """
    dsq_diffs = 2 * (t_p - t_c) / t_p.size(0)
    return dsq_diffs

def dmodel_dw(t_u, w, b):
    """ 
    return weight derivative
    """
    return t_u

def dmodel_db(t_u, w, b):
    """ 
    return bias derivative
    """
    return 1.0

def grad_fn(t_u, t_c, t_p, w, b):
    """ 
    Compute linear function gradient in repsect of partial derivatives
    """
    dloss_dtp = dloss(t_p, t_c)
    dloss_dw = dloss_dtp * dmodel_dw(t_u, w, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w, b)
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])
    
    
def training(n_epochs, params, t_u, t_c, lr):
    for epoch in range(1, n_epochs + 1):
        w, b = params

        t_p = model(t_u, w, b)
        loss = loss_fn(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p, w, b)

        params = params - lr * grad

        print(f"Epoch {epoch}, Loss {loss}")
        print(f"    Params {params}")
        print(f"    Grad {grad}")

    return params

def show_scatterplot(t_c, t_u):
    plt.scatter(t_c, t_u)
    plt.show()
    

def main():
    # create tensor
    t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])
    t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])

    # show scatterplot
    show_scatterplot(t_c, t_u)
    
    # train
    params = torch.tensor([1.0, 0.0])
    training(n_epochs=100, params=params, t_u=t_u, t_c=t_c, lr=1e-4)


if __name__ == "__main__":
    main()



