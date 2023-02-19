# Inspiration: https://pytorch.org/tutorials/beginner/examples_tensor/polynomial_numpy.html

# Goal: we want to approximate transcendant functions using 
# (1) Taylor Series: we want to discover the values of the coefficients for f = a + b*x + c*x^2 + ... => formulas: https://people.math.sc.edu/girardi/m142/handouts/10sTaylorPolySeries.pdf
# (2) Using Neural Networks: NN can be seen as methods to approximate non linear functions 
# (3) Lagrange Interpolation: 
# (4) Fourier Series:

# Methodology for Taylor Series Approximation: 
# 1) generate data and transform them using the function we want to approximate 
# 2) initialize (n+1) weights to approximate function with polynomial of degree n 
# 3) update weights using 

import numpy as np
import scipy as sc
import torch
import math

#  ---------------------------------------------------------------------------

def get_taylor_coefficients_grad_descent(f, n, loss_fn=None, lr=1e-6, num_points=2000, epochs=2000):
    """ 
    Function that approximate transcendental function using taylor approximation 
    calculated numerically using gradient descent and backprop

    Parameters
    ----------
    f: numpy.function
        function to be approximated using taylor series (sin, cos, tan, ...)
    n: int
        polynomial degree
    loss_fn: 
        loss function to use 
    lr: float
        learning rate
    num_points: int
        number of points to be generated
    epochs: int
        number of epochs


    Returns
    -------
    coefficients: list
        list of the polynomial coefficients. If polynomial is degree n, returns list of size n+1
    """
    # generate points using the function
    x = np.linspace(-math.pi, math.pi, num_points)
    y = f(x)

    #  initialize coefficients
    coefficients = [np.random.randn() for _ in range(n+1)]

    # update weights using gradient descent
    for epoch in range(epochs):
        # get current taylor approximation predictions
        prediction = sum([coeff * x**i for i, coeff in enumerate(coefficients)])

        # compute loss function using MSE
        loss = np.square(prediction - y).sum()

        # compute gradient for each coefficients
        gradient_y = 2.0 * (prediction - y)
        gradients_coefficients = [(gradient_y * x ** i).sum() for i in range(len(coefficients))]

        # updating weights
        updated_coefficients = [ coefficients[i] - lr * gradients_coefficients[i]  for i in range(n+1) ]
        coefficients = updated_coefficients

    return coefficients

def get_taylor_coefficients_torch(f, n, loss_fn=None, lr=1e-6, num_points=2000, epochs=2000):
    # generate data
    x = torch.linspace(-math.pi, math.pi, num_points)
    y = f(x)

    # generate polynomial: w0 + w1*x1 + w2*x2 + .. (n times)
    p = torch.tensor([i for i in range(n)])
    xx = x.unsqueeze(-1).pow(p)

    # initialize coefficients
    model = torch.nn.Sequential(
        torch.nn.Linear(n, 1),
        torch.nn.Flatten(0, 1),
    )
    loss_fn = torch.nn.MSELoss(reduction='sum')

    # update weights
    for epoch in range(epochs):
        pred = model(xx)
        loss = loss_fn(pred, y)
        model.zero_grad()
        loss.backward()

        # update weights with gradient descent
        with torch.no_grad():
            for param in model.parameters():
                param -= lr * param.grad

    linear_layer = model[0]
    coefficients = [ linear_layer.bias.item() ]
    coefficients.extend(linear_layer.weight[0].detach().numpy())

    return coefficients

#  ---------------------------------------------------------------------------

def test_get_taylor_coefficients():
    # FIXME: doesn't work with n>3
    print(get_taylor_coefficients_grad_descent(np.sin, 3))
    print(get_taylor_coefficients_grad_descent(np.cos, 3))
    print(get_taylor_coefficients_grad_descent(np.tan, 3))

def test_get_taylor_coefficients_torch():
    print(get_taylor_coefficients_torch(np.sin, 3))
    print(get_taylor_coefficients_torch(np.cos, 3))
    print(get_taylor_coefficients_torch(np.tan, 3))

def main():
    #  test_get_taylor_coefficients()
    test_get_taylor_coefficients_torch()
    


if __name__ == "__main__":
    main()

