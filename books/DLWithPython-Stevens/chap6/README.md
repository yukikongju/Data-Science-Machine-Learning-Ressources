# Chap 6 - Using a Neural Network to fit the data

[Code](https://github.com/deep-learning-with-pytorch/dlwpt-code/tree/master/p1ch6)

- [X] Neural network: coding a neural network with our own model and loss function
- [ ] Activation Function: plotting the activation function
- [X] NN Module: using pytorch nn module class


**Notes**

The activation function must be: 
    1. be differentiable
    2. non-linear

```python

  seq_model = nn.Sequential(OrderedDict([
      ('conv1', nn.Conv2d(3, 16, kernel_size=3, padding=1)),
      ('act1', nn.Tanh()),
      ('pool1', nn.MaxPool2d(2)),
      ('conv2', nn.Conv2d(16, 8, kernel_size=3, padding=1)),
      ('act2', nn.Tanh()),
      ('pool2', nn.MaxPool2d(2)),
      ('fct1', nn.Linear(8*8*8, 32)),
      ('fct2', nn.Linear(32, 2)),
      ('act3', nn.Tanh()),
  ]))
```



