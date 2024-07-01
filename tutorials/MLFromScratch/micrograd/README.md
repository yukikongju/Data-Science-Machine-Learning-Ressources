# micrograd

How does backpropagation being implemented in the backend? We aim to do it 
from scratch, starting from a neuron to an MLP

Classes:
- [ ] Value
- [ ] Layer
- [ ] MLP


# Concepts

**Forward Pass - How to compute the gradient of each weight: dL/dw**


**Backward Pass - How to update weight value**

- when we apply the backward prop, it's updating the two previous node
- we call backprop() for all of the weights node in its reversed topological order


# Resources

- [Andrej Kaparthy - The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2&ab_channel=AndrejKarpathy)

