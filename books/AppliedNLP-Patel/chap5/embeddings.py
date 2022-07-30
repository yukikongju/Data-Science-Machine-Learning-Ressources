import torch

from random import randrange

### Create Dummy data


def generate_one_hot_vector(n):
    """ 
    Generate random one hot vector of size n as tensor

    Examples
    --------
    >>> generate_one_hot_vector(3)
    >>> tensor([0., 0., 1.])
    #  >>> [0,0,1]
    """
    t = torch.zeros(n, dtype=int)
    i = randrange(0, n)
    t[i] = 1
    return t
    


### Vectorize vector

t = generate_one_hot_vector(100)

embedding = torch.nn.Embedding(num_embeddings = 100, embedding_dim = 10)
#  print(list(embedding.parameters()))
print(embedding(t))

