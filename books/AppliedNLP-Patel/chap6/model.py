import torch 
import torch.nn.functional as F

from torch import optim, nn

class DummyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(DummyRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.fct1 = nn.Linear(hidden_size, 8)
        self.fct2 = nn.Linear(8, input_size)

    def forward(self, x, h=None):
        if h is None:
            _, x = self.rnn(x)
        else: 
            _, x = self.rnn(x, h)
        x = self.fct1(x)
        x = self.fct2(x)
        return x

class RNNEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_dim):
        super(RNNEmbedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, embedding_dim)
        self.fct1 = nn.Linear(embedding_dim, 8)
        self.fct2 = nn.Linear(8, vocab_size)

    def forward(self, x, h=None):
        x = self.embed(x).unsqueeze(0) # size: [1 x 1 x context_dim]
        if h is None:
            _, x = self.rnn(x)
        else: 
            _, x = self.rnn(x, h)
        x = self.fct1(x)
        x = torch.sigmoid(self.fct2(x))
        return x
        

class RNNEmbeddingFlatten(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_dim):
        super(RNNEmbeddingFlatten, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, embedding_dim)
        self.fct1 = nn.Linear(embedding_dim * context_dim, 8)
        self.fct2 = nn.Linear(8, vocab_size)

    def forward(self, x, h=None):
        x = self.embed(x).unsqueeze(0) # size: [1 x 1 x context_dim]
        if h is None:
            _, x = self.rnn(x)
        else: 
            _, x = self.rnn(x, h)
        #  x = x.reshape(x.shape[0], -1) # size: [1 x (embedding_dim * context_dim)]
        x = x.view((1,-1))
        x = self.fct1(x)
        x = torch.sigmoid(self.fct2(x))
        return x


class BidirectionalRNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_dim):
        super(BidirectionalRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, embedding_dim, batch_first=True, 
                bidirectional=True)
        self.fct1 = nn.Linear(embedding_dim * context_dim, 12)
        self.fct2 = nn.Linear(12, vocab_size)

    def forward(self, x):
        x = self.embed(x).unsqueeze(0)
        _, x = self.rnn(x)
        # size: [1 x (embedding_dim*context_dim)] ; [(embedding_dim*context_dim) x 12]
        x = torch.cat((x[0], x[1]), dim=-1) 
        x = self.fct1(x)
        x = torch.tanh(self.fct2(x))
        return x

def test_bidirectional_rnn():
    vocab_size, embedding_dim, context_dim = 50, 10, 2
    t = torch.tensor([4, 18]) # size: 1 x context_dim
    model = BidirectionalRNN(vocab_size, embedding_dim, context_dim)
    print(model(t))
        

def test_nn_embedding_flatten():
    vocab_size, embedding_dim, context_dim = 50, 10, 2
    t = torch.tensor([4, 18]) # size: 1 x context_dim
    model = RNNEmbeddingFlatten(vocab_size, embedding_dim, context_dim)
    print(model(t))


def test_rnn_embedding():
    vocab_size, embedding_dim, context_dim = 50, 10, 2
    t = torch.tensor([4, 18]) # size: 1 x context_dim
    model = RNNEmbedding(vocab_size, embedding_dim, context_dim)
    print(model(t))

def test_dummy_rnn_embedding():
    vocab_size, embedding_dim, context_dim = 50, 10, 2
    embed = nn.Embedding(vocab_size, embedding_dim)
    t = torch.tensor([4, 18]) # size: 1 x context_dim
    t1 = embed(t).unsqueeze(0)
    rnn = nn.RNN(embedding_dim, embedding_dim)
    print(rnn(t1))


def test_dummy_rnn():
    t = torch.randn(7, 5, 12)
    model = DummyRNN(12, 22, 4)
    print(model(t))
    
def test_rnn():
    t = torch.randn(7, 5, 12)
    h = torch.randn(4, 5, 22)
    model = nn.RNN(input_size=12, hidden_size=22, num_layers=4)
    _, t1 = model(t)
    _, t2 = model(t, h)
    print(t1.shape, t2.shape)
    print(model)
    fct1 = nn.Linear(in_features=22, out_features=12)
    t3 = fct1(t2)

        
if __name__ == "__main__":
    #  test_rnn()
    #  test_dummy_rnn()
    #  test_dummy_rnn_embedding()
    #  test_rnn_embedding()
    test_nn_embedding_flatten()
    test_bidirectional_rnn()

