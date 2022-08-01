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
        x = self.embed(x).unsqueeze(0) # size: [1 x 1 x contex_dim]
        if h is None:
            _, x = self.rnn(x)
        else: 
            _, x = self.rnn(x, h)
        x = self.fct1(x)
        x = torch.sigmoid(self.fct2(x))
        return x
        

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
    test_dummy_rnn_embedding()
    test_rnn_embedding()

