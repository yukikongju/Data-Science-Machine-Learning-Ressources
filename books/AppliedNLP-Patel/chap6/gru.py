import torch

from torch import nn

class SimpleGRU(nn.Module):

    def __init__(self, vocab_size, embedding_size, context_size):
        super(SimpleGRU, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, embedding_size)
        self.fct1 = nn.Linear(embedding_size, 12)
        self.fct2 = nn.Linear(12, context_size)

    def forward(self, x):
        x = self.embed(x).unsqueeze(0)
        _, x = self.gru(x)
        x = self.fct1(x)
        x = self.fct2(x)
        return x
        
class BidirectionalGRU(nn.Module):

    def __init__(self, vocab_size, embedding_size, context_size):
        super(BidirectionalGRU, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, embedding_size, batch_first=True, 
                bidirectional=True)
        self.fct1 = nn.Linear(embedding_size * context_size, 12)
        self.fct2 = nn.Linear(12, context_size)

    def forward(self, x):
        x = self.embed(x).unsqueeze(0)
        _, x = self.gru(x)
        x = x.view((1,-1))
        x = self.fct1(x)
        x = self.fct2(x)
        return x

def test_bidirectional_gru():
    vocab_size, embedding_dim, context_size = 50, 10, 2
    t = torch.tensor([3,21]) # (['word1', 'word2'], 'word3'] => ([3, 21], 45)
    model = BidirectionalGRU(vocab_size, embedding_dim, context_size)
    print(model(t))

def test_simple_gru():
    vocab_size, embedding_dim, context_size = 50, 10, 2
    t = torch.tensor([3,21]) # (['word1', 'word2'], 'word3'] => ([3, 21], 45)
    model = SimpleGRU(vocab_size, embedding_dim, context_size)
    print(model(t))
    

if __name__ == "__main__":
    test_simple_gru()
    test_bidirectional_gru()
        


