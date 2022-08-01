import torch

from torch import nn


class SimpleLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(SimpleLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)
        self.fct1 = nn.Linear(embedding_dim, 12)
        self.fct2 = nn.Linear(12, context_size)


    def forward(self, x):
        x = self.embed(x).unsqueeze(0)
        x, _ = self.lstm(x)
        x = self.fct1(x)
        x = self.fct2(x)
        return x

class BidirectionalLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(BidirectionalLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, batch_first=True, 
                bidirectional=True)
        self.fct1 = nn.Linear(embedding_dim * context_size, 12)
        self.fct2 = nn.Linear(12, context_size)


    def forward(self, x):
        x = self.embed(x).unsqueeze(0)
        x, _ = self.lstm(x)
        x = self.fct1(x)
        x = self.fct2(x)
        return x


def test_bidirectional_lstm():
    vocab_size, embedding_dim, context_size = 50, 10, 2
    t = torch.tensor([3,21]) # (['word1', 'word2'], 'word3'] => ([3, 21], 45)
    model = BidirectionalLSTM(vocab_size, embedding_dim, context_size)
    print(model(t))

def test_simple_lstm():
    vocab_size, embedding_dim, context_size = 50, 10, 2
    t = torch.tensor([3,21]) # (['word1', 'word2'], 'word3'] => ([3, 21], 45)
    model = SimpleLSTM(vocab_size, embedding_dim, context_size)
    print(model(t))
    
def test_lstm():
    vocab_size, embedding_dim, context_size = 50, 10, 2
    t = torch.tensor([3, 21]).unsqueeze(0)
    embed = nn.Embedding(vocab_size, embedding_dim)
    lstm = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)

    t1 = embed(t)
    t2 = lstm(t1)
    print(t2)
    
if __name__ == "__main__":
    test_lstm()
    test_simple_lstm()
    test_bidirectional_lstm()
