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
    test_rnn()
    test_dummy_rnn()

