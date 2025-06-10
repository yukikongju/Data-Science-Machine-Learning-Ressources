import torch.nn as nn

class Decoder(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(num_embeddings=input_size, 
                                      embedding_dim=embedding_size)
        self.rnn = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x, hidden, cell):
        out = x.unsqueeze(0) # x shape: (N, ) but we want (1, N)
        embedding = self.dropout(self.embedding(out)) # embedding shape: (1, N, embedding_size)
        out, (hidden, cell) = self.rnn(embedding, (hidden, cell)) # rnn shape: (1, N, hidden_size)
        out = self.fc(out) # output shape: (N, output_size)
        outputs = out.squeeze(0)
        return outputs
        
