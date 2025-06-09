import torch
import torch.nn as nn
from typing import Tuple

class Encoder(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_size)
        self.rnn = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x shape: (seq_length, N)
        embedding = self.dropout(self.embedding(x)) # shape: (seq_length, N, embedding_size)
        out, (hidden, cell) = self.rnn(embedding)
        return out, hidden, cell

