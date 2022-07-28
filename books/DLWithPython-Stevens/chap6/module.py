import torch

import torch.nn as nn
from torch import optim

class CustomModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden_linear = nn.Linear(1, 8)
        self.hidden_activation = nn.Tanh()
        self.output_linear = nn.Linear(8, 1)

    def forward(self, input):
        hidden_t = self.hidden_linear(input)
        activated_t = self.hidden_activation(hidden_t)
        output_t = self.output_linear(activated_t)

        return output_t

model = CustomModel()
print(model)
        

