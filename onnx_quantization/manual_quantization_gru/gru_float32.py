import torch
import torch.nn as nn
from torch.onnx import export
import numpy as np

class ReimplementedGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ReimplementedGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_ir = nn.Linear(input_size, hidden_size)
        self.W_hr = nn.Linear(hidden_size, hidden_size)

        self.W_iz = nn.Linear(input_size, hidden_size)
        self.W_hz = nn.Linear(hidden_size, hidden_size)

        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_hn = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, hidden):
        r_t = torch.sigmoid(self.W_ir(x) + self.W_hr(hidden))
        z_t = torch.sigmoid(self.W_iz(x) + self.W_hz(hidden))
        n_t = torch.tanh(self.W_in(x) + r_t * self.W_hn(hidden))
        #ones = torch.ones(z_t.shape)
        hidden = (1 - z_t) * n_t + z_t * hidden
        return hidden


input_size = 3
hidden_size = 4
model = ReimplementedGRU(input_size, hidden_size)
dummy_input = torch.randn(1, input_size)
hidden = torch.zeros(1, hidden_size)
torch.onnx.export(model, (dummy_input, hidden), "gru_reimplemented_1.onnx")