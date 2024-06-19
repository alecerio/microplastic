import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(41, 32)
        self.fc2 = nn.Linear(32, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

model = MLP()
export_onnx_model = False
if export_onnx_model:
    mlp_input = torch.randn(1, 41, dtype=torch.float32)
    torch.onnx.export(model, mlp_input, "C/models/MLP.onnx", export_params=True, opset_version=14)