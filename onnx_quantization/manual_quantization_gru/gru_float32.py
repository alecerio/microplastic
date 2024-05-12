import torch
import torch.nn as nn
from torch.onnx import export
import numpy as np

def linear_weight_init(tensor, start_value=0.1, increment=0.05):
    num_rows, num_cols = tensor.size()    
    new_tensor = torch.empty_like(tensor)
    for i in range(num_rows):
        row_start = start_value + i * increment
        new_tensor[i] = torch.arange(row_start, row_start + (num_cols-1) * increment, num_cols)
    return new_tensor

def linear_bias_init(tensor, start_value=0.1, increment=0.05):
    tensor_size = tensor.size()   
    temp_tensor = []
    for i in range(0, tensor_size[0]):
        row_start = start_value + i * increment
        temp_tensor.append(row_start)
    new_tensor = torch.tensor(temp_tensor)
    return new_tensor

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

        # Apply the new initialization method
        self.W_ir.weight.data = linear_weight_init(self.W_ir.weight.data, start_value=-1.5, increment=0.01)
        self.W_hr.weight.data = linear_weight_init(self.W_hr.weight.data, start_value=-1, increment=0.01)
        self.W_iz.weight.data = linear_weight_init(self.W_iz.weight.data, start_value=0.5, increment=0.01)
        self.W_hz.weight.data = linear_weight_init(self.W_hz.weight.data, start_value=0, increment=0.01)
        self.W_in.weight.data = linear_weight_init(self.W_in.weight.data, start_value=0.5, increment=0.01)
        self.W_hn.weight.data = linear_weight_init(self.W_hn.weight.data, start_value=1, increment=0.01)

        self.W_ir.bias.data = linear_bias_init(self.W_ir.bias.data, start_value=-1.5, increment=0.01)
        self.W_hr.bias.data = linear_bias_init(self.W_hr.bias.data, start_value=-1, increment=0.02)
        self.W_iz.bias.data = linear_bias_init(self.W_iz.bias.data, start_value=-0.5, increment=0.03)
        self.W_hz.bias.data = linear_bias_init(self.W_hz.bias.data, start_value=0, increment=0.04)
        self.W_in.bias.data = linear_bias_init(self.W_in.bias.data, start_value=1, increment=0.05)
        self.W_hn.bias.data = linear_bias_init(self.W_hn.bias.data, start_value=0.5, increment=0.06)

    def forward(self, x, hidden):
        Wiz = self.W_iz(x)
        print(f"Wiz: {Wiz}")

        Whz = self.W_hz(hidden)
        print(f"Whz: {Whz}")

        Wir = self.W_ir(x)
        print(f"Wir: {Wir}")

        Whr = self.W_hr(hidden)
        print(f"Whr: {Whr}")

        Win = self.W_in(x)
        print(f"Win: {Win}")

        Whn = self.W_hn(hidden)
        print(f"Whn: {Whn}")

        W_iz_hz = Wiz + Whz
        print(f"Wiz + Whz: {W_iz_hz}")

        W_ir_hr = Wir + Whr
        print(f"Wir + Whr: {W_ir_hr}")

        z_t = torch.sigmoid(W_iz_hz)
        print(f"zt: {z_t}")

        r_t = torch.sigmoid(W_ir_hr)
        print(f"rt: {r_t}")

        zt_h = z_t * hidden
        print(f"zt * hidden: {zt_h}")

        rt_Whn = r_t * Whn
        print(f"rt * Whn: {rt_Whn}")
        
        W_in_rt_hn = Win +rt_Whn 
        print(f"Win + rt * Whn: {W_in_rt_hn}")

        n_t = torch.tanh(W_in_rt_hn)
        print(f"nt: {n_t}")

        _1_zt = 1 - z_t
        print(f"1-zt: {_1_zt}")

        _1_zt_nt = _1_zt * n_t
        print(f"(1-zt)*nt: {_1_zt_nt}")

        outh = _1_zt_nt + zt_h
        print(f"outh: {outh}")
        return outh


input_size = 3
hidden_size = 4
model = ReimplementedGRU(input_size, hidden_size)

onnx_Gemm_0 = torch.tensor([0.001, -0.002, 0.003], dtype=torch.float32) # input
onnx_Gemm_1 = torch.tensor([0.001, -0.002, 0.003, 0.004], dtype=torch.float32) # hidden state
y = model(onnx_Gemm_0, onnx_Gemm_1)
