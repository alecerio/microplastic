import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) 
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def quantize(W, S, Z):
    return np.round(W/S + Z)

def dequantize(Wq, S, Z):
    return S*(Wq-Z)

def quantize_input(input):
    y_scale = 0.003917609341442585
    y_zero_point = 0
    return quantize(input, y_scale, y_zero_point)

def dequantize_output(qy):
    y_scale = 0.003917609341442585
    y_zero_point = 0
    return dequantize(qy, y_scale, y_zero_point)

def quant_softmax_lut(minimum, maximum, nbins):
    x = np.zeros((nbins,), dtype=np.float32)
    y = np.zeros((nbins,), dtype=np.float32)
    for i in range(0, nbins):
        val = minimum+i*(maximum-minimum)/nbins
        x[i] = val
        y[i] = np.exp(val)
    return [x, y]

def access_lut(x_axis, y_axis, x):
    if x < x_axis[0]:
        return y_axis[0]
    for i in range(0, len(x_axis)-1):
        if x >= x_axis[i] and x < x_axis[i+1]:
            return np.interp(x, [x_axis[i], x_axis[i+1]], [y_axis[i], y_axis[i+1]])
    return y_axis[len(x_axis)-1]


input_data = np.array(
    [0.0, 1.0, 2.0, 3.0, 4.0]
).astype(np.float32)

qx = quantize_input(input_data)

[lutx, luty] = quant_softmax_lut(-10, 10, 256)

x = dequantize_output(qx)

maxx = max(x)
exps = np.zeros((len(x),), dtype=np.float32)
for i in range(0, len(x)):
    exps[i] = access_lut(lutx, luty, x[i])
sum_exps = sum(exps)

sm = np.zeros((len(x),), dtype=np.float32)
for i in range(0, len(x)):
    sm[i] = exps[i] / sum_exps

qsm = quantize_input(sm)

y = dequantize_output(qsm)
print(y)