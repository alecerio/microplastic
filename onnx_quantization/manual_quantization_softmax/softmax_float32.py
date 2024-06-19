import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) 
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

input_data = np.array(
    [0.0, -0.01, 0.02, -0.03]
).astype(np.float32)

y = softmax(input_data)
print(y)