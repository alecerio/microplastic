import numpy as np

def range_q(n):
    return [-2**(n-1), 2**(n-1)-1]

def range_r(W):
    return [np.min(W), np.max(W)]

def quantize(W, S, Z):
    return np.round(W/S + Z)

def dequantize(Wq, S, Z):
    return S*(Wq-Z)


def quantize_input(input):
    y_scale = 0.003917609341442585
    y_zero_point = 0
    return quantize(input, y_scale, y_zero_point)

def compute_qgemm(qx):
    Sx = 0.003917609341442585
    Zx = 0
    qw = np.array([
        [ 122, -28, 66, -87 ],
        [ 0, 76, -50, 119 ],
        [ 0, -127, 94, -85 ]
    ]).astype(np.int8)
    Sw = 0.0033770152367651463
    Zw = 0

    qb = np.array(
        [ -35547, 23187, 10704 ]
    ).astype(np.int32)

    Sy = 0.006249960046261549
    Zy = 117

    c = (Sw * Sx) / Sy
    a = qx - Zx
    a = np.transpose(a)
    b = c * (np.dot(qw, a) + qb)
    b = b.astype(np.int8)
    return b + Zy

def dequantize_output(qy):
    Sy = 0.006249960046261549
    Zy = 117
    return dequantize(qy, Sy, Zy)



# input
input = np.array(
    [0.0, -0.01, 0.02, -0.03]
).astype(np.float32)
qx = quantize_input(input)

# qgemm
qy = compute_qgemm(qx)

output = dequantize_output(qy)
print(output)


