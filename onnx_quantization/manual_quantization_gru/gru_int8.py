import numpy as np

# Wiz
Wiz_qw = np.array([
    [ 120, 120, 120 ],
    [ 122, 122, 122 ],
    [ 125, 125, 125 ],
    [ 127, 127, 127 ]
]).astype(np.int8)

Wiz_qb = np.array(
    [ -30873, -29021, -27168, -25316 ]
).astype(np.int32)

# Whz
Whz_qw = np.array([
    [ 0, 0, 0, 0 ],
    [ 42, 42, 42, 42 ],
    [ 85, 85, 85, 85 ],
    [ 127, 127, 127, 127 ]
]).astype(np.int8)

Whz_qb = np.array(
    [ 0, 43255, 86510, 129766 ]
).astype(np.int32)

# Wir
Wir_qw = np.array([
    [ -127, -127, -127 ],
    [ -126, -126, -126 ],
    [ -125, -125, -125 ],
    [ -124, -124, -124 ]
]).astype(np.int8)

Wir_qb = np.array(
    [ -32725, -32507, -32289, -32071 ]
).astype(np.int32)

# Whr
Whr_qw = np.array([
    [ -127, -127, -127, -127 ],
    [ -126, -126, -126, -126 ],
    [ -124, -124, -124, -124 ],
    [ -123, -123, -123, -123 ]
]).astype(np.int8)

Whr_qb = np.array(
    [ -32441, -31793, -31144, -30495 ]
).astype(np.int32)

# Win
Win_qw = np.array([
    [ 120, 120, 120 ],
    [ 122, 122, 122 ],
    [ 125, 125, 125 ],
    [ 127, 127, 127 ]
]).astype(np.int8)

Win_qb = np.array(
    [ 61746, 64833, 67920, 71008 ]
).astype(np.int32)

# Whn
Whn_qw = np.array([
    [ 123, 123, 123, 123 ],
    [ 125, 125, 125, 125 ],
    [ 126, 126, 126, 126 ],
    [ 127, 127, 127, 127 ]
]).astype(np.int8)

Whn_qb = np.array(
    [  15748, 17638, 19528, 21418 ]
).astype(np.int32)

# helper methods

def range_q(n):
    return [-2**(n-1), 2**(n-1)-1]

def range_r(W):
    return [np.min(W), np.max(W)]

def quantize(W, S, Z):
    res = np.round(W/S + Z)
    clipped_res = np.empty_like(res)
    for i, v in enumerate(res):
        if v > 127: clipped_res[i] = 127
        elif v < -128: clipped_res[i] = -128
        else: clipped_res[i] = v
    return clipped_res

def dequantize(Wq, S, Z):
    return S*(Wq.astype(np.float32)-Z)
    clipped_res = np.empty_like(res)
    for i, v in enumerate(res):
        if v > 127: clipped_res[i] = 127
        elif v < -128: clipped_res[i] = -128
        else: clipped_res[i] = v
    return clipped_res

def qgemm(qx, Sx, Zx, qw, Sw, qb, Sy, Zy):
    c = (Sw * Sx) / Sy
    a = np.int32(qx) - np.int32(Zx)
    prod = np.dot(qw.astype(np.int32), a.astype(np.int32)).astype(np.int32)
    b = c * (prod + qb)
    b = np.clip(b, -128, 127)
    b = b.astype(np.int8)
    return b + Zy

def qlinadd(qa, Sa, Za, qb, Sb, Zb, Sy, Zy):
    #a = (Sa / Sy) * (qa.astype(np.int32) - np.int32(Za)).astype(np.int32)
    #b = (Sb / Sy) * (qb.astype(np.int32) - np.int32(Zb)).astype(np.int32)
    #print((qb - Zb).astype(np.int32))
    print(" ------------------- ")
    a = (Sa / Sy) * (qa - Za).astype(np.int32)
    b = (Sb / Sy) * (qb - Zb).astype(np.int32)

    print(qa)
    print(Za)
    print(qb)
    print(Zb)
    print(Sa)
    print(Sb)
    print(Sy)
    print(Zy)

    sum_ab = a + b

    qmin, qmax = np.iinfo(np.int8).min, np.iinfo(np.int8).max
    qy = np.clip(np.round(sum_ab + Zy), qmin, qmax).astype(np.int8)

    return qy
    #return (Sa/Sy) * (qa-Za) + (Sb/Sy) * (qb-Zb) + Zy
    alpha = (Sa / Sy)
    beta = (Sb / Sy)
    Za32 = np.int32(Za)
    qa32 = qa.astype(np.int32)
    qaz32 = (qa32 - Za32).astype(np.int32)
    Zb32 = np.int32(Zb)
    qb32 = qb.astype(np.int32)
    qbz32 = (qb32 - Zb32).astype(np.int32)
    qy32 = (alpha * qaz32).astype(np.int32) + (beta * qbz32).astype(np.int32) + np.int32(Zy) 
    #qy8 = qy32.astype(np.int8)
    res = np.empty_like(qy32)
    for i, e in enumerate(qy32):
        if e < -128: res[i] = -128
        elif e >= 127: res[i] = 127
        else: res[i] = e

    return res

def qlinmul(qa, Sa, Za, qb, Sb, Zb, Sy, Zy):
    #product_ab = (qa.astype(np.int32) - Za) * (qb.astype(np.int32) - Zb)
    #scale_factor = Sa * Sb / Sy
    #scaled_product = product_ab * scale_factor
    #qmin, qmax = np.iinfo(np.int8).min, np.iinfo(np.int8).max
    #qy = np.clip(np.round(scaled_product + Zy), qmin, qmax).astype(np.int8)
    #return qy
    c = (Sa * Sb) / Sy
    #return c * (qa-Za) * (qb-Zb) + Zy

    Za32 = np.int32(Za)
    qa32 = qa.astype(np.int32)
    a_ = (qa32 - Za32).astype(np.int32)
    
    Zb32 = np.int32(Zb)
    qb32 = qb.astype(np.int32)
    b_ = (qb32 - Zb32)

    qy32 = (c * a_ * b_).astype(np.int32) + np.int32(Zy)
    qy = np.clip(np.round(qy32), -128, 127).astype(np.int8)
    return qy32

def qlinsigmoid(qx, Sx, Zx, Sy, Zy):
    x = dequantize(qx, Sx, Zx)
    y = 1 / (1 + np.exp(-x))
    qy = quantize(y, Sy, Zy)
    return qy

def tanh(x):
    e_p = np.exp(x)
    e_n = np.exp(-x)
    return (e_p - e_n) / (e_p + e_n)

def subf32(a, b):
    return a - b

def gru_int8():
    onnx_Gemm_0 = np.array([0.001, -0.002, 0.003]).astype(np.float32) # input
    onnx_Gemm_1 = np.array([0.001, -0.002, 0.003, 0.004]).astype(np.float32) # hidden state

    # onnx gemm 0 quantize linear
    onnx_Gemm_0_quantized = quantize(W=onnx_Gemm_0, S=0.0038807899691164494, Z=-128)
    #print(dequantize(Wq=onnx_Gemm_0_quantized, S=0.0038807899691164494, Z=-128))

    # onnx gemm 1 quantize linear
    onnx_Gemm_1_quantized = quantize(W=onnx_Gemm_1, S=0.003914752043783665, Z=-128)
    # print(dequantize(Wq=onnx_Gemm_1_quantized, S=0.003914752043783665, Z=-128))

    # Wiz gemm quant
    Wiz_Gemm_output_0_quantized = qgemm(
        qx=onnx_Gemm_0_quantized, 
        Sx=0.0038807899691164494, 
        Zx=-128, 
        qw=Wiz_qw,
        Sw=0.0041732280515134335, 
        qb=Wiz_qb, 
        Sy=0.004839869681745768, 
        Zy=-69)
    print(Wiz_Gemm_output_0_quantized)
    print("Wiz: ", dequantize(Wq=Wiz_Gemm_output_0_quantized, S=0.004839869681745768, Z=-69))

    # Whz gemm quant
    Whz_Gemm_output_0_quantized = qgemm(
        qx=onnx_Gemm_1_quantized, 
        Sx=0.003914752043783665, 
        Zx=-128, 
        qw=Whz_qw,
        Sw=0.000236220468650572, 
        qb=Whz_qb, 
        Sy=0.000859871506690979, 
        Zy=-128)
    #print(Whz_Gemm_output_0_quantized)
    print("Whz: ", dequantize(Wq=Whz_Gemm_output_0_quantized, S=0.000859871506690979, Z=-128))
    
    # Wir gemm quant
    Wir_Gemm_output_0_quantized = qgemm(
        qx=onnx_Gemm_0_quantized, 
        Sx=0.0038807899691164494, 
        Zx=-128, 
        qw=Wir_qw,
        Sw=0.011811023578047752, 
        qb=Wir_qb, 
        Sy=0.020957935601472855, 
        Zy=127)
    #print(Wir_Gemm_output_0_quantized)
    print("Wir: ", dequantize(Wq=Wir_Gemm_output_0_quantized, S=0.020957935601472855, Z=127))

    # Whr gemm quant
    Whr_Gemm_output_0_quantized = qgemm(
        qx=onnx_Gemm_1_quantized, 
        Sx=0.003914752043783665, 
        Zx=-128, 
        qw=Whr_qw,
        Sw=0.007874015718698502, 
        qb=Whr_qb, 
        Sy=0.01689767837524414, 
        Zy=127)
    print("Whr: ", dequantize(Wq=Whr_Gemm_output_0_quantized, S=0.01689767837524414, Z=127))
    
    # Win gemm quant
    Win_Gemm_output_0_quantized = qgemm(
        qx=onnx_Gemm_0_quantized, 
        Sx=0.0038807899691164494, 
        Zx=-128, 
        qw=Win_qw,
        Sw=0.0041732280515134335, 
        qb=Win_qb, 
        Sy=0.009836508892476559, 
        Zy=-128)
    print("Win: ", dequantize(Wq=Win_Gemm_output_0_quantized, S=0.009836508892476559, Z=-128))
    
    # Whn gemm quant
    Whn_Gemm_output_0_quantized = qgemm(
        qx=onnx_Gemm_1_quantized, 
        Sx=0.003914752043783665, 
        Zx=-128, 
        qw=Whn_qw,
        Sw=0.008110236376523972, 
        qb=Whn_qb, 
        Sy=0.016032058745622635, 
        Zy=-128)
    #print(Whn_Gemm_output_0_quantized)
    print("Whn: ", dequantize(Wq=Whn_Gemm_output_0_quantized, S=0.016032058745622635, Z=-128))
    
    # add 1 quant
    Add_1_output_0_quantized = qlinadd(
        qa=Wiz_Gemm_output_0_quantized,
        Sa=0.004839869681745768,
        Za=-69,
        qb=Whz_Gemm_output_0_quantized,
        Sb=0.000859871506690979,
        Zb=-128,
        Sy=0.0056219203397631645,
        Zy=-77
    )
    print("Wiz + Whz: ", dequantize(Wq=Add_1_output_0_quantized, S=0.0056219203397631645, Z=-77))

    # add
    Add_output_0_quantized = qlinadd(
        qa=Wir_Gemm_output_0_quantized,
        Sa=0.020957935601472855,
        Za=127,
        qb=Whr_Gemm_output_0_quantized,
        Sb=0.01689767837524414,
        Zb=127,
        Sy=0.03715033829212189,
        Zy=127
    )
    print("Wir + Whr: ", dequantize(Wq=Add_output_0_quantized, S=0.03715033829212189, Z=127))

    # sigmoid 1 quant
    Sigmoid_1_output_quantized = qlinsigmoid(
        qx=Add_1_output_0_quantized,
        Sx=0.0056219203397631645,
        Zx=-77,
        Sy=0.002976849442347884,
        Zy=-128
    )
    print("zt: ", dequantize(Wq=Sigmoid_1_output_quantized, S=0.002976849442347884, Z=-128))

    # sigmoid quant
    Sigmoid_output_quantized = qlinsigmoid(
        qx=Add_output_0_quantized,
        Sx=0.03715033829212189,
        Zx=127,
        Sy=0.000033313972380710766,
        Zy=-128
    )
    print("rt: ", dequantize(Wq=Sigmoid_output_quantized, S=0.000033313972380710766, Z=-128))

    Mul_2_output_0_quantized = qlinmul(
        qa=Sigmoid_1_output_quantized,
        Sa=0.002976849442347884,
        Za=-128,
        qb=onnx_Gemm_1_quantized,
        Sb=0.003914752043783665,
        Zb=-128,
        Sy=0.0028020692989230156,
        Zy=-128
    )
    print("zt * hidden: ", dequantize(Wq=Mul_2_output_0_quantized, S=0.0028020692989230156, Z=-128))

    Mul_output_0_quantized = qlinmul(
        qa=Sigmoid_output_quantized,
        Sa=0.000033313972380710766,
        Za=-128,
        qb=Whn_Gemm_output_0_quantized,
        Sb=0.016032058745622635,
        Zb=-128,
        Sy=0.00007471089338650927,
        Zy=-128
    )
    print("rt * Whn: ", dequantize(Wq=Mul_output_0_quantized, S=0.00007471089338650927, Z=-128))

    Add_2_output_0_quantized = qlinadd(
        qa=Win_Gemm_output_0_quantized,
        Sa=0.009836508892476559,
        Za=-128,
        qb=Mul_output_0_quantized,
        Sb=0.00007471089338650927,
        Zb=-128,
        Sy=0.00983863603323698,
        Zy=-128
    )
    print("Win + rt * Whn:", dequantize(Wq=Add_2_output_0_quantized, S=0.00983863603323698, Z=-128))

    Add_2_output_0 = dequantize(Wq=Add_2_output_0_quantized, S=0.00983863603323698, Z=-128)
    Tanh_output_0 = tanh(Add_2_output_0)
    Tanh_output_0_quantized = quantize(W=Tanh_output_0, S=0.0038699908182024956, Z=-128)
    print("nt: ", dequantize(Wq=Tanh_output_0_quantized, S=0.0038699908182024956, Z=-128))
    
    Sigmoid_1_output_0 = dequantize(Wq=Sigmoid_1_output_quantized, S=0.002976849442347884, Z=-128)
    Sub_output_0 = subf32(1, Sigmoid_1_output_0)
    Sub_output_0_quantized = quantize(W=Sub_output_0, S=0.002239143243059516, Z=-128)
    print("1-zt: ", dequantize(Wq=Sub_output_0_quantized, S=0.002239143243059516, Z=-128))

    Mul_1_output_0_quantized = qlinmul(
        qa=Sub_output_0_quantized,
        Sa=0.002239143243059516,
        Za=-128,
        qb=Tanh_output_0_quantized,
        Sb=0.0038699908182024956,
        Zb=-128,
        Sy=0.0018845913000404835,
        Zy=-128
    )
    print("(1-zt)*nt: ", dequantize(Wq=Mul_1_output_0_quantized, S=0.0018845913000404835, Z=-128))

    _31_quantized = qlinadd(
        qa=Mul_1_output_0_quantized,
        Sa=0.0018845913000404835,
        Za=-128,
        qb=Mul_2_output_0_quantized,
        Sb=0.0028020692989230156,
        Zb=-128,
        Sy=0.003882633987814188,
        Zy=-128
    )
    print(_31_quantized)
    print("outh: ", dequantize(Wq=_31_quantized, S=0.003882633987814188, Z=-128))





def test_qlinadd():
    a = np.array([0.015, 0.025, 0.035]).astype(np.float32)
    b = np.array([0.045, 0.055, 0.065]).astype(np.float32)
    qa = quantize(W=a, S=0.001, Z=0)
    qb = quantize(W=b, S=0.002, Z=0)
    qy = qlinadd(qa=qa, Sa=0.001, Za=0, qb=qb, Sb=0.002, Zb=0, Sy=0.003, Zy=0)
    print(qy)
    y = dequantize(qy, 0.003, 0)
    print(y)

def test_qlinmul():
    a = np.array([0.15, 0.25, 0.35]).astype(np.float32)
    b = np.array([0.45, 0.55, 0.65]).astype(np.float32)
    qa = quantize(W=a, S=0.001, Z=0)
    qb = quantize(W=b, S=0.002, Z=0)
    qy = qlinmul(qa=qa, Sa=0.001, Za=0, qb=qb, Sb=0.002, Zb=0, Sy=0.003, Zy=0)
    print(qy)
    y = dequantize(qy, 0.003, 0)
    print(y)

def test_qlinsigmoid():
    x = np.array([0.085, 0.095, 0.09]).astype(np.float32)
    y = 1 / (1 + np.exp(-x))
    print(f"sigmoid float32: {y}")

    qx = quantize(x, 0.5799347069114447, 64)
    print(f"qx: {qx}")
    qz = qlinsigmoid(qx, 0.5799347069114447, 64, 0.2949759131297469, 0)
    z = dequantize(qz, 0.2949759131297469, 0)
    print(f"sigmoid after quantization: {z}")


gru_int8()
#test_qlinadd()
#test_qlinmul()
#test_qlinsigmoid()