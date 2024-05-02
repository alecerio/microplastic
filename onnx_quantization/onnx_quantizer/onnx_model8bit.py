import torch
import torch.nn as nn
import torch.onnx
import numpy as np
import torch
import onnx
from onnxruntime import quantization
from onnxruntime.quantization import QuantFormat
from onnxruntime.quantization import QuantType


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 3)  # Input layer to hidden layer

    def forward(self, x):
        x = self.fc1(x)
        return x

# Create a dummy input tensor
dummy_input = torch.randn(1, 4, requires_grad=True)
model = SimpleNN()

# export fp32 model to onnx
model_fp32_path = "simple_nn.onnx"

torch.onnx.export(model,                                         # model
                  dummy_input,                                         # model input
                  model_fp32_path,                                  # path
                  export_params=True,                               # store the trained parameter weights inside the model file
                  opset_version=14,                                 # the ONNX version to export the model to
                  do_constant_folding=True,                         # constant folding for optimization
                  input_names = ['input'],                          # input names
                  output_names = ['output'],                        # output names
)


# Load your ONNX model
model_onnx = onnx.load(model_fp32_path)
onnx.checker.check_model(model_onnx)

# Prepare the model for quantization
model_prep_path = 'simple_nn_prep.onnx'
quantization.shape_inference.quant_pre_process(model_fp32_path, model_prep_path, skip_symbolic_shape=False)


class QuantizationDataReader(quantization.CalibrationDataReader):
    def __init__(self, input_name, num_calibration_samples=100):
        self.input_name = input_name
        self.num_calibration_samples = num_calibration_samples
        self.current_index = 0
    def get_next(self):
        if self.current_index < self.num_calibration_samples:
            input_data = np.random.rand(1, 4).astype(np.float32)  # Adjust shape if needed
            self.current_index += 1
            return {input_name: input_data}  # Use the correct input name from the model
        else:
            return None


input_name = model_onnx.graph.input[0].name
print(input_name)
qdr = QuantizationDataReader(input_name)
# these are options that can be changed depending on the hardware
q_static_opts = {"ActivationSymmetric":False,
                 "WeightSymmetric":True}
op_types_to_quantize = ['Gemm', 'Relu']

# this Quantize Dequantize Quantize "QDQ" format is a way to simulate the quantization process
# the quantize operator can be selected with the quant_format parameter QuantFormat.QOperator or QuantFormat.QDQ
quant_format = QuantFormat.QOperator
model_int8_path = 'static_quantized_operator_model.onnx'
quantized_model = quantization.quantize_static(model_input=model_prep_path,
                                               model_output=model_int8_path,
                                               calibration_data_reader=qdr,        
                                               weight_type=QuantType.QInt8,
                                               activation_type=QuantType.QUInt8,
                                               quant_format=quant_format,
                                               op_types_to_quantize=op_types_to_quantize,)

print("Quantized model saved to:", model_int8_path)
qdr = QuantizationDataReader(input_name) # Reset the data reader
model_int8_path = 'static_quantized_QDQ_model.onnx'
quantized_model = quantization.quantize_static(model_input=model_prep_path,
                                               model_output=model_int8_path,
                                               calibration_data_reader=qdr,        
                                               weight_type=QuantType.QInt8,
                                               activation_type=QuantType.QUInt8,
                                               op_types_to_quantize=op_types_to_quantize,)
