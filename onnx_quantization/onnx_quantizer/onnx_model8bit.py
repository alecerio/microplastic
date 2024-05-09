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
    def __init__(self, input_size, hidden_size):
        super(SimpleNN, self).__init__()
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

# Create a dummy input tensor
dummy_input = torch.randn(1, 3, requires_grad=True)
hidden = torch.zeros(1, 4)
model = SimpleNN(3, 4)

#torch.save(model.state_dict(), 'model.pth')
#model.load_state_dict(torch.load('model.pth'))

# export fp32 model to onnx
print("Create float32 model ...")
model_fp32_path = "simple_nn.onnx"
torch.onnx.export(model,                                         # model
                  (dummy_input, hidden),                                         # model input
                  model_fp32_path,                                  # path
                  export_params=True,                               # store the trained parameter weights inside the model file
                  opset_version=14,                                 # the ONNX version to export the model to
                  do_constant_folding=True,                         # constant folding for optimization
                  input_names = ['onnx::Gemm_0', 'onnx::Gemm_1'],                          # input names
                  output_names = ['31'],                        # output names
)


# Load your ONNX model
model_onnx = onnx.load(model_fp32_path)
onnx.checker.check_model(model_onnx)

# Prepare the model for quantization
model_prep_path = 'simple_nn_prep.onnx'
quantization.shape_inference.quant_pre_process(model_fp32_path, model_prep_path, skip_symbolic_shape=False)


class QuantizationDataReader(quantization.CalibrationDataReader):
    def __init__(self, input_name, sizes, num_calibration_samples=100):
        self.input_name = input_name
        self.num_calibration_samples = num_calibration_samples
        self.current_index = 0
        self.sizes = sizes
    def get_next(self):
        if self.current_index < self.num_calibration_samples:
            input_data_1 = np.random.rand(1, self.sizes[0]).astype(np.float32)  # Adjust shape if needed
            input_data_2 = np.random.rand(1, self.sizes[1]).astype(np.float32)  # Adjust shape if needed
            self.current_index += 1
            #return {self.input_name[self.current_index-1]: input_data}  # Use the correct input name from the model
            return {self.input_name[0]: input_data_1, self.input_name[1]: input_data_2}
        else:
            return None


input_names = [input.name for input in model_onnx.graph.input]
sizes = [3, 4]
#input_name = model_onnx.graph.input[0].name
#input_name_hidden = model_onnx.graph.input[1].name
#input_names = [input_name, input_name_hidden]
print(input_names)
qdr = QuantizationDataReader(input_names, sizes)
# these are options that can be changed depending on the hardware
q_static_opts = {"ActivationSymmetric":False,
                 "WeightSymmetric":True}
op_types_to_quantize = ['Gemm', 'Add', 'Sigmoid', 'Mul', 'Sub', 'Tanh']

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
qdr = QuantizationDataReader(input_names, sizes) # Reset the data reader
model_int8_path = 'static_quantized_QDQ_model.onnx'
quantized_model = quantization.quantize_static(model_input=model_prep_path,
                                               model_output=model_int8_path,
                                               calibration_data_reader=qdr,        
                                               weight_type=QuantType.QInt8,
                                               activation_type=QuantType.QUInt8,
                                               op_types_to_quantize=op_types_to_quantize,)
