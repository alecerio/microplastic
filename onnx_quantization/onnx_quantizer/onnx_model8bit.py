import torch
import torch.nn as nn
import torch.onnx
import numpy as np
import torch
import onnx
from onnxruntime import quantization
from onnxruntime.quantization import QuantFormat
from onnxruntime.quantization import QuantType

##################################################################
#                   START MODEL DEFINITION                       #
##################################################################

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
        r_t = torch.sigmoid(self.W_ir(x) + self.W_hr(hidden))
        z_t = torch.sigmoid(self.W_iz(x) + self.W_hz(hidden))
        n_t = torch.tanh(self.W_in(x) + r_t * self.W_hn(hidden))

        hidden = (1 - z_t) * n_t + z_t * hidden
        return hidden

##################################################################
#                     END MODEL DEFINITION                       #
##################################################################

class QuantizationDataReader(quantization.CalibrationDataReader):
    def __init__(self, input_name, sizes, num_calibration_samples=100):
        self.input_name = input_name
        self.num_calibration_samples = num_calibration_samples
        self.current_index = 0
        self.sizes = sizes
    def get_next(self):
        if self.current_index < self.num_calibration_samples:
            result = {}
            for index, input_name in enumerate(input_names):
                input_data = np.random.rand(1, self.sizes[index]).astype(np.float32)
                result[input_name] = input_data
            self.current_index += 1
            return result
        else:
            return None

def _gen_dummy_data(input_sizes):
    dummy_inputs = []
    for input_size in input_sizes:
        dummy = torch.randn(1, input_size, requires_grad=True)
        dummy_inputs.append(dummy)
    return dummy_inputs

def _gen_f32(model, dummy_inputs, input_names, onnx_name, output_names):
    print("Create float32 model ...")
    model_fp32_path = onnx_name
    torch.onnx.export(model,                                         # model
                      tuple(dummy_inputs),                                         # model input
                      model_fp32_path,                                  # path
                      export_params=True,                               # store the trained parameter weights inside the model file
                      opset_version=14,                                 # the ONNX version to export the model to
                      do_constant_folding=True,                         # constant folding for optimization
                      input_names = input_names,                          # input names
                      output_names = output_names,                        # output names
    )

def _gen_staticq(input_names, input_sizes, model_prep_path, q_onnx_name, op_types_to_quantize):
    qdr = QuantizationDataReader(input_names, input_sizes)
    q_static_opts = {"ActivationSymmetric":False,
                     "WeightSymmetric":True}

    quant_format = QuantFormat.QOperator
    quantized_model = quantization.quantize_static(model_input=model_prep_path,
                                                   model_output=q_onnx_name,
                                                   calibration_data_reader=qdr,        
                                                   weight_type=QuantType.QInt8,
                                                   activation_type=QuantType.QInt8,
                                                   quant_format=quant_format,
                                                   op_types_to_quantize=op_types_to_quantize,)

    print("Quantized model saved to:", q_onnx_name)

def run_onnx_quantizer(model, input_names, input_sizes, output_names, op_types_to_quantize):
    dummy_inputs = _gen_dummy_data(input_sizes)

    model_fp32_path = 'model_f32.onnx'
    _gen_f32(model, dummy_inputs, input_names, model_fp32_path, output_names)

    # load f32 model
    model_onnx = onnx.load(model_fp32_path)
    onnx.checker.check_model(model_onnx)

    # Prepare the model for quantization
    model_prep_path = 'model_f32_prep.onnx'
    quantization.shape_inference.quant_pre_process(model_fp32_path, model_prep_path, skip_symbolic_shape=False)

    _gen_staticq(input_names, input_sizes, model_prep_path, 'model_i8.onnx', op_types_to_quantize)


##################################################################
#                        RUN QUANTIZER                           #
##################################################################

model = SimpleNN(3, 4)
input_sizes = [3, 4]
input_names = ['onnx::Gemm_0', 'onnx::Gemm_1']
output_names = ['31']
op_types_to_quantize = ['Gemm', 'Add', 'Sigmoid', 'Mul', 'Sub', 'Tanh']
run_onnx_quantizer(model, input_names, input_sizes, output_names, op_types_to_quantize)