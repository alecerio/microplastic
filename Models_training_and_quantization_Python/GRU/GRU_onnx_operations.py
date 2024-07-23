import numpy as np
import torch
import onnx
import onnxruntime as rt
from onnxruntime import quantization
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

# Define a data reader class for ONNX quantization
class QuantizationDataReader(CalibrationDataReader):
    def __init__(self, input_name, sizes, num_calibration_samples=100):
        self.input_name = input_name
        self.num_calibration_samples = num_calibration_samples
        self.current_index = 0
        self.sizes = sizes
    
    def get_next(self):
        # Generate random data samples for calibration
        if self.current_index < self.num_calibration_samples:
            result = {}
            for index, input_name in enumerate(self.input_name):
                input_data = np.random.rand(1, self.sizes[index]).astype(np.float32)
                result[input_name] = input_data
            self.current_index += 1
            return result
        else:
            return None

# Extract input vector and hidden state from the model for a given index
def extract_vector_and_hidden_state(model, X_train, index=0):
    model.eval()
    with torch.no_grad():
        input_vector = torch.tensor(X_train[index], dtype=torch.float32).to(model.device)
        hidden_state = torch.zeros(1, model.hidden_size).to(model.device)
        hidden_state = model(input_vector.unsqueeze(0), hidden_state)
        return input_vector.cpu().numpy(), hidden_state.cpu().numpy()

# Generate real data inputs for the model
def _gen_real_data(model, X_train, index):
    inputs = []
    input_vector, hidden_state = extract_vector_and_hidden_state(model, X_train, index)
    inputs.append(torch.tensor(input_vector, dtype=torch.float32, requires_grad=True).unsqueeze(0))
    hidden_tensor = torch.tensor(hidden_state, dtype=torch.float32, requires_grad=True)
    inputs.append(hidden_tensor)
    return inputs

# Generate and save the float32 ONNX model
def _gen_f32(model, dummy_inputs, input_names, onnx_name, output_names):
    print("Create float32 model ...")
    model_fp32_path = onnx_name
    torch.onnx.export(model,
                      tuple(dummy_inputs),
                      model_fp32_path,
                      export_params=True,
                      opset_version=14,
                      do_constant_folding=True,
                      input_names=input_names,
                      output_names=output_names)

# Generate and save the quantized ONNX model
def _gen_staticq(input_names, input_sizes, model_prep_path, q_onnx_name, op_types_to_quantize):
    qdr = QuantizationDataReader(input_names, input_sizes)
    quant_format = QuantFormat.QOperator
    quantized_model = quantize_static(model_input=model_prep_path,
                                      model_output=q_onnx_name,
                                      calibration_data_reader=qdr,
                                      weight_type=QuantType.QInt8,
                                      activation_type=QuantType.QInt8,
                                      quant_format=quant_format,
                                      op_types_to_quantize=op_types_to_quantize)
    print("Quantized model saved to:", q_onnx_name)

# Main function to run ONNX quantization
def run_onnx_quantizer(model, X_train, input_names, input_sizes, output_names, op_types_to_quantize):
    index = np.random.randint(0, len(X_train))
    dummy_inputs = _gen_real_data(model, X_train, index)
    model_fp32_path = 'gru_model_f32.onnx'
    _gen_f32(model, dummy_inputs, input_names, model_fp32_path, output_names)

    model_onnx = onnx.load(model_fp32_path)
    onnx.checker.check_model(model_onnx)
    model_prep_path = 'gru_model_f32_prep.onnx'
    quantization.shape_inference.quant_pre_process(model_fp32_path, model_prep_path, skip_symbolic_shape=False)

    _gen_staticq(input_names, input_sizes, model_prep_path, 'gru_model_i8.onnx', op_types_to_quantize)
    
    return 'gru_model_i8.onnx'

# Test the quantized ONNX model
def test_onnx_model(onnx_model_file, X_test, hidden_size, i):
    sess = rt.InferenceSession(onnx_model_file)
    input_name_0 = sess.get_inputs()[0].name
    input_name_1 = sess.get_inputs()[1].name
    
    test_input_0 = np.array([X_test[i]]).astype(np.float32)
    test_input_1 = np.zeros((1, hidden_size)).astype(np.float32)
    
    pred = sess.run(None, {input_name_0: test_input_0, input_name_1: test_input_1})[0]
    pred_class = np.argmax(pred, axis=1)
    return pred_class
