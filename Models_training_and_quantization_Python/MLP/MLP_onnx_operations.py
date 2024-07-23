import numpy as np
import torch
import onnx
import onnxruntime as rt
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat

# Class to read calibration data for quantization
class MyCalibrationDataReader(CalibrationDataReader):
    def __init__(self, X_test, n_samples=100):
        self.data = X_test[:n_samples]  # Use only a subset of test data for calibration
        self.data_index = 0  # Index to keep track of current position in the data

    def get_next(self):
        # Provides the next input data for calibration
        if self.data_index < len(self.data):
            input_data = {'input': np.expand_dims(self.data[self.data_index], axis=0).astype(np.float32)}
            self.data_index += 1
            return input_data
        else:
            return None  # No more data available

# Function to convert a PyTorch model to ONNX format
def convert_to_onnx(model, input_shape, file_name='mlp_model.onnx'):
    model = model.to('cpu')  # Move the model to CPU
    dummy_input = torch.randn(input_shape).to('cpu')  # Create a dummy input for conversion
    torch.onnx.export(
        model, dummy_input, file_name,
        export_params=True,  # Export the model parameters
        opset_version=14,  # ONNX opset version
        do_constant_folding=True,  # Optimize constants during export
        input_names=['input'],  # Name of the input node
        output_names=['output']  # Name of the output node
    )
    return file_name

# Function to quantize an ONNX model
def quantize_model(onnx_model_file, X_test):
    quantized_model_file = 'mlp_model_quant.onnx'
    calibrator = MyCalibrationDataReader(X_test)  # Create the calibration data reader
    quantize_static(onnx_model_file, quantized_model_file, calibrator, quant_format=QuantFormat.QOperator)
    return quantized_model_file

# Function to check for the presence of quantization operators in an ONNX model
def check_quantization(onnx_model_file):
    model = onnx.load(onnx_model_file)
    operators = {node.op_type for node in model.graph.node}
    quantization_operators = {'QuantizeLinear', 'DequantizeLinear'}
    has_quantization = quantization_operators & operators

    if has_quantization:
        print("The ONNX model contains quantization operators:", has_quantization)
    else:
        print("The ONNX model does not contain quantization operators.")

# Function to test an ONNX model
def test_onnx_model(onnx_model_file, X_test, y_test, i):
    sess = rt.InferenceSession(onnx_model_file)  # Create an inference session
    input_name = sess.get_inputs()[0].name  # Get the input node name

    test_input = np.array([X_test[i]]).astype(np.float32)  # Prepare the input data
    pred = sess.run(None, {input_name: test_input})[0]  # Run the inference
    pred_class = np.argmax(pred, axis=1)  # Get the predicted class
    return pred_class
