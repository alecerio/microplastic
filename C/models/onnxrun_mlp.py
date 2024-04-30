import torch
import torch.nn as nn
import onnxruntime
import os
import onnx

# define input
input_data = torch.ones(1, 41) 

# create session
script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)
session = onnxruntime.InferenceSession(script_directory + "/MLP.onnx")

# inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
result = session.run([output_name], {input_name: input_data.numpy()})

# print the result
print("Input:")
print(input_data)
print()
print("Output:")
print(result[0])
