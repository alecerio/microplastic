import onnxruntime as ort
import numpy as np
import time

num_iterations = 100000

########################################################
#                PUT I/O VARIABLES HERE                #
########################################################

model_path = 'qmlp.onnx'
dummy_input = None
session = None
exp_collector = []

########################################################

def setup():
    session = ort.InferenceSession(model_path)
    input_shape = (1, 41)
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

def run():
    input_name = session.get_inputs()[0].name
    for _ in range(num_iterations):
        start_time = time.time()
        session.run(None, {input_name: dummy_input})
        end_time = time.time()
        cpu_inference_time = end_time - start_time
        exp_collector.append(cpu_inference_time)

def postprocessing():
    average = sum(exp_collector) / num_iterations
    maxval = max(exp_collector)
    print("average: {average}")
    print("maximum: {maxval}")