import onnxruntime as ort
import numpy as np

########################################################
#                PUT I/O VARIABLES HERE                #
########################################################

model_path = 'qgru_model_i8.onnx'
exp_collector = []

########################################################

def run():
    dummy_input = np.ones((1, 41), dtype=np.float32)
    dummy_hidden = np.ones((1, 8), dtype=np.float32)
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    hidden_name = session.get_inputs()[1].name
    output = session.run(None, {input_name: dummy_input, hidden_name: dummy_hidden})
    print(output)

if __name__ == "__main__":
    run()