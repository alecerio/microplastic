# Neural Architecture Search and Model Training

This repository contains all the necessary Python code for performing Neural Architecture Search (NAS), training Multilayer Perceptron (MLP)
and Gated Recurrent Unit (GRU) models, and quantizing these models to INT8 using ONNX. Below, you'll find a detailed explanation of each
all the components and instructions on how to use the scripts provided.


## Repository Structure

 `MLP/`: Contains scripts for training MLP models and performing ONNX operations.
  - `MLP_main.py`: Main script for training MLP models, converting to ONNX, and quantizing.
  - `MLP_model.py`: Defines the MLP model architecture and training functions.
  - `MLP_onnx_operations.py`: Contains functions for converting PyTorch models to ONNX format and quantizing the ONNX models.
  - `MLP_utils.py`: Utility functions for creating datasets and calculating parameters.

- `GRU/`: Contains scripts for training GRU models and performing ONNX operations.
  - `GRU_main.py`: Main script for training GRU models, converting to ONNX, and quantizing.
  - `GRU_model.py`: Defines the GRU model architecture and training functions.
  - `GRU_onnx_operations.py`: Contains functions for converting PyTorch models to ONNX format and quantizing the ONNX models.
  - `GRU_utils.py`: Utility functions for creating datasets and calculating parameters.


## Training Models

This section includes scripts for training two types of neural networks: Multilayer Perceptrons (MLP) and Gated Recurrent Units (GRU).

### MLP Training

To train, quantize, and test an MLP model, navigate to the `MLP` directory and use the following command:

```bash
cd MLP
python MLP_main.py
```

### GRU Training

To train, quantize, and test an MLP model, navigate to the `GRU` directory and use the following command:

```bash
cd GRU
python GRU_main.py
```


## Requirements

This project requires Python 3.8 or higher, along with the following libraries:
- PyTorch
- ONNX
- ONNXRuntime
- NumPy
- SciPy
- scikit-learn

You can install the necessary dependencies via pip:

```bash
pip install -r requirements.txt
```
