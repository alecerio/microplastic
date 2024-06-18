Certo, ecco il contenuto completo del file README in formato markdown, pronto per essere utilizzato su GitHub:

```markdown
# Neural Architecture Search and Model Training

This repository contains all the necessary Python code for performing Neural Architecture Search (NAS), training Multilayer Perceptron (MLP) and Gated Recurrent Unit (GRU) models, and quantizing these models to INT8 using ONNX. Below, you'll find a detailed explanation of each component and instructions on how to use the scripts provided.

## Repository Structure

- `nas/`: Contains scripts and modules necessary for performing Neural Architecture Search.
- `training/`: Includes Python scripts for training MLP and GRU models.
- `quantization/`: Scripts for quantizing trained models to the INT8 format using ONNX.

## Neural Architecture Search (NAS)

Neural Architecture Search (NAS) is a process by which we automate the design of artificial neural networks. It is particularly useful for optimizing model architectures in a more efficient way than manual design.

### Usage

Navigate to the `nas/` directory:

```bash
cd nas
```

Run the main script to start the architecture search:

```bash
python nas_main.py
```

## Training Models

This section includes scripts for training two types of neural networks: Multilayer Perceptrons (MLP) and Gated Recurrent Units (GRU).

### MLP Training

To train an MLP model, use the following command:

```bash
python train_mlp.py
```

### GRU Training

For training a GRU model, execute:

```bash
python train_gru.py
```

## Model Quantization

Quantization is the process of reducing the precision of the numbers used to represent a model's parameters, which typically leads to reductions in model size and increases in inferencing speed, with a minimal decrease in accuracy.

### Quantizing to INT8

Navigate to the `quantization/` directory:

```bash
cd quantization
```

Run the quantization script to convert models to INT8 format using ONNX:

```bash
python quantize_to_int8.py
```

## Requirements

This project requires Python 3.8 or higher, along with the following libraries:
- PyTorch
- ONNX
- ONNXRuntime

You can install the necessary dependencies via pip:

```bash
pip install -r requirements.txt
```

## Contributing

Contributions to this project are welcome. Please fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

