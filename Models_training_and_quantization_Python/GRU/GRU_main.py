import os
import numpy as np
import torch
import random
import warnings
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import scipy.stats as stats
from GRU_model import train_model, save_pytorch_model, predict
from GRU_onnx_operations import run_onnx_quantizer, test_onnx_model
from GRU_utils import create_dataset, calcolaParametri

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

# Set seed for reproducibility
seed = 42
np.random.seed(seed)  # Set seed for numpy random number generation
torch.manual_seed(seed)  # Set seed for PyTorch random number generation

# Device configuration: Use GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Define time points and Gaussian parameters for the signal
    t = np.arange(-20, 21)
    sigma_gau = 5
    g = np.exp(-t ** 2 / sigma_gau ** 2)  # Gaussian function
    g = np.resize(g, (41, 1))  # Reshape for compatibility

    # Define range and parameters for signal creation
    amin = 0.6
    amax = 1.4
    R = np.arange(1, 101)  # Range for testing
    M = 10000  # Number of samples

    sigma_noise = 0.5
    P1 = 0.05
    
    Sigma = (sigma_noise ** 2) * np.diag(np.ones(g.size))  # Covariance matrix for noise

    # Create directory for model saving and set it as current working directory
    original_path = os.getcwd()
    path = os.path.join(original_path, f'GRU_model_{sigma_noise}_{P1}')
    os.makedirs(path, exist_ok=True)
    os.chdir(path)

    # Compute parameters for dataset generation
    Ta, Pe = calcolaParametri(g, Sigma, amin, amax, P1, sigma_noise)
    adelta = amax - amin

    # Create training dataset
    train_dataset_segnali = create_dataset(P1, sigma_noise, M, adelta, g, amin)
    X_train = train_dataset_segnali[:, :-1]  # Feature matrix
    y_train = train_dataset_segnali[:, -1]   # Labels

    # Train the model and save the best model
    model, best_accuracy, best_params = train_model(X_train, y_train, epochs=30)
    save_pytorch_model(model, os.path.join(path, 'gru_model.pth'))

    # Define ONNX quantization parameters
    input_size = X_train.shape[1]
    hidden_size = best_params['hidden_size']
    input_sizes = [input_size, hidden_size]
    input_names = ['onnx::Gemm_0', 'onnx::Gemm_1']
    output_names = ['output_31']
    op_types_to_quantize = ['Gemm', 'Add', 'Sigmoid', 'Mul', 'Sub', 'Tanh']
    
    # Create dataset for quantization
    test_q_dataset_segnali = create_dataset(P1, sigma_noise, M, adelta, g, amin)
    X_test_q = test_q_dataset_segnali[:, :-1]
    
    # Quantize the model and return the quantized ONNX model path
    quantized_model = run_onnx_quantizer(model, X_test_q, input_names, input_sizes, output_names, op_types_to_quantize)

    # Lists to store performance metrics
    accuracies = []
    accuracies_q = []
    precisions = []
    precisions_q = []
    recalls = []
    recalls_q = []
    f1s = []
    f1s_q = []

    # Evaluate performance for each test run
    for r in R:
        print(f"Test {r}")
        print("-------")
        
        # Create new dataset for testing
        test_dataset_segnali = create_dataset(P1, sigma_noise, M, adelta, g, amin)
        X_test = test_dataset_segnali[:, :-1]
        y_test = test_dataset_segnali[:, -1]

        # Predict using the PyTorch model
        y_preds = predict(model, X_test)
        accuracy = accuracy_score(y_test, y_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_preds, average='binary')
        
        # Store metrics
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        print(f"PyTorch Model Accuracy: {accuracy}")
        print(f"PyTorch Model Precision: {precision}")
        print(f"PyTorch Model Recall: {recall}")
        print(f"PyTorch Model F1-score: {f1}")

        # Predict using the ONNX model
        y_preds_q = []
        for i in range(len(X_test)):
            pred_q = test_onnx_model(quantized_model, X_test, hidden_size, i)
            y_preds_q.append(pred_q)
        y_preds_q = [float(item) for array in y_preds_q for item in array.flatten()]
        
        # Compute metrics for ONNX model
        accuracy_q = accuracy_score(y_test, y_preds_q)
        precision_q, recall_q, f1_q, _ = precision_recall_fscore_support(y_test, y_preds_q, average='binary')
        
        # Store metrics
        accuracies_q.append(accuracy_q)
        precisions_q.append(precision_q)
        recalls_q.append(recall_q)
        f1s_q.append(f1_q)
        print("-------")
        print(f"ONNX Model Accuracy: {accuracy_q}")
        print(f"ONNX Model Precision: {precision_q}")
        print(f"ONNX Model Recall: {recall_q}")
        print(f"ONNX Model F1-score: {f1_q}")
        print("-------")
        print("-------")

    # Restore the original working directory
    os.chdir(original_path)

# Entry point of the script
if __name__ == "__main__":
    main()
