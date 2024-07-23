import os
import numpy as np
import torch
import warnings
from MLP_model import train_model, save_pytorch_model
from MLP_onnx_operations import convert_to_onnx, quantize_model, check_quantization, test_onnx_model
from MLP_utils import create_dataset, calcolaParametri
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

warnings.filterwarnings("ignore")

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Generate the Gaussian signal g
    t = np.arange(-20, 21)
    sigma_gau = 5
    g = np.exp(-t ** 2 / sigma_gau ** 2)
    g = np.resize(g, (41, 1))

    # Define amplitude range and other parameters
    amin = 0.6
    amax = 1.4
    R = np.arange(1, 101)
    M = 10000

    sigma_noise = 0.5
    P1 = 0.05
    
    # Generate the noise covariance matrix
    Sigma = (sigma_noise ** 2) * np.diag(np.ones(g.size))

    # Create directory for saving models
    original_path = os.getcwd()
    path = os.path.join(original_path, f'MLP_model_{sigma_noise}_{P1}')
    os.makedirs(path, exist_ok=True)
    os.chdir(path)

    # Calculate decision threshold and error probability
    Ta, Pe = calcolaParametri(g, Sigma, amin, amax, P1, sigma_noise)
    adelta = amax - amin

    # Create training dataset
    train_dataset_segnali = create_dataset(P1, sigma_noise, M, adelta, g, amin)
    X_train = train_dataset_segnali[:, :-1]
    y_train = train_dataset_segnali[:, -1]

    # Train the model
    model, best_accuracy, best_params = train_model(X_train, y_train, epochs=30)
    save_pytorch_model(model, os.path.join(path, 'mlp_model.pth'))

    # Convert trained model to ONNX format
    onnx_model = convert_to_onnx(model, (1, 41))
    
    # Create test dataset for quantization
    test_q_dataset_segnali = create_dataset(P1, sigma_noise, M, adelta, g, amin)
    X_test_q = torch.tensor(test_q_dataset_segnali[:, :-1], dtype=torch.float32)
    
    # Quantize the ONNX model
    quantized_model = quantize_model(onnx_model, X_test_q)
    check_quantization(quantized_model)

    # Initialize lists to store metrics
    accuracies = []
    accuracies_q = []
    precisions = []
    precisions_q = []
    recalls = []
    recalls_q = []
    f1s = []
    f1s_q = []

    # Test the models over multiple runs
    for r in R:
        print(f"Test {r}")
        print("-------")
        
        # Create test dataset
        test_dataset_segnali = create_dataset(P1, sigma_noise, M, adelta, g, amin)
        X_test = test_dataset_segnali[:, :-1]
        y_test = test_dataset_segnali[:, -1]
        
        # Evaluate PyTorch model
        y_preds = model.predict(torch.tensor(X_test, dtype=torch.float32).to(device))
        accuracy = accuracy_score(y_test, y_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_preds, average='binary')
        
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        print(f"PyTorch Model Accuracy: {accuracy}")
        print(f"PyTorch Model Precision: {precision}")
        print(f"PyTorch Model Recall: {recall}")
        print(f"PyTorch Model F1-score: {f1}")
        
        # Evaluate quantized ONNX model
        y_preds_q = []
        for i in list(range(len(X_test))):
            pred_q = test_onnx_model(quantized_model, X_test, y_test, i)
            y_preds_q.append(pred_q)
        y_preds_q = [float(item) for array in y_preds_q for item in array.flatten()]
        
        accuracy_q = accuracy_score(y_test, y_preds_q)
        precision_q, recall_q, f1_q, _ = precision_recall_fscore_support(y_test, y_preds_q, average='binary')
        
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

    # Return to the original directory
    os.chdir(original_path)

if __name__ == "__main__":
    main()
