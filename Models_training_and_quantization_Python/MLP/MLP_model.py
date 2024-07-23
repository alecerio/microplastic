import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import accuracy_score

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the neural network model
class Net(nn.Module):
    def __init__(self, num_layers, num_neurons):
        super(Net, self).__init__()
        layers = []
        input_size = 41  # Example input size, modify as per your input features
        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, num_neurons))
            layers.append(nn.ReLU())
            input_size = num_neurons
        layers.append(nn.Linear(num_neurons, 2))  # Output layer for binary classification
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
    # Method for making predictions
    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X = X.to(next(self.parameters()).device)
            outputs = self(X)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

# Function to train the model
def train_model(X_train, y_train, epochs=10):
    # Convert training data to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    # Parameter grid for hyperparameter tuning
    param_grid = {
        'num_layers': [1, 2, 3],  # Number of hidden layers
        'num_neurons': [8, 16, 32, 64]  # Number of neurons in each layer
    }

    best_accuracy = 0
    best_params = None
    best_model = None
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)  # Stratified K-Folds cross-validator

    # Iterate over all combinations of hyperparameters
    for params in ParameterGrid(param_grid):
        model = Net(num_layers=params['num_layers'], num_neurons=params['num_neurons']).to(device)
        criterion = nn.CrossEntropyLoss()  # Loss function for classification
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

        fold_accuracies = []
        print("-------")
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_train, y_train)):
            print(f"Fold {fold + 1}")
            print("-------")

            X_fold_train, X_fold_test = X_train[train_idx], X_train[test_idx]
            y_fold_train, y_fold_test = y_train[train_idx], y_train[test_idx]

            dataset = TensorDataset(X_fold_train, y_fold_train)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=True)  # DataLoader for batching

            # Training loop
            for epoch in range(epochs):
                model.train()
                for inputs, labels in dataloader:
                    optimizer.zero_grad()
                    outputs = model(inputs.to(device))
                    loss = criterion(outputs, labels.to(device))
                    loss.backward()
                    optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                outputs = model(X_fold_test.to(device))
                _, predicted = torch.max(outputs, 1)
                accuracy = accuracy_score(y_fold_test.numpy(), predicted.cpu().numpy())
                fold_accuracies.append(accuracy)

        mean_fold_accuracy = np.mean(fold_accuracies)

        # Check if this combination of hyperparameters is the best so far
        if mean_fold_accuracy > best_accuracy:
            best_accuracy = mean_fold_accuracy
            best_params = params
            best_model = model

    print(f'Best Accuracy: {best_accuracy}')
    print(f'Best Parameters: {best_params}')

    return best_model, best_accuracy, best_params

# Function to save the trained model
def save_pytorch_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
