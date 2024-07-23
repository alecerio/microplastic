import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import accuracy_score

# Set seed for reproducibility
seed = 42
np.random.seed(seed)  # Set seed for numpy random number generation
torch.manual_seed(seed)  # Set seed for PyTorch random number generation

# Device configuration: Use GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the GRU model class
class MyGRU1(nn.Module):
    def __init__(self, input_size, hidden_size, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(MyGRU1, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        # Define GRU weights as linear layers
        self.W_ir = nn.Linear(input_size, hidden_size).to(device)  # Input gate weight for reset gate
        self.W_hr = nn.Linear(hidden_size, hidden_size).to(device)  # Hidden state weight for reset gate
        self.W_iz = nn.Linear(input_size, hidden_size).to(device)  # Input gate weight for update gate
        self.W_hz = nn.Linear(hidden_size, hidden_size).to(device)  # Hidden state weight for update gate
        self.W_in = nn.Linear(input_size, hidden_size).to(device)  # Input gate weight for new hidden state
        self.W_hn = nn.Linear(hidden_size, hidden_size).to(device)  # Hidden state weight for new hidden state

    def forward(self, x, hidden):
        x = x.to(self.device)
        hidden = hidden.to(self.device)
        
        # Calculate reset gate
        r_t = torch.sigmoid(self.W_ir(x) + self.W_hr(hidden))
        # Calculate update gate
        z_t = torch.sigmoid(self.W_iz(x) + self.W_hz(hidden))
        # Calculate new hidden state
        n_t = torch.tanh(self.W_in(x) + r_t * self.W_hn(hidden))
        # Compute final hidden state
        new_hidden = (1 - z_t) * n_t + z_t * hidden
        return new_hidden

# Define a function to make predictions using the trained model
def predict(model, X_test):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        # Initialize hidden state
        hidden = torch.zeros(X_test.shape[0], model.hidden_size).to(model.device)
        # Get model outputs
        outputs = model(torch.tensor(X_test, dtype=torch.float32).to(model.device), hidden)
        # Get predicted classes
        _, predicted = torch.max(outputs, 1)
    return predicted.cpu().numpy()

# Define a function to train the model with cross-validation
def train_model(X_train, y_train, epochs=10):
    # Convert training data to PyTorch tensors and move to device
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)

    # Define hyperparameter grid for grid search
    param_grid = {
        'hidden_size': [8, 16, 32, 64]  # List of hidden sizes to try
    }

    best_accuracy = 0
    best_params = None
    best_model = None
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)  # Initialize StratifiedKFold

    # Grid search over parameter combinations
    for params in ParameterGrid(param_grid):
        input_size = X_train.shape[1]  # Number of input features
        model = MyGRU1(input_size=input_size, hidden_size=params['hidden_size']).to(device)  # Initialize model
        criterion = nn.CrossEntropyLoss()  # Define loss function
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # Define optimizer
        
        fold_accuracies = []  # List to store accuracies for each fold
        print("-------")
        
        # Cross-validation loop
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_train.cpu().numpy(), y_train.cpu().numpy())):
            print(f"Fold {fold + 1}")
            print("-------")

            # Split data into training and testing sets for this fold
            X_fold_train, X_fold_test = X_train[train_idx], X_train[test_idx]
            y_fold_train, y_fold_test = y_train[train_idx], y_train[test_idx]

            # Create DataLoader for batch processing
            dataset = TensorDataset(X_fold_train, y_fold_train)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

            # Training loop
            for epoch in range(epochs):
                model.train()  # Set model to training mode
                for inputs, labels in dataloader:
                    optimizer.zero_grad()  # Clear previous gradients
                    hidden = torch.zeros(inputs.size(0), model.hidden_size).to(device)  # Initialize hidden state
                    outputs = model(inputs.to(device), hidden)  # Forward pass
                    loss = criterion(outputs, labels.to(device))  # Compute loss
                    loss.backward()  # Backward pass
                    optimizer.step()  # Update weights

            # Evaluation
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                hidden = torch.zeros(X_fold_test.size(0), model.hidden_size).to(device)
                outputs = model(X_fold_test.to(device), hidden)
                _, predicted = torch.max(outputs, 1)
                accuracy = accuracy_score(y_fold_test.cpu().numpy(), predicted.cpu().numpy())  # Compute accuracy
                fold_accuracies.append(accuracy)  # Store accuracy for this fold

        # Compute mean accuracy across all folds
        mean_fold_accuracy = np.mean(fold_accuracies)

        # Update best model if current model is better
        if mean_fold_accuracy > best_accuracy:
            best_accuracy = mean_fold_accuracy
            best_params = params
            best_model = model

    print(f'Best Accuracy: {best_accuracy}')
    print(f'Best Parameters: {best_params}')

    return best_model, best_accuracy, best_params

# Define a function to save the trained PyTorch model
def save_pytorch_model(model, path):
    torch.save(model.state_dict(), path)  # Save model state_dict
    print(f"Model saved to {path}")
