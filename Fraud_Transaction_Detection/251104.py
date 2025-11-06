#251104.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import(
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)
import torch
import torch.nn as nn
import torch.optim as optim

# Visualization Settings
# plt.rcParams['font size'] = 12
# plt.rcParams['figure.figsize'] = (10,6)
# plt.rcParams['axes.grid'] = True
             
# Data Generation (Class Imbalance)
# Normal Transaction : 95%
# Fraud Transaction : 5%

# Let's create imbalanced data (Normal : 95%, Fraud : 5%)

X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    weights=[0.95,0.05],
    flip_y=0.01,
    random_state=42
)

# Check class distribution
np.unique(y, return_counts=True)  # unique() removes duplicate values
unique, counts = np.unique(y, return_counts=True) # np.unique() shows the number of samples in each class
print("normal_transaction(0): ", counts[0])
print("fraud_transaction(1): ", counts[1])

# Data Split (Stratified Sampling)
# Stratified Option >> maintains clss ratio between train and test sets

X_train, X_test, y_train, y_test =\
train_test_split(X,y, test_size=0.2, stratify=y, random_state=42)

print(f"fraud ratio in training data: {y_train.sum()/len(y_train)*100: .2f}%"  )
print(f"fraud ratio in test data: {y_test.sum()/len(y_test)*100: .2f}%"  )

# Standardization
# (x-x.mean)/std
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Training data
X_test_scaled = scaler.transform(X_test) # Test data
# hold out

print(f"Raw Data Range:[{X_train.min():.2f}, {X_train.max():.2f}]")
print(f"Standardization range:[{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
print(f"Mean after Standardization:[{X_train_scaled.mean():.2f}]")
print(f"Standard deviation after standardization :[{X_train_scaled.std():.2f}]")

# Pytorch Tensor Conversion
X_train_tensor = torch.FloatTensor(X_train_scaled) # Number of samples, Number of features
y_train_tensor = torch.FloatTensor(y_train).view(-1,1)

X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test).view(-1,1)

print(X_train_tensor.shape, y_train_tensor.shape)
print(X_test_tensor.shape, y_test_tensor.shape)

# Model Definition
# Create a 3-layer Neural Network Model 
class FraudDetectionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x