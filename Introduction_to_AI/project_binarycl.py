# project_binarycl.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import(
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans

# Seed fixed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 한글 깨짐 방지 
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("4th practice : Introduction to AI - Intergrated Execution")
print("="*70)

# print("="*70)
# Part 1 : Model Definition
# print("="*70)

print("\n[Part 1] Model Definition ing")

class BinaryClassification(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    # Forward
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.layer3(x))
        return x
    
class Regressor(nn.Module):
    """MLP Regressor"""
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)
    
print("finish model definition")

# print("="*70)
# Part 2 : Supervised Training
# print("="*70)
print("\n[Part2] Supervise Training - Classification and Regression")

# Classification data generation and training


X_class, y_class = make_classification(
    n_samples=1000, n_features=20, n_informative=15,
    n_redundant=5, weights=[0.7, 0.3], random_state=42
)

X_train_c, X_temp_c, y_train_c, y_temp_c = train_test_split(
    X_class, y_class, test_size=0.4, random_state=42, stratify=y_class
)

X_val_c, X_test_c, y_val_c, y_test_c = train_test_split(
    X_temp_c, y_temp_c, test_size=0.5, random_state=42, stratify=y_temp_c
)

# Normalization (Standardization: mean 0, variance 1) scaling
scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_val_c_scaled = scaler_c.transform(X_val_c)
X_test_c_scaled = scaler_c.transform(X_test_c)

# Transform scaled data into tensor
X_train_c_t = torch.FloatTensor(X_train_c_scaled)
y_train_c_t = torch.FloatTensor(y_train_c).unsqueeze(1)
X_val_c_t = torch.FloatTensor(X_val_c_scaled)
y_val_c_t = torch.FloatTensor(y_val_c).unsqueeze(1)