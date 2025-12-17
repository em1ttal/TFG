import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
from tqdm import tqdm


# Set random seed for reproducibility, always 42!!
rng = np.random.default_rng(42)
torch.manual_seed(42)

# Generate multi-class spirals dataset
N = 300  # Samples per class
num_classes = 4  # Number of spiral classes

all_data = []

for class_idx in range(num_classes):
    # Generate theta values
    theta = np.sqrt(rng.random(N)) * 2 * pi
    
    # Calculate the angle offset for this spiral
    angle_offset = (2 * pi * class_idx) / num_classes
    
    # Generate spiral with rotation
    r = 2 * theta + pi
    
    # Apply rotation to spread spirals evenly
    x = np.cos(theta + angle_offset) * r
    y = np.sin(theta + angle_offset) * r
    
    # Stack coordinates
    data = np.array([x, y]).T
    
    # Add noise to data, Normal distribution with mean 0 and standard deviation 0.5
    data = data + rng.standard_normal((N, 2)) * 0.5
    
    # Add labels
    labeled_data = np.append(data, np.full((N, 1), class_idx), axis=1)
    all_data.append(labeled_data)

# Combine all classes
res = np.vstack(all_data)
rng.shuffle(res)  # Use generator's shuffle method

print(f"Dataset shape: {res.shape}")
print(f"Total samples: {len(res)}")
print(f"Number of classes: {num_classes}")
for i in range(num_classes):
    print(f"Class {i} samples: {np.sum(res[:, 2] == i)}")