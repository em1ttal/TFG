import numpy as np

print("=" * 80)
print("BATCH NORMALIZATION vs LAYER NORMALIZATION")
print("=" * 80)

# Example data: 4 students, 3 features each
data = np.array([
    [7.0,  85.0, 6.0],   # student 1
    [5.0,  90.0, 8.0],   # student 2
    [9.0,  75.0, 7.0],   # student 3
    [6.0,  80.0, 5.0]    # student 4
])

print("\nOriginal Data (4 students × 3 features):")
print("         Feature 0  Feature 1  Feature 2")
print("         (hours)    (score)    (sleep)")
for i, row in enumerate(data):
    print(f"Student {i+1}:  {row[0]:6.1f}     {row[1]:6.1f}     {row[2]:6.1f}")

print("\n" + "=" * 80)
print("BATCH NORMALIZATION (BatchNorm1d)")
print("Normalizes ACROSS the batch, FOR EACH feature separately")
print("=" * 80)

# BatchNorm: normalize each COLUMN (feature) independently
print("\nStep 1: Compute mean and std for EACH FEATURE across the batch")
print("-" * 80)

batch_means = np.mean(data, axis=0)  # mean along batch dimension
batch_stds = np.std(data, axis=0, ddof=0)   # std along batch dimension

print(f"Feature 0 (hours):  mean = {batch_means[0]:.2f}, std = {batch_stds[0]:.2f}")
print(f"Feature 1 (score):  mean = {batch_means[1]:.2f}, std = {batch_stds[1]:.2f}")
print(f"Feature 2 (sleep):  mean = {batch_means[2]:.2f}, std = {batch_stds[2]:.2f}")

print("\nStep 2: Normalize each feature using its own mean and std")
print("-" * 80)

epsilon = 1e-5  # small constant for numerical stability
batch_normalized = np.zeros_like(data)

for j in range(data.shape[1]):  # for each feature
    print(f"\nFeature {j}:")
    for i in range(data.shape[0]):  # for each student
        normalized_value = (data[i, j] - batch_means[j]) / (batch_stds[j] + epsilon)
        batch_normalized[i, j] = normalized_value
        print(f"  Student {i+1}: ({data[i,j]:.1f} - {batch_means[j]:.2f}) / {batch_stds[j]:.2f} = {normalized_value:.4f}")

print("\nBatchNorm Result (before scale and shift):")
print("         Feature 0  Feature 1  Feature 2")
for i, row in enumerate(batch_normalized):
    print(f"Student {i+1}:  {row[0]:8.4f}   {row[1]:8.4f}   {row[2]:8.4f}")

# With learnable parameters
gamma = np.array([1.0, 1.0, 1.0])  # scale (initialized to 1)
beta = np.array([0.0, 0.0, 0.0])   # shift (initialized to 0)

batch_normalized_scaled = gamma * batch_normalized + beta

print("\nAfter applying γ (scale) and β (shift):")
print(f"γ = {gamma}, β = {beta}")
print("Result (same as above since γ=1, β=0):")
print("         Feature 0  Feature 1  Feature 2")
for i, row in enumerate(batch_normalized_scaled):
    print(f"Student {i+1}:  {row[0]:8.4f}   {row[1]:8.4f}   {row[2]:8.4f}")

print("\n" + "=" * 80)
print("KEY INSIGHT FOR BATCHNORM:")
print("=" * 80)
print("""
- Each FEATURE is normalized independently
- Uses statistics computed ACROSS the batch
- Feature 0 mean/std computed from all 4 students' feature 0 values
- Feature 1 mean/std computed from all 4 students' feature 1 values
- Feature 2 mean/std computed from all 4 students' feature 2 values

After normalization:
- Each feature has mean ≈ 0, std ≈ 1 (across the batch)
- Students are still different from each other
""")

print("\n" + "=" * 80)
print("LAYER NORMALIZATION (LayerNorm)")
print("Normalizes ACROSS features, FOR EACH sample separately")
print("=" * 80)

print("\nStep 1: Compute mean and std for EACH STUDENT across their features")
print("-" * 80)

layer_means = np.mean(data, axis=1)  # mean along feature dimension
layer_stds = np.std(data, axis=1, ddof=0)   # std along feature dimension

for i in range(data.shape[0]):
    print(f"Student {i+1}: mean = {layer_means[i]:.2f}, std = {layer_stds[i]:.2f}")
    print(f"  Features: {data[i]} → mean of these 3 values = {layer_means[i]:.2f}")

print("\nStep 2: Normalize each student using THEIR OWN mean and std")
print("-" * 80)

layer_normalized = np.zeros_like(data)

for i in range(data.shape[0]):  # for each student
    print(f"\nStudent {i+1} (mean={layer_means[i]:.2f}, std={layer_stds[i]:.2f}):")
    for j in range(data.shape[1]):  # for each feature
        normalized_value = (data[i, j] - layer_means[i]) / (layer_stds[i] + epsilon)
        layer_normalized[i, j] = normalized_value
        print(f"  Feature {j}: ({data[i,j]:.1f} - {layer_means[i]:.2f}) / {layer_stds[i]:.2f} = {normalized_value:.4f}")

print("\nLayerNorm Result (before scale and shift):")
print("         Feature 0  Feature 1  Feature 2")
for i, row in enumerate(layer_normalized):
    print(f"Student {i+1}:  {row[0]:8.4f}   {row[1]:8.4f}   {row[2]:8.4f}")

print("\n" + "=" * 80)
print("KEY INSIGHT FOR LAYERNORM:")
print("=" * 80)
print("""
- Each SAMPLE (student) is normalized independently
- Uses statistics computed ACROSS features for that sample
- Student 1 normalized using Student 1's own mean/std
- Student 2 normalized using Student 2's own mean/std
- etc.

After normalization:
- Each student's features have mean ≈ 0, std ≈ 1
- Removes differences in scale between students
""")

print("\n" + "=" * 80)
print("VISUAL COMPARISON")
print("=" * 80)

print("""
BATCHNORM: Normalizes DOWN columns (across batch)
           
           Feature 0  Feature 1  Feature 2
Student 1:   [7.0]      [85.0]     [6.0]
Student 2:   [5.0]      [90.0]     [8.0]
Student 3:   [9.0]      [75.0]     [7.0]
Student 4:   [6.0]      [80.0]     [5.0]
              ↓          ↓          ↓
           compute    compute    compute
           mean/std   mean/std   mean/std
           for col    for col    for col
           
Each column gets normalized to mean=0, std=1


LAYERNORM: Normalizes ACROSS rows (across features)
           
           Feature 0  Feature 1  Feature 2
           ----------------------------------------→ compute mean/std
Student 1:   [7.0       85.0       6.0]              for this row
           ----------------------------------------→ compute mean/std  
Student 2:   [5.0       90.0       8.0]              for this row
           ----------------------------------------→ compute mean/std
Student 3:   [9.0       75.0       7.0]              for this row
           ----------------------------------------→ compute mean/std
Student 4:   [6.0       80.0       5.0]              for this row

Each row gets normalized to mean=0, std=1
""")

print("\n" + "=" * 80)
print("WHICH ONE TO USE?")
print("=" * 80)
print("""
BATCHNORM1D:
✓ Good for: CNNs, fully-connected networks with large batches
✓ Learns different statistics for each feature
✗ Doesn't work well with small batch sizes (unstable statistics)
✗ Different behavior in training vs evaluation
  - Training: uses batch statistics
  - Evaluation: uses running average statistics

LAYERNORM:
✓ Good for: Transformers, RNNs, small batch sizes
✓ Batch size independent (works even with batch_size=1)
✓ Same behavior in training and evaluation
✗ Treats all features the same way

For your thesis:
- BatchNorm is more common in traditional neural networks
- Both can help with training stability
- For machine unlearning, both add extra parameters to consider
""")

print("\n" + "=" * 80)
print("PYTORCH USAGE")
print("=" * 80)
print("""
# BatchNorm1d
import torch.nn as nn

bn = nn.BatchNorm1d(num_features=3)
# For input shape: (batch_size, num_features)
# Example: (128, 3) → normalizes each of 3 features across 128 samples

# LayerNorm  
ln = nn.LayerNorm(normalized_shape=3)
# For input shape: (batch_size, num_features)
# Example: (128, 3) → normalizes each of 128 samples across their 3 features

# In your network:
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),  # Add BatchNorm after Linear
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),  # Add BatchNorm after Linear
            nn.ReLU(),
            # ... etc
        )
""")

print("\n" + "=" * 80)
print("WHAT GETS TRAINED?")
print("=" * 80)
print("""
BatchNorm1d(num_features=16) has:
- 16 γ (gamma/scale) parameters - one per feature
- 16 β (beta/shift) parameters - one per feature
- Total: 32 trainable parameters
- Plus: 16 running means + 16 running vars (not trained, just tracked)

LayerNorm(normalized_shape=16) has:
- 16 γ (gamma/scale) parameters - one per feature
- 16 β (beta/shift) parameters - one per feature  
- Total: 32 trainable parameters

These are LEARNED during training to allow the network to undo
the normalization if needed!
""")

print("=" * 80)