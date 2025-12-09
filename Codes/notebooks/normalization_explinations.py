import numpy as np

print("=" * 80)
print("COMPREHENSIVE GUIDE TO NORMALIZATION TECHNIQUES")
print("=" * 80)

# Example data: 4 students, 6 features each (divisible by 2 and 3 for GroupNorm demo)
data = np.array([
    [7.0,  85.0, 6.0, 3.0, 92.0, 4.0],   # student 1
    [5.0,  90.0, 8.0, 2.0, 88.0, 6.0],   # student 2
    [9.0,  75.0, 7.0, 5.0, 95.0, 3.0],   # student 3
    [6.0,  80.0, 5.0, 4.0, 90.0, 5.0]    # student 4
])

print("\nOriginal Data (4 samples × 6 features):")
print("         F0     F1     F2     F3     F4     F5")
for i, row in enumerate(data):
    print(f"Sample {i+1}: {row[0]:5.1f}  {row[1]:5.1f}  {row[2]:5.1f}  {row[3]:5.1f}  {row[4]:5.1f}  {row[5]:5.1f}")

epsilon = 1e-5  # small constant for numerical stability

# =============================================================================
# 1. BATCH NORMALIZATION
# =============================================================================
print("\n" + "=" * 80)
print("1. BATCH NORMALIZATION (BatchNorm1d)")
print("   Normalizes ACROSS the batch, FOR EACH feature separately")
print("=" * 80)

print("""
VISUAL:
         F0     F1     F2     F3     F4     F5
Sample 1: [7.0]  [85.0] [6.0]  [3.0]  [92.0] [4.0]
Sample 2: [5.0]  [90.0] [8.0]  [2.0]  [88.0] [6.0]
Sample 3: [9.0]  [75.0] [7.0]  [5.0]  [95.0] [3.0]
Sample 4: [6.0]  [80.0] [5.0]  [4.0]  [90.0] [5.0]
           ↓      ↓      ↓      ↓      ↓      ↓
         mean   mean   mean   mean   mean   mean
         std    std    std    std    std    std
         
Each COLUMN (feature) normalized independently using batch statistics.
""")

batch_means = np.mean(data, axis=0)
batch_stds = np.std(data, axis=0, ddof=0)
batch_normalized = (data - batch_means) / (batch_stds + epsilon)

print("Statistics per feature:")
for j in range(data.shape[1]):
    print(f"  F{j}: mean={batch_means[j]:6.2f}, std={batch_stds[j]:5.2f}")

print("\nBatchNorm Result:")
print("         F0       F1       F2       F3       F4       F5")
for i, row in enumerate(batch_normalized):
    print(f"Sample {i+1}: {row[0]:7.3f}  {row[1]:7.3f}  {row[2]:7.3f}  {row[3]:7.3f}  {row[4]:7.3f}  {row[5]:7.3f}")

print("""
✓ Each feature column has mean ≈ 0, std ≈ 1
✓ Learns running statistics during training
✗ Depends on batch size (unstable with small batches)
✗ Different behavior in train vs eval mode
""")

# =============================================================================
# 2. LAYER NORMALIZATION
# =============================================================================
print("\n" + "=" * 80)
print("2. LAYER NORMALIZATION (LayerNorm)")
print("   Normalizes ACROSS features, FOR EACH sample separately")
print("=" * 80)

print("""
VISUAL:
         F0     F1     F2     F3     F4     F5
         ←―――――――――――――――――――――――――――――――――――――→ mean/std for row
Sample 1: [7.0   85.0   6.0    3.0   92.0   4.0]
         ←―――――――――――――――――――――――――――――――――――――→ mean/std for row
Sample 2: [5.0   90.0   8.0    2.0   88.0   6.0]
         ←―――――――――――――――――――――――――――――――――――――→ mean/std for row
Sample 3: [9.0   75.0   7.0    5.0   95.0   3.0]
         ←―――――――――――――――――――――――――――――――――――――→ mean/std for row
Sample 4: [6.0   80.0   5.0    4.0   90.0   5.0]

Each ROW (sample) normalized independently using its own statistics.
""")

layer_means = np.mean(data, axis=1, keepdims=True)
layer_stds = np.std(data, axis=1, keepdims=True, ddof=0)
layer_normalized = (data - layer_means) / (layer_stds + epsilon)

print("Statistics per sample:")
for i in range(data.shape[0]):
    print(f"  Sample {i+1}: mean={layer_means[i,0]:6.2f}, std={layer_stds[i,0]:5.2f}")

print("\nLayerNorm Result:")
print("         F0       F1       F2       F3       F4       F5")
for i, row in enumerate(layer_normalized):
    print(f"Sample {i+1}: {row[0]:7.3f}  {row[1]:7.3f}  {row[2]:7.3f}  {row[3]:7.3f}  {row[4]:7.3f}  {row[5]:7.3f}")

print("""
✓ Each sample row has mean ≈ 0, std ≈ 1
✓ Batch-size independent (works with batch_size=1)
✓ Same behavior in train and eval mode
✓ Used in Transformers, RNNs
""")

# =============================================================================
# 3. INSTANCE NORMALIZATION
# =============================================================================
print("\n" + "=" * 80)
print("3. INSTANCE NORMALIZATION (InstanceNorm1d)")
print("   Similar to LayerNorm - each sample normalized independently")
print("=" * 80)

print("""
VISUAL (for 1D tabular data, very similar to LayerNorm):
         F0     F1     F2     F3     F4     F5
         ←―――――――――――――――――――――――――――――――――――――→ normalize this instance
Sample 1: [7.0   85.0   6.0    3.0   92.0   4.0]
         ←―――――――――――――――――――――――――――――――――――――→ normalize this instance
Sample 2: [5.0   90.0   8.0    2.0   88.0   6.0]
...

For images (InstanceNorm2d), it normalizes each channel of each image
independently - used heavily in style transfer!

In our 1D case: effectively same as LayerNorm but with different
default affine parameters and designed for "channel-like" features.
""")

# For 1D data, InstanceNorm is essentially the same computation as LayerNorm
instance_normalized = layer_normalized.copy()  # Same result for 1D

print("InstanceNorm1d Result (same as LayerNorm for 1D tabular):")
print("         F0       F1       F2       F3       F4       F5")
for i, row in enumerate(instance_normalized):
    print(f"Sample {i+1}: {row[0]:7.3f}  {row[1]:7.3f}  {row[2]:7.3f}  {row[3]:7.3f}  {row[4]:7.3f}  {row[5]:7.3f}")

print("""
✓ Each instance normalized completely independently
✓ Popular in style transfer (removes style information)
✓ No batch statistics stored
Note: For 1D tabular data, essentially equivalent to LayerNorm
""")

# =============================================================================
# 4. GROUP NORMALIZATION
# =============================================================================
print("\n" + "=" * 80)
print("4. GROUP NORMALIZATION (GroupNorm)")
print("   Divides features into GROUPS, normalizes within each group per sample")
print("=" * 80)

print("""
VISUAL (with 2 groups of 3 features each):
         ┌─── Group 1 ───┐  ┌─── Group 2 ───┐
         F0     F1     F2    F3     F4     F5
Sample 1: [7.0   85.0   6.0]  [3.0   92.0   4.0]
           └─ normalize ─┘    └─ normalize ─┘
           
Sample 2: [5.0   90.0   8.0]  [2.0   88.0   6.0]
           └─ normalize ─┘    └─ normalize ─┘
           
Each group within each sample is normalized independently.
Middle ground between LayerNorm (1 group) and InstanceNorm (N groups).
""")

num_groups = 2
features_per_group = data.shape[1] // num_groups
group_normalized = np.zeros_like(data)

print(f"Using {num_groups} groups with {features_per_group} features each")
print("\nStatistics per sample per group:")

for i in range(data.shape[0]):
    for g in range(num_groups):
        start_idx = g * features_per_group
        end_idx = (g + 1) * features_per_group
        group_data = data[i, start_idx:end_idx]
        
        group_mean = np.mean(group_data)
        group_std = np.std(group_data, ddof=0)
        
        group_normalized[i, start_idx:end_idx] = (group_data - group_mean) / (group_std + epsilon)
        
        print(f"  Sample {i+1}, Group {g+1} (F{start_idx}-F{end_idx-1}): mean={group_mean:6.2f}, std={group_std:5.2f}")

print("\nGroupNorm Result (2 groups):")
print("         F0       F1       F2    |   F3       F4       F5")
print("         ←―――― Group 1 ―――――→    |   ←―――― Group 2 ―――――→")
for i, row in enumerate(group_normalized):
    print(f"Sample {i+1}: {row[0]:7.3f}  {row[1]:7.3f}  {row[2]:7.3f}  |  {row[3]:7.3f}  {row[4]:7.3f}  {row[5]:7.3f}")

print("""
✓ Flexible: num_groups=1 → LayerNorm, num_groups=num_features → InstanceNorm
✓ Batch-size independent
✓ Good for varying batch sizes
✓ Used in object detection, segmentation models
""")

# =============================================================================
# 5. RMS NORMALIZATION (RMSNorm)
# =============================================================================
print("\n" + "=" * 80)
print("5. RMS NORMALIZATION (RMSNorm)")
print("   Like LayerNorm but WITHOUT mean centering - only scales by RMS")
print("=" * 80)

print("""
VISUAL:
LayerNorm:  normalized = (x - mean) / std
RMSNorm:    normalized = x / RMS(x)    ← No mean subtraction!

Where RMS(x) = sqrt(mean(x²))

         F0     F1     F2     F3     F4     F5
         ←―――――――――――――――――――――――――――――――――――――→ compute RMS for row
Sample 1: [7.0   85.0   6.0    3.0   92.0   4.0]
         
RMS = sqrt((7² + 85² + 6² + 3² + 92² + 4²) / 6) = sqrt(2638.17) ≈ 51.36

Then divide each value by RMS (no centering!)
""")

# RMSNorm: normalize by root mean square (no mean subtraction)
rms = np.sqrt(np.mean(data ** 2, axis=1, keepdims=True) + epsilon)
rms_normalized = data / rms

print("RMS values per sample:")
for i in range(data.shape[0]):
    print(f"  Sample {i+1}: RMS = {rms[i,0]:.4f}")

print("\nRMSNorm Result:")
print("         F0       F1       F2       F3       F4       F5")
for i, row in enumerate(rms_normalized):
    print(f"Sample {i+1}: {row[0]:7.4f}  {row[1]:7.4f}  {row[2]:7.4f}  {row[3]:7.4f}  {row[4]:7.4f}  {row[5]:7.4f}")

print("""
✓ Simpler than LayerNorm (fewer operations)
✓ Better gradient flow (no mean computation in backprop)
✓ Used in LLaMA, GPT-3.5+, Gemma, and modern LLMs
✓ Potentially less "memory" of training data (relevant for unlearning)
✗ Values are NOT centered around 0
""")

# =============================================================================
# 6. COMPARISON TABLE
# =============================================================================
print("\n" + "=" * 80)
print("COMPARISON: SAME DATA, DIFFERENT NORMALIZATIONS")
print("=" * 80)

print("\nSample 1 normalized values:")
print("-" * 70)
print(f"{'Method':<15} {'F0':>8} {'F1':>8} {'F2':>8} {'F3':>8} {'F4':>8} {'F5':>8}")
print("-" * 70)
print(f"{'Original':<15} {data[0,0]:8.2f} {data[0,1]:8.2f} {data[0,2]:8.2f} {data[0,3]:8.2f} {data[0,4]:8.2f} {data[0,5]:8.2f}")
print(f"{'BatchNorm':<15} {batch_normalized[0,0]:8.3f} {batch_normalized[0,1]:8.3f} {batch_normalized[0,2]:8.3f} {batch_normalized[0,3]:8.3f} {batch_normalized[0,4]:8.3f} {batch_normalized[0,5]:8.3f}")
print(f"{'LayerNorm':<15} {layer_normalized[0,0]:8.3f} {layer_normalized[0,1]:8.3f} {layer_normalized[0,2]:8.3f} {layer_normalized[0,3]:8.3f} {layer_normalized[0,4]:8.3f} {layer_normalized[0,5]:8.3f}")
print(f"{'InstanceNorm':<15} {instance_normalized[0,0]:8.3f} {instance_normalized[0,1]:8.3f} {instance_normalized[0,2]:8.3f} {instance_normalized[0,3]:8.3f} {instance_normalized[0,4]:8.3f} {instance_normalized[0,5]:8.3f}")
print(f"{'GroupNorm(2)':<15} {group_normalized[0,0]:8.3f} {group_normalized[0,1]:8.3f} {group_normalized[0,2]:8.3f} {group_normalized[0,3]:8.3f} {group_normalized[0,4]:8.3f} {group_normalized[0,5]:8.3f}")
print(f"{'RMSNorm':<15} {rms_normalized[0,0]:8.4f} {rms_normalized[0,1]:8.4f} {rms_normalized[0,2]:8.4f} {rms_normalized[0,3]:8.4f} {rms_normalized[0,4]:8.4f} {rms_normalized[0,5]:8.4f}")
print("-" * 70)

# =============================================================================
# 7. WEIGHT NORMALIZATION (Conceptual)
# =============================================================================
print("\n" + "=" * 80)
print("7. WEIGHT NORMALIZATION (WeightNorm) - CONCEPTUAL")
print("   Normalizes WEIGHTS of layers, not activations")
print("=" * 80)

print("""
DIFFERENT PARADIGM: Instead of normalizing layer outputs (activations),
Weight Normalization reparameterizes the weight vectors.

Original weight vector: w
Weight Norm splits it into:
  - Direction: v (learnable vector)
  - Magnitude: g (learnable scalar)
  
  w = g * (v / ||v||)
  
NOTE: The final weights w are the SAME, but during training, the network
learns g and v separately. This decoupling can improve optimization!
  
VISUAL (for a weight matrix with 6 input features, 2 output neurons):
                        
Original:               Weight Normalized:
┌─────────────┐         ┌─────────────┐
│  w11  w12   │    →    │ g·v11/||v|| │   g = magnitude (scalar)
│  w21  w22   │         │ g·v21/||v|| │   v = direction (vector)
│  w31  w32   │         │ g·v31/||v|| │   ||v|| = norm of v
│  w41  w42   │         │ g·v41/||v|| │   
│  w51  w52   │         │ g·v51/||v|| │   Each COLUMN is normalized
│  w61  w62   │         │ g·v61/||v|| │   independently (per output neuron)
└─────────────┘         └─────────────┘
  6×2 matrix              6×2 matrix

Key Insight: Separates "what direction" from "how much"
Each column (output neuron) has its own magnitude g and direction v/||v||
""")

# Demonstrate with example weights
# For 6 input features → 2 output neurons, weight matrix should be 6×2
example_weights = np.array([
    [3.0, 4.0],   # weights for feature 0
    [1.0, 2.0],   # weights for feature 1
    [2.0, 1.0],   # weights for feature 2
    [1.5, 3.0],   # weights for feature 3
    [2.5, 1.5],   # weights for feature 4
    [3.5, 2.5]    # weights for feature 5
])
print("Example weight matrix (6 input features × 2 output neurons):")
print(example_weights)
print(f"Shape: {example_weights.shape} (matches our 6-feature data!)")

# Weight normalization for each column (output neuron)
print("\nDemonstrating the reparameterization (w = g * v/||v||):")
for col in range(example_weights.shape[1]):
    v = example_weights[:, col]
    norm_v = np.linalg.norm(v)
    g = norm_v  # magnitude
    direction = v / norm_v
    print(f"\nOutput neuron {col}: weight vector v = {v}")
    print(f"  ||v|| = {norm_v:.4f}")
    print(f"  direction = v/||v|| = {direction}")
    print(f"  g (magnitude) = {g:.4f}")
    print(f"  Reconstructed: g * direction = {g * direction}")
    print(f"  ✓ Same as original! (g * direction = v)")

print("""
IMPORTANT: The reconstructed weights ARE the same as the original!
The benefit is HOW the network LEARNS during training:
  - Instead of learning w directly, it learns g and v separately
  - Gradient updates to g (magnitude) don't affect direction
  - Gradient updates to v (direction) are automatically normalized
  - This can lead to faster/more stable optimization
  
Example during training:
  - If you want weights to be "stronger", just increase g
  - If you want to change "what" the neuron responds to, adjust v
  - These two aspects are decoupled in the gradient updates!
""")

print("""
✓ Decouples weight magnitude from direction
✓ Can improve optimization (smoother loss landscape)
✓ Applied to WEIGHTS, not a separate layer
✗ Not compatible with nn.Sequential (wraps layers)
✗ Different implementation approach in PyTorch:
    from torch.nn.utils import weight_norm
    layer = weight_norm(nn.Linear(in, out))
""")

# =============================================================================
# 8. SPECTRAL NORMALIZATION (Conceptual)
# =============================================================================
print("\n" + "=" * 80)
print("8. SPECTRAL NORMALIZATION (SpectralNorm) - CONCEPTUAL")
print("   Constrains the spectral norm (largest singular value) of weights")
print("=" * 80)

print("""
WHAT IS SPECTRAL NORM?
The spectral norm σ(W) is the largest singular value of a matrix W.
It represents the maximum "stretching" factor of the transformation.

Spectral Normalization divides weights by their spectral norm:
  W_normalized = W / σ(W)
  
This ensures the Lipschitz constant of the layer ≤ 1.

VISUAL (using our 6×2 weight matrix):
Original matrix W:        After Spectral Norm:
┌─────────────┐           ┌─────────────┐
│  3.0   4.0  │           │  W/σ(W)     │  ← Divided by σ(W)
│  1.0   2.0  │           │             │
│  2.0   1.0  │     →     │  All values │
│  1.5   3.0  │           │  scaled     │
│  2.5   1.5  │           │  down       │
│  3.5   2.5  │           │             │
└─────────────┘           └─────────────┘
  6×2 matrix                6×2 matrix
  σ(W) = largest           σ(W_norm) = 1.0
  singular value
""")

# Compute spectral norm of example weights
U, S, Vt = np.linalg.svd(example_weights)
spectral_norm = S[0]  # Largest singular value
spectral_normalized = example_weights / spectral_norm

print(f"Example weight matrix:")
print(example_weights)
print(f"\nSingular values: {S}")
print(f"Spectral norm (largest): {spectral_norm:.4f}")
print(f"\nSpectrally normalized weights:")
print(spectral_normalized)
print(f"New spectral norm: {np.linalg.svd(spectral_normalized)[1][0]:.4f}")

print("""
✓ Stabilizes GAN training (prevents mode collapse)
✓ Controls Lipschitz constant of network
✓ Used in discriminators of GANs
✗ Not for general-purpose training
✗ Applied to weights, not a layer in Sequential:
    from torch.nn.utils import spectral_norm
    layer = spectral_norm(nn.Linear(in, out))
""")

# =============================================================================
# 9. SPATIAL NORMALIZATIONS (2D/3D) - Conceptual
# =============================================================================
print("\n" + "=" * 80)
print("9. SPATIAL NORMALIZATIONS (2D/3D) - CONCEPTUAL")
print("   For images and volumetric data with spatial structure")
print("=" * 80)

print("""
These normalizations are designed for data with SPATIAL dimensions:
- Images: (Batch, Channels, Height, Width)
- Videos/3D: (Batch, Channels, Depth, Height, Width)

NOT APPLICABLE to 1D tabular data like yours!

┌─────────────────────────────────────────────────────────────────────┐
│ BATCHNORM2D                                                         │
│                                                                     │
│ For each channel, compute mean/std across (batch, height, width)    │
│                                                                     │
│ Image batch:    Channel 0        Channel 1        Channel 2         │
│                ┌─────────┐      ┌─────────┐      ┌─────────┐        │
│ Sample 1:      │ H × W   │      │ H × W   │      │ H × W   │        │
│                └─────────┘      └─────────┘      └─────────┘        │
│                ┌─────────┐      ┌─────────┐      ┌─────────┐        │
│ Sample 2:      │ H × W   │      │ H × W   │      │ H × W   │        │
│                └─────────┘      └─────────┘      └─────────┘        │
│                    ↓                ↓                ↓              │
│                 mean/std         mean/std         mean/std          │
│              (across batch      (across batch    (across batch      │
│               & spatial)         & spatial)       & spatial)        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ INSTANCENORM2D                                                      │
│                                                                     │
│ For each sample, for each channel, normalize across spatial dims    │
│                                                                     │
│ Single image:  Channel 0        Channel 1        Channel 2          │
│                ┌─────────┐      ┌─────────┐      ┌─────────┐        │
│                │ H × W   │      │ H × W   │      │ H × W   │        │
│                │normalize│      │normalize│      │normalize│        │
│                │  this   │      │  this   │      │  this   │        │
│                └─────────┘      └─────────┘      └─────────┘        │
│                 mean/std         mean/std         mean/std          │
│              (across spatial)  (across spatial) (across spatial)    │
│                                                                     │
│ Used heavily in STYLE TRANSFER - removes style information!         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ LOCAL RESPONSE NORMALIZATION (LRN) - Historical                     │
│                                                                     │
│ Normalizes across nearby channels at each spatial position          │
│ Used in AlexNet (2012), now mostly replaced by BatchNorm            │
│                                                                     │
│ At position (x, y), normalize channel c using channels [c-n, c+n]   │
│                                                                     │
│                    Channel c-1  Channel c  Channel c+1              │
│                    ┌─────────┐ ┌─────────┐ ┌─────────┐              │
│                    │    ·    │ │    ·    │ │    ·    │              │
│                    │ (x,y) ←─┼─┼─(x,y)──┼─┼→(x,y)   │              │
│                    │    ·    │ │    ·    │ │    ·    │              │
│                    └─────────┘ └─────────┘ └─────────┘              │
│                    Local neighborhood normalization                 │
└─────────────────────────────────────────────────────────────────────┘
""")

print("""
WHY THESE DON'T FIT TABULAR DATA:
- Tabular data shape: (batch_size, num_features)
- No spatial dimensions (height, width)
- No meaningful "channel" concept
- Features are independent columns, not spatial neighbors
""")

# =============================================================================
# 10. SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY: WHICH NORMALIZATION TO USE?")
print("=" * 80)

print("""
┌────────────────────┬─────────────────────┬─────────────────────┬─────────────────────┐
│ Normalization      │ Normalizes Over     │ Best For            │ Implemented?        │
├────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ BatchNorm1d        │ Batch (per feature) │ Large batches, MLPs │ ✓ In notebook       │
│ LayerNorm          │ Features (per sample)│ Small batches, RNNs │ ✓ In notebook       │
│ InstanceNorm1d     │ Features (per sample)│ Style-independent   │ ✓ In notebook       │
│ GroupNorm          │ Feature groups      │ Varying batch sizes │ ✓ In notebook       │
│ RMSNorm            │ Features (RMS only) │ Modern LLMs         │ ✓ In notebook       │
│ None               │ -                   │ Baseline            │ ✓ In notebook       │
├────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ WeightNorm         │ Weight vectors      │ Optimization        │ ✗ Different API     │
│ SpectralNorm       │ Weight matrix       │ GANs                │ ✗ Different API     │
├────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
│ BatchNorm2d/3d     │ Batch + spatial     │ Images/Videos       │ ✗ Need spatial dims │
│ InstanceNorm2d/3d  │ Spatial (per sample)│ Style transfer      │ ✗ Need spatial dims │
│ LocalResponseNorm  │ Local channels      │ CNNs (old)          │ ✗ Need spatial dims │
└────────────────────┴─────────────────────┴─────────────────────┴─────────────────────┘
""")

# =============================================================================
# 11. MACHINE UNLEARNING IMPLICATIONS
# =============================================================================
print("\n" + "=" * 80)
print("MACHINE UNLEARNING IMPLICATIONS")
print("=" * 80)

print("""
Different normalizations have different implications for machine unlearning:

┌────────────────────┬─────────────────────────────────────────────────────────────┐
│ Normalization      │ Unlearning Consideration                                    │
├────────────────────┼─────────────────────────────────────────────────────────────┤
│ BatchNorm          │ ⚠️  Stores running_mean and running_var that accumulate     │
│                    │    information about ALL training data (including forget    │
│                    │    set). May need to reset or adjust these statistics.      │
├────────────────────┼─────────────────────────────────────────────────────────────┤
│ LayerNorm          │ ✓  No running statistics - only learns γ and β per feature │
│                    │    Less "memory" of specific training samples               │
├────────────────────┼─────────────────────────────────────────────────────────────┤
│ InstanceNorm       │ ✓  Each sample normalized independently                     │
│                    │    Minimal cross-sample information leakage                 │
├────────────────────┼─────────────────────────────────────────────────────────────┤
│ GroupNorm          │ ✓  No running statistics, batch-independent                 │
│                    │    Good middle-ground for unlearning                        │
├────────────────────┼─────────────────────────────────────────────────────────────┤
│ RMSNorm            │ ✓✓ Simplest - only learns scale (γ), no shift (β)          │
│                    │    Fewer learned parameters = less "memory"                 │
│                    │    No mean computation = potentially cleaner gradients      │
├────────────────────┼─────────────────────────────────────────────────────────────┤
│ None               │ ✓✓ No normalization parameters to worry about              │
│                    │    But may have training stability issues                   │
└────────────────────┴─────────────────────────────────────────────────────────────┘

RECOMMENDATION FOR MACHINE UNLEARNING EXPERIMENTS:
1. Compare BatchNorm vs others (BatchNorm has running stats to consider)
2. RMSNorm and LayerNorm are cleaner for unlearning analysis
3. Consider that noise injection affects normalization params too!
""")

# =============================================================================
# 12. PYTORCH CODE REFERENCE
# =============================================================================
print("\n" + "=" * 80)
print("PYTORCH CODE REFERENCE")
print("=" * 80)

print("""
# ============================================================================
# ACTIVATION NORMALIZATIONS (used as layers in nn.Sequential)
# ============================================================================

import torch.nn as nn

# BatchNorm1d - for tabular data
bn = nn.BatchNorm1d(num_features=64)
# Input: (batch_size, 64) → Output: (batch_size, 64)

# LayerNorm - for tabular data  
ln = nn.LayerNorm(normalized_shape=64)
# Input: (batch_size, 64) → Output: (batch_size, 64)

# InstanceNorm1d - for tabular data
ins = nn.InstanceNorm1d(num_features=64, affine=True)
# Input: (batch_size, 64) → Output: (batch_size, 64)

# GroupNorm - for tabular data
gn = nn.GroupNorm(num_groups=4, num_channels=64)
# Input: (batch_size, 64) → Output: (batch_size, 64)
# Note: 64 must be divisible by num_groups (4)

# RMSNorm - custom implementation (not in PyTorch)
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight

# ============================================================================
# WEIGHT NORMALIZATIONS (wrap layers, different API)
# ============================================================================

from torch.nn.utils import weight_norm, spectral_norm

# Weight Normalization - wraps a layer
layer = weight_norm(nn.Linear(64, 32))
# Access: layer.weight_v (direction), layer.weight_g (magnitude)

# Spectral Normalization - wraps a layer  
layer = spectral_norm(nn.Linear(64, 32))
# Constrains spectral norm to 1

# ============================================================================
# SPATIAL NORMALIZATIONS (for images)
# ============================================================================

# BatchNorm2d - for images
bn2d = nn.BatchNorm2d(num_features=64)  # 64 channels
# Input: (batch, 64, H, W) → Output: (batch, 64, H, W)

# InstanceNorm2d - for images (style transfer)
in2d = nn.InstanceNorm2d(num_features=64)
# Input: (batch, 64, H, W) → Output: (batch, 64, H, W)

# LocalResponseNorm - for images (old, AlexNet-era)
lrn = nn.LocalResponseNorm(size=5)  # Normalize across 5 nearby channels
# Input: (batch, C, H, W) → Output: (batch, C, H, W)
""")

print("=" * 80)
print("END OF NORMALIZATION GUIDE")
print("=" * 80)
