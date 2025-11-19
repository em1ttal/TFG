import torch
import torch.nn as nn

# Simple model: just 3 parameters for clarity
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor([2.0]))
        self.w2 = nn.Parameter(torch.tensor([1.5]))
        self.w3 = nn.Parameter(torch.tensor([0.8]))
    
    def forward(self, x):
        return self.w1 * x + self.w2 * x**2 + self.w3 * x**3

model = TinyModel()

# Forget data
x_forget = torch.tensor([2.0])
y_forget = torch.tensor([15.0])

# Compute loss and gradients
model.zero_grad()
y_pred = model(x_forget)
loss = (y_forget - y_pred) ** 2
loss.backward()

print("Predictions and Loss:")
print(f"  Predicted: {y_pred.item():.2f}")
print(f"  Actual: {y_forget.item():.2f}")
print(f"  Loss: {loss.item():.2f}")
print()

# Extract gradients
print("Gradients (∇_θ L):")
print(f"  ∂L/∂w1 = {model.w1.grad.item():.4f}")
print(f"  ∂L/∂w2 = {model.w2.grad.item():.4f}")
print(f"  ∂L/∂w3 = {model.w3.grad.item():.4f}")
print()

# Compute gradient norms (for single parameters, norm = absolute value)
grad_norms = {
    'w1': abs(model.w1.grad.item()),
    'w2': abs(model.w2.grad.item()),
    'w3': abs(model.w3.grad.item())
}

print("Gradient Norms (||∇_θ L||):")
for name, norm in grad_norms.items():
    print(f"  ||∇_{name} L|| = {norm:.4f}")
print()

# Find maximum gradient norm
max_grad_norm = max(grad_norms.values())
print(f"Maximum gradient norm: {max_grad_norm:.4f}")
print()

# Compute α(θ) for each parameter
print("Scaling factors α(θ) = ||∇_θ L|| / max(||∇_θ L||):")
alphas = {}
for name, norm in grad_norms.items():
    alpha = norm / max_grad_norm
    alphas[name] = alpha
    print(f"  α({name}) = {norm:.4f} / {max_grad_norm:.4f} = {alpha:.4f}")