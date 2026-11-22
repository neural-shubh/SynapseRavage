"""
PROJECT SYMBIOTE — Experiment 2: Recovery by Fine-Tuning
Run independently: python project_symbiote_recovery.py

Dependencies:
    pip install torch numpy matplotlib scikit-learn

This script rebuilds the same baseline and infected starting point so it
does not depend on variables from the infection experiment's notebook.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

SEED = 42
INFECTION_INTENSITY = 0.50
torch.manual_seed(SEED)
np.random.seed(SEED)

# 1. Recreate the same dataset and split
X, y = make_moons(n_samples=2000, noise=0.2, random_state=SEED)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


# 2. Same MLP architecture
class SymbioteNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.network(x)


criterion = nn.CrossEntropyLoss()

def evaluate(model):
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        loss = criterion(logits, y_test).item()
        acc = (logits.argmax(dim=1) == y_test).float().mean().item()
    return acc * 100, loss


# 3. Train a healthy model using the same baseline procedure
healthy_model = SymbioteNet()
baseline_optimizer = optim.Adam(healthy_model.parameters(), lr=0.01)

for _ in range(100):
    healthy_model.train()
    baseline_optimizer.zero_grad()
    baseline_loss = criterion(healthy_model(X_train), y_train)
    baseline_loss.backward()
    baseline_optimizer.step()

healthy_weights = copy.deepcopy(healthy_model.state_dict())
healthy_acc, _ = evaluate(healthy_model)

# 4. Apply reproducible Gaussian weight corruption
generator = torch.Generator().manual_seed(SEED + 1)
infected_weights = {}
for name, param in healthy_weights.items():
    noise = torch.randn(param.shape, generator=generator, dtype=param.dtype)
    infected_weights[name] = param + INFECTION_INTENSITY * noise

infected_model = SymbioteNet()
infected_model.load_state_dict(infected_weights)
infected_acc, infected_loss = evaluate(infected_model)

print(f"Healthy baseline: {healthy_acc:.2f}%")
print(f"Infected starting point: {infected_acc:.2f}% | loss {infected_loss:.4f}")

# 5. Fine-tune the infected model (not a fresh model)
recovery_optimizer = optim.Adam(infected_model.parameters(), lr=0.001)
recovery_epochs = 100
recovery_losses = []
recovery_accuracies = []

for epoch in range(recovery_epochs):
    infected_model.train()
    recovery_optimizer.zero_grad()
    train_loss = criterion(infected_model(X_train), y_train)
    train_loss.backward()
    recovery_optimizer.step()

    test_acc, test_loss = evaluate(infected_model)
    recovery_losses.append(train_loss.item())
    recovery_accuracies.append(test_acc)

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch + 1:3d}/{recovery_epochs}] | "
            f"train loss {train_loss.item():.4f} | "
            f"test loss {test_loss:.4f} | test accuracy {test_acc:.2f}%"
        )

recovered_acc, recovered_loss = evaluate(infected_model)

# 6. Final comparison
print("\nFINAL COMPARISON")
print(f"Healthy:   {healthy_acc:.2f}%")
print(f"Infected:  {infected_acc:.2f}%")
print(f"Recovered: {recovered_acc:.2f}%")
print(f"Recovery gain over infected: {recovered_acc - infected_acc:+.2f} percentage points")

# 7. Plot recovery curve and loss
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(range(1, recovery_epochs + 1), recovery_accuracies, label="Recovered model")
axes[0].axhline(healthy_acc, linestyle="--", label=f"Healthy baseline ({healthy_acc:.1f}%)")
axes[0].set(title="Symbiote recovery", xlabel="Recovery epoch", ylabel="Test accuracy (%)")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(range(1, recovery_epochs + 1), recovery_losses)
axes[1].set(title="Recovery training loss", xlabel="Recovery epoch", ylabel="Training loss")
axes[1].grid(True)

fig.tight_layout()
fig.savefig("symbiote_recovery.png", dpi=160)
plt.show()

print("\nSaved: symbiote_recovery.png")
