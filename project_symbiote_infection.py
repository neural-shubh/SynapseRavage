"""
PROJECT SYMBIOTE — Experiment 1: Progressive Weight Infection
Run: python project_symbiote_infection.py

Dependencies:
    pip install torch numpy matplotlib scikit-learn
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
torch.manual_seed(SEED)
np.random.seed(SEED)

# 1. Synthetic two-moons dataset
X, y = make_moons(n_samples=2000, noise=0.2, random_state=SEED)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


# 2. Small MLP
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


def accuracy_and_loss(model):
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        loss = nn.CrossEntropyLoss()(logits, y_test).item()
        acc = (logits.argmax(dim=1) == y_test).float().mean().item()
    return acc * 100, loss


def plot_boundary(model, title, filename):
    model.eval()
    x_axis = torch.linspace(-2, 3, 220)
    y_axis = torch.linspace(-1.5, 2, 220)
    xx, yy = torch.meshgrid(x_axis, y_axis, indexing="xy")
    grid = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)

    with torch.no_grad():
        pred = model(grid).argmax(dim=1).reshape(xx.shape)

    plt.figure(figsize=(7, 5))
    plt.contourf(xx.numpy(), yy.numpy(), pred.numpy(), alpha=0.32, cmap="coolwarm")
    plt.scatter(
        X_test[:, 0].numpy(), X_test[:, 1].numpy(),
        c=y_test.numpy(), cmap="coolwarm", edgecolors="black", s=13
    )
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    plt.show()


# 3. Train healthy baseline
model = SymbioteNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X_train), y_train)
    loss.backward()
    optimizer.step()

healthy_weights = copy.deepcopy(model.state_dict())
healthy_acc, healthy_loss = accuracy_and_loss(model)
print(f"Healthy baseline: {healthy_acc:.2f}% accuracy | loss {healthy_loss:.4f}")

plot_boundary(model, "Healthy neural network", "symbiote_healthy_boundary.png")

# 4. Generate one fixed noise direction per parameter.
# Reusing the same noise direction at each intensity makes the comparison fairer.
generator = torch.Generator().manual_seed(SEED + 1)
noise_tensors = {
    name: torch.randn(param.shape, generator=generator, dtype=param.dtype)
    for name, param in healthy_weights.items()
}

def infect_model(target_model, baseline_state, intensity):
    infected_state = {}
    for name, param in baseline_state.items():
        infected_state[name] = param + intensity * noise_tensors[name]
    target_model.load_state_dict(infected_state)
    return target_model


# 5. Measure progressive corruption
infection_levels = [0.0, 0.05, 0.10, 0.20, 0.50, 1.0]
results = []

for intensity in infection_levels:
    infect_model(model, healthy_weights, intensity)
    acc, loss = accuracy_and_loss(model)
    results.append((intensity, acc, loss))
    print(f"Infection {intensity:>4.2f} | accuracy {acc:6.2f}% | loss {loss:.4f}")

# 6. Plot degradation
levels = [row[0] for row in results]
accuracies = [row[1] for row in results]
losses = [row[2] for row in results]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(levels, accuracies, marker="o")
axes[0].set(title="Model degradation", xlabel="Noise intensity", ylabel="Test accuracy (%)")
axes[0].grid(True)
axes[1].plot(levels, losses, marker="o")
axes[1].set(title="Test loss under corruption", xlabel="Noise intensity", ylabel="Cross-entropy loss")
axes[1].grid(True)
fig.tight_layout()
fig.savefig("symbiote_degradation.png", dpi=160)
plt.show()

# 7. Visualize a heavily infected copy
infect_model(model, healthy_weights, 0.50)
plot_boundary(model, "Symbiote infection — intensity 0.50", "symbiote_infected_boundary.png")

print("\nSaved: symbiote_degradation.png, symbiote_healthy_boundary.png,")
print("       symbiote_infected_boundary.png")
