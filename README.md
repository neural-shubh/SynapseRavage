<div align="center">

# 🕷️ SYNAPSERAVAGE

### `NEURAL NETWORK CORRUPTION & RECOVERY LAB`

**INFECT. DEGRADE. RECOVER.**

*A tiny PyTorch experiment inspired by Spider-Man's symbiote — except the host is a neural network.*

<img src="assets/synapseravage-hologram.png" alt="SynapseRavage holographic laboratory dashboard" width="100%">

`PYTORCH` · `NEURAL NETWORKS` · `ROBUSTNESS` · `FINE-TUNING`

</div>

---

## ◈ SYSTEM BRIEFING

> **LAB ACCESS: PETER PARKER**
>
> A healthy neural network enters the lab. A symbiote made of Gaussian noise meets its parameters. What happens to the model's learned decision boundary?
>
> Then comes the interesting part: can fine-tuning help the infected network recover?

**SynapseRavage** is a playful, small-scale experiment that explores two ideas:

- **Infection:** perturb a trained model's parameters and measure how its predictions change.
- **Recovery:** fine-tune the corrupted model and observe how much performance returns.

This is an educational robustness experiment—not malware, not a model exploit, and not a tool for modifying anyone else's models without permission.

## ◈ THE TWO EXPERIMENTS

<table>
<tr>
<td width="50%" valign="top">

### 🔴 01 — SYMBIOTE INFECTION

Train a small multilayer perceptron on the two-moons dataset, preserve its healthy weights, then inject progressively stronger Gaussian noise into a separate copy.

**Watch for:**
- Accuracy changes
- Test-loss changes
- Decision-boundary distortion

**Run:**
```bash
python project_symbiote_infection.py
```

</td>
<td width="50%" valign="top">

### 🔵 02 — SYMBIOTE RECOVERY

Start from a corrupted copy of the trained network and fine-tune it on the original training data.

**Watch for:**
- Recovery across epochs
- Training-loss behaviour
- Recovered vs. infected vs. healthy accuracy

**Run:**
```bash
python project_symbiote_recovery.py
```

</td>
</tr>
</table>

## ◈ EXPERIMENT READOUT

One observed run from the notebook:

| Specimen | Test accuracy |
|---|---:|
| Healthy network | **96.50%** |
| Infected network (noise intensity 0.50) | **64.50%** |
| Fine-tuned network | **76.00%** |

**Recovery gain:** `+11.50 percentage points` over the infected starting point.

The recovery run showed that fine-tuning improved performance, but did not restore the original baseline in that run.

> Results can vary with random initialization, noise samples, training settings, and execution order. Treat these values as one observed run, not guaranteed outputs. The scripts set seeds to improve repeatability.

## ◈ LAB EQUIPMENT

- Python
- PyTorch
- NumPy
- Matplotlib
- scikit-learn

No GPU required. The experiments are small enough to run on a CPU or in Google Colab.

## ◈ BOOT SEQUENCE

Clone or download the project, then install the dependencies:

```bash
pip install torch numpy matplotlib scikit-learn
```

Run either experiment independently:

```bash
python project_symbiote_infection.py
```

```bash
python project_symbiote_recovery.py
```

The scripts generate plots in the current working directory. The infection script saves degradation and decision-boundary plots; the recovery script saves the recovery curves.

## ◈ WHAT'S HAPPENING UNDER THE HOOD?

The infection experiment adds Gaussian noise to each parameter:

\[
W_{\text{infected}} = W_{\text{healthy}} + \lambda \epsilon
\]

where:

- \(W_{\text{healthy}}\) is the original trained parameter tensor,
- \(\epsilon\) is Gaussian noise,
- \(\lambda\) controls the corruption intensity.

The recovery experiment then fine-tunes the infected model on the original training data.

**Important:** this is a deliberately simple demonstration. It applies parameter noise directly to a toy MLP; it does not establish that the same corruption will affect every architecture or real-world model in the same way.

## ◈ SAFETY / CONTAINMENT

SynapseRavage is intended for models you own or are explicitly authorized to test.

- Keep the original checkpoint unchanged.
- Work on a copy of the model.
- Evaluate changes in a controlled environment.
- Do not alter, disrupt, or deploy modifications to third-party models without explicit authorization.

The scripts in this project demonstrate local parameter perturbation and recovery on a synthetic dataset. They do not connect to or attack remote services.

## ◈ PROJECT STATUS

<pre>
[✓] Healthy model trained
[✓] Gaussian parameter corruption
[✓] Progressive degradation measured
[✓] Decision boundaries visualized
[✓] Fine-tuning recovery experiment
[✓] Healthy / infected / recovered comparison
</pre>

<div align="center">

### 🕸️ END OF TRANSMISSION

**Even a neural network can have a bad symbiote day.**

`SYNAPSERAVAGE // LAB NOTES // 001`

</div>
