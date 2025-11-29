# maf-made-capsule-anomaly
# MAF + MADE for Anomaly Detection on Capsule Dataset

This repository contains the practical implementation of a **Masked Autoregressive Flow (MAF)** with a **MADE** network for density estimation and anomaly detection on a capsule inspection dataset.

The model is trained only on *normal* samples and then used to detect anomalies based on the **negative log-likelihood (NLL)**.

Dataset
The code assumes the capsule dataset is organized as:

data/capsule/
├── train/
│   └── good/             # normal training images
├── test/
│   ├── good/             # normal test images
│   ├── crack/
│   ├── scratch/
│   ├── poke/
│   ├── faulty_imprint/
│   └── squeeze/
└── ground_truth/         # (not used here)

Preprocessing

All images are resized to 64×64.
Converted to RGB and then to tensors in [0, 1].
Flattened to a vector of dimension 64 × 64 × 3 = 12,288.
No additional data augmentation is used.
Labels for anomaly detection
During evaluation, we group test classes as:
label 0 (normal): good
label 1 (anomaly): crack, scratch, poke, faulty_imprint, squeeze

Model
Masked Autoregressive Flow (MAF)
The model implements a standard MAF:
Base distribution: multivariate standard normal N(0, I) in 12,288 dimensions.
Flow: stack of several MAFBlock layers.
Each MAFBlock is a bijective transformation of the form:
z = (x − mean(x)) * exp(−log_scale(x))
where mean and log_scale are predicted by a MADE network.
The model returns log p(x) via change-of-variables:
log p(x) = log p(z) + sum log |det ∂z/∂x|.

MADE network

Each MAFBlock uses a MADE network with:
Input dimension: D = 12,288

Architecture:

Linear(D → 512) → ReLU
Linear(512 → 512) → ReLU
Linear(512 → 2D)  → split into mean and log_scale
Custom binary masks in MaskedLinear enforce the autoregressive structure.
log_scale is clamped to a finite range (e.g. [-5, 5]) for numerical stability.
The whole model is implemented in PyTorch using fully connected layers.

Training

Objective and hyperparameters
We maximize the likelihood of normal samples:
Objective: maximize log p(x) on train/good
→ equivalently minimize average NLL = −log p(x).

Main hyperparameters (reference run):

Image size: 64×64×3
Optimizer: Adam
Learning rate: 1e-3
Batch size: 8
Number of epochs: 100
Number of flow layers: e.g. 7 MAFBlocks

Training time

On a GTX 1050 Ti (4 GB VRAM):
Average time per epoch ≈ 28 seconds
Total training time for 100 epochs ≈ 46–47 minutes

NLL curve

The training notebook logs NLL at each epoch and plots a curve similar to:
Epoch 1: NLL around +8.3e3
NLL decreases monotonically and stabilizes around −4.27e4 by epoch 100
This plot is saved as:
results/nll_curve.png

Sampling

To generate synthetic images from the learned model:
Sample z from the base normal distribution N(0, I).
Apply the inverse of each MAFBlock in reverse order.
Reshape to (3, 64, 64), clamp to [0, 1].

Visualize the images with matplotlib.

Because the inverse pass is autoregressive, sampling is slow:

On the reference run, generating 5 samples took ~900–1300 seconds total,
i.e. around 200–270 seconds per 64×64 RGB image.

Generated samples are saved under results/samples/.

Anomaly detection

After training on train/good, we evaluate on all test classes:
For each test image x, compute log p(x) with the trained flow.
Define anomaly score as NLL(x) = −log p(x).
Visualize histograms of NLL for:
good vs all defect classes combined
(saved as results/nll_hist_good_defect.png).
Compute ROC curve and AUROC:
AUROC in the reference run ≈ 0.785
ROC plot saved as results/roc_curve.png.
Choose a threshold on NLL (e.g. around −4.2×10^4) to separate normal vs anomaly.

Compute:

Confusion matrix
Accuracy
Precision / Recall / F1 for both classes
In the reference experiment:
Overall accuracy ≈ 0.86
For anomalies:

precision ≈ 0.90
recall ≈ 0.93
F1 ≈ 0.91
For normals:

precision ≈ 0.60
recall ≈ 0.52

These numbers show that the model is quite good at catching anomalies, but still misclassifies a non-negligible fraction of normal samples as anomalous.

How to run
Put the dataset in data/capsule/ with the structure described above.
Install the environment.
Start Jupyter:
jupyter notebook
Open:
notebooks/maf_made_capsule.ipynb
Run all cells:
training
sampling
anomaly evaluation
plots and metrics
