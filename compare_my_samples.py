"""
Compare your algorithm's sampled points against the labeled ground truth.

Workflow:
  1. Run analyze_groundtruth.py first to produce gt_convexity_labels.csv.
  2. Put your algorithm's sampled points in a CSV with columns
     (Length, Risk, TravelTime), one row per sample.
  3. Run this script.

If your algorithm is linear-regret-minimizing (i.e. equivalent to weighted-sum
scalarization in expectation), the prediction is that almost all samples land
in the *convex* region of the front. Significant samples in the *concave*
region would contradict that theoretical limitation.
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

from pareto_convexity import (
    load_pareto_csv,
    label_ground_truth,
    classify_by_region,
    summarize,
)

GT_PATH      = "groundTruth_converted_8.csv"
SAMPLES_PATH = "my_algorithm_samples.csv"   # <-- change to your sample file


# 1. Load ground truth and recompute its labels (or read from saved CSV)
gt, _ = load_pareto_csv(GT_PATH)
labels0, scores = label_ground_truth(gt, k=10, tau=0.0, normalize=True)
tau = float(np.percentile(np.abs(scores), 25))
gt_labels = np.where(scores >  tau, "convex",
            np.where(scores < -tau, "concave", "linear"))


# 2. Load your samples (assumed to be a simple 3-column CSV: Length, Risk, TravelTime)
samples = []
with open(SAMPLES_PATH, "r") as f:
    reader = csv.reader(f)
    next(reader, None)                       # skip header
    for r in reader:
        if len(r) >= 3:
            samples.append([float(r[0]), float(r[1]), float(r[2])])
samples = np.asarray(samples, float)
print(f"Loaded {len(samples)} samples from {SAMPLES_PATH}")


# 3. Classify each sample by the region of GT it sits closest to.
# This is the right method when samples may be slightly dominated by the front
# (sit inside its envelope) -- they inherit the region label, not be scored
# on their own (possibly misleading) geometric position.
sample_labels = classify_by_region(samples, gt, gt_labels, k=5, normalize=True)
summarize(gt_labels,     "Ground truth")
summarize(sample_labels, "My algorithm")


# 4. Compare proportions
print("\nExpected vs observed:")
for region in ("convex", "concave", "linear"):
    gt_pct  = 100 * (gt_labels     == region).mean()
    smp_pct = 100 * (sample_labels == region).mean()
    print(f"  {region:7s}  GT={gt_pct:5.1f}%   samples={smp_pct:5.1f}%   "
          f"delta={smp_pct - gt_pct:+5.1f} pp")


# 5. Plot side-by-side projections
palette = {"convex": "tab:blue", "concave": "tab:red", "linear": "0.7"}
pairs = [(0, 1, "Length", "Risk"),
         (0, 2, "Length", "TravelTime"),
         (1, 2, "Risk",   "TravelTime")]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (i, j, ni, nj) in zip(axes, pairs):
    for reg, c in palette.items():
        m_gt = gt_labels == reg
        ax.scatter(gt[m_gt, i], gt[m_gt, j], c=c, s=12, alpha=0.4,
                   label=f"GT {reg}")
    for reg, c in palette.items():
        m_s = sample_labels == reg
        ax.scatter(samples[m_s, i], samples[m_s, j], c=c, marker="*", s=120,
                   edgecolors="black", linewidth=0.5, label=f"sample {reg}")
    ax.set_xlabel(ni); ax.set_ylabel(nj)
    ax.set_title(f"{ni} vs {nj}")
axes[0].legend(fontsize=7, loc="upper right")
plt.tight_layout()
plt.savefig("sample_vs_groundtruth.png", dpi=130)
print("\nSaved comparison plot to sample_vs_groundtruth.png")
