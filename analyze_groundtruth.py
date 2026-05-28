"""
Convexity analysis of the 3D Pareto ground truth.

Inputs : groundTruth_converted_8.csv  (Length, Risk, TravelTime + paths + weights)
Outputs: gt_convexity_labels.csv      (one row per GT point, with label + score)
         gt_convexity_diagnostic.png  (score histogram + 3D + 2D projections)
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from pareto_convexity import (
    load_pareto_csv,
    classify_pareto_convexity_lp,
    summarize,
)

CSV_PATH = "./Artificial/concave_pf.csv"
#CSV_PATH = "./Apr/groundTruth_converted_8.csv"
OUT_DIR  = "."

# 1. Load -----------------------------------------------------------------
gt, weights = load_pareto_csv(CSV_PATH)
N, M = gt.shape

labels = classify_pareto_convexity_lp(gt)

print(f"Convex : {np.sum(labels == 'convex')} ")
print(f"Concave : {np.sum(labels == 'concave')} ")

# ----------------------------------------------------

summarize(labels, "Ground truth")

# 4. Save per-point labels -----------------------------------------------
out_csv = f"{OUT_DIR}/gt_convexity_labels.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["idx", "Length", "Risk", "TravelTime",
                 "label",
                "weight_Length", "weight_Risk", "weight_TravelTime"])
    for i in range(N):
        w.writerow([i, gt[i, 0], gt[i, 1], gt[i, 2],
                    f"{labels[i]}", labels[i],
                    weights[i, 0], weights[i, 1], weights[i, 2]])
print(f"Wrote per-point labels to {out_csv}")

# 5. Diagnostic figure ----------------------------------------------------
palette = {"convex": "tab:blue", "concave": "tab:red", "linear": "0.6"}
fig = plt.figure(figsize=(13, 9))

# 5b. 3D scatter
ax3 = fig.add_subplot(2, 3, 2, projection="3d")
for reg, c in palette.items():
    m = labels == reg
    ax3.scatter(gt[m, 0], gt[m, 1], gt[m, 2], c=c, s=10, alpha=0.7, label=reg)
ax3.set_xlabel("Length"); ax3.set_ylabel("Risk"); ax3.set_zlabel("TravelTime")
ax3.set_title("3D Pareto front by region")
ax3.legend(fontsize=8)

# 5c. (text panel)
ax_blank = fig.add_subplot(2, 3, 3)
ax_blank.axis("off")
text = (f"N = {N} GT points\n"
        f"LP tolerance = 1e-4\n\n"
        f"convex : {int((labels=='convex').sum())} "
        f"({100*(labels=='convex').mean():.1f}%)\n"
        f"concave: {int((labels=='concave').sum())} "
        f"({100*(labels=='concave').mean():.1f}%)\n"
        f"linear : {int((labels=='linear').sum())} "
        f"({100*(labels=='linear').mean():.1f}%)")
ax_blank.text(0.05, 0.95, text, va="top", ha="left",
              family="monospace", fontsize=10)

# 5d-f. 2D projections
pairs = [(0, 1, "Length", "Risk"),
         (0, 2, "Length", "TravelTime"),
         (1, 2, "Risk",   "TravelTime")]
for k, (i_obj, j_obj, ni, nj) in enumerate(pairs):
    ax = fig.add_subplot(2, 3, 4 + k)
    for reg, c in palette.items():
        m = labels == reg
        ax.scatter(gt[m, i_obj], gt[m, j_obj], c=c, s=10, alpha=0.65, label=reg)
    ax.set_xlabel(ni); ax.set_ylabel(nj)
    ax.set_title(f"{ni}  vs  {nj}")
    if k == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
out_fig = f"{OUT_DIR}/gt_convexity_diagnostic.png"
plt.savefig(out_fig, dpi=130)
print(f"Saved diagnostic plot to {out_fig}")
