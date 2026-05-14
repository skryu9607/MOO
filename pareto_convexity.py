"""
Pareto Front Convexity / Concavity Detection
=============================================
"""
import csv
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# =============================================================================
# CSV loader for the "Length, Risk, TravelTime, Paths.x, Paths.y, Cost, Weight"
# format where Weight wraps to its own physical row.
# =============================================================================
def load_pareto_csv(path, obj_cols=(0, 1, 2)):
    """
    Load a Pareto CSV that either:
      1) spans two physical rows (objectives on row 1, weights on row 2)
      2) spans a single row (objectives first, weights at the end)

    Returns
    -------
    obj     : (N, len(obj_cols)) ndarray of objective values
    weights : (N, 3) ndarray of scalarization weights, NaN if absent
    """
    with open(path, "r") as f:
        rows = list(csv.reader(f))

    obj_rows, w_rows = [], []
    i = 1  # skip header
    while i < len(rows):
        r = rows[i]
        
        # Skip empty rows
        if not r:
            i += 1
            continue

        # Ensure the row has enough columns to extract objectives
        if len(r) >= max(obj_cols) + 1:
            try:
                objs = [float(r[c]) for c in obj_cols]
            except ValueError:
                i += 1
                continue
                
            # Case 1: Weights are in the NEXT row (e.g., groundTruth_converted_8_res10.csv)
            if i + 1 < len(rows) and len(rows[i + 1]) == 1 and ";" in rows[i + 1][0]:
                weights = [float(x) for x in rows[i + 1][0].split(";")]
                obj_rows.append(objs)
                w_rows.append(weights)
                i += 2
            
            # Case 2: Weights are in the SAME row (e.g., convex_pf.csv)
            else:
                try:
                    # Attempt to parse the last 3 columns as weights
                    weights = [float(x) for x in r[-3:]]
                    obj_rows.append(objs)
                    w_rows.append(weights)
                except ValueError:
                    # If they aren't floats (e.g., missing or strings), default to NaNs
                    obj_rows.append(objs)
                    w_rows.append([np.nan] * 3)
                i += 1
        else:
            i += 1
            
    return np.asarray(obj_rows, float), np.asarray(w_rows, float)
# =============================================================================
# Min-max normalization helper
# =============================================================================
def _minmax(arr, lo, hi):
    span = hi - lo
    span = np.where(span < 1e-12, 1.0, span)
    return (arr - lo) / span

# =============================================================================
# 2D method: signed cross product of consecutive edges (exact, very robust)
# =============================================================================
def _label_2d(gt, tau, ideal):
    """
    Sort by f1 ascending, then for each interior triple of points compute
    cross = (p_i - p_{i-1}) x (p_{i+1} - p_i), normalized by edge norms.

    For a Pareto front going from upper-left to lower-right with ideal at the
    lower-left corner, the cross product is POSITIVE where the curve bulges
    toward the ideal (convex) and NEGATIVE where it bulges away (concave).
    """
    N = len(gt)
    order = np.argsort(gt[:, 0])
    s = gt[order]

    scores = np.zeros(N)
    labels = np.full(N, "linear", dtype=object)

    for j in range(1, N - 1):
        v1 = s[j]     - s[j - 1]
        v2 = s[j + 1] - s[j]
        n1n2 = np.linalg.norm(v1) * np.linalg.norm(v2)
        if n1n2 < 1e-12:
            continue
        scores[order[j]] = (v1[0] * v2[1] - v1[1] * v2[0]) / n1n2

    # Orient so that positive = "toward ideal" regardless of how the data
    # is sorted in objective space.
    extremes = s[[0, -1]]
    chord_mid = extremes.mean(axis=0)
    front_centroid = s.mean(axis=0)
    if np.linalg.norm(front_centroid - ideal) > np.linalg.norm(chord_mid - ideal):
        scores = -scores

    labels[scores >  tau] = "convex"
    labels[scores < -tau] = "concave"
    return labels, scores


# =============================================================================
# General M-D method: kNN + SVD local hyperplane + curvature normalization
# =============================================================================
def _curv_score(point, neighbors, ideal):
    c        = neighbors.mean(axis=0)
    centered = neighbors - c
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    normal   = Vt[-1]                       # normal to best-fit hyperplane
    if np.dot(ideal - c, normal) < 0:       # orient toward ideal
        normal = -normal
    signed_dist = float(np.dot(point - c, normal))
    var = float(np.mean(np.sum(centered ** 2, axis=1)))
    return 0.0 if var < 1e-12 else 2.0 * signed_dist / var


def _label_md(gt, k, tau, ideal):
    N      = len(gt)
    tree   = cKDTree(gt)
    labels = np.full(N, "linear", dtype=object)
    scores = np.zeros(N)
    for i, p in enumerate(gt):
        _, idx = tree.query(p, k=min(k + 1, N))
        nbrs   = gt[idx[1:]]
        s      = _curv_score(p, nbrs, ideal)
        scores[i] = s
        if s >  tau: labels[i] = "convex"
        if s < -tau: labels[i] = "concave"
    return labels, scores


# =============================================================================
# Public API
# =============================================================================
import numpy as np
from scipy.optimize import linprog

def classify_pareto_convexity_lp(gt):

    N, M = gt.shape
    
    labels = np.full(N, "convex", dtype=object)
    
    ptp = np.ptp(gt, axis=0)
    ptp[ptp < 1e-12] = 1.0
    norm_gt = (gt - np.min(gt, axis=0)) / ptp
    
    for i in range(N):
        # N-1 lambda_j variables for the convex combination, and 1 variable for t
        c = np.zeros(N)
        # last one, t, which is to be minized. 
        c[-1] = 1.0  
        
        other_gt = np.delete(norm_gt, i, axis=0)
        # A_ub  sum(lambda_j * x_{j,m}) - t <= x_{i,m}
        A_ub = np.zeros((M, N))
        A_ub[:, :-1] = other_gt.T
        A_ub[:, -1] = -1.0
        
        b_ub = norm_gt[i]
        
        A_eq = np.zeros((1, N))
        A_eq[0, :-1] = 1.0
        b_eq = np.array([1.0])
        
        bounds = [(0, None)] * (N - 1) + [(None, None)]
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if res.success:
            t = res.x[-1]
            if t > 0:
                labels[i] = "convex"
            else:
                labels[i] = "concave"
            
    return labels


def classify_against_truth(points, gt, k=None, tau=None, ideal=None, normalize=False):
    """
    Classify candidate points using the local geometry of the ground truth.

    For M = 2: each candidate point is inserted into the sorted ground-truth
    sequence and given the cross-product score of its surrounding triple.
    For M >= 3: SVD test on its kNN in the ground truth.

     this scores the candidate's own position relative to the local
    hyperplane. If the candidate is dominated by the front (sits "inside"
    the envelope, away from the ideal), the sign will skew negative
    regardless of local region. For dominated candidates, prefer
    `classify_by_region`, which inherits the GT region label instead.
    """
    points = np.asarray(points, dtype=float)
    gt     = np.asarray(gt,     dtype=float)

    if normalize:
        lo, hi = gt.min(axis=0), gt.max(axis=0)
        gt     = _minmax(gt,     lo, hi)
        points = _minmax(points, lo, hi)
        ideal_eff = np.zeros(gt.shape[1]) if ideal is None else _minmax(np.asarray(ideal, float), lo, hi)
    else:
        ideal_eff = gt.min(axis=0) if ideal is None else np.asarray(ideal, float)

    _, M   = gt.shape

    if M == 2:
        tau = 0.003 if tau is None else tau
        order = np.argsort(gt[:, 0])
        s_gt  = gt[order]
        extremes = s_gt[[0, -1]]
        chord_mid = extremes.mean(axis=0)
        flip = (np.linalg.norm(s_gt.mean(axis=0) - ideal_eff) >
                np.linalg.norm(chord_mid - ideal_eff))

        f1 = s_gt[:, 0]
        labels = np.full(len(points), "linear", dtype=object)
        scores = np.zeros(len(points))
        for i, p in enumerate(points):
            j = int(np.searchsorted(f1, p[0]))
            if 0 <= j < len(s_gt) and np.allclose(s_gt[j], p):
                if j == 0 or j == len(s_gt) - 1:
                    continue
                p_left, p_right = s_gt[j - 1], s_gt[j + 1]
            else:
                if j == 0 or j >= len(s_gt):
                    continue
                p_left, p_right = s_gt[j - 1], s_gt[j]
            v1 = p       - p_left
            v2 = p_right - p
            n1n2 = np.linalg.norm(v1) * np.linalg.norm(v2)
            if n1n2 < 1e-12:
                continue
            sc = (v1[0] * v2[1] - v1[1] * v2[0]) / n1n2
            if flip:
                sc = -sc
            scores[i] = sc
            if sc >  tau: labels[i] = "convex"
            if sc < -tau: labels[i] = "concave"
        return labels, scores

    # M >= 3 -------------------------------------------------------------------
    k   = max(M + 1, 4) if k is None else k
    tau = 0.05 if tau is None else tau
    tree = cKDTree(gt)
    labels = np.full(len(points), "linear", dtype=object)
    scores = np.zeros(len(points))
    for i, p in enumerate(points):
        _, idx = tree.query(p, k=min(k, len(gt)))
        nbrs   = gt[idx]
        sc     = _curv_score(p, nbrs, ideal_eff)
        scores[i] = sc
        if sc >  tau: labels[i] = "convex"
        if sc < -tau: labels[i] = "concave"
    return labels, scores


def classify_by_region(points, gt, gt_labels, k=5, normalize=False):
    """
    For each candidate point, look up the k nearest ground-truth points and
    assign the majority region label among them. This is the right method
    when candidates may be dominated by the front (sit inside its envelope):
    the candidate inherits the *region* it projects into rather than being
    scored on its own (possibly misleading) geometric position.

    Parameters
    ----------
    points    : (N', M) candidate points
    gt        : (N,  M) ground-truth points
    gt_labels : (N,) array of labels already produced by `label_ground_truth`
    k         : int, number of GT neighbors to vote (default 5)
    normalize : bool, rescale objectives to [0,1] using gt min/max for the
                kNN distance computation.

    Returns
    -------
    labels : (N',) array of 'convex' | 'concave' | 'linear'
    """
    points = np.asarray(points, dtype=float)
    gt     = np.asarray(gt,     dtype=float)
    gt_labels = np.asarray(gt_labels)

    if normalize:
        lo, hi = gt.min(axis=0), gt.max(axis=0)
        gt     = _minmax(gt,     lo, hi)
        points = _minmax(points, lo, hi)

    tree = cKDTree(gt)
    out  = np.empty(len(points), dtype=object)
    for i, p in enumerate(points):
        _, idx = tree.query(p, k=min(k, len(gt)))
        nbr_labels = gt_labels[idx]
        # majority vote
        vals, counts = np.unique(nbr_labels, return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


def summarize(labels, name=""):
    n = len(labels)
    counts = {k: int(np.sum(labels == k)) for k in ("convex", "concave", "linear")}
    print(f"--- {name} ({n} points) ---" if name else f"--- {n} points ---")
    for k, v in counts.items():
        pct = 100.0 * v / n if n else 0.0
        print(f"  {k:7s}: {v:4d}  ({pct:5.1f}%)")
    return counts

if __name__ == "__main__":


    rng = np.random.default_rng(0)

    # Convex arc toward origin: (0,2) -> (1,1)
    th1 = np.linspace(0, np.pi / 2, 80)
    x1, y1 = 1 - np.cos(th1), 2 - np.sin(th1)
    # Concave arc away from origin: (1,1) -> (2,0)
    th2 = np.linspace(0, np.pi / 2, 80)
    x2, y2 = 1 + np.sin(th2), np.cos(th2)
    gt = np.vstack([np.column_stack([x1, y1]),
                    np.column_stack([x2, y2])])

    conv_pool = np.where(gt[:, 0] <= 1.0)[0]
    baseline  = gt[rng.choice(conv_pool, size=15, replace=False)]
    mine      = gt[rng.choice(len(gt),   size=25, replace=False)]

    gt_lbl, _ = label_ground_truth(gt)
    b_lbl,  _ = classify_against_truth(baseline, gt)
    m_lbl,  _ = classify_against_truth(mine,     gt)

    summarize(gt_lbl, "Ground truth")
    summarize(b_lbl,  "Baseline")
    summarize(m_lbl,  "My algorithm")

    palette = {"convex": "tab:blue", "concave": "tab:red", "linear": "0.6"}
    fig, ax = plt.subplots(figsize=(7, 6))
    for region, color in palette.items():
        m = gt_lbl == region
        ax.scatter(gt[m, 0], gt[m, 1], c=color, s=18, alpha=0.55,
                   label=f"GT {region}")
    ax.scatter(baseline[:, 0], baseline[:, 1], marker="x", c="black",
               s=80, linewidth=2, label="Baseline")
    ax.scatter(mine[:, 0], mine[:, 1], marker="*", c="gold",
               edgecolors="black", s=160, label="My algorithm")
    ax.set_xlabel("f1"); ax.set_ylabel("f2")
    ax.set_title("Pareto-front concavity check (synthetic 2D)")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig("/home/claude/convexity_demo.png", dpi=120)
    print("Saved demo plot to convexity_demo.png")
