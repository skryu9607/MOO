import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import argparse
from scipy.spatial.distance import pdist, squareform, cdist

def identify_pareto_efficient(costs):
    """ Finds non-dominated points. """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_dominated = np.any(np.all(costs <= c, axis=1) & np.any(costs < c, axis=1))
            if is_dominated:
                is_efficient[i] = False
    return is_efficient

def calculate_spread(pareto_points, reference_corners=None):
    """
    Calculates Generalized Spread (Delta).
    Formula: Delta = (Sum(d_e) + Sum(|d_i - d_bar|)) / (Sum(d_e) + N * d_bar)
    """
    N = len(pareto_points)
    if N < 2: return 0.0

    # 1. Neighbor Distances (d_i)
    dist_matrix = squareform(pdist(pareto_points, metric='euclidean'))
    np.fill_diagonal(dist_matrix, np.inf)
    d_i = np.min(dist_matrix, axis=1)
    d_bar = np.mean(d_i)
    
    # 2. Deviation (Uniformity)
    sum_deviation = np.sum(np.abs(d_i - d_bar))
    
    # 3. Extent (d_e)
    sum_d_e = 0.0
    if reference_corners is not None and len(reference_corners) > 0:
        dists = cdist(reference_corners, pareto_points, metric='euclidean')
        sum_d_e = np.sum(np.min(dists, axis=1))

    # 4. Final Metric
    delta = (sum_d_e + sum_deviation) / (sum_d_e + N * d_bar)
    return delta

def main():
    parser = argparse.ArgumentParser(description='Plot 3D Pareto Front with Spread Metrics')
    parser.add_argument('filename', type=str, nargs='?', default='RPS_log_scenario_0.txt')
    args = parser.parse_args()

    if not os.path.exists(args.filename):
        print(f"Error: File '{args.filename}' not found.")
        return

    # 1. Load Data
    try:
        df = pd.read_csv(args.filename, on_bad_lines='skip')
        df.columns = [c.strip() for c in df.columns]
    except Exception as e:
        print(f"Error: {e}")
        return

    if not all(col in df.columns for col in ['f1', 'f2', 'f3']):
        print("Error: Columns f1, f2, f3 required.")
        return

    # 2. Filter Non-Dominated Solutions
    costs = df[['f1', 'f2', 'f3']].values
    mask = identify_pareto_efficient(costs)
    pareto_df = df[mask].copy()
    
    # Sort by f1 (Distance) for cleaner output
    pareto_df = pareto_df.sort_values(by='f1')

    # ==========================================
    # PRINT NON-DOMINATED VALUES
    # ==========================================
    print(f"\n{'='*60}")
    print(f" NON-DOMINATED SOLUTIONS (Pareto Front)")
    print(f"{'='*60}")
    print(f"{'Iter':<6} | {'f1 (Dist)':<12} | {'f2 (Risk)':<12} | {'f3 (Time)':<12}")
    print(f"{'-'*6} | {'-'*12} | {'-'*12} | {'-'*12}")
    
    for index, row in pareto_df.iterrows():
        iter_val = int(row['Iteration']) if 'Iteration' in row else index
        print(f"{iter_val:<6} | {row['f1']:<12.4f} | {row['f2']:<12.4f} | {row['f3']:<12.4f}")
    print(f"{'='*60}\n")

    # 3. Identify Corners vs Inner
    tol = 1e-4
    is_corner = (pareto_df['w1'] > 1.0-tol) | \
                (pareto_df['w2'] > 1.0-tol) | \
                ((pareto_df['w1'] < tol) & (pareto_df['w2'] < tol)) 
    
    corners_df = pareto_df[is_corner]
    inner_df = pareto_df[~is_corner]
    
    all_pts = pareto_df[['f1','f2','f3']].values
    inner_pts = inner_df[['f1','f2','f3']].values
    corners_pts = corners_df[['f1','f2','f3']].values

    # ==========================================
    # CALCULATE SPREAD
    # ==========================================
    
    # Case A: Included
    spread_included = calculate_spread(all_pts, reference_corners=None)
    
    # Case B: Missing
    if len(corners_pts) > 0 and len(inner_pts) > 1:
        spread_missing = calculate_spread(inner_pts, reference_corners=corners_pts)
    else:
        spread_missing = 0.0

    print(f"--- Spread Metric Comparison ---")
    print(f"[Case A] Corners INCLUDED (Full Set):   {spread_included:.5f}")
    print(f"[Case B] Corners MISSING (Inner Only):  {spread_missing:.5f}")

    # ==========================================
    # PLOTTING
    # ==========================================
    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    # Dominated
    dominated_df = df[~mask]
    ax1.scatter(dominated_df['f1'], dominated_df['f2'], dominated_df['f3'], 
                c='lightgray', s=15, alpha=0.3, label='Dominated')

    # Inner Points
    sc = ax1.scatter(inner_df['f1'], inner_df['f2'], inner_df['f3'], 
                     c='blue', s=60, edgecolors='k', label='Inner Solutions')
    
    # Corners
    ax1.scatter(corners_df['f1'], corners_df['f2'], corners_df['f3'], 
                c='red', marker='x', s=150, linewidth=2, label='True Corners')

    ax1.set_xlabel('Distance')
    ax1.set_ylabel('Risk')
    ax1.set_zlabel('Time')
    ax1.set_title(f'Pareto Front\nIncluded: {spread_included:.4f} | Missing: {spread_missing:.4f}')
    ax1.legend()
    ax1.view_init(elev=30, azim=45)

    # Weights
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(dominated_df['w1'], dominated_df['w2'], c='lightgray', s=15, alpha=0.3)
    ax2.scatter(inner_df['w1'], inner_df['w2'], c='blue', s=50, edgecolors='k', label='Inner')
    ax2.scatter(corners_df['w1'], corners_df['w2'], c='red', marker='x', s=100, label='Corners')
    
    ax2.plot([0, 1], [1, 0], 'r--', label='Boundary')
    ax2.plot([0, 0], [0, 1], 'k--', alpha=0.2)
    ax2.plot([0, 1], [0, 0], 'k--', alpha=0.2)
    ax2.set_xlabel('Weight 1')
    ax2.set_ylabel('Weight 2')
    ax2.set_title('Weight Space')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
