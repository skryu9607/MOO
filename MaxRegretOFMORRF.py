import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist

def calculate_spread_and_regret(csv_file):
    # --- 1. Load and Clean Data ---
    df = pd.read_csv(csv_file, skipinitialspace=True)
    
    # Filter for valid numeric data columns
    target_cols = ['Length', 'Risk', 'TravelTime']
    for col in target_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows that were weights (which became NaNs)
    df_clean = df.dropna(subset=target_cols).copy()
    data = df_clean[target_cols].values
    
    if len(data) < 3:
        print("Not enough data points.")
        return

    # --- 2. Normalization (Critical for Regret) ---
    # We normalize by the max value of each objective found in the set
    max_vals = np.max(data, axis=0)
    data_norm = data / max_vals

    # --- 3. Identify Anchors (Ideal Extremes) ---
    # Find the row indices that minimize each objective
    anchor_indices = [np.argmin(data_norm[:, i]) for i in range(3)]
    anchors_norm = data_norm[anchor_indices]

    # --- 4. Calculate Spread (3D) ---
    spread = get_spread_3d(data_norm, anchors_norm)

    # --- 5. Calculate Max Regret (Sampling) ---
    max_regret = get_max_regret_sampling(data_norm, anchors_norm)
    
    print(f"Dataset Size: {len(data)} solutions")
    print(f"Max Values (for normalization): {max_vals}")
    print("-" * 30)
    print(f"Spread Value:     {spread:.4f}")
    print(f"Max Regret Value: {max_regret:.4f}")

def get_spread_3d(points, anchors):
    """Calculates uniformity of the spread."""
    N = len(points)
    
    # Neighbor Distances
    dist_matrix = squareform(pdist(points, metric='euclidean'))
    np.fill_diagonal(dist_matrix, np.inf)
    d_i = np.min(dist_matrix, axis=1)
    d_bar = np.mean(d_i)
    
    # Distance to Extremes (Should be 0 if anchors are in the set)
    dists_to_anchors = cdist(anchors, points, metric='euclidean')
    d_ke_sum = np.sum(np.min(dists_to_anchors, axis=1))

    # Formula
    numerator = d_ke_sum + np.sum(np.abs(d_i - d_bar))
    denominator = d_ke_sum + (N * d_bar)
    
    return numerator / denominator if denominator != 0 else 0

def get_max_regret_sampling(points, anchors, num_samples=100000):
    """Estimates Max Regret by sampling weights."""
    # 1. Generate random weights on simplex (w1+w2+w3=1)
    weights = np.random.dirichlet((1, 1, 1), num_samples)
    
    # 2. Linear Lower Bound P(w)
    # The plane passing through the anchors (Ideal Cost for weight w)
    # P(w) = w dot [min_f1, min_f2, min_f3]
    ideal_vals = np.array([anchors[0,0], anchors[1,1], anchors[2,2]])
    P_w = np.dot(weights, ideal_vals)
    
    # 3. Current Upper Bound
    # For each weight, find the solution in 'points' that gives min cost
    # Cost matrix: (NumSamples, NumPoints)
    all_costs = np.dot(weights, points.T)
    best_costs = np.min(all_costs, axis=1)
    
    # 4. Regret = Best_Found - Ideal
    regrets = best_costs - P_w
    
    return np.max(regrets)

if __name__ == "__main__":
    calculate_spread_and_regret("results_sub_66_iterations_8000.csv")
