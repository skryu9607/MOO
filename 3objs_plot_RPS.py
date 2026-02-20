import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import argparse
from scipy.spatial.distance import pdist, squareform

# ----------------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ----------------------------------------------------------------------------------

def identify_pareto_efficient(costs):
    """ Finds non-dominated points (Pareto Front). """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # strictly less in at least one, less or equal in all
            is_dominated = np.any(np.all(costs <= c, axis=1) & np.any(costs < c, axis=1))
            if is_dominated:
                is_efficient[i] = False
    return is_efficient

def calculate_spread(inner_points, corner_points):
    """ Calculates Generalized Spread (Delta). """
    if len(corner_points) > 0:
        all_pareto = np.vstack((corner_points, inner_points)) if len(inner_points) > 0 else corner_points
    else:
        all_pareto = inner_points

    N = len(all_pareto)
    if N < 2: return 0.0

    dist_matrix = squareform(pdist(all_pareto, metric='euclidean'))
    np.fill_diagonal(dist_matrix, np.inf)
    d_i = np.min(dist_matrix, axis=1)
    d_bar = np.mean(d_i)
    
    sum_deviation = np.sum(np.abs(d_i - d_bar))
    sum_d_e = 0.0 
    
    delta = (sum_d_e + sum_deviation) / (sum_d_e + N * d_bar)
    return delta

def load_data_robust(filename):
    """ Robust data loader. """
    try:
        df = pd.read_csv(filename, on_bad_lines='skip')
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

# ----------------------------------------------------------------------------------
# MAIN SCRIPT
# ----------------------------------------------------------------------------------




def main():
    parser = argparse.ArgumentParser(description='Plot 3D Pareto Front with Unified Weight Triangles')
    parser.add_argument('filename', type=str, nargs='?', default='RPS_log_batch_scenario_0_size_2.txt')
    #parser.add_argument('filename', type=str, nargs='?', default='RPS_log_scenario_1.txt')
    
    args = parser.parse_args()

    if not os.path.exists(args.filename):
        print(f"Error: File '{args.filename}' not found.")
        return

    # 1. Load Data
    df = load_data_robust(args.filename)
    if df is None or df.empty:
        return

    print(f"Processing File: {args.filename} ({len(df)} rows)")

    required_cols = ['f1', 'f2', 'f3', 'w1', 'w2', 'MaxRegret']
    has_parents = all(c in df.columns for c in ['wd', 'wr', 'wt'])
    
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing columns. Needed: {required_cols}")
        return

    # Log Transform Risk (f2)
    if (df['f2'] <= 0).any():
        df['f2'] = df['f2'].clip(lower=1e-9)
    df['f2_log'] = np.log10(df['f2']) 

    # 2. Identify Pareto Front
    costs = df[['f1', 'f2_log', 'f3']].values
    mask = identify_pareto_efficient(costs)
    df['is_pareto'] = mask
    
    # 3. Separate Data for Metrics
    pareto_df = df[mask].copy()
    initial_indices = [0, 1, 2]
    corners_df = pareto_df[pareto_df.index.isin(initial_indices)]
    inner_pareto_df = pareto_df[~pareto_df.index.isin(initial_indices)].sort_values(by='f1')

    # Calculate Spread
    spread_metric = calculate_spread(inner_pareto_df[['f1','f2_log','f3']].values, 
                                     corners_df[['f1','f2_log','f3']].values)
    print(f"Spread Metric: {spread_metric:.5f}")

    # ==========================================
    # PLOTTING
    # ==========================================
    fig = plt.figure(figsize=(18, 8))

    # --- PLOT 1: 3D Objective Space ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    # Dominated Points
    dominated_df = df[~mask]
    ax1.scatter(dominated_df['f1'], dominated_df['f2_log'], dominated_df['f3'], 
                c='lightgray', s=20, alpha=0.3, label='Dominated')

    # Pareto Points (Colored by MaxRegret)
    if not pareto_df.empty:
        p_scatter = ax1.scatter(pareto_df['f1'], pareto_df['f2_log'], pareto_df['f3'], 
                            c=pareto_df['MaxRegret'], cmap='viridis', s=60, edgecolors='k', label='Pareto Front')
        cbar = fig.colorbar(p_scatter, ax=ax1, shrink=0.5, label='Max Regret')
    
    # Highlight Corners
    if not corners_df.empty:
        ax1.scatter(corners_df['f1'], corners_df['f2_log'], corners_df['f3'], 
                    s=150, facecolors='none', edgecolors='red', linewidth=2, label='Corners')

    # Axis Setup
    ax1.set_xlabel('Distance (f1)')
    ax1.set_ylabel('Log Risk (f2)')
    ax1.set_zlabel('Time (f3)')
    ax1.view_init(elev=20, azim=-60)
    ax1.set_xlim(df['f1'].min(), df['f1'].max())
    ax1.set_title(f'Pareto Front (Color = Max Regret)\nSpread: {spread_metric:.4f}')
    ax1.legend(loc='upper left')

    # --- PLOT 2: Weight Space with Unified Triangles ---
    ax2 = fig.add_subplot(1, 2, 2)

    # Background: All weights
    ax2.scatter(df['w1'], df['w2'], c='lightgray', s=10, alpha=0.1)

    triangle_count = 0

    if has_parents:
        # Loop through EVERY row with valid parents
        for idx, row in df.iterrows():
            try:
                if pd.isna(row['wd']):
                    continue

                p_ids = [int(row['wd']), int(row['wr']), int(row['wt'])]
                
                if max(p_ids) >= len(df):
                    continue

                # Retrieve Parent Weights
                parents = df.loc[p_ids]
                
                if len(parents) == 3:
                    pts = parents[['w1', 'w2']].values
                    pts = np.vstack([pts, pts[0]]) # Close loop
                    
                    # UNIFIED TRIANGLE STYLE
                    # All triangles are black, thin, and semi-transparent
                    # We do NOT distinguish between Pareto and Dominated triangles here
                    ax2.plot(pts[:,0], pts[:,1], linestyle='-', color='black', 
                             linewidth=0.5, alpha=0.2, zorder=1)
                    
                    # Draw Child Point (Distinguish Points Only)
                    color = 'blue' if row['is_pareto'] else 'gray'
                    size = 30 if row['is_pareto'] else 15
                    z = 3 if row['is_pareto'] else 2
                    
                    ax2.scatter(row['w1'], row['w2'], color=color, s=size, alpha=0.8, zorder=z)
                    
                    triangle_count += 1

            except Exception:
                continue

    print(f"Successfully drew {triangle_count} sampling triangles.")

    # Explicitly highlight corners
    if not corners_df.empty:
        ax2.scatter(corners_df['w1'], corners_df['w2'], c='red', marker='x', s=100, zorder=10, label='Initial Corners')

    # Boundaries
    ax2.plot([0, 1], [1, 0], 'r--', alpha=0.5, label='Simplex Boundary')
    
    ax2.set_xlabel('Weight 1 (w1)')
    ax2.set_ylabel('Weight 2 (w2)')
    ax2.set_title(f'Weight Sampling History)')
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # Legend - Simplified
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='black', lw=1, alpha=0.5, label='Sampling Triangle'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Pareto Sample'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='Dominated Sample'),
                    Line2D([0], [0], marker='x', color='r', markersize=10, label='Corners', linestyle='None')]
    ax2.legend(handles=custom_lines, loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
