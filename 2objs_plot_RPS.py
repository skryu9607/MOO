import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.spatial.distance import pdist, squareform, cdist

def identify_pareto_efficient(costs):
    """ Finds non-dominated points across the entire provided set. """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # strictly better in at least one, better or equal in all
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

def analyze_rps_log_2d(csv_file):
    print(f"Reading log file: {csv_file}...")
    
    try:
        df = pd.read_csv(csv_file, skipinitialspace=True)
        costs = df[['f1', 'f2']].values
        df['is_pareto'] = identify_pareto_efficient(costs)
        df['is_pareto'] = df['is_pareto'].astype(bool)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # --- 1. Dynamic Label & Scale Parsing ---
    # Expected format: "RPS2D_log_OBJ1_OBJ2.txt" -> parts[-2] is OBJ1, parts[-1] is OBJ2
    filename = os.path.basename(csv_file)
    name_parts = filename.replace(".txt", "").split('_')
    
    # Defaults
    obj1_name = "Objective 1"
    obj2_name = "Objective 2"
    
    # Attempt to extract names if format matches standard
    if len(name_parts) >= 3:
        obj1_name = name_parts[-2].capitalize() # e.g. "Distance"
        obj2_name = name_parts[-1].capitalize() # e.g. "Risk" or "Time"

    # Determine Scaling
    # "Whenever the risk is included that label must be expressed in log scale."
    
    # Setup F1
    if "risk" in obj1_name.lower():
        df['plot_f1'] = np.log10(df['f1']) # Add small epsilon to avoid log(0)
        x_label = f"Log10({obj1_name})"
    else:
        df['plot_f1'] = df['f1']
        x_label = f"{obj1_name} (f1)"

    # Setup F2
    if "risk" in obj2_name.lower():
        df['plot_f2'] = np.log10(df['f2'])
        y_label = f"Log10({obj2_name})"
    else:
        df['plot_f2'] = df['f2']
        y_label = f"{obj2_name} (f2)"

    print(f"Detected Objectives: {obj1_name} vs {obj2_name}")

    # --- 2. Separate Data and Identify Corners ---
    # Explicitly grab corners (-1, -2) and Search Data (>= 0)
    df_corners_raw = df[df['Iteration'].isin([-1, -2])].copy()
    df_search_raw  = df[df['Iteration']>=-3].copy()

    if df_search_raw.empty:
        print("Error: No positive iteration data found.")
        return

    # --- 3. Filter Non-Dominated (Search Data Only) ---
    search_costs = df_search_raw[['plot_f1', 'plot_f2']].values
    mask_search = identify_pareto_efficient(search_costs)
    
    inner_df = df_search_raw[mask_search].copy()
    dominated_df = df_search_raw[~mask_search].copy()
    
    inner_df = inner_df.sort_values(by='f1')

    # --- 4. Process Corners ---
    # We assume -1 and -2 are strictly the corners.
    corners_df = df_corners_raw.copy()
    
    # Prepare Point Arrays
    inner_pts = inner_df[['plot_f1', 'plot_f2']].values
    corners_pts = corners_df[['plot_f1', 'plot_f2']].values
    
    # Combined Set (Inner + Corners) for "Included" metric
    if not corners_df.empty:
        df_combined = pd.concat([inner_df, corners_df], ignore_index=True)
        combined_costs = df_combined[['plot_f1', 'plot_f2']].values
        mask_combined = identify_pareto_efficient(combined_costs)
        all_pareto_df = df_combined[mask_combined].copy()
        all_pts = all_pareto_df[['plot_f1', 'plot_f2']].values
    else:
        all_pts = inner_pts
        print("Warning: No corners (-1, -2) found in file.")

    # ==========================================
    # PRINT NON-DOMINATED VALUES
    # ==========================================
    print(f"\n{'='*60}")
    print(f" NON-DOMINATED SOLUTIONS (Inner Search)")
    print(f"{'='*60}")
    print(f"{'Iter':<6} | {'f1 (' + obj1_name + ')':<15} | {'f2 (' + obj2_name + ')':<15} | {'MaxRegret':<10}")
    print(f"{'-'*6} | {'-'*15} | {'-'*15} | {'-'*10}")
    
    for index, row in inner_df.iterrows():
        iter_val = int(row['Iteration'])
        print(f"{iter_val:<6} | {row['f1']:<15.4f} | {row['f2']:<15.4f} | {row['MaxRegret']:<10.4f}")
    
    if not corners_df.empty:
        print(f"{'-'*60}")
        print(f" ANCHORS (Iterations -1, -2)")
        for index, row in corners_df.iterrows():
            iter_val = int(row['Iteration'])
            print(f"{iter_val:<6} | {row['f1']:<15.4f} | {row['f2']:<15.4f} | {row['MaxRegret']:<10.4f}")
    print(f"{'='*60}\n")

    # ==========================================
    # CALCULATE SPREAD
    # ==========================================
    
    # Case A: Included (Treat the Combined set as self-contained)
    spread_included = calculate_spread(all_pts, reference_corners=None)
    
    # Case B: Missing (Inner points relative to Explicit Corners)
    if len(corners_pts) > 0 and len(inner_pts) > 0:
        spread_missing = calculate_spread(inner_pts, reference_corners=corners_pts)
    else:
        spread_missing = 0.0

    print(f"--- Spread Metric Comparison ---")
    print(f"[Case A] Corners INCLUDED (Combined Front): {spread_included:.5f}")
    print(f"[Case B] Corners MISSING (Inner Only):      {spread_missing:.5f}")

    # ==========================================
    # VISUALIZATION
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- Plot 1: Objective Space ---
    
    # 1. Dominated Points
    ax1.scatter(dominated_df['plot_f1'], dominated_df['plot_f2'], 
                c='lightgray', s=30, alpha=0.4, label='Dominated')

    # 2. Inner Pareto Points
    if not inner_df.empty:
        sc = ax1.scatter(inner_df['plot_f1'], inner_df['plot_f2'], 
                        c=inner_df['MaxRegret'], cmap='viridis', 
                        s=80, edgecolors='k', zorder=3, label='Inner Solutions')
        plt.colorbar(sc, ax=ax1).set_label('Max Regret')

    # 3. Corner Points (Explicit -1, -2)
    if not corners_df.empty:
        ax1.scatter(corners_df['plot_f1'], corners_df['plot_f2'], 
                    c='red', marker='x', s=150, linewidth=2, 
                    zorder=4, label='Anchors (-1, -2)')

    # 4. Connect the dots
    # Use the combined Pareto front (Inner + Corners) to draw the line if available
    if 'all_pareto_df' in locals() and not all_pareto_df.empty:
        line_data = all_pareto_df.sort_values(by='plot_f1')
        ax1.plot(line_data['plot_f1'], line_data['plot_f2'], 'b-', alpha=0.3)
    elif not inner_df.empty:
        line_data = inner_df.sort_values(by='plot_f1')
        ax1.plot(line_data['plot_f1'], line_data['plot_f2'], 'b-', alpha=0.3)

    # 5. Annotations
    for i in range(len(inner_df)):
        ax1.annotate(str(int(inner_df['Iteration'].iloc[i])), 
                     (inner_df['plot_f1'].iloc[i], inner_df['plot_f2'].iloc[i]), 
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_title(f'Pareto Front ({obj1_name} vs {obj2_name})\nIncluded: {spread_included:.4f} | Missing: {spread_missing:.4f}')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # --- Plot 2: Weight Space ---
    
    # Dominated
    ax2.scatter(dominated_df['w1'], dominated_df['w2'], c='lightgray', s=20, alpha=0.3)
    
    # Inner
    if not inner_df.empty:
        ax2.scatter(inner_df['w1'], inner_df['w2'], 
                    c=inner_df['MaxRegret'], cmap='viridis', 
                    s=60, edgecolors='k', label='Inner')
    
    # Corners
    if not corners_df.empty:
        ax2.scatter(corners_df['w1'], corners_df['w2'], 
                    c='red', marker='x', s=100, label='Anchors')
        
    ax2.plot([0, 1], [1, 0], 'r--', label='Simplex Boundary')
    ax2.set_xlabel('Weight 1')
    ax2.set_ylabel('Weight 2')
    ax2.set_title('Search Pattern (Weight Space)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage: Replace with your actual filename
    
    #analyze_rps_log_2d("RPS2D_log_distance_risk.txt")
    #analyze_rps_log_2d("RPS2D_log_distance_time.txt")
    analyze_rps_log_2d("RPS2D_log_risk_time.txt")
