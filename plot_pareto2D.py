import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.spatial.distance import pdist, squareform

def calculate_spread_metric(x_data, y_data, ref_bounds_f1, ref_bounds_f2):
    """
    Calculates Spread (delta) using specific Reference Bounds found from 'flagged' rows.
    """
    points = np.column_stack((x_data, y_data))
    N = len(points)
    if N < 2: return 0.0

    # 1. Neighbor Distances (d_i)
    dist_matrix = squareform(pdist(points, metric='euclidean'))
    np.fill_diagonal(dist_matrix, np.inf)
    d_i = np.min(dist_matrix, axis=1)
    d_bar = np.mean(d_i)
    
    # 2. Distance to Extremes (d_k^e)
    d_ke_sum = 0.0
    
    # Current Search Extremes
    curr_min_f1, curr_max_f1 = np.min(x_data), np.max(x_data)
    curr_min_f2, curr_max_f2 = np.min(y_data), np.max(y_data)
    
    # Reference (Flagged) Extremes
    ref_min_f1, ref_max_f1 = ref_bounds_f1
    ref_min_f2, ref_max_f2 = ref_bounds_f2

    # Calculate Gaps
    # Note: We take absolute difference. Even if current result is "better" (smaller) 
    # due to stochastic noise, the gap is just the distance between them.
    if ref_min_f1 is not None: d_ke_sum += abs(curr_min_f1 - ref_min_f1)
    if ref_max_f1 is not None: d_ke_sum += abs(curr_max_f1 - ref_max_f1)
    
    if ref_min_f2 is not None: d_ke_sum += abs(curr_min_f2 - ref_min_f2)
    if ref_max_f2 is not None: d_ke_sum += abs(curr_max_f2 - ref_max_f2)

    # 3. Final Calculation
    numerator = d_ke_sum + np.sum(np.abs(d_i - d_bar))
    denominator = d_ke_sum + (N * d_bar)
    
    if denominator == 0: return 0.0
    return numerator / denominator

def analyze_rps_log_2d(csv_file):
    print(f"Reading log file: {csv_file}...")
    
    try:
        df = pd.read_csv(csv_file, skipinitialspace=True)
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # --- 1. Separating Flags (Anchors) from Data ---
    # Flags are negative iterations (-2, -1, etc.)
    df_flags = df[df['Iteration'] < 0]
    df_data  = df[df['Iteration'] >= 0]
    
    if df_data.empty:
        print("Error: No positive iteration data found to plot.")
        return

    # --- 2. Extract Reference Bounds from Flags ---
    # We assume:
    #   - One flag minimizes f1 (Best Distance)
    #   - One flag minimizes f2 (Best Risk)
    
    # Get raw values from flags
    flag_f1_values = df_flags['f1'].values
    flag_f2_values = df_flags['f2'].values
    
    # Determine bounds based on the Flags provided
    # If flags exist, we use their min/max as the "Ideal" range
    if not df_flags.empty:
        # We assume the flags represent the extreme limits found during initialization
        ref_min_f1 = np.min(flag_f1_values)
        ref_max_f1 = np.max(flag_f1_values) # The other anchor usually has the worst f1
        
        ref_min_f2 = np.min(flag_f2_values)
        ref_max_f2 = np.max(flag_f2_values) # The other anchor usually has the worst f2
    else:
        # Fallback if no negative flags exist
        ref_min_f1, ref_max_f1 = None, None
        ref_min_f2, ref_max_f2 = None, None
        print("Warning: No negative iteration flags found. d_k^e will be 0.")

    # --- 3. Prepare Data for Plotting & Metric ---
    # Use only the search data (Iteration >= 0)
    x_data = df_data['f1'].values
    y_data_raw = df_data['f2'].values
    
    # Check naming for Labels/Log Scale
    filename = os.path.basename(csv_file)
    if "risk" in filename.lower():
        # Apply Log Scale to Data AND Reference Bounds
        y_data = np.log10(y_data_raw + 1)
        ref_min_f2 = np.log10(ref_min_f2 + 1) if ref_min_f2 is not None else None
        ref_max_f2 = np.log10(ref_max_f2 + 1) if ref_max_f2 is not None else None
        y_label = "Log10(Risk) (f2)"
    else:
        y_data = y_data_raw
        y_label = "Objective 2 (f2)"

    # --- 4. Calculate Spread ---
    spread_val = calculate_spread_metric(
        x_data, y_data, 
        ref_bounds_f1=(ref_min_f1, ref_max_f1), 
        ref_bounds_f2=(ref_min_f2, ref_max_f2)
    )
    
    print(f"Reference Bounds Used:")
    print(f"  f1: {ref_min_f1:.4f} to {ref_max_f1:.4f}")
    print(f"  f2: {ref_min_f2:.4f} to {ref_max_f2:.4f}")
    print(f"Calculated Spread (delta): {spread_val:.4f}")

    # --- 5. Visualization ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Sort for line plot
    sort_idx = np.argsort(x_data)
    
    # Plot Data
    ax1.plot(x_data[sort_idx], y_data[sort_idx], 'b-', alpha=0.5, label='Pareto Front')
    sc = ax1.scatter(x_data, y_data, c=df_data['MaxRegret'], cmap='viridis', s=80, edgecolors='k', zorder=3)
    
    # Plot the Flags (Anchors) as Red X's to see them
    if not df_flags.empty:
        # Transform flag y-data if log scale
        if "risk" in filename.lower():
            flag_y_plot = np.log10(flag_f2_values + 1)
        else:
            flag_y_plot = flag_f2_values
            
        ax1.scatter(flag_f1_values, flag_y_plot, c='red', marker='x', s=100, label='Anchors (Flags)', zorder=4)

    # Annotations
    for i in range(len(df_data)):
        ax1.annotate(str(int(df_data['Iteration'].iloc[i])), 
                     (x_data[i], y_data[i]), xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax1.set_xlabel('Distance (f1)')
    ax1.set_ylabel(y_label)
    ax1.set_title(f'Pareto Front\nSpread (δ): {spread_val:.4f}')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    plt.colorbar(sc, ax=ax1).set_label('Max Regret')

    # Weight Space (Standard)
    ax2.scatter(df_data['w1'], df_data['w2'], c=df_data['MaxRegret'], cmap='plasma', edgecolors='k')
    ax2.plot([0, 1], [1, 0], 'r--', label='Simplex')
    ax2.set_xlabel('w1'); ax2.set_ylabel('w2')
    ax2.set_title('Search Pattern')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_rps_log_2d("RPS2D_log_distance_time.txt")
