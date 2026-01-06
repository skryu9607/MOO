import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
import numpy as np
import os
from scipy.spatial.distance import pdist, squareform, cdist

def calculate_spread_metric_3d(x_data, y_data, z_data, ideal_extremes):
    """
    Calculates Spread (delta) in 3D using Euclidean distance to specific Ideal Extremes (Flags).
    
    Args:
        x_data, y_data, z_data: Arrays of objective values.
        ideal_extremes: List or Array of [f1, f2, f3] coordinates for the anchor points.
    """
    points = np.column_stack((x_data, y_data, z_data))
    N = len(points)
    if N < 2: return 0.0

    # 1. Neighbor Distances (d_i) - 3D Euclidean
    dist_matrix = squareform(pdist(points, metric='euclidean'))
    np.fill_diagonal(dist_matrix, np.inf)
    d_i = np.min(dist_matrix, axis=1)
    d_bar = np.mean(d_i)
    
    # 2. Distance to Extremes (d_k^e)
    # We calculate the distance from each "Ideal Flag" to its closest neighbor in the current set.
    d_ke_sum = 0.0
    
    if ideal_extremes is not None and len(ideal_extremes) > 0:
        ideals_arr = np.array(ideal_extremes)
        # Calculate distances between Ideals (rows) and Current Points (cols)
        dists_to_ideals = cdist(ideals_arr, points, metric='euclidean')
        # For each Ideal, find the closest point in our data
        closest_dists = np.min(dists_to_ideals, axis=1)
        d_ke_sum = np.sum(closest_dists)

    # 3. Final Calculation
    numerator = d_ke_sum + np.sum(np.abs(d_i - d_bar))
    denominator = d_ke_sum + (N * d_bar)
    
    if denominator == 0: return 0.0
    return numerator / denominator

def analyze_rps_log_3d(csv_file):
    print(f"Reading log file: {csv_file}...")
    
    try:
        df = pd.read_csv(csv_file, skipinitialspace=True)
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Check for 3D columns
    required_cols = ['f1', 'f2', 'f3']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV must contain columns {required_cols}")
        return

    # --- 1. Separating Flags (Anchors) from Data ---
    df_flags = df[df['Iteration'] < 0]
    df_data  = df[df['Iteration'] >= 0]
    
    if df_data.empty:
        print("Error: No positive iteration data found to plot.")
        return

    # --- 2. Data Preparation & Log Scaling ---
    # Assuming f1=Dist, f2=Risk, f3=Time. 
    # Logic: Apply log to f2 if filename implies 'risk'. 
    
    x_data = df_data['f1'].values
    y_data_raw = df_data['f2'].values
    z_data = df_data['f3'].values
    
    # Handle Flags (Raw)
    flag_x = df_flags['f1'].values
    flag_y_raw = df_flags['f2'].values
    flag_z = df_flags['f3'].values

    filename = os.path.basename(csv_file)
    y_label = "Objective 2 (f2)"
    
    # Apply Log Scale to Risk (f2) if detected
    if "risk" in filename.lower():
        print("Log scaling applied to f2 (Risk)...")
        y_data = np.log10(y_data_raw + 1)
        flag_y = np.log10(flag_y_raw + 1)
        y_label = "Log10(Risk) (f2)"
    else:
        y_data = y_data_raw
        flag_y = flag_y_raw

    # --- 3. Extract Ideal Extremes for Metric ---
    ideal_extremes = []
    if not df_flags.empty:
        # We treat every negative iteration row as an "Ideal Anchor"
        # Make sure to use the transformed (log) values if applicable
        ideal_extremes = np.column_stack((flag_x, flag_y, flag_z))
    else:
        print("Warning: No negative iteration flags found. d_k^e will be 0.")

    # --- 4. Calculate Spread ---
    spread_val = calculate_spread_metric_3d(x_data, y_data, z_data, ideal_extremes)
    
    print(f"Flags (Anchors) found: {len(ideal_extremes)}")
    print(f"Calculated Spread (delta): {spread_val:.4f}")

    # --- 5. Visualization (3D) ---
    fig = plt.figure(figsize=(16, 7))

    # Subplot 1: 3D Pareto Front
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    # Plot Data
    sc = ax1.scatter(x_data, y_data, z_data, c=df_data['MaxRegret'], cmap='viridis', s=50, edgecolors='k', alpha=0.8)
    
    # Plot Flags (Anchors)
    if len(ideal_extremes) > 0:
        ax1.scatter(flag_x, flag_y, flag_z, c='red', marker='x', s=100, label='Anchors', zorder=10)

    ax1.set_xlabel('Distance (f1)')
    ax1.set_ylabel(y_label)
    ax1.set_zlabel('Time (f3)')
    ax1.set_title(f'3D Pareto Front\nSpread (δ): {spread_val:.4f}')
    ax1.legend()
    
    # Add Colorbar for Max Regret
    plt.colorbar(sc, ax=ax1, fraction=0.03, pad=0.1).set_label('Max Regret')

    # Subplot 2: Weight Space (Projected)
    # Since w1+w2+w3=1, plotting w1 vs w2 is sufficient to show distribution
    ax2 = fig.add_subplot(1, 2, 2)
    sc2 = ax2.scatter(df_data['w1'], df_data['w2'], c=df_data['MaxRegret'], cmap='plasma', edgecolors='k')
    
    # Draw simplex boundary (w1 + w2 <= 1)
    ax2.plot([0, 1], [1, 0], 'r--', label='Simplex Boundary (w3=0)')
    ax2.plot([0, 0], [0, 1], 'k--', alpha=0.3)
    ax2.plot([0, 1], [0, 0], 'k--', alpha=0.3)
    
    ax2.set_xlabel('w1')
    ax2.set_ylabel('w2')
    ax2.set_title('Search Pattern (Projected Weights)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Ensure your log file has columns: Iteration, f1, f2, f3, w1, w2, w3, MaxRegret
    analyze_rps_log_3d("RPS_log.txt")
