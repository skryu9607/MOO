import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def analyze_rps_log(csv_file):
    print(f"Reading log file: {csv_file}...")
    
    try:
        df = pd.read_csv(csv_file, skipinitialspace=True)
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # --- Setup Visualization ---
    fig = plt.figure(figsize=(16, 7))

    # ==========================================
    # --- Plot 1: 3D Cost Space (Pareto Front) ---
    # ==========================================
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Prepare data (Keep Log scale logic consistent)
    x_data = df['f1']
    y_data = np.log10(df['f2'] + 1) # Log scale for Risk
    z_data = df['f3']
    
    # Scatter Plot
    sc = ax1.scatter(x_data, y_data, z_data, c=df['MaxRegret'], cmap='viridis', s=80, edgecolors='k')
    
    # Trace Line
    ax1.plot(x_data, y_data, z_data, color='gray', alpha=0.5, linestyle='--')

    # --- NEW: Add Labels to 3D Points ---
    for i in range(len(df)):
        # We grab the text to display (Iteration number)
        label_text = str(int(df['Iteration'][i]))
        
        # We place the text at the transformed coordinates (x, y, z)
        # We add a tiny offset to Z so the text floats slightly above the dot
        ax1.text(x_data[i], y_data[i], z_data[i] + 0.1, label_text, color='black', fontsize=9)

    ax1.set_xlabel('Distance (f1)')
    ax1.set_ylabel('Log10(Risk) (f2)') 
    ax1.set_zlabel('Time (f3)')
    ax1.set_title('Approximated Pareto Front\n(Labels = Iteration Number)')
    
    cbar = plt.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Max Regret')

    # ==========================================
    # --- Plot 2: 2D Weight Space (Simplex) ---
    # ==========================================
    ax2 = fig.add_subplot(122)
    
    sc2 = ax2.scatter(df['w1'], df['w2'], c=df['MaxRegret'], cmap='plasma', s=80, edgecolors='k')
    
    # Simplex Boundary
    ax2.plot([0, 1], [1, 0], 'r--', linewidth=2, label='Simplex Boundary (w3=0)')
    ax2.plot([0, 0], [0, 1], 'k-', linewidth=1)
    ax2.plot([0, 1], [0, 0], 'k-', linewidth=1)
    
    # Annotate Iteration Numbers (Already in your code, kept for completeness)
    for i, txt in enumerate(df['Iteration']):
        ax2.annotate(str(int(txt)), (df['w1'][i], df['w2'][i]), 
                     xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax2.set_xlabel('Weight 1 (Distance)')
    ax2.set_ylabel('Weight 2 (Risk)')
    ax2.set_title('Search Pattern in Weight Space (Simplex)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #analyze_rps_log("RPS_log_convex_problem.txt")
   #analyze_rps_log("RPS_log_interesting.txt")
    #analyze_rps_log("RPS_log_sharp_risk.txt")
    #analyze_rps_log("RPS_log_100.txt")
    #analyze_rps_log("RPS_log_50.txt")
    analyze_rps_log("RPS_log.txt")
