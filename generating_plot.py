import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

# ----------------------------------------------------------------------------------
# UTILITY FUNCTION
# ----------------------------------------------------------------------------------
def identify_pareto_efficient(costs):
    """ Finds non-dominated points to plot only the clean Pareto front """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # strictly better in at least one, better or equal in all
            is_dominated = np.any(np.all(costs <= c, axis=1) & np.any(costs < c, axis=1))
            if is_dominated:
                is_efficient[i] = False
    return is_efficient

# ----------------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------------
# Define the datasets to plot. The script will skip any files that are not found.
datasets = [
    {
        'name': 'Ground Truth',
        'file': 'groundTruth_scenario_0.csv',
        'cols': {'f1': 'Length', 'f2': 'Risk', 'f3': 'TravelTime'},
        'color': 'black',
        'marker': '*'
    },
    {
        'name': 'RPS Batch 1',
        'file': 'RPS_log_scenario_0_database.txt',
        'cols': {'f1': 'Cost_Distance', 'f2': 'Cost_Risk', 'f3': 'Cost_Time'},
        'color': 'blue',
        'marker': 'o'
    },
    {
        'name': 'RPS Batch 2',
        'file': 'RPS_log_batch_scenario_0_size_2_database.txt',
        'cols': {'f1': 'Cost_Distance', 'f2': 'Cost_Risk', 'f3': 'Cost_Time'},
        'color': 'green',
        'marker': 's'
    },
    {
        'name': 'RPS Batch 4',
        'file': 'RPS_log_batch_scenario_0_size_4_database.txt',
        'cols': {'f1': 'Cost_Distance', 'f2': 'Cost_Risk', 'f3': 'Cost_Time'},
        'color': 'orange',
        'marker': '^'
    },
    {
        'name': 'RPS Batch 8',
        'file': 'RPS_log_batch_scenario_0_size_8_database.txt',
        'cols': {'f1': 'Cost_Distance', 'f2': 'Cost_Risk', 'f3': 'Cost_Time'},
        'color': 'red',
        'marker': 'D'
    }
]

# ----------------------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------------------
loaded_data = {}

for ds in datasets:
    if os.path.exists(ds['file']):
        try:
            # Skip bad lines and initial spaces in column names
            df = pd.read_csv(ds['file'], on_bad_lines='skip', skipinitialspace=True)
            
            # Dynamically map the columns (ignoring trailing/leading whitespaces)
            col_map = {}
            for target, expected in ds['cols'].items():
                for c in df.columns:
                    if c.strip() == expected:
                        col_map[expected] = c
                        break
            
            # Extract raw objective values
            f1 = df[col_map[ds['cols']['f1']]].values
            f2 = df[col_map[ds['cols']['f2']]].values
            f3 = df[col_map[ds['cols']['f3']]].values
            # Calculate Pareto Efficiency on the dataset to only plot optimal points
            costs = np.column_stack((f1, f2, f3))
            mask = identify_pareto_efficient(costs)
            
            loaded_data[ds['name']] = {
                'f1': f1[mask],
                'f2': f2[mask],
                'f3': f3[mask],
                'color': ds['color'],
                'marker': ds['marker']
            }
            print(f"Loaded {ds['name']}: {sum(mask)} Pareto points out of {len(df)} total.")
        except Exception as e:
            print(f"Failed to process {ds['name']}: {e}")
    else:
        print(f"File not found, skipping: {ds['file']}")

# ----------------------------------------------------------------------------------
# 2. PLOT FIGURE 1: 2D PROJECTIONS (1x3 Panel)
# ----------------------------------------------------------------------------------
fig1, axes = plt.subplots(1, 3, figsize=(18, 5))

for name, data in loaded_data.items():
    # subplot 1: Distance vs Risk (f1 vs f2)
    axes[0].scatter(data['f1'], data['f2'], c=data['color'], marker=data['marker'], label=name, s=60, alpha=0.7)
    # subplot 2: Distance vs Time (f1 vs f3)
    axes[1].scatter(data['f1'], data['f3'], c=data['color'], marker=data['marker'], label=name, s=60, alpha=0.7)
    # subplot 3: Risk vs Time (f2 vs f3)
    axes[2].scatter(data['f2'], data['f3'], c=data['color'], marker=data['marker'], label=name, s=60, alpha=0.7)

# Format Subplot 1 (Risk is often large, using log scale is recommended)
axes[0].set_xlabel('Distance')
axes[0].set_ylabel('Risk (Log Scale)')
axes[0].set_yscale('log')
axes[0].set_title('scenario_0 : Distance vs Risk')
axes[0].legend()

# Format Subplot 2
axes[1].set_xlabel('Distance')
axes[1].set_ylabel('Time')
axes[1].set_title('scenario_0 : Distance vs Time')
axes[1].legend()

# Format Subplot 3
axes[2].set_xlabel('Risk (Log Scale)')
axes[2].set_ylabel('Time')
axes[2].set_xscale('log')
axes[2].set_title('scenario_0 : Risk vs Time')
axes[2].legend()

plt.tight_layout()
fig1.savefig('figure1_2d_projections.png', dpi=300)
print("Saved Figure 1 as 'figure1_2d_projections.png'")

# ----------------------------------------------------------------------------------
# 3. PLOT FIGURE 2: 3D PARETO FRONT
# ----------------------------------------------------------------------------------
fig2 = plt.figure(figsize=(10, 8))
ax3d = fig2.add_subplot(111, projection='3d')

for name, data in loaded_data.items():
    # Note: Applying log10 to Risk (f2) for better visualization in 3D space
    safe_risk = np.clip(data['f2'], 1e-9, None)
    ax3d.scatter(data['f1'], np.log10(safe_risk), data['f3'], 
                 c=data['color'], marker=data['marker'], label=name, s=70, alpha=0.8)

ax3d.set_xlabel('Distance')
ax3d.set_ylabel('Log10(Risk)')
ax3d.set_zlabel('Time')
ax3d.set_title('scenario_0 : 3D Pareto Front Comparison')
ax3d.legend()

# Set an optimal viewing angle
ax3d.view_init(elev=20, azim=-60)

plt.tight_layout()
fig2.savefig('figure2_3d_scatter.png', dpi=300)
print("Saved Figure 2 as 'figure2_3d_scatter.png'")

plt.show()
