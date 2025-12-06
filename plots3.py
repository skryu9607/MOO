import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import re
import os
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

# ---------------------------------------------------------
# 1. Custom Data Parser
# ---------------------------------------------------------
def parse_raw_data(filename):
    """
    Reads the custom two-line record format and returns a DataFrame without Pareto analysis.
    The format expects one line for metrics/path data and the next line for Weights.
    """
    data = []
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: '{filename}' not found.")
        return pd.DataFrame()

    if len(lines) > 0:
        lines = lines[1:] # Skip header line if present

    for i in range(0, len(lines), 2):
        if i + 1 >= len(lines):
            continue

        metric_line = lines[i].strip()
        weight_line = lines[i+1].strip().strip('"')

        if not metric_line or not weight_line:
            continue
            
        # Regex: Length, Risk, TravelTime, "Paths.x", "Paths.y", Fitness
        match = re.match(r'^([^,]+),([^,]+),([^,]+),"([^"]+)","([^"]+)",([^,]+)', metric_line)
        
        if match:
            try:
                record = {
                    "Length": float(match.group(1)),
                    "Risk": float(match.group(2)),
                    "TravelTime": float(match.group(3)),
                    "Paths.x": match.group(4),
                    "Paths.y": match.group(5),
                    "Fitness": float(match.group(6)),
                    "Weights": weight_line,
                }
                data.append(record)
            except ValueError as e:
                 print(f"Skipping line due to numerical error ({e}): {metric_line}")
        else:
             print(f"Skipping badly formatted metric line: {metric_line}")

    return pd.DataFrame(data)

# ---------------------------------------------------------
# 2. Pareto Front Identification Logic
# ---------------------------------------------------------

def is_dominated(p1, p2, objectives):
    is_strictly_better = False
    for obj in objectives:
        if p1[obj] > p2[obj]:
            return False 
        elif p1[obj] < p2[obj]:
            is_strictly_better = True
    return is_strictly_better

def find_pareto_front(df):
    objectives = ['Length', 'Risk', 'TravelTime']
    num_solutions = len(df)
    is_dominated_flag = [False] * num_solutions
    
    for i in range(num_solutions):
        if is_dominated_flag[i]:
            continue
        for j in range(num_solutions):
            if i == j:
                continue
            p_i = df.iloc[i]
            p_j = df.iloc[j]
            if is_dominated(p_j, p_i, objectives):
                is_dominated_flag[i] = True
                break 
            
    df['is_pareto'] = ~pd.Series(is_dominated_flag)
    return df

# ---------------------------------------------------------
# 3. Load Data & Preprocess
# ---------------------------------------------------------

#results_filename = 'results_sub_66_iterations_8000.csv'
#results_filename = 'results_sub_231_iterations_10000.csv'
#results_filename = 'results_sub_21_iterations_6000.csv'
results_filename = 'results.csv'
#results_filename = 'results_sub_496_iterations_20000.csv'
#results_filename = 'results_sub_496_iterations_20000_finer.csv'
#results_filename = 'results_nonzero_iterations_80000.csv'
#results_filename = 'results_sub_496_iterations'
df = parse_raw_data(results_filename)

if df.empty:
    print("DataFrame is empty. Exiting.")
    exit()

df['ID'] = range(len(df)) 
df['Risk'] = df['Risk'] / 1.0
df = find_pareto_front(df)

df_pareto = df[df['is_pareto']]
df_dominated = df[~df['is_pareto']]

# Utopia Point
utopia_point = {
    'Length': df['Length'].min(),
    'Risk': df['Risk'].min(),
    'TravelTime': df['TravelTime'].min()
}
print(f"Utopia Point: Length={utopia_point['Length']:.4f}, Risk={utopia_point['Risk']:.4f}, Time={utopia_point['TravelTime']:.4f}")
# ---------------------------------------------------------
# Sequential (pairwise) cost vector comparison
# ---------------------------------------------------------

def compare_cost_vectors(df, weight_list, atol=1e-4):
    """
    df: parsed DataFrame
    weight_list: list of weight vectors in order (e.g.,
                 [[0,0,1],[0,0.01,0.99],[0,0.02,0.98],...])
    Performs pairwise comparison: w[i] → w[i+1]
    """

    def parse_w(s):
        return [float(x) for x in s.split(";")]

    def find_idx(w):
        matches = df.index[df["Weights"].apply(lambda s: np.allclose(parse_w(s), w, atol=atol))]
        if len(matches) == 0:
            raise ValueError(f"Weight {w} not found!")
        return matches[0]

    def cost_vec(row):
        return np.array([row["Length"], row["Risk"], row["TravelTime"]], dtype=float)

    print("\n==============================================")
    print(" SEQUENTIAL COST VECTOR COMPARISON REPORT")
    print("==============================================\n")

    for i in range(len(weight_list) - 1):
        w_from = weight_list[i]
        w_to   = weight_list[i+1]

        idx_from = find_idx(w_from)
        idx_to   = find_idx(w_to)

        c_from = cost_vec(df.iloc[idx_from])
        c_to   = cost_vec(df.iloc[idx_to])

        delta = c_to - c_from
        norm_delta = np.linalg.norm(delta)

        print(f"From Weight {w_from} (Index {idx_from})")
        print(f"To   Weight {w_to}   (Index {idx_to})")
        print(f"Cost(from) = {c_from}")
        print(f"Cost(to)   = {c_to}")
        print(f"ΔCost      = {delta}")
        print(f"‖ΔCost‖    = {norm_delta:.6f}")
        print("----------------------------------------------")

    print("\n==============================================\n")

# ---------------------------------------------------------
# Request 3: Create ID-Weight Mapping Table
# ---------------------------------------------------------
mapping_file = "ID_Weight_Mapping.csv"
df[['ID', 'Weights']].to_csv(mapping_file, index=False)
print(f"\n[Info] ID matching table saved to '{mapping_file}'")

# ---------------------------------------------------------
# 4. Plotting Configuration
# ---------------------------------------------------------
colors = cm.rainbow(np.linspace(0, 1, len(df)))

# ---------------------------------------------------------
# Figure 1: Workspace (Path Visualization) - Selected Only
# ---------------------------------------------------------
plt.figure(figsize=(14, 10)) 

# [NEW] Obstacle Plotting
obstacle_center = (11, 13)
obstacle_radius = 3.0
circle = patches.Circle(obstacle_center, obstacle_radius, 
                        edgecolor='red', 
                        facecolor='red', 
                        alpha=0.3, 
                        zorder=1, 
                        label='Obstacle')
plt.gca().add_patch(circle)
plt.scatter(obstacle_center[0], obstacle_center[1], marker='x', color='red', s=100, zorder=2) 


# Filter Logic for Trajectories
# Target Weights (Approximate matching needed due to float strings)
# targets = [
#     [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
#     [0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5],
#     [1/3, 1/3, 1/3]
# ]
targets = [
    [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
]

selected_indices = []
found_targets = []

# 1. Find Target Weights
for idx, row in df.iterrows():
    # Parse string weights "w1;w2;w3" to list of floats
    w_str = row['Weights'].split(';')
    w_vals = [float(x) for x in w_str]
    
    # Check if this matches any target (allow small epsilon error)
    for t in targets:
        if np.allclose(w_vals, t, atol=0.0001):
            selected_indices.append(idx)
            found_targets.append(idx)
            break

# 2. Pick Random 43 from the rest
remaining_indices = [i for i in df.index if i not in selected_indices]
random_sample_count = min(43, len(remaining_indices))
random_indices = random.sample(remaining_indices, random_sample_count)

final_plot_indices = selected_indices + random_indices
final_plot_indices = sorted(list(set(final_plot_indices))) # Sort and remove duplicates

print(f"[Info] Figure 1 will plot {len(final_plot_indices) + 1} paths (Targets: {len(found_targets)}, Random: {len(random_indices)})")

# --- Plot Paths (Only Selected) ---
pure_targets = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

for index in final_plot_indices:
    row = df.iloc[index]
    try:
        path_x = [float(x) for x in row['Paths.x'].split(';') if x.strip()]
        path_y = [float(y) for y in row['Paths.y'].split(';') if y.strip()]

        # Weights string for label
        weights_str = row['Weights'].replace(';', ', ')
        path_label = f"w=[{weights_str}]"
        
        # Parse weights for checking
        w_vals = [float(x) for x in row['Weights'].split(';')]

        # [MODIFIED] Special styling for pure weights
        is_pure = False
        for pt in pure_targets:
            if np.allclose(w_vals, pt, atol=0.0001):
                is_pure = True
                break
        
        if is_pure:
            # 단일 목적함수(Pure weights)인 경우: 진하게, 다른 마커
            lw = 3.0      # 더 두껍게
            alpha = 1.0   # 불투명
            marker_style = 'd' # Diamond marker (기본 'o'와 다르게)
            line_style = ':'
            zorder_val = 10 # 맨 위에 그리기
        else:
            # 일반 경로
            lw = 2.0 if row['is_pareto'] else 0.5
            alpha = 1.0 if row['is_pareto'] else 0.5
            marker_style = 'o' # Circle
            line_style = ':'
            zorder_val = 3

        plt.plot(path_x, path_y,
                 marker=marker_style,
                 markersize=6 if is_pure else 2, # Pure weights는 마커도 조금 더 크게
                 linewidth=lw,
                 linestyle=line_style,
                 color=colors[index], 
                 label=path_label,
                 alpha=alpha,
                 zorder=zorder_val) 

    except Exception as e:
        print(f"Error parsing path at index {index}: {e}")

# Start/Goal (Plot after paths to ensure visibility)
if final_plot_indices:
    first_path_row = df.iloc[final_plot_indices[0]]
    start_x = float(first_path_row['Paths.x'].split(';')[0])
    start_y = float(first_path_row['Paths.y'].split(';')[0])
    goal_x = float(first_path_row['Paths.x'].split(';')[-1])
    goal_y = float(first_path_row['Paths.y'].split(';')[-1])

    plt.scatter(start_x, start_y, c='green', s=100, marker='^', zorder=15, label='Start')
    plt.scatter(goal_x, goal_y, c='red', s=100, marker='*', zorder=15, label='Goal')


plt.title('Figure 1: Selected Paths with Obstacle')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
# 범례 위치 조정 
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-small', title="Weights") 
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout() 


# ---------------------------------------------------------
# Figure 2: 3D Cost Space (Objective Space)
# ---------------------------------------------------------
fig2 = plt.figure(figsize=(10, 10))
ax = fig2.add_subplot(111, projection='3d') 

# Plot dominated
ax.scatter(df_dominated['Length'], df_dominated['Risk'], df_dominated['TravelTime'], 
           s=50, c='lightgray', edgecolors='gray', linewidths=0.5, alpha=0.3)

# Plot Pareto
ax.scatter(df_pareto['Length'], df_pareto['Risk'], df_pareto['TravelTime'], 
           s=150, 
           c=df_pareto.index.map(lambda i: colors[i]),
           edgecolors='black', 
           linewidths=1.5,
           alpha=1.0,
           zorder=3)

# Utopia Point
ax.scatter(utopia_point['Length'], utopia_point['Risk'], utopia_point['TravelTime'],
           marker='*', s=200, c='black', edgecolors='black', zorder=6) 

# # Request 2: Only Index ID on points, Remove Legend
# for i, row in df.iterrows():
#     if row['is_pareto'] or i in final_plot_indices:
#         ax.text(row['Length'], row['Risk'], row['TravelTime'], 
#                 f"P{row['ID']}",  # Only ID
#                 fontsize=9, 
#                 ha='left', 
#                 va='bottom', 
#                 fontweight='bold', 
#                 color='black', 
#                 zorder=10 
#                 )
ax.set_title('Figure 2: 3D Objective Space (ID Only)')
ax.set_xlabel('Length (Min)')
ax.set_ylabel('Risk (Min)') 
ax.set_zlabel('Time (Min)')
ax.view_init(elev=20, azim=-60)


# ---------------------------------------------------------
# Figure 3: 2D Projection Plots (3 Subplots)
# ---------------------------------------------------------
fig3, axes = plt.subplots(1, 3, figsize=(18, 6))

plot_configs = [
    ('Length', 'Risk', 'Length vs. Risk'),
    ('Length', 'TravelTime', 'Length vs. Time'),
    ('Risk', 'TravelTime', 'Risk vs. Time')
]

for ax_sub, (x_col, y_col, title) in zip(axes, plot_configs):
    # Dominated
    ax_sub.scatter(df_dominated[x_col], df_dominated[y_col], 
               s=30, c='lightgray', edgecolors='gray', linewidths=0.5, alpha=0.3)
    
    # Pareto
    ax_sub.scatter(df_pareto[x_col], df_pareto[y_col], 
               s=100, 
               c=df_pareto.index.map(lambda i: colors[i]),
               edgecolors='black', 
               linewidths=1.0,
               alpha=1.0,
               zorder=3)
    
    # Utopia
    ax_sub.scatter(utopia_point[x_col], utopia_point[y_col], 
               marker='*', s=200, c='black', edgecolors='black', zorder=4) 

    # Request 4: Only Index ID on points, Remove Legend
    for i, row in df.iterrows():
        if row['is_pareto'] or i in final_plot_indices:
            ax_sub.text(row[x_col], row[y_col], 
                    f"P{row['ID']}", 
                    fontsize=8, 
                    ha='right', 
                    va='bottom', 
                    color='black',
                    zorder=5)

    ax_sub.set_title(title)
    ax_sub.set_xlabel(x_col)
    ax_sub.set_ylabel(y_col)
    ax_sub.grid(True, linestyle='--', alpha=0.7)
df = parse_raw_data(results_filename)
df['ID'] = range(len(df))
df = find_pareto_front(df)
compare_cost_vectors(
    df,          # baseline
    # weight_list=[
    #     [0.00, 0.00, 1.00],
    #     [0.00, 0.01, 0.99],
    #     [0.00, 0.02, 0.98],
    #     [0.00, 0.03, 0.97],             
    # ]
    # weight_list = [[0.00,1.00,0.00],
    #                [0.01,0.99,0.00],
    #                [0.02,0.98,0.00],
    #                [0.03,0.97,0.00]]
    weight_list=[[1.00,0.00,0.00],
                 [0.99,0.00,0.01],
                 [0.98,0.00,0.02],
                 [0.97,0.00,0.03]]
)    

# ---------------------------------------------------------
## ✨ Save Figures
# ---------------------------------------------------------
plt.figure(1).savefig('Figure1_Path_Visualization.png', dpi=300, bbox_inches='tight')
fig2.savefig('Figure2_3D_Cost_Space.png', dpi=300, bbox_inches='tight')
fig3.savefig('Figure3_2D_Projections.png', dpi=300, bbox_inches='tight')

print("Figures saved. showing plots...")
plt.show()


