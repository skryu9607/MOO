import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import re
import os
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# =========================================================
# CONFIGURATION
# =========================================================
SCENARIO_ID = 8 # Set this to 0, 1, 2, 3, 4, or 5 to match your C++ scenario
# results_filename = 'groundTruth_converted_12.csv' # for scenario 0,1,2

#results_filename = './Apr/groundTruth_converted_4_496_copy.csv' # for scenario 3,4,5

results_filename = './Apr/groundTruth_converted_8.csv' # for scenario 6,7 
#results_filename = 'groundTruth_converted_8_res30.csv' # for scenario 8 

def draw_environment(ax, scenario):

    # Always add boundary (0.0 to 40.0)
    boundary = patches.Rectangle((0.0, 0.0), 40.0, 40.0, 
                                 linewidth=2, edgecolor='black', facecolor='none', zorder=10)
    ax.add_patch(boundary)
    ax.set_xlim(-2, 42)
    ax.set_ylim(-2, 42)

    # Scenarios 3, 4, 5, 6 ,7: Smooth Velocity Change — show actual speed_smooth gradient
    if scenario in [3, 4, 5, 6, 7 , 8 , 9 ]:
        # Compute speed_smooth(y) matching the C++ implementation
        y_arr = np.linspace(0.0, 40.0, 400)
        speed_slow, speed_fast = 2.0, 100.0
        transition_mid, k = 15.0, 0.2
        speed_arr = speed_fast + (1.0 / (1.0 + np.exp(-k * (y_arr - transition_mid)))) * (speed_slow - speed_fast)

        # Draw as a vertical gradient strip on the right side of the workspace
        strip_x0, strip_width = 41.0, 3.0
        speed_img = speed_arr.reshape(-1, 1)
        im = ax.imshow(speed_img, extent=[strip_x0, strip_x0 + strip_width, 0.0, 40.0],
                        origin='lower', aspect='auto', cmap='RdYlGn', vmin=speed_slow, vmax=speed_fast,
                        alpha=0.8, zorder=1)

        # Also shade the workspace background with the same gradient (subtle)
        workspace_img = np.tile(speed_arr.reshape(-1, 1), (1, 2))
        ax.imshow(workspace_img, extent=[0.0, 40.0, 0.0, 40.0],
                  origin='lower', aspect='auto', cmap='RdYlGn', vmin=speed_slow, vmax=speed_fast,
                  alpha=0.10, zorder=0)

        # Reference speed annotations on the strip
        ref_ys = [2.0, 10.0, 15.0, 18.0, 23.0, 30.0, 38.0]
        for ry in ref_ys:
            sp = speed_fast + (1.0 / (1.0 + np.exp(-k * (ry - transition_mid)))) * (speed_slow - speed_fast)
            ax.text(strip_x0 + strip_width + 0.3, ry, f'{sp:.0f}',
                    fontsize=7, va='center', ha='left', color='black')
            ax.plot([strip_x0, strip_x0 + strip_width], [ry, ry],
                    color='black', linewidth=0.3, alpha=0.5)

        ax.text(strip_x0 + strip_width / 2, 41.5, 'Speed', fontsize=8,
                ha='center', fontweight='bold')
        # Mark the transition midpoint
        ax.axhline(y=transition_mid, color='olive', linewidth=1.0, linestyle='--', alpha=0.5, zorder=2)
        ax.text(0.5, transition_mid + 0.5, f'Transition mid (y={transition_mid})',
                color='olive', alpha=0.8, fontsize=8, zorder=2)

        # Widen xlim to fit the speed strip
        ax.set_xlim(-2, strip_x0 + strip_width + 5)

    # Obstacle Mapping based on C++ configureEnvironment
    if scenario == 1:
        # Single Circle
        ax.add_patch(patches.Circle((11.0, 13.0), 3.0, edgecolor='red', facecolor='red', alpha=0.3, zorder=3))
        
    elif scenario == 2:
        # Two Circles
        ax.add_patch(patches.Circle((11.0, 13.0), 3.0, edgecolor='red', facecolor='red', alpha=0.3, zorder=3))
        ax.add_patch(patches.Circle((11.0, 21.0), 2.0, edgecolor='red', facecolor='red', alpha=0.3, zorder=3))
        
    elif scenario == 4:
        # Smooth Velocity Change + Single Circle
        ax.add_patch(patches.Circle((11.0, 13.0), 3.0, edgecolor='red', facecolor='red', alpha=0.3, zorder=3))
        
    elif scenario == 5:
        # Slit obstacles (RectangularObstacle(x_min, x_max, y_min, y_max))
        # matplotlib Rectangle is ((x_min, y_min), width, height)
        # Slit 1: 6.0 to 17.0 (w=11.0), 9.0 to 13.0 (h=4.0)
        ax.add_patch(patches.Rectangle((6.0, 9.0), 11.0, 4.0, 
                                       edgecolor='red', facecolor='red', alpha=0.3, zorder=3))
        # Slit 2: 6.0 to 17.0 (w=11.0), 17.0 to 21.0 (h=4.0)
        ax.add_patch(patches.Rectangle((6.0, 17.0), 11.0, 4.0, 
                                       edgecolor='red', facecolor='red', alpha=0.3, zorder=3))
        # Slit 3: 6.0 to 17.0 (w=11.0), 25.0 to 29.0 (h=4.0)
        ax.add_patch(patches.Rectangle((6.0, 25.0), 11.0, 4.0, 
                                       edgecolor='red', facecolor='red', alpha=0.3, zorder=3))
    elif scenario == 6:
        # Two Gaussian Risk Fields (Circles with gradient)
        ax.add_patch(patches.Circle((11.0, 13.0), 3.0, edgecolor='red', facecolor='red', alpha=0.3, zorder=3))
        ax.add_patch(patches.Circle((11.0, 17.0), 3.0, edgecolor='red', facecolor='red', alpha=0.15, zorder=3))
    elif scenario == 7:
        ax.add_patch(patches.Circle((11.0, 13.0), 3.0, edgecolor='red', facecolor='red', alpha=0.3, zorder=3))
    elif scenario == 8:
        ax.add_patch(patches.Circle((11.0, 13.0), 3.0, edgecolor='red', facecolor='red', alpha=0.3, zorder=3))
        ax.add_patch(patches.Circle((11.0, 17.0), 3.0, edgecolor='red', facecolor='red', alpha=0.3, zorder=3))
    elif scenario == 9:
        ax.add_patch(patches.Circle((11.0, 13.0), 3.0, edgecolor='red', facecolor='red', alpha=0.3, zorder=3))
        ax.add_patch(patches.Circle((11.0, 17.0), 3.0, edgecolor='red', facecolor='red', alpha=0.3, zorder=3))

# =========================================================
# 2. Custom Data Parser
# =========================================================
def parse_raw_data(filename):
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
             pass # Silently skip badly formatted lines

    return pd.DataFrame(data)

# =========================================================
# 3. Convex Hull & Pareto Logic
# =========================================================
def plot_convex_hull_3d(ax, df_local, cols=['Length', 'Risk', 'TravelTime']):
    points = df_local[cols].values
    if len(points) < 4:
        print("Not enough points for Convex Hull (need >= 4).")
        return
    try:
        hull = ConvexHull(points)
        for s in hull.simplices:
            tri = Poly3DCollection([points[s]], alpha=0.1, color='cyan', linewidths=0.5, edgecolors='blue')
            ax.add_collection3d(tri)
        print(f"Convex Hull plotted with {len(hull.simplices)} facets.")
    except Exception as e:
        print(f"Could not plot Convex Hull: {e}")

def is_dominated(p1, p2, objectives):
    is_strictly_better = False
    for obj in objectives:
        if p1[obj] > p2[obj]:
            return False 
        elif p1[obj] < p2[obj]:
            is_strictly_better = True
    return is_strictly_better

def find_pareto_front(df_local):
    objectives = ['Length', 'Risk', 'TravelTime']
    num_solutions = len(df_local)
    is_dominated_flag = [False] * num_solutions
    
    for i in range(num_solutions):
        if is_dominated_flag[i]:
            continue
        for j in range(num_solutions):
            if i == j:
                continue
            if is_dominated(df_local.iloc[j], df_local.iloc[i], objectives):
                is_dominated_flag[i] = True
                break 
            
    df_local['is_pareto'] = ~pd.Series(is_dominated_flag)
    return df_local

# =========================================================
# 4. Load Data & Preprocess
# =========================================================
df = parse_raw_data(results_filename)

if df.empty:
    print("DataFrame is empty. Exiting.")
else:
    df['ID'] = range(len(df)) 
    #df['Risk'] = range(len(df))
    df['Risk'] = np.log10(df['Risk'] / 1.0) 
    #df['TravelTime'] = 0
    #df['Risk'] = 0
    df['Length'] = 0
    #df['Risk'] = df['Risk']/ df['Risk'].max()
    df = find_pareto_front(df)

    df_pareto = df[df['is_pareto']]
    df_dominated = df[~df['is_pareto']]

    utopia_point = {
        'Length': df['Length'].min(),
        'Risk': df['Risk'].min(),
        'TravelTime': df['TravelTime'].min()
    }
    print(f"Utopia Point: Length={utopia_point['Length']:.4f}, Risk={utopia_point['Risk']:.4f}, Time={utopia_point['TravelTime']:.4f}")

    # =========================================================
    # 5. Plotting Configuration
    # =========================================================
    colors = cm.rainbow(np.linspace(0, 1, len(df)))

    # --- Figure 1: Workspace (Path Visualization) ---
    plt.figure(figsize=(14, 10)) 
    
    # Draw configured environment
    draw_environment(plt.gca(), SCENARIO_ID)

    targets = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    selected_indices = []
    
    for idx, row in df.iterrows():
        try:
            w_str = row['Weights'].split(';')
            w_vals = [float(x) for x in w_str]
            for t in targets:
                if np.allclose(w_vals, t, atol=0.0001):
                    selected_indices.append(idx)
                    break
        except:
            pass

    remaining_indices = [i for i in df.index if i not in selected_indices]
    random_sample_count = min(33, len(remaining_indices))
    random_indices = random.sample(remaining_indices, random_sample_count)

    final_plot_indices = sorted(list(set(selected_indices + random_indices)))
    pure_targets = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    for index in final_plot_indices:
        row = df.iloc[index]
        try:
            path_x = [float(x) for x in row['Paths.x'].split(';') if x.strip()]
            path_y = [float(y) for y in row['Paths.y'].split(';') if y.strip()]

            w_vals = [float(x) for x in row['Weights'].split(';')]
            is_pure = any(np.allclose(w_vals, pt, atol=0.0001) for pt in pure_targets)
            
            if is_pure:
                lw, alpha, marker_style, line_style, zorder_val = 3.0, 1.0, 'd', '-', 10
            else:
                lw = 2.0 if row['is_pareto'] else 0.5
                alpha = 1.0 if row['is_pareto'] else 0.5
                marker_style, line_style, zorder_val = 'o', ':', 3

            # Named labels for pure corner weights (legend only, no inline annotations)
            pure_names = {0: 'Pure Dist', 1: 'Pure Risk', 2: 'Pure Time'}
            w_label = None
            for pi, pt in enumerate(pure_targets):
                if np.allclose(w_vals, pt, atol=0.0001):
                    w_label = pure_names[pi]
                    break

            plt.plot(path_x, path_y, marker=marker_style, markersize=3 if is_pure else 2, 
                     linewidth=lw, linestyle=line_style, color=colors[index], 
                     alpha=alpha, zorder=zorder_val,
                     label=w_label)

        except Exception as e:
            print(f"Error parsing path at index {index}: {e}")

    if final_plot_indices:
        first_path_row = df.iloc[final_plot_indices[0]]
        start_x = float(first_path_row['Paths.x'].split(';')[0])
        start_y = float(first_path_row['Paths.y'].split(';')[0])
        goal_x = float(first_path_row['Paths.x'].split(';')[-1])
        goal_y = float(first_path_row['Paths.y'].split(';')[-1])
        plt.scatter(start_x, start_y, c='green', s=100, marker='^', zorder=15, label='Start')
        plt.scatter(goal_x, goal_y, c='red', s=100, marker='*', zorder=15, label='Goal')

    plt.title(f'Figure 1: Selected Paths (Scenario {SCENARIO_ID})')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(loc='upper right', fontsize=9, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout() 

    # --- Figure 2: 3D Cost Space ---
    fig2 = plt.figure(figsize=(10, 10))
    ax = fig2.add_subplot(111, projection='3d') 

    ax.scatter(df_dominated['Length'], df_dominated['Risk'], df_dominated['TravelTime'], 
               s=50, c='lightgray', edgecolors='gray', linewidths=0.5, alpha=0.3)
    ax.scatter(df_pareto['Length'], df_pareto['Risk'], df_pareto['TravelTime'], 
               s=150, c=df_pareto.index.map(lambda i: colors[i]),
               edgecolors='black', linewidths=1.5, alpha=1.0, zorder=3)
    ax.scatter(utopia_point['Length'], utopia_point['Risk'], utopia_point['TravelTime'],
               marker='*', s=200, c='black', edgecolors='black', zorder=6) 

    plot_convex_hull_3d(ax, df_pareto, cols=['Length', 'Risk', 'TravelTime'])

    ax.set_title(f'Figure 2: 3D Objective Space with Pareto Convex Hull (Scenario {SCENARIO_ID})')
    ax.set_xlabel('Length (Min)')
    ax.set_ylabel('Risk (Min)') 
    ax.set_zlabel('Time (Min)')
    ax.view_init(elev=20, azim=-60)

    # --- Figure 3: 2D Projections ---
    fig3, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig3.suptitle(f'Figure 3: 2D Projections (Scenario {SCENARIO_ID})', fontsize=16)
    
    plot_configs = [
        ('Length', 'Risk', 'Length vs. Risk'),
        ('Length', 'TravelTime', 'Length vs. Time'),
        ('Risk', 'TravelTime', 'Risk vs. Time')
    ]

    for ax_sub, (x_col, y_col, title) in zip(axes, plot_configs):
        ax_sub.scatter(df_dominated[x_col], df_dominated[y_col], 
                   s=30, c='lightgray', edgecolors='gray', linewidths=0.5, alpha=0.3)
        ax_sub.scatter(df_pareto[x_col], df_pareto[y_col], 
                   s=100, c=df_pareto.index.map(lambda i: colors[i]),
                   edgecolors='black', linewidths=1.0, alpha=1.0, zorder=3)
        ax_sub.scatter(utopia_point[x_col], utopia_point[y_col], 
                   marker='*', s=200, c='black', edgecolors='black', zorder=4) 

        ax_sub.set_title(title)
        ax_sub.set_xlabel(x_col)
        ax_sub.set_ylabel(y_col)
        ax_sub.grid(True, linestyle='--', alpha=0.7)

    # --- Save Figures ---
    plt.figure(1).savefig(f'Figure1_Path_Visualization_Scenario_{SCENARIO_ID}.png', dpi=300, bbox_inches='tight')
    fig2.savefig(f'Figure2_3D_Cost_Space_Scenario_{SCENARIO_ID}.png', dpi=300, bbox_inches='tight')
    fig3.savefig(f'Figure3_2D_Projections_Scenario_{SCENARIO_ID}.png', dpi=300, bbox_inches='tight')

    print(f"Figures saved for Scenario {SCENARIO_ID}. Showing plots...")
    plt.show()
