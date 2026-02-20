import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import re
import os
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

# [NEW] Imports for Convex Hull
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
# [NEW] Convex Hull Function
# ---------------------------------------------------------
def plot_convex_hull_3d(ax, df, cols=['Length', 'Risk', 'TravelTime']):
    """
    Computes and plots the 3D Convex Hull of the points in df.
    """
    points = df[cols].values
    
    # Convex Hull requires at least 4 points in 3D (tetrahedron)
    if len(points) < 4:
        print("Not enough points for Convex Hull (need >= 4).")
        return

    try:
        hull = ConvexHull(points)
        
        # Plot the facets (triangles)
        for s in hull.simplices:
            # Create a triangle from the simplex points
            tri = Poly3DCollection([points[s]], alpha=0.1, color='cyan', linewidths=0.5, edgecolors='blue')
            ax.add_collection3d(tri)
            
        print(f"Convex Hull plotted with {len(hull.simplices)} facets.")
        
    except Exception as e:
        print(f"Could not plot Convex Hull: {e}")

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

# Ensure this matches your data file name
results_filename = 'groundTruth_converted_0.csv'
#results_filename = 'groundTruth_converted_1.csv'
# results_filename = 'groundTruth_converted_2.csv'

df = parse_raw_data(results_filename)

if df.empty:
    print("DataFrame is empty. Exiting.")
else:
    df['ID'] = range(len(df)) 
    # Note: Log scale on Risk might distort Convex Hull visual linearity, but valid for topological analysis
    df['Risk'] = np.log10(df['Risk'] / 1.0) 
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
    # 4. Plotting Configuration
    # ---------------------------------------------------------
    colors = cm.rainbow(np.linspace(0, 1, len(df)))

    # ---------------------------------------------------------
    # Figure 1: Workspace (Path Visualization)
    # ---------------------------------------------------------
    plt.figure(figsize=(14, 10)) 

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

    targets = [
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
    ]

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
    random_sample_count = min(43, len(remaining_indices))
    random_indices = random.sample(remaining_indices, random_sample_count)

    final_plot_indices = selected_indices + random_indices
    final_plot_indices = sorted(list(set(final_plot_indices)))

    pure_targets = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    for index in final_plot_indices:
        row = df.iloc[index]
        try:
            path_x = [float(x) for x in row['Paths.x'].split(';') if x.strip()]
            path_y = [float(y) for y in row['Paths.y'].split(';') if y.strip()]

            w_vals = [float(x) for x in row['Weights'].split(';')]
            is_pure = False
            for pt in pure_targets:
                if np.allclose(w_vals, pt, atol=0.0001):
                    is_pure = True
                    break
            
            if is_pure:
                lw = 3.0      
                alpha = 1.0   
                marker_style = 'd' 
                line_style = ':'
                zorder_val = 10 
            else:
                lw = 2.0 if row['is_pareto'] else 0.5
                alpha = 1.0 if row['is_pareto'] else 0.5
                marker_style = 'o'
                line_style = ':'
                zorder_val = 3

            plt.plot(path_x, path_y,
                     marker=marker_style,
                     markersize=6 if is_pure else 2, 
                     linewidth=lw,
                     linestyle=line_style,
                     color=colors[index], 
                     alpha=alpha,
                     zorder=zorder_val) 

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

    plt.title('Figure 1: Selected Paths with Obstacle')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
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

    # [NEW] Plot Convex Hull ONLY on Pareto Front
    # We pass df_pareto instead of df to draw hull only on Pareto points
    plot_convex_hull_3d(ax, df_pareto, cols=['Length', 'Risk', 'TravelTime'])

    ax.set_title('Figure 2: 3D Objective Space with Pareto Convex Hull')
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

        # # Only Index ID on points
        # for i, row in df.iterrows():
        #     if row['is_pareto'] or i in final_plot_indices:
        #         ax_sub.text(row[x_col], row[y_col], 
        #                 f"P{row['ID']}", 
        #                 fontsize=8, 
        #                 ha='right', 
        #                 va='bottom', 
        #                 color='black',
        #                 zorder=5)

        ax_sub.set_title(title)
        ax_sub.set_xlabel(x_col)
        ax_sub.set_ylabel(y_col)
        ax_sub.grid(True, linestyle='--', alpha=0.7)

    # ---------------------------------------------------------
    #  Save Figures
    # ---------------------------------------------------------
    plt.figure(1).savefig('Figure1_Path_Visualization.png', dpi=300, bbox_inches='tight')
    fig2.savefig('Figure2_3D_Cost_Space.png', dpi=300, bbox_inches='tight')
    fig3.savefig('Figure3_2D_Projections.png', dpi=300, bbox_inches='tight')

    print("Figures saved. showing plots...")
    plt.show()
