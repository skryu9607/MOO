import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

# =============================
# Utility: Pareto Front Checker
# =============================
def is_pareto_front(points):
    """
    Determine which points belong to the Pareto front.
    Each row in `points` represents a solution vector (e.g., [Length, Cost1]).
    Lower values are assumed to be better in all dimensions.

    Returns:
        pareto_mask: Boolean array, True if the point is non-dominated.
    """
    points = np.array(points)
    n_points = points.shape[0]
    pareto_mask = np.ones(n_points, dtype=bool)

    for i in range(n_points):
        if pareto_mask[i]:
            # Any point that dominates point i will cause i to be removed
            dominates = np.all(points <= points[i], axis=1) & np.any(points < points[i], axis=1)
            pareto_mask[dominates] = False
            pareto_mask[i] = True  # Keep the current point as True
    return pareto_mask


# =============================
# Load RRT* Tree structure
# =============================
tree_filename = 'final_trees_data_subproblem_0.txt'
tree_nodes = {}
tree_edges = []

line_regex = re.compile(
    r'\s*(\d+)\s*\(([^,]+),\s*([^\)]+)\)\s*->\s*(\d+)\s*\(([^,]+),\s*([^\)]+)\)'
)

try:
    with open(tree_filename, 'r') as f:
        lines = f.readlines()
        for line in lines[3:]:
            match = line_regex.search(line)
            if match:
                child_id = int(match.group(1))
                child_x = float(match.group(2))
                child_y = float(match.group(3))
                parent_id = int(match.group(4))
                parent_x = float(match.group(5))
                parent_y = float(match.group(6))

                tree_nodes[child_id] = (child_x, child_y)
                tree_nodes[parent_id] = (parent_x, parent_y)

                edge = ([parent_x, child_x], [parent_y, child_y])
                tree_edges.append(edge)

except FileNotFoundError:
    print(f"Error: '{tree_filename}' not found.")
except Exception as e:
    print(f"Error: {e}")


# =============================
# Load results data
# =============================
results_filename = 'results.csv'
try:
    df = pd.read_csv(results_filename)
    df.columns = df.columns.str.strip()
except FileNotFoundError:
    print(f"Error: '{results_filename}' not found.")
    exit()


# =============================
# Draw RRT* Tree
# =============================
plt.figure(figsize=(12, 10))

# Plot all tree edges
if tree_edges:
    for edge in tree_edges:
        plt.plot(edge[0], edge[1],
                 color='grey', linestyle='-', linewidth=1.0,
                 alpha=0.6, zorder=1)

# Plot nodes
if tree_nodes:
    for node_id, (x, y) in tree_nodes.items():
        plt.scatter(x, y, s=50, color='gray', alpha=0.6, zorder=1)
        plt.text(x, y - 0.1, str(node_id),
                 fontsize=7, color='black', alpha=0.7,
                 ha='center', va='top', zorder=2)

# Plot candidate paths from results.csv
for index, row in df.iterrows():
    path_x = [float(x) for x in row['Paths.x'].split(';')]
    path_y = [float(y) for y in row['Paths.y'].split(';')]
    plt.plot(path_x, path_y, marker='o', markersize=5, linewidth=2.0,
             label=f'Path {index + 1}', zorder=3)

plt.title('Path Visualization with RRT* Tree')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')


# =============================
# Pareto Front Calculation
# =============================
# Assume we want to minimize both 'Length' and 'Cost1'
points = df[['Length', 'Cost1']].values
pareto_mask = is_pareto_front(points)

# Add Pareto front info to dataframe
df['ParetoFront'] = pareto_mask


# =============================
# Cost Space Visualization
# =============================
plt.figure(figsize=(10, 8))

# Non-dominated (Pareto front) points in red
plt.scatter(df.loc[df['ParetoFront'], 'Length'],
            df.loc[df['ParetoFront'], 'Cost1'],
            s=120, facecolors='none', edgecolors='red', linewidths=2,
            label='Pareto Front')

# Dominated points in blue
plt.scatter(df.loc[~df['ParetoFront'], 'Length'],
            df.loc[~df['ParetoFront'], 'Cost1'],
            s=100, facecolors='none', edgecolors='blue', linewidths=1.5,
            label='Dominated')

for i, row in df.iterrows():
    plt.text(row['Length'], row['Cost1'] + 0.2, f"{i+1}", fontsize=9, ha='center')

plt.title('Cost Space Visualization with Pareto Front')
plt.xlabel('Path Length')
plt.ylabel('Travel Time (Cost1)')
plt.legend()
plt.grid(True)
plt.show()
