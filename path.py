import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

try:
    paths_data = pd.read_csv('pareto_paths.csv')
except FileNotFoundError:
    print("Error: pareto_paths.csv not found.")
    exit()

start_pos = (0, 0)
goal_pos = (90, 90)

plt.figure(figsize=(10, 10))

path_ids = paths_data['path_id'].unique()
colors = cm.rainbow(np.linspace(0, 1, len(path_ids)))

for path_id, color in zip(path_ids, colors):
    path = paths_data[paths_data['path_id'] == path_id]
    plt.plot(path['x'], path['y'], color=color, alpha=0.8, linewidth=2, label=f'Path {path_id}')

plt.scatter(start_pos[0], start_pos[1], c='green', s=200, marker='o', label='Start', zorder=5)
plt.scatter(goal_pos[0], goal_pos[1], c='red', s=200, marker='X', label='Goal', zorder=5)


plt.title('Pareto Optimal Paths')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.axis('equal') #
plt.legend()
plt.show()
