import pandas as pd
import matplotlib.pyplot as plt

filename = 'results.csv'
df = pd.read_csv(filename)
df.columns = df.columns.str.strip()
plt.figure(figsize=(10, 8))

for index, row in df.iterrows():
    
    path_x_str = row['Paths.x'].split(';')
    path_y_str = row['Paths.y'].split(';')

    path_x = [float(x) for x in path_x_str]
    path_y = [float(y) for y in path_y_str]
    
    plt.plot(path_x, path_y, '-o', label=f'Path {index+1}')

plt.title('Path Visualization (10 Paths)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')


print("\nPlot 2: Cost Space...")
plt.figure(figsize=(10, 8)) 

plt.scatter(df['Length'], df['Cost1'], 
            s=100,             
            facecolors='none', 
            edgecolors='blue', 
            linewidths=1.5)    

plt.title('Cost Space Visualization (10 Solutions)')
plt.xlabel('Path Length (Cost 1)')
plt.ylabel('Cost 2')
plt.grid(True)
plt.show()

