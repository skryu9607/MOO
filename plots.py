import pandas as pd
import matplotlib.pyplot as plt
import re 


tree_filename = 'final_trees_data_subproblem_0.txt'

tree_nodes = {} 
# [ ([parent_x, child_x], [parent_y, child_y]) ]
tree_edges = [] 

#  Child ID (x, y) -> Parent ID (x, y) 
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
    print(f"Error: '{tree_filename}'.")
except Exception as e:
    print(f"{e}")




results_filename = 'results.csv'
try:
    df = pd.read_csv(results_filename)
    df.columns = df.columns.str.strip()
    
except FileNotFoundError:
    exit()




plt.figure(figsize=(12, 10)) 


if tree_edges:

    for edge in tree_edges:
        # edge[0] = [parent_x, child_x]
        # edge[1] = [parent_y, child_y]
        plt.plot(edge[0], edge[1],
                 color='grey',       
                 linestyle='-',      
                 linewidth=1.0,      
                 alpha=0.6,        
                 zorder=1)          


if tree_nodes:

    for node_id, (x, y) in tree_nodes.items():

        plt.scatter(x, y, 
                    s=50,             
                    color='gray', 
                    alpha=0.6, 
                    zorder=1)         
        
  
        plt.text(x, y - 0.1, str(node_id), 
                 fontsize=7,         
                 color='black', 
                 alpha=0.7, 
                 ha='center',        
                 va='top',           
                 zorder=2)           
for index, row in df.iterrows():
    
    path_x_str = row['Paths.x'].split(';')
    path_y_str = row['Paths.y'].split(';')

    path_x = [float(x) for x in path_x_str]
    path_y = [float(y) for y in path_y_str]
    
    plt.plot(path_x, path_y,
             marker='o',             
             markersize=5,           
             linewidth=2.0,          
             label=f'Path {index+1}',
             zorder=3)             

plt.title('Path Visualization with RRT* Tree ')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')



plt.figure(figsize=(10, 8)) 

plt.scatter(df['Length'], df['Cost1'], 
            s=100,             
            facecolors='none', 
            edgecolors='blue', 
            linewidths=1.5)    

plt.title('Cost Space Visualization ')
plt.xlabel('Path Length (Cost 1)')
plt.ylabel('Cost 2')
plt.grid(True)
plt.show()

