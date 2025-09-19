import numpy as np



class Node: 
    def __init__(self, x,y,z,parent, id = None):
        self.x = x
        self.y = y
        self.z = z
        self.id = id
        self.parent = parent
    def get_coordinates(self):
        return np.array([self.x,self.y,self.z])
    def __repr__(self):
        return f"Node(id={self.id}, x={self.x}, y={self.y}, z={self.z})"
    def distnace_to(self, other_node):
        return np.linalg.norm(self.get_coordinates() - other_node.get_coordinates())
    def update_parent(self, new_parent):
        self.parent = new_parent
    
