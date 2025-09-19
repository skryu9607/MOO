import numpy as np
import math
import random
import time
import os
import psutil
import concurrent.futures
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# A constant for infinity, useful for initial costs.
INF = float('inf')

class Vertex:
    """
    Represents a vertex (or node) in the configuration space.
    For the unicycle, the state is (x, y, theta).
    """
    def __init__(self, state):
        self.state = np.array(state, dtype=np.float64)

    @property
    def position(self):
        # For compatibility with older functions, return only (x, y)
        return self.state[:2]
        
    def __repr__(self):
        return f"Vertex({self.state})"

class Tree:
    """
    Represents a single tree in the forest.
    It stores the parent of each vertex, the cost to reach it,
    and the trajectory segment from its parent.
    """
    def __init__(self, initial_vertex, num_objectives):
        self.parents = {initial_vertex: None}
        self.costs = {initial_vertex: np.zeros(num_objectives)}
        self.trajectories = {initial_vertex: [initial_vertex.state]}

    def get_full_path_to_vertex(self, vertex):
        """
        Reconstructs the entire path (a list of states) from the root to a given vertex.
        """
        path_segments = []
        current = vertex
        while self.parents.get(current) is not None:
            path_segments.append(self.trajectories[current])
            current = self.parents[current]
        
        if not path_segments:
            return []
        
        # Combine all segments into one continuous path
        full_path = list(path_segments[-1]) 
        for seg in reversed(path_segments[:-1]):
            full_path.extend(seg[1:])
        return full_path

class KinodynamicMORRF:
    """
    Implements a Multi-Objective RRT* for a Kinodynamic Unicycle model.
    This version includes iteration-level parallelism and dynamic normalization of objectives.
    """
    def __init__(self, start_state, goal_pos, bounds, obstacle_list,
                 num_iterations=5000, num_subproblems=30, 
                 control_bounds=([0.5, 2.0], [-np.pi/4, np.pi/4]),
                 integration_time=0.5, integration_steps=10, 
                 near_radius_gamma=20.0, max_workers=None):
        
        self.start_vertex = Vertex(start_state)
        self.goal_pos = np.array(goal_pos)
        self.bounds = np.array(bounds).reshape((2, 2))
        self.obstacle_list = obstacle_list
        self.num_iterations = num_iterations
        self.num_subproblems = num_subproblems
        self.K = 2  # Number of objectives
        self.dim = 2 # Dimension of the position space
        self.near_radius_gamma = near_radius_gamma

        # --- Kinodynamic Parameters ---
        self.control_bounds = control_bounds
        self.integration_time = integration_time
        self.integration_steps = integration_steps
        
        # --- Algorithm Data Structures ---
        self.vertices = [self.start_vertex]
        self.cost_functions = [self._cost_travel_distance, self._cost_obstacle_risk]
        self.reference_trees = [Tree(self.start_vertex, 1) for _ in range(self.K)]
        self.subproblem_trees = [Tree(self.start_vertex, self.K) for _ in range(self.num_subproblems)]
        self.lambda_vectors = [np.array([i / (self.num_subproblems - 1), 1 - (i / (self.num_subproblems - 1))])
                               for i in range(self.num_subproblems)]
        self.max_workers = max_workers
        
        # --- Dynamic Normalization Bounds ---
        self.utopia_point = np.full(self.K, INF)   # Tracks min cost for each objective
        self.nadir_point = np.full(self.K, -INF)  # Tracks max cost for each objective

    def run_with_perf_monitor(self):
        """
        Runs the planner while monitoring its performance (time and memory).
        """
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024) # in MB
        
        print("--- Starting Kinodynamic MORRF* Planner (Dynamic Normalization & Parallelism) ---")
        print(f"Using {self.max_workers if self.max_workers else os.cpu_count()} worker threads.")
        print(f"Initial memory usage: {mem_before:.2f} MB")
        
        start_time = time.perf_counter()
        self.run()
        end_time = time.perf_counter()
        
        mem_after = process.memory_info().rss / (1024 * 1024) # in MB
        
        print("\n--- Performance Report ---")
        print(f"Total Computation Time: {end_time - start_time:.4f} seconds")
        print(f"Final memory usage: {mem_after:.2f} MB")
        print(f"Memory consumed by planner: {mem_after - mem_before:.2f} MB")
        print("--------------------------")

    def run(self):
        """
        Executes the main MORRF* loop using iteration-level parallelism.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for i in range(self.num_iterations):
                x_rand = self._sample()
                x_nearest = self._nearest(x_rand)
                best_new_vertex, _ = self._find_best_motion(x_nearest, x_rand)

                if best_new_vertex:
                    self.vertices.append(best_new_vertex)
                    neighbors = self._near(best_new_vertex)
                    
                    # Extend reference trees in parallel to find normalization bounds
                    ref_futures = [executor.submit(self._extend_ref, tree, best_new_vertex, neighbors, k)
                                   for k, tree in enumerate(self.reference_trees)]
                    concurrent.futures.wait(ref_futures)
                    
                    # Update normalization bounds before extending subproblem trees
                    self._update_normalization_bounds()
                    
                    # Extend subproblem trees in parallel using the new bounds
                    sub_futures = [executor.submit(self._extend_sub, tree, best_new_vertex, neighbors, m)
                                   for m, tree in enumerate(self.subproblem_trees)]
                    concurrent.futures.wait(sub_futures)

                if i % 1000 == 0 and i > 0:
                    print(f"  Iteration {i}...")

    def _fitness(self, cost_vector, lambda_m):
        """
        Calculates the Tchebycheff fitness after applying dynamic normalization.
        """
        # Calculate the current range of costs found so far
        cost_range = self.nadir_point - self.utopia_point
        epsilon = 1e-8 # Prevent division by zero
        
        # Normalize the cost vector to be within [0, 1]
        normalized_cost = (cost_vector - self.utopia_point) / (cost_range + epsilon)
        
        # Calculate fitness using the normalized cost
        return np.max(lambda_m * normalized_cost)

    def _update_normalization_bounds(self):
        """
        Updates the utopia (min) and nadir (max) points using the reference trees.
        """
        for k, ref_tree in enumerate(self.reference_trees):
            if not ref_tree.costs: continue
            
            all_costs = np.array([cost[0] for cost in ref_tree.costs.values() if cost is not None])
            if all_costs.size == 0: continue
            
            min_cost_k = np.min(all_costs)
            max_cost_k = np.max(all_costs)
            
            self.utopia_point[k] = min(self.utopia_point[k], min_cost_k)
            self.nadir_point[k] = max(self.nadir_point[k], max_cost_k)

    def _find_best_motion(self, from_vertex, to_pos):
        """
        Samples multiple control inputs and returns the motion that gets closest to the target position.
        """
        best_dist = INF
        best_vertex = None
        best_segment = None
        for _ in range(10): # Number of motion samples
            v, segment, _ = self._steer(from_vertex, to_pos)
            if v and self._is_path_obstacle_free(segment):
                dist = np.linalg.norm(v.position - to_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_vertex = v
                    best_segment = segment
        return best_vertex, best_segment

    def _extend_ref(self, tree, x_new, neighbors, k):
        """
        Extends a reference tree with RRT* logic (parent selection and rewiring).
        """
        cost_func = self.cost_functions[k]
        min_cost = INF
        best_parent = None
        best_segment = None
        
        # 1. Find the best parent in the neighborhood
        for x_near in neighbors:
            if x_near in tree.costs:
                _, segment, _ = self._steer(x_near, x_new.position)
                if segment and self._is_path_obstacle_free(segment):
                    cost = tree.costs[x_near][0] + cost_func(segment)
                    if cost < min_cost:
                        min_cost = cost
                        best_parent = x_near
                        best_segment = segment
        
        if best_parent:
            tree.parents[x_new] = best_parent
            tree.costs[x_new] = np.array([min_cost])
            tree.trajectories[x_new] = best_segment

            # 2. Rewire the neighbors
            for x_near in neighbors:
                if x_near is not best_parent:
                    _, segment, _ = self._steer(x_new, x_near.position)
                    if segment and self._is_path_obstacle_free(segment):
                        potential_new_cost = tree.costs[x_new][0] + cost_func(segment)
                        if potential_new_cost < tree.costs[x_near][0]:
                            tree.parents[x_near] = x_new
                            tree.costs[x_near] = np.array([potential_new_cost])
                            tree.trajectories[x_near] = segment

    def _extend_sub(self, tree, x_new, neighbors, m):
        """
        Extends a subproblem tree with RRT* logic using the normalized fitness function.
        """
        lambda_m = self.lambda_vectors[m]
        min_fitness = INF
        best_parent = None
        best_cost_vec = np.full(self.K, INF)
        best_segment = None

        # 1. Find the parent that minimizes Tchebycheff fitness
        for x_near in neighbors:
            if x_near in tree.costs:
                _, segment, _ = self._steer(x_near, x_new.position)
                if segment and self._is_path_obstacle_free(segment):
                    segment_cost = np.array([f(segment) for f in self.cost_functions])
                    cost_vec = tree.costs[x_near] + segment_cost
                    fitness = self._fitness(cost_vec, lambda_m)
                    if fitness < min_fitness:
                        min_fitness = fitness
                        best_parent = x_near
                        best_cost_vec = cost_vec
                        best_segment = segment
        
        if best_parent:
            tree.parents[x_new] = best_parent
            tree.costs[x_new] = best_cost_vec
            tree.trajectories[x_new] = best_segment

            # 2. Rewire the neighbors
            for x_near in neighbors:
                 if x_near is not best_parent:
                    _, segment, _ = self._steer(x_new, x_near.position)
                    if segment and self._is_path_obstacle_free(segment):
                        segment_cost = np.array([f(segment) for f in self.cost_functions])
                        potential_cost_vec = tree.costs[x_new] + segment_cost
                        current_fitness = self._fitness(tree.costs[x_near], lambda_m)
                        potential_fitness = self._fitness(potential_cost_vec, lambda_m)
                        if potential_fitness < current_fitness:
                            tree.parents[x_near] = x_new
                            tree.costs[x_near] = potential_cost_vec
                            tree.trajectories[x_near] = segment
    
    def _near(self, center_vertex):
        """
        Returns vertices within a ball of a specific radius around a center vertex.
        """
        n = len(self.vertices)
        radius = self.near_radius_gamma * (math.log(n) / n if n > 1 else 1.0)**(1/self.dim)
        return [v for v in self.vertices if np.linalg.norm(v.position - center_vertex.position) <= radius]

    def visualize_results(self):
        """
        Generates two plots: the paths in the configuration space and the
        Pareto front approximation in the objective space.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('Kinodynamic MORRF* Results', fontsize=16)

        # Plot 1: Configuration Space
        ax1.set_title('Configuration Space Paths')
        ax1.set_xlim(self.bounds[0, 0], self.bounds[0, 1])
        ax1.set_ylim(self.bounds[1, 0], self.bounds[1, 1])
        ax1.set_aspect('equal', adjustable='box')
        ax1.plot(self.start_vertex.state[0], self.start_vertex.state[1], 'go', ms=10, label='Start')
        ax1.plot(self.goal_pos[0], self.goal_pos[1], 'r*', ms=15, label='Goal')
        for obs in self.obstacle_list: ax1.add_patch(Circle(obs[:2], obs[2], fc='gray', alpha=0.7))
        ax1.grid(True)
        
        # Plot 2: Objective Space
        ax2.set_title('Objective Space (Pareto Front Approximation)')
        ax2.set_xlabel('Objective 1: Travel Distance')
        ax2.set_ylabel('Objective 2: Obstacle Risk')
        ax2.grid(True)
        
        goal_vertex = self._nearest(self.goal_pos)
        final_path_costs = []
        sol_count = 0
        
        for tree in self.subproblem_trees:
            if goal_vertex in tree.costs:
                sol_count += 1
                cost_vector = tree.costs[goal_vertex]
                final_path_costs.append(cost_vector)
                
                full_path_states = tree.get_full_path_to_vertex(goal_vertex)
                if full_path_states:
                    path_coords = np.array(full_path_states)[:, :2]
                    ax1.plot(path_coords[:, 0], path_coords[:, 1], '-', lw=1.5, alpha=0.7)
        
        print(f"Number of solutions found: {sol_count}")
        if final_path_costs:
            costs = np.array(final_path_costs)
            ax2.scatter(costs[:, 0], costs[:, 1], c='blue', alpha=0.8, label='Solutions')
        
        ax1.legend()
        ax2.legend()
        plt.show()

    def _sample(self):
        """ Returns an independent, uniformly distributed sample from the space. """
        return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])

    def _nearest(self, position):
        """ Finds the vertex in the shared list closest to the given position. """
        min_dist = INF
        nearest_vertex = None
        for v in self.vertices:
            dist = np.linalg.norm(v.position - position)
            if dist < min_dist:
                min_dist = dist
                nearest_vertex = v
        return nearest_vertex

    def _steer(self, from_vertex, to_position):
        """
        Generates a new state by simulating unicycle dynamics with a random control input.
        """
        v = random.uniform(self.control_bounds[0][0], self.control_bounds[0][1])
        w = random.uniform(self.control_bounds[1][0], self.control_bounds[1][1])
        control = [v, w]
        path_segment = [from_vertex.state]
        current_state = np.copy(from_vertex.state)
        dt = self.integration_time / self.integration_steps
        
        for _ in range(self.integration_steps):
            x, y, theta = current_state
            x_new, y_new = x + v * np.cos(theta) * dt, y + v * np.sin(theta) * dt
            theta_new = np.arctan2(np.sin(theta + w * dt), np.cos(theta + w * dt))
            current_state = np.array([x_new, y_new, theta_new])
            path_segment.append(current_state)
            
        final_state = path_segment[-1]
        if not (self.bounds[0, 0] <= final_state[0] <= self.bounds[0, 1] and \
                self.bounds[1, 0] <= final_state[1] <= self.bounds[1, 1]):
            return None, [], []
            
        return Vertex(final_state), path_segment, control

    def _is_path_obstacle_free(self, path_segment):
        """ Checks if the entire discretized path segment is collision-free. """
        return all(np.linalg.norm(state[:2] - obs[:2]) > obs[2] for state in path_segment for obs in self.obstacle_list)

    def _cost_travel_distance(self, path_segment):
        """ Objective 1: Minimize travel distance along the curved path. """
        return sum(np.linalg.norm(path_segment[i][:2] - path_segment[i+1][:2]) for i in range(len(path_segment) - 1))

    def _cost_obstacle_risk(self, path_segment):
        """
        Objective 2: Minimize risk by maximizing clearance from obstacles.
        Cost is inversely proportional to the minimum distance to an obstacle.
        """
        if not self.obstacle_list:
            return 0.0
        min_dist_to_obs = min(np.linalg.norm(s[:2] - o[:2]) - o[2] for s in path_segment for o in self.obstacle_list)
        return 1.0 / (min_dist_to_obs + 0.1)

# --- Example Usage ---
if __name__ == '__main__':
    START = (0, 0, 0)
    GOAL = (9, 9)
    BOUNDS = (0, 10, 0, 10)
    OBSTACLES = [(3, 3, 1), (7, 2, 1.2), (5, 7, 1.5), (2, 8, 0.8)]
    
    kin_morrf_planner = KinodynamicMORRF(
        start_state=START, 
        goal_pos=GOAL, 
        bounds=BOUNDS, 
        obstacle_list=OBSTACLES,
        num_iterations=3000,
        num_subproblems=30,
        max_workers=os.cpu_count()
    )
    
    kin_morrf_planner.run_with_perf_monitor()
    kin_morrf_planner.visualize_results()
