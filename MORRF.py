import numpy as np
import math
import random
import time
import os
import psutil # You need to install this: pip install psutil
import concurrent.futures
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
# A constant for infinity, useful for initial costs.
INF = float('inf')

class Vertex:
    """
    Represents a vertex (or node) in the configuration space.
    All trees in the forest share the same set of vertices.
    """
    def __init__(self, position):
        # The coordinates of the vertex in the d-dimensional space.
        self.position = np.array(position)

    def __repr__(self):
        return f"Vertex({self.position})"

class Tree:
    """
    Represents a single tree structure within the random forest.
    Each tree has its own set of edges and a way of calculating costs.
    """
    def __init__(self, initial_vertex):
        # A map from a vertex to its parent vertex in this tree.
        self.parents = {initial_vertex: None}
        # A map from a vertex to the cost to reach it from the root.
        self.costs = {initial_vertex: 0.0}

    def get_path_to_vertex(self, vertex):
        """ Reconstructs the path from the root to a given vertex. """
        path = []
        current = vertex
        while current is not None:
            path.append(current)
            current = self.parents.get(current)
        return path[::-1] # Return reversed path (start to end)

class MORRF:
    """
    Implementation of the Multi-Objective Rapidly-exploring Random Forest* (MORRF*)
    [cite_start]algorithm as described in the paper[cite: 366, 400].
    """
    def __init__(self, start_pos, goal_pos, bounds, obstacle_check_func,obstacle_list, cost_functions,
                 num_iterations=2000, num_subproblems=10, step_size=0.1, near_radius_gamma=1.0):
        """
        Initializes the MORRF* planner.

        Args:
            start_pos (tuple): The starting coordinates.
            goal_pos (tuple): The goal coordinates.
            bounds (tuple): The boundaries of the configuration space, e.g., (min_x, max_x, min_y, max_y).
            obstacle_check_func (callable): A function `f(point1, point2)` that returns
                                            True if the line segment is obstacle-free.
            cost_functions (list[callable]): A list of K functions `f(point1, point2)` that
                                             return the cost of the segment for each objective.
            num_iterations (int): The number of iterations to run (N).
            num_subproblems (int): The number of subproblem trees to create (M).
            step_size (float): The maximum distance for the STEER function (eta).
            near_radius_gamma (float): A constant for calculating the NEAR radius.
        """
        # (All other initializations are the same)
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.num_iterations = num_iterations
        self.num_subproblems = num_subproblems
        self.K = len(cost_functions)  # Number of objectives
        self.dim = len(start_pos)      # Dimensionality of the space
        self.step_size = step_size
        self.gamma = near_radius_gamma
        self.obstacle_list = obstacle_list
        # --- User-defined functions ---
        self.obstacle_free = obstacle_check_func
        self.cost_functions = cost_functions

        # [cite_start]--- Core data structures [cite: 367, 462] ---
        self.start_vertex = Vertex(start_pos)
        self.goal_vertex = Vertex(goal_pos) # Note: Goal handling is an extension
        self.vertices = [self.start_vertex] # The shared set of vertices for all trees

        # K reference trees, one for each objective
        self.reference_trees = [Tree(self.start_vertex) for _ in range(self.K)]

        # M subproblem trees for finding the Pareto front
        self.subproblem_trees = [Tree(self.start_vertex) for _ in range(self.num_subproblems)]

        # --- Algorithm parameters ---
        self.bounds = np.array(bounds).reshape((self.dim, 2))
        self.lambda_vectors = self._generate_lambda_vectors()
        # [cite_start]Utopia point, initialized to infinity [cite: 453, 529]
        self.utopia_point = np.full(self.K, INF)

    def _generate_lambda_vectors(self):
        """
        [cite_start]Generates M weight vectors for the subproblems[cite: 539].
        For 2D, it creates uniformly spaced vectors. For higher dimensions,
        you might consider more sophisticated sampling.

        Question for you: Is this simple linear spacing sufficient for your
        problem, or do you need a different distribution of weights?
        """
        if self.K == 2:
            return [np.array([i / (self.num_subproblems - 1), 1 - (i / (self.num_subproblems - 1))])
                    for i in range(self.num_subproblems)]
        else:
            # For >2 objectives, random sampling is simpler.
            vectors = np.random.rand(self.num_subproblems, self.K)
            return vectors / np.sum(vectors, axis=1, keepdims=True)

    def run(self):
        """
        Executes the main MORRF* loop (Algorithm 1).
        """
        print(f"Running MORRF* for {self.num_iterations} iterations...")
        for i in range(self.num_iterations):
            x_rand = self._sample()
            x_nearest = self._nearest(x_rand)
            x_new_pos = self._steer(x_nearest.position, x_rand)
            
            # Skip if STEER returns a position that's already a vertex
            if any(np.array_equal(x_new_pos, v.position) for v in self.vertices):
                continue
            
            x_new = Vertex(x_new_pos)

            if self.obstacle_free(x_nearest.position, x_new.position):
                self.vertices.append(x_new)

                # 5. Extend reference trees (Algorithm 2)
                for k, ref_tree in enumerate(self.reference_trees):
                    # FIX: Pass x_nearest to the extend function.
                    self._extend_ref(ref_tree, x_new, x_nearest, k)

                self._update_utopia_point()

                # 7. Extend subproblem trees (Algorithm 3)
                for m, sub_tree in enumerate(self.subproblem_trees):
                    # FIX: Pass x_nearest to the extend function.
                    self._extend_sub(sub_tree, x_new, x_nearest, m)

            if i % 500 == 0 and i > 0:
                print(f"  Iteration {i}... Current Utopia Point: {self.utopia_point}")

        print("MORRF* planning finished.") 
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

    def _steer(self, from_pos, to_pos):
        """ Returns a point at most `step_size` away from `from_pos` in the direction of `to_pos`. """
        direction = to_pos - from_pos
        dist = np.linalg.norm(direction)
        if dist <= self.step_size:
            return to_pos
        return from_pos + direction / dist * self.step_size

    def _near(self, center_vertex):
        """
        [cite_start]Returns a set of vertices within a ball of a specific radius around a point[cite: 523].
        The radius decreases as the number of vertices increases.
        """
        n = len(self.vertices)
        radius = self.gamma * (math.log(n) / n)**(1/self.dim)
        near_vertices = []
        for v in self.vertices:
            if v is not center_vertex and np.linalg.norm(v.position - center_vertex.position) <= radius:
                near_vertices.append(v)
        return near_vertices

    def _get_path_cost_vector(self, tree, vertex):
        """
        Calculates the total cost vector from the root to a vertex by summing
        up segment costs along the path in that specific tree.
        """
        cost_vector = np.zeros(self.K)
        path = tree.get_path_to_vertex(vertex)
        for i in range(len(path) - 1):
            p1 = path[i].position
            p2 = path[i+1].position
            for k in range(self.K):
                cost_vector[k] += self.cost_functions[k](p1, p2)
        return cost_vector

    def _update_utopia_point(self):
        """
        Updates the Utopia reference vector. Each element is the minimum cost found
        so far by the corresponding reference tree for any path to the goal.
        Question for you: The paper isn't explicit on how the Utopia point is defined
        if a goal region isn't reached yet. A common approach is to use the minimum
        cost seen so far among all vertices. Is this assumption acceptable?
        """
        # For simplicity, we'll find the minimum cost vertex in each reference tree.
        for k, ref_tree in enumerate(self.reference_trees):
            min_cost_k = min(ref_tree.costs.values())
            self.utopia_point[k] = min(self.utopia_point[k], min_cost_k)

    def _fitness(self, cost_vector, lambda_m):
        """
        [cite_start]Calculates the fitness using the Tchebycheff method[cite: 425, 532].
        """
        return np.max(lambda_m * np.abs(cost_vector - self.utopia_point))

    def _extend_ref(self, tree, x_new, x_nearest, k):
        """
        Extends a reference tree with the new vertex `x_new` (Algorithm 2).
        This involves finding the best parent for `x_new` and rewiring neighbors.
        """
        # FIX: First, add x_new to the tree with x_nearest as its parent.
        # This establishes a baseline cost and prevents the KeyError.
        initial_cost = tree.costs[x_nearest] + self.cost_functions[k](x_nearest.position, x_new.position)
        tree.parents[x_new] = x_nearest
        tree.costs[x_new] = initial_cost

        x_min = x_nearest
        min_cost = initial_cost

        # 1. Find the best parent for x_new from its neighbors
        for x_near in self._near(x_new):
            if self.obstacle_free(x_near.position, x_new.position):
                cost = tree.costs[x_near] + self.cost_functions[k](x_near.position, x_new.position)
                if cost < min_cost:
                    min_cost = cost
                    x_min = x_near
        
        # If a better parent was found, update the tree entry for x_new
        if x_min is not x_nearest:
            tree.parents[x_new] = x_min
            tree.costs[x_new] = min_cost

        # 2. Rewire the tree: Check if x_new offers a better path for its neighbors
        for x_near in self._near(x_new):
            if x_near is not x_min and self.obstacle_free(x_new.position, x_near.position):
                new_potential_cost = tree.costs[x_new] + self.cost_functions[k](x_new.position, x_near.position)
                if new_potential_cost < tree.costs[x_near]:
                    tree.parents[x_near] = x_new
                    tree.costs[x_near] = new_potential_cost

    def _extend_sub(self, tree, x_new, x_nearest, m):
        """
        Extends a subproblem tree (Algorithm 3), using Tchebycheff fitness.
        """
        lambda_m = self.lambda_vectors[m]

        # --- Step 1: Find the best parent for x_new based on fitness ---
        # FIX: First, connect x_new via x_nearest to establish a baseline fitness.
        c_vec_initial = self._get_path_cost_vector(tree, x_nearest) + self._get_path_cost_vector_direct(x_nearest, x_new)
        initial_fitness = self._fitness(c_vec_initial, lambda_m)
        tree.parents[x_new] = x_nearest
        tree.costs[x_new] = initial_fitness # 'costs' for subproblem trees store fitness

        x_min = x_nearest
        min_fitness = initial_fitness

        # Now, search neighbors for a parent that provides better fitness.
        for x_near in self._near(x_new):
            if self.obstacle_free(x_near.position, x_new.position):
                c_vec_near = self._get_path_cost_vector(tree, x_near) + self._get_path_cost_vector_direct(x_near, x_new)
                fitness = self._fitness(c_vec_near, lambda_m)
                if fitness < min_fitness:
                    min_fitness = fitness
                    x_min = x_near

        # Update parent and fitness if a better one was found.
        if x_min is not x_nearest:
            tree.parents[x_new] = x_min
            tree.costs[x_new] = min_fitness

        # --- Step 2: Rewire the neighbors ---
        for x_near in self._near(x_new):
            if x_near is not x_min and self.obstacle_free(x_new.position, x_near.position):
                c_vec_potential = self._get_path_cost_vector(tree, x_new) + self._get_path_cost_vector_direct(x_new, x_near)
                potential_fitness = self._fitness(c_vec_potential, lambda_m)

                # Use .get() for safety, though the key should always exist.
                if potential_fitness < tree.costs.get(x_near, INF):
                    tree.parents[x_near] = x_new
                    tree.costs[x_near] = potential_fitness
    def _get_path_cost_vector_direct(self, v_from, v_to):
        """ Helper to calculate the cost vector for a single segment. """
        cost_vector = np.zeros(self.K)
        for k in range(self.K):
            cost_vector[k] = self.cost_functions[k](v_from.position, v_to.position)
        return cost_vector
    def visualize_results(self):
        """
        Generates two plots: the paths in the configuration space and the
        Pareto front approximation in the objective space.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('MORRF* Results', fontsize=16)

        # --- Plot 1: Configuration Space (Paths) ---
        ax1.set_title('Configuration Space')
        ax1.set_xlim(self.bounds[0, 0], self.bounds[0, 1])
        ax1.set_ylim(self.bounds[1, 0], self.bounds[1, 1])
        ax1.set_aspect('equal', adjustable='box')

        # Plot Start and Goal
        ax1.plot(self.start_pos[0], self.start_pos[1], 'go', markersize=10, label='Start')
        ax1.plot(self.goal_pos[0], self.goal_pos[1], 'r*', markersize=15, label='Goal')

        # Plot Obstacles
        for obs in self.obstacle_list:
            # Assuming rectangular obstacles for this example: (x, y, width, height)
            ax1.add_patch(Rectangle((obs[0], obs[2]), obs[1]-obs[0], obs[3]-obs[2], facecolor='gray', alpha=0.7))

        # --- Plot 2: Objective Space (Pareto Front) ---
        ax2.set_title('Objective Space (Pareto Front Approximation)')
        ax2.set_xlabel('Objective 1: Travel Distance')
        ax2.set_ylabel('Objective 2: Risk')

        final_path_costs = []
        goal_vertex = self._nearest(np.array(self.goal_pos))

        # Extract paths and calculate their final costs
        for tree in self.subproblem_trees:
            path_to_goal = tree.get_path_to_vertex(goal_vertex)
            if not path_to_goal:
                continue

            # Plot path in configuration space
            path_coords = np.array([v.position for v in path_to_goal])
            ax1.plot(path_coords[:, 0], path_coords[:, 1], '-')

            # Calculate total cost vector for the path
            cost_vector = np.zeros(self.K)
            for i in range(len(path_to_goal) - 1):
                p1 = path_to_goal[i].position
                p2 = path_to_goal[i+1].position
                for k in range(self.K):
                    cost_vector[k] += self.cost_functions[k](p1, p2)
            final_path_costs.append(cost_vector)

        # Plot costs in objective space
        if final_path_costs:
            costs = np.array(final_path_costs)
            ax2.scatter(costs[:, 0], costs[:, 1], c='blue', label='Solutions')

        ax1.legend()
        ax2.grid(True)
        ax2.legend()
        plt.show()
if __name__ == '__main__':
    # Define problem
    START = (0, 0)
    GOAL = (10, 10)
    BOUNDS = (0, 10, 0, 10)
    # Define rectangular obstacles: (x_min, x_max, y_min, y_max)
    OBSTACLES = [(4, 6, 4, 6)]

    def is_obstacle_free(p1, p2):
        # A simple check (can be improved with line intersection algorithms)
        for obs in OBSTACLES:
            if obs[0] < p1[0] < obs[1] and obs[2] < p1[1] < obs[3]: return False
            if obs[0] < p2[0] < obs[1] and obs[2] < p2[1] < obs[3]: return False
        return True

    def cost_distance(p1, p2): return np.linalg.norm(p1 - p2)
    def cost_risk(p1, p2):
        risk_center = np.array([7, 3])
        if np.linalg.norm((p1 + p2) / 2 - risk_center) < 2.0:
            return np.linalg.norm(p1 - p2) * 10
        return np.linalg.norm(p1 - p2)

    # Initialize and Run Planner
    planner = MORRF(
        start_pos=START,
        goal_pos=GOAL,
        bounds=BOUNDS,
        obstacle_list=OBSTACLES,
        obstacle_check_func=is_obstacle_free,# Pass obstacles
        cost_functions=[cost_distance, cost_risk],
        # (other params)
    )
    planner.run()

    # --- Visualize the results ---
    planner.visualize_results()
