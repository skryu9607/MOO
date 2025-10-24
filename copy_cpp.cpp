// Point Mass Multi Objective RRForest* // 
// Edited by SeungKeol Ryu 
// Oct 1. 2025 // 
#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <limits>
#include <random>
#include <algorithm>
#include <fstream>
#include <map>

// Basic constatns // 
constexpr double PI = 3.14159265358979323846;
constexpr double X_MAX = 100.0;
constexpr double Y_MAX = 100.0;

struct State{
    // Default 
    double x = 0.0;
    double y = 0.0;

};
struct Node{
    int id;
    State state;
    // Constructor 
    Node(int i, const State& s) : id(i), state(s){}

};
// A series of Nodes
struct Trajectory{
    std::vector<State> path;
};


// --- Utility Functions  --- // 
double normalizeAngle(double angle);
double stateDistance(const State& s1, const State& s2);
State sampleState();
// Trajectory steer(const State& s_from, const State& s_to);
Trajectory Line(const State& s_from, const State& s_to);
bool isObstacleFree(const Trajectory& traj); // now : always true.
std::vector<double> calculateLineCost(const Trajectory& traj);
double calculateTchebycheffFitness(const std::vector<double>& cost_vec, const std::vector<double>& lambda, const std::vector<double>& z_utop);

// Tree class
class Tree{
    public: 
    // Indicates the map of the relationship of "parent" and "children"
    std::map<int,int> parent_map;
    // Indicates the map of the cost of each node.
    std::map<int, std::vector<double>> cost_map;
    // fitness map for subproblem trees.
    std::map<int, double> fitness_map;
    // 
    Tree() = default;
    int num_objectives = 2;
    std::vector<double> getCost(int node_id) const{
        // if exists, return the cost vector of the node.
        if (cost_map.count(node_id)){
            return cost_map.at(node_id);
        }
        // if not, return infinity vector.
        return std::vector<double>(num_objectives, std::numeric_limits<double>::infinity());
    }
};
class MORRFPlanner{
    public:
    MORRFplanner(const State& start, const State& goal, double threshold, int number_objectives, int number_subproblems): 
    start(start), goal(goal), threshold(threshold),num_objectives(number_objectives), num_sub(number_subproblems){

    }

    }




}
// Main dish // 
class MORRFPlanner {
private:
    State start_state;
    State goal_state;
    double goal_threshold;
    size_t num_objectives;
    size_t num_subproblems;

    // Centeralized Node Storage = a certain graph : Every trees share a set of the same nodes. 
    std::vector<std::shared_ptr<Node>> G_nodes;
    // Single objective 
    std::vector<Tree> reference_trees;
    // Single objectives combined with different weights (lambdas)
    std::vector<Tree> subproblem_trees;
    std::vector<std::vector<double>> lambdas;
    // Utopian point.
    std::vector<double> z_utop;

public:
    MORRFPlanner(const State& start, const State& goal, double threshold, size_t number_objectives, size_t number_subproblems)
        : start_state(start), goal_state(goal), goal_threshold(threshold), num_objectives(number_objectives), num_subproblems(number_subproblems) {
        
        // Centralized node has the start node must be included and shared with all trees. 
        auto root_node = std::make_shared<Node>(0, start_state);
        G_nodes.push_back(root_node);

        // Reference trees initializes
        reference_trees.resize(num_objectives);
        for (size_t k = 0; k < num_objectives; ++k) {
            auto initial_cost = std::vector<double>(num_objectives, std::numeric_limits<double>::infinity());
            initial_cost[k] = 0.0;
            reference_trees[k].cost_map[0] = initial_cost;
        }
        
        // Subproblem trees initializes
        subproblem_trees.resize(num_subproblems);
        for (size_t m = 0; m < num_subproblems; ++m) {
             subproblem_trees[m].cost_map[0] = std::vector<double>(num_objectives, 0.0);
             // Subproblem trees need fitness_map.
             subproblem_trees[m].fitness_map[0] = 0.0;
        }

        // lambda
        lambdas.resize(num_subproblems);
        for (size_t m = 0; m < num_subproblems; ++m) {
            lambdas[m].resize(num_objectives);
            if (num_subproblems > 1) {
                double val = static_cast<double>(m) / (num_subproblems - 1);
                lambdas[m][0] = (1.0 - val ) / 128; // always path length
                lambdas[m][1] = val / 4.3865;       // in this case, smoothness (angle term)
            } else {
                lambdas[m][0] = 1.0; lambdas[m][1] = 0.0;
            }
        }


        z_utop.assign(num_objectives, std::numeric_limits<double>::infinity());
    }

    std::shared_ptr<Node> getNearestNode(const State& s) {
        std::shared_ptr<Node> nearest = nullptr;
        double min_dist = std::numeric_limits<double>::infinity();
        for (const auto& node : G_nodes) {
            double dist = stateDistance(node->state, s);
            if (dist < min_dist) {
                min_dist = dist;
                nearest = node;
            }
        }
        return nearest;
    }
    std::vector<std::shared_ptr<Node>> getNearNodes(const State& s, double radius) {
            std::vector<std::shared_ptr<Node>> near_nodes;
            for (const auto& node : G_nodes) {
                if (stateDistance(node->state, s) <= radius) {
                    near_nodes.push_back(node);
                }
            }
            return near_nodes;
        }
    void run(int max_iterations);
    
    std::vector<std::vector<double>> getSolutions() {
        std::vector<std::vector<double>> final_solutions;

        // 1. search for each subproblem tree
        for (const auto& tree : subproblem_trees) {
            
            double min_fitness = std::numeric_limits<double>::infinity();
            int best_node_id = -1;

            // 2. find the best one by each subproblem tree
            for (const auto& node : G_nodes) {
                double dist = std::hypot(node->state.x - goal_state.x, node->state.y - goal_state.y);
                if (dist < goal_threshold) {
                    // check the fitness map.
                    if (tree.fitness_map.count(node->id)) {
                        // the lowest fitness value is the optimal one. 
                        if (tree.fitness_map.at(node->id) < min_fitness) {
                            min_fitness = tree.fitness_map.at(node->id);
                            best_node_id = node->id;
                        }
                    }
                }
            }
            // 3. Therefore, the node which has the lowest fitness value is the optimal one which has the lowest cost value = non-dominated one. 
            if (best_node_id != -1) {
                // get the cost from costmap.
                final_solutions.push_back(tree.cost_map.at(best_node_id));
            }
        }
        return final_solutions;
    }
    
    void saveCostsToCSV(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return;
        }
        auto solutions = getSolutions();
        std::cout << "Found " << solutions.size() << " solutions near the goal." << std::endl;
        file << "cost1_length,cost2_smoothness\n";
        for (const auto& cost : solutions) {
            file << cost[0] << "," << cost[1] << "\n";
        }
        file.close();
        std::cout << "Solution costs saved to " << filename << std::endl;
    }

    void saveFinalSolutionsToCSV() {
        // 1. 
        std::ofstream solutions_file("pareto_solutions.csv");
        std::ofstream paths_file("pareto_paths.csv");

        if (!solutions_file.is_open() || !paths_file.is_open()) {
            std::cerr << "Error: Could not open output CSV files." << std::endl;
            return;
        }

        // 2. Headers
        solutions_file << "path_id,cost1_length,cost2_smoothness\n";
        paths_file << "path_id,x,y\n";

        // 3. 
        for (size_t m = 0; m < subproblem_trees.size(); ++m) {
            const auto& tree = subproblem_trees[m];
            
            double min_fitness = std::numeric_limits<double>::infinity();
            int best_node_id = -1;

            // 4. 
            for (const auto& node : G_nodes) {
                double dist = std::hypot(node->state.x - goal_state.x, node->state.y - goal_state.y);
                if (dist < goal_threshold) {
                    if (tree.fitness_map.count(node->id) && tree.fitness_map.at(node->id) < min_fitness) {
                        min_fitness = tree.fitness_map.at(node->id);
                        best_node_id = node->id;
                    }
                }
            }

            std::cout << "Subproblem Tree " << m << ": Best Node ID = " << best_node_id << std::endl;
            // 5. 
            if (best_node_id != -1) {
                // 5a. costs
                const auto& cost = tree.cost_map.at(best_node_id);
                std::cout << cost[0] << "," << cost[1] << std::endl;

                // 5b. backtracking
                std::vector<State> path;
                int current_node_id = best_node_id;
                size_t i = 0;
                while (tree.parent_map.count(current_node_id)) {
                    i = i + 1;
                    if (i % 2000000 == 0){
                        std::cout << i << " th backtracking..." << std::endl;  
                    } 
                    path.push_back(G_nodes[current_node_id]->state);
                    current_node_id = tree.parent_map.at(current_node_id);
                }
                path.push_back(G_nodes[0]->state); 
                std::reverse(path.begin(), path.end());

                for (const auto& state : path) {
                    paths_file << m << "," << state.x << "," << state.y << "\n";
                }
            }
        }

        // 6. close the files
        solutions_file.close();
        paths_file.close();
        
        std::cout << "Final solutions and paths saved successfully." << std::endl;
    }

};

void MORRFPlanner::run(int max_iterations) {
    for (int i = 0; i < max_iterations; ++i) {
        State s_rand = sampleState();
        auto x_nearest = getNearestNode(s_rand);

        Trajectory traj_initial = Line(x_nearest->state, s_rand);
        if (traj_initial.path.size() < 2 || !isObstacleFree(traj_initial)) continue;
        
        const State& s_new_state = traj_initial.path.back();

        // New node gets a new ID
        size_t new_node_id = G_nodes.size();
        auto x_new_node = std::make_shared<Node>(new_node_id, s_new_state);
        

        double radius = 15.0; 
        auto near_nodes = getNearNodes(s_new_state, radius);

        // --- Reference Trees Update (ChooseParent & Rewire) ---
        for (size_t k = 0; k < num_objectives; ++k) {
            auto& tree = reference_trees[k];

            // 1. ChooseParent: find the best parent of x_new_node among near_nodes
            size_t best_parent_id = x_nearest->id;
            double min_cost = tree.getCost(x_nearest->id, num_objectives)[k] 
                            + calculateLineCost(traj_initial)[k];

            for (const auto& near_node : near_nodes) {
                Trajectory traj_from_near = Line(near_node->state, s_new_state);
                if (near_node->id == best_parent_id) {
                    continue;
                }
                if (!isObstacleFree(traj_from_near)) continue;

                double cost_via_near = tree.getCost(near_node->id, num_objectives)[k]
                                     + calculateLineCost(traj_from_near)[k];
                
                if (cost_via_near < min_cost) {
                    min_cost = cost_via_near;
                    best_parent_id = near_node->id;
                }
            }

            // Update the tree with the new node
            tree.parent_map[new_node_id] = best_parent_id;
            auto final_parent_cost = tree.getCost(best_parent_id, num_objectives);
            final_parent_cost[k] = min_cost;
            tree.cost_map[new_node_id] = final_parent_cost;
            
            // 2. Rewire: Check the x_new_node can be a better parent for near_nodes
            for (const auto& near_node : near_nodes) {
                Trajectory traj_to_near = Line(s_new_state, near_node->state);
                if (!isObstacleFree(traj_to_near)) continue;

                double cost_via_new = tree.getCost(new_node_id, num_objectives)[k] 
                                    + calculateLineCost(traj_to_near)[k];
                
                if (cost_via_new < tree.getCost(near_node->id, num_objectives)[k]) {
                    tree.parent_map[near_node->id] = new_node_id;
                    auto updated_cost = tree.getCost(near_node->id, num_objectives);
                    updated_cost[k] = cost_via_new;
                    tree.cost_map[near_node->id] = updated_cost;
                }
            }
        }

        // x_new_node is addded to the centralized node storage, graph G. 
        G_nodes.push_back(x_new_node);

        // // --- Utopiaian point update ---
        // for (size_t k = 0; k < num_objectives; ++k) {
        //     z_utop[k] = std::min(z_utop[k], reference_trees[k].getCost(new_node_id, num_objectives)[k]);
        // }
        
        double dist_to_goal = std::hypot(x_new_node->state.x - goal_state.x, x_new_node->state.y - goal_state.y);
        if (dist_to_goal < goal_threshold) {
            
            // If the new node is close to the goal, update the utopian point.
            for (size_t k = 0; k < num_objectives; ++k) {
                double cost_in_tree_k = reference_trees[k].getCost(new_node_id, num_objectives)[k];
                if (cost_in_tree_k != std::numeric_limits<double>::infinity()) {
                    z_utop[k] = std::min(z_utop[k], cost_in_tree_k);
                }
            }
        }
        // --- Subproblem Trees Update (ChooseParent & Rewire) ---
        if (z_utop[0] == std::numeric_limits<double>::infinity() && z_utop[1] == std::numeric_limits<double>::infinity()) {
            // Utopian point is not updated yet, skip subproblem tree update.
            continue;
        }
        for (size_t m = 0; m < num_subproblems; ++m) {
            auto& tree = subproblem_trees[m];

            // 1. Choose Parent
            size_t best_parent_id = x_nearest->id;
            std::vector<double> best_cost_vec(num_objectives); 

            auto initial_parent_cost = tree.getCost(x_nearest->id, num_objectives);
            auto initial_seg_cost = calculateLineCost(traj_initial);
            std::vector<double> cost_vec(num_objectives);
            for(size_t obj=0; obj<num_objectives; ++obj) cost_vec[obj] = initial_parent_cost[obj] + initial_seg_cost[obj];
            double min_fitness = calculateTchebycheffFitness(cost_vec, lambdas[m], z_utop);

            for (const auto& near_node : near_nodes) {
                Trajectory traj_from_near = Line(near_node->state, s_new_state);
                if (!isObstacleFree(traj_from_near)) continue;

                auto parent_cost = tree.getCost(near_node->id, num_objectives);
                auto seg_cost = calculateLineCost(traj_from_near);
                for(size_t obj=0; obj<num_objectives; ++obj) cost_vec[obj] = parent_cost[obj] + seg_cost[obj];
                double fitness_via_near = calculateTchebycheffFitness(cost_vec, lambdas[m], z_utop);

                if (fitness_via_near < min_fitness) {
                    min_fitness = fitness_via_near;
                    best_parent_id = near_node->id;
                    best_cost_vec = cost_vec;
                    
                }
            }
            tree.cost_map[new_node_id] = best_cost_vec;
            tree.fitness_map[new_node_id] = min_fitness;
            tree.parent_map[new_node_id] = best_parent_id;
            //  if (i > 0 && i % 500 == 0){
            //     std::cout << "Minimum Fitness : " << min_fitness << std::endl;
            // }
            
            // 2. Rewire (subproblem tree)
            for (const auto& near_node : near_nodes) {
                if (near_node->id == best_parent_id) {
                    continue;
                }
                Trajectory traj_to_near = Line(s_new_state, near_node->state);
                 if (!isObstacleFree(traj_to_near)) continue;

                auto new_parent_cost = tree.getCost(new_node_id, num_objectives);
                auto seg_cost = calculateLineCost(traj_to_near);
                for(size_t obj=0; obj<num_objectives; ++obj) cost_vec[obj] = new_parent_cost[obj] + seg_cost[obj];
                double fitness_via_new = calculateTchebycheffFitness(cost_vec, lambdas[m], z_utop);

                if (fitness_via_new < tree.fitness_map[near_node->id]) {
                     tree.parent_map[near_node->id] = new_node_id;
                     tree.cost_map[near_node->id] = cost_vec;
                     tree.fitness_map[near_node->id] = fitness_via_new;
                }
            }
        }
        if (i > 0 && i % 2000 == 0) {
            std::cout << "Iteration " << i << "... Graph Nodes: " << G_nodes.size() << std::endl;
        }
        if (i % 500 == 0) {
            std::cout << "Current Utopian Point: " << z_utop[0] << ", " << z_utop[1] << std::endl;
        }
    }
    std::cout << "Planning finished after " << max_iterations << " iterations." << std::endl;
}


int main() {
    // 1. problem specification
    State start = {0.0, 0.0, PI / 4.0};
    State goal = {90.0, 90.0, 0.0};
    size_t num_objectives = 2;
    size_t num_subproblems = 5; 
    double goal_threshold = 3.0;
    int iterations = 2000;

    // 2. construct the planner "instance"
    MORRFPlanner planner(start, goal, goal_threshold, num_objectives, num_subproblems);
    
    // 3. running the planner
    std::cout << "Starting MORRF* planning with unicycle dynamics..." << std::endl;
    std::cout << "The number of subproblems : " << num_subproblems << "   " << "Max Iterations :" << iterations << std::endl;
    planner.run(iterations);
    
    // 4. saving results to csv 
    std::cout << "Planning finished. Saving results to CSV..." << std::endl;
    //planner.saveCostsToCSV("pareto_costs.csv");
    planner.saveFinalSolutionsToCSV();
    std::cout << "Saving paths results to CSV ..." << std::endl;
    //planner.savePathsToCSV("pareto_paths.csv");
    return 0;

}
// --- Silverwares (utility functions)  ---
double normalizeAngle(double angle) {
    angle = fmod(angle + PI, 2.0 * PI);
    if (angle < 0.0) angle += 2.0 * PI;
    return angle - PI;

}

double stateDistance(const State& s1, const State& s2) {
    double dx = s1.x - s2.x;
    double dy = s1.y - s2.y;

    double dtheta = normalizeAngle(s1.theta - s2.theta);

    return std::sqrt(dx * dx + dy * dy + 1.0 * (dtheta * dtheta));
}

State sampleState() {
    static std::mt19937 gen(std::random_device{}());
    static std::uniform_real_distribution<> posx(0, X_MAX);
    static std::uniform_real_distribution<> posy(0, Y_MAX);
    static std::uniform_real_distribution<> ang_dis(-PI, PI);
    return {posx(gen), posy(gen), ang_dis(gen)};
}

bool isObstacleFree(const Trajectory& traj) {
    // in this case, always true
    return true;
}
Trajectory Line(const State& s_from, const State& s_to) {
    Trajectory traj;
    traj.path.push_back(s_from);
    double dist = std::hypot(s_to.x - s_from.x, s_to.y - s_from.y);
    int num_steps = static_cast<int>(dist / 1.0); // 1.0 : step size
    if (num_steps < 1) num_steps = 1;

    for (int i = 1; i <= num_steps; ++i) {
        double ratio = static_cast<double>(i) / num_steps;
        State intermediate;
        intermediate.x = s_from.x + ratio * (s_to.x - s_from.x);
        intermediate.y = s_from.y + ratio * (s_to.y - s_from.y);
        intermediate.theta = normalizeAngle(s_from.theta + ratio * normalizeAngle(s_to.theta - s_from.theta));
        traj.path.push_back(intermediate);
    }
    traj.duration = dist; // assuming constant speed of 1 unit/sec
    return traj;
}
std::vector<double> calculateLineCost(const Trajectory& traj) {
    if (traj.path.size() < 2) return {0.0, 0.0};
    double path_length = 0.0;
    for (size_t i = 0; i < traj.path.size() - 1; ++i) {
        path_length += std::sqrt(std::hypot(traj.path[i+1].x - traj.path[i].x, traj.path[i+1].y - traj.path[i].y));
    }
    double total_angle_change = 0.0;
    for (size_t i = 0; i < traj.path.size() - 1; ++i) {
        double angle_diff = normalizeAngle(traj.path[i+1].theta - traj.path[i].theta);
        total_angle_change += angle_diff * angle_diff; 
    }
    double smoothness_cost = total_angle_change;
    // 128 :  90 * sqrt (2) Straight line , 2.4674 : empirical max value, 90 degree turn
    return {path_length/1, smoothness_cost/1};
    //return {path_length/128, smoothness_cost/4.3865};
}

double calculateTchebycheffFitness(const std::vector<double>& cost_vec, const std::vector<double>& lambda, const std::vector<double>& z_utop) {
    double max_val = -1.0;
    for (size_t k = 0; k < cost_vec.size(); ++k) {
        max_val = std::max(max_val, lambda[k] * std::abs(cost_vec[k] - z_utop[k]));
        //std::cout << "Cost[" << k << "]: " << cost_vec[k] << ", Lambda[" << k << "]: " << lambda[k] << ", z_utop[" << k << "]: " << z_utop[k] << std::endl;
    }
    return max_val;
}
