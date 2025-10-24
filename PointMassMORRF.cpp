// Point Mass Multi Objective RRForest* // 
// Edited by SeungKeol Ryu 
// Oct 1. 2025 // 
#include <iostream>
#include <vector>
#include <cmath>
#include <unordered_set>
#include <sstream>
#include <memory>
#include <limits>
#include <random>
#include <algorithm>
#include <fstream>
#include <map>
#include <stdexcept> 

// Basic constatns // 
double PI = 3.14159265358979323846;
double X_MAX = 21.0;
double Y_MAX = 21.0;

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
struct Trajectory{
    std::vector<State> path;    
};
struct SolutionSet{
    std::vector<double> cost_vector;
    std::vector<State> path;
    double fitness;
};


// std::vector<double> + overloadings
std::vector<double> operator+(const std::vector<double>& a, const std::vector<double>& b) {

    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of the same size to be added.");
    }

    std::vector<double> result;
    result.reserve(a.size());

    for (size_t i = 0; i < a.size(); ++i) {
        result.push_back(a[i] + b[i]);
    }

    return result;
}
// --- Utility Functions  --- // 
double stateDistance(const State& s1, const State& s2) {
    double dx = s1.x - s2.x;
    double dy = s1.y - s2.y;
    return std::sqrt(dx * dx + dy * dy);
}
// Trajectory steer(const State& s_from, const State& s_to);
Trajectory Line(const State& s_from, const State& s_to);
bool isObstacleFree(const Trajectory& traj); // now : always true.
double calculateTchebycheffFitness(const std::vector<double>& cost_vec, const std::vector<double>& lambda, const std::vector<double>& z_utop);
std::vector<double> calculateSegmentCost(const State& s_from, const State& s_to);
// Tree class
class Tree{
    public: 
    // Indicates the map of the relationship of "parent" and "children"
    std::map<int, int> parent_map;
    // For propagation of cost changes. 
    std::map<int, std::vector<int>> children_map;
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
        std::cout << "Node ID " << node_id << " not found in cost_map. Returning infinity vector." << std::endl;
        return std::vector<double>(num_objectives, std::numeric_limits<double>::infinity());
    }
};

class MORRFplanner{
    public:
    State start;
    State goal;
    double threshold;
    int num_objectives;
    int num_sub;
    
    MORRFplanner(const State& start, const State& goal, double threshold, int number_objectives, int number_subproblems): 
    start(start), goal(goal), threshold(threshold),num_objectives(number_objectives), num_sub(number_subproblems){
        // Add the start node
        auto root_node = std::make_shared<Node> (0, start);
        G_nodes.push_back(root_node);
        auto zero_cost = std::vector<double>(num_objectives, 0.0);
        //Reference trees Initialization
        reference_trees.resize(num_objectives);
        for (size_t k = 0; k < num_objectives; ++k){
            auto initial_cost = std::vector<double>(num_objectives,std::numeric_limits<double>::infinity());
            reference_trees[k].cost_map[0] = zero_cost;
        }
        //Subproblem trees Initialization
        subproblem_trees.resize(num_sub);
        for (size_t k = 0; k < num_sub; ++k){
            // if (subproblem_trees[k].cost_map.count(0) != 0){
            //     subproblem_trees[k].cost_map[0] = std::vector<double>(num_objectives, 0.0);
            // }
            subproblem_trees[k].cost_map[0] = std::vector<double>(num_objectives, 0.0);
            subproblem_trees[k].fitness_map[0] = 0.0;
        }
        // Lambda : uniform distributions
        lambdas.resize(num_sub);
        for (size_t k = 0; k < num_sub; ++k){
            lambdas[k].resize(num_objectives);
            double kk = k;
            double val = kk / (num_sub - 1);
            std::vector<double> ref = {1.0,1.0};
            lambdas[k][0] = (1.0 - val) / ref[0];
            lambdas[k][1] = val / ref[1];
        }

    }
    
    bool isAncestor(const std::map<int, int>& parent_map, int start_node_id, int potential_ancestor_id) {
        if (start_node_id == potential_ancestor_id) {
            return true;
        }
        int current_node_id = start_node_id;
        
        while (parent_map.count(current_node_id)) { 
            current_node_id = parent_map.at(current_node_id);
            //std::cout << "Checking Node ID: " << current_node_id << std::endl;
            if (current_node_id == 0) {
                return false;
            }
            if (current_node_id == potential_ancestor_id) {
                return true; 
            }
            if (current_node_id == parent_map.at(current_node_id)){
                //std::cout << "Cycle detected in parent_map!" << std::endl;
                return true; // Cycle detected
            }

        }
        return false; 
    }

    std::vector<std::shared_ptr<Node>> getNearNodes(const State&s , double radius){
        std::vector<std::shared_ptr<Node>> near_nodes;
        for (const std::shared_ptr<Node>& node : G_nodes){
            if (stateDistance(node->state, s) <= radius){
                if( node->state.x == s.x && node->state.y == s.y) continue;
                near_nodes.push_back(node);

            }
        }
        return near_nodes;
    }

    std::shared_ptr<Node> getNearestNode(const State&s, double radius){
        std::shared_ptr<Node> nearest;
        double current_min_dist = std::numeric_limits<double>::infinity();
        if (getNearNodes(s,radius).size() == 0) {
            for (const std::shared_ptr<Node>& node : G_nodes){
                double dist = stateDistance(node->state,s);
                if( node->state.x == s.x && node->state.y == s.y) {
                    continue;
                }
                else if (dist <= current_min_dist){
                
                    current_min_dist = dist;
                    nearest = node;  
                }
            }
        }
        else {
            for (const std::shared_ptr<Node>& node : getNearNodes(s,radius)){
                double dist = stateDistance(node->state,s);
                if( node->state.x == s.x && node->state.y == s.y) continue;
                    if (dist <= current_min_dist){   
                        current_min_dist = dist;
                        nearest = node;
                    }
            }
        }
        return nearest;
    }
    std::vector<SolutionSet> getSolutions(){
        std::vector<SolutionSet> final_solutions;
        SolutionSet final_solution;

        // Search for each subproblem tree
        for (int k = 0; k < num_sub; ++k){
            Tree tree = subproblem_trees[k];
            std::cout << "Searching for subproblem tree " << k << std::endl;
            //for (const Tree& tree : subproblem_trees){
            
            std::vector<double> min_cost = std::vector<double>(num_objectives,std::numeric_limits<double>::infinity());
            double min_fitness = std::numeric_limits<double>::infinity();
            // For detecting the cycle.
            std::unordered_set<int> visited_nodes;
            int best_node_id = -1;

            std::vector<std::shared_ptr<Node>> candidates_nodes;
            for (const std::shared_ptr<Node>& node: G_nodes){
                double dist = stateDistance(node->state, goal);
                
                if (dist < threshold){
                    candidates_nodes.push_back(node);
                    std::cout << "candidate node ID: " << node->id << " at distance " << dist << std::endl;
                    std::cout << "z_utop: " << z_utop[0] << ", " << z_utop[1] << std::endl;
                    double current_min_fitness = calculateTchebycheffFitness(tree.getCost(node->id), lambdas[k], z_utop);
                    //double current_min_fitness = tree.fitness_map.at(node->id);
                    //std::cout << "size of fitness_map: " << tree.fitness_map.size() << std::endl;
                    if (current_min_fitness < min_fitness){
                        min_fitness = current_min_fitness;
                        best_node_id = node->id;
                        std::cout << "Best Node ID updated to: " << best_node_id << std::endl;
                        min_cost = tree.cost_map.at(best_node_id);
                        std::cout  << "min_cost updated to: " << min_cost[0] << ", " << min_cost[1] << std::endl;
                        std::cout << "min_fitness updated to: " << min_fitness << std::endl;
                    }
                }
                else{
                    continue;
                }
                //std::cout << "Best Node ID updated to: " << best_node_id << std::endl;
            }
            std::vector<State> each_best_path;
            //visited_nodes.push_back(start->id);
            std::cout << "BackTracking starts" << std::endl;
            while (tree.parent_map.count(best_node_id) > 0 && best_node_id != 0){
                
                std::cout << "Best Node ID: " << best_node_id << std::endl;
                // checking the cycle
                if (visited_nodes.count(best_node_id)){
                    std::cout << "Best Node ID: " << best_node_id << std::endl;
                    std::cout << "Parent Node ID: " << tree.parent_map.at(best_node_id) << std::endl;
                    std::cout << "Cycle detected in the path!" << std::endl;
                    break;
                }

                each_best_path.push_back(G_nodes[best_node_id]->state);
                // checking the cycle
                visited_nodes.insert(best_node_id);
               
                // backtracking
                best_node_id = tree.parent_map.at(best_node_id);
                std::cout << "Backtracked to Node ID: " << best_node_id << std::endl;
            }
            each_best_path.push_back(G_nodes[0]->state);
            std::cout << "visited_nodes size: " << visited_nodes.size() << std::endl;
            std::cout << "While loop ends" << std::endl;
            final_solution.cost_vector = min_cost;
            final_solution.path = each_best_path;
            final_solution.fitness = min_fitness;
            final_solutions.push_back(final_solution);
        }
        return final_solutions;
    }
    void saveCostsToCSV(const std::string& filename){
        std::ofstream file(filename);
        if (!file.is_open()){
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return;
        }
        std::vector<SolutionSet> Results = getSolutions();
        file << "Length, Cost1, Paths.x, Paths.y ,Fitness, \n";
        for (const SolutionSet& result : Results){
            file << result.cost_vector[0] << "," << result.cost_vector[1] << ",";
            std::stringstream path_x_stream;
            std::stringstream path_y_stream;
            for (size_t i = 0; i < result.path.size(); ++i){
                path_x_stream << result.path[i].x;
                path_y_stream << result.path[i].y;
                if (i != result.path.size() - 1){
                    path_x_stream << ";";
                    path_y_stream << ";";
                }
            }
            file << "\"" << path_x_stream.str() << "\","
             << "\"" << path_y_stream.str() << "\","
             << result.fitness << "\n";
        }

    }
    void run(int max_iterations);
    void saveTreesToTxt(const std::string& prefix = "tree_data") const;
    private: 
    // Need to be protected.
    std::vector<std::shared_ptr<Node>> G_nodes;
    std::vector<Tree> reference_trees;
    std::vector<Tree> subproblem_trees;
    std::vector<std::vector<double>> lambdas;
    std::vector<double> ref = {1.0,1.0};
    //std::vector<double> z_utop = {std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()};
    std::vector<double> z_utop = {0.0,0.0};

};
void MORRFplanner::saveTreesToTxt(const std::string& prefix) const {
    // Subproblem Trees -> .txt file
    for (size_t i = 0; i < subproblem_trees.size(); ++i) {
        std::string filename = prefix + "_subproblem_" + std::to_string(i) + ".txt";
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open " << filename << std::endl;
            continue;
        }

        file << "Parent Map for Subproblem Tree " << i << std::endl;
        file << "======================================" << std::endl;
        file << "Child ID (x, y) -> Parent ID (x, y)" << std::endl;
        
        // 
        for (const auto& [child_id, parent_id] : subproblem_trees[i].parent_map) {
            if (child_id < G_nodes.size() && parent_id < G_nodes.size()) {
                const State& child_state = G_nodes[child_id]->state;
                const State& parent_state = G_nodes[parent_id]->state;
                // "Child node -> Parent node" format
                file << "    " << child_id 
                     << " (" << child_state.x << ", " << child_state.y << ")"
                     << " -> " 
                     << parent_id 
                     << " (" << parent_state.x << ", " << parent_state.y << ")" 
                     << std::endl;
            }
        }
        
        file.close();
        std::cout << "Saved parent map with states to " << filename << std::endl;
    }
};
State sampleState(){
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis_x(0.0, X_MAX);
    static std::uniform_real_distribution<> dis_y(0.0, Y_MAX);

    State s;
    s.x = dis_x(gen);
    s.y = dis_y(gen);
    return s;

}
// In this case, steering function is a line segment function. 
State steer(const State&s_from, const State&s_to){
    double direction = atan2(s_to.y - s_from.y, s_to.x - s_from.x);
    State new_state;
    double eta = 2.0;
    new_state.x = s_from.x + eta * std::cos(direction);
    new_state.y = s_from.y + eta * std::sin(direction);
    return new_state;
}
Trajectory line(const State& s_from, const State& s_to){
    Trajectory traj;
    traj.path.push_back(s_from);
    double dist = std::hypot(s_to.x - s_from.x, s_to.y - s_from.y);
    int num_steps = 11;

    for (int i = 1; i < num_steps; ++i){
        double ratio = i / num_steps;
        State intermediate;
        intermediate.x = s_from.x + ratio * (s_to.x - s_from.x);
        intermediate.y = s_from.y + ratio * (s_to.y - s_from.y);
        traj.path.push_back(intermediate);
    }
    traj.path.push_back(s_to);
    return traj;
}
bool isObstacleFree(const Trajectory& traj){
    return true;
}
void propagateCostToChildren(Tree& tree, int parent_id) {
    
    double parent_cost = tree.cost_map[parent_id];

    if (tree.children_map.count(parent_id) == 0) {
        return; 
    }

    for (int child_id : tree.children_map[parent_id]) {
        
        double segment_cost = calculateSegmentCost(G_nodes[parent_id]->state, G_nodes[child_id]->state);
        double new_child_cost = parent_cost + segment_cost;

        tree.cost_map[child_id] = new_child_cost;
        propagateCostToChildren(tree, child_id);
    }
}

void printParentMap(const std::map<int, int>& parent_map, const std::string& tree_name) {
    std::cout << "--- Printing Parent Map for: " << tree_name << " ---" << std::endl;
    if (parent_map.empty()) {
        std::cout << "Map is empty." << std::endl;
        return;
    }
    // C++17 structured binding
    for (const auto& [child, parent] : parent_map) {
        std::cout << "  Node " << child << " -> (Parent) " << parent << std::endl;
    }
    std::cout << "-----------------------------------------" << std::endl;
}
void printChildrenMap(const std::map<int, std::vector<int>>& children_map, const std::string& tree_name) {
    std::cout << "--- Printing Children Map for: " << tree_name << " ---" << std::endl;
    if (children_map.empty()) {
        std::cout << "Map is empty." << std::endl;
        return;
    }
    // C++17 structured binding
    for (int child_id : tree.children_map[parent_id]) {
        std::cout << "  Parent_id : " << parent_id << " -> Children " << child_id << std::endl;
    }
    std::cout << "-----------------------------------------" << std::endl;
}
//std::vector<std::shared_ptr<Node>> MORRFplanner::ExtendTrees(G_nodes,)
void MORRFplanner::run(int max_iterations){
    // Tree Initialization is done
    // Start the main loop
    for (size_t i = 0; i < max_iterations; ++i){
        State x_rand = sampleState();
        std::cout << "------ Iteration " << i << " | Sampled State: (" << x_rand.x << ", " << x_rand.y << ")" << std::endl;
        double radius = 30.0 * std::sqrt((std::log(G_nodes.size() + 1.0) / (G_nodes.size() + 1.0)));
        
        //double radius =  5.0;
        std::shared_ptr<Node> NstNode = getNearestNode(x_rand,radius);
        std::cout<<"NstNode ID: " << NstNode->id << std::endl;
        
        const State& new_state = steer(NstNode->state, x_rand);

        // Add the new node to G_nodes

        int new_node_id = G_nodes.size();
        auto new_node = std::make_shared<Node>(new_node_id, new_state);
        G_nodes.push_back(new_node); // Line 2 <- Extend Ref
        if (isObstacleFree(line(NstNode->state,new_state))){
            // REFERENCE TREES EXTEND -- Oct 3rd. 2025

            for (size_t k = 0; k < num_objectives; ++k){
                if (new_node->id == NstNode->id) continue; // Line 1
                Tree& tree = reference_trees[k];
                
                State x_min = NstNode->state;
                int id_min = NstNode->id; // Line 3

                tree.parent_map[new_node->id] = NstNode->id; // Line 3
                tree.children_map[NstNode->id].push_back(new_node->id); // For cost propagation
                tree.cost_map[NstNode->id] = tree.getCost(NstNode->id); // Line 3
                tree.cost_map[new_node->id] = tree.cost_map[NstNode->id] + calculateSegmentCost(NstNode->state,new_node->state); // Line 3
                
                std::vector<std::shared_ptr<Node>> NrNodes = getNearNodes(x_min, radius); // Line 4
                for (std::shared_ptr<Node> NrNode:NrNodes){ // Line 5
                    if (isObstacleFree(line(new_node->state, NrNode->state))){ // Line 6

                        double ck_new = tree.getCost(NrNode->id)[k] // Line 7
                        + calculateSegmentCost(NrNode->state,new_node->state)[k];
                        std::cout << "NewNode's cost : " << tree.getCost(new_node->id)[k] << std::endl;
                        std::cout << "Ck_new : " << ck_new << std::endl;
                        if (ck_new < tree.cost_map[new_node->id][k]){ // Line 8
                            x_min = NrNode->state;  // Line 9
                            id_min = NrNode->id; // Line 9
                            tree.parent_map[new_node_id] = id_min;
                            //tree.children_map[id_min].push_back(new_node_id); // For cost propagation.
                            //double segment_cost_k = calculateSegmentCost(G_nodes[id_min]->state, new_node->state)[k];
                            if (tree.cost_map.count(new_node->id) == 0) {
                                tree.cost_map[new_node->id][k] = std::numeric_limits<double>::infinity();
                            }
                            tree.cost_map[new_node->id][k] = ck_new;

                            std::cout << "NewNode ID "<< new_node->id << " updated parent to " << id_min << " with cost " << tree.cost_map[new_node->id][k] << std::endl;
                        }
                    }

                }
                tree.children_map[id_min].push_back(new_node->id); // For cost propagation.
                // Rewiring step
                for (std::shared_ptr<Node> NrNode: NrNodes){
                    if (NrNode->id == id_min) continue; // Line 11
                    if (isObstacleFree(line(new_node->state,NrNode->state))){
                        double ck_new_2 = tree.getCost(new_node->id)[k] 
                        + calculateSegmentCost(new_node->state,NrNode->state)[k];
                        if (ck_new_2 < tree.getCost(NrNode->id)[k]){
                            if (!isAncestor(tree.parent_map,new_node->id, NrNode->id)){
                                tree.parent_map[NrNode->id] = new_node_id; // Line 15-17
                                tree.children_map[new_node->id].push_back(NrNode->id); // For cost propagation.
                                tree.cost_map[NrNode->id][k] = ck_new_2; // Line 15-17
                                propagateCostToChildren(tree, NrNode->id); // Propagate cost changes

                                }
                            }
                        }
                    }
                }
            }


            double min_cost_0 = std::numeric_limits<double>::infinity();
            for (auto const& [node_id, cost_vector] : reference_trees[0].cost_map) {
                if (stateDistance(G_nodes[node_id]->state, goal) < threshold) {
                    if (cost_vector[0] < min_cost_0) {
                        min_cost_0 = cost_vector[0];
                        z_utop[0] = min_cost_0; 
                    }
                }
            }
            double min_cost_1 = std::numeric_limits<double>::infinity();
            for (auto const& [node_id, cost_vector] : reference_trees[1].cost_map) {
                if (stateDistance(G_nodes[node_id]->state, goal) < threshold) {
                    if (cost_vector[1] < min_cost_1) {
                        min_cost_1 = cost_vector[1];
                        z_utop[1] = min_cost_1;
                    }
                }
            }
            // Update the utopian point
            if( i > 0 && i % 200 == 0){
                std::cout << "Iter " << i << " | z_utop : " << z_utop[0] << ", " << z_utop[1] << std::endl;
            }
            std::cout << "-------- SubProblem Trees ... -----------" << std::endl;
            std::cout << "Size of G_nodes : " << G_nodes.size() << std::endl;
            
            // SUBPROBLEM TREES EXTEND -- Oct 5rd. 2025
            for (size_t k = 0; k < num_sub; ++k){
                std::cout << "-------Subproblem tree " << k << " ------------" << std::endl;
                if (new_node->id == NstNode->id) continue; // Line 1
                Tree& tree = subproblem_trees[k];

                int id_min = NstNode->id; // Line 3
                State x_min = NstNode->state; // Line 3
                tree.parent_map[new_node->id] = NstNode->id;
                tree.children_map[NstNode->id].push_back(new_node->id);
                std::vector<std::shared_ptr<Node>> NrNodes = getNearNodes(x_min, radius); // Line 4
                tree.cost_map[NstNode->id] = tree.getCost(NstNode->id); // Line 3
                std::cout << "Nst Node "<< NstNode->id <<  " State: " << NstNode->state.x << ", " << NstNode->state.y << std::endl;
                std::cout << "Cost map of " << NstNode->id << " : " << tree.cost_map[NstNode->id][0] << ", " << tree.cost_map[NstNode->id][1] << std::endl;
                tree.cost_map[new_node->id] = tree.cost_map[NstNode->id] + calculateSegmentCost(NstNode->state,new_node->state); // Line 3
                std::cout << "New Node "<<new_node->id <<  " State: " << new_node->state.x << ", " << new_node->state.y << std::endl;
                std::cout << "Cost map of " << new_node->id << " : " << tree.cost_map[new_node->id][0] << ", " << tree.cost_map[new_node->id][1] << std::endl;
                
                //std::cout << "NrNodes size: " << NrNodes.size() << std::endl;
                //std::cout << "subproblem tree's size: " << tree.cost_map.size() << std::endl;
                // Checking the new node can find a better parent among near nodes.
                for (std::shared_ptr<Node> NrNode:NrNodes){
                    if (isObstacleFree(line(new_node->state,NrNode->state))){
                        std::vector<double> cost_vec = tree.getCost(NrNode->id)
                        + calculateSegmentCost(NrNode->state,new_node->state); // Line 7
                        std::cout << "COST_VEC OF NEW NODE through NrNode : " << cost_vec[0] << ", " << cost_vec[1] << std::endl;
                        double eta_current = calculateTchebycheffFitness(cost_vec, lambdas[k], z_utop); // Line 8
                        std::cout << "ETA CURRENT : " << eta_current << std::endl;
                        std::vector<double> new_cost = tree.getCost(new_node->id); // Line 9
                        std::cout << "COST VECTOR OF NEW NODE : " << new_cost[0] << ", " << new_cost[1] << std::endl;
                        double eta_new = calculateTchebycheffFitness(new_cost, lambdas[k], z_utop); // Line 10
                        std::cout << "ETA NEW : " << eta_new << std::endl;
                        if (eta_current < eta_new){
                            x_min = NrNode->state;
                            id_min = NrNode->id;

                            tree.parent_map[new_node_id] = id_min; // Line 13
                            tree.cost_map[new_node->id] = cost_vec; // Line 13
                            std::cout << "NrNode ID: " << NrNode->id << std::endl;
                            std::cout << "cost_Vec " << cost_vec[0] << ", " << cost_vec[1] << std::endl;
                            std::cout << "z_utop " << z_utop[0] << ", " << z_utop[1] << std::endl;
                            std::cout << "eta_current " << eta_current << " eta_new " << eta_new << std::endl;
                        }
                    }
                }
                //tree.parent_map[new_node_id] = id_min; // Line 13
                //tree.cost_map[new_node->id] = tree.getCost(id_min) + calculateSegmentCost(x_min,new_node->state); // Line 3
                std::cout << "After New Node "<<new_node->id <<  " State: " << new_node->state.x << ", " << new_node->state.y << std::endl;
                std::cout << "After Cost map of " << new_node->id << " : " << tree.cost_map[new_node->id][0] << ", " << tree.cost_map[new_node->id][1] << std::endl;
                
                // Rewiring step : checking the new node can be a parent of near nodes.
                // Propagation of cost change by switiching the parent.
                //std::cout << "Edge set is updated" << std::endl;
                for (std::shared_ptr<Node> NrNode:NrNodes){
                    if (NrNode->id == id_min) continue; // Line 11
                    if (isObstacleFree(line(new_node->state,NrNode->state))){
                        std::vector<double> cost_vec_2 = tree.getCost(new_node->id)
                        + calculateSegmentCost(new_node->state,NrNode->state);
                        double eta_current_2 = calculateTchebycheffFitness(cost_vec_2, lambdas[k], z_utop);
                        std::vector<double> near_cost = tree.getCost(NrNode->id);
                        double eta_near = calculateTchebycheffFitness(near_cost, lambdas[k], z_utop);
                        if (eta_current_2 < eta_near){
                            if (!isAncestor(tree.parent_map, new_node->id, NrNode->id)){
                                tree.parent_map[NrNode->id] = new_node->id; // Line 21-23
                                tree.cost_map[NrNode->id] = cost_vec_2; // Line 21-23
                           }
                        }
                    }
                }
            }
        }
    }
}
double calculateTchebycheffFitness(const std::vector<double>& cost_vec, const std::vector<double>& lambda, const std::vector<double>& z_utop) {
    double max_val = -1.0;
    // Improve the worst element.
    for (size_t k = 0; k < cost_vec.size(); ++k) {
        max_val = std::max(max_val, lambda[k] * std::abs(cost_vec[k] - z_utop[k]));
    }
    
    return max_val;
}

std::vector<double> calculateSegmentCost(const State& s_from, const State& s_to){
    std::vector<double> cost(2.0,0.0);
    cost[0] = stateDistance(s_from,s_to);

    // double sx = (s_from.x + s_to.x) / 2;
    // double sy = (s_from.y + s_to.y) / 2;
    // const double obstacle_cx = 10.0;
    // const double obstacle_cy = 5.0;
    // const double radius = 5.0;
    // const double max_risk = 10000;

    // double dx = sx - obstacle_cx;
    // double dy = sy - obstacle_cy;

    // double distance = std::sqrt(dx * dx + dy * dy);

    // if (distance <= radius) {
    //     //cost[1] =  std::numeric_limits<double>::infinity();
    //     cost[1] = max_risk;
    // } else {
    //     //cost[1] = 1.0 / distance;
    //     cost[1] = distance;
    // }
    cost[1] = cost[0];
    return cost;
}

int main(){

    State start = {0.0, 1.0};
    State goal = {20.0,15.0}; 
    int num_objectives = 2;
    int num_subproblems = 3;
    double threshold = 0.5;
    int iterations = 5000;
    MORRFplanner planner(start, goal, threshold, num_objectives, num_subproblems);
    // 3. running the planner
    std::cout << "Starting MORRF* planning with point mass..." << std::endl;
    std::cout << "The number of subproblems : " << num_subproblems << "   " << "Max Iterations :" << iterations << std::endl;

    planner.run(iterations);
    
    // 4. saving results to csv 
    std::cout << "Planning finished." << std::endl;
    //planner.saveCostsToCSV("pareto_costs.csv");
    std::cout << "Saving tree structures to .txt files..." << std::endl;
    planner.saveTreesToTxt("final_trees_data");
    planner.saveCostsToCSV("results.csv");
    std::cout << "Saving paths results to CSV ..." << std::endl;
    return 0;

};

