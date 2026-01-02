// Kinematic Point Mass Multi Objective RRForest* // 
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
#include <queue>
#include <cstdlib>


// Basic constatns // 
double PI = 3.14159265358979323846;
double X_MAX = 25.0;
double Y_MAX = 30.0;

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
    std::vector<double> weight;
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
bool isObstacleFree(const Trajectory& traj); // Not always true. 
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
    int num_objectives = 3;
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
    int num_divs;
    
    MORRFplanner(const State& start, const State& goal, double threshold, int number_objectives, int divisions): 
    start(start), goal(goal), threshold(threshold),num_objectives(number_objectives), num_divs(divisions){
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

        // Lambda : uniform distributions
        lambdas.clear();
        for (int i = 0; i <= num_divs; ++i) {
            for (int j = 0; j <= num_divs - i; ++j) {
                int k = num_divs - i - j; 

                std::vector<double> vec(3);
                vec[0] = (double)i / num_divs;
                vec[1] = (double)j / num_divs;
                vec[2] = (double)k / num_divs;

                lambdas.push_back(vec);
            }
        }
        // lambdas.clear();
        // std::vector<std::vector<double>>Test_Lambda_Set = 
        //                     {{0.0, 0.0, 1.0},
        //                     {0.0, 1.0, 0.0},
        //                     {1.0, 0.0, 0.0},
        //                     {0.99, 0.0, 0.01},
        //                     {0.0, 0.99, 0.01},
        //                     {0.01, 0.99, 0.0},
        //                     {0.01, 0.0, 0.99},
        //                     {0.0, 0.01, 0.99},
        //                     {0.99, 0.01, 0.0},
        //                     {0.98, 0.0, 0.02},
        //                     {0.0, 0.98, 0.02},
        //                     {0.02, 0.98, 0.0},
        //                     {0.02, 0.0, 0.98},
        //                     {0.0, 0.02, 0.98},
        //                     {0.98, 0.02, 0.0},
        //                     {0.97, 0.0, 0.03},
        //                     {0.0, 0.97, 0.03},
        //                     {0.03, 0.97, 0.0},
        //                     {0.03, 0.0, 0.97},
        //                     {0.0, 0.03, 0.97},
        //                     {0.97, 0.03, 0.0},
        //                     {0.9, 0.05, 0.05},
        //                     {0.05, 0.9, 0.05},
        //                     {0.05, 0.9, 0.05},
        //                     {0.05, 0.05, 0.9},
        //                     {0.05, 0.05, 0.9},
        //                     {0.9, 0.05, 0.05},
        //                     {1/3, 1/3, 1/3}};
        // lambdas = Test_Lambda_Set;
        std::cout << "The number of lambdas generated: " << lambdas.size() << std::endl;
        int num_sub = lambdas.size();
        num_sub = (num_divs + 2) * (num_divs + 1) /2;
        if (num_objectives == 3 && num_sub != lambdas.size()){
            throw std::runtime_error("Error in generating lambdas for 3 objectives.");
        };
        //Subproblem trees Initialization
        subproblem_trees.resize(num_sub);
        for (size_t k = 0; k < num_sub; ++k){
            // if (subproblem_trees[k].cost_map.count(0) != 0){
            //     subproblem_trees[k].cost_map[0] = std::vector<double>(num_objectives, 0.0);
            // }
            subproblem_trees[k].cost_map[0] = std::vector<double>(num_objectives, 0.0);
            subproblem_trees[k].fitness_map[0] = 0.0;
        }
        std::cout << "The number of subproblems : " << num_sub << std::endl;


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
        int num_sub;
        num_sub = (num_divs + 2) * (num_divs + 1) /2;
        // Search for each subproblem tree
        for (int k = 0; k < num_sub; ++k){
            Tree tree = subproblem_trees[k];
            //std::cout << "Searching for subproblem tree " << k << std::endl;
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
                    //std::cout << "candidate node ID: " << node->id << " at distance " << dist << std::endl;
                    //std::cout << "z_utop: " << z_utop[0] << ", " << z_utop[1] << std::endl;
                    double current_min_fitness = calculateTchebycheffFitness(tree.getCost(node->id) + calculateSegmentCost(node->state,goal), lambdas[k], z_utop);
                    //double current_min_fitness = tree.fitness_map.at(node->id);
                    //std::cout << "size of fitness_map: " << tree.fitness_map.size() << std::endl;
                    if (current_min_fitness < min_fitness){
                        min_fitness = current_min_fitness;
                        best_node_id = node->id;
                        //std::cout << "Best Node ID updated to: " << best_node_id << std::endl;
                        min_cost = tree.cost_map.at(best_node_id);
                        //std::cout  << "min_cost updated to: " << min_cost[0] << ", " << min_cost[1] << std::endl;
                        //std::cout << "min_fitness updated to: " << min_fitness << std::endl;
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
                
                //std::cout << "Best Node ID: " << best_node_id << std::endl;
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
                //std::cout << "Backtracked to Node ID: " << best_node_id << std::endl;
            }
            each_best_path.push_back(G_nodes[0]->state);
            std::cout << "visited_nodes size: " << visited_nodes.size() << std::endl;
            std::cout << "While loop ends" << std::endl;
            final_solution.cost_vector = min_cost;
            final_solution.path = each_best_path;
            final_solution.fitness = min_fitness;
            final_solution.weight = lambdas[k];
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
        file << "Length, Risk, TravelTime, Paths.x, Paths.y , Fitness, Weights \n";
        for (const SolutionSet& result : Results){
            file << result.cost_vector[0] << "," << result.cost_vector[1] << "," << result.cost_vector[2] << ",";
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
             std::stringstream weights_stream;
            for (size_t i = 0; i < result.weight.size(); ++i) {
                weights_stream << result.weight[i];
                if (i != result.weight.size() - 1) {
                    weights_stream << ";"; 
                }
            }
            file << "\"" << weights_stream.str() << "\"\n";
        }

    }
    void run(int max_iterations);
    void saveParentMapToTxt(const std::string& prefix = "tree_data") const;
    void saveChildrenMapToTxt(const std::string& prefix) const;
    private: 
    // Need to be protected.
    std::vector<std::shared_ptr<Node>> G_nodes;
    std::vector<Tree> reference_trees;
    std::vector<Tree> subproblem_trees;
    std::vector<std::vector<double>> lambdas;
    std::vector<double> ref = {1.0,1.0,1.0};
    //std::vector<double> z_utop = {std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()};
    std::vector<double> z_utop = {24.41,0.0,0.0};

};
void MORRFplanner::saveParentMapToTxt(const std::string& prefix) const {
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
        //std::cout << "Saved parent map with states to " << filename << std::endl;
    }
};
void MORRFplanner::saveChildrenMapToTxt(const std::string& prefix) const {
    for (size_t i = 0; i < subproblem_trees.size(); ++i) {
        std::string filename = prefix + "_subproblem_children_" + std::to_string(i) + ".txt";
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open " << filename << std::endl;
            continue;
        }

        file << "Children Map for Subproblem Tree " << i << std::endl;
        file << "======================================" << std::endl;
        file << "Parent ID (x, y) -> [List of Children IDs (x, y)]" << std::endl;
        
        const auto& children_map = subproblem_trees[i].children_map;

        // children_map. (key: parent_id, value: vector<int> children_list)
        for (const auto& [parent_id, children_list] : children_map) {
            if (parent_id >= G_nodes.size()) continue; 
            const State& parent_state = G_nodes[parent_id]->state;
            file << "    " << parent_id 
                 << " (" << parent_state.x << ", " << parent_state.y << ") -> [ ";
            for (int child_id : children_list) {
                if (child_id < G_nodes.size()) {
                    const State& child_state = G_nodes[child_id]->state;
                    file << child_id << " (" << child_state.x << ", " << child_state.y << "); ";
                } else {
                    file << child_id << " (State N/A); ";
                }
            }
            file << "]" << std::endl; 
        }
        
        file.close();
        //std::cout << "Saved children map with states to " << filename << std::endl;
    }
};
State sampleState(){
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis_x(0, +X_MAX);
    static std::uniform_real_distribution<> dis_y(-5, +Y_MAX);
    //static std::uniform_real_distribution<> dis_y(-50, +25);
    State s;
    s.x = dis_x(gen);
    s.y = dis_y(gen);
    if (stateDistance(s,State{11.0,13.0}) >= 3.0){
        return s;
    }

}
// In this case, steering function is a line segment function. 
State steer(const State&s_from, const State&s_to){
    double direction = atan2(s_to.y - s_from.y, s_to.x - s_from.x);
    State new_state;
    double dist = stateDistance(s_from, s_to);
    double eta = 1.0;
    if (dist <= eta) {
        new_state = s_to;
    } else {
        double ratio = eta / dist;
        new_state.x = s_from.x + ratio * (s_to.x - s_from.x);
        new_state.y = s_from.y + ratio * (s_to.y - s_from.y);
    }
    return new_state;
}
Trajectory line(const State& s_from, const State& s_to){
    Trajectory traj;
    traj.path.push_back(s_from);
    double dist = std::hypot(s_to.x - s_from.x, s_to.y - s_from.y);
    int num_steps = 31;

    for (int i = 1; i <= num_steps; ++i){
        double ratio = (double) i / num_steps;
        State intermediate;
        intermediate.x = s_from.x + ratio * (s_to.x - s_from.x);
        intermediate.y = s_from.y + ratio * (s_to.y - s_from.y);
        traj.path.push_back(intermediate);
    } 
    //traj.path.push_back(s_to);
    return traj;
}
bool isObstacleFree(const Trajectory& traj){
    State s_from = traj.path.front();
    State s_to = traj.path.back();
    double threshold = 3.0;
    int num_steps = 31;
    for (int i = 0; i <= num_steps; ++i){
        double ratio = static_cast<double>(i) / num_steps; 
        State intermediate;
        intermediate.x = s_from.x + ratio * (s_to.x - s_from.x);
        intermediate.y = s_from.y + ratio * (s_to.y - s_from.y);
        double dist = stateDistance(intermediate, State{11.0,13.0});
        if (dist <= threshold){
            return false;
        }
        else{
            continue;
        }
    } 
    return true;
}
void propagateCostToChildren(Tree& tree, int parent_id, const std::vector<std::shared_ptr<Node>>& G_nodes) {
    std::queue<int> q;
    q.push(parent_id);
    // Using BFS
    while(!q.empty()){
        int parent_id = q.front();
        q.pop();
        std::vector<double> parent_cost = tree.cost_map[parent_id];
        if (tree.children_map.count(parent_id) == 0) {
            continue; 
        }
        for (int child_id : tree.children_map[parent_id]){
            std::vector<double> segment_cost = calculateSegmentCost(G_nodes[parent_id]->state, G_nodes[child_id]->state);
            std::vector<double> new_child_cost = parent_cost + segment_cost;

            tree.cost_map[child_id] = new_child_cost;

            //std::cout << "Propagated cost to Child ID " << child_id << ": " << std::endl;
            q.push(child_id);

        }
    }
}

//std::vector<std::shared_ptr<Node>> MORRFplanner::ExtendTrees(G_nodes,)
void MORRFplanner::run(int max_iterations){
    // Tree Initialization is done
    // Start the main loop
    for (size_t i = 0; i < max_iterations; ++i){

        State x_rand = sampleState();

        //std::cout << "------ Iteration " << i << " | Sampled State: (" << x_rand.x << ", " << x_rand.y << ")" << std::endl;
        //double search_radius = 30.0 * std::sqrt((std::log(G_nodes.size() + 1.0) / (G_nodes.size() + 1.0)));
        double search_radius = 1.0;
        std::shared_ptr<Node> NstNode = getNearestNode(x_rand,search_radius);
        //std::cout<<"NstNode ID: " << NstNode->id << std::endl;
        
        const State& new_state = steer(NstNode->state, x_rand);

        // Add the new node to G_nodes
        int new_node_id = G_nodes.size();
        auto new_node = std::make_shared<Node>(new_node_id, new_state);
        if (isObstacleFree(line(NstNode->state,new_state))){
            // REFERENCE TREES EXTEND -- Oct 3rd. 2025
            G_nodes.push_back(new_node); // Line 2 <- Extend Ref
            i = G_nodes.size(); // Reset the iteration count based on the size of G_nodes.
            for (size_t k = 0; k < num_objectives; ++k){
                if (new_node->id == NstNode->id) continue; // Line 1
                Tree& tree = reference_trees[k];
                
                State x_min = NstNode->state;
                int id_min = NstNode->id; // Line 3

                tree.parent_map[new_node->id] = NstNode->id; // Line 3
                //tree.children_map[NstNode->id].push_back(new_node->id); // For cost propagation
                tree.cost_map[NstNode->id] = tree.getCost(NstNode->id); // Line 3
                tree.cost_map[new_node->id] = tree.cost_map[NstNode->id] + calculateSegmentCost(NstNode->state,new_node->state); // Line 3
                
                std::vector<std::shared_ptr<Node>> NrNodes = getNearNodes(new_node->state, search_radius); // Line 4
                for (std::shared_ptr<Node> NrNode:NrNodes){ // Line 5
                    if (isObstacleFree(line(new_node->state, NrNode->state))){ // Line 6

                        double ck_new = tree.getCost(NrNode->id)[k] + calculateSegmentCost(NrNode->state,new_node->state)[k];  // Line 7
                        //std::cout << "NewNode's cost : " << tree.getCost(new_node->id)[k] << std::endl;
                        //std::cout << "Ck_new : " << ck_new << std::endl;
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

                            //std::cout << "NewNode ID "<< new_node->id << " updated parent to " << id_min << " with cost " << tree.cost_map[new_node->id][k] << std::endl;
                        }
                    }
                
                }
                tree.children_map[id_min].push_back(new_node->id); // For cost propagation.
                // Rewiring step
                for (std::shared_ptr<Node> NrNode: NrNodes){
                    if (NrNode->id == id_min) continue; // Line 11
                    if (isObstacleFree(line(new_node->state,NrNode->state))){
                        //std::cout << tree.getCost(new_node->id)[k] << std::endl;
                        //std::cout << calculateSegmentCost(new_node->state,NrNode->state)[k] << std::endl;
                        double ck_new_2 = tree.getCost(new_node->id)[k] 
                        + calculateSegmentCost(new_node->state,NrNode->state)[k];
                        if (ck_new_2 < tree.getCost(NrNode->id)[k]){
                            if (!isAncestor(tree.parent_map,new_node->id, NrNode->id)){
                                if (NrNode->id == 0) {
                                continue;
                                }
                                //
                                if (tree.parent_map.count(NrNode->id)) {
                                    int old_parent_id = tree.parent_map.at(NrNode->id);
                                    
                        
                                    if (tree.children_map.count(old_parent_id)) {
                                        auto& old_children_list = tree.children_map.at(old_parent_id);
                                        auto it = std::remove(old_children_list.begin(), old_children_list.end(), NrNode->id);
                                        old_children_list.erase(it, old_children_list.end());
                                    }
                                }
                                //

                                tree.parent_map[NrNode->id] = new_node_id; // Line 15-17
                                tree.children_map[new_node->id].push_back(NrNode->id); // For cost propagation.
                                tree.cost_map[NrNode->id][k] = ck_new_2; // Line 15-17
                                propagateCostToChildren(tree, NrNode->id,G_nodes); // Propagate cost changes

                                }
                            }
                        }
                    }
                }
            }
        else{
            continue;
        }
        //std::cout << "The size of G_nodes : " << G_nodes.size() << std::endl;
        // double min_cost_0 = std::numeric_limits<double>::infinity();
        // for (auto const& [node_id, cost_vector] : reference_trees[0].cost_map) {
        //     if (stateDistance(G_nodes[node_id]->state, goal) < threshold) {
        //         if (cost_vector[0] + calculateSegmentCost(G_nodes[node_id]->state, goal)[0] < min_cost_0) {
        //             min_cost_0 = cost_vector[0] + calculateSegmentCost(G_nodes[node_id]->state, goal)[0];
        //             z_utop[0] = min_cost_0; 
        //             //std::cout << "Node ID : " << node_id << " with cost_vector[0] : " << cost_vector[0] << std::endl;
        //         }

        //     }
        // }
        // double min_cost_1 = std::numeric_limits<double>::infinity();
        // for (auto const& [node_id, cost_vector] : reference_trees[1].cost_map) {
        //     if (stateDistance(G_nodes[node_id]->state, goal) < threshold) {
        //         if (cost_vector[1] + calculateSegmentCost(G_nodes[node_id]->state, goal)[1] < min_cost_1) {
        //             min_cost_1 = cost_vector[1] + calculateSegmentCost(G_nodes[node_id]->state, goal)[1];
        //             z_utop[1] = min_cost_1;
        //         }
        //     }
        // }
        //z_utop = {0.,0.};
        // Update the utopian point
        //std::cout << "-------- SubProblem Trees ... -----------" << std::endl;
        //std::cout << "Size of G_nodes : " << G_nodes.size() << std::endl;
        
        // SUBPROBLEM TREES EXTEND -- Oct 5rd. 2025
        int num_sub;
        num_sub = (num_divs + 2) * (num_divs + 1) /2;
        for (size_t k = 0; k < num_sub; ++k){
            //std::cout << "-------Subproblem tree " << k << " ------------" << std::endl;
            if (new_node->id == NstNode->id) continue; // Line 1
            Tree& tree = subproblem_trees[k];

            int id_min = NstNode->id; // Line 3
            State x_min = NstNode->state; // Line 3
            
            tree.parent_map[new_node->id] = NstNode->id;
            //std::cout << "New Node ID : " << new_node->id << "  NstNode ID : " << NstNode->id << std::endl;
            std::vector<std::shared_ptr<Node>> NrNodes = getNearNodes(new_node->state, search_radius); // Line 4

            //tree.cost_map[NstNode->id] = tree.getCost(NstNode->id); // Line 3
            tree.cost_map[new_node->id] = tree.cost_map[NstNode->id] + calculateSegmentCost(NstNode->state,new_node->state); // Line 3
        
            // Checking the new node can find a better parent among near nodes.
            std::vector<double> min_cost_vec = tree.getCost(new_node->id); 
            if (min_cost_vec != tree.cost_map[new_node->id]){
                std::cout << "Mismatch in cost map for New Node ID " << new_node->id << std::endl;
            }
            // NOT using all path.
            //zhat_utop = z_utop(v);
            //std::cout << "New Node ID " << new_node->id << std::endl;
            std::vector<double> z_hat_utop(num_objectives);
            //std::cout << "The size of reference trees[0] : " << reference_trees[0].cost_map.size() << std::endl;
            //std::cout << "The size of reference trees[1] : " << reference_trees[1].cost_map.size() << std::endl;
            z_hat_utop[0] = reference_trees[0].getCost(new_node->id)[0];
            z_hat_utop[1] = reference_trees[1].getCost(new_node->id)[1];
            z_hat_utop[2] = reference_trees[2].getCost(new_node->id)[2];
            double eta_min = calculateTchebycheffFitness(min_cost_vec, lambdas[k], z_hat_utop);
            //std::cout << "eta minim : " << eta_min << std::endl;
            for (std::shared_ptr<Node> NrNode:NrNodes){
                if (isObstacleFree(line(new_node->state,NrNode->state))){
                    std::vector<double> cost_vec = tree.getCost(NrNode->id) 
                    + calculateSegmentCost(NrNode->state,new_node->state); // Line 7
                    double eta_current = calculateTchebycheffFitness(cost_vec, lambdas[k], z_hat_utop); // Line 8
                    //std::cout << "Nr Node ID : " << NrNode->id << " Cost Vec: " << cost_vec[0] << ", " << cost_vec[1] << "eta CURRENT " << eta_current << std::endl;
                    //std::cout << "z_hat_utop (" << new_node->id << ") : " << z_hat_utop[0] << ", " << z_hat_utop[1] << ", " << z_hat_utop[2] << std::endl;
                    if (eta_current < eta_min){
                        x_min = NrNode->state;
                        id_min = NrNode->id;
                        eta_min = eta_current;
                        min_cost_vec = cost_vec;
                        tree.parent_map[new_node_id] = id_min; // Line 13
                        tree.cost_map[new_node->id] = min_cost_vec; // Line 13
                        //std::cout << "NrNode ID: " << NrNode->id << std::endl;
                        //std::cout << "cost_Vec " << cost_vec[0] << ", " << cost_vec[1] << std::endl;
                        //std::cout << "z_utop " << z_utop[0] << ", " << z_utop[1] << std::endl;
                        //std::cout << "eta_current " << eta_current << " eta_min " << eta_min << std::endl;
                    }
                }
            }
            tree.children_map[id_min].push_back(new_node->id); // For cost propagation, remember the children node.
            
            //tree.parent_map[new_node_id] = id_min; // Line 13
            //tree.cost_map[new_node->id] = tree.getCost(id_min) + calculateSegmentCost(x_min,new_node->state); // Line 3
            //std::cout << "After New Node "<<new_node->id <<  " State: " << new_node->state.x << ", " << new_node->state.y << std::endl;
            //std::cout << "After Cost map of " << new_node->id << " : " << tree.cost_map[new_node->id][0] << ", " << tree.cost_map[new_node->id][1] << std::endl;
            
            // Rewiring step : checking the new node can be a parent of near nodes.
            // Propagation of cost change by switiching the parent.
            //std::cout << "Edge set is updated" << std::endl;
            for (std::shared_ptr<Node> NrNode:NrNodes){
                if (NrNode->id == id_min) continue; // Line 11
                if (isObstacleFree(line(new_node->state,NrNode->state))){
                    std::vector<double> cost_vec_2 = tree.getCost(new_node->id) 
                    + calculateSegmentCost(new_node->state,NrNode->state);
                    std::vector<double> z_hat_utop(num_objectives);
                    z_hat_utop[0] = reference_trees[0].getCost(NrNode->id)[0];
                    z_hat_utop[1] = reference_trees[1].getCost(NrNode->id)[1];
                    z_hat_utop[2] = reference_trees[2].getCost(NrNode->id)[2];
                    double eta_current_2 = calculateTchebycheffFitness(cost_vec_2, lambdas[k], z_hat_utop);
                    std::vector<double> near_cost = tree.getCost(NrNode->id);
                    double eta_near = calculateTchebycheffFitness(near_cost, lambdas[k], z_hat_utop);
                    if (eta_current_2 < eta_near){
                        // Rewiring happens
                        if (NrNode->id == 0) {
                            continue;
                        }
                        if (!isAncestor(tree.parent_map, new_node->id, NrNode->id)){
                            //
                            if (tree.parent_map.count(NrNode->id)) {
                                int old_parent_id = tree.parent_map.at(NrNode->id);
                                
                                if (tree.children_map.count(old_parent_id)) {
                                    auto& old_children_list = tree.children_map.at(old_parent_id);
                                    auto it = std::remove(old_children_list.begin(), old_children_list.end(), NrNode->id);
                                    old_children_list.erase(it, old_children_list.end());
                                }
                            }
                            //
                            //std::cout << "Rewiring: NrNode ID " << NrNode->id << " changes parent to NewNode ID " << new_node->id << std::endl;
                            tree.parent_map[NrNode->id] = new_node->id; // Line 21-23
                            tree.cost_map[NrNode->id] = cost_vec_2; // Line 21-23
                            tree.children_map[new_node->id].push_back(NrNode->id);
                            propagateCostToChildren(tree, NrNode->id,G_nodes); // Propagate cost changes
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
    std::vector<double> cost(3.0,0.0);
    //cost[0] = stateDistance(s_from,s_to)/20.0;
    cost[0] = stateDistance(s_from,s_to);
    const double obstacle_cx = 11.0;
    const double obstacle_cy = 13.0;
    const double radius = 3.0;
    double risk = 0.0;
    int num_steps = 16;
    double intermediate_dist = 0.0;
    State previous_intermediate_risk = State{s_from.x, s_from.y};
    double inverse_risk_segment;
    double sum_segment_risk;
    State intermediate_State_risk;
    State CenterOfSegment;
    double R = 5000;
    for (int i = 1; i <= num_steps; ++i){
        double ratio = (double)i / num_steps;
        
        intermediate_State_risk.x = s_from.x + ratio * (s_to.x - s_from.x);
        intermediate_State_risk.y = s_from.y + ratio * (s_to.y - s_from.y);
        CenterOfSegment = State{
            (intermediate_State_risk.x + previous_intermediate_risk.x) / 2.0,
            (intermediate_State_risk.y + previous_intermediate_risk.y) / 2.0
        };
        // std::cout << "Step " << i << std::endl;
        // std::cout << "Ratio" << ratio << std::endl;
        // std::cout << "Intermediate State: " << intermediate_State_risk.x << ", " << intermediate_State_risk.y << std::endl;
        // std::cout << "Previous Intermediate State: " << previous_intermediate_risk.x << ", " << previous_intermediate_risk.y << std::endl;
        // std::cout << "CenterOfSegment: " << CenterOfSegment.x << ", " << CenterOfSegment.y << std::endl;
        // std::cout << "distance between segment: " << stateDistance(intermediate_State_risk, previous_intermediate_risk) << std::endl;
        inverse_risk_segment = 1/((stateDistance(CenterOfSegment, State{11.0,13.0}) - radius) * (stateDistance(CenterOfSegment, State{11.0,13.0}) - radius)) ;
        if (inverse_risk_segment < 0.0){
            inverse_risk_segment = 0.001;
        }

        previous_intermediate_risk = State{intermediate_State_risk.x, intermediate_State_risk.y};

        sum_segment_risk += inverse_risk_segment;
        //std::cout << "risk " << inverse_risk_segment << std::endl;
    }
    //risk = std::min(R,1.0 /sum_segment_risk)
    //dist /= num_steps; 
    risk = 1. * (sum_segment_risk) * stateDistance(s_from,s_to)/num_steps;
    cost[1] = risk;
    if (risk < 0.0) {
        std::cout << "Risk" << risk << std::endl;
    }
    
    // Cost[2] : travel time.
    State previous_intermediate_traveltime = State{s_from.x, s_from.y};
    double speed;
    double Time;
    double distance_segment;
    State intermediate_traveltime;
    for (int i = 1; i <= num_steps; ++i){
        double ratio = (double)i / num_steps;
        //std::cout << "Ratio" << ratio << std::endl;
        intermediate_traveltime.x = s_from.x + ratio * (s_to.x - s_from.x);
        intermediate_traveltime.y = s_from.y + ratio * (s_to.y - s_from.y);
        if (intermediate_traveltime.y < 13.0){
            speed = 100; // Highway
        }
        else{
            speed = 2;
        }
        distance_segment = stateDistance(intermediate_traveltime, previous_intermediate_traveltime);
        Time += distance_segment / speed;
        previous_intermediate_traveltime = intermediate_traveltime;
    }

    //cost[2] = Time/2.1;
    cost[2] = Time;
    //cost[2] = cost[1];

    return cost;
}

int main(){

    State start = {1.0, 15.0};
    State goal = {21.0,15.0};
    int num_objectives = 3;
    int divisions = 10;
    double threshold = 0.25;
    int iterations = 2000;
    MORRFplanner planner(start, goal, threshold, num_objectives, divisions);
    // 3. running the planner
    std::cout << "Starting MORRF* planning with Kinematic point mass..." << std::endl;

    planner.run(iterations);
    
    // 4. saving results to csv 
    std::cout << "Planning finished." << std::endl;
    std::string folder_suffix = "divisions_" + std::to_string(divisions) + "_iterations_" + std::to_string(iterations);


    std::string parent_map_dir = "./final_trees_data/" + folder_suffix;
    std::string cmd1 = "mkdir -p " + parent_map_dir;
    system(cmd1.c_str()); 

    std::cout << "Saving parent maps to: " << parent_map_dir << std::endl;
    planner.saveParentMapToTxt(parent_map_dir + "/tree_data"); 


    std::string child_map_dir = "./parent_data/" + folder_suffix;
    
    std::string cmd2 = "mkdir -p " + child_map_dir;
    system(cmd2.c_str());

    std::cout << "Saving children maps to: " << child_map_dir << std::endl;
    planner.saveChildrenMapToTxt(child_map_dir + "/children_map");

    planner.saveCostsToCSV("results.csv");
    std::cout << "Saving paths results to CSV ..." << std::endl;
    return 0;

};
