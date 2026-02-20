/*
 * Regret-Based Pareto Front Sampling (C++ Implementation)
 * Integrated with Custom Cost Function (Distance, Risk, Time)
 *
 * Dependencies: OMPL, Gurobi C++, Boost
 *
 * Compilation Command:
 g++ -m64 -g RPS.cpp -o RPS \
 -I/opt/gurobi1300/linux64/include \
 -L/opt/gurobi1300/linux64/lib \
 -I/home/seung/ompl/src \
 -L/home/seung/ompl/build/lib \
 -I/usr/include/eigen3 \
 -lgurobi_c++ -lgurobi130 -lompl -lpthread
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <limits>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>

// Gurobi
#include "gurobi_c++.h"

// OMPL Core
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/base/OptimizationObjective.h>

// OMPL Planners
#include <ompl/geometric/planners/informedtrees/BITstar.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>

namespace ob = ompl::base;
namespace og = ompl::geometric;

// ==========================================
// 1. Data Structures & Cost Helpers
// ==========================================

using Vector = std::vector<double>;

// User-defined State struct for cost calculation
struct State {
    double x;
    double y;
};

// Represents a sample in our database: {weight, cost_vector}
struct SampledCost {
    int id; // ID in the database
    Vector w; // Weight Vector
    Vector f; // Objective Vector f(s) = [Dist, Risk, Time]
};
struct Neighborhood{
    // distance, risk, time
    int id_d,id_r, id_t; // Indices of the corners in the database
    double max_regret; // the worst case regret found in this traiangle
    Vector candidate_w; // The weight that causes this max regret -> pivot. 
    bool is_duplicate; // Whether this candidate_w is a duplicate of an existing one in the database.
};
struct Simplex{
    std::vector<int> corner_ids; // Indices of the corners in the database
};
// This is the result of the LP solver.
struct RegretResult {
    double max_regret;
    Vector worst_w;
};
class Obstacle{
    public: 
    virtual ~Obstacle() = default;
    virtual bool CheckCollision(const State& s) const = 0;
    virtual double getClearance(const State& s) const = 0;
};
class CircularObstacle : public Obstacle {
    public:
    CircularObstacle(double cx, double cy, double r) : center_x(cx), center_y(cy), radius(r) {}
    double getClearance(const State& s) const override {

    double dist = std::sqrt(std::pow(s.x - center_x, 2) + std::pow(s.y - center_y, 2));
        return dist - radius;
    }
    bool CheckCollision(const State& s) const override {
        return getClearance(s) <= 0.1;
    }

    private:
    double center_x;
    double center_y;
    double radius;
};
class BoundaryObstacle :public Obstacle {
    double min_val, max_val;
    public :
    BoundaryObstacle(double min_v, double max_v) : min_val(min_v), max_val(max_v) {}
    double getClearance(const State& s) const override {
        double dist_x = std::min(s.x - min_val, max_val - s.x);
        double dist_y = std::min(s.y - min_val, max_val - s.y);
        return std::min(dist_x, dist_y);
    }
    bool CheckCollision(const State& s) const {
            // Collision if outside the box
        return (s.x < min_val || s.x > max_val || s.y < min_val || s.y > max_val);
    }
};

class Environment {
    std::vector<std::shared_ptr<Obstacle>> obstacles;
    int integration_steps;
    public:
    void addObstacle(const std::shared_ptr<Obstacle>& obs) {
        obstacles.push_back(obs);
    }
    bool checkCollision(const State& s) const {
        for (const auto& obs : obstacles) {
            if (obs->CheckCollision(s)) {
                return true;
            }
        }
        return false;
    }
    double getRiskAtState(const State& s) const {
        double total_risk = 0.0;
        for (const auto& obs : obstacles) {
            double clearance = obs->getClearance(s);
            if (clearance <= 0.1) {
                total_risk += 1e6; // High risk for collision
            } else {
                total_risk += 1.0 / (clearance * clearance + 1e-3);
            }
        }
        return std::min(total_risk, 1e6);
    }
    double getEuclideanDist(const State& s1, const State& s2) const {
        return std::sqrt(std::pow(s1.x - s2.x, 2) + std::pow(s1.y - s2.y, 2));  
    }
    std::vector<double> calculateSegmentCost(const State& s_from, const State& s_to) const {
        std::vector<double> cost(3, 0.0);

        // 1. Cost[0]: Distance
        cost[0] = getEuclideanDist(s_from, s_to);

        // 2. Cost[1]: Risk 
        double sum_segment_risk = 0.0;
        State prev_state = s_from;
        State curr_state;
        State center_state;

        int steps = 2501; 

        for (int i = 1; i <= steps; ++i) {
            double ratio = (double)i / steps;
            curr_state.x = s_from.x + ratio * (s_to.x - s_from.x);
            curr_state.y = s_from.y + ratio * (s_to.y - s_from.y);

            center_state.x = (curr_state.x + prev_state.x) / 2.0;
            center_state.y = (curr_state.y + prev_state.y) / 2.0;

            sum_segment_risk += getRiskAtState(center_state);
            prev_state = curr_state;
        }

        cost[1] = 1.0 * sum_segment_risk * cost[0] / steps;

        // 3. Cost[2]: Travel Time
        // Logic: Lower Y is fast (Highway), Higher Y is slow
        double Time = 0.0;
        prev_state = s_from;

        for (int i = 1; i <= steps; ++i) {
            double ratio = (double)i / steps;
            curr_state.x = s_from.x + ratio * (s_to.x - s_from.x);
            curr_state.y = s_from.y + ratio * (s_to.y - s_from.y);

            double speed = (curr_state.y < 13.0) ? 100.0 : 2.0;
            double seg_dist = getEuclideanDist(curr_state, prev_state);
            Time += seg_dist / speed;
            
            prev_state = curr_state;
        }
        cost[2] = Time;

        return cost;
    }
};
std::shared_ptr<Environment> global_env;

void printVector(const std::string& label, const Vector& v) {
    std::cout << label << ": [ ";
    for (auto d : v) std::cout << d << " ";
    std::cout << "]" << std::endl;
}
std::ofstream logFile;
// Function to save the database to a CSV file
void saveDatabaseToCSV(const std::string& filename, const std::vector<SampledCost>& database) {
    std::ofstream outFile(filename);

    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    outFile << "ID,W_Distance,W_Risk,W_Time,Cost_Distance,Cost_Risk,Cost_Time\n";

    // Use 'const auto&' to reference the original data directly (zero copying!)
    for (const auto& s : database) {
        outFile << s.id << ",";
        
        // Write the weights
        for (size_t i = 0; i < s.w.size(); ++i) {
            outFile << s.w[i];
            if (i < s.w.size() - 1) outFile << ","; 
        }
        
        outFile << ","; 
        
        // Write the costs
        for (size_t i = 0; i < s.f.size(); ++i) {
            outFile << s.f[i];
            if (i < s.f.size() - 1) outFile << ",";
        }
        outFile << "\n"; 
    }

    outFile.close();
    std::cout << "Database successfully saved to: " << filename << std::endl;
}
// ==========================================
// 2. OMPL Planner Setup
// ==========================================

class CustomWeightedObjective : public ob::OptimizationObjective {
public:
    CustomWeightedObjective(const ob::SpaceInformationPtr &si, const Vector& weights)
        : ob::OptimizationObjective(si), weights(weights) {
        description_ = "Weighted Distance/Risk/Time";
    }

    ob::Cost stateCost(const ob::State *s) const override {
        return ob::Cost(0.0);
    }

    ob::Cost motionCost(const ob::State *s1, const ob::State *s2) const override {
        // ""Take this generic state s1, and let mee use it "AS" a RealVectorStateSpace         
        // ::StateType
        const auto* p1 = s1->as<ob::RealVectorStateSpace::StateType>();
        const auto* p2 = s2->as<ob::RealVectorStateSpace::StateType>();

        State st1 = {p1->values[0], p1->values[1]};
        State st2 = {p2->values[0], p2->values[1]};

        // 2. Calculate the multi objectives vector
        std::vector<double> obj_vecs = global_env->calculateSegmentCost(st1, st2);

        // 3. Weight them: w0*Dist + w1*Risk + w2*Time
        double cost_scalar = 0.0;
        for(size_t i = 0; i< weights.size() && i < obj_vecs.size(); ++i){
            cost_scalar += weights[i] * std::pow(obj_vecs[i], 1);
        }
        return ob::Cost(std::pow(cost_scalar, 1.0/1.0));
    }

private:
    Vector weights;
};
bool isStateValid(const ob::State *state) {
    const auto *pos = state->as<ob::RealVectorStateSpace::StateType>();
    bool ans = global_env->checkCollision({pos->values[0], pos->values[1]});
    return !ans;
}
// Evaluate the full path to get the [Dist, Risk, Time] vector
Vector evaluatePathCosts(og::PathGeometric& path) {
    Vector total_costs(3, 0.0);
    const auto& states = path.getStates();

    for (size_t i = 0; i < states.size() - 1; ++i) {
        const auto* p1 = states[i]->as<ob::RealVectorStateSpace::StateType>();
        const auto* p2 = states[i+1]->as<ob::RealVectorStateSpace::StateType>();
        
        State st1 = {p1->values[0], p1->values[1]};
        State st2 = {p2->values[0], p2->values[1]};

        std::vector<double> segment_costs = global_env->calculateSegmentCost(st1, st2);

        for(int k=0; k<3; ++k) total_costs[k] += segment_costs[k];
    }
    return total_costs;
}



// Using OMPL Solver. Inputs : weight, and setup. Outputs: Cost Vector. 
Vector solvePlanningProblem(const Vector& w, og::SimpleSetup& setup) {
    setup.clear();
    auto planner(std::make_shared<og::RRTstar>(setup.getSpaceInformation()));
    planner->setRange(1.0); 
    setup.setPlanner(planner);
    auto obj = std::make_shared<CustomWeightedObjective>(setup.getSpaceInformation(), w);
    setup.setOptimizationObjective(obj);
    double prev_cost = std::numeric_limits<double>::infinity();
    double current_cost = std::numeric_limits<double>::infinity();
    
    // double time_slice = 30.0;           
    // double improvement_threshold = 0.0001; // 0.1% imporovement
    // int max_batches = 10;             
    // int batch_count = 0;

    // setup.solve(time_slice);
    // if (setup.haveExactSolutionPath()) {
    //     current_cost = setup.getSolutionPath().cost(obj).value();
    // }
    // batch_count++;

    // while (batch_count < max_batches) {
    //     if (current_cost == std::numeric_limits<double>::infinity()) {
    //         setup.solve(time_slice);
    //         if (setup.haveExactSolutionPath()) {
    //             current_cost = setup.getSolutionPath().cost(obj).value();
    //         }
    //     } 
    //     else {
    //         prev_cost = current_cost;
    //         setup.solve(time_slice);

    //         double new_cost = setup.getSolutionPath().cost(obj).value();

    //         double improvement = (prev_cost - new_cost) / prev_cost;

    //         if (improvement < improvement_threshold) {
    //             std::cout << "Converged at batch " << batch_count << " (Imp: " << improvement << ")" << std::endl;
    //             break; 
    //         }
    //         current_cost = new_cost;
    //     }
    //     batch_count++;
    // }
    setup.solve(900.0);
    return evaluatePathCosts(setup.getSolutionPath());
}

// ==========================================
// 3. Gurobi LP Solver (Equation 12.) -> To find the max regret and its weight.
// ==========================================
// Solve the Max Regret LP given the current database of SampledCosts.
// s.t. 
// w is inside the convex hull of the corner weights
// R <= Known solution cost at w 
RegretResult solveMaxRegretLP(const std::vector<SampledCost>& corners, const std::vector<double>& global_max_costs) {
    int num_objectives = 3; // Must be 3 in out case. 
    int k = corners.size(); // number of corners in the neighborhood
   try {
        GRBEnv env = GRBEnv(true);
        env.set("LogFile", "gurobi.log");
        env.start();
        env.set(GRB_IntParam_OutputFlag, 0);
        // Create am empty and initialized model. 
        GRBModel model = GRBModel(env);
        // Definition of variables.
        std::vector<GRBVar> lambda(k);
        for(int i=0; i<k; ++i) 
            lambda[i] = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "lambda");

        std::vector<GRBVar> w(num_objectives);
        for(int j=0; j<num_objectives; ++j) 
            w[j] = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "w");

        GRBVar R = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "Regret");

        // w definition : a linear combination of corner weights. 
        for(int j=0; j<num_objectives; ++j) {
            GRBLinExpr expr = 0;
            for(int i=0; i<k; ++i) expr += lambda[i] * corners[i].w[j];
            model.addConstr(w[j] == expr);
        }

        // Sum lambdas = 1
        GRBLinExpr sum_lambdas = 0;
        for(int i=0; i<k; ++i) sum_lambdas += lambda[i];
        model.addConstr(sum_lambdas == 1.0);

        // Lower Bound Calculation : P(w)
        std::vector<double> u_corners(k);
        for(int i=0; i<k; ++i) {
            double dot = 0.0;
            for(int j=0; j<num_objectives; ++j) dot += corners[i].w[j] * corners[i].f[j] / global_max_costs[j];
            u_corners[i] = dot;
        }

        GRBLinExpr LB = 0;
        for(int i=0; i<k; ++i) LB += lambda[i] * u_corners[i];

        // Regret Constraints: R <= w*f(s^j)- P(w) for each corner j
        for(int i=0; i<k; ++i) {
            GRBLinExpr w_dot_fs = 0.0;
            for(int j=0; j<num_objectives; ++j) w_dot_fs += w[j] * corners[i].f[j] / global_max_costs[j];
            model.addConstr(R <= w_dot_fs - LB);
        }

        model.setObjective(GRBLinExpr(R), GRB_MAXIMIZE);
        model.optimize();

        Vector res_w;
        for(int j=0; j<num_objectives; ++j) res_w.push_back(w[j].get(GRB_DoubleAttr_X));
        return {R.get(GRB_DoubleAttr_X), res_w};

    } catch(GRBException e) {
        std::cerr << "Gurobi Error: " << e.getMessage() << std::endl;
        return {-1.0, {}};
    }
}
// ==========================================
// Scenarios.
void configureEnvironment(int scenario_id) {
    // 1. Reset the global environment
    global_env = std::make_shared<Environment>();

    // 2. ALWAYS Add Boundary (Required for risk calculation 0-30)
    global_env->addObstacle(std::make_shared<BoundaryObstacle>(0.0, 40.0));

    // 3. Add Specific Obstacles based on ID
    switch(scenario_id) {
        case 0: 
            // [Case 0: Non-obstacle case]
            // Only the boundary exists. The space is empty.
            std::cout << "Loading Scenario: Empty Space (Boundary Only)" << std::endl;
            break;

        case 1:
            // [Case 1: Single Obstacle] 
            // Original setup: Boundary + 1 Circle
            std::cout << "Loading Scenario: Single Circle" << std::endl;
            global_env->addObstacle(std::make_shared<CircularObstacle>(11.0, 13.0, 3.0));
            break;

        case 2:
            // [Case 2: Two Obstacles]
            // Boundary + 2 Circles (Example: One low, one high)
            std::cout << "Loading Scenario: Two Circles" << std::endl;
            global_env->addObstacle(std::make_shared<CircularObstacle>(11.0, 13.0, 3.0));
            global_env->addObstacle(std::make_shared<CircularObstacle>(11.0, 21.0, 2.0));
            break;
        default:
            std::cout << "Unknown Scenario. Defaulting to Empty." << std::endl;
            break;
    }
}


// ==========================================
// 5. Main Loop

int main(int argc, char* argv[]) {
    // 1. Determine Scenario ID (Default to 1 if not provided)
    int scenario = 1; 
    if (argc > 1) {
        scenario = std::stoi(argv[1]);
    }

    // 2. Configure Environment based on Scenario
    configureEnvironment(scenario);

    // 3. Create Dynamic Log Filename
    std::string filename = "RPS_log_scenario_" + std::to_string(scenario) + ".txt";
    logFile.open(filename);
    
    // Check if file opened successfully
    if (!logFile.is_open()) {
        std::cerr << "Error: Could not open log file " << filename << std::endl;
        return 1;
    }

    std::cout << "Saving data to: " << filename << std::endl;
    logFile << "Iteration, w1,w2,w3, f1,f2,f3, MaxRegret, is_duplicate, wd,wr,wt\n";
    // ------------------------------------------
    // A. Environment Setup
    // ------------------------------------------
    global_env = std::make_shared<Environment>();
    
    // 1. Add Circular Obstacle
    global_env->addObstacle(std::make_shared<CircularObstacle>(11.0, 13.0, 3.0));
    
    // 2. Add Boundary as Obstacle (0 to 30)
    // This makes the risk increase as you approach 0.0 or 30.0
    double boundary_min = 0.0;
    double boundary_max = 40.0;
    global_env->addObstacle(std::make_shared<BoundaryObstacle>(boundary_min,boundary_max));

    auto stateSpace = std::make_shared<ob::RealVectorStateSpace>(2);
    stateSpace->setBounds(boundary_min, boundary_max); 
    og::SimpleSetup setup(stateSpace);
    ob::ScopedState<> start(stateSpace);
    setup.setStateValidityChecker(isStateValid);
    start[0] = 1.0; start[1] = 15.0;
    ob::ScopedState<> goal(stateSpace);
    goal[0] = 21.0; goal[1] = 15.0;
    int duplicate_count = 0;
    setup.setStartAndGoalStates(start, goal);

    // 2. Initialize Algorithm
    std::vector<SampledCost> database;
    int num_obj = 3; // Distance, Risk, Time

    // Standard Basis Weights (Corners of the 3-obj simplex)
    std::vector<Vector> corner_weights = {
        {1.0, 0.0, 0.0}, // Pure Distance
        {0.0, 1.0, 0.0}, // Pure Risk
        {0.0, 0.0, 1.0}  // Pure Time
    };
    std::vector<Vector> corner_case;
    if (scenario == 1){
    // Pre-calculated corners (Optionally use solvePlanningProblem here)
    corner_case = {
        {20.1494,101.356,10.0747}, 
        {31.6239,1.7663,15.8119},
        {25.4269,143.952,2.21921}
    };
    }
    else if (scenario == 2){
    corner_case = {
        {20.1936,49.5278,10.0968},
        {43.9008,2.97585,21.9504},
        {25.3565,101.833,2.2156}
    };
    }
    else {
    corner_case = {
            {20.0154,0.959798,10.0077},
            {20.3631,0.959532,10.1816},
            {24.5904,2.81677,2.21509}
        };
    }
    std::cout << "--- Initializing Corners ---" << std::endl;
    std::vector<double> global_max_costs(3, 1.0); 
    for (int i = 0; i < 3 ; ++i) {
        Vector f = corner_case[i];
        database.push_back({i, corner_weights[i], f});
        for(int k=0; k<3; ++k) {
            if(f[k] > global_max_costs[k]) global_max_costs[k] = f[k];
        }
        logFile << i-num_obj << "," << corner_weights[i][0] << "," << corner_weights[i][1] << ", " << corner_weights[i][2] << ", "
                << f[0] << "," << f[1] << ", " << f[2] << ", " << 0.0 << "\n";
    }
    std::list<Neighborhood> neighborhoods;

    // Create the FIRST neighborhood.

    Neighborhood initial_neighborhood;
    initial_neighborhood.id_d = 0;
    initial_neighborhood.id_r = 1;
    initial_neighborhood.id_t = 2;
    std::vector<SampledCost> initial_corners = {
        database[initial_neighborhood.id_d],
        database[initial_neighborhood.id_r],
        database[initial_neighborhood.id_t]
    };
    RegretResult initial_regret = solveMaxRegretLP(initial_corners,global_max_costs);
    initial_neighborhood.max_regret = initial_regret.max_regret;
    initial_neighborhood.candidate_w = initial_regret.worst_w;
    initial_neighborhood.is_duplicate = false; // First one cannot be duplicate.
    neighborhoods.push_back(initial_neighborhood);
    std::cout << "Initial Max Regret: " << initial_neighborhood.max_regret << std::endl;

    //----- LOOP -----
    int Buget_K = 32; 
    double threshold_duplicate = 0.001; // If the new cost is within 1% of an existing cost, consider it a duplicate.
    for(int k=0; k<Buget_K; ++k) {
        std::cout << "\n--- Iteration " << k << " ---" << std::endl;
        double max_global_regret = -1.0;
        auto best_it = neighborhoods.begin();
        for(auto it = neighborhoods.begin(); it != neighborhoods.end(); ++it) {
            if(it->max_regret > max_global_regret && !it->is_duplicate) {
                max_global_regret = it->max_regret;
                best_it = it;
            }
        }
        std::cout << "Selected Neighborhood with Max Regret: " << max_global_regret << std::endl;
        std::cout << "Iteration " << k << ": Solving for weights " << max_global_regret << " Triangle Corners IDs: "
                  << best_it->id_d << ", " << best_it->id_r << ", " << best_it->id_t << std::endl;
        
        if(max_global_regret < 0.0005) {
            std::cout << "Converged." << std::endl;
            logFile << k << "," << best_it->candidate_w[0] << "," << best_it->candidate_w[1] << "," << best_it->candidate_w[2] << max_global_regret << ", " << 
                best_it->is_duplicate << "," << database[best_it->id_d].w[0] << "," << database[best_it->id_d].w[1] << "," << database[best_it->id_d].w[2] << "," <<
                database[best_it->id_r].w[0] << "," << database[best_it->id_r].w[1] << "," << database[best_it->id_r].w[2] << "," <<
                database[best_it->id_t].w[0] << "," << database[best_it->id_t].w[1] << "," << database[best_it->id_t].w[2] << "\n";
            logFile.flush();
            break;
        }
        // plan for the candidate weight
        Vector new_w = best_it->candidate_w; // Pivot weight
        Vector new_f = solvePlanningProblem(new_w, setup);
        int new_id = database.size();


        int d = best_it->id_d;
        int r = best_it->id_r;
        int t = best_it->id_t;
        std::vector<double> w_d = database[best_it->id_d].w;
        std::vector<double> w_r = database[best_it->id_r].w;
        std::vector<double> w_t = database[best_it->id_t].w;

        // 1. CHECK FOR DUPLICATES
        bool is_duplicate = false;
        // Checking the corners of the triangle.
        for(size_t i=0; i<database.size(); ++i) {
            double dist = 0.0;
            // Calculate Euclidean distance in weight Space
            for(int j=0; j<3; ++j) dist += std::pow(new_w[j] - database[i].w[j], 2);
            
            // Tolerance: If cost vector is within 0.01 of an existing one, it's a duplicate.
            if(std::sqrt(dist) < threshold_duplicate) { 
                is_duplicate = true;
                break;
            }
        }
        // 2. HANDLE THE RESULT
        if (is_duplicate) {
            best_it->is_duplicate = true; // Flag the neighborhood hard to solve LP by gurobi.
            continue;
        }

        printVector("New weight", new_w);
        printVector("New cost", new_f);
        database.push_back({new_id, new_w, new_f});
        logFile << k << "," << new_w[0] << "," << new_w[1] << "," << new_w[2] << ", "
                    << new_f[0] << "," << new_f[1] << "," << new_f[2] << ", "
                    << max_global_regret << "," << is_duplicate << "," <<w_d[0]<< "," <<w_d[1]<< "," 
                    <<w_d[2] <<"," <<w_r[0]<< "," <<w_r[1]<< "," <<w_r[2] 
                    <<"," <<w_t[0]<< "," <<w_t[1]<< "," <<w_t[2] << "\n";
        logFile.flush();

        // Remove the used neighborhood
        neighborhoods.erase(best_it);
    
        int sets[3][3] = {
            {d, r, new_id},
            {d, new_id, t},
            {new_id, r, t}
        };
        
        // Create 3 new neighborhoods
        for(int i=0; i< num_obj; ++i) {
            Neighborhood n;
            n.id_d = sets[i][0];
            n.id_r = sets[i][1];
            n.id_t = sets[i][2];
            std::vector<SampledCost> corners = {
                database[n.id_d],
                database[n.id_r],
                database[n.id_t]
            };
            RegretResult regret = solveMaxRegretLP(corners,global_max_costs);
            n.max_regret = regret.max_regret;
            n.candidate_w = regret.worst_w;
            n.is_duplicate = false; // Initially assume it's not a duplicate. It will be checked in the next iteration.
            neighborhoods.push_back(n);
            std::cout << "New Neighborhood Corners IDs: "
                      << n.id_d << ", " << n.id_r << ", " << n.id_t 
                      << " with Max Regret: " << n.max_regret << std::endl;
        }
    }
    std::cout << "Final Database:" << std::endl;
    for(auto s : database) {
        std::cout << "ID: " << s.id << " W: [ ";
        for(auto w : s.w) std::cout << w << " ";
        std::cout << "] F: [ ";
        for(auto f : s.f) std::cout << f << " ";
        std::cout << "]" << std::endl;
    }
    logFile.close();
    std::cout << "RPS Finished. Total Samples: " << database.size() << std::endl;
    std::cout << "RPS Completed." << std::endl;

    std::string db_filename = filename;
    
    // Find where the ".csv" or ".txt" is
    size_t dotPos = db_filename.find_last_of('.'); 
    
    if (dotPos != std::string::npos) {
        // If it found a dot, insert "_database" right before it
        // "my_log_run.csv" -> "my_log_run_database.csv"
        db_filename.insert(dotPos, "_database"); 
    } else {
        // If there is no extension, just append it
        // "my_log_run" -> "my_log_run_database.csv"
        db_filename += "_database.csv"; 
    }

    // 2. Call the save function using the newly generated name!
    saveDatabaseToCSV(db_filename, database);

    return 0;
}

    // Metric time!
    // 1. Pareto Optimality : using truth ground one.
