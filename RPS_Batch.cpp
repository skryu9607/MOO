/*
 * Regret-Based Pareto Front Sampling (C++ Implementation) - BATCH VERSION
 * Refactored with Command Line Batch Size Control
 *
 * Dependencies: OMPL, Gurobi C++, Boost
 *
 * Compilation Command:
g++ -m64 -g RPS_Batch.cpp -o RPS_Batch \
-I/opt/gurobi1300/linux64/include \
-L/opt/gurobi1300/linux64/lib \
-I/home/seung/ompl/src \
-L/home/seung/ompl/build/lib \
-I/usr/include/eigen3 \
-lgurobi_c++ -lgurobi130 -lompl -lpthread \
-fopenmp
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
#include <list>
#include <omp.h>
#include <mutex>
// Gurobi
#include "gurobi_c++.h"

// OMPL Core
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/base/OptimizationObjective.h>

// OMPL Planners
#include <ompl/geometric/planners/rrt/RRTstar.h>

namespace ob = ompl::base;
namespace og = ompl::geometric;

// ==========================================
// 1. Core Data Structures
// ==========================================

using Vector = std::vector<double>;

struct State {
    double x;
    double y;
};

// Represents a sample in our database: {weight, cost_vector}
struct SampledCost {
    int id;       // ID in the database
    Vector w;     // Weight Vector
    Vector f;     // Cost Vector f(s) = [Dist, Risk, Time]
};

struct Neighborhood {
    int id_d, id_r, id_t; // Indices of the corners in the database
    double max_regret;    // The worst case regret found in this triangle
    Vector new_w;   // The weight that causes this max regret -> pivot.
    bool is_duplicate; // Whether this new_w is a duplicate of an existing one in the database.
};

struct RegretResult {
    double max_regret;
    Vector worst_w;
};
struct BatchResult{
    Vector new_w;
    Vector new_f;
    int id_d, id_r, id_t;
};
// ==========================================
// 2. Obstacle & Environment System
// ==========================================

// Abstract Base Class for all Obstacles
class Obstacle {
public:
    virtual ~Obstacle() = default;

    // Returns true if the state is INSIDE the obstacle (collision)
    virtual bool CheckCollision(const State& s) const = 0;

    // Returns distance to the surface of the obstacle.
    // > 0: Outside (safe), < 0: Inside (collision)
    virtual double getClearance(const State& s) const = 0;
};

// Circular Obstacle Implementation
class CircularObstacle : public Obstacle {
    double cx, cy, radius;
public:
    CircularObstacle(double x, double y, double r) : cx(x), cy(y), radius(r) {}

    bool CheckCollision(const State& s) const override {
        // Simple collision check with small buffer
        return getClearance(s) <= 0.0;
    }

    double getClearance(const State& s) const override {
        double dist = std::sqrt(std::pow(s.x - cx, 2) + std::pow(s.y - cy, 2));
        return dist - radius;
    }
};

// Boundary Obstacle Implementation
class BoundaryObstacle : public Obstacle {
    double min_val, max_val;
public:
    BoundaryObstacle(double min_v, double max_v) : min_val(min_v), max_val(max_v) {}

    bool CheckCollision(const State& s) const override {
        // Collision if outside the box
        return (s.x < min_val || s.x > max_val || s.y < min_val || s.y > max_val);
    }

    double getClearance(const State& s) const override {
        // Clearance is the distance to the NEAREST wall.
        // If outside, clearance is negative.
        double dist_x_min = s.x - min_val;
        double dist_x_max = max_val - s.x;
        double dist_y_min = s.y - min_val;
        double dist_y_max = max_val - s.y;

        return std::min({dist_x_min, dist_x_max, dist_y_min, dist_y_max});
    }
};

// Environment Manager
class Environment {
    std::vector<std::shared_ptr<Obstacle>> obstacles;

public:
    void addObstacle(std::shared_ptr<Obstacle> obs) {
        obstacles.push_back(obs);
    }

    // Check if a state is valid (no collisions)
    bool isValid(const State& s) const {
        for (const auto& obs : obstacles) {
            // Using a small buffer 0.1 as per your original code
            if (obs->getClearance(s) <= 0.1) return false;
        }
        return true;
    }

    // Calculate aggregated Risk at a specific point
    double getPointRisk(const State& s) const {
        double total_risk = 0.0;
        const double max_risk_penalty = 1e6; 

        for (const auto& obs : obstacles) {
            double clearance = obs->getClearance(s);
            
            if (clearance <= 0.1) {
                return max_risk_penalty;
            } 
            
            // Risk = 1 / (dist^2)
            double val = (clearance * clearance) + 1e-3;
            total_risk += 1.0 / val;
        }
        return std::min(total_risk, max_risk_penalty);
    }

    double getEuclideanDist(const State& s1, const State& s2) const {
        return std::sqrt(std::pow(s1.x - s2.x, 2) + std::pow(s1.y - s2.y, 2));
    }

    // Returns vector: [Travel Distance, Risk, Travel Time]
    std::vector<double> calculateSegmentCost(const State& s_from, const State& s_to) const {
        std::vector<double> cost(3, 0.0);

        // 1. Cost[0]: Distance
        cost[0] = getEuclideanDist(s_from, s_to);

        // 2. Cost[1]: Risk (Numerical Integration)
        double sum_segment_risk = 0.0;
        State prev_state = s_from;
        State curr_state;
        State center_state;

        int steps = 1001; 

        for (int i = 1; i <= steps; ++i) {
            double ratio = (double)i / steps;
            curr_state.x = s_from.x + ratio * (s_to.x - s_from.x);
            curr_state.y = s_from.y + ratio * (s_to.y - s_from.y);

            center_state.x = (curr_state.x + prev_state.x) / 2.0;
            center_state.y = (curr_state.y + prev_state.y) / 2.0;

            sum_segment_risk += getPointRisk(center_state);
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
std::ofstream logFile;

// ==========================================
// 3. Environment Configuration Helper
// ==========================================

void configureEnvironment(int scenario_id) {
    global_env = std::make_shared<Environment>();

    // Always add boundary (0 to 30)
    global_env->addObstacle(std::make_shared<BoundaryObstacle>(0.0, 30.0));

    switch(scenario_id) {
        case 0: // No Obstacles
            std::cout << "Scenario 0: Empty Space" << std::endl;
            break;
        case 1: // One Circle
            std::cout << "Scenario 1: Single Circle" << std::endl;
            global_env->addObstacle(std::make_shared<CircularObstacle>(11.0, 13.0, 3.0));
            break;
        case 2: // Two Circles
            std::cout << "Scenario 2: Two Circles" << std::endl;
            global_env->addObstacle(std::make_shared<CircularObstacle>(11.0, 13.0, 3.0));
            global_env->addObstacle(std::make_shared<CircularObstacle>(11.0, 21.0, 2.0));
            break;
        default:
            std::cout << "Unknown Scenario. Defaulting to Empty." << std::endl;
            break;
    }
}

// ==========================================
// 4. OMPL Custom Objective & Validity
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
        const auto* p1 = s1->as<ob::RealVectorStateSpace::StateType>();
        const auto* p2 = s2->as<ob::RealVectorStateSpace::StateType>();

        State st1 = {p1->values[0], p1->values[1]};
        State st2 = {p2->values[0], p2->values[1]};

        std::vector<double> obj_costs = global_env->calculateSegmentCost(st1, st2);

        double scalar_cost = 0.0;
        for(size_t i = 0; i < weights.size() && i < obj_costs.size(); ++i){
            scalar_cost += weights[i] * obj_costs[i];
        }
        return ob::Cost(scalar_cost);
    }

private:
    Vector weights;
};

bool isStateValid(const ob::State *state) {
    const auto *pos = state->as<ob::RealVectorStateSpace::StateType>();
    // Delegate to Environment
    return global_env->isValid({pos->values[0], pos->values[1]});
}

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

// ==========================================
// 5. Solvers (OMPL & Gurobi)
// ==========================================

Vector solvePlanningProblem(const Vector& w, og::SimpleSetup& setup,double planning_time = 600.0) {
    setup.clear();
    auto planner(std::make_shared<og::RRTstar>(setup.getSpaceInformation()));
    planner->setRange(1.0); 
    setup.setPlanner(planner);
    
    auto obj = std::make_shared<CustomWeightedObjective>(setup.getSpaceInformation(), w);
    setup.setOptimizationObjective(obj);

    double prev_cost = std::numeric_limits<double>::infinity();
    double current_cost = std::numeric_limits<double>::infinity();
    
    // double time_slice = 30.0;           
    // double improvement_threshold = 0.0001; 
    // int max_batches = 20;             
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
    //             break; 
    //         }
    //         current_cost = new_cost;
    //     }
    //     batch_count++;
    // }
    setup.solve(planning_time * 1.5); // Solve for the entire planning time in one go, since we're doing batch processing.
    return evaluatePathCosts(setup.getSolutionPath());
}


RegretResult solveMaxRegretLP(const std::vector<SampledCost>& corners, const std::vector<double>& global_max_costs) {
    int num_objectives = 3; 
    int k = corners.size(); 
    try {
        GRBEnv env = GRBEnv(true);
        env.set("LogFile", "gurobi.log");
        env.start();
        env.set(GRB_IntParam_OutputFlag, 0);
        
        GRBModel model = GRBModel(env);
        
        std::vector<GRBVar> lambda(k);
        for(int i=0; i<k; ++i) 
            lambda[i] = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "lambda");

        std::vector<GRBVar> w(num_objectives);
        for(int j=0; j<num_objectives; ++j) 
            w[j] = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "w");

        GRBVar R = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "Regret");

        for(int j=0; j<num_objectives; ++j) {
            GRBLinExpr expr = 0;
            for(int i=0; i<k; ++i) expr += lambda[i] * corners[i].w[j];
            model.addConstr(w[j] == expr);
        }

        GRBLinExpr sum_lambdas = 0;
        for(int i=0; i<k; ++i) sum_lambdas += lambda[i];
        model.addConstr(sum_lambdas == 1.0);

        std::vector<double> u_corners(k);
        for(int i=0; i<k; ++i) {
            double dot = 0.0;
            for(int j=0; j<num_objectives; ++j) dot += corners[i].w[j] * corners[i].f[j] / global_max_costs[j];
            u_corners[i] = dot;
        }

        GRBLinExpr LB = 0;
        for(int i=0; i<k; ++i) LB += lambda[i] * u_corners[i];

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

void printVector(const std::string& label, const Vector& v) {
    std::cout << label << ": [ ";
    for (auto d : v) std::cout << d << " ";
    std::cout << "]" << std::endl;
}
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
std::vector<double> solveBatchItem(const std::vector<double> w, int thread_id,const std::vector<double>& global_max_costs,double planning_time = 600.0) {
    // Re-create the space/setup locally for this thread
    auto stateSpace = std::make_shared<ob::RealVectorStateSpace>(2);
    stateSpace->setBounds(0.0, 40.0);
    og::SimpleSetup local_setup(stateSpace);
    local_setup.setStateValidityChecker(isStateValid);

    ob::ScopedState<> start(stateSpace);
    start[0] = 1.0; start[1] = 15.0;
    ob::ScopedState<> goal(stateSpace);
    goal[0] = 21.0; goal[1] = 15.0;
    local_setup.setStartAndGoalStates(start, goal);

    auto obj = std::make_shared<CustomWeightedObjective>(
        local_setup.getSpaceInformation(), w
    );
    local_setup.setOptimizationObjective(obj);
    auto planner = std::make_shared<og::RRTstar>(local_setup.getSpaceInformation());
    planner->setRange(1.0);
    local_setup.setPlanner(planner);
    local_setup.solve(planning_time); 
    Vector f_res(3, 1.0); 
    if (local_setup.haveExactSolutionPath()) {
        // Calculate the actual vector cost of the path
        f_res = evaluatePathCosts(local_setup.getSolutionPath());
    } else {
        std::cout << "[Thread " << thread_id << "] Failed to find solution." << std::endl;
    }

    
    return f_res;
}

// ==========================================
// 6. Main Execution
// ==========================================

int main(int argc, char* argv[]) {
    // ------------------------------------------
    // Command Line Arguments Parsing
    // ------------------------------------------
    int scenario = 1;       // Default Scenario
    int target_batch_size = 4; // Default Batch Size

    // Arg 1: Scenario ID
    if (argc > 1) {
        scenario = std::stoi(argv[1]);
    }
    // Arg 2: Batch Size
    if (argc > 2) {
        target_batch_size = std::stoi(argv[2]);
        if (target_batch_size < 1) target_batch_size = 1; 
    }

    std::cout << ">>> Config: Scenario = " << scenario 
              << ", Batch Size = " << target_batch_size << std::endl;

    // 2. Configure Environment
    configureEnvironment(scenario);

    // 3. Create Dynamic Log Filename (Includes both Scenario and BatchSize)
    std::string filename = "RPS_log_batch_scenario_" + std::to_string(scenario) + 
                           "_size_" + std::to_string(target_batch_size) + ".txt";
    logFile.open(filename);
    
    if (!logFile.is_open()) {
        std::cerr << "Error: Could not open log file " << filename << std::endl;
        return 1;
    }
    std::cout << "Saving data to: " << filename << std::endl;
    logFile << "Iteration, w1,w2,w3, f1,f2,f3, MaxRegret, is_duplicate, wd,wr,wt\n";

    // ------------------------------------------
    // RPS Initialization
    // ------------------------------------------
    std::vector<SampledCost> database;
    int num_obj = 3; 
    int duplicate_count = 0;

    std::vector<Vector> corner_weights = {
        {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}
    };
    std::vector<Vector> corner_case;
    if (scenario == 1){
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
                << f[0] << "," << f[1] << "," << f[2] << ", " << 0.0 << "\n";
    }

    std::list<Neighborhood> neighborhoods;

    // Initial Neighborhood
    Neighborhood initial_neighborhood;
    initial_neighborhood.id_d = 0; initial_neighborhood.id_r = 1; initial_neighborhood.id_t = 2;
    std::vector<SampledCost> initial_corners = { database[0], database[1], database[2] };
    RegretResult initial_regret = solveMaxRegretLP(initial_corners, global_max_costs);
    initial_neighborhood.max_regret = initial_regret.max_regret;
    initial_neighborhood.new_w = initial_regret.worst_w;
    initial_neighborhood.is_duplicate = false; // First one cannot be duplicate.
    neighborhoods.push_back(initial_neighborhood);

    // ------------------------------------------
    // Batch Main Loop
    // ------------------------------------------
    //int Buget_K = (int) 32/target_batch_size; // Following the original paper. 
    // MAXIMUM iteration is Buget_K + num_obj.
    int Buget_K = 30;

    for(int k=0; k<Buget_K; ++k) {
        std::cout << "\n--- Iteration " << k << " ---" << std::endl;

        // 1. Sort Neighborhoods (Highest Regret First)
        neighborhoods.sort([](const Neighborhood& a, const Neighborhood& b) {
            return a.max_regret > b.max_regret;
        });

        // 2. Select Batch
        std::vector<Neighborhood> batch_to_process;
        int count = 0;
        
        while(!neighborhoods.empty() && count < target_batch_size) {
            // Stop if regret is too low
            if (neighborhoods.front().max_regret < 0.0005) break;
            batch_to_process.push_back(neighborhoods.front());
            neighborhoods.pop_front();
            count++;
        }
        
        if (batch_to_process.empty()) {
            std::cout << "Converged! No high regret regions left." << std::endl;
            break;
        }

        std::cout << "Processing Batch Size: " << batch_to_process.size() << std::endl;

        // Parallel execution

        struct PlanResult {
            Vector f;
            Vector w;
        };
        std::vector<PlanResult> batch_results(batch_to_process.size());

        #pragma omp parallel for schedule(dynamic)
        for(int i=0; i<batch_to_process.size(); ++i) {
            int tid = omp_get_thread_num();
            
            Vector f_res = solveBatchItem(batch_to_process[i].new_w, tid, global_max_costs);
            
            // Store locally
            batch_results[i] = {f_res, batch_to_process[i].new_w};

            #pragma omp critical
            {
                std::cout << "   [Thread " << tid << "] Finished sample " << i+1 << "/" << batch_to_process.size() << std::endl;
            }
        }
        double threshold_duplicate = 0.001;
        for (int i=0; i<batch_to_process.size(); ++i) {
            auto* task = &batch_to_process[i];
            const auto& res  = batch_results[i];
            
            int new_id = (int)database.size();
            
            int d = task->id_d;
            int r = task->id_r;
            int t = task->id_t;
            std::vector<double> w_d = database[d].w;
            std::vector<double> w_r = database[r].w;
            std::vector<double> w_t = database[t].w;
            // Check the duplicate 
            bool is_duplicate = false;
            for (const auto& entry : database) {
                double dist_weight = std::sqrt(std::pow(entry.w[0] - res.w[0], 2)
                 + std::pow(entry.w[1] - res.w[1], 2) +
                  std::pow(entry.w[2] - res.w[2], 2));
                if (dist_weight < threshold_duplicate){
                    is_duplicate = true;
                    break;
                }
            }
            // Add the random noise to the weight. (Continue)
            if(is_duplicate){
                task->is_duplicate = true;
                continue; // Skip the subdivision if it's a duplicate.
            }   
            // Log
            database.push_back({new_id, res.w, res.f});
            logFile << k << "," << res.w[0] << "," << res.w[1] << "," << res.w[2] << ", "
                    << res.f[0] << "," << res.f[1] << "," << res.f[2] << ", " 
                    << task->max_regret << "," << is_duplicate << "," <<w_d[0]<< "," <<w_d[1]<< "," 
                    <<w_d[2] <<"," <<w_r[0]<< "," <<w_r[1]<< "," <<w_r[2] 
                    <<"," <<w_t[0]<< "," <<w_t[1]<< "," <<w_t[2] << "\n";
            // 

            // Define the 3 smaller triangles 
            // Triangle 1: {d, r, new}
            // Triangle 2: {d, new, t}
            // Triangle 3: {new, r, t}
            int sets[3][3] = {
                {d, r, new_id},
                {d, new_id, t},
                {new_id, r, t}
            };
            // Subdivision
            for (int j = 0; j < 3 ; ++j) {
                Neighborhood n_child;
                n_child.id_d = sets[j][0];
                n_child.id_r = sets[j][1];
                n_child.id_t = sets[j][2];
                n_child.is_duplicate = false; // Initially assume it's not a duplicate. Will check later.
                std::vector<SampledCost> corners = {
                    database[n_child.id_d],
                    database[n_child.id_r],
                    database[n_child.id_t]
                };

                // Solve LP for new sub-neighborhood
                RegretResult lp_res = solveMaxRegretLP(corners, global_max_costs);
                
                n_child.new_w = lp_res.worst_w;
                n_child.max_regret = lp_res.max_regret;

                if (n_child.max_regret > 0.0005) {
                    neighborhoods.push_back(n_child);
                }
                else {
                    std::cout << " max_Regret is zero!! Do not need to be subdivided" << std::endl;
                }
            }
        }
        
        // Log flush
        logFile.flush();
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
    }


    // for(int k=0; k<Buget_K; ++k) {
    //     std::cout << "\n--- Iteration " << k << " ---" << std::endl;
    //     double max_global_regret = 0.0;

    //     // Sort neighborhoods by max_regret descending
    //     neighborhoods.sort([](const Neighborhood& a, const Neighborhood& b) {
    //         return a.max_regret > b.max_regret;
    //     });
    //     // Neighborhood : id_d, id_r, id_t, max_regret, new_w.
    //     std::vector<Neighborhood> batch_to_process;
    //     int count = 0;
    //     auto it = neighborhoods.begin();
        
    //     while(!neighborhoods.empty() && count < target_batch_size) {
    //         batch_to_process.push_back(neighborhoods.front());
    //         neighborhoods.pop_front();
    //         count++;
    //     }
        
    //     if (batch_to_process.empty() || batch_to_process[0].max_regret < 0.0005) {
    //         std::cout << "Converged! Max Regret: " << batch_to_process[0].max_regret << std::endl;
    //         break;
    //     }
    //     // plan for the candidate weight
    //     // 3. Parallel Execution (The "Batch" Advantage)
    //     // BatchResult : worst_w, max_regret.
    //     std::vector<std::vector<double>> batch_f(batch_to_process.size(), std::vector<double>(num_obj));
    //     // databse added.
    //     for(int i=0; i<batch_to_process.size(); ++i) {
    //         int new_id = (int)database.size() + i;
    //         int tid = omp_get_thread_num();
    //         batch_f[i] = solveBatchItem(batch_to_process[i].new_w, tid, global_max_costs);
    //         database.push_back({new_id, batch_to_process[i].new_w, batch_f[i]});
    //         {
    //             std::cout << "   [Thread " << tid << "] Finished sample " << i+1 << "/" << batch_to_process.size() << std::endl;
    //         }
    //     }
    //     //SampleCost : int id, Vector w, Vector f.
    //     //Split the K several Neighborhoods into 3 * K samples.
    //     // 4. Sequential Update
    //     for (int i=0; i<batch_to_process.size(); ++i){
    //         const auto& res = batch_to_process[i];
    //         int id_d = res.id_d;
    //         int id_r = res.id_r;
    //         int id_t = res.id_t;
    //         std::cout << "Processing neighborhood with pivot w = ["
    //                   << res.new_w[0] << ", " << res.new_w[1] << ", " << res.new_w[2] 
    //                   << "], max regret = " << res.max_regret << std::endl;
    //         std::vector<std::vector<double>> subCorners_index = {
    //             {id_d, id_r, database.size() - batch_to_process.size() + i},
    //             {id_d, database.size() - batch_to_process.size() + i, id_t},
    //             {database.size() - batch_to_process.size() + i, id_r, id_t}
    //         };
    //         std::vector<std::vector<double>> subCorners_costs = {
    //             {database[id_d].f, database[id_r].f, database[database.size() - batch_to_process.size() + i].f},
    //             {database[id_d].f, database[database.size() - batch_to_process.size() + i].f, database[id_t].f},
    //             {database[database.size() - batch_to_process.size() + i].f, database[id_r].f, database[id_t].f}
    //         };
    //         for (int j = 0; j < 3 ; ++j) {
    //             Neighborhood n_child;
    //             n_child.id_d = subCorners_index[j][0];
    //             n_child.id_r = subCorners_index[j][1];
    //             n_child.id_t = subCorners_index[j][2];
    //             std::vector<SampledCost> corners = {
    //                 database[n_child.id_d],
    //                 database[n_child.id_r],
    //                 database[n_child.id_t]
    //             };
    //             RegretResult lp_res = solveMaxRegretLP(
    //                 corners, global_max_costs
    //             );
    //             n_child.new_w = lp_res.worst_w;
    //             n_child.max_regret = lp_res.max_regret;

    //             neighborhoods.push_back(n_child);
    //         }
    //     }
            
    // }

    // logFile.close();
    // std::cout << "RPS Batch Finished. Total Samples: " << database.size() << std::endl;
    // std::cout << "Number of duplicates: " << duplicate_count << std::endl;
    // return 0;
}
