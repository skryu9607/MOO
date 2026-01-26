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
#include <list>

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
    Vector candidate_w;   // The weight that causes this max regret -> pivot.
};

struct RegretResult {
    double max_regret;
    Vector worst_w;
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
// Treats the area OUTSIDE the bounds as an obstacle
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

        int steps = 10001; 

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

Vector solvePlanningProblem(const Vector& w, og::SimpleSetup& setup) {
    setup.clear();
    auto planner(std::make_shared<og::RRTstar>(setup.getSpaceInformation()));
    planner->setRange(1.5); 
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
    setup.solve(300.0);
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
    logFile << "Iteration, w1,w2,w3, f1,f2,f3, MaxRegret\n";

    // ------------------------------------------
    // OMPL Setup
    // ------------------------------------------
    auto stateSpace = std::make_shared<ob::RealVectorStateSpace>(2);
    stateSpace->setBounds(0.0, 40.0); 
    og::SimpleSetup setup(stateSpace);
    setup.setStateValidityChecker(isStateValid);

    ob::ScopedState<> start(stateSpace);
    start[0] = 1.0; start[1] = 15.0;
    ob::ScopedState<> goal(stateSpace);
    goal[0] = 21.0; goal[1] = 15.0;
    
    setup.setStartAndGoalStates(start, goal);

    // ------------------------------------------
    // RPS Initialization
    // ------------------------------------------
    std::vector<SampledCost> database;
    int num_obj = 3; 
    int duplicate_count = 0;

    std::vector<Vector> corner_weights = {
        {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}
    };
    
    // Pre-calculated corners (Optionally use solvePlanningProblem here)
    std::vector<Vector> corner_case = {
        {20.1717,86.1769,10.0858}, {53.6036,0.330123,26.8018},{25.0371,265.625,2.16954}
    };

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
    initial_neighborhood.candidate_w = initial_regret.worst_w;
    neighborhoods.push_back(initial_neighborhood);

    // ------------------------------------------
    // Batch Main Loop
    // ------------------------------------------
    int MAX_ITER = 20; 

    for(int k=0; k<MAX_ITER; ++k) {
        std::cout << "\n--- Iteration " << k << " ---" << std::endl;

        if (neighborhoods.empty()) break;

        // 1. SORT neighborhoods by Max Regret (Descending)
        neighborhoods.sort([](const Neighborhood& a, const Neighborhood& b) {
            return a.max_regret > b.max_regret;
        });

        // 2. CHECK CONVERGENCE
        double current_max_global_regret = neighborhoods.front().max_regret;
        std::cout << "Max Global Regret in queue: " << current_max_global_regret << std::endl;
        
        if(current_max_global_regret < 0.005) {
            std::cout << "Converged." << std::endl;
            break;
        }

        // 3. EXTRACT BATCH based on command line argument 'target_batch_size'
        std::vector<Neighborhood> batch_to_process;
        int count = 0;
        
        while(!neighborhoods.empty() && count < target_batch_size) {
            batch_to_process.push_back(neighborhoods.front());
            neighborhoods.pop_front();
            count++;
        }

        // 4. PROCESS BATCH
        std::cout << "Processing batch of " << batch_to_process.size() << " neighborhoods." << std::endl;

        for (auto& best_neighborhood : batch_to_process) {
            std::cout << "  > Planning for weight (Regret: " << best_neighborhood.max_regret << ")" << std::endl;

            // --- PLAN ---
            Vector new_w = best_neighborhood.candidate_w;
            Vector new_f = solvePlanningProblem(new_w, setup);
            int new_id = database.size();

            // --- DUPLICATE CHECK ---
            bool is_duplicate = false;
            for(size_t i=0; i<database.size(); ++i) {
                double dist = 0.0;
                for(int j=0; j<3; ++j) dist += std::pow(new_f[j] - database[i].f[j], 2);
                if(std::sqrt(dist) < 0.01) { 
                    is_duplicate = true; break; 
                }
            }

            if (is_duplicate) {
                duplicate_count++;
                std::cout << "    Duplicate detected! Discarding neighborhood." << std::endl;
                continue; 
            }

            // --- SAVE NEW SAMPLE ---
            database.push_back({new_id, new_w, new_f});
            printVector("    New Cost", new_f);
            
            logFile << k << "," << new_w[0] << "," << new_w[1] << "," << new_w[2] << ", "
                    << new_f[0] << "," << new_f[1] << "," << new_f[2] << ", "
                    << best_neighborhood.max_regret << "\n";
            logFile.flush();

            // --- SUBDIVIDE ---
            int d = best_neighborhood.id_d;
            int r = best_neighborhood.id_r;
            int t = best_neighborhood.id_t;

            int sets[3][3] = { {d, r, new_id}, {d, new_id, t}, {new_id, r, t} };
            
            for(int i=0; i< num_obj; ++i) {
                Neighborhood n;
                n.id_d = sets[i][0]; n.id_r = sets[i][1]; n.id_t = sets[i][2];
                std::vector<SampledCost> sub_corners = {
                    database[n.id_d], database[n.id_r], database[n.id_t]
                };
                
                RegretResult res = solveMaxRegretLP(sub_corners, global_max_costs);
                n.max_regret = res.max_regret;
                n.candidate_w = res.worst_w;
                neighborhoods.push_back(n); 
            }
        }
    }

    logFile.close();
    std::cout << "RPS Batch Finished. Total Samples: " << database.size() << std::endl;
    std::cout << "Number of duplicates: " << duplicate_count << std::endl;
    return 0;
}
