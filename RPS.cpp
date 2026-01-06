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

// Helper: Euclidean Distance
double getEuclideanDist(const State& s1, const State& s2) {
    return std::sqrt(std::pow(s1.x - s2.x, 2) + std::pow(s1.y - s2.y, 2));
}


// Cost Function for a segment between s_from and s_to. 
// Returns vector: [Travel Distance, Risk, Travel Time]

std::vector<double> calculateSegmentCost(const State& s_from, const State& s_to) {
    std::vector<double> cost(3, 0.0);

    // 1. Cost[0]: Distance
    cost[0] = getEuclideanDist(s_from, s_to);

    // 2. Cost[1]: Risk
    const double obstacle_cx = 11.0;
    const double obstacle_cy = 13.0;
    const double radius = 3.0;
    State obstacle = {obstacle_cx, obstacle_cy};

    double risk = 0.0;
    int num_steps = 10001;
    
    double sum_segment_risk = 0.0; 
    
    State previous_intermediate_risk = s_from;
    State intermediate_State_risk;
    State CenterOfSegment;

    for (int i = 1; i <= num_steps; ++i) {
        double ratio = (double)i / num_steps;
        
        intermediate_State_risk.x = s_from.x + ratio * (s_to.x - s_from.x);
        intermediate_State_risk.y = s_from.y + ratio * (s_to.y - s_from.y);
        
        CenterOfSegment = {
            (intermediate_State_risk.x + previous_intermediate_risk.x) / 2.0,
            (intermediate_State_risk.y + previous_intermediate_risk.y) / 2.0
        };

        // (Distance - Radius)^2
        double dist_to_obs = getEuclideanDist(CenterOfSegment, obstacle);
        double inverse_risk_segment = 1.0 / ((dist_to_obs - radius) * (dist_to_obs - radius) + 1e-3);
        
        if (inverse_risk_segment < 0.0){
                    inverse_risk_segment = 0.001;
        }

        previous_intermediate_risk = intermediate_State_risk;
        sum_segment_risk += inverse_risk_segment;
    }

    // Risk Formula
    risk = 1.0 * (sum_segment_risk) * cost[0] / num_steps;
    cost[1] = risk;

    // 3. Cost[2]: Travel Time
    State previous_intermediate_traveltime = s_from;
    State intermediate_traveltime;
    double Time = 0.0;

    for (int i = 1; i <= num_steps; ++i) {
        double ratio = (double)i / num_steps;
        intermediate_traveltime.x = s_from.x + ratio * (s_to.x - s_from.x);
        intermediate_traveltime.y = s_from.y + ratio * (s_to.y - s_from.y);
        
        double speed;
        // Logic: Lower Y is fast (Highway), Higher Y is slow
        if (intermediate_traveltime.y < 13.0) {
            speed = 100.0; // Highway
        } else {
            speed = 2.0;   // Slow zone
        }
        
        double distance_segment = getEuclideanDist(intermediate_traveltime, previous_intermediate_traveltime);
        Time += distance_segment / speed;
        previous_intermediate_traveltime = intermediate_traveltime;
    }
    cost[2] = Time;

    return cost;
}

std::vector<double> metricOfTrajectory(const std::vector<State>& trajectory) {
    std::vector<double> total_costs(3, 0.0); // [Distance, Risk, Time]

    for (size_t i = 0; i < trajectory.size() - 1; ++i) {
        std::vector<double> segment_costs = calculateSegmentCost(trajectory[i], trajectory[i + 1]);
        for (int k = 0; k < 3; ++k) {
            total_costs[k] += segment_costs[k];
        }
    }
    return total_costs;
}
// Represents a sample in our database: {weight, cost_vector}
struct SampledCost {
    int id; // ID in the database
    Vector w; // Weight Vector
    Vector f; // Cost Vector f(s) = [Dist, Risk, Time]
};
struct Neighborhood{
    // distance, risk, time
    int id_d,id_r, id_t; // Indices of the corners in the database
    double max_regret; // the worst case regret found in this traiangle
    Vector candidate_w; // The weight that causes this max regret -> pivot. 
};
struct Simplex{
    std::vector<int> corner_ids; // Indices of the corners in the database
};
// This is the result of the LP solver.
struct RegretResult {
    double max_regret;
    Vector worst_w;
};

void printVector(const std::string& label, const Vector& v) {
    std::cout << label << ": [ ";
    for (auto d : v) std::cout << d << " ";
    std::cout << "]" << std::endl;
}
std::ofstream logFile;
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
        std::vector<double> obj_costs = calculateSegmentCost(st1, st2);

        // 3. Weight them: w0*Dist + w1*Risk + w2*Time
        double scalar_cost = 0.0;
        for(size_t i = 0; i< weights.size() && i < obj_costs.size(); ++i){
            scalar_cost += weights[i] * obj_costs[i];
        }
        return ob::Cost(scalar_cost);
    }

private:
    Vector weights;
};

// Evaluate the full path to get the [Dist, Risk, Time] vector
Vector evaluatePathCosts(og::PathGeometric& path) {
    Vector total_costs(3, 0.0);
    const auto& states = path.getStates();

    for (size_t i = 0; i < states.size() - 1; ++i) {
        const auto* p1 = states[i]->as<ob::RealVectorStateSpace::StateType>();
        const auto* p2 = states[i+1]->as<ob::RealVectorStateSpace::StateType>();
        
        State st1 = {p1->values[0], p1->values[1]};
        State st2 = {p2->values[0], p2->values[1]};

        std::vector<double> segment_costs = calculateSegmentCost(st1, st2);

        for(int k=0; k<3; ++k) total_costs[k] += segment_costs[k];
    }
    return total_costs;
}



// Using OMPL Solver. Inputs : weight, and setup. Outputs: Cost Vector. 
Vector solvePlanningProblem(const Vector& w, og::SimpleSetup& setup) {
    auto obj = std::make_shared<CustomWeightedObjective>(setup.getSpaceInformation(), w);
    setup.setOptimizationObjective(obj);
    setup.clear();
    double prev_cost = std::numeric_limits<double>::infinity();
    double current_cost = std::numeric_limits<double>::infinity();

    double time_slice = 3.0;           
    double improvement_threshold = 0.0001; // 0.1% imporovement
    int max_batches = 20;             
    int batch_count = 0;

    setup.solve(time_slice);
    if (setup.haveExactSolutionPath()) {
        current_cost = setup.getSolutionPath().cost(obj).value();
    }
    batch_count++;

    while (batch_count < max_batches) {
        if (current_cost == std::numeric_limits<double>::infinity()) {
            setup.solve(time_slice);
            if (setup.haveExactSolutionPath()) {
                current_cost = setup.getSolutionPath().cost(obj).value();
            }
        } 
        else {
            prev_cost = current_cost;
            setup.solve(time_slice);

            double new_cost = setup.getSolutionPath().cost(obj).value();

            double improvement = (prev_cost - new_cost) / prev_cost;

            if (improvement < improvement_threshold) {
                std::cout << "Converged at batch " << batch_count << " (Imp: " << improvement << ")" << std::endl;
                break; 
            }
            current_cost = new_cost;
        }
        batch_count++;
    }
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

// Collision Checking
bool isStateValid(const ob::State *state) {
    const auto *pos = state->as<ob::RealVectorStateSpace::StateType>();
    double x = pos->values[0];
    double y = pos->values[1];

    // Define the Obstacle (Must match the one in calculateSegmentCost)
    double obs_x = 11.0;
    double obs_y = 13.0;
    double radius = 3.0; 

    double dist = std::sqrt(std::pow(x - obs_x, 2) + std::pow(y - obs_y, 2));

    // Valid if distance is greater than radius
    return dist > radius + 0.1; 
}
// ==========================================
// 5. Main Loop

int main() {
    logFile.open("RPS_log.txt");
    logFile << "Iteration, w1,w2,w3, f1,f2,f3, MaxRegret\n";
    auto stateSpace = std::make_shared<ob::RealVectorStateSpace>(2);
    stateSpace->setBounds(0.0, 30.0); 
    og::SimpleSetup setup(stateSpace);
    ob::ScopedState<> start(stateSpace);
    setup.setStateValidityChecker(isStateValid);
    start[0] = 1.0; start[1] = 15.0;
    ob::ScopedState<> goal(stateSpace);
    goal[0] = 21.0; goal[1] = 15.0;
    int duplicate_count = 0;
    setup.setStartAndGoalStates(start, goal);
    setup.setPlanner(std::make_shared<og::RRTstar>(setup.getSpaceInformation()));

    // 2. Initialize Algorithm
    std::vector<SampledCost> database;
    int num_obj = 3; // Distance, Risk, Time

    // Standard Basis Weights (Corners of the 3-obj simplex)
    std::vector<Vector> corner_weights = {
        {1.0, 0.0, 0.0}, // Pure Distance
        {0.0, 1.0, 0.0}, // Pure Risk
        {0.0, 0.0, 1.0}  // Pure Time
    };

    std::cout << "--- Initializing Corners ---" << std::endl;

    // 1. Initialize Global Max Tracker
    // Start with 1.0 to avoid division by zero initially, or small epsilon
    std::vector<double> global_max_costs(3, 1.0); 
    for (int i = 0; i < 3 ; ++i) {
        Vector f = solvePlanningProblem(corner_weights[i], setup);
        database.push_back({i, corner_weights[i], f});

        // UPDATE GLOBAL MAX
        for(int k=0; k<3; ++k) {
            if(f[k] > global_max_costs[k]) global_max_costs[k] = f[k];
        }
        logFile << i-num_obj << "," << corner_weights[i][0] << "," << corner_weights[i][1] << ", " << corner_weights[i][2] << ", "
                << f[0] << "," << f[1] << ", " << f[2]
                << "" << "\n";
    }
    std::cout << "Global Max Costs after initialization: "
              << global_max_costs[0] << ", "
              << global_max_costs[1] << ", "
              << global_max_costs[2] << std::endl;
    std::list<Neighborhood> neighborhoods;

    // Create the FIRST neighborhood (the whole triangle, 0 - 1 - 2)

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
    neighborhoods.push_back(initial_neighborhood);
    std::cout << "Initial Max Regret: " << initial_neighborhood.max_regret << std::endl;
    
    //----- LOOP -----
    int MAX_ITER = 100; 
    
    for(int k=0; k<MAX_ITER; ++k) {
        std::cout << "\n--- Iteration " << k << " ---" << std::endl;

        auto best_it = neighborhoods.begin();
        double max_global_regret = -1.0;
        for(auto it = neighborhoods.begin(); it != neighborhoods.end(); ++it) {
            if(it->max_regret > max_global_regret) {
                max_global_regret = it->max_regret;
                best_it = it;
            }
        }
        std::cout << "Selected Neighborhood with Max Regret: " << max_global_regret << std::endl;
        std::cout << "Iteration " << k << ": Solving for weights " << max_global_regret << " Triangle Corners IDs: "
                  << best_it->id_d << ", " << best_it->id_r << ", " << best_it->id_t << std::endl;
        
        if(max_global_regret < 0.005) {
            std::cout << "Converged." << std::endl;
            break;
        }
        // plan for the candidate weight
        Vector new_w = best_it->candidate_w; // Pivot weight
        Vector new_f = solvePlanningProblem(new_w, setup);
        int new_id = database.size();
        // 1. CHECK FOR DUPLICATES
        bool is_duplicate = false;
        int duplicate_id = -1;

        for(size_t i=0; i<database.size(); ++i) {
            double dist = 0.0;
            // Calculate Euclidean distance in Cost Space
            for(int j=0; j<3; ++j) dist += std::pow(new_f[j] - database[i].f[j], 2);
            
            // Tolerance: If cost vector is within 0.01 of an existing one, it's a duplicate.
            if(std::sqrt(dist) < 0.01) { 
                is_duplicate = true;
                duplicate_id = i;
                continue;
            }
        }
        
        // 2. HANDLE THE RESULT
        if (is_duplicate) {
            duplicate_count++;
            std::cout << "Duplicate detected! (Identical to ID " << duplicate_id << ")" << std::endl;
            
            // CRITICAL: Do NOT add to database. Do NOT subdivide.
            // Simply remove the current neighborhood from the priority queue.
            // This tells the algorithm: "There is nothing more to find in this specific direction."
            neighborhoods.erase(best_it);
            //std::cout << "Number of duplicates so far: " << duplicate_count << std::endl;
            // Continue to next iteration to pick the NEXT best neighborhood
            continue; 
        }
        database.push_back({new_id, new_w, new_f});
        printVector("New weight", new_w);
        printVector("New cost", new_f);
        logFile << k << "," << new_w[0] << "," << new_w[1] << "," << new_w[2] << ", "
                << new_f[0] << "," << new_f[1] << "," << new_f[2] << ", "
                << max_global_regret << "\n";
        logFile.flush();
        int d = best_it->id_d;
        int r = best_it->id_r;
        int t = best_it->id_t;
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
    std::cout << "Number of duplicates : " << duplicate_count<< std::endl;
    std::cout << "RPS Completed." << std::endl;
    return 0;

}
