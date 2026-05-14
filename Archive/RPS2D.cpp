/*
 * Regret-Based Pareto Front Sampling (C++ Implementation)
 * Integrated with Custom Cost Function (Risk, Time) for 2D Pareto Fronts
 * Features: Generalized Linearly Independent Checking & Multi-Scenario Environment
 *
 * Dependencies: OMPL, Gurobi C++, Boost, Eigen3
 *
 * Compilation Command:
 g++ -m64 -g RPS2D.cpp -o RPS_2D \
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

// Eigen
#include <Eigen/Dense>

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

// Represents a sample in our database: {weight, cost_vector}
struct SampledCost {
    int id; // ID in the database
    Vector w; // Weight Vector
    Vector f; // Cost Vector f(s) = [Risk, Time]
};

// Generalized Neighborhood definition (works for 2D, 3D, etc.)
struct Neighborhood {
    std::vector<int> corner_ids; // Indices of the corners in the database
    double max_regret; // the worst case regret found in this simplex
    Vector candidate_w; // The weight that causes this max regret -> pivot. 
    bool is_duplicate; 
};

struct Simplex {
    std::vector<int> corner_ids; // Indices of the corners in the database
};

// This is the result of the LP solver.
struct RegretResult {
    double max_regret;
    Vector worst_w;
};

// ==========================================
// Environment and Obstacles (From RPS.cpp)
// ==========================================
class Obstacle {
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

class BoundaryObstacle : public Obstacle {
    double min_val, max_val;
public:
    BoundaryObstacle(double min_v, double max_v) : min_val(min_v), max_val(max_v) {}
    double getClearance(const State& s) const override {
        double dist_x = std::min(s.x - min_val, max_val - s.x);
        double dist_y = std::min(s.y - min_val, max_val - s.y);
        return std::min(dist_x, dist_y);
    }
    bool CheckCollision(const State& s) const override {
        // Collision if outside the box
        return (s.x < min_val || s.x > max_val || s.y < min_val || s.y > max_val);
    }
};

class RectangularObstacle : public Obstacle {
    double x_min, x_max, y_min, y_max;
public:
    RectangularObstacle(double x_min_v, double x_max_v, double y_min_v, double y_max_v) 
        : x_min(x_min_v), x_max(x_max_v), y_min(y_min_v), y_max(y_max_v) {}
    bool CheckCollision(const State& s) const override {
        return (s.x >= x_min && s.x <= x_max && s.y >= y_min && s.y <= y_max);
    }
    double getClearance(const State& s) const override {
        if (CheckCollision(s)) {
            double dist_left = s.x - x_min;
            double dist_right = x_max - s.x;
            double dist_bottom = s.y - y_min;
            double dist_top = y_max - s.y;
            return -std::min({dist_left, dist_right, dist_bottom, dist_top});
        } else {
            double dist_x = std::max({x_min - s.x, 0.0, s.x - x_max});
            double dist_y = std::max({y_min - s.y, 0.0, s.y - y_max});
            return std::sqrt(dist_x * dist_x + dist_y * dist_y);
        }
    }
};

class Environment {
    std::vector<std::shared_ptr<Obstacle>> obstacles;
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
    double speed_jump(double y) const {
        return (y < 13.0) ? 100.0 : 2.0;
    }
    double speed_smooth(double y) const {
        // Smooth transition between 100 and 2 around transition_mid
        double speed_slow = 2.0;
        double speed_fast = 100.0;
        double transition_mid = 18.0;
        double k = 1.0; // +- 5. 
        double exp_term = std::exp(-k * (y - transition_mid));
        double ratio = 1.0 / (1.0 + exp_term);

        return speed_fast + ratio * (speed_slow - speed_fast);
    }
    std::vector<double> calculateSegmentCost(const State& s_from, const State& s_to) const {
        std::vector<double> cost(2, 0.0); // 2D: [Risk, Time]

        // 1. Cost[0]: Risk
        double sum_segment_risk = 0.0;
        State prev_state = s_from;
        State curr_state;
        State center_state;

        int steps = 5001; 

        for (int i = 1; i <= steps; ++i) {
            double ratio = (double)i / steps;
            curr_state.x = s_from.x + ratio * (s_to.x - s_from.x);
            curr_state.y = s_from.y + ratio * (s_to.y - s_from.y);

            center_state.x = (curr_state.x + prev_state.x) / 2.0;
            center_state.y = (curr_state.y + prev_state.y) / 2.0;

            sum_segment_risk += getRiskAtState(center_state);
            prev_state = curr_state;
        }

        cost[0] = 1.0 * sum_segment_risk * getEuclideanDist(s_from, s_to) / steps;

        // 2. Cost[1]: Travel Time
        double Time = 0.0;
        prev_state = s_from;

        for (int i = 1; i <= steps; ++i) {
            double ratio = (double)i / steps;
            curr_state.x = s_from.x + ratio * (s_to.x - s_from.x);
            curr_state.y = s_from.y + ratio * (s_to.y - s_from.y);

            // Use speed_jump or speed_smooth based on preference
            double speed = speed_smooth(curr_state.y);
            double seg_dist = getEuclideanDist(curr_state, prev_state);
            Time += seg_dist / speed;
            
            prev_state = curr_state;
        }
        cost[1] = Time;

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

void saveDatabaseToCSV(const std::string& filename, const std::vector<SampledCost>& database) {
    std::ofstream outFile(filename);

    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    outFile << "ID,W_Risk,W_Time,Cost_Risk,Cost_Time\n";

    for (const auto& s : database) {
        outFile << s.id << ",";
        for (size_t i = 0; i < s.w.size(); ++i) {
            outFile << s.w[i];
            if (i < s.w.size() - 1) outFile << ","; 
        }
        outFile << ","; 
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
        description_ = "Weighted Risk/Time"; 
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
        for(size_t i = 0; i< weights.size() && i < obj_costs.size(); ++i){
            scalar_cost += weights[i] * obj_costs[i];
        }
        return ob::Cost(scalar_cost);
    }

private:
    Vector weights;
};

// Collision Checking using the Environment
bool isStateValid(const ob::State *state) {
    const auto *pos = state->as<ob::RealVectorStateSpace::StateType>();
    return !global_env->checkCollision({pos->values[0], pos->values[1]});
}

// Evaluate the full path to get the [Risk, Time] vector
Vector evaluatePathCosts(og::PathGeometric& path) {
    int num_obj = 2;
    Vector total_costs(num_obj, 0.0);
    const auto& states = path.getStates();
    for (size_t i = 0; i < states.size() - 1; ++i) {
        const auto* p1 = states[i]->as<ob::RealVectorStateSpace::StateType>();
        const auto* p2 = states[i+1]->as<ob::RealVectorStateSpace::StateType>();
        
        State st1 = {p1->values[0], p1->values[1]};
        State st2 = {p2->values[0], p2->values[1]};

        std::vector<double> segment_costs = global_env->calculateSegmentCost(st1, st2);

        for(int k=0; k < num_obj; ++k) total_costs[k] += segment_costs[k];
    }
    return total_costs;
}

// Using OMPL Solver. Inputs : weight, and setup. Outputs: Cost Vector. 
Vector solvePlanningProblem(const Vector& w, og::SimpleSetup& setup) {
    setup.clear();
    auto planner(std::make_shared<og::RRTstar>(setup.getSpaceInformation()));
    planner->setRange(2.0); 
    setup.setPlanner(planner);

    auto obj = std::make_shared<CustomWeightedObjective>(setup.getSpaceInformation(), w);
    setup.setOptimizationObjective(obj);

    setup.solve(60.0); // Adjusted solve time for demonstration
    return evaluatePathCosts(setup.getSolutionPath());
}

// ==========================================
// 3. Gurobi LP Solver & Neighborhood Utils
// ==========================================

RegretResult solveMaxRegretLP(const std::vector<SampledCost>& corners, const std::vector<double>& global_max_costs) {
    int num_objectives = corners[0].w.size(); // Generalize to n
    int k = corners.size(); // number of corners in the neighborhood
   try {
        GRBEnv env = GRBEnv(true);
        env.set("LogFile", "gurobi.log");
        env.start();
        GRBModel model = GRBModel(env);
        model.set(GRB_IntParam_OutputFlag, 0);

        std::vector<GRBVar> lambda(k);
        for(int i=0; i<k; ++i) 
            lambda[i] = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "lambda");

        std::vector<GRBVar> w(num_objectives);
        for(int j=0; j<num_objectives; ++j) 
            w[j] = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "w");

        GRBVar X = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "Regret");

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

        GRBLinExpr P = 0;
        for(int i=0; i<k; ++i) P += lambda[i] * u_corners[i];

        for(int i=0; i<k; ++i) {
            GRBLinExpr w_dot_fs = 0.0;
            for(int j=0; j<num_objectives; ++j) w_dot_fs += w[j] * corners[i].f[j] / global_max_costs[j];
            model.addConstr(X <= w_dot_fs - P);
        }

        model.setObjective(GRBLinExpr(X), GRB_MAXIMIZE);
        model.optimize();
        
        Vector res_w;
        for(int j=0; j<num_objectives; ++j) res_w.push_back(w[j].get(GRB_DoubleAttr_X));
        return {X.get(GRB_DoubleAttr_X), res_w};

    } catch(GRBException e) {
        std::cerr << "Gurobi Error: " << e.getMessage() << std::endl;
        return {-1.0, {}};
    }
}

// Advanced Linear Independence Checker
bool isLinearlyIndependent(const std::vector<int>& corner_ids, const std::vector<SampledCost>& database, int num_obj){
    int n = corner_ids.size();
    Eigen::MatrixXd A(num_obj, n);
    for (int i = 0; i < n ; ++i ){
        const Vector& wi = database[corner_ids[i]].w;
        for (int j = 0; j < num_obj; ++j){
            A(j,i) = wi[j];
        }
    }
    double det = A.determinant(); // Works properly for n=num_obj (e.g. 2x2, 3x3)
    return std::abs(det) > 1e-9;
}

// Generalized Neighborhood Splitter
void splitNeighborhood(const Neighborhood& N, int new_id, const std::vector<SampledCost>& database, int num_obj, const std::vector<double>& global_max_costs, std::list<Neighborhood>& neighborhoods){
    int n = N.corner_ids.size(); // For 2D, this is 2 (a line segment)
    for (int i = 0 ; i < n ; ++i){
        Neighborhood child;
        child.corner_ids = N.corner_ids; 
        child.corner_ids[i] = new_id; // Replace one corner with the new pivot
        
        if (!isLinearlyIndependent(child.corner_ids, database, num_obj)){
            continue; 
        }
        std::vector<SampledCost> corners;
        for (int id : child.corner_ids){
            corners.push_back(database[id]);
        }
        RegretResult regret = solveMaxRegretLP(corners, global_max_costs);
        child.max_regret = regret.max_regret;
        child.candidate_w = regret.worst_w;
        child.is_duplicate = false;
        neighborhoods.push_back(child);
        
        std::cout << "New Neighborhood Corners IDs: ";
        for (int id : child.corner_ids) std::cout << id << " ";
        std::cout << "with Max Regret: " << child.max_regret << std::endl;
    }
}


// ==========================================
// 4. Scenarios
// ==========================================
void configureEnvironment(int scenario_id) {
    global_env = std::make_shared<Environment>();
    global_env->addObstacle(std::make_shared<BoundaryObstacle>(0.0, 40.0));

    switch(scenario_id) {
        case 0: 
            std::cout << "Loading Scenario: Empty Space (Boundary Only)" << std::endl;
            break;
        case 1:
            std::cout << "Loading Scenario: Single Circle" << std::endl;
            global_env->addObstacle(std::make_shared<CircularObstacle>(11.0, 13.0, 3.0));
            break;
        case 2:
            std::cout << "Loading Scenario: Two Circles" << std::endl;
            global_env->addObstacle(std::make_shared<CircularObstacle>(11.0, 13.0, 3.0));
            global_env->addObstacle(std::make_shared<CircularObstacle>(11.0, 21.0, 2.0));
            break;
        case 3: 
            std::cout << "Loading Scenario: Smooth Velocity Change (No Obstacles)" << std::endl;
            break;
        case 4: 
            std::cout << "Loading Scenario: Smooth Velocity Change + Single Circle" << std::endl;
            global_env->addObstacle(std::make_shared<CircularObstacle>(11.0, 13.0, 3.0));
            break;
        case 5: 
            std::cout << "Loading Scenario: Smooth Velocity Change + Rectangular Obstacles" << std::endl;
            global_env->addObstacle(std::make_shared<RectangularObstacle>(6.0,17.0,9.0,13.0));
            global_env->addObstacle(std::make_shared<RectangularObstacle>(6.0,17.0,17.0,21.0));
            global_env->addObstacle(std::make_shared<RectangularObstacle>(6.0,17.0,25.0,29.0));
            break;
        default:
            std::cout << "Unknown Scenario. Defaulting to Empty." << std::endl;
            break;
    }
}


// ==========================================
// 5. Main Loop
// ==========================================

int main(int argc, char* argv[]) {
    int scenario = 1; 
    if (argc > 1) {
        scenario = std::stoi(argv[1]);
    }

    configureEnvironment(scenario);

    std::string filename = "RPS2D_log_scenario_" + std::to_string(scenario) + ".txt";
    logFile.open(filename);
    
    if (!logFile.is_open()) {
        std::cerr << "Error: Could not open log file " << filename << std::endl;
        return 1;
    }

    std::cout << "Saving data to: " << filename << std::endl;
    logFile << "Iteration, w1,w2, f1,f2, MaxRegret, is_duplicate\n";

    auto stateSpace = std::make_shared<ob::RealVectorStateSpace>(2);
    stateSpace->setBounds(0.0, 40.0); 
    og::SimpleSetup setup(stateSpace);
    ob::ScopedState<> start(stateSpace);
    setup.setStateValidityChecker(isStateValid);
    start[0] = 1.0; start[1] = 15.0;
    ob::ScopedState<> goal(stateSpace);
    goal[0] = 21.0; goal[1] = 15.0;
    
    int duplicate_count = 0;
    setup.setStartAndGoalStates(start, goal);

    std::vector<SampledCost> database;
    int num_obj = 2; // Risk, Time

    std::vector<Vector> corner_weights = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        //{0.0, 0.0, 1.0} 
    };
    std::vector<Vector> corner_case;
    
    if (scenario == 1){
        corner_case = {
            {20.1494,101.356,10.0747}, 
            {31.6239,1.7663,15.8119},
            {25.4269,143.952,2.21921}
        };
    } else if (scenario == 2){
        corner_case = {
            {20.1936,49.5278,10.0968},
            {43.9008,2.97585,21.9504},
            {25.3565,101.833,2.2156}
        };
    } else if (scenario == 0) {
        corner_case = {
            {20.0154,0.959798,10.0077},
            {20.3631,0.959532,10.1816},
            {24.5904,2.81677,2.21509}
        };
    } else if (scenario == 3) {
        corner_case = {
            {20.648,0.98232,0.217853},
            {20.8237,0.975242,0.226832},
            {20.8425,0.98513,0.213617}
        };
    } else if (scenario == 4) {
        corner_case = {
            {21.4985,62.0615,0.239344},
            {30.9436,1.94354,4.81113},
            {21.2453,70.2198,0.234114}
        };
    } else if (scenario == 5) {
        corner_case = {
            {22.0413,34.364,0.240193},
            {22.2211,10.4827,0.231839},
            {22.5295,12.0934,0.237025}
        };
    }
    std::cout << "--- Initializing Corners ---" << std::endl;

    std::vector<double> global_max_costs(num_obj, 1.0); 
    for (int i = 0; i < num_obj ; ++i) {
        Vector f = corner_cases[i];
        database.push_back({i, corner_weights[i], f});
        
        for(int k=0; k < num_obj; ++k) {
            if(f[k] > global_max_costs[k]) global_max_costs[k] = f[k];
        }
        logFile << i-num_obj << "," << corner_weights[i][0] << "," << corner_weights[i][1] << ", "
                << f[0] << "," << f[1] << ", " << 0.0 << ", 0\n";
    }

    std::list<Neighborhood> neighborhoods;

    Neighborhood initial_neighborhood;
    initial_neighborhood.corner_ids = {0, 1};
    std::vector<SampledCost> initial_corners = {
        database[0],
        database[1]
    };
    RegretResult initial_regret = solveMaxRegretLP(initial_corners, global_max_costs);
    initial_neighborhood.max_regret = initial_regret.max_regret;
    initial_neighborhood.candidate_w = initial_regret.worst_w;
    initial_neighborhood.is_duplicate = false;
    neighborhoods.push_back(initial_neighborhood);
    
    std::cout << "Initial Max Regret: " << initial_neighborhood.max_regret << std::endl;

    int MAX_ITER = 30; 
    double threshold_duplicate = 0.01; 
    
    for(int k=0; k<MAX_ITER; ++k) {
        std::cout << "\n--- Iteration " << k << " ---" << std::endl;

        auto best_it = neighborhoods.begin();
        double max_global_regret = -1.0;
        for(auto it = neighborhoods.begin(); it != neighborhoods.end(); ++it) {
            if(it->max_regret > max_global_regret && !it->is_duplicate) {
                max_global_regret = it->max_regret;
                best_it = it;
            }
        }
        
        std::cout << "Selected Neighborhood with Max Regret: " << max_global_regret << std::endl;
        std::cout << "Iteration " << k << ": Solving for weights " << max_global_regret << " Line Segment Corners IDs: ";
        for (int id : best_it->corner_ids) std::cout << id << " ";
        std::cout << std::endl;
        
        if(max_global_regret < 0.00005) {
            std::cout << "Converged." << std::endl;
            break;
        }

        Vector new_w = best_it->candidate_w; 
        Vector new_f = solvePlanningProblem(new_w, setup);
        int new_id = database.size();
        
        bool is_duplicate = false;
        int duplicate_id = -1;

        for(size_t i=0; i<database.size(); ++i) {
            double dist = 0.0;
            // Calculate Euclidean distance in Cost Space for convergence/duplicate check
            for(int j=0; j<2; ++j) dist += std::pow(new_f[j] - database[i].f[j], 2);
            
            if(std::sqrt(dist) < threshold_duplicate) { 
                is_duplicate = true;
                duplicate_id = i;
                break;
            }
        }
        
        std::vector<int> old_corner_ids = best_it->corner_ids;

        if (is_duplicate) {
            duplicate_count++;
            std::cout << "Duplicate detected! (Identical to ID " << duplicate_id << "). Discarding neighborhood." << std::endl;
            logFile << k << "," << new_w[0] << "," << new_w[1] << ", "
                    << new_f[0] << "," << new_f[1] << ", "
                    << max_global_regret << ", 1\n";
            logFile.flush();
            neighborhoods.erase(best_it);
            continue; 
        }

        database.push_back({new_id, new_w, new_f});
        printVector("New weight", new_w);
        printVector("New cost", new_f);
        
        logFile << k << "," << new_w[0] << "," << new_w[1] << ", "
                << new_f[0] << "," << new_f[1] << ", "
                << max_global_regret << ", 0\n";
        logFile.flush();
        
        neighborhoods.erase(best_it);
    
        Neighborhood old_N;
        old_N.corner_ids = old_corner_ids;
        splitNeighborhood(old_N, new_id, database, num_obj, global_max_costs, neighborhoods);
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

    std::string db_filename = filename;
    size_t dotPos = db_filename.find_last_of('.'); 
    if (dotPos != std::string::npos) {
        db_filename.insert(dotPos, "_database"); 
    } else {
        db_filename += "_database.csv"; 
    }

    saveDatabaseToCSV(db_filename, database);

    return 0;
}
