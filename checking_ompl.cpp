/*
 * OMPL Planner using RRT* with RPS Performance Checking (No Smoothing)
 * * Refactored with Object-Oriented Environment and Obstacle Handling
 * * Logic:
 * - Solves using RRT* with iterative convergence loop.
 * - Uses shared Environment class for consistent metrics with RPS.
 *
 * Compilation:
 g++ -m64 -O3 groundtruth_ompl.cpp -o groundTruth \
 -I/home/seung/ompl/src \
 -L/home/seung/ompl/build/lib \
 -I/usr/include/eigen3 \
 -lompl -lpthread
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <limits>
#include <set>



#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/base/OptimizationObjective.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>

namespace ob = ompl::base;
namespace og = ompl::geometric;

// ==========================================
// 1. Obstacle & Environment System
// ==========================================

struct StateStruct {
    double x, y;
};

// Abstract Base Class for all Obstacles
class Obstacle {
public:
    virtual ~Obstacle() = default;
    virtual bool CheckCollision(const StateStruct& s) const = 0;
    virtual double getClearance(const StateStruct& s) const = 0;
};

// Circular Obstacle Implementation
class CircularObstacle : public Obstacle {
    double cx, cy, radius;
public:
    CircularObstacle(double x, double y, double r) : cx(x), cy(y), radius(r) {}

    bool CheckCollision(const StateStruct& s) const override {
        return getClearance(s) <= 0.0;
    }

    double getClearance(const StateStruct& s) const override {
        double dist = std::sqrt(std::pow(s.x - cx, 2) + std::pow(s.y - cy, 2));
        return dist - radius;
    }
};

// Boundary Obstacle Implementation
class BoundaryObstacle : public Obstacle {
    double min_val, max_val;
public:
    BoundaryObstacle(double min_v, double max_v) : min_val(min_v), max_val(max_v) {}

    bool CheckCollision(const StateStruct& s) const override {
        return (s.x < min_val || s.x > max_val || s.y < min_val || s.y > max_val);
    }

    double getClearance(const StateStruct& s) const override {
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

    // Check validity (Collision Free)
    bool isValid(const StateStruct& s) const {
        for (const auto& obs : obstacles) {
            // Using small buffer 0.1 consistent with RPS code
            if (obs->getClearance(s) <= 0.1) return false;
        }
        return true;
    }

    // Calculate aggregated Risk at a specific point
    double getPointRisk(const StateStruct& s) const {
        double total_risk = 0.0;
        const double max_risk_penalty = 1e6; 

        for (const auto& obs : obstacles) {
            double clearance = obs->getClearance(s);
            if (clearance <= 0.1) return max_risk_penalty;
            
            // Risk = 1 / (dist^2)
            double val = (clearance * clearance) + 1e-3;
            total_risk += 1.0 / val;
        }
        return std::min(total_risk, max_risk_penalty);
    }

    double getEuclideanDist(const StateStruct& s1, const StateStruct& s2) const {
        return std::sqrt(std::pow(s1.x - s2.x, 2) + std::pow(s1.y - s2.y, 2));
    }

    // Returns vector: [Travel Distance, Risk, Travel Time]
    std::vector<double> calculateSegmentCost(const StateStruct& s_from, const StateStruct& s_to) const {
        std::vector<double> cost(3, 0.0);

        // 1. Cost[0]: Distance
        cost[0] = getEuclideanDist(s_from, s_to);

        // 2. Cost[1]: Risk (Numerical Integration)
        double sum_segment_risk = 0.0;
        StateStruct prev_state = s_from;
        StateStruct curr_state;
        StateStruct center_state;

        // High fidelity integration steps (Consistent with RPS.cpp)
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

// ==========================================
// 2. Environment Configuration Helper
// ==========================================

void configureEnvironment(int scenario_id) {
    global_env = std::make_shared<Environment>();

    // Always add boundary (0 to 40)
    global_env->addObstacle(std::make_shared<BoundaryObstacle>(0.0, 40.0));

    switch(scenario_id) {
        case 0: // No Obstacles
            std::cout << "Scenario 0: Empty Space (Boundary Only)" << std::endl;
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
// 3. OMPL Classes
// ==========================================

class CustomWeightedObjective : public ob::OptimizationObjective {
public:
    CustomWeightedObjective(const ob::SpaceInformationPtr &si, const std::vector<double>& weights)
        : ob::OptimizationObjective(si), weights_(weights) {}

    ob::Cost stateCost(const ob::State *s) const override { return ob::Cost(0.0); }
    ob::Cost motionCostHeuristic(const ob::State *s1, const ob::State *s2) const override { return ob::Cost(0.0); }

    ob::Cost motionCost(const ob::State *s1, const ob::State *s2) const override {
        const auto* p1 = s1->as<ob::RealVectorStateSpace::StateType>();
        const auto* p2 = s2->as<ob::RealVectorStateSpace::StateType>();
        StateStruct st1 = {p1->values[0], p1->values[1]};
        StateStruct st2 = {p2->values[0], p2->values[1]};
        
        // Use Global Environment for Consistent Cost Calculation
        std::vector<double> c = global_env->calculateSegmentCost(st1, st2);
        
        double sum = 0.0;
        for(size_t i=0; i<weights_.size() && i<c.size(); ++i) sum += weights_[i]*c[i];
        return ob::Cost(sum);
    }
private:
    std::vector<double> weights_;
};

class ObstacleValidityChecker : public ob::StateValidityChecker {
public:
    ObstacleValidityChecker(const ob::SpaceInformationPtr& si) : ob::StateValidityChecker(si) {}
    bool isValid(const ob::State* state) const override {
        const auto* s = state->as<ob::RealVectorStateSpace::StateType>();
        // Delegate to Global Environment
        return global_env->isValid({s->values[0], s->values[1]});
    }
};

// ==========================================
// 4. IO Helpers
// ==========================================

std::vector<double> parseWeights(std::string wStr) {
    std::vector<double> w;
    std::string clean;
    for(char c : wStr) {
        if(c != '\"' && c != ' ' && c != '\r' && c != '\n') clean += c;
    }
    std::replace(clean.begin(), clean.end(), ';', ' ');
    std::stringstream ss(clean);
    double temp;
    while(ss >> temp) w.push_back(temp);
    while(w.size() < 3) w.push_back(0.0);
    return w;
}

std::vector<std::string> extractWeightsFromCSV(const std::string& filename) {
    std::vector<std::string> weightsList;
    std::ifstream file(filename);
    if (!file.is_open()) { std::cerr << "Error opening file " << filename << "\n"; return weightsList; }
    std::string line;
    int lineNum = 0;
    while (std::getline(file, line)) {
        lineNum++;
        if (lineNum == 1 || lineNum % 2 == 0 || line.empty()) continue;
        weightsList.push_back(line);
    }
    return weightsList;
}
// ==========================================
// 6. Resume Helper (Add this before main)
// ==========================================
std::set<std::string> loadProcessedWeights(const std::string& filename) {
    std::set<std::string> processed;
    std::ifstream file(filename);
    if (!file.is_open()) return processed;

    std::string line;
    // Skip header if possible, but the logic below handles it anyway
    while (std::getline(file, line)) {
        // Format is: ..., "w1;w2;w3"
        // We look for the last pair of quotes
        size_t lastQuote = line.find_last_of('"');
        if (lastQuote == std::string::npos) continue;
        
        size_t firstQuoteOfLast = line.find_last_of('"', lastQuote - 1);
        if (firstQuoteOfLast == std::string::npos) continue;

        // Extract the content inside the quotes: "0.5;0.2;0.3"
        std::string wStr = line.substr(firstQuoteOfLast + 1, lastQuote - firstQuoteOfLast - 1);
        if (!wStr.empty()) {
            processed.insert(wStr);
        }
    }
    return processed;
}
// ==========================================
//  Main Function
// ==========================================
int main(int argc, char* argv[]) {
    // ------------------------------------------
    // 1. Configuration (Manual Setup)
    // ------------------------------------------
    int scenario = 1; // 0: Empty, 1: One Circle, 2: Two Circles
    
    // !!! SET YOUR TARGET WEIGHTS HERE !!!
    // Format: {Distance, Risk, Time}
    std::vector<double> target_weights = {0.0, 0.0, 1.0}; 

    std::cout << "--- Running Single Weight Case ---" << std::endl;
    std::cout << "Scenario: " << scenario << std::endl;
    std::cout << "Weights:  Dist=" << target_weights[0] 
              << ", Risk=" << target_weights[1] 
              << ", Time=" << target_weights[2] << std::endl;

    configureEnvironment(scenario);

    // 2. Setup Space and Bounds
    auto space(std::make_shared<ob::RealVectorStateSpace>(2));
    ob::RealVectorBounds bounds(2);
    bounds.setLow(0.0); bounds.setHigh(30.0);
    space->setBounds(bounds);

    // 3. Setup SimpleSetup
    og::SimpleSetup setup(space);
    ob::SpaceInformationPtr si = setup.getSpaceInformation();
    setup.setStateValidityChecker(std::make_shared<ObstacleValidityChecker>(si));

    // 4. Define Start and Goal
    ob::ScopedState<> start(space);
    ob::ScopedState<> goal(space);
    start[0] = 1.0; start[1] = 15.0;
    goal[0] = 21.0; goal[1] = 15.0;
    setup.setStartAndGoalStates(start, goal);

    // 5. Set Objective
    auto obj = std::make_shared<CustomWeightedObjective>(si, target_weights);
    setup.setOptimizationObjective(obj);

    // 6. Set Planner (RRT*)
    auto planner(std::make_shared<og::RRTstar>(si));
    planner->setRange(1.0);
    setup.setPlanner(planner);

    // ------------------------------------------
    // 7. Solve (Iterative Convergence)
    // ------------------------------------------
    std::cout << "Solving..." << std::endl;

    double prev_cost = std::numeric_limits<double>::infinity();
    double current_cost = std::numeric_limits<double>::infinity();
    
    // Performance settings
    double time_slice = 20;     // Seconds per batch (Lowered for single test speed)
    int max_batches = 10;        // Max iterations
    double improvement_threshold = 0.001;
    int batch_count = 0;

    // Initial Solve
    setup.solve(time_slice);
    if (setup.haveExactSolutionPath()) {
        current_cost = setup.getSolutionPath().cost(obj).value();
    }
    batch_count++;

    // Convergence Loop
    while (batch_count < max_batches) {
        std::cout << "  Batch " << batch_count << " Cost: " << current_cost << std::endl;

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
                std::cout << "  -> Converged! (Improvement < " << improvement_threshold << ")" << std::endl;
                break; 
            }
            current_cost = new_cost;
        }
        batch_count++;
    }

    // ------------------------------------------
    // 8. Results
    // ------------------------------------------
    if (setup.haveExactSolutionPath()) {
        std::cout << "\n--- Solution Found ---" << std::endl;
        og::PathGeometric& path = setup.getSolutionPath();
        
        // Calculate specific metrics
        double total_dist = 0.0, total_risk = 0.0, total_time = 0.0;
        const auto& states = path.getStates();

        for (size_t i = 0; i < states.size() - 1; ++i) {
            const auto* s1 = states[i]->as<ob::RealVectorStateSpace::StateType>();
            const auto* s2 = states[i+1]->as<ob::RealVectorStateSpace::StateType>();
            StateStruct st1 = {s1->values[0], s1->values[1]};
            StateStruct st2 = {s2->values[0], s2->values[1]};
            
            std::vector<double> seg = global_env->calculateSegmentCost(st1, st2);
            total_dist += seg[0];
            total_risk += seg[1];
            total_time += seg[2];
        }
        
        double weighted_fitness = total_dist*target_weights[0] + 
                                  total_risk*target_weights[1] + 
                                  total_time*target_weights[2];

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Total Length:      " << total_dist << std::endl;
        std::cout << "Total Risk:        " << total_risk << std::endl;
        std::cout << "Total Travel Time: " << total_time << std::endl;
        std::cout << "Weighted Cost:     " << weighted_fitness << std::endl;
        std::cout << "Waypoints:         " << states.size() << std::endl;

    } else {
        std::cout << "No solution found." << std::endl;
    }

    return 0;
}
