/*
 * Regret-Based Pareto Front Sampling with Weighted p-Norm Scalarization
 * (C++ Implementation)
 * Integrated with Custom Cost Function (Distance, Risk, Time)
 *
 * This extends the original MRPS algorithm (Botros et al., 2024) by replacing
 * the weighted sum scalarization (p=1) with a weighted p-norm scalarization.
 * The LP for max-regret is solved in the transformed lambda-space where
 *   lambda = phi^p(w),  with phi^p_i(w) = w_i^p / sum_j w_j^p
 * and the transformed objectives are g^p(f) = (f_1^p, ..., f_n^p).
 * In lambda-space the scalarization becomes a weighted sum of g^p(f),
 * so the concavity/LP structure of MRPS is preserved.
 *
 * Dependencies: OMPL, Gurobi C++, Boost
 *
 * Compilation Command:
 g++ -m64 -g RPS_pnorm.cpp -o RPS_pnorm \
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
// 0. Global p-norm parameter
// ==========================================
// p=1  => original weighted sum (MRPS)
// p=2  => weighted 2-norm
// p>=5 => approaches Tchebycheff; captures more of the Pareto front
double P_NORM = 10.0;

// ==========================================
// 1. Data Structures & Cost Helpers
// ==========================================

using Vector = std::vector<double>;

struct State {
    double x;
    double y;
};

struct SampledCost {
    int id;
    Vector w;       // Weight vector in w-space (on the simplex)
    Vector f;       // Objective vector f(s) = [Dist, Risk, Time]
    Vector lambda;  // Transformed weight: lambda = phi^p(w)
    Vector gp_f;    // Transformed objectives: g^p(f) = [f1^p, f2^p, f3^p]
};

struct Neighborhood {
    int id_d, id_r, id_t;
    double max_regret;
    Vector candidate_w;      // Best pivot weight in w-space
    Vector candidate_lambda; // Best pivot weight in lambda-space
    bool is_duplicate;
};

struct RegretResult {
    double max_regret;
    Vector worst_lambda; // In lambda-space
    Vector worst_w;      // Mapped back to w-space
};

// ==========================================
// p-Norm Transform Utilities
// ==========================================

// phi^p: w-space -> lambda-space
//   lambda_i = w_i^p / sum_j w_j^p
Vector phi_p(const Vector& w, double p) {
    int n = w.size();
    Vector lambda(n);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        lambda[i] = std::pow(w[i], p);
        sum += lambda[i];
    }
    if (sum > 0.0) {
        for (int i = 0; i < n; ++i) lambda[i] /= sum;
    }
    return lambda;
}

// (phi^p)^{-1}: lambda-space -> w-space
//   w_i = lambda_i^{1/p} / sum_j lambda_j^{1/p}
Vector phi_p_inverse(const Vector& lambda, double p) {
    int n = lambda.size();
    Vector w(n);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        w[i] = std::pow(std::max(lambda[i], 0.0), 1.0 / p);
        sum += w[i];
    }
    if (sum > 0.0) {
        for (int i = 0; i < n; ++i) w[i] /= sum;
    }
    return w;
}

// g^p: f -> (f_1^p, ..., f_n^p)
Vector g_p(const Vector& f, double p) {
    Vector z(f.size());
    for (size_t i = 0; i < f.size(); ++i) {
        z[i] = std::pow(f[i], p);
    }
    return z;
}

// ==========================================
// Environment (unchanged from original)
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
    double center_x, center_y, radius;
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
        return (s.x < min_val || s.x > max_val || s.y < min_val || s.y > max_val);
    }
};

class Environment {
    std::vector<std::shared_ptr<Obstacle>> obstacles;
public:
    void addObstacle(const std::shared_ptr<Obstacle>& obs) { obstacles.push_back(obs); }
    bool checkCollision(const State& s) const {
        for (const auto& obs : obstacles)
            if (obs->CheckCollision(s)) return true;
        return false;
    }
    double getRiskAtState(const State& s) const {
        double total_risk = 0.0;
        for (const auto& obs : obstacles) {
            double clearance = obs->getClearance(s);
            if (clearance <= 0.1) total_risk += 1e6;
            else total_risk += 1.0 / (clearance * clearance + 1e-3);
        }
        return std::min(total_risk, 1e6);
    }
    double getEuclideanDist(const State& s1, const State& s2) const {
        return std::sqrt(std::pow(s1.x - s2.x, 2) + std::pow(s1.y - s2.y, 2));
    }
    std::vector<double> calculateSegmentCost(const State& s_from, const State& s_to) const {
        std::vector<double> cost(3, 0.0);
        cost[0] = getEuclideanDist(s_from, s_to);

        double sum_segment_risk = 0.0;
        State prev_state = s_from;
        State curr_state, center_state;
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
    outFile << "ID,W_Distance,W_Risk,W_Time,Lambda_1,Lambda_2,Lambda_3,"
            << "Cost_Distance,Cost_Risk,Cost_Time,GpF_1,GpF_2,GpF_3\n";
    for (const auto& s : database) {
        outFile << s.id << ",";
        for (size_t i = 0; i < s.w.size(); ++i) {
            outFile << s.w[i];
            if (i < s.w.size() - 1) outFile << ",";
        }
        outFile << ",";
        for (size_t i = 0; i < s.lambda.size(); ++i) {
            outFile << s.lambda[i];
            if (i < s.lambda.size() - 1) outFile << ",";
        }
        outFile << ",";
        for (size_t i = 0; i < s.f.size(); ++i) {
            outFile << s.f[i];
            if (i < s.f.size() - 1) outFile << ",";
        }
        outFile << ",";
        for (size_t i = 0; i < s.gp_f.size(); ++i) {
            outFile << s.gp_f[i];
            if (i < s.gp_f.size() - 1) outFile << ",";
        }
        outFile << "\n";
    }
    outFile.close();
    std::cout << "Database saved to: " << filename << std::endl;
}

// ==========================================
// 2. OMPL Planner Setup — p-Norm Objective
// ==========================================

// The planner minimizes ||f(s)||_w^p = (sum_i (w_i * f_i)^p)^{1/p}
// which is equivalent to minimizing sum_i lambda_i * f_i^p  in lambda-space.
// Since OMPL accumulates motionCost additively along segments, and
//   (sum_i (w_i f_i)^p)^{1/p} is NOT additive over path segments for p>1,
// we instead optimize the p-th power of the p-norm per segment:
//   sum_i (w_i * f_i^{segment})^p
// and accumulate this additively. The total is then sum over segments of
// this quantity, which equals sum_i w_i^p * (sum_seg f_i^{seg})^p only
// approximately. For exact optimality we'd need a different framework,
// but this is the standard practical approach (cf. Botros et al. using
// discretized turning radius search, also approximate).
//
// More precisely: we accumulate  sum_i lambda_i * (f_i^{segment})^p
// where lambda = phi^p(w). This makes the per-segment cost a weighted
// sum of transformed per-segment objectives, preserving additivity.

class CustomWeightedObjective : public ob::OptimizationObjective {
public:
    CustomWeightedObjective(const ob::SpaceInformationPtr& si, const Vector& weights, double p)
        : ob::OptimizationObjective(si), weights(weights), p_norm(p) {
        // Precompute lambda = phi^p(w) for the additive cost
        lambda = phi_p(weights, p_norm);
        description_ = "Weighted p-Norm (p=" + std::to_string(p_norm) + ") Distance/Risk/Time";
    }

    ob::Cost stateCost(const ob::State* s) const override {
        return ob::Cost(0.0);
    }

    ob::Cost motionCost(const ob::State* s1, const ob::State* s2) const override {
        const auto* p1 = s1->as<ob::RealVectorStateSpace::StateType>();
        const auto* p2 = s2->as<ob::RealVectorStateSpace::StateType>();

        State st1 = {p1->values[0], p1->values[1]};
        State st2 = {p2->values[0], p2->values[1]};

        std::vector<double> obj_vecs = global_env->calculateSegmentCost(st1, st2);

        // Compute: sum_i lambda_i * f_i^p  (additive in lambda-space)
        double cost_scalar = 0.0;
        for (size_t i = 0; i < lambda.size() && i < obj_vecs.size(); ++i) {
            cost_scalar += lambda[i] * std::pow(obj_vecs[i], p_norm);
        }
        return ob::Cost(cost_scalar);
    }

private:
    Vector weights;
    Vector lambda;
    double p_norm;
};

bool isStateValid(const ob::State* state) {
    const auto* pos = state->as<ob::RealVectorStateSpace::StateType>();
    return !global_env->checkCollision({pos->values[0], pos->values[1]});
}

Vector evaluatePathCosts(og::PathGeometric& path) {
    Vector total_costs(3, 0.0);
    const auto& states = path.getStates();
    for (size_t i = 0; i < states.size() - 1; ++i) {
        const auto* p1 = states[i]->as<ob::RealVectorStateSpace::StateType>();
        const auto* p2 = states[i + 1]->as<ob::RealVectorStateSpace::StateType>();
        State st1 = {p1->values[0], p1->values[1]};
        State st2 = {p2->values[0], p2->values[1]};
        std::vector<double> segment_costs = global_env->calculateSegmentCost(st1, st2);
        for (int k = 0; k < 3; ++k) total_costs[k] += segment_costs[k];
    }
    return total_costs;
}

Vector solvePlanningProblem(const Vector& w, og::SimpleSetup& setup) {
    setup.clear();
    auto planner(std::make_shared<og::RRTstar>(setup.getSpaceInformation()));
    planner->setRange(1.0);
    setup.setPlanner(planner);

    // Use p-norm objective
    auto obj = std::make_shared<CustomWeightedObjective>(setup.getSpaceInformation(), w, P_NORM);
    setup.setOptimizationObjective(obj);
    setup.solve(900.0);
    return evaluatePathCosts(setup.getSolutionPath());
}

// ==========================================
// 3. Gurobi LP Solver — Transformed Lambda-Space
// ==========================================
//
// The key insight: in lambda-space with transformed objectives g^p(f),
// the problem has the SAME structure as the original MRPS LP.
//
// u_tilde(lambda) = min_s  sum_i lambda_i * f_i(s)^p   is concave in lambda
// (pointwise min of affine functions).
//
// The LP finds the lambda* in the convex hull of corner lambdas that
// maximizes the regret bound, then maps back to w-space via (phi^p)^{-1}.

RegretResult solveMaxRegretLP(const std::vector<SampledCost>& corners,
                              const std::vector<double>& global_max_costs) {
    int num_objectives = 3;
    int k = corners.size();

    try {
        GRBEnv env = GRBEnv(true);
        env.set("LogFile", "gurobi.log");
        env.start();
        env.set(GRB_IntParam_OutputFlag, 0);
        GRBModel model = GRBModel(env);

        // Variables: barycentric coords for convex hull in LAMBDA-space
        std::vector<GRBVar> mu(k);
        for (int i = 0; i < k; ++i)
            mu[i] = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "mu");

        // lambda variables (in transformed space)
        std::vector<GRBVar> lam(num_objectives);
        for (int j = 0; j < num_objectives; ++j)
            lam[j] = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "lam");

        GRBVar R = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "Regret");

        // lambda = convex combination of corner lambdas
        for (int j = 0; j < num_objectives; ++j) {
            GRBLinExpr expr = 0;
            for (int i = 0; i < k; ++i)
                expr += mu[i] * corners[i].lambda[j];
            model.addConstr(lam[j] == expr);
        }

        // sum mu = 1
        GRBLinExpr sum_mu = 0;
        for (int i = 0; i < k; ++i) sum_mu += mu[i];
        model.addConstr(sum_mu == 1.0);

        // Lower bound P(lambda): linear interpolant of u_tilde at corner lambdas
        // u_tilde(lambda_i) = sum_j lambda_i_j * gp_f_i_j / normalization
        std::vector<double> u_corners(k);
        for (int i = 0; i < k; ++i) {
            double dot = 0.0;
            for (int j = 0; j < num_objectives; ++j) {
                // Normalized transformed objective
                double gp_norm = corners[i].gp_f[j] / std::pow(global_max_costs[j], P_NORM);
                dot += corners[i].lambda[j] * gp_norm;
            }
            u_corners[i] = dot;
        }

        GRBLinExpr LB = 0;
        for (int i = 0; i < k; ++i) LB += mu[i] * u_corners[i];

        // Regret constraints: R <= lambda . gp_f_i / norm - P(lambda)  for each corner i
        for (int i = 0; i < k; ++i) {
            GRBLinExpr lam_dot_gf = 0.0;
            for (int j = 0; j < num_objectives; ++j) {
                double gp_norm = corners[i].gp_f[j] / std::pow(global_max_costs[j], P_NORM);
                lam_dot_gf += lam[j] * gp_norm;
            }
            model.addConstr(R <= lam_dot_gf - LB);
        }

        model.setObjective(GRBLinExpr(R), GRB_MAXIMIZE);
        model.optimize();

        // Extract lambda* and map back to w-space
        Vector res_lambda(num_objectives);
        for (int j = 0; j < num_objectives; ++j)
            res_lambda[j] = lam[j].get(GRB_DoubleAttr_X);

        Vector res_w = phi_p_inverse(res_lambda, P_NORM);

        return {R.get(GRB_DoubleAttr_X), res_lambda, res_w};

    } catch (GRBException e) {
        std::cerr << "Gurobi Error: " << e.getMessage() << std::endl;
        return {-1.0, {}, {}};
    }
}

// ==========================================
// 4. Scenarios (unchanged)
// ==========================================

void configureEnvironment(int scenario_id) {
    global_env = std::make_shared<Environment>();
    global_env->addObstacle(std::make_shared<BoundaryObstacle>(0.0, 40.0));

    switch (scenario_id) {
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
        default:
            std::cout << "Unknown Scenario. Defaulting to Empty." << std::endl;
            break;
    }
}

// ==========================================
// 5. Main Loop
// ==========================================

int main(int argc, char* argv[]) {
    // Parse arguments: scenario [p_norm]
    int scenario = 1;
    if (argc > 1) scenario = std::stoi(argv[1]);
    if (argc > 2) P_NORM = std::stod(argv[2]);

    std::cout << "=== p-Norm MRPS ===" << std::endl;
    std::cout << "p = " << P_NORM << std::endl;

    configureEnvironment(scenario);

    std::string filename = "RPS_pnorm_p" + std::to_string((int)P_NORM)
                         + "_scenario_" + std::to_string(scenario) + ".txt";
    logFile.open(filename);
    if (!logFile.is_open()) {
        std::cerr << "Error: Could not open log file " << filename << std::endl;
        return 1;
    }
    std::cout << "Saving data to: " << filename << std::endl;
    logFile << "Iteration,w1,w2,w3,lam1,lam2,lam3,f1,f2,f3,gp1,gp2,gp3,MaxRegret,is_duplicate\n";

    // Environment setup
    global_env = std::make_shared<Environment>();
    global_env->addObstacle(std::make_shared<CircularObstacle>(11.0, 13.0, 3.0));

    double boundary_min = 0.0, boundary_max = 40.0;
    global_env->addObstacle(std::make_shared<BoundaryObstacle>(boundary_min, boundary_max));

    auto stateSpace = std::make_shared<ob::RealVectorStateSpace>(2);
    stateSpace->setBounds(boundary_min, boundary_max);
    og::SimpleSetup setup(stateSpace);
    setup.setStateValidityChecker(isStateValid);
    ob::ScopedState<> start(stateSpace);
    start[0] = 1.0; start[1] = 15.0;
    ob::ScopedState<> goal(stateSpace);
    goal[0] = 21.0; goal[1] = 15.0;
    setup.setStartAndGoalStates(start, goal);

    // Initialize database
    std::vector<SampledCost> database;
    int num_obj = 3;

    // Corner weights (canonical basis in w-space)
    // Note: for p>1, phi^p maps these to the same canonical basis in lambda-space
    std::vector<Vector> corner_weights = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}
    };

    std::vector<Vector> corner_case;
    if (scenario == 1) {
        corner_case = {
            {20.1494, 101.356, 10.0747},
            {31.6239, 1.7663, 15.8119},
            {25.4269, 143.952, 2.21921}
        };
    } else if (scenario == 2) {
        corner_case = {
            {20.1936, 49.5278, 10.0968},
            {43.9008, 2.97585, 21.9504},
            {25.3565, 101.833, 2.2156}
        };
    } else {
        corner_case = {
            {20.0154, 0.959798, 10.0077},
            {20.3631, 0.959532, 10.1816},
            {24.5904, 2.81677, 2.21509}
        };
    }

    std::cout << "--- Initializing Corners ---" << std::endl;
    std::vector<double> global_max_costs(3, 1.0);

    for (int i = 0; i < 3; ++i) {
        Vector f = corner_case[i];
        Vector lam = phi_p(corner_weights[i], P_NORM);
        Vector gpf = g_p(f, P_NORM);

        database.push_back({i, corner_weights[i], f, lam, gpf});

        for (int k = 0; k < 3; ++k) {
            if (f[k] > global_max_costs[k]) global_max_costs[k] = f[k];
        }

        logFile << (i - num_obj) << ","
                << corner_weights[i][0] << "," << corner_weights[i][1] << "," << corner_weights[i][2] << ","
                << lam[0] << "," << lam[1] << "," << lam[2] << ","
                << f[0] << "," << f[1] << "," << f[2] << ","
                << gpf[0] << "," << gpf[1] << "," << gpf[2] << ","
                << 0.0 << "," << 0 << "\n";
    }

    std::list<Neighborhood> neighborhoods;

    // Create initial neighborhood
    Neighborhood initial_neighborhood;
    initial_neighborhood.id_d = 0;
    initial_neighborhood.id_r = 1;
    initial_neighborhood.id_t = 2;
    std::vector<SampledCost> initial_corners = {
        database[0], database[1], database[2]
    };

    RegretResult initial_regret = solveMaxRegretLP(initial_corners, global_max_costs);
    initial_neighborhood.max_regret = initial_regret.max_regret;
    initial_neighborhood.candidate_w = initial_regret.worst_w;
    initial_neighborhood.candidate_lambda = initial_regret.worst_lambda;
    initial_neighborhood.is_duplicate = false;
    neighborhoods.push_back(initial_neighborhood);
    std::cout << "Initial Max Regret: " << initial_neighborhood.max_regret << std::endl;
    printVector("Initial candidate w", initial_neighborhood.candidate_w);
    printVector("Initial candidate lambda", initial_neighborhood.candidate_lambda);

    // ----- MAIN LOOP -----
    int Budget_K = 32;
    double threshold_duplicate = 0.001;
    int duplicate_count = 0;

    for (int k = 0; k < Budget_K; ++k) {
        std::cout << "\n--- Iteration " << k << " ---" << std::endl;

        // Find neighborhood with largest regret
        double max_global_regret = -1.0;
        auto best_it = neighborhoods.begin();
        for (auto it = neighborhoods.begin(); it != neighborhoods.end(); ++it) {
            if (it->max_regret > max_global_regret && !it->is_duplicate) {
                max_global_regret = it->max_regret;
                best_it = it;
            }
        }

        std::cout << "Selected Neighborhood with Max Regret: " << max_global_regret << std::endl;
        std::cout << "Triangle Corners IDs: "
                  << best_it->id_d << ", " << best_it->id_r << ", " << best_it->id_t << std::endl;

        if (max_global_regret < threshold_duplicate) {
            std::cout << "Converged." << std::endl;
            logFile << k << "," << best_it->candidate_w[0] << "," << best_it->candidate_w[1]
                    << "," << best_it->candidate_w[2] << ","
                    << best_it->candidate_lambda[0] << "," << best_it->candidate_lambda[1]
                    << "," << best_it->candidate_lambda[2] << ","
                    << ",,,,,,," << max_global_regret << ",CONVERGED\n";
            logFile.flush();
            break;
        }

        // Solve planning problem for candidate weight
        Vector new_w = best_it->candidate_w;
        Vector new_f = solvePlanningProblem(new_w, setup);
        Vector new_lambda = phi_p(new_w, P_NORM);
        Vector new_gpf = g_p(new_f, P_NORM);
        int new_id = database.size();

        // Check for duplicate
        bool is_duplicate = false;
        for (size_t i = 0; i < database.size(); ++i) {
            double dist = 0.0;
            for (int j = 0; j < 3; ++j)
                dist += std::pow(new_w[j] - database[i].w[j], 2);
            if (std::sqrt(dist) < threshold_duplicate) {
                is_duplicate = true;
                break;
            }
        }

        int d = best_it->id_d, r = best_it->id_r, t = best_it->id_t;

        if (is_duplicate) {
            std::cout << "Duplicate weight found. Discarding neighborhood." << std::endl;
            logFile << k << ",DUPLICATE," << new_w[0] << "," << new_w[1] << "," << new_w[2] << "\n";
            logFile.flush();
            neighborhoods.erase(best_it);
            ++duplicate_count;
            continue;
        }

        printVector("New weight (w-space)", new_w);
        printVector("New weight (lambda-space)", new_lambda);
        printVector("New cost f(s)", new_f);
        printVector("Transformed g^p(f)", new_gpf);

        database.push_back({new_id, new_w, new_f, new_lambda, new_gpf});

        logFile << k << ","
                << new_w[0] << "," << new_w[1] << "," << new_w[2] << ","
                << new_lambda[0] << "," << new_lambda[1] << "," << new_lambda[2] << ","
                << new_f[0] << "," << new_f[1] << "," << new_f[2] << ","
                << new_gpf[0] << "," << new_gpf[1] << "," << new_gpf[2] << ","
                << max_global_regret << "," << is_duplicate << "\n";
        logFile.flush();

        // Remove used neighborhood, create 3 new ones
        neighborhoods.erase(best_it);

        int sets[3][3] = {
            {d, r, new_id},
            {d, new_id, t},
            {new_id, r, t}
        };

        for (int i = 0; i < num_obj; ++i) {
            Neighborhood n;
            n.id_d = sets[i][0];
            n.id_r = sets[i][1];
            n.id_t = sets[i][2];
            std::vector<SampledCost> corners = {
                database[n.id_d], database[n.id_r], database[n.id_t]
            };

            RegretResult regret = solveMaxRegretLP(corners, global_max_costs);
            n.max_regret = regret.max_regret;
            n.candidate_w = regret.worst_w;
            n.candidate_lambda = regret.worst_lambda;
            n.is_duplicate = false;
            neighborhoods.push_back(n);

            std::cout << "New Neighborhood IDs: "
                      << n.id_d << ", " << n.id_r << ", " << n.id_t
                      << " | Max Regret: " << n.max_regret << std::endl;
        }
    }

    // Final output
    std::cout << "\nFinal Database (p=" << P_NORM << "):" << std::endl;
    for (const auto& s : database) {
        std::cout << "ID:" << s.id << " w:[";
        for (auto w : s.w) std::cout << w << " ";
        std::cout << "] lam:[";
        for (auto l : s.lambda) std::cout << l << " ";
        std::cout << "] f:[";
        for (auto f : s.f) std::cout << f << " ";
        std::cout << "] g^p(f):[";
        for (auto g : s.gp_f) std::cout << g << " ";
        std::cout << "]" << std::endl;
    }

    logFile.close();
    std::cout << "p-Norm RPS Finished. p=" << P_NORM
              << " | Total Samples: " << database.size()
              << " | Duplicates: " << duplicate_count << std::endl;

    // Save database
    std::string db_filename = filename;
    size_t dotPos = db_filename.find_last_of('.');
    if (dotPos != std::string::npos) db_filename.insert(dotPos, "_database");
    else db_filename += "_database.csv";
    saveDatabaseToCSV(db_filename, database);

    return 0;
}
