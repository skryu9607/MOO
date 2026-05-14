/*
 * Regret-Based Pareto Front Sampling (C++ Implementation) - BATCH VERSION v2
 * 
 * Key changes from v1:
 *   1. Extracts ALL feasible solutions from OMPL's ProblemDefinition::getSolutions()
 *      RRT* internally stores every improving solution it finds. We harvest all of them.
 *   2. Extracts near-goal paths from the RRT* tree via PlannerData.
 *      The tree contains vertices near the goal that were explored but not selected
 *      as the best path — these are topologically diverse free Pareto candidates.
 *   3. Softened speed transition (sigmoid).
 *   4. Non-dominance filtering on all harvested candidates.
 *
 * Dependencies: OMPL, Gurobi C++, Boost
 *
 * Compilation:
g++ -m64 -g plain.cpp -o plain \
-I/opt/gurobi1300/linux64/include \
-L/opt/gurobi1300/linux64/lib \
-I/home/seung/ompl/src \
-L/home/seung/ompl/build/lib \
-I/usr/include/eigen3 \
-lgurobi_c++ -lgurobi130 -lompl -lpthread \
-fopenmp
 *
 * Usage: ./RPS_Batch_v2 [scenario] [batch_size] [steepness] [planning_time] [goal_radius]
 *   scenario      : 0=empty, 1=one circle, 2=two circles  (default: 1)
 *   batch_size    : parallel batch size                     (default: 4)
 *   steepness     : sigmoid sharpness for speed transition  (default: 2.0)
 *   planning_time : seconds per planning query              (default: 600.0)
 *   goal_radius   : radius for near-goal tree harvesting    (default: 2.0)
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
#include <set>
#include <omp.h>
#include <mutex>

// Gurobi
#include "gurobi_c++.h"

// OMPL Core
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/PlannerData.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/base/OptimizationObjective.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>

namespace ob = ompl::base;
namespace og = ompl::geometric;

// ==========================================
// 1. Core Data Structures
// ==========================================

using Vector = std::vector<double>;

struct State { double x, y; };

struct SampledCost {
    int    id;
    Vector w;              // Weight vector used for planning
    Vector f;              // Cost vector [Dist, Risk, Time]
    bool   is_harvested;   // true = came from an intermediate / tree path
};

struct Neighborhood {
    int    id_d, id_r, id_t;
    double max_regret;
    Vector new_w;
    bool   is_duplicate;
};

struct RegretResult {
    double max_regret;
    Vector worst_w;
};

// Everything returned from a single planning query
struct PlanningResult {
    Vector              final_f;          // Best solution's cost vector
    std::vector<Vector> all_solution_fs;  // ALL solutions from pdef->getSolutions()
    std::vector<Vector> tree_path_fs;     // Paths reconstructed from PlannerData tree
};

// ==========================================
// 2. Obstacle & Environment System
// ==========================================

class Obstacle {
public:
    virtual ~Obstacle() = default;
    virtual bool   CheckCollision(const State& s) const = 0;
    virtual double getClearance(const State& s)   const = 0;
};

class CircularObstacle : public Obstacle {
    double cx, cy, radius;
public:
    CircularObstacle(double x, double y, double r) : cx(x), cy(y), radius(r) {}
    bool CheckCollision(const State& s) const override { return getClearance(s) <= 0.0; }
    double getClearance(const State& s) const override {
        return std::sqrt((s.x-cx)*(s.x-cx) + (s.y-cy)*(s.y-cy)) - radius;
    }
};

class BoundaryObstacle : public Obstacle {
    double min_val, max_val;
public:
    BoundaryObstacle(double lo, double hi) : min_val(lo), max_val(hi) {}
    bool CheckCollision(const State& s) const override {
        return (s.x < min_val || s.x > max_val || s.y < min_val || s.y > max_val);
    }
    double getClearance(const State& s) const override {
        return std::min({s.x - min_val, max_val - s.x, s.y - min_val, max_val - s.y});
    }
};

class Environment {
    std::vector<std::shared_ptr<Obstacle>> obstacles;
    double v_fast = 100.0, v_slow = 2.0, y_trans = 13.0, steep = 2.0;

public:
    void addObstacle(std::shared_ptr<Obstacle> o) { obstacles.push_back(o); }

    void setSpeedParams(double fast, double slow, double yt, double s) {
        v_fast = fast; v_slow = slow; y_trans = yt; steep = s;
    }

    bool isValid(const State& s) const {
        for (const auto& o : obstacles)
            if (o->getClearance(s) <= 0.1) return false;
        return true;
    }

    double getPointRisk(const State& s) const {
        double total = 0.0;
        for (const auto& o : obstacles) {
            double c = o->getClearance(s);
            if (c <= 0.1) return 1e6;
            total += 1.0 / (c * c + 1e-3);
        }
        return std::min(total, 1e6);
    }

    static double euclidean(const State& a, const State& b) {
        return std::sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y));
    }

    // Sigmoid speed: fast at low y, slow at high y
    double getSpeed(double y) const {
        return v_slow + (v_fast - v_slow) / (1.0 + std::exp(steep * (y - y_trans)));
    }

    // [distance, risk, time]
    std::vector<double> calculateSegmentCost(const State& from, const State& to) const {
        std::vector<double> cost(3, 0.0);
        cost[0] = euclidean(from, to);

        const int steps = 1001;
        double risk_sum = 0.0, time_acc = 0.0;
        State prev = from, cur, mid;

        for (int i = 1; i <= steps; ++i) {
            double t = (double)i / steps;
            cur.x = from.x + t * (to.x - from.x);
            cur.y = from.y + t * (to.y - from.y);

            mid.x = (cur.x + prev.x) * 0.5;
            mid.y = (cur.y + prev.y) * 0.5;
            risk_sum += getPointRisk(mid);

            time_acc += euclidean(cur, prev) / getSpeed(cur.y);
            prev = cur;
        }
        cost[1] = risk_sum * cost[0] / steps;
        cost[2] = time_acc;
        return cost;
    }
};

std::shared_ptr<Environment> global_env;
std::ofstream logFile;

// Global config (set from main, read by solver threads — immutable during run)
double G_PLANNING_TIME = 600.0;
double G_GOAL_RADIUS   = 2.0;
double G_BOUNDS_LO     = 0.0;
double G_BOUNDS_HI     = 30.0;
double G_START_X = 1.0, G_START_Y = 15.0;
double G_GOAL_X  = 21.0, G_GOAL_Y = 15.0;

// ==========================================
// 3. Non-dominance utilities
// ==========================================

bool dominates(const Vector& a, const Vector& b) {
    bool strict = false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] > b[i] + 1e-9) return false;
        if (a[i] < b[i] - 1e-9) strict = true;
    }
    return strict;
}

std::vector<Vector> filterNonDominated(
    const std::vector<Vector>& candidates,
    const std::vector<SampledCost>& database)
{
    std::vector<Vector> result;
    for (size_t ci = 0; ci < candidates.size(); ++ci) {
        const auto& c = candidates[ci];
        bool dominated = false;

        for (const auto& e : database)
            if (dominates(e.f, c)) { dominated = true; break; }
        if (dominated) continue;

        for (size_t oi = 0; oi < candidates.size(); ++oi) {
            if (oi == ci) continue;
            if (dominates(candidates[oi], c)) { dominated = true; break; }
        }
        if (dominated) continue;

        for (const auto& r : result)
            if (dominates(r, c)) { dominated = true; break; }

        if (!dominated) result.push_back(c);
    }
    return result;
}

// ==========================================
// 4. Environment Configuration
// ==========================================

void configureEnvironment(int id) {
    global_env = std::make_shared<Environment>();
    global_env->addObstacle(std::make_shared<BoundaryObstacle>(G_BOUNDS_LO, G_BOUNDS_HI));

    switch (id) {
        case 0: std::cout << "Scenario 0: Empty" << std::endl; break;
        case 1:
            std::cout << "Scenario 1: Single Circle" << std::endl;
            global_env->addObstacle(std::make_shared<CircularObstacle>(11.0, 13.0, 3.0));
            break;
        case 2:
            std::cout << "Scenario 2: Two Circles" << std::endl;
            global_env->addObstacle(std::make_shared<CircularObstacle>(11.0, 13.0, 3.0));
            global_env->addObstacle(std::make_shared<CircularObstacle>(11.0, 21.0, 2.0));
            break;
        default: std::cout << "Unknown scenario, using empty." << std::endl; break;
    }
}

// ==========================================
// 5. OMPL Objective & Validity
// ==========================================

class CustomWeightedObjective : public ob::OptimizationObjective {
    Vector weights;
public:
    CustomWeightedObjective(const ob::SpaceInformationPtr& si, const Vector& w)
        : ob::OptimizationObjective(si), weights(w) {
        description_ = "Weighted Dist/Risk/Time";
    }
    ob::Cost stateCost(const ob::State*) const override { return ob::Cost(0.0); }
    ob::Cost motionCost(const ob::State* s1, const ob::State* s2) const override {
        const auto* p1 = s1->as<ob::RealVectorStateSpace::StateType>();
        const auto* p2 = s2->as<ob::RealVectorStateSpace::StateType>();
        auto obj = global_env->calculateSegmentCost(
            {p1->values[0], p1->values[1]}, {p2->values[0], p2->values[1]});
        double s = 0.0;
        for (size_t i = 0; i < weights.size() && i < obj.size(); ++i)
            s += weights[i] * obj[i];
        return ob::Cost(s);
    }
};

bool isStateValid(const ob::State* state) {
    const auto* p = state->as<ob::RealVectorStateSpace::StateType>();
    return global_env->isValid({p->values[0], p->values[1]});
}

// Evaluate full cost vector from a list of states
Vector evaluatePathCosts(const std::vector<const ob::State*>& states) {
    Vector total(3, 0.0);
    for (size_t i = 0; i + 1 < states.size(); ++i) {
        const auto* p1 = states[i]->as<ob::RealVectorStateSpace::StateType>();
        const auto* p2 = states[i+1]->as<ob::RealVectorStateSpace::StateType>();
        auto seg = global_env->calculateSegmentCost(
            {p1->values[0], p1->values[1]}, {p2->values[0], p2->values[1]});
        for (int k = 0; k < 3; ++k) total[k] += seg[k];
    }
    return total;
}

Vector evaluatePathCosts(og::PathGeometric& path) {
    const auto& s = path.getStates();
    std::vector<const ob::State*> sv(s.begin(), s.end());
    return evaluatePathCosts(sv);
}

// ==========================================
// 6. Tree Path Harvesting from PlannerData
// ==========================================

// Trace path from vertex back to start through parent edges.
std::vector<unsigned int> traceToRoot(const ob::PlannerData& pd, unsigned int idx) {
    std::vector<unsigned int> path;
    std::set<unsigned int> visited;

    unsigned int cur = idx;
    while (true) {
        if (visited.count(cur)) return {};  // cycle
        visited.insert(cur);
        path.push_back(cur);

        if (pd.isStartVertex(cur)) break;

        std::vector<unsigned int> parents;
        pd.getIncomingEdges(cur, parents);
        if (parents.empty()) return {};  // disconnected

        cur = parents[0];  // RRT tree: one parent per node
    }

    std::reverse(path.begin(), path.end());
    return path;
}

// Find all vertices within `radius` of the goal, trace each to start,
// evaluate cost vector including last-mile to goal.
std::vector<Vector> harvestTreePaths(og::SimpleSetup& setup, double radius) {
    std::vector<Vector> results;

    ob::PlannerData pd(setup.getSpaceInformation());
    setup.getPlannerData(pd);
    if (pd.numVertices() == 0) return results;

    State goal_st = {G_GOAL_X, G_GOAL_Y};

    for (unsigned int i = 0; i < pd.numVertices(); ++i) {
        const auto* p = pd.getVertex(i).getState()->as<ob::RealVectorStateSpace::StateType>();
        State vs = {p->values[0], p->values[1]};
        double d2g = Environment::euclidean(vs, goal_st);
        if (d2g > radius) continue;

        auto idx_path = traceToRoot(pd, i);
        if (idx_path.size() < 2) continue;

        // Build state list
        std::vector<const ob::State*> sp;
        sp.reserve(idx_path.size());
        for (auto vi : idx_path)
            sp.push_back(pd.getVertex(vi).getState());

        Vector f = evaluatePathCosts(sp);

        // Add last-mile cost from near-goal vertex to actual goal
        if (d2g > 1e-6) {
            const auto* last = sp.back()->as<ob::RealVectorStateSpace::StateType>();
            auto seg = global_env->calculateSegmentCost(
                {last->values[0], last->values[1]}, goal_st);
            for (int k = 0; k < 3; ++k) f[k] += seg[k];
        }

        results.push_back(f);
    }
    return results;
}

// ==========================================
// 7. Main Solver — Extracts ALL Feasible Solutions
// ==========================================

PlanningResult solveBatchItemWithHarvesting(const Vector& w, int thread_id)
{
    // ---- Thread-local OMPL setup ----
    auto space = std::make_shared<ob::RealVectorStateSpace>(2);
    space->setBounds(G_BOUNDS_LO, G_BOUNDS_HI);
    og::SimpleSetup setup(space);
    setup.setStateValidityChecker(isStateValid);

    ob::ScopedState<> start(space), goal(space);
    start[0] = G_START_X; start[1] = G_START_Y;
    goal[0]  = G_GOAL_X;  goal[1]  = G_GOAL_Y;
    setup.setStartAndGoalStates(start, goal);

    auto obj = std::make_shared<CustomWeightedObjective>(setup.getSpaceInformation(), w);
    setup.setOptimizationObjective(obj);

    auto planner = std::make_shared<og::RRTstar>(setup.getSpaceInformation());
    planner->setRange(0.5);
    setup.setPlanner(planner);

    // ==================================================================
    // Solve for the full planning time.
    // OMPL's RRT* internally calls pdef->addSolutionPath() every time
    // it finds a better path. All solutions are retained in memory.
    // ==================================================================
    setup.solve(G_PLANNING_TIME);

    PlanningResult result;

    // ==================================================================
    // SOURCE 1: pdef->getSolutions()
    //
    // Returns ALL solution paths that RRT* found during the entire
    // planning session, sorted by cost (best first).
    //
    // Typical count: 10-100+ solutions depending on planning time.
    // Each represents a distinct snapshot of the improving solution.
    // Their [dist, risk, time] vectors can differ wildly — a path
    // that's suboptimal under weight w might be non-dominated in
    // objective space.
    // ==================================================================
    auto pdef = setup.getProblemDefinition();
    const auto& solutions = pdef->getSolutions();

    std::cout << "  [T" << thread_id << "] pdef->getSolutions(): "
              << solutions.size() << " paths" << std::endl;

    for (size_t i = 0; i < solutions.size(); ++i) {
        auto* gp = solutions[i].path_->as<og::PathGeometric>();
        if (!gp) continue;
        result.all_solution_fs.push_back(evaluatePathCosts(*gp));
    }

    if (!result.all_solution_fs.empty()) {
        result.final_f = result.all_solution_fs[0];  // best solution
    } else {
        result.final_f = {1e6, 1e6, 1e6};
        std::cout << "  [T" << thread_id << "] WARNING: No solution found." << std::endl;
        return result;
    }

    // ==================================================================
    // SOURCE 2: PlannerData tree — near-goal vertex paths
    //
    // RRT* builds a tree with potentially thousands of vertices.
    // Some vertices land near the goal but through different routes
    // (different topologies — above/below obstacles, highway vs
    // surface). These are paths the planner *explored* but didn't
    // select as optimal under this weight w.
    //
    // We trace each near-goal vertex back to start and evaluate its
    // full cost vector. Many of these will be dominated, but the
    // non-dominated ones are free Pareto samples.
    // ==================================================================
    result.tree_path_fs = harvestTreePaths(setup, G_GOAL_RADIUS);

    std::cout << "  [T" << thread_id << "] PlannerData tree: "
              << result.tree_path_fs.size() << " near-goal paths" << std::endl;

    return result;
}

// ==========================================
// 8. Gurobi LP Solver (Max Regret)
// ==========================================

RegretResult solveMaxRegretLP(const std::vector<SampledCost>& corners,
                               const std::vector<double>& gmax)
{
    int M = 3, K = (int)corners.size();
    try {
        GRBEnv env(true);
        env.set("LogFile", "");
        env.start();
        env.set(GRB_IntParam_OutputFlag, 0);
        GRBModel model(env);

        std::vector<GRBVar> lam(K);
        for (int i = 0; i < K; ++i)
            lam[i] = model.addVar(0, 1, 0, GRB_CONTINUOUS);

        std::vector<GRBVar> w(M);
        for (int j = 0; j < M; ++j)
            w[j] = model.addVar(0, 1, 0, GRB_CONTINUOUS);

        GRBVar R = model.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS, "R");

        for (int j = 0; j < M; ++j) {
            GRBLinExpr e = 0;
            for (int i = 0; i < K; ++i) e += lam[i] * corners[i].w[j];
            model.addConstr(w[j] == e);
        }

        GRBLinExpr sl = 0;
        for (int i = 0; i < K; ++i) sl += lam[i];
        model.addConstr(sl == 1.0);

        std::vector<double> u(K);
        for (int i = 0; i < K; ++i) {
            double d = 0;
            for (int j = 0; j < M; ++j) d += corners[i].w[j] * corners[i].f[j] / gmax[j];
            u[i] = d;
        }

        GRBLinExpr LB = 0;
        for (int i = 0; i < K; ++i) LB += lam[i] * u[i];

        for (int i = 0; i < K; ++i) {
            GRBLinExpr wf = 0;
            for (int j = 0; j < M; ++j) wf += w[j] * corners[i].f[j] / gmax[j];
            model.addConstr(R <= wf - LB);
        }

        model.setObjective(GRBLinExpr(R), GRB_MAXIMIZE);
        model.optimize();

        Vector rw;
        for (int j = 0; j < M; ++j) rw.push_back(w[j].get(GRB_DoubleAttr_X));
        return {R.get(GRB_DoubleAttr_X), rw};

    } catch (GRBException& e) {
        std::cerr << "Gurobi Error: " << e.getMessage() << std::endl;
        return {-1.0, {}};
    }
}

// ==========================================
// 9. Utility
// ==========================================

void saveDatabaseToCSV(const std::string& fn, const std::vector<SampledCost>& db) {
    std::ofstream out(fn);
    if (!out) { std::cerr << "Cannot open " << fn << std::endl; return; }
    out << "ID,W_Dist,W_Risk,W_Time,Cost_Dist,Cost_Risk,Cost_Time,Is_Harvested\n";
    for (const auto& s : db) {
        out << s.id;
        for (auto v : s.w) out << "," << v;
        for (auto v : s.f) out << "," << v;
        out << "," << (s.is_harvested ? 1 : 0) << "\n";
    }
    out.close();
    std::cout << "Database saved: " << fn << std::endl;
}

// ==========================================
// 10. Main
// ==========================================

int main(int argc, char* argv[]) {
    int    scenario   = 1;
    int    batch_size = 4;
    double steepness  = 0.5;
    G_PLANNING_TIME   = 600.0;
    G_GOAL_RADIUS     = 0.5;

    if (argc > 1) scenario        = std::stoi(argv[1]);
    if (argc > 2) batch_size      = std::stoi(argv[2]);
    if (argc > 3) steepness       = std::stod(argv[3]);
    if (argc > 4) G_PLANNING_TIME = std::stod(argv[4]);
    if (argc > 5) G_GOAL_RADIUS   = std::stod(argv[5]);

    std::cout << "=== RPS v2 ===" << std::endl;
    std::cout << "Scenario:      " << scenario << std::endl;
    std::cout << "Batch size:    " << batch_size << std::endl;
    std::cout << "Steepness:     " << steepness << std::endl;
    std::cout << "Planning time: " << G_PLANNING_TIME << "s" << std::endl;
    std::cout << "Goal radius:   " << G_GOAL_RADIUS << std::endl;

    configureEnvironment(scenario);
    global_env->setSpeedParams(100.0, 2.0, 13.0, steepness);

    std::string filename = "RPS_v2_sc" + std::to_string(scenario)
                         + "_b" + std::to_string(batch_size)
                         + "_s" + std::to_string((int)steepness) + ".csv";
    logFile.open(filename);
    if (!logFile) { std::cerr << "Cannot open " << filename << std::endl; return 1; }
    logFile << "Iteration,w1,w2,w3,f1,f2,f3,MaxRegret,Source\n";

    // ------------------------------------------
    // Corner initialization
    // ------------------------------------------
    // NOTE: With softened speed you should re-solve these fresh.
    // Replace hardcoded values with:
    //   for (int i = 0; i < 3; ++i) {
    //       auto pr = solveBatchItemWithHarvesting(corner_weights[i], 0);
    //       corner_f[i] = pr.final_f;
    //       // harvest pr.all_solution_fs and pr.tree_path_fs too
    //   }
    std::vector<SampledCost> database;
    int harvested_total = 0;

    std::vector<Vector> corner_w = {{1,0,0}, {0,1,0}, {0,0,1}};
    std::vector<Vector> corner_f;

    if (scenario == 1)
        corner_f = {{20.1494,101.356,10.0747},{31.6239,1.7663,15.8119},{25.4269,143.952,2.21921}};
    else if (scenario == 2)
        corner_f = {{20.1936,49.5278,10.0968},{43.9008,2.97585,21.9504},{25.3565,101.833,2.2156}};
    else
        corner_f = {{20.0154,0.959798,10.0077},{20.3631,0.959532,10.1816},{24.5904,2.81677,2.21509}};

    std::vector<double> gmax(3, 1.0);
    for (int i = 0; i < 3; ++i) {
        database.push_back({i, corner_w[i], corner_f[i], false});
        for (int k = 0; k < 3; ++k)
            if (corner_f[i][k] > gmax[k]) gmax[k] = corner_f[i][k];
        logFile << -1 << "," << corner_w[i][0] << "," << corner_w[i][1] << ","
                << corner_w[i][2] << "," << corner_f[i][0] << "," << corner_f[i][1]
                << "," << corner_f[i][2] << ",0,corner\n";
    }

    // Initial neighborhood
    std::list<Neighborhood> neighborhoods;
    {
        std::vector<SampledCost> c = {database[0], database[1], database[2]};
        auto rr = solveMaxRegretLP(c, gmax);
        neighborhoods.push_back({0, 1, 2, rr.max_regret, rr.worst_w, false});
        std::cout << "Initial max regret: " << rr.max_regret << std::endl;
    }

    // ------------------------------------------
    // Main loop
    // ------------------------------------------
    int budget = 32;

    for (int iter = 0; iter < budget; ++iter) {
        std::cout << "\n=== Iteration " << iter << " ===" << std::endl;

        neighborhoods.sort([](const Neighborhood& a, const Neighborhood& b) {
            return a.max_regret > b.max_regret;
        });

        std::vector<Neighborhood> batch;
        while (!neighborhoods.empty() && (int)batch.size() < batch_size) {
            if (neighborhoods.front().max_regret < 0.0005) break;
            batch.push_back(neighborhoods.front());
            neighborhoods.pop_front();
        }
        if (batch.empty()) {
            std::cout << "Converged." << std::endl;
            break;
        }

        std::cout << "Batch: " << batch.size()
                  << "  top regret: " << batch[0].max_regret << std::endl;

        // Parallel planning
        std::vector<PlanningResult> results(batch.size());

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < (int)batch.size(); ++i) {
            int tid = omp_get_thread_num();
            results[i] = solveBatchItemWithHarvesting(batch[i].new_w, tid);
        }

        // Sequential integration
        for (int i = 0; i < (int)batch.size(); ++i) {
            const auto& pr   = results[i];
            const auto& task = batch[i];
            int new_id = (int)database.size();

            // ---- Add final (best) solution ----
            database.push_back({new_id, task.new_w, pr.final_f, false});
            for (int j = 0; j < 3; ++j)
                if (pr.final_f[j] > gmax[j]) gmax[j] = pr.final_f[j];

            logFile << iter << "," << task.new_w[0] << "," << task.new_w[1] << ","
                    << task.new_w[2] << "," << pr.final_f[0] << "," << pr.final_f[1]
                    << "," << pr.final_f[2] << "," << task.max_regret << ",planned\n";

            // ---- Harvest ALL intermediate OMPL solutions ----
            // pdef->getSolutions()[0] = best (already added), rest are older/suboptimal
            if (pr.all_solution_fs.size() > 1) {
                std::vector<Vector> intermediates(
                    pr.all_solution_fs.begin() + 1, pr.all_solution_fs.end());
                auto keepers = filterNonDominated(intermediates, database);

                for (const auto& hf : keepers) {
                    int hid = (int)database.size();
                    database.push_back({hid, task.new_w, hf, true});
                    harvested_total++;
                    for (int j = 0; j < 3; ++j)
                        if (hf[j] > gmax[j]) gmax[j] = hf[j];
                    logFile << iter << "," << task.new_w[0] << "," << task.new_w[1]
                            << "," << task.new_w[2] << "," << hf[0] << "," << hf[1]
                            << "," << hf[2] << ",0,ompl_solution\n";
                }
                std::cout << "  OMPL solutions: " << intermediates.size()
                          << " candidates -> " << keepers.size() << " kept" << std::endl;
            }

            // ---- Harvest near-goal tree paths ----
            if (!pr.tree_path_fs.empty()) {
                auto keepers = filterNonDominated(pr.tree_path_fs, database);

                for (const auto& tf : keepers) {
                    int tid2 = (int)database.size();
                    database.push_back({tid2, task.new_w, tf, true});
                    harvested_total++;
                    for (int j = 0; j < 3; ++j)
                        if (tf[j] > gmax[j]) gmax[j] = tf[j];
                    logFile << iter << "," << task.new_w[0] << "," << task.new_w[1]
                            << "," << task.new_w[2] << "," << tf[0] << "," << tf[1]
                            << "," << tf[2] << ",0,tree_path\n";
                }
                std::cout << "  Tree paths: " << pr.tree_path_fs.size()
                          << " candidates -> " << keepers.size() << " kept" << std::endl;
            }

            // ---- Subdivide (only final solution participates) ----
            int d = task.id_d, r = task.id_r, t = task.id_t;
            int sets[3][3] = {{d, r, new_id}, {d, new_id, t}, {new_id, r, t}};

            for (int j = 0; j < 3; ++j) {
                Neighborhood child;
                child.id_d = sets[j][0];
                child.id_r = sets[j][1];
                child.id_t = sets[j][2];
                child.is_duplicate = false;

                std::vector<SampledCost> corners = {
                    database[child.id_d], database[child.id_r], database[child.id_t]
                };
                auto rr = solveMaxRegretLP(corners, gmax);
                child.max_regret = rr.max_regret;
                child.new_w = rr.worst_w;

                if (child.max_regret > 0.0005)
                    neighborhoods.push_back(child);
            }
        }
        logFile.flush();
    }

    // ------------------------------------------
    // Summary
    // ------------------------------------------
    std::cout << "\n=============================" << std::endl;
    std::cout << "RPS v2 Complete" << std::endl;
    std::cout << "Total in database:  " << database.size() << std::endl;
    std::cout << "  Planned (RPS):    " << database.size() - harvested_total << std::endl;
    std::cout << "  Harvested (free): " << harvested_total << std::endl;
    std::cout << "=============================" << std::endl;

    logFile.close();

    std::string db_fn = filename;
    size_t dot = db_fn.find_last_of('.');
    if (dot != std::string::npos) db_fn.insert(dot, "_database");
    else db_fn += "_database.csv";
    saveDatabaseToCSV(db_fn, database);

    return 0;
}
