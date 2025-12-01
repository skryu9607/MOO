/*
 * MRPS Hybrid Implementation
 * 1. Regret LP Solver: Gurobi Optimizer
 * 2. Robot Planner: OMPL (RRT*)
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <queue>
#include <memory>

// --- Gurobi Header ---
#include "gurobi_c++.h"

// --- OMPL Headers ---
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/base/objectives/StateCostIntegralObjective.h>
#include <ompl/base/objectives/OptimizationObjective.h>

using namespace std;
namespace ob = ompl::base;
namespace og = ompl::geometric;

// ---------------------------------------------------------
// 1. Basic Structures
// ---------------------------------------------------------

using Vector = std::vector<double>;

double dot(const Vector& a, const Vector& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    };
    return sum;
}

void printVec(const Vector& v) {
    std::cout << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << (i > 0 ? ", " : "") << std::fixed << std::setprecision(4) << v[i];
    };
    std::cout << "]";
}

// In this case, this is "f".
struct PlannerSolution {
    Vector objectives; // [Cost1, Cost2]
};

class RobotPlanner {
public:
    virtual PlannerSolution solve(const Vector& weights) = 0;
    virtual int getNumObjectives() const = 0;
    virtual ~RobotPlanner() = default;
};

// ---------------------------------------------------------
// 2. Gurobi Regret Solver (LP Solver)
// ---------------------------------------------------------
// MRPS 알고리즘 내부의 Regret Maximization 문제를 Gurobi로 해결
class GurobiRegretSolver{
    GRBEnv& env;
    public:
    struct Result{
        bool success;
        double objectiveValue;
        Vector solution;
    };
    GurobiRegretSolver(GRBEnv& env_ref) : env(env_ref) {}
    // Solve : Maximize c^T x s.t. Ax <= b, x >= 0
    Result solve(const std::vector<Vector>& A, const Vector& b, const Vector& c){
        GRBModel model(env);
        model.set(GRB_IntParam_OutputFlag,0);
        int num_vars = c.size();
        int num_constrs = b.size();

        std::vector<GRBVar> vars(num_vars);
        for (int i = 0; i < num_vars; ++i){
            vars[i] = model.addVar(0.0, GRB_infinity, 0.0, GRB_continou)

        }
    }
}
