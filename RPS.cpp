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

double getEuclideanDist(const StateXY& a, const StateXY& b) {
    return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2));
}

std::vector<double> calculateSegmentCost(const StateXY& s_from, const StateXY& s_to) {
    std::vector<double> cost(3, 0.0);

    // 1. Cost[0]: Distance
    cost[0] = getEuclideanDist(s_from, s_to);

    // 2. Cost[1]: Risk
    const double obstacle_cx = 11.0;
    const double obstacle_cy = 13.0;
    const double radius = 3.0;
    StateXY obstacle = {obstacle_cx, obstacle_cy};

    double risk = 0.0;
    int num_steps = 16;
    
    double sum_segment_risk = 0.0; 
    
    StateXY previous_intermediate_risk = s_from;
    StateXY intermediate_State_risk;
    StateXY CenterOfSegment;

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
        double inverse_risk_segment = (dist_to_obs - radius) * (dist_to_obs - radius);
        
        if (inverse_risk_segment < 0.0) { /
            inverse_risk_segment = 0.001;
        }
        if (inverse_risk_segment < 0.001) inverse_risk_segment = 0.001;

        previous_intermediate_risk = intermediate_State_risk;
        sum_segment_risk += inverse_risk_segment;
    }

    // Risk Formula
    // 10 * (1/sum) * length / steps
    if (sum_segment_risk < 1e-9) sum_segment_risk = 1e-9;
    risk = 1.0 * (1.0 / sum_segment_risk) * cost[0] / num_steps;
    cost[1] = risk;

    // 3. Cost[2]: Travel Time
    StateXY previous_intermediate_traveltime = s_from;
    StateXY intermediate_traveltime;
    double Time = 0.0; // 초기화 필수

    for (int i = 1; i <= num_steps; ++i) {
        double ratio = (double)i / num_steps;
        intermediate_traveltime.x = s_from.x + ratio * (s_to.x - s_from.x);
        intermediate_traveltime.y = s_from.y + ratio * (s_to.y - s_from.y);
        
        double speed;
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
void printVec(const Vector& v) {
    std::cout << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << (i > 0 ? ", " : "") << std::fixed << std::setprecision(4) << v[i];
    };
    std::cout << "]";
}
struct PlannerSolution {
    Vector objectives; // [Cost1, Cost2,...]
};
struct Sample{
    Vector weights; // Weight vector
    PlannerSolution s; // Robot trajectory/Cost
    double u_val;// Optimal scalar cost = w * f(s)
};
struct StateXY {
    double x, y;
};

struct Neighborhood{
    std::vector<int> vertex_indices;
    double max_regret;
    Vector w_candidate;
    bool operator<(const Neighborhood& other) const {
        return max_regret < other.max_regret;
    }
};
struct Sample {
    Vector weights;
    PlannerSolution s;
    double u_val;
};

// Neighborhood (Simplex)
struct Neighborhood {
    std::vector<int> vertex_indices;
    double max_regret;
    Vector w_candidate;

    bool operator<(const Neighborhood& other) const {
        return max_regret < other.max_regret;
    }
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
class RegretMaximizationLP{
    GRBEnv& env;
public:
    RegretMaximizationLP(GRBEnv& env_ref) : env(env_ref) {}
    // Solve : Maximize c^T x s.t. Ax <= b, x >= 0
    bool solve(Neighborhood& N, const std::vector<const Sample*>& vertices){
        GRBModel model(env);
        model.set(GRB_IntParam_OutputFlag,0);
        int n = vertices.size();
        int n_objs = vertices[0]->weights.size();

        std::vector<GRBVar> alpha(n);
        for (int i = 0; i < num_vars; ++i){
            alpha[i] = model.addVar(0.0, GRB_infinity, 0.0, GRB_continous,"alpha_"+std::to_string(i));
        }
        // eta : Linear approximation of upper bound
        GRBVar eta = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "eta");
        model.update();
        // --- 2. Objective (Equation 12).
        // maximize x - P(w) 
        // maximize eta - sum(alpha_i * u_i)
        GRBLinExpr lower_bound_approx = 0.0;

        for (int i = 0; i < num_vars; ++i){
            lower_bound_approx += vertices[i]->u_val * alpha[i];
        }
        model.setObjective(eta - lower_bound_approx, GRB_MAXIMIZE);
        // --- 3. Constraints
        GRBLinExpr sum_alpha = 0.0;
        for (int i = 0; i < num_vars; ++i){
            sum_alpha += alpha[i];
        }
        model.addConstr(sum_alpha == 1.0, "SumAlpha");
        // Upper Bound Constriants, A * X >= 0.
        for (int k = 0;k < n ; ++k){
            Vector f_s_k = vertices[k]->s.objectives;
            GRBLinExpr w_dot_fsk = 0.0;
            for (int j = 0; j < n ; ++j){
                double val = dot(vertices[j]->weights, f_s_k);
                w_dot_fsk += val * alpha[j];
            }
            model.addConstr(eta <= w_dot_fsk, "UpperBound_"+std::to_string(k));
        }
        model.optimize();
        if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL){
             N.max_regret= model.get(GRB_DoubleAttr_ObjVal);
            if (N.max_regret < 0.0){
                N.max_regret = 0.0;
                std::cout<< "Max Regret is negative, set to zero." << std::endl;
            }
            int dim = vertices[0]->weights.size();
            N.w_candidate.assign(dim,0.0);
            for (int i = 0; i < n; ++i){
                double alpha_val = alpha[i].get(GRB_DoubleAttr_X);
                for (int d = 0; d < dim; ++d){
                    N.w_candidate[d] += alpha_val * vertices[i]->weights[d];
                };
            };
            return true;
        }else{
            return false;
        }
    }
};
/*
OMPL Planner Implementation
*/
// 사용자 정의 Cost Objective (Weighted Sum of 3 Objectives)
class CustomWeightedObjective : public ob::OptimizationObjective {
    Vector weights; // [w_dist, w_risk, w_time]

public:
    CustomWeightedObjective(const ob::SpaceInformationPtr& si, const Vector& w) 
        : ob::OptimizationObjective(si), weights(w) {
        // 비용 합산 방식: 누적 (Path Cost = Sum of segment costs)
        setCostThreshold(ob::Cost(std::numeric_limits<double>::infinity()));
    }

    // Motion Cost: 두 상태 사이의 비용 계산
    ob::Cost motionCost(const ob::State* s1, const ob::State* s2) const override {
        const auto* start = s1->as<ob::SE2StateSpace::StateType>();
        const auto* end = s2->as<ob::SE2StateSpace::StateType>();

        StateXY p1 = {start->getX(), start->getY()};
        StateXY p2 = {end->getX(), end->getY()};

        // 사용자 로직 호출
        Vector costs = calculateSegmentCost(p1, p2);

        // 스칼라화: w * c
        double weighted_cost = 0.0;
        for(size_t i=0; i<3; ++i) {
            weighted_cost += weights[i] * costs[i];
        }

        return ob::Cost(weighted_cost);
    }

    // State Cost는 0 (Motion Cost에 모두 포함됨)
    ob::Cost stateCost(const ob::State* s) const override {
        return ob::Cost(0.0);
    }
    
    // 비용 결합 방식: 덧셈
    ob::Cost combineCosts(ob::Cost c1, ob::Cost c2) const override {
        return ob::Cost(c1.value() + c2.value());
    }
};

// Validity Checker (장애물과 경계)
class MyValidityChecker : public ob::StateValidityChecker {
public:
    MyValidityChecker(const ob::SpaceInformationPtr& si) : ob::StateValidityChecker(si) {}
    bool isValid(const ob::State* state) const override {
        const auto* se2state = state->as<ob::SE2StateSpace::StateType>();
        double x = se2state->getX();
        double y = se2state->getY();
        
        // Map Bounds (0~20) - OMPL bounds 설정과 일치해야 함
        if (x < 0 || x > 25 || y < 0 || y > 25) return false;

        // 원형 장애물 (11, 13, r=3) 내부만 아니면 됨
        // (Risk Cost는 경계 근처에서 높아지지만, Validity는 충돌 여부만 판단)
        // 충돌 여부를 Risk 계산과 분리해도 되고, Risk가 무한대인 곳을 Invalid로 봐도 됨.
        // 여기서는 물리적 충돌(반경 2.5 이내)만 Invalid로 처리하고, 
        // 3.0 반경 근처는 높은 Cost로 처리하겠습니다.
        double dist_sq = pow(x - 11.0, 2) + pow(y - 13.0, 2);
        return dist_sq >= (3.0 * 3.0); 
    }
};

class OMPLRobotPlanner : public RobotPlanner {
    ob::StateSpacePtr space;
    ob::SpaceInformationPtr si;
    og::SimpleSetupPtr ss;

public:
    OMPLRobotPlanner() {
        space = std::make_shared<ob::SE2StateSpace>();
        // 사용자 좌표계가 (11,13)을 포함하므로 0~20으로 확장
        ob::RealVectorBounds bounds(2);
        bounds.setLow(0.0);
        bounds.setHigh(25.0); 
        space->as<ob::SE2StateSpace>()->setBounds(bounds);

        si = std::make_shared<ob::SpaceInformation>(space);
        si->setStateValidityChecker(std::make_shared<MyValidityChecker>(si));
        si->setup();

        ss = std::make_shared<og::SimpleSetup>(si);
        auto planner = std::make_shared<og::RRTstar>(si);
        planner->setRange(0.5); // Step size
        ss->setPlanner(planner);
    }

    int getNumObjectives() const override { return 3; } // 3개로 변경됨

    PlannerSolution solve(const Vector& weights) override {
        ss->clear();
        
        // Start & Goal (맵 크기에 맞춰 조정)
        ob::ScopedState<ob::SE2StateSpace> start(space);
        start->setX(1.0); start->setY(15.0); start->setYaw(0.0);
        
        ob::ScopedState<ob::SE2StateSpace> goal(space);
        goal->setX(21.0); goal->setY(15.0); goal->setYaw(0.0);
        
        ss->setStartAndGoalStates(start, goal);

        // 1. 가중치가 적용된 Objective 설정
        auto combinedObj = std::make_shared<CustomWeightedObjective>(si, weights);
        ss->setOptimizationObjective(combinedObj);
        ss->setup();

        // 2. Solve (Time budget 1.0s)
        ob::PlannerStatus solved = ss->solve(1.0);

        PlannerSolution sol;
        sol.objectives = std::vector<double>(3, 0.0);

        if (solved) {
            ss->simplifySolution();
            auto path = ss->getSolutionPath();
            const auto& states = path.getStates();

            // 3. 최적 경로에 대해 [Dist, Risk, Time] 각각 재계산
            for (size_t i = 0; i < states.size() - 1; ++i) {
                const auto* s1 = states[i]->as<ob::SE2StateSpace::StateType>();
                const auto* s2 = states[i+1]->as<ob::SE2StateSpace::StateType>();
                
                StateXY p1 = {s1->getX(), s1->getY()};
                StateXY p2 = {s2->getX(), s2->getY()};

                Vector segment_costs = calculateSegmentCost(p1, p2);
                
                // 누적
                for(int k=0; k<3; ++k) {
                    sol.objectives[k] += segment_costs[k];
                }
            }
        } else {
            // 실패 시 페널티
            sol.objectives = {1e5, 1e5, 1e5};
        }
        return sol;
    }
};

/*
MRPS Algorithm Implementation
*/
class RegretBasedSampler{
    RobotPlanner& planner;
    RegretMaximizationLP solver;
    std::vector<Sample> samples;
    std::priority_queue<Neighborhood> neighborhood_queue;
    int n_objs;
    public:
    RegretBasedSampler(RobotPlanner& p, GRBEnv& env): planner(p), solver(env){
        n_objs = planner.getNumObjectives();
    }
    const std::vector<Sample>& getSamples() const{
        return samples;
    };
    void evaluateNeighborhood(Neighborhood& N){
        // Collect vertex points
        std::vector<const Sample*> vertices;
        for (int idx : N.vertex_indices){
            vertices.push_back(&samples[idx]);
        }
        solver.solve(N,vertices);
    }
    void run(int K){
        std::vector<int> indices;
        for (int i =0; i < n_objs; ++i){
            Vector w(n_objs,0.0);
            w[i] = 1.0;
            std::cout << "Init " << i << "-th Objective with weight: ";
            PlannerSolution s = planner.solve(w);
            double u = dot(w,s.objectives);
            std::cout << "Cost: "; printVec(s.objectives); std::cout << endl;
            samples.push_back(Sample{w,s,u});
            indices.push_back(i);
        }
    }



};
