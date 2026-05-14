/*
 * OMPL Planner using RRT* with RPS Performance Checking (No Smoothing)
 * * Refactored with Object-Oriented Environment and Obstacle Handling
 * * Logic:
 * - Solves using RRT* with iterative convergence loop.
 * - Uses shared Environment class for consistent metrics with RPS.
 *
 * Compilation:
g++ -m64 -O3 corners_ompl.cpp -o corners_checking \
 -I/home/seung/ompl/src/ \
 -L/home/seung/ompl/build/src/ompl \
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

#include <ompl/base/TypedSpaceInformation.h>
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
    virtual double getRiskContribution(const StateStruct& s) const {
        double clearance = getClearance(s);
        if (clearance <= 0.1) return 1e6;
        double val = clearance * clearance + 1e-3;
        return 1.0 / val;
    }
    virtual bool isHardConstraint() const { return true; }
};

// Circular Obstacle Implementation
class CircularObstacle : public Obstacle {
    double cx, cy, radius;
public:
    CircularObstacle(double x, double y, double r) : cx(x), cy(y), radius(r) {}

    bool CheckCollision(const StateStruct& s) const override {
        return getClearance(s) <= 0.1;
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

// Rectangular Obstacle Implementation (Added from groundtruth_ompl.cpp)
class RectangularObstacle : public Obstacle {
    double x_min, x_max, y_min, y_max;
public:
    RectangularObstacle(double x_min_v, double x_max_v, double y_min_v, double y_max_v) 
        : x_min(x_min_v), x_max(x_max_v), y_min(y_min_v), y_max(y_max_v) {}

    bool CheckCollision(const StateStruct& s) const override {
        return (s.x >= x_min && s.x <= x_max && s.y >= y_min && s.y <= y_max);
    }

    double getClearance(const StateStruct& s) const override {
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
class GaussianRiskField : public Obstacle {
    double cx, cy, sigma, amplitude;
public:
    GaussianRiskField(double x, double y, double s, double A)
        : cx(x), cy(y), sigma(s), amplitude(A) {}

    // Soft field: never collides.
    bool CheckCollision(const StateStruct& /*s*/) const override {
        return false;
    }

    // Return a large positive clearance so any legacy threshold check passes.
    // (Actual risk is computed via getRiskContribution below.)
    double getClearance(const StateStruct& /*s*/) const override {
        return std::numeric_limits<double>::infinity();
    }

    double getRiskContribution(const StateStruct& s) const override {
        double dx = s.x - cx;
        double dy = s.y - cy;
        double r2 = dx * dx + dy * dy;
        return amplitude * std::exp(-r2 / (2.0 * sigma * sigma));
    }

    bool isHardConstraint() const override { return false; }
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
            if (!obs->isHardConstraint()) continue;
            // Using small buffer 0.1 consistent with RPS code
            if (obs->getClearance(s) <= 0.1) return false;
        }
        return true;
    }

    double speed_jump(double y) const {
        return (y < 13.0) ? 100.0 : 2.0;
    }

    double speed_smooth(double y) const {
        // Smooth transition between 100 and 2 around 
        double speed_slow = 2.0;
        double speed_fast = 100.0;
        double transition_mid = 15.0;
        double k = 0.2; // +- 5. 
        double exp_term = std::exp(-k * (y - transition_mid));
        double ratio = 1.0 / (1.0 + exp_term);

        return speed_fast + ratio * (speed_slow - speed_fast);
    }
    double speed_bell(double y) const {
    // Bell-curve velocity centered at y=15, symmetric.
    // Fastest at the centerline (y=15), slows down on both sides.
    // Killing the up/down asymmetry that caused Sc8's bifurcation.
    double speed_slow = 2.0;
    double speed_fast = 100.0;
    double y_peak     = 9.0;      // ← 직선 경로(y=15)에서 4 unit 떨어진 곳에 속도 peak
    double sigma_v    = 3.0;
    double dy = y - y_peak;
    double bell = std::exp(-(dy*dy) / (2.0 * sigma_v * sigma_v));
    return speed_slow + (speed_fast - speed_slow) * bell;
    }
    double speed_bell_24(double y) const {
        // For scenario 24 : 
        double speed_slow = 2.0;
        double speed_fast = 100.0;
        double y_peak     = 10.0;
        double sigma_v    = 3.0;       // 좁게 → 빠른 영역이 또렷하게 분리됨
        double dy = y - y_peak;
        double bell = std::exp(-(dy*dy) / (2.0 * sigma_v * sigma_v));
    return speed_slow + (speed_fast - speed_slow) * bell;
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
        int steps = 101; 

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

            // Updated to match groundtruth_ompl.cpp logic
            double speed = speed_bell(curr_state.y);
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
        case 3: // Velocity Change is smooth (not jump)
            std::cout << "Scenario 3: Smooth Velocity Change (No Obstacles)" << std::endl;
            break;
        case 4 : // Velocity Change is smooth (not jump) + One Circle
            std::cout << "Scenario 4: Smooth Velocity Change + Single Circle" << std::endl;
            global_env->addObstacle(std::make_shared<CircularObstacle>(11.0, 13.0, 3.0));
            break;
        case 5 : // Slit obstacle + Smooth velocity change (Predicting holes on Pareto front.)
            std::cout << "Scenario 5: Smooth Velocity Change + Rectangular Obstacles" << std::endl;
            global_env->addObstacle(std::make_shared<RectangularObstacle>(6.0,17.0,9.0,13.0));
            global_env->addObstacle(std::make_shared<RectangularObstacle>(6.0,17.0,17.0,21.0));
            global_env->addObstacle(std::make_shared<RectangularObstacle>(6.0,17.0,25.0,29.0));
            break;
        case 6 : // Clean convex PF: Soft Gaussian risk field + Smooth velocity change
            std::cout << "Scenario 6: Smooth Velocity Change + Soft Gaussian Risk Field (Convex PF)" << std::endl;
            global_env->addObstacle(std::make_shared<GaussianRiskField>(
                11.0,   // center x
                13.0,   // center y  (below direct line y=15, inside fast region)
                3.0,    // sigma     (spatial spread)
                8.0));  // amplitude (peak risk at center)
            global_env->addObstacle(std::make_shared<GaussianRiskField>(
                11.0,   // center x
                17.0,   // center y  (below direct line y=15, inside fast region)
                3.0,    // sigma     (spatial spread)
                4.0));  // amplitude (peak risk at center)
            break;
        case 7 : // Clean convex PF: Soft Gaussian risk field + Smooth velocity change
            std::cout << "Scenario 7: Smooth Velocity Change + one Soft Gaussian Risk Field (Convex PF)" << std::endl;
            global_env->addObstacle(std::make_shared<GaussianRiskField>(
                11.0,   // center x
                13.0,   // center y  (below direct line y=15, inside fast region)
                3.0,    // sigma     (spatial spread)
                8.0));  // amplitude (peak risk at center)
            break;
        case 8 : // Clean convex PF: Soft Gaussian risk field + Smooth velocity change
            std::cout << "Scenario 8: Smooth Velocity Change (more smooth) + Soft Gaussian Risk Field (Convex PF)" << std::endl;
            global_env->addObstacle(std::make_shared<GaussianRiskField>(
                11.0,   // center x
                13.0,   // center y  (below direct line y=15, inside fast region)
                3.0,    // sigma     (spatial spread)
                8.0));  // amplitude (peak risk at center)
            global_env->addObstacle(std::make_shared<GaussianRiskField>(
                11.0,   // center x
                17.0,   // center y  (below direct line y=15, inside fast region)
                3.0,    // sigma     (spatial spread)
                4.0));  // amplitude (peak risk at center)
            break;
        case 12 :
            std::cout << "Scenario 12: Aligned 3-objective convex PF (modified Sc8)" << std::endl;
    
            // 아래쪽 Gaussian: 약하게 + Time 빠른 영역에 위치
            // → 살짝 들어가도 Risk 약간만 늘고 Time은 크게 절약
            global_env->addObstacle(std::make_shared<GaussianRiskField>(
                11.0,   // x
                13.0,   // y  (속도 빠른 영역, y<15)
                3.5,    // sigma  (약간 넓게)
                3.0));  // amplitude  (약하게)
            
            // 위쪽 Gaussian: 강하게 + 더 위로 + 더 넓게
            // → 위로 우회하는 경로를 강력하게 페널라이즈해서 dominated로 만듦
            global_env->addObstacle(std::make_shared<GaussianRiskField>(
                11.0,   // x
                19.0,   // y  (속도 느린 영역에서 더 위로)
                4.0,    // sigma  (넓게)
                15.0)); // amplitude  (강하게, 8보다 두 배 가까이)
            break;
        case 13 : 
            std::cout << "Scenario 13 : Aligned 3-objective convex PF (modified Sc8)" << std::endl;
    
            // 아래쪽 Gaussian: 약하게 + Time 빠른 영역에 위치
            // → 살짝 들어가도 Risk 약간만 늘고 Time은 크게 절약
            global_env->addObstacle(std::make_shared<GaussianRiskField>(
                11.0,   // x
                13.0,   // y  (속도 빠른 영역, y<15)
                3.5,    // sigma  (약간 넓게)
                3.0));  // amplitude  (약하게)
            
            // 위쪽 Gaussian: 강하게 + 더 위로 + 더 넓게
            // → 위로 우회하는 경로를 강력하게 페널라이즈해서 dominated로 만듦
            global_env->addObstacle(std::make_shared<GaussianRiskField>(
                11.0,   // x
                19.0,   // y  (속도 느린 영역에서 더 위로)
                4.0,    // sigma  (넓게)
                12.0)); // amplitude  (강하게, 8보다 두 배 가까이)
            break;
        case 14 : 
            std::cout << "Scenario 14 : Aligned 3-objective convex PF (modified Sc8)" << std::endl;
    
            // 아래쪽 Gaussian: 약하게 + Time 빠른 영역에 위치
            // → 살짝 들어가도 Risk 약간만 늘고 Time은 크게 절약
            global_env->addObstacle(std::make_shared<GaussianRiskField>(
                11.0,   // x
                13.0,   // y  (속도 빠른 영역, y<15)
                3.5,    // sigma  (약간 넓게)
                3.0));  // amplitude  (약하게)
            
            // 위쪽 Gaussian: 강하게 + 더 위로 + 더 넓게
            // → 위로 우회하는 경로를 강력하게 페널라이즈해서 dominated로 만듦
            global_env->addObstacle(std::make_shared<GaussianRiskField>(
                11.0,   // x
                21.0,   // y  (속도 느린 영역에서 더 위로)
                4.0,    // sigma  (넓게)
                12.0)); // amplitude  (강하게, 8보다 두 배 가까이)
            break;
        case 21 :
            // Bell-curve velocity + Single off-center Gaussian (가장 보수적)
            // 속도가 대칭이니 좌우 대칭 분기를 막기 위해 Gaussian을 한 쪽으로 offset
            global_env->addObstacle(std::make_shared<GaussianRiskField>(
                11.0,   // x
                18.0,   // y  ← 직선 경로(y=15) 위쪽
                4.0,    // sigma
                8.0));  // amplitude
            break;

        case 22 :
            // Bell-curve velocity + Sc12 스타일 비대칭 두 Gaussian
            // 속도장도 바꾸고 Risk도 비대칭화 (가장 강한 분기 차단)
            std::cout << "Scenario 16: Wider Three-Corner Separation" << std::endl;
    
            // 속도 peak를 더 아래(y=8)로
            // → Time-min은 더 큰 우회를 요구 → Length와 더 강하게 경쟁
            // (speed_smooth에서 y_peak=8.0으로 변경)
            
            global_env->addObstacle(std::make_shared<GaussianRiskField>(
                11.0,
                20.0,   // ← Risk 우회도 더 멀리 가야 효과
                4.0,
                8.0));
            break;

        case 23 :
            // Bell-curve velocity + Single central Gaussian (가장 깨끗한 3-목적 trade-off)
            // 속도장이 대칭이라 분기 위험이 본질적으로 작음 → Gaussian도 중앙에 둘 수 있음
            std::cout << "Scenario 23: Single Circle + Bell curve" << std::endl;
            global_env->addObstacle(std::make_shared<CircularObstacle>(11.0, 13.0, 3.0));     
        case 24 :
                std::cout << "Scenario 24: Two Hard Obstacles Block Upper Detour" << std::endl;
    
            // 메인 obstacle: 직선 경로 위쪽
            global_env->addObstacle(std::make_shared<CircularObstacle>(
                11.0, 17.0, 2.5));
            
            // 차단용 obstacle: 더 위쪽에 배치 → 위로 우회를 어렵게 만듦
            // 위로 가려면 y=24 이상 가야 하므로 길이 손해가 너무 큼 → dominated
            global_env->addObstacle(std::make_shared<CircularObstacle>(
                11.0, 22.0, 2.0));
        break;
        case 25 :
            std::cout << "Scenario 25: Narrowed upper Gaussian for gap filling" << std::endl;
            
            // 아래쪽: Sc8과 완전히 동일 유지
            global_env->addObstacle(std::make_shared<GaussianRiskField>(
                11.0, 13.0, 3.0, 8.0));
            
            // 위쪽: sigma 좁히고 amplitude 키움 (3.0/4.0 → 1.5/6.0)
            global_env->addObstacle(std::make_shared<GaussianRiskField>(
                11.0, 17.0, 1.5, 6.0));
            break;
        case 26:
            std::cout << "Scenario 26: Wider upper Gaussian for gap filling" << std::endl;
            global_env->addObstacle(std::make_shared<GaussianRiskField>(11.0, 13.0, 3.0, 8.0));
            global_env->addObstacle(std::make_shared<GaussianRiskField>(11.0, 17.0, 2.5, 6.0));
            break;

        case 27:
            std::cout << "Scenario 27: Stronger lower Gaussian for gap filling" << std::endl;
            global_env->addObstacle(std::make_shared<GaussianRiskField>(11.0, 13.0, 3.0, 10.0));
            global_env->addObstacle(std::make_shared<GaussianRiskField>(11.0, 17.0, 1.5, 6.0));
            break;
        case 31:
             std::cout << "Sc8 with bell-curve velocity (peak below direct line)" << std::endl;
            global_env->addObstacle(std::make_shared<GaussianRiskField>(11.0, 13.0, 3.0, 8.0));
            global_env->addObstacle(std::make_shared<GaussianRiskField>(11.0, 17.0, 3.0, 4.0));
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
// 6. Resume Helper
// ==========================================
std::set<std::string> loadProcessedWeights(const std::string& filename) {
    std::set<std::string> processed;
    std::ifstream file(filename);
    if (!file.is_open()) return processed;

    std::string line;
    while (std::getline(file, line)) {
        size_t lastQuote = line.find_last_of('"');
        if (lastQuote == std::string::npos) continue;
        
        size_t firstQuoteOfLast = line.find_last_of('"', lastQuote - 1);
        if (firstQuoteOfLast == std::string::npos) continue;

        std::string wStr = line.substr(firstQuoteOfLast + 1, lastQuote - firstQuoteOfLast - 1);
        if (!wStr.empty()) {
            processed.insert(wStr);
        }
    }
    return processed;
}
void smoothWaypoints(og::PathGeometric& path, int window = 5) {
    auto& states = path.getStates();
    const int n = states.size();
    if (n < window + 2) return;
    
    const int half = window / 2;

    std::vector<double> xs(n), ys(n);
    for (int i = 0; i < n; ++i) {
        const auto* s = states[i]->as<ob::RealVectorStateSpace::StateType>();
        xs[i] = s->values[0];
        ys[i] = s->values[1];
    }
    
    for (int i = half; i < n - half; ++i) {
        double sx = 0.0, sy = 0.0;
        for (int j = -half; j <= half; ++j) {
            sx += xs[i + j];
            sy += ys[i + j];
        }
        auto* s = states[i]->as<ob::RealVectorStateSpace::StateType>();
        s->values[0] = sx / window;
        s->values[1] = sy / window;
    }
}
// ==========================================
//  Main Function
// ==========================================
int main(int argc, char* argv[]) {
    // ------------------------------------------
    // 1. Configuration (Manual Setup)
    // ------------------------------------------
    int scenario = 1; // Default
    if (argc > 1) {
        scenario = std::stoi(argv[1]);
    }
    configureEnvironment(scenario);
    
    // Target corner cases for isolated trajectory checks
    std::vector<std::vector<double>> corner_cases ={
        {1.0,0.0,0.0}, 
        {0.0,1.0,0.0}, 
        {0.0,0.0,1.0}, 
    };
    
    for (auto & target_weights : corner_cases) {
        std::cout << "\n--- Running Single Weight Case ---" << std::endl;
        std::cout << "Scenario: " << scenario << std::endl;
        std::cout << "Weights:  Dist=" << target_weights[0] 
                  << ", Risk=" << target_weights[1] 
                  << ", Time=" << target_weights[2] << std::endl;

        configureEnvironment(scenario);

        // 2. Setup Space and Bounds
        auto space(std::make_shared<ob::RealVectorStateSpace>(2));
        ob::RealVectorBounds bounds(2);
        bounds.setLow(0.0); bounds.setHigh(40.0);
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
        planner->setRange(2.0);
        setup.setPlanner(planner);

        // ------------------------------------------
        // 7. Solve (Iterative Convergence)
        // ------------------------------------------
        std::cout << "Solving..." << std::endl;

        double prev_cost = std::numeric_limits<double>::infinity();
        double current_cost = std::numeric_limits<double>::infinity();
        
        // Performance settings
        double time_slice = 360;     // Seconds per batch
        // Initial Solve
        setup.solve(time_slice);

        // ------------------------------------------
        // 8. Results: Save Individual Trajectory File
        // ------------------------------------------
        if (setup.haveExactSolutionPath()) {
            
            og::PathGeometric& path = setup.getSolutionPath();

            path.interpolate(200);
            smoothWaypoints(path, 5); // Simple moving average smoothing (window=5)
            
            // Construct Filename: trajectory_{w1}_{w2}_{w3}_S{scenario}.txt
            std::stringstream ss;
            ss << "trajectory_" 
               << std::fixed << std::setprecision(1) 
               << target_weights[0] << "_" 
               << target_weights[1] << "_" 
               << target_weights[2] 
               << "_S" << scenario << ".txt"; 
               
            std::string filename = ss.str();

            const auto& states = path.getStates();
            std::stringstream ss_x, ss_y;
            double total_dist = 0.0, total_risk = 0.0, total_time = 0.0;
            for (size_t i = 0; i < states.size(); ++i) {
                const auto* s = states[i]->as<ob::RealVectorStateSpace::StateType>();
                ss_x << s->values[0] << (i < states.size()-1 ? ";" : "");
                ss_y << s->values[1] << (i < states.size()-1 ? ";" : "");

                if (i < states.size() - 1) {
                    const auto* n = states[i+1]->as<ob::RealVectorStateSpace::StateType>();
                    StateStruct st1 = {s->values[0], s->values[1]};
                    StateStruct st2 = {n->values[0], n->values[1]};
                    
                    // Use Environment to calculate Final Metrics
                    std::vector<double> seg = global_env->calculateSegmentCost(st1, st2);
                    total_dist += seg[0]; total_risk += seg[1]; total_time += seg[2];
                }
            }
            
            double Cost = total_dist*target_weights[0] + total_risk*target_weights[1] + total_time*target_weights[2];
            std::stringstream cleanW; cleanW << target_weights[0] << ";" << target_weights[1] << ";" << target_weights[2];
            // Write to File
            std::ofstream outFile(filename);
            if (outFile.is_open()) {
                path.printAsMatrix(outFile);
                outFile << total_dist << "," << total_risk << "," << total_time << "," 
                << "\"" << ss_x.str() << "\"," << "\"" << ss_y.str() << "\"," 
                << Cost << "," << "\"" << cleanW.str() << "\"\n";
                outFile.close();
                std::cout << ">> Saved trajectory to: " << filename << std::endl;
            } else {
                std::cerr << "!! Error opening file: " << filename << std::endl;
            }

            
        }
         else {
            std::cout << ">> No exact solution found for these weights." << std::endl;
        }
    }
    return 0;
}
