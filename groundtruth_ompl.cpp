/*
 * OMPL Planner using RRT* with RPS Performance Checking
 * Refactored: scenario_id is the single top-level switch.
 * Path post-processing: interpolate(200) + moving-average smoothing.
 *
 * Compilation:
 g++ -m64 -O3 groundtruth_ompl.cpp -o groundTruth \
 -I/home/seung/ompl/src/ \
 -L/home/seung/ompl/build/src/ompl \
 -I/usr/include/eigen3 \
 -lompl -lpthread
 *
 * Usage:
 *   ./groundTruth <scenario_id> [weight_resolution]
 *   e.g.  ./groundTruth 8 70
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

// =============================================================
// 1. Obstacles
// =============================================================

struct StateStruct {
    double x, y;
};

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
// Penerate through just source of risk. 
class GaussianRiskField : public Obstacle {
    double cx, cy, sigma, amplitude;
public:
    GaussianRiskField(double x, double y, double s, double A)
        : cx(x), cy(y), sigma(s), amplitude(A) {}

    bool CheckCollision(const StateStruct&) const override { return false; }

    double getClearance(const StateStruct&) const override {
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

// =============================================================
// 2. Scenario Configuration
// =============================================================

enum class SpeedModel {
    JUMP,
    SMOOTH,
    BELL_CURVE,
    CONSTANT
};

struct ScenarioConfig {
    int    scenario_id  = 0;
    std::string description;

    SpeedModel speed_model = SpeedModel::SMOOTH;
    double speed_slow      = 2.0;
    double speed_fast      = 100.0;
    double speed_mid       = 15.0;
    double speed_k         = 0.2;
    double speed_sigma     = 6.0;

    bool   normalize       = false;
    double norm_length     = 1.0;
    double norm_risk       = 1.0;
    double norm_time       = 1.0;
};

// =============================================================
// 3. Environment
// =============================================================

class Environment {
    std::vector<std::shared_ptr<Obstacle>> obstacles;
    ScenarioConfig config;

public:
    void addObstacle(std::shared_ptr<Obstacle> obs) {
        obstacles.push_back(obs);
    }

    void setConfig(const ScenarioConfig& c) { config = c; }
    const ScenarioConfig& getConfig() const { return config; }

    bool isValid(const StateStruct& s) const {
        for (const auto& obs : obstacles) {
            if (!obs->isHardConstraint()) continue;
            if (obs->getClearance(s) <= 0.1) return false;
        }
        return true;
    }

    double getSpeed(double y) const {
        switch (config.speed_model) {
            case SpeedModel::JUMP:
                return (y < config.speed_mid) ? config.speed_fast : config.speed_slow;
            case SpeedModel::SMOOTH: {
                double exp_term = std::exp(-config.speed_k * (y - config.speed_mid));
                double ratio = 1.0 / (1.0 + exp_term);
                return config.speed_fast + ratio * (config.speed_slow - config.speed_fast);
            }
            case SpeedModel::BELL_CURVE: {
                double dy = y - config.speed_mid;
                double bell = std::exp(-(dy * dy) / (2.0 * config.speed_sigma * config.speed_sigma));
                return config.speed_slow + (config.speed_fast - config.speed_slow) * bell;
            }
            case SpeedModel::CONSTANT:
            default:
                return config.speed_fast;
        }
    }

    double getPointRisk(const StateStruct& s) const {
        double total_risk = 0.0;
        const double max_risk_penalty = 1e6;
        for (const auto& obs : obstacles) {
            total_risk += obs->getRiskContribution(s);
            if (total_risk >= max_risk_penalty) return max_risk_penalty;
        }
        return std::min(total_risk, max_risk_penalty);
    }

    double getEuclideanDist(const StateStruct& s1, const StateStruct& s2) const {
        return std::sqrt(std::pow(s1.x - s2.x, 2) + std::pow(s1.y - s2.y, 2));
    }

    // Returns [Length, Risk, Time]
    std::vector<double> calculateSegmentCost(const StateStruct& s_from, const StateStruct& s_to) const {
        std::vector<double> cost(3, 0.0);
        const int steps = 101;

        cost[0] = getEuclideanDist(s_from, s_to);

        double sum_risk = 0.0;
        double sum_time = 0.0;
        StateStruct prev = s_from;

        for (int i = 1; i <= steps; ++i) {
            double r = static_cast<double>(i) / steps;
            StateStruct curr {
                s_from.x + r * (s_to.x - s_from.x),
                s_from.y + r * (s_to.y - s_from.y)
            };
            StateStruct mid {
                (curr.x + prev.x) / 2.0,
                (curr.y + prev.y) / 2.0
            };

            sum_risk += getPointRisk(mid);
            sum_time += getEuclideanDist(prev, curr) / getSpeed(curr.y);

            prev = curr;
        }

        cost[1] = sum_risk * cost[0] / steps;
        cost[2] = sum_time;

        if (config.normalize) {
            cost[0] /= config.norm_length;
            cost[1] /= config.norm_risk;
            cost[2] /= config.norm_time;
        }

        return cost;
    }
};

std::shared_ptr<Environment> global_env;

// =============================================================
// 4. Environment Configuration
// =============================================================

void configureEnvironment(int scenario_id) {
    global_env = std::make_shared<Environment>();
    global_env->addObstacle(std::make_shared<BoundaryObstacle>(0.0, 40.0));

    ScenarioConfig cfg;
    cfg.scenario_id = scenario_id;

    switch (scenario_id) {
        case 0:
            cfg.description = "Empty space";
            cfg.speed_model = SpeedModel::CONSTANT;
            break;

        case 1:
            cfg.description = "Single circle + jump velocity";
            cfg.speed_model = SpeedModel::JUMP;
            cfg.speed_mid   = 13.0;
            global_env->addObstacle(std::make_shared<CircularObstacle>(11.0, 13.0, 3.0));
            break;

        case 2:
            cfg.description = "Two circles + jump velocity";
            cfg.speed_model = SpeedModel::JUMP;
            cfg.speed_mid   = 13.0;
            global_env->addObstacle(std::make_shared<CircularObstacle>(11.0, 13.0, 3.0));
            global_env->addObstacle(std::make_shared<CircularObstacle>(11.0, 21.0, 2.0));
            break;

        case 3:
            cfg.description = "Smooth velocity, no obstacles";
            cfg.speed_model = SpeedModel::SMOOTH;
            break;

        case 4:
            cfg.description = "Smooth velocity + single circle (normalized)";
            cfg.speed_model = SpeedModel::SMOOTH;
            cfg.normalize   = true;
            cfg.norm_length = 26.2519;
            cfg.norm_risk   = 607006.0;
            cfg.norm_time   = 0.656958;
            global_env->addObstacle(std::make_shared<CircularObstacle>(11.0, 13.0, 3.0));
            break;

        case 5:
            cfg.description = "Smooth velocity + slit obstacles";
            cfg.speed_model = SpeedModel::SMOOTH;
            global_env->addObstacle(std::make_shared<RectangularObstacle>(6.0, 17.0, 9.0, 13.0));
            global_env->addObstacle(std::make_shared<RectangularObstacle>(6.0, 17.0, 17.0, 21.0));
            global_env->addObstacle(std::make_shared<RectangularObstacle>(6.0, 17.0, 25.0, 29.0));
            break;

        case 6:
            cfg.description = "Smooth velocity + two Gaussians";
            cfg.speed_model = SpeedModel::SMOOTH;
            global_env->addObstacle(std::make_shared<GaussianRiskField>(11.0, 13.0, 3.0, 8.0));
            global_env->addObstacle(std::make_shared<GaussianRiskField>(11.0, 17.0, 3.0, 4.0));
            break;

        case 7:
            cfg.description = "Smooth velocity + single Gaussian";
            cfg.speed_model = SpeedModel::SMOOTH;
            global_env->addObstacle(std::make_shared<GaussianRiskField>(11.0, 13.0, 3.0, 8.0));
            break;

        case 8:
            cfg.description = "Smooth velocity (k=0.2) + two Gaussians";
            cfg.speed_model = SpeedModel::SMOOTH;
            cfg.normalize   = true;
            cfg.norm_length = 41.124;
            cfg.norm_risk   = 73.1902;
            cfg.norm_time   = 2.81075;
            global_env->addObstacle(std::make_shared<GaussianRiskField>(11.0, 13.0, 3.0, 8.0));
            global_env->addObstacle(std::make_shared<GaussianRiskField>(11.0, 17.0, 3.0, 4.0));
            break;

        case 12:
            cfg.description = "Modified Sc8 (A=15) normalized";
            cfg.speed_model = SpeedModel::SMOOTH;
            cfg.normalize   = true;
            cfg.norm_length = 22.3147;
            cfg.norm_risk   = 1.33705;
            cfg.norm_time   = 0.446142;
            global_env->addObstacle(std::make_shared<GaussianRiskField>(11.0, 13.0, 3.5, 3.0));
            global_env->addObstacle(std::make_shared<GaussianRiskField>(11.0, 19.0, 4.0, 15.0));
            break;

        case 13:
            cfg.description = "Modified Sc8 (A=12) normalized";
            cfg.speed_model = SpeedModel::SMOOTH;
            cfg.normalize   = true;
            cfg.norm_length = 22.3143;
            cfg.norm_risk   = 1.34155;
            cfg.norm_time   = 0.44209;
            global_env->addObstacle(std::make_shared<GaussianRiskField>(11.0, 13.0, 3.5, 3.0));
            global_env->addObstacle(std::make_shared<GaussianRiskField>(11.0, 19.0, 4.0, 12.0));
            break;

        case 14:
            cfg.description = "Modified Sc8 (y=21) normalized";
            cfg.speed_model = SpeedModel::SMOOTH;
            cfg.normalize   = true;
            cfg.norm_length = 22.3485;
            cfg.norm_risk   = 1.35039;
            cfg.norm_time   = 0.444471;
            global_env->addObstacle(std::make_shared<GaussianRiskField>(11.0, 13.0, 3.5, 3.0));
            global_env->addObstacle(std::make_shared<GaussianRiskField>(11.0, 21.0, 4.0, 12.0));
            break;

        case 21:
            cfg.description = "Bell-curve velocity + off-center Gaussian";
            cfg.speed_model = SpeedModel::BELL_CURVE;
            cfg.speed_mid   = 11.0;
            cfg.speed_sigma = 4.0;
            global_env->addObstacle(std::make_shared<GaussianRiskField>(11.0, 18.0, 4.0, 8.0));
            break;

        case 22:
            cfg.description = "Bell-curve velocity + far Gaussian";
            cfg.speed_model = SpeedModel::BELL_CURVE;
            cfg.speed_mid   = 11.0;
            cfg.speed_sigma = 4.0;
            global_env->addObstacle(std::make_shared<GaussianRiskField>(11.0, 20.0, 4.0, 8.0));
            break;

        case 23:
            cfg.description = "Bell-curve velocity + single hard circle";
            cfg.speed_model = SpeedModel::BELL_CURVE;
            cfg.speed_mid   = 11.0;
            cfg.speed_sigma = 4.0;
            global_env->addObstacle(std::make_shared<CircularObstacle>(11.0, 13.0, 3.0));
            break;

        case 24:
            cfg.description = "Narrow bell-curve + two hard obstacles";
            cfg.speed_model = SpeedModel::BELL_CURVE;
            cfg.speed_mid   = 10.0;
            cfg.speed_sigma = 3.0;
            global_env->addObstacle(std::make_shared<CircularObstacle>(11.0, 17.0, 2.5));
            global_env->addObstacle(std::make_shared<CircularObstacle>(11.0, 22.0, 2.0));
            break;

        case 25:
            cfg.description = "Sc8 with narrowed upper Gaussian";
            cfg.speed_model = SpeedModel::SMOOTH;
            cfg.normalize   = true;
            // 20.0001,0.959651,0.392362
            // 20.2029,0.958411,0.410625
            // 21.8051,1.17829,0.350297
            cfg.norm_length = 22.3485;
            cfg.norm_risk   = 1.35039;
            cfg.norm_time   = 0.444471;
            global_env->addObstacle(std::make_shared<GaussianRiskField>(11.0, 13.0, 3.0, 8.0));
            global_env->addObstacle(std::make_shared<GaussianRiskField>(11.0, 17.0, 1.5, 6.0));
            break;
        case 26:
            cfg.description = "Sc25 with widened upper Gaussian (sigma 1.5 -> 2.5)";
            cfg.speed_model = SpeedModel::SMOOTH;
            cfg.normalize   = true;
            //
            // 20.0001,0.959647,0.39258
            // 20.218,0.958415,0.411677
            // 21.8154,1.18277,0.350289
            cfg.norm_length = 21.8154;
            cfg.norm_risk   = 1.18277;
            cfg.norm_time   = 0.411677;
            global_env->addObstacle(std::make_shared<GaussianRiskField>(11.0, 13.0, 3.0, 8.0));
            global_env->addObstacle(std::make_shared<GaussianRiskField>(11.0, 17.0, 2.5, 6.0));
            break;
        case 27:
            cfg.description = "Sc25 with stronger lower Gaussian (A 8 -> 10)";
            cfg.speed_model = SpeedModel::SMOOTH;
            //
            //20.0002,0.959677,0.392427
            //20.2071,0.958412,0.410887
            // 21.789,1.17963,0.350292
            cfg.normalize   = true;
            cfg.norm_length = 21.789;
            cfg.norm_risk   = 1.17963;
            cfg.norm_time   = 0.410887;
            global_env->addObstacle(std::make_shared<GaussianRiskField>(11.0, 13.0, 3.0, 10.0));
            global_env->addObstacle(std::make_shared<GaussianRiskField>(11.0, 17.0, 1.5, 6.0));
            break;
        case 31:
            cfg.description = "Sc8 with bell-curve velocity (peak below direct line)";
            cfg.speed_model = SpeedModel::BELL_CURVE;
            cfg.speed_mid   = 9.0;        // velocity peak far below y=15
            cfg.speed_sigma = 3.0;        // narrow peak
            cfg.normalize   = true;
            // 20.0003,0.95973,1.30326
            // 20.2125,0.958409,1.65974
            // 25.6831,2.31764,0.422187
            cfg.norm_length = 25.6831;
            cfg.norm_risk   = 2.31764;
            cfg.norm_time   = 1.65974;
            global_env->addObstacle(std::make_shared<GaussianRiskField>(11.0, 13.0, 3.0, 8.0));
            global_env->addObstacle(std::make_shared<GaussianRiskField>(11.0, 17.0, 3.0, 4.0));
            break;
        default:
            cfg.description = "Unknown scenario";
            break;
    }

    global_env->setConfig(cfg);

    static const char* speed_names[] = {"JUMP", "SMOOTH", "BELL_CURVE", "CONSTANT"};
    std::cout << "Scenario " << scenario_id << ": " << cfg.description << "\n"
              << "  Speed: " << speed_names[static_cast<int>(cfg.speed_model)]
              << " (mid=" << cfg.speed_mid;
    if (cfg.speed_model == SpeedModel::SMOOTH)     std::cout << ", k=" << cfg.speed_k;
    if (cfg.speed_model == SpeedModel::BELL_CURVE) std::cout << ", sigma=" << cfg.speed_sigma;
    std::cout << ")\n";
    if (cfg.normalize) {
        std::cout << "  Normalize: L/" << cfg.norm_length
                  << ", R/" << cfg.norm_risk
                  << ", T/" << cfg.norm_time << "\n";
    }
}

// =============================================================
// 5. OMPL Classes
// =============================================================

class CustomWeightedObjective : public ob::OptimizationObjective {
public:
    CustomWeightedObjective(const ob::SpaceInformationPtr &si, const std::vector<double>& weights)
        : ob::OptimizationObjective(si), weights_(weights) {}

    ob::Cost stateCost(const ob::State*) const override { return ob::Cost(0.0); }
    ob::Cost motionCostHeuristic(const ob::State*, const ob::State*) const override { return ob::Cost(0.0); }

    ob::Cost motionCost(const ob::State *s1, const ob::State *s2) const override {
        const auto* p1 = s1->as<ob::RealVectorStateSpace::StateType>();
        const auto* p2 = s2->as<ob::RealVectorStateSpace::StateType>();
        StateStruct st1 = {p1->values[0], p1->values[1]};
        StateStruct st2 = {p2->values[0], p2->values[1]};

        std::vector<double> c = global_env->calculateSegmentCost(st1, st2);

        double sum = 0.0;
        for (size_t i = 0; i < weights_.size() && i < c.size(); ++i) sum += weights_[i] * c[i];
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
        return global_env->isValid({s->values[0], s->values[1]});
    }
};

// =============================================================
// 6. IO Helpers
// =============================================================

std::vector<double> parseWeights(std::string wStr) {
    std::vector<double> w;
    std::string clean;
    for (char c : wStr) {
        if (c != '\"' && c != ' ' && c != '\r' && c != '\n') clean += c;
    }
    std::replace(clean.begin(), clean.end(), ';', ' ');
    std::stringstream ss(clean);
    double temp;
    while (ss >> temp) w.push_back(temp);
    while (w.size() < 3) w.push_back(0.0);
    return w;
}

// Robust resume: only accepts lines that end with closing quote.
// Truncated lines (from a crash mid-write) are skipped, so those
// weights will be reprocessed on the next run.
std::set<std::string> loadProcessedWeights(const std::string& filename) {
    std::set<std::string> processed;
    std::ifstream file(filename);
    if (!file.is_open()) return processed;

    std::string line;
    int line_num = 0;
    int skipped_malformed = 0;

    while (std::getline(file, line)) {
        line_num++;
        if (line_num == 1) continue;
        if (line.empty()) continue;

        if (line.back() != '"') {
            skipped_malformed++;
            continue;
        }

        size_t lastQuote = line.find_last_of('"');
        if (lastQuote == std::string::npos) continue;

        size_t firstQuoteOfLast = line.find_last_of('"', lastQuote - 1);
        if (firstQuoteOfLast == std::string::npos) continue;

        std::string wStr = line.substr(firstQuoteOfLast + 1,
                                        lastQuote - firstQuoteOfLast - 1);
        if (!wStr.empty()) {
            processed.insert(wStr);
        }
    }

    if (skipped_malformed > 0) {
        std::cout << ">>> Warning: " << skipped_malformed
                  << " malformed line(s) skipped (likely from a previous crash).\n"
                  << "    Those weights will be reprocessed.\n";
    }

    return processed;
}

// Moving-average smoothing on path waypoints (endpoints are preserved).
void smoothWaypoints(og::PathGeometric& path, int window = 5) {
    auto& states = path.getStates();
    const int n = static_cast<int>(states.size());
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

// =============================================================
// 7. Main
// =============================================================

int main(int argc, char* argv[]) {
    int scenario = 1;
    int weight_resolution = 30;

    if (argc > 1) scenario = std::stoi(argv[1]);
    if (argc > 2) weight_resolution = std::stoi(argv[2]);

    std::cout << ">>> Scenario: " << scenario
              << ", Weight resolution: 1/" << weight_resolution << std::endl;

    configureEnvironment(scenario);

    auto space(std::make_shared<ob::RealVectorStateSpace>(2));
    ob::RealVectorBounds bounds(2);
    bounds.setLow(0.0); bounds.setHigh(40.0);
    space->setBounds(bounds);

    og::SimpleSetup setup(space);
    ob::SpaceInformationPtr si = setup.getSpaceInformation();
    setup.setStateValidityChecker(std::make_shared<ObstacleValidityChecker>(si));

    ob::ScopedState<> start(space);
    ob::ScopedState<> goal(space);
    start[0] = 1.0; start[1] = 15.0;
    goal[0] = 21.0; goal[1] = 15.0;
    setup.setStartAndGoalStates(start, goal);

    // Generate simplex lattice weights (interval = 1/N -> (N+1)(N+2)/2 weights)
    std::vector<std::string> weightStrings;
    weightStrings.reserve((weight_resolution + 1) * (weight_resolution + 2) / 2);

    for (int i = 0; i <= weight_resolution; ++i) {
        for (int j = 0; j <= weight_resolution - i; ++j) {
            int k = weight_resolution - i - j;
            double w0 = static_cast<double>(i) / weight_resolution;
            double w1 = static_cast<double>(j) / weight_resolution;
            double w2 = static_cast<double>(k) / weight_resolution;

            std::stringstream ss;
            ss << w0 << ";" << w1 << ";" << w2;
            weightStrings.push_back(ss.str());
        }
    }

    std::cout << ">>> Generated " << weightStrings.size()
              << " weights at interval 1/" << weight_resolution << std::endl;

    if (weightStrings.empty()) {
        std::cerr << "No weights generated\n";
        return 1;
    }

    std::string outFileName = "Normalized_groundTruth_scenario_"
                            + std::to_string(scenario)
                            + "_res" + std::to_string(weight_resolution)
                            + ".csv";

    std::set<std::string> processedWeights = loadProcessedWeights(outFileName);
    bool fileExists = !processedWeights.empty();

    std::ofstream outFile;
    if (fileExists) {
        std::cout << ">>> Resuming: Found " << processedWeights.size()
                  << " existing entries in " << outFileName << std::endl;
        outFile.open(outFileName, std::ios::app);
    } else {
        std::cout << ">>> Starting fresh: " << outFileName << std::endl;
        outFile.open(outFileName);
        outFile << "Length, Risk, TravelTime, Paths.x, Paths.y, Cost, Weight\n";
        outFile.flush();
    }

    int skipped_count = 0;
    for (const auto& wStr : weightStrings) {
        std::vector<double> w = parseWeights(wStr);
        if (w.empty()) continue;
        std::stringstream cleanW;
        cleanW << w[0] << ";" << w[1] << ";" << w[2];

        if (processedWeights.count(cleanW.str())) {
            skipped_count++;
            if (skipped_count % 10 == 0)
                std::cout << "\rSkipped " << skipped_count
                          << " existing entries..." << std::flush;
            continue;
        }
        setup.clear();

        auto obj = std::make_shared<CustomWeightedObjective>(si, w);
        setup.setOptimizationObjective(obj);

        auto planner(std::make_shared<og::RRTstar>(si));
        planner->setRange(2.0);
        setup.setPlanner(planner);

        std::cout << "Weight [" << w[0] << ", " << w[1] << ", " << w[2]
                  << "] -> Solving..." << std::endl;

        double prev_cost = std::numeric_limits<double>::infinity();
        double current_cost = std::numeric_limits<double>::infinity();

        double time_slice = 30.0;
        double improvement_threshold = 0.0005;
        int max_batches = 4;
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
            } else {
                prev_cost = current_cost;
                setup.solve(time_slice);

                double new_cost = setup.getSolutionPath().cost(obj).value();
                double improvement = (prev_cost - new_cost) / prev_cost;

                if (improvement < improvement_threshold) {
                    std::cout << "  Converged at batch " << batch_count
                              << " (Imp: " << improvement << ")" << std::endl;
                    break;
                }
                current_cost = new_cost;
            }
            batch_count++;
        }

        if (setup.haveExactSolutionPath()) {
            og::PathGeometric& path = setup.getSolutionPath();

            path.interpolate(200);
            smoothWaypoints(path, 5);

            double total_dist = 0.0, total_risk = 0.0, total_time = 0.0;
            const auto& states = path.getStates();
            std::stringstream ss_x, ss_y;

            for (size_t i = 0; i < states.size(); ++i) {
                const auto* s = states[i]->as<ob::RealVectorStateSpace::StateType>();
                ss_x << s->values[0] << (i < states.size() - 1 ? ";" : "");
                ss_y << s->values[1] << (i < states.size() - 1 ? ";" : "");

                if (i < states.size() - 1) {
                    const auto* n = states[i + 1]->as<ob::RealVectorStateSpace::StateType>();
                    StateStruct st1 = {s->values[0], s->values[1]};
                    StateStruct st2 = {n->values[0], n->values[1]};

                    std::vector<double> seg = global_env->calculateSegmentCost(st1, st2);
                    total_dist += seg[0]; total_risk += seg[1]; total_time += seg[2];
                }
            }

            double Cost = total_dist * w[0] + total_risk * w[1] + total_time * w[2];
            std::stringstream cleanW2; cleanW2 << w[0] << ";" << w[1] << ";" << w[2];

            outFile << total_dist << "," << total_risk << "," << total_time << ","
                    << "\"" << ss_x.str() << "\"," << "\"" << ss_y.str() << "\","
                    << Cost << "," << "\"" << cleanW2.str() << "\"\n";
            outFile.flush();
        } else {
            std::stringstream cleanW2; cleanW2 << w[0] << ";" << w[1] << ";" << w[2];
            std::cout << "  Failed to find path." << std::endl;
            outFile << "0,0,0,\"\",\"\",0,\"" << cleanW2.str() << "\"\n";
            outFile.flush();
        }
    }
    outFile.close();
    std::cout << "Processing Complete. Data saved to " << outFileName << std::endl;
    return 0;
}
