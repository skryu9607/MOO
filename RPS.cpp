#include "RPS_IPlanner.h"
#include <stdexcept>
#include <memory>
#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <limits>

class MockPlanner : public IPlanner {
public:
    ComplexMockPlanner(int objectives) {

        pf_points = {
            {1.0, 15.0}, // A
            {2.0, 10.0}, // B
            {3.0, 7.0},  // C
            {5.0, 5.0},  // D
            {7.0, 3.0},  // E
            {10.0, 2.0}, // F
            {15.0, 1.0}  // G
        };
    }

    /**
     * @brief w·f를 최소화하는 해 (PF 점들 중 하나)의 f 벡터를 반환합니다.
     */
    Vector solve(const Vector& w) override {
        double w1 = w[0];
        double w2 = w[1];
        
        double min_cost = std::numeric_limits<double>::infinity();
        Vector best_f = pf_points[0];

        // 7개의 점을 모두 순회하며 w·f가 가장 작은 점을 찾음
        for (const auto& f_point : pf_points) {
            double cost = w1 * f_point[0] + w2 * f_point[1];
            if (cost < min_cost) {
                min_cost = cost;
                best_f = f_point;
            }
        }
        return best_f;
    }

private:
    Solutions pf_points; // f-vector들의 집합
};
class RegretSampler {
    private:
    std::unique_ptr<IPlanner> planner;
    Solutions omega; 
    int m;
    Solutions omega_L;

    Vector findNextWeight(){
        Vector w_star(m,0.0);
        double max_regret = -std::numeric_limits<double>::infinity();
        int num_grid_samples = 101;
        for (int i = 0; i< num_grid_samples;++i){
            Vector w_grid(m);
            w_grid[0] = (double) i / (num_grid_samples-1);
            w_grid[1] = 1.0 - w_grid[0];

            double x_w = std::numeric_limits<double>::infinity();
            for(const auto& sol : omega){
                double w_dot_f = w_grid[0] * sol.objectives[0] + w_grid[1] * sol.objectives[1];
                x_w = std::min(x_w,w_dot_f);
            }
            double p_w = -std::numeric_limits<double>::infinity();
            for(const auto& sol : omega){
                double w_minus_w_prime_dot_f = 0.0;
                for (int j = 0;j<m;++j){
                    w_minus_w_prime_dot_f += (w_grid[j] - sol.weight[j]) * sol.objectives[j];
                }
                double lower_bound_term = sol.cost + w_minus_w_prime_dot_f;
                p_w = std::max(p_w,lower_bound_term);
            }
            double current_regret_upper_bound = x_w - p_w;
            if (current_regret_upper_bound > max_regret){
                max_regret = current_regret_upper_bound;
                w_star = w_grid;
                if (max_regret == 0.0) {
                    break;
                }
            }
        }
        std::cout << "Worst Regret : " << max_regret << " at w*: [" << w_star[0] << ", " << w_star[1] << "]" << std::endl;
        
        return w_star;
    
    };

    public:
        RegretSampler(std::unique_ptr<IPlanner> p, int num_objectives):planner(std::move(p)),m(num_objectives){}

    void initialize(){
        omega.clear();
        std::cout << "Initializing Omega with vertex weights..." << std::endl;
        for (int i = 0;i<m;++i){
            Vector w_vertex(m,0.0);
            w_vertex[i] = 1.0; 
            Vector obj = planner->solve(w_vertex); 
            omega.emplace_back(w_vertex,obj);
        }
    }
    void run(int K) {
        if (omega.empty()){
            initialize(); 
        }
        while (omega.size()< K) {
            std::cout << "Repeat this process " << omega.size() + 1 << "/" << K << "..." << std::endl;
            Vector w_star = findNextWeight();
            Vector f_star = planner->solve(w_star);
            omega.emplace_back(w_star,f_star);
        if 
        }

        std::cout << "Total " << K << " samples are done." << std::endl;
    }
    const Solutions& getOmega() const{return omega;}
};

int main(){
    int m = 2;
    int K = 6;
    try { 
        auto planner_ptr = std::make_unique<MockPlanner>(m);
        RegretSampler sampler(std::move(planner_ptr),m);
        sampler.run(K);
        std::cout << "Final Omega set:" << std::endl;
        for (const auto& sol : sampler.getOmega()){
            std::cout << " w' = [" << sol.weight[0] << ", " << sol.weight[1] << "]"
                      << " -> f(s*) = [" << sol.objectives[0] << ", " << sol.objectives[1] << "]"
                      << " (cost u(w') = " << sol.cost << ")" << std::endl;
        }
    }
    catch (const std::exception& e){
        std::cerr << "Error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
