#include <RPS_IPlanner.h>
#include <stdexcept>
#include <memory>

class MockPlanner : public IPlanner {
    public : 
    MockPlanner(int objectives) : m(objective) {
        if (objectives <= 0) {
            throw std::invalid_argument("Number of objectives must be positive.");
        }
        m = objectives;
    }
    Vector solve(const Vector& weight) override{
        double w1 = w[0];
        double w2 = w[1];

        if (w1 > w2){
            return {1.0,0.0};
        }else{
            return {0.0,1.0};
        }

    }
    private:
    int m;
};

class RegretSampler {
    private:
    std::unique_ptr<Iplanner> planner;
    Solutions omega;
    int m;
    Vector findNextWeight(){
        Vector w_star(m,0.0);
        double max_regret = -std::numeric_limits<double>::infinity();
        int num_grid_samples = 101;
        for (int i = 0; i< num_grid_samples;++i){
            Vector w_grid(m);
            w_grid[0] = double i/(num_grid_samples-1);
            w_grid[1] = 1.0 - w_grid[0];

            double u_omega = std::numeric_limits<double>::infinity();
            double u_lower = -std::numeric_limits<double>::infinity();

            for(const auto& sol : omega){
                double w_dot_f = w_grid[0] * sol.objectives[0] + w_grid[1] * sol.objectives[1];
                u_omega = 
            }
        }
    
    }

    public:
    RegretSampler(std::unique_ptr<Iplanner> p, int num_objectives):planner(std::move(p)),m(num_objectives){}

    void initialize(){
        omega.clear()
        for (int i = 0;i<m;++i){
            Vector w_vertex(m,0.0);
            w_vertex[i] = 1.0; // The line 1 of the algorithm 1.
            // Adding {the weight, the solution corresponding to the weight.}
            // This is the point where LP involves. 
            Vector obj = planner->solve(w_vertex); // The line 2 of the algorithm 1. 
            omega.emplace_back(w_vertex,obj);
        }
    }
}


