#include "gurobi_c++.h"
#include <iostream>

int main() {
    try {
        // 1. Initialize Gurobi environment
        GRBEnv env = GRBEnv(true);
        env.set("LogFile", "gurobi_log.log");
        env.start();

        // 2. Create a new model
        GRBModel model = GRBModel(env);

        // 3. Add decision variables (x, y >= 0)
        GRBVar x = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "x");
        GRBVar y = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "y");

        // 4. Set objective: maximize 3x + 4y
        model.setObjective(3 * x + 4 * y, GRB_MAXIMIZE);

        // 5. Add constraints
        model.addConstr(2 * x + y <= 8, "c0");
        model.addConstr(x + 2 * y <= 8, "c1");

        // 6. Optimize the model
        model.optimize();

        // 7. Display results
        if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
            std::cout << "Optimal objective value: "
                      << model.get(GRB_DoubleAttr_ObjVal) << std::endl;
            std::cout << "x = " << x.get(GRB_DoubleAttr_X)
                      << ", y = " << y.get(GRB_DoubleAttr_X) << std::endl;
        } else {
            std::cout << "Optimization did not find an optimal solution."
                      << std::endl;
        }

    } catch (GRBException &e) {
        std::cerr << "Gurobi error code: " << e.getErrorCode() << std::endl;
        std::cerr << e.getMessage() << std::endl;
    } catch (...) {
        std::cerr << "Unknown error during optimization." << std::endl;
    }

    return 0;
}
