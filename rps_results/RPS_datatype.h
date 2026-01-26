/*
Implementation of "Regret based Pareto Sampling for multi objective optimization (RPS)"

Edited : Nov 7 Fri, 2025
SeungKeol Ryu

*/
#pragma once
#include <vector>
#include <cmath>

struct Solution{
    std::vector<double> weight;
    std::vector<double> objectives;
    double cost;

    Solution(const std::vector<double>& w, const std::vector<double>& obj)
        : weight(w), objectives(obj) {
        cost = 0.0;
        for (size_t i = 0; i < objectives.size(); ++i) {
            cost += weight[i] * objectives[i];
        }
    }
};

using Solutions = std::vector<Solution>;

