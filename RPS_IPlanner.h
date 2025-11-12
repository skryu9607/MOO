#pragma once
#include "RPS_datatype.h"

/*
Virtual : interface class for planner. 
I know what it does, but I don't know how to make it doing it. 


*/


using Vector = std::vector<double>;
class IPlanner {
    public: 
    virtual ~IPlanner() = default;
    virtual Vector solve(const Vector& weight) = 0;
};
