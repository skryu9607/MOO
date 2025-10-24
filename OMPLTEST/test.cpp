#include<memory>
#include<random>
#include<iostream>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/base/spaces/SE2StateSpace.h>
namespace ob = ompl::base;
namespace og = ompl::geometric;

bool isStatevalid(const ob::State *state){
    return true;
}

void planWithSimpleSetup(){
    auto space(std::make_shared<ob::SE2StateSpace>());
    ob::RealVectorBounds bounds(2);
    bounds.setLow(-1);
    bounds.setHigh(1);

    space -> setBounds(bounds);

    ompl::geometric::SimpleSetup ss(space);
    ss.setStateValidityChecker([](const ob::State *state){
        return isStatevalid(state);
    });
    ob::ScopedState<> start(space);
    start.random();

    ob::ScopedState<> goal(space);
    static std::mt19937 gen(std::random_device{}());
    goal.random();
    ss.setStateAndGoalState(start,goal);
    // Setup is finished.

    ob::PlannerStatus solved = ss.solve(1.0);
    if (solved)
    {
        std::cout << "Found solution:" << std::endl;
        // print the path to screen
        ss.simplifySolution();
        ss.getSolutionPath().print(std::cout);
    }
}
int main(){
    planWithSimpleSetup();
    return 0;
}

