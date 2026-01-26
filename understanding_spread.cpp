/*
Undestanding the metric, spread value. 
Three big questions to answer :
1. Is it monotonically increasing with respect to number of points sampled?

*/

#include<stdio.h>


struct State {
    double x;
    double y;
};
double spread(const std::vector<double>& d_i, const std::vector<double>& extremes){
    int N = d_i.size();
    if (N == 0) return 1.0;
    double d_f = extremes[0];
    double d_l = extremes[1];
    double d_bar = sum(d_i)/N;
};
double sum(const std::vector<double>& vec){
    double s = 0.0;
    for (double v : vec) s += v;
    return s;
};
double 
int main(){
    double extremes[2] = {1.0,3.0};
    std::vector<double> d_i1 = {0.5, 1.0, 1.5, 2.0};



}


