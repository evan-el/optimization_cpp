#include <Eigen/Dense>
#include <iostream>

#include "example_prob.hpp"

// TODO: implement a QP solver to solve each quadratic subproblem step
int main()
{
    const int max_iter = 10000;
    const double convergence_tol = 1.0e-3;

    VectorXd dec_var_guess(3,1);
    dec_var_guess.setZero();

    MySqpProblem prob = MySqpProblem(max_iter, convergence_tol);
    prob.setDecVar(dec_var_guess);
    prob.solve();
    
    VectorXd sol = prob.getDecVar();

    std::cout << "Solution: " << sol << std::endl;

    return 0;
}