#include <Eigen/Dense>
#include <iostream>
#include <chrono>

#include "example_prob.hpp"

void runOptimalControlExample()
{
    const int max_iter = 1; // if the problem is a quadratic program, you can set max_iter to 1.
    const double convergence_tol = 1.0e-3;

    SingleIntegratorProb prob = SingleIntegratorProb(max_iter, convergence_tol);
    VectorXd dec_var_guess(prob.num_dec_vars_,1);
    dec_var_guess.setZero();

    prob.setDecVar(dec_var_guess);
    auto start = std::chrono::high_resolution_clock::now();
    prob.solve();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Elapsed time: " << duration.count()*1e-6 << " seconds" << std::endl;

    VectorXd sol = prob.getDecVar();

    std::cout << "Solution: " << sol << std::endl;
}

void runSqpExample()
{
    const int max_iter = 100;
    const double convergence_tol = 1.0e-3;
    ExampleSqpProblem ex_prob = ExampleSqpProblem(max_iter, convergence_tol);
    VectorXd example_dec_var_guess(ex_prob.num_dec_vars_,1);
    example_dec_var_guess.setZero();

    ex_prob.setDecVar(example_dec_var_guess);
    auto ex_start = std::chrono::high_resolution_clock::now();
    ex_prob.solve();
    auto ex_end = std::chrono::high_resolution_clock::now();
    auto ex_duration = std::chrono::duration_cast<std::chrono::microseconds>(ex_end - ex_start);
    std::cout << "Elapsed time: " << ex_duration.count()*1e-6 << " seconds" << std::endl;

    VectorXd ex_sol = ex_prob.getDecVar();
    std::cout << "Solution: " << ex_sol << std::endl;
}

int main()
{
    runOptimalControlExample();
    runSqpExample();

    return 0;
}