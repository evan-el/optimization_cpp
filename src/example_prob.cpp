#include <Eigen/Dense>

#include "example_prob.hpp"

double MySqpProblem::objective()
{
    return w1*dec_var(0)*dec_var(0) + w2*dec_var(1)*dec_var(1) + w3*dec_var(2)*dec_var(2);
}

void MySqpProblem::eqCons()
{
    eq_cons_val(0) = 5.0*dec_var(0) - 1;
}

void MySqpProblem::ineqCons()
{

}

void MySqpProblem::gradLagrangian()
{
    // w.r.t. decision vars
    grad_lagrangian(0) = 2.0*w1*dec_var(0) + eq_cons_mult(0)*5.0;
    grad_lagrangian(1) = 2.0*w2*dec_var(1);
    grad_lagrangian(2) = 2.0*w3*dec_var(2);

    // w.r.t. equality constraint multipliers
    grad_lagrangian(3) = eq_cons_val(0);
}