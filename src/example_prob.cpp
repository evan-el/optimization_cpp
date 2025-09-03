#include <Eigen/Dense>

#include "example_prob.hpp"

double MySqpProblem::objective()
{
    return w1*exp(-dec_var(0)) + w2*dec_var(1)*dec_var(1) + w3*dec_var(2)*dec_var(2);
}

void MySqpProblem::eqCons()
{
    eq_cons_val(0) = 5.0*dec_var(2) - 1.0;
}

void MySqpProblem::ineqCons()
{
    ineq_cons_val(0) = dec_var(0) + dec_var(2) - 3.0;
}

void MySqpProblem::gradObjective()
{
    grad_objective(0) = -dec_var(0)*w1*exp(-dec_var(0));
    grad_objective(1) = 2.0*w2*dec_var(1);
    grad_objective(2) = 2.0*w3*dec_var(2);
}

void MySqpProblem::hessianObjective()
{
    hessian_objective(0,0) = dec_var(0)*dec_var(0)*w1*exp(-dec_var(0));
    hessian_objective(1,1) = 2.0*w2;
    hessian_objective(2,2) = 2.0*w3;
}

void MySqpProblem::gradEqCons()
{
    grad_eq_cons(0,2) = 5.0;
}

void MySqpProblem::gradIneqCons()
{
    grad_ineq_cons(0,0) = 1.0;
    grad_ineq_cons(0,2) = 1.0;
}