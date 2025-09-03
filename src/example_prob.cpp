#include <Eigen/Dense>

#include "example_prob.hpp"

double ExampleSqpProblem::objective()
{
    return w1*exp(-dec_var(0)) + w2*dec_var(1)*dec_var(1) + w3*dec_var(2)*dec_var(2);
}

void ExampleSqpProblem::eqCons()
{
    eq_cons_val(0) = 5.0*dec_var(2) - 1.0;
}

void ExampleSqpProblem::ineqCons()
{
    ineq_cons_val(0) = dec_var(0) + dec_var(2) - 3.0;
}

void ExampleSqpProblem::gradObjective()
{
    grad_objective(0) = -dec_var(0)*w1*exp(-dec_var(0));
    grad_objective(1) = 2.0*w2*dec_var(1);
    grad_objective(2) = 2.0*w3*dec_var(2);
}

void ExampleSqpProblem::hessianObjective()
{
    hessian_objective(0,0) = dec_var(0)*dec_var(0)*w1*exp(-dec_var(0));
    hessian_objective(1,1) = 2.0*w2;
    hessian_objective(2,2) = 2.0*w3;
}

void ExampleSqpProblem::gradEqCons()
{
    grad_eq_cons(0,2) = 5.0;
}

void ExampleSqpProblem::gradIneqCons()
{
    grad_ineq_cons(0,0) = 1.0;
    grad_ineq_cons(0,2) = 1.0;
}

// --------------- Single Integrator -----------------
double SingleIntegratorProb::objective()
{
    double obj = 0.0;
    for (int k=0; k<num_time_steps*(num_states+num_ctrl); k=k+num_states+num_ctrl)
    {
        obj += w1*dec_var(k)*dec_var(k) + w2*dec_var(k+1)*dec_var(k+1);
    }
    return obj;
}

void SingleIntegratorProb::eqCons()
{
    eq_cons_val(0) = dec_var(0) - v_0;
    int ind = 1;
    for (int k=0; k<(num_time_steps-1)*(num_states+num_ctrl); k=k+num_states+num_ctrl)
    {
        eq_cons_val(ind) = dec_var(k+2) - dec_var(k) - dec_var(k+1)*dt;
        ind++;
    }
    // std::cout << "eq_cons_val: " << eq_cons_val << std::endl;
}

void SingleIntegratorProb::ineqCons()
{
    int ind = 0;
    for (int k=0; k<num_time_steps*(num_states+num_ctrl); k = k+num_states+num_ctrl)
    {
        ineq_cons_val(ind)   = accel_min - dec_var(k+1);
        ineq_cons_val(ind+1) = dec_var(k+1) - accel_max;
        ind += 2;
    }
    // std::cout << "ineq_cons_val: " << ineq_cons_val << std::endl;
}

void SingleIntegratorProb::gradObjective()
{
    int ind = 0;
    for (int k=0; k<num_time_steps*(num_states+num_ctrl); k = k+num_states+num_ctrl)
    {
        grad_objective(ind)   = 2.0*w1*(dec_var(k) - v_des);
        grad_objective(ind+1) = 2.0*w2*dec_var(k+1);
        ind += 2;
    }
    // std::cout << "dec_var: " << dec_var << std::endl;
    // std::cout << "grad_objective: " << grad_objective << std::endl;
}

void SingleIntegratorProb::hessianObjective()
{
    for (int k=0; k<num_dec_vars; k = k+num_states+num_ctrl)
    {
        hessian_objective(k,k) = 2.0*w1;
        hessian_objective(k+1,k+1) = 2.0*w2;
    }
    // std::cout << "hessian_objective: " << hessian_objective << std::endl;
}

void SingleIntegratorProb::gradEqCons()
{
    grad_eq_cons(0,0) = 1.0;
    int ind = 1;
    for (int k=0; k<(num_time_steps-1)*(num_states+num_ctrl); k = k+num_states+num_ctrl)
    {
        grad_eq_cons(ind,k) = -1.0;
        grad_eq_cons(ind,k+1) = -dt;
        grad_eq_cons(ind,k+2) = 1.0;
        ind++;
    }
    // std::cout << "grad_eq_cons: " << grad_eq_cons << std::endl;
}

void SingleIntegratorProb::gradIneqCons()
{
    int ind = 0;
    for (int k=0; k<num_time_steps*(num_states+num_ctrl); k = k+num_states+num_ctrl)
    {
        // std::cout << "grad_ineq_cons: " << grad_ineq_cons << std::endl;
        grad_ineq_cons(ind,k+1) = -1.0;
        grad_ineq_cons(ind+1,k+1) = 1.0;
        ind += 2;
    }
    // std::cout << "grad_ineq_cons: " << grad_ineq_cons << std::endl;
}