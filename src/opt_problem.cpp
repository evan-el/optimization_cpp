#include "opt_problem.hpp"

OptProblem::~OptProblem() {}

double OptProblem::lagrangian()
{   
    eqCons();
    ineqCons();
    return objective() + eq_cons_mult.dot(eq_cons_val) + ineq_cons_mult.dot(ineq_cons_val);
}

SqpProblem::SqpProblem(int max_iter_, double conv_tol_, int num_dec_vars_, 
    int num_eq_cons_, int num_ineq_cons_, double bfgs_eps_, bool use_bfgs_apprx_) : 
    max_iter(max_iter_),
    conv_tol(conv_tol_),
    num_dec_vars(num_dec_vars_),
    num_eq_cons(num_eq_cons_),
    num_ineq_cons(num_ineq_cons_),
    bfgs_eps(bfgs_eps_),
    use_bfgs_apprx(use_bfgs_apprx_)
{
    const int dim_lagrangian = num_dec_vars + num_eq_cons + num_ineq_cons;
    grad_lagrangian.resize(dim_lagrangian,1);
    grad_lagrangian_prev.resize(dim_lagrangian,1);
    hessian_lagrangian.resize(dim_lagrangian, dim_lagrangian);
    inv_hessian_lagrangian.resize(dim_lagrangian, dim_lagrangian);
    step.resize(dim_lagrangian,1);
  
    inv_hessian_lagrangian.setZero();
    hessian_lagrangian.setZero();
    grad_lagrangian_prev.setZero();
    grad_lagrangian.setZero();
    step.setZero();

    dec_var.resize(num_dec_vars);
    eq_cons_val.resize(num_eq_cons);
    ineq_cons_val.resize(num_ineq_cons);
    eq_cons_mult.resize(num_eq_cons);
    ineq_cons_mult.resize(num_ineq_cons);

    dec_var.setZero();
    eq_cons_val.setZero();
    ineq_cons_val.setZero();
    eq_cons_mult.setZero();
    ineq_cons_mult.setZero();
}

SqpProblem::~SqpProblem() {}

void SqpProblem::hessianLagrangian()
{
    bfgsHessianApprx();
}

void SqpProblem::bfgsHessianApprx()
{
    MatrixXd identity = MatrixXd::Identity(inv_hessian_lagrangian.rows(), inv_hessian_lagrangian.rows());
    VectorXd delta_grad = grad_lagrangian - grad_lagrangian_prev;
    double rho = 1.0/(delta_grad.transpose()*step + bfgs_eps);
    inv_hessian_lagrangian = (identity-rho*step*delta_grad.transpose())*inv_hessian_lagrangian*(identity-rho*delta_grad*step.transpose()) + rho*step*step.transpose();
    std::cout << "grad: " << grad_lagrangian << std::endl;
    std::cout << "delta_grad: " << delta_grad << std::endl;
    std::cout << "step: " << step << std::endl;
    std::cout << inv_hessian_lagrangian << std::endl;
}

void SqpProblem::solve()
{
    eqCons();
    ineqCons();
    gradLagrangian();
    if (use_bfgs_apprx)
    {
        inv_hessian_lagrangian.setIdentity();
    }
    else
    {
        hessianLagrangian();
    }

    int k = 0;
    while (!isConverged() && k<max_iter)
    {
        step = -inv_hessian_lagrangian*grad_lagrangian;
        dec_var = dec_var + step(Eigen::seq(0,num_dec_vars-1));
        eq_cons_mult = eq_cons_mult + step(Eigen::seq(num_dec_vars,num_dec_vars+num_eq_cons-1));
        ineq_cons_mult = ineq_cons_mult + step(Eigen::seq(num_dec_vars+num_eq_cons,num_dec_vars+num_eq_cons+num_ineq_cons-1));
        
        std::cout << "step: " << step << std::endl;
        std::cout << "eq_cons_mult: " << eq_cons_mult << std::endl;
        std::cout << "ineq_cons_mult: " << ineq_cons_mult << std::endl;
        std::cout << "eq_cons_val: " << eq_cons_val << std::endl;
        std::cout << "ineq_cons_val: " << ineq_cons_val << std::endl;
        std::cout << "dec_var: " << dec_var << std::endl;
        std::cout << "objective: " << objective() << std::endl;
        
        eqCons();
        ineqCons();
        grad_lagrangian_prev = grad_lagrangian;
        gradLagrangian();

        std::cout << "grad_lagrangian_prev: " << grad_lagrangian_prev << std::endl;
        std::cout << "grad_lagrangian: " << grad_lagrangian << std::endl;
        
        hessianLagrangian();
        k++;
    }
}

bool SqpProblem::isConverged()
{
    return (grad_lagrangian.norm()<conv_tol && eq_cons_val.norm()<conv_tol && ineq_cons_val.norm()<conv_tol);
}