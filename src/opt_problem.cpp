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
    grad_objective.resize(num_dec_vars,1);
    grad_objective_prev.resize(num_dec_vars,1);
    hessian_objective.resize(num_dec_vars, num_dec_vars);
    inv_hessian_objective.resize(num_dec_vars, num_dec_vars);
    step.resize(num_dec_vars,1);
    grad_eq_cons.resize(num_eq_cons, num_dec_vars);
    grad_ineq_cons.resize(num_ineq_cons, num_dec_vars);

    grad_objective.setZero();
    grad_objective_prev.setZero();
    hessian_objective.setZero();
    inv_hessian_objective.setZero();
    grad_eq_cons.setZero();
    grad_ineq_cons.setZero();
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

    osqp_instance.upper_bounds.resize(num_eq_cons+num_ineq_cons,1);
    osqp_instance.lower_bounds.resize(num_eq_cons+num_ineq_cons,1);
    
    osqp_settings.verbose = false;
}

SqpProblem::~SqpProblem() {}

void SqpProblem::hessianObjective()
{
    bfgsHessianApprx();
}

void SqpProblem::bfgsHessianApprx()
{
    MatrixXd identity = MatrixXd::Identity(inv_hessian_objective.rows(), inv_hessian_objective.rows());
    VectorXd delta_grad = grad_objective - grad_objective_prev;
    double rho = 1.0/(delta_grad.transpose()*step + bfgs_eps);

    hessian_objective = hessian_objective + ((delta_grad*delta_grad.transpose())/rho) - (hessian_objective*step*step.transpose()*hessian_objective.transpose())/(step.transpose()*hessian_objective*step + bfgs_eps);

    inv_hessian_objective = (identity-rho*step*delta_grad.transpose())*inv_hessian_objective*(identity-rho*delta_grad*step.transpose()) + rho*step*step.transpose();

    // std::cout << "grad: " << grad_objective << std::endl;
    // std::cout << "delta_grad: " << delta_grad << std::endl;
    // std::cout << "step: " << step << std::endl;
}

void SqpProblem::solve()
{
    eqCons();
    ineqCons();
    gradObjective();
    gradEqCons();
    gradIneqCons();
    if (use_bfgs_apprx)
    {
        hessian_objective.setIdentity();
        inv_hessian_objective.setIdentity();
    }
    else
    {
        hessianObjective();
    }

    int k = 0;
    while (!isConverged() && k<max_iter)
    {
        // std::cout << "using bfgs: " << use_bfgs_apprx << std::endl;
        // std::cout << hessian_objective << std::endl;
        osqp_instance.objective_matrix = hessian_objective.sparseView();
        osqp_instance.objective_vector = grad_objective.sparseView();

        MatrixXd cons_mat_dense(num_eq_cons+num_ineq_cons, num_dec_vars);
        cons_mat_dense.block(0, 0, num_eq_cons, num_dec_vars) = grad_eq_cons;
        cons_mat_dense.block(num_eq_cons, 0, num_ineq_cons, num_dec_vars) = grad_ineq_cons;

        osqp_instance.constraint_matrix = cons_mat_dense.sparseView();
        osqp_instance.lower_bounds.segment(0, num_eq_cons) = -eq_cons_val;
        osqp_instance.upper_bounds.segment(0, num_eq_cons) = -eq_cons_val;
        osqp_instance.lower_bounds.segment(num_eq_cons, num_ineq_cons).setConstant(-std::numeric_limits<double>::infinity());
        osqp_instance.upper_bounds.segment(num_eq_cons, num_ineq_cons) = -ineq_cons_val;

        auto osqp_solver_status = osqp_solver.Init(osqp_instance, osqp_settings);
        // std::cout << osqp_solver_status << std::endl;
        osqp::OsqpExitCode exit_code = osqp_solver.Solve();
        step = osqp_solver.primal_solution();
        VectorXd mult = osqp_solver.dual_solution();
        
        // std::cout << step << std::endl;
        // std::cout << mult << std::endl;

        dec_var = dec_var + step(Eigen::seq(0,num_dec_vars-1));
        eq_cons_mult = mult(Eigen::seq(0,num_eq_cons-1));
        ineq_cons_mult = mult(Eigen::seq(num_eq_cons,num_eq_cons+num_ineq_cons-1));
        
        // std::cout << "step: " << step << std::endl;
        // std::cout << "eq_cons_mult: " << eq_cons_mult << std::endl;
        // std::cout << "ineq_cons_mult: " << ineq_cons_mult << std::endl;
        // std::cout << "eq_cons_val: " << eq_cons_val << std::endl;
        // std::cout << "ineq_cons_val: " << ineq_cons_val << std::endl;
        // std::cout << "dec_var: " << dec_var << std::endl;
        // std::cout << "objective: " << objective() << std::endl;
        
        eqCons();
        ineqCons();
        grad_objective_prev = grad_objective;
        gradObjective();
        gradEqCons();
        gradIneqCons();

        // std::cout << "grad_lagrangian_prev: " << grad_objective_prev << std::endl;
        // std::cout << "grad_lagrangian: " << grad_objective << std::endl;
        
        hessianObjective();
        k++;
    }
    // std::cout << "iterations: " << k << std::endl;
}

bool SqpProblem::isConverged()
{
    // TODO: Could check the norm of the gradient of lagrangian instead of step. This could reduce extra iterations.
    // std::cout << "step norm: " << step.norm() << std::endl;
    return (step.norm()<conv_tol && eq_cons_val.norm()<conv_tol && (ineq_cons_val.array()<conv_tol).all());
}