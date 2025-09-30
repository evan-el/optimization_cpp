#ifndef OPT_PROBLEM_HPP
#define OPT_PROBLEM_HPP

#include <Eigen/Dense>
#include <iostream>

#include "osqp++.h"

using VectorXd = Eigen::VectorXd;
using MatrixXd = Eigen::MatrixXd;

class OptProblem
{
protected:
    OptProblem() {}
    VectorXd dec_var;
    VectorXd eq_cons_val;
    VectorXd ineq_cons_val;
    VectorXd eq_cons_mult;
    VectorXd ineq_cons_mult;

public:
    virtual ~OptProblem() = 0;

    // minimum interface every optimization problem should have
    virtual double objective() = 0;
    virtual void eqCons() = 0;
    virtual void ineqCons() = 0;

    VectorXd getDecVar() const
    {
        return dec_var;
    }

    void setDecVar(VectorXd& dec_var_)
    {
        dec_var = dec_var_;
    }

    // Source for construction of lagrangian, gradient, and hessian of lagrangian: 
    // https://optimization.cbe.cornell.edu/index.php?title=Sequential_quadratic_programming
    //
    // eq_cons_mult and ineq_cons_mult are the lagrange multipliers
    virtual double lagrangian();

    virtual void solve() = 0;

};
class SqpProblem : public OptProblem
{
protected:
    SqpProblem(int max_iter_, double conv_tol_, int num_dec_vars_, 
        int num_eq_cons_, int num_ineq_cons_, double bfgs_eps_, bool use_bfgs_apprx_);

    VectorXd grad_objective;
    VectorXd grad_objective_prev;

    MatrixXd hessian_objective;
    MatrixXd inv_hessian_objective;

    MatrixXd grad_eq_cons;
    MatrixXd grad_ineq_cons;

    VectorXd step;

    VectorXd dec_var_prev;
   
    const int max_iter;
    const double conv_tol;
    const int num_dec_vars;
    const int num_eq_cons;
    const int num_ineq_cons;
    const double bfgs_eps;
    const bool use_bfgs_apprx;

    osqp::OsqpInstance osqp_instance;
    osqp::OsqpSettings osqp_settings;
    osqp::OsqpSolver osqp_solver;

public:
    virtual ~SqpProblem() = 0;
    
    void reset();

    virtual void gradObjective() = 0;

    // Calculates hessian_lagrangian and inv_hessian_lagrangian matrices
    virtual void hessianObjective();

    virtual void gradEqCons() = 0;

    virtual void gradIneqCons() = 0;

    // Calculates hessian_lagrangian and inv_hessian_lagrangian matrices using BFGS approximation.
    //
    // Sources on BFGS Hessian apprx: 
    // https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm
    // https://towardsdatascience.com/bfgs-in-a-nutshell-an-introduction-to-quasi-newton-methods-21b0e13ee504/
    void bfgsHessianApprx();

    virtual void solve() override;

    virtual bool isConverged();

};

#endif // OPT_PROBLEM_HPP