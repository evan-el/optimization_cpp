#ifndef OPT_PROBLEM_HPP
#define OPT_PROBLEM_HPP

#include <Eigen/Dense>
#include <iostream>

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
    SqpProblem(int max_iter_, double conv_tol_, int num_dec_vars_, int num_eq_cons_, int num_ineq_cons_);
    
    // The size on all the VectorXd's and MatrixXd's needs to be set in the constructor of the derived class.
    VectorXd grad_lagrangian;
    VectorXd grad_lagrangian_prev;

    MatrixXd hessian_lagrangian;
    MatrixXd inv_hessian_lagrangian;

    VectorXd step;

    VectorXd dec_var_prev;
   
    const int max_iter;
    const double conv_tol;
    const int num_dec_vars;
    const int num_eq_cons;
    const int num_ineq_cons;


public:
    virtual ~SqpProblem() = 0;
    
    virtual void gradLagrangian() = 0;

    // Sources on BFGS Hessian apprx: 
    // https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm
    // https://towardsdatascience.com/bfgs-in-a-nutshell-an-introduction-to-quasi-newton-methods-21b0e13ee504/
    virtual void hessianLagrangian();

    virtual void solve() override;

    virtual bool isConverged();

};

#endif // OPT_PROBLEM_HPP