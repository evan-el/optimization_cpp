#ifndef OPT_PROBLEM_HPP
#define OPT_PROBLEM_HPP

#include <Eigen/Dense>

using VectorXd = Eigen::VectorXd;
using MatrixXd = Eigen::MatrixXd;

class OptProblem
{
protected:
    OptProblem() {}

public:
    virtual ~OptProblem() = 0;

    // minimum interface every optimization problem should have
    virtual double objective(const VectorXd& dec_var) const = 0;
    virtual void eq_cons(VectorXd& cons_val, const VectorXd& dec_var) const = 0;
    virtual void ineq_cons(VectorXd& cons_val, const VectorXd& dec_var) const = 0;

    // Source for construction of lagrangian, gradient, and hessian of lagrangian: 
    // https://optimization.cbe.cornell.edu/index.php?title=Sequential_quadratic_programming
    //
    // eq_cons_mult and ineq_cons_mult are the lagrange multipliers
    virtual double lagrangian(VectorXd& eq_cons_val, VectorXd& ineq_cons_val, const VectorXd& dec_var, 
        const VectorXd& eq_cons_mult, const VectorXd& ineq_cons_mult)
    {   
        eq_cons(eq_cons_val, dec_var);
        ineq_cons(ineq_cons_val, dec_var);

        return objective(dec_var) + eq_cons_mult.dot(eq_cons_val) + ineq_cons_mult.dot(ineq_cons_val);
    }

};

#endif // OPT_PROBLEM_HPP