#include <Eigen/Dense>
#include <iostream>

#include "opt_problem.hpp"

// example sqp
// class MySqpProblem : public OptProblem
// {

// public:
//     ~MySqpProblem() override {}

//     double objective(const VectorXd& dec_var) override
//     {
//         return 100.0*dec_var(0)*dec_var(0) + 20.0*dec_var(1)*dec_var(1) + 10.0*dec_var(2)*dec_var(2);
//     }

//     void eqCons(VectorXd& cons_val, const VectorXd& dec_var) const override
//     {
//         if (cons_val.rows()<NUM_EQ_CONS)
//         {
//             cons_val.resize(NUM_EQ_CONS, 1);
//         }

//         cons_val(0) = 5.0*dec_var(0) - 1;
//     }

//     void ineqCons(VectorXd& cons_val, const VectorXd& dec_var) const override
//     {
//         if (cons_val.rows()<NUM_INEQ_CONS)
//         {
//             cons_val.resize(NUM_INEQ_CONS, 1);
//         }

//         cons_val(0) = 5.0*dec_var(0) - 10;
//         cons_val(1) = dec_var(1) - 1;
//         cons_val(2) = dec_var(1) + 1;
//         cons_val(3) = dec_var(2) - 2;
//     }

//     static constexpr int NUM_DEC_VARS = 3;
//     static constexpr int NUM_EQ_CONS = 1;
//     static constexpr int NUM_INEQ_CONS = 4;
// };

class MySqpProblem : public SqpProblem
{
public:
    MySqpProblem(const int max_iter_, const double conv_tol_) : 
        SqpProblem(max_iter_, conv_tol_, num_dec_vars_, num_eq_cons_, num_ineq_cons_) {}

    ~MySqpProblem() override {}

    double objective() override
    {
        return 10.0*dec_var(0)*dec_var(0) + 20.0*dec_var(1)*dec_var(1) + 10.0*dec_var(2)*dec_var(2);
    }

    void eqCons() override
    {
        eq_cons_val(0) = 5.0*dec_var(0) - 1;
    }

    void ineqCons() override
    {
        ineq_cons_val(0) = dec_var(2) - 2;
    }

    void gradLagrangian() override
    {
        // w.r.t. decision vars
        grad_lagrangian(0) = 2.0*10.0*dec_var(0) + eq_cons_mult(0)*5.0;
        grad_lagrangian(1) = 2.0*20.0*dec_var(1);
        grad_lagrangian(2) = 2.0*10.0*dec_var(2);

        // w.r.t. equality constraint multipliers
        grad_lagrangian(3) = eq_cons_val(0);

        // w.r.t. inequality constraint multipliers
        // grad_lagrangian(4) = ineq_cons_val(0);
    }

    // void hessianLagrangian() override
    // {
    //     MatrixXd eps(hessian_lagrangian.rows(), hessian_lagrangian.rows());
    //     eps.setIdentity();
    //     eps = eps*1e-15;
    //     hessian_lagrangian.setZero();
    //     hessian_lagrangian(0,0) = 2.0*10.0; // ddL/dx1^2
    //     hessian_lagrangian(1,0) = 2.0*20.0;
    //     hessian_lagrangian(2,0) = 2.0*10.0;
    //     hessian_lagrangian(3,0) = 5.0;
    //     hessian_lagrangian(0,1) = 2.0*20.0;
    //     hessian_lagrangian(0,2) = 2.0*10.0;
    //     hessian_lagrangian(0,3) = 5.0;
    //     hessian_lagrangian = hessian_lagrangian + eps;
    //     std::cout << hessian_lagrangian << std::endl;
    //     inv_hessian_lagrangian = hessian_lagrangian.inverse();
    // }

private:
    static const int num_dec_vars_ = 3;
    static const int num_eq_cons_ = 1;
    static const int num_ineq_cons_ = 0;
    static const int bfgs_eps; // used to prevent div by 0 when calculating the inv hessian apprx from bfgs
};

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

    // std::cout << "Equality constraints values: " << prob.eq_cons_val<< std::endl;
    // std::cout << "Inequality constraints values: " << prob.ineq_cons_val << std::endl;
    // std::cout << "Lagrangian: " << prob.lagrangian << std::endl;

    return 0;
}