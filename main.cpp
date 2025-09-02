#include <Eigen/Dense>
#include <iostream>

#include "opt_problem.hpp"

// example sqp
class MySqpProblem : public OptProblem
{

public:
    ~MySqpProblem() override {}

    double objective(const VectorXd& dec_var) const override
    {
        return 100.0*dec_var(0)*dec_var(0) + 20.0*dec_var(1)*dec_var(1) + 10.0*dec_var(2)*dec_var(2);
    }

    void eq_cons(VectorXd& cons_val, const VectorXd& dec_var) const override
    {
        if (cons_val.rows()<NUM_EQ_CONS)
        {
            cons_val.resize(NUM_EQ_CONS, 1);
        }

        cons_val(0) = 5.0*dec_var(0) - 1;
    }

    void ineq_cons(VectorXd& cons_val, const VectorXd& dec_var) const override
    {
        if (cons_val.rows()<NUM_INEQ_CONS)
        {
            cons_val.resize(NUM_INEQ_CONS, 1);
        }

        cons_val(0) = 5.0*dec_var(0) - 10;
        cons_val(1) = dec_var(1) - 1;
        cons_val(2) = dec_var(1) + 1;
        cons_val(3) = dec_var(2) - 2;
    }

    static constexpr int NUM_DEC_VARS = 3;
    static constexpr int NUM_EQ_CONS = 1;
    static constexpr int NUM_INEQ_CONS = 4;
};

int main()
{
    MySqpProblem prob = MySqpProblem();
    VectorXd dec_vars_init(MySqpProblem::NUM_DEC_VARS,1);
    dec_vars_init << 1.0, 4.0, 2.0;
    double obj = prob.objective(dec_vars_init);

    VectorXd eq_cons_val;
    VectorXd ineq_cons_val;
    VectorXd eq_cons_mult(MySqpProblem::NUM_EQ_CONS,1);
    VectorXd ineq_cons_mult(MySqpProblem::NUM_INEQ_CONS,1);

    double lagrangian = prob.lagrangian(eq_cons_val,
        ineq_cons_val,
        dec_vars_init,
        eq_cons_mult,
        ineq_cons_mult);

    std::cout << "Equality constraints values: " << eq_cons_val<< std::endl;
    std::cout << "Inequality constraints values: " << ineq_cons_val << std::endl;
    std::cout << "Lagrangian: " << lagrangian << std::endl;

    return 0;
}