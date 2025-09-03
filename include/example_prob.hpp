#include <Eigen/Dense>

#include "opt_problem.hpp"

class MySqpProblem : public SqpProblem
{
public:
    MySqpProblem(const int max_iter_, const double conv_tol_) : 
        SqpProblem(max_iter_, conv_tol_, num_dec_vars_, 
            num_eq_cons_, num_ineq_cons_, bfgs_eps_, use_bfgs_apprx_) {}

    ~MySqpProblem() override {}

    double objective() override;

    void eqCons() override;

    void ineqCons() override;

    void gradObjective() override;

    // must overload hessianLagrangian if you don't want to use bfgs apprx
    void hessianObjective() override;

    void gradEqCons() override;

    void gradIneqCons() override;

private:
    static constexpr int num_dec_vars_ = 3;
    static constexpr int num_eq_cons_ = 1;
    static constexpr int num_ineq_cons_ = 1;
    static constexpr double bfgs_eps_ = 1.0e-20;
    static constexpr bool use_bfgs_apprx_ = false;

    double w1 = 20.0;
    double w2 = 1.0;
    double w3 = 1.0;
};