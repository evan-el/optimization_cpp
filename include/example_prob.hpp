#include <Eigen/Dense>

#include "opt_problem.hpp"

class ExampleSqpProblem : public SqpProblem
{
public:
    ExampleSqpProblem(const int max_iter_, const double conv_tol_) : 
        SqpProblem(max_iter_, conv_tol_, num_dec_vars_, 
            num_eq_cons_, num_ineq_cons_, bfgs_eps_, use_bfgs_apprx_) {}

    ~ExampleSqpProblem() override {}

    double objective() override;

    void eqCons() override;

    void ineqCons() override;

    void gradObjective() override;

    // must overload hessianLagrangian if you don't want to use bfgs apprx
    void hessianObjective() override;

    void gradEqCons() override;

    void gradIneqCons() override;

    static constexpr int num_dec_vars_ = 3;
    static constexpr int num_eq_cons_ = 1;
    static constexpr int num_ineq_cons_ = 1;
    static constexpr double bfgs_eps_ = 1.0e-20;
    static constexpr bool use_bfgs_apprx_ = false;

private:

    double w1 = 20.0;
    double w2 = 1.0;
    double w3 = 1.0;
};

class SingleIntegratorProb : public SqpProblem
{
public:
    SingleIntegratorProb(const int max_iter_, const double conv_tol_) : 
        SqpProblem(max_iter_, conv_tol_, num_dec_vars_, 
            num_eq_cons_, num_ineq_cons_, bfgs_eps_, use_bfgs_apprx_) {}

    ~SingleIntegratorProb() override {}

    double objective() override;

    void eqCons() override;

    void ineqCons() override;

    void gradObjective() override;

    // must overload hessianLagrangian if you don't want to use bfgs apprx
    void hessianObjective() override;

    void gradEqCons() override;

    void gradIneqCons() override;


    static constexpr int num_time_steps = 2000;
    static constexpr double dt = 0.1;

    static constexpr int num_states = 1;
    static constexpr int num_ctrl = 1;

    static constexpr int num_dec_vars_ = (num_states+num_ctrl)*num_time_steps;
    static constexpr int num_eq_cons_ = num_states*(num_time_steps-1) + num_states;
    static constexpr int num_ineq_cons_ = 2*num_ctrl*num_time_steps;

private:
    static constexpr double bfgs_eps_ = 1.0e-20;
    static constexpr bool use_bfgs_apprx_ = false;

    double w1 = 2.0;
    double w2 = 1.0;
    double v_des = 20.0;
    double v_0 = 5.0;
    double accel_min = -2.0;
    double accel_max = 0.5;
};