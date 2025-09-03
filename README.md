# optimization_cpp
A C++17 library for solving constrained optimization problems. 

This project currently implements a [Sequential Quadratic Programming]( https://en.wikipedia.org/wiki/Sequential_quadratic_programming)  (SQP) method for solving [nonlinear programs](https://en.wikipedia.org/wiki/Nonlinear_programming)  (NLP's). On each iteration of the SQP solver, the optimization problem is approximated with a quadratic cost and linear constraints. This quadratic subproblem is then passed to a QP solver which solves for the next step in the decision variables. This is repeated until the convergence criteria are met. [OSQP](https://github.com/osqp/osqp) is used to solve the quadratic subproblems in this project using the [osqp-cpp](https://github.com/google/osqp-cpp) interface.

The interface is defined in opt_problem.hpp in the SqpProblem class. Override the functions in this class to implement your own objective and constraint functions and their hessian and gradients. Examples of this are given in example_prob.hpp/.cpp.

## Build
From the root directory of this project:
```
mkdir build
cd build
cmake ..
make
```

To run the examples (from the build directory):
```
./run_examples
```

## References

[Cornell University Computational Optimization Open Textbook](https://optimization.cbe.cornell.edu/index.php?title=Sequential_quadratic_programming)

https://en.wikipedia.org/wiki/Sequential_quadratic_programming


https://github.com/osqp/osqp

https://github.com/google/osqp-cpp