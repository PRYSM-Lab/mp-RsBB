# RsBB/ mp-RsBB

This repository contains implementations of robust optimization algorithms for nonconvex pooling problems formulated in [Pyomo](https://github.com/Pyomo/pyomo). The repository includes the source code for the `RsBB` and `mp-RsBB` algorithms as presented in 
_A. Marousi, V. M. Charitopoulos, 2025. Global and robust optimization for non-convex quadratic programs. [arXiv, 2503.07310](https://arxiv.org/abs/2503.07310)_ and _A. Marousi, V. M. Charitopoulos, 2026. Accelerated robust spatial Branch-and-Bound algorithm via multi-parametric programming: Application to pooling problems. Under review_, as well as implementations of benchmark solution approaches used in the accompanying computational studies. Both algorithms rely on the [`pooling-network`](https://github.com/cog-imperial/pooling-network) library to build the pooling problem instances as [Pyomo](https://github.com/Pyomo/pyomo) optimization problems. 

## RsBB
The Robust spatial-Branch-and-Bound (`RsBB`) algorithm is an integration of the robust cutting planes and spatial-Branch-and-Bound algorithms implemented for the pq-formulation of benchmark pooling problems. 


## mp-RsBB
The multi-parametric Robust spatial-Branch-and-Bound (`mp-RsBB`) algorithm replaces the online evaluation of the lower-level problem in the robust cutting plane iterations with a multi-parametric problem (`mpPooling`) that is solved offline once. The `mpPooling` is solved using the [`PPOPT`](https://github.com/TAMUparametric/PPOPT) solver and the obtained multi-parametric solutions are passed in `mp-RsBB` where they are used for function evaluations.

## Benchmark approaches
### Dual reformulation
Implementation of the dual reformulation approach for robust pooling problems used as a benchmark in the computational studies.
### PyROS implementation
Implementation of the robust cutting planes approach via [PyROS](https://github.com/Pyomo/pyomo/blob/6.8.0/doc/OnlineDocs/contributed_packages/pyros.rst) for robust pooling problems used as a benchmark in the computational studies.
