import pyomo.environ as pe
from timeit import default_timer as timer
import time
import pandas as pd
import numpy as np
from pooling_network.instances.literature import literature_problem_data
from pooling_network.instances.data import pooling_problem_from_data
from dual_pq_block import PoolingPQFormulationDual

instance_library=['haverly1','haverly2','haverly3','bental4','bental5','foulds2','adhya1','adhya2','adhya3','adhya4']
pert=1
global_solver='gurobi'
solver=pe.SolverFactory(global_solver)
solver.options['TimeLimit'] = 3600  
set_size=[0.05]
avail_sets=['box','polyhedron','ellipse']


def QP_dual(instance_name,ro,_pert,unc_set):
    use_flow=False
    name=instance_name
    problem = pooling_problem_from_data(literature_problem_data(name))
    model = pe.ConcreteModel()
    model.pooling = PoolingPQFormulationDual()
    model.pooling.define_uncertainty_parameters(ro,_pert)
    model.pooling.set_pooling_problem(problem)
    model.pooling.rebuild()
    model.pooling.add_objective(use_flow_cost=use_flow)
    model.pooling.robust_uncertainty_lower_constraint(unc_set)
    model.pooling.robust_uncertainty_upper_constraint(unc_set) 
    model.pooling.auxiliary_uncertainty_gamma_constraint(unc_set)
   
    return model


for unc_set_ in avail_sets:

    for ro in set_size:
        for instance_name in instance_library:
            modelQP_dual=QP_dual(instance_name,ro,pert,unc_set_)
            res=solver.solve(modelQP_dual,tee=True)

        
