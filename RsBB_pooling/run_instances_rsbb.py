import pyomo
import pyomo.environ as pe
import pandas as pd
from pooling_network.instances.literature import literature_problem_data
from pooling_network.instances.data import pooling_problem_from_data
from pooling_network.formulation.pq_block import PoolingPQFormulation
from pqLP_block import PoolingPQLPFormulation
from RsBB import  RobustspatialBranchAndBound, NodeExhaustion, TimeLimit

instance_library=['haverly1','haverly2','haverly3','bental4','bental5','foulds2','adhya1','adhya2','adhya3','adhya4']
set_size=[0.05]
branching_method='bisection'
local_solver='gams:conopt4'
linear_solver='appsi_highs'
avail_sets=['box','ellipse','polyhedron']
define_path ='/path/to/results/directory'
Epsilon=1e-5
rob_tol=1e-3

def QP(instance_name):
  #Build QP model for the pooling problem
    use_flow=False    
    name=instance_name
    problem = pooling_problem_from_data(literature_problem_data(name))
    model = pe.ConcreteModel()
    model.pooling = PoolingPQFormulation()
    model.pooling.set_pooling_problem(problem)
    model.pooling.rebuild()
    model.pooling.add_objective(use_flow_cost=use_flow)
   
    return model

def LP(instance_name):
  #Build LP model for the relaxed pooling problem
    use_flow=False    
    name=instance_name
    problem = pooling_problem_from_data(literature_problem_data(name))
    model = pe.ConcreteModel()
    model.pooling = PoolingPQLPFormulation()
    model.pooling.set_pooling_problem(problem)
    model.pooling.rebuild()
    model.pooling.add_objective(use_flow_cost=use_flow)
    
        
    return model
      

for unc_set_ in avail_sets:
    if unc_set_=='ellipse':
        robust_solver='gurobi'
    else:
        robust_solver=linear_solver


    for pert in set_size:
        for instance_name in instance_library:
            modelQP=QP(instance_name)
            modelQP.pooling.cuts=pe.ConstraintList()
            
            modelLP=LP(instance_name)
            modelLP.pooling.cuts=pe.ConstraintList()

            try:
                RsBB=RobustspatialBranchAndBound(instance_name,branching_method,local_solver,linear_solver,robust_solver,modelQP,modelLP,define_path,pert,Epsilon,rob_tol,unc_set_)
    
            except NodeExhaustion as e:
                print(f"Exiting routine for {instance_name}: {e}")
                print('explored',  e.nexplored)
            except TimeLimit as e:
                print(f"Exiting routine due to time limit {instance_name}: {e}")                                                    
                print('explored',  e.nexplored)
            continue

       

