import pyomo.environ as pe
import numpy as np
from pooling_network.instances.literature import literature_problem_data
from pooling_network.instances.data import pooling_problem_from_data
from pooling_network.pooling import index_set_k, index_set_ik
from pq_block_pyros import PoolingPQFormulationPRS 
import pyomo.contrib.pyros as pyros

instance_library=['haverly1','haverly2','haverly3','bental4','bental5','foulds2','adhya1','adhya2','adhya3','adhya4']

local_solver_='gurobi'
global_solver_='gurobi'
solve_globally=True
glob=str(solve_globally)
unc_set_=['box','ellipse','polyhedron']
perturbations=[0.05]


def generate_ksi_box_set(problem,ro_):
    bounds=[]
    for i,k in m.c_nom_index:
        ub=ro_
        lb=-ro_
        bounds.append([lb,ub])
    return pyros.BoxSet(bounds=bounds)

def generate_ksi_polyhedral_set(problem,ro_):
    # Initialize A matrix and b vector
    A = []
    b = []    
    ik_list = list(index_set_ik(problem))  # Flatten the (i, k) index set
    num_params = len(ik_list)

    for i in range(2 ** num_params):
        coeffs = [1 if (i >> j) & 1 == 1 else -1 for j in range(num_params)]
        A.append(coeffs)
        b.append(ro_)

    # Return the PyROS PolyhedralSet
    return pyros.PolyhedralSet(lhs_coefficients_mat=A, rhs_vec=b)

def generate_ksi_ellipsoidal_set(problem, ro_):
    # Determine number of uncertain parameters
    num_params = len(list(index_set_ik(problem)))

    shape_mat = ( ro_**2) * np.eye(num_params)

    # Ellipsoidal set with center at origin
    return pyros.EllipsoidalSet(center=[0]*num_params, shape_matrix=shape_mat)

avail_sets=unc_set_

for unc_set_ in avail_sets:
    for pert in perturbations:
        for instance_name in instance_library:
            use_flow=False        
            name=instance_name
            # Define problem configuration
            problem = pooling_problem_from_data(literature_problem_data(name))
            model = pe.ConcreteModel()
            model.pooling = PoolingPQFormulationPRS()
            model.pooling.set_pooling_problem(problem)
            model.pooling.rebuild()
            model.pooling.add_objective(use_flow_cost=use_flow)
            m=model.pooling
            
            uncertain_parameters=[m.ksi]
            if unc_set_=='box':
                uncertainty_set_ =generate_ksi_box_set(problem,pert)
            elif unc_set_=='polyhedron':
                uncertainty_set_=generate_ksi_polyhedral_set(problem,pert)
            elif unc_set_=='ellipse':
                uncertainty_set_=generate_ksi_ellipsoidal_set(problem,pert)


        

            pyros_solver = pe.SolverFactory('pyros')



            first_stage_variables=[m.q,m.y,m.z,m.v]
            second_stage_variables=[]
            try:

                results= pyros_solver.solve(model = model,
                                                first_stage_variables = first_stage_variables,
                                                second_stage_variables = second_stage_variables,
                                                uncertain_params = uncertain_parameters,
                                                uncertainty_set = uncertainty_set_,
                                                local_solver = local_solver_,
                                                global_solver= global_solver_,
                                                options = {
                                                    "objective_focus": pyros.ObjectiveType.worst_case,                                        
                                                    "time_limit":3600,
                                                    "solve_master_globally": solve_globally,
                                                    "load_solution":True
                                                })

                z_ro = results.final_objective_value
                solve_time = results.time
                iterations = results.iterations
                final_condition = results.pyros_termination_condition 
    
            except RuntimeError as e:
                print(f"Error solving problem {instance_name}: {e}")
                continue  # Skip to the next model





