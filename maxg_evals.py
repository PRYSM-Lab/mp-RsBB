# Complementary functions and robust-lower-level problem evaluations for the RsBB algorithm
import pyomo.environ as pe
import numpy as np
from pooling_network.network import Node,Network
from pooling_network.instances.literature import literature_problem_data
from pooling_network.instances.data import pooling_problem_from_data
from pooling_network.pooling import (index_set_ij,
    index_set_i,
    index_set_il,
    index_set_ilj,
    index_set_jk,
    index_set_lj,
    index_set_ik)
import sys


#Generate additional index sets 
def index_set_ik(network: Network):
    for input in network.nodes_at_layer(0):
        for k in input.attr['quality'].keys():
            yield input.name, k
            
def index_set_ijk(network: Network):
    for input in network.nodes_at_layer(0):
        for output in network.nodes_at_layer(2):
            for k in input.attr['quality'].keys():
                yield input.name, output.name, k

def index_set_k(network: Network):
    for input in network.nodes_at_layer(0):
        for k in input.attr['quality'].keys():
            yield  k   
        break 

def nominal_quality(problem: Network): 
    c_nom={}
    for i,k in index_set_ik(problem):
        inp=problem.nodes[i]
        c_nom[i,k]=inp.attr['quality'][k]
    return c_nom


def ksi_box_abs(m,i,k,ro_):
    '''
    Generate bounds for the box uncertainty set
    '''
    return -ro_,m.ksi[i,k], ro_

def ksi_polyhedral(m,k,problem:Network):
    '''
    Generate bounds for the polyhedral uncertainty set
    '''

    ik_filtered = [(ii, kval) for ii, kval in index_set_ik(problem) if kval == k]
    expr=0
    for i,k in ik_filtered :

        if not hasattr(m, 't_ksi'):  
            m.t_ksi = pe.Var(m.ik, domain=pe.NonNegativeReals) 

        m.add_component(f'abs_constraint_ksi_pos_{i}_{k}', pe.Constraint(expr=m.ksi[i, k] <= m.t_ksi[i, k]))
        m.add_component(f'abs_constraint_ksi_neg_{i}_{k}', pe.Constraint(expr=-m.ksi[i, k] <= m.t_ksi[i, k]))
        expr += m.t_ksi[i, k]
    
    constraint_name = f'ksi_polyhedral_constraint_{k}'
    m.add_component(constraint_name, pe.Constraint(expr=expr <= m.G[k]))
    
def ksi_ellipsoidal(m,k,problem:Network,ro_):
    '''
    Generate bounds for the ellipsoidal uncertainty set
    '''
    expr=0
    ik_filtered = [(ii, kval) for ii, kval in index_set_ik(problem) if kval == k]
    for i,k in ik_filtered:
        expr+=m.ksi[i,k]**2/ro_**2

    m.add_component(f'ellipsoidal_set_{k}', pe.Constraint(expr=expr <= 1))
    


def gamma_set(problem: Network,ro_):
    '''
    Initialize bound for polyhedral uncertainty set
    '''
    gamma_init={}
    gen = index_set_i(problem)
    for i,k in index_set_ik(problem):
        gamma_init[k]=ro_
    return gamma_init

def psi_set(problem: Network):
    '''
    Initialise bound for box uncertainty set
    '''
    psi_init={}
    for i,k in index_set_ik(problem):
        psi_init[i,k]=0
    return psi_init

def nominal_quality_mp(name):
    '''
    Nominal quality for the mp-RsBB approach
    ''' 
    problem = pooling_problem_from_data(literature_problem_data(name))
    c_nom={}
    for i,k in index_set_ik(problem):
        inp=problem.nodes[i]
        c_nom[i,k]=inp.attr['quality'][k]
    return c_nom
      

def constant_perturbation(problem:Network, pert):
    '''
    Evaluate the perturbation parameter
    '''
    c_pert={}
    c_nom=nominal_quality(problem)
    for i,k in index_set_ik(problem):
        c_pert[i,k]=round(c_nom[i,k]*pert,4)
    return c_pert

def uncertain_quality(m,i,k,problem: Network):
    '''
    Model the parameter for the uncertain quality constraint
    '''
    return m.c[i,k]==m.c_nom[i,k]+m.ksi[i,k]*m.c_pert[i,k]

def product_quality(problem: Network):
    '''
    Define bounds on the product quality
    '''
    p_upper={}
    p_lower={}
    for j,k in index_set_jk(problem):
        out=problem.nodes[j]
        if out.attr['quality_lower'] is None:
            out.attr['quality_lower'] = {}
        for key, val in out.attr['quality_lower'].items():
            if val is None:
                out.attr['quality_lower'][key] = 0        
        p_lower[j,k] = out.attr['quality_lower'].get(k, 0)
        p_upper[j,k]=out.attr['quality_upper'][k]
    return p_lower,p_upper

def g_obj(m,j,k,problem,upper):
    '''
    Model the objective function for the robust lower-level problem
    '''
    expr = 0
    flow = 0
    for l in [ll for ll, jj in index_set_lj(problem) if jj == j]:
        flow += m.y[l,j]
        for i in [ii for ii, ll in index_set_il(problem) if ll == l]:    
            expr += m.c[i,k]*m.v[i,l, j]
    for i in [ii for ii, jj in index_set_ij(problem) if jj == j]:
        flow += m.z[i, j]
        expr += m.c[i,k]*m.z[(i, j)]
    if upper: #for the upper quality constraint
        quality=product_quality(problem)[1][j,k]
        g_expr=expr - quality*flow
    else: #for the lower quality constraint
        quality=product_quality(problem)[0][j,k]
        g_expr=quality*flow-expr
    return g_expr 

def evaluate_g(name,j,k,v_U,y_U,z_U,unc,upper):
    '''
    Function evaluation for the objective of the lower-level problem
    '''
    expr = 0
    flow = 0
    problem = pooling_problem_from_data(literature_problem_data(name))
    for l in [ll for ll, jj in index_set_lj(problem) if jj == j]:
        flow += y_U[l,j]
        for i in [ii for ii, ll in index_set_il(problem) if ll == l]:    
            expr += unc[i,k]*v_U[i,l, j]
            
    for i in [ii for ii, jj in index_set_ij(problem) if jj == j]:
        flow += z_U[i, j]
        expr += unc[i,k]*z_U[(i, j)]
    if upper:
        quality=product_quality(problem)[1][j,k]
        obj_val=expr- quality*flow
    else:
        quality=product_quality(problem)[0][j,k]   
        obj_val= quality*flow-expr 
    return obj_val 

def prod_quality_cuts(name,m,j,k,unc,upper):
    '''
    Model robust cut constraints for the selected uncertainty samples
    '''
    expr = 0
    flow = 0
    problem = pooling_problem_from_data(literature_problem_data(name))
    for l in [ll for ll, jj in m.pooling.y.index_set() if jj == j]:
        flow += m.pooling.y[l,j]
        for i in [ii for ii, ll in m.pooling.q.index_set() if ll == l]:    
            expr += unc[i,k]*m.pooling.v[i,l, j]
            
    for i in [ii for ii, jj in m.pooling.z.index_set() if jj == j]:
        flow += m.pooling.z[i, j]
        expr += unc[i,k]*m.pooling.z[(i, j)]
    if upper:
        quality=product_quality(problem)[1][j,k]
        cut_expr=expr- quality*flow
    else:
        quality=product_quality(problem)[0][j,k]   
        cut_expr= quality*flow-expr
    return cut_expr<=0



def robust_lower_level_problem(name,v_solution,y_solution,z_solution,j,k,pert,solve_upper,define_set,ro_):
    '''
    Model the robust lower-level problem
    '''
    problem = pooling_problem_from_data(literature_problem_data(name))
    m=pe.ConcreteModel(f"LLP_model_{j}_{k}")
    idx_ik_filt=[(i, kval) for (i, kval) in index_set_ik(problem) if kval == k]
    c_nom_full = nominal_quality(problem)
    c_pert_full = constant_perturbation(problem, 1)
    c_nom_loc={(i, kval): c_nom_full[i, kval] for (i, kval) in idx_ik_filt}
    c_pert_loc={(i, kval): c_pert_full[i, kval] for (i, kval) in idx_ik_filt}

    m.ik=pe.Set(initialize=idx_ik_filt,dimen=2)
    m.c_nom=pe.Param(m.ik,  initialize=c_nom_loc)
    m.c_pert=pe.Param(m.ik, initialize=c_pert_loc)
    m.c=pe.Var(m.ik)
    m.ksi=pe.Var(m.ik)
    #Use the solution of the upper-level problem as parameters
    m.v=pe.Param(index_set_ilj(problem),initialize=v_solution, mutable=True)
    m.y=pe.Param(index_set_lj(problem), initialize=y_solution, mutable=True)
    m.z=pe.Param(index_set_ij(problem), initialize=z_solution, mutable=True)
    if define_set=='box':
        m.uncertainty_set=pe.Constraint(m.ik, rule=lambda m, i, k: ksi_box_abs(m,i,k,ro_))
    elif define_set=='polyhedron':
        m.G=pe.Param(index_set_k(problem), initialize=gamma_set(problem,ro_))
        ksi_polyhedral(m, k, problem) 
    elif define_set=='ellipse':
        ksi_ellipsoidal(m,k,problem,ro_)

    
    m.c_formulation=pe.Constraint(m.ik,  rule=lambda m, i, k:uncertain_quality(m,i,k,problem))

    obj_expr=g_obj(m,j,k,problem,solve_upper)
    m.obj=pe.Objective(expr=obj_expr, sense=pe.maximize)
    return m,c_nom_full

