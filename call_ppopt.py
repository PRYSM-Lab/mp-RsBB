import pyomo.environ as pe
import numpy as np
from pooling_network.network import Node,Network
from pooling_network.instances.literature import literature_problem_data
from pooling_network.instances.data import pooling_problem_from_data
from pooling_network.pooling import (index_set_ij,
    index_set_i,
    index_set_il,
    index_set_ilj,
    index_set_ik)
from ppopt.mpqp_program import MPLP_Program


def nominal_quality(problem: Network,k_fix):
    '''
    Evaluate nominal quality
    '''
    c_nom={}
    c_hat_dict={}
    for i,k in index_set_ik(problem):
        inp=problem.nodes[i]
        c_nom[i,k]=inp.attr['quality'][k]
        if k==k_fix: # I am generating an mpP for each j,k couple
            c_hat_dict[i,k]=inp.attr['quality'][k]
    c_hat=np.array(list(c_hat_dict.values()))
    #print('C_hat is',c_hat)
    return c_hat,c_nom    
    
def mp_x_constraints(unc_set,size_x,ro_,upper):
    '''
    Generate constraints for MPP variables imposed by the selected uncertainty set.
    '''
    if upper:#upper quality constraint
        A_box=np.eye(size_x)
        b_box= ro_*np.ones((1, size_x)).reshape(-1, 1)
        b_box_zero=0*np.ones((1, size_x)).reshape(-1, 1)
    else:# lower quality constraint
        A_box=-np.eye(size_x)
        b_box= ro_*np.ones((1, size_x)).reshape(-1, 1)
        b_box_zero=0*np.ones((1, size_x)).reshape(-1, 1)

    if unc_set=='box':#Here I am using only the u+, u- for the box
        A=A_box
        b=b_box
    elif unc_set=='polyhedron':
        b_poly_base=ro_*np.ones((1, size_x+1)).reshape(-1, 1)
        A_pos=np.vstack((A_box, np.ones((1, size_x)),-A_box))
        A_neg =  np.vstack((A_box,-np.ones((1, size_x)),-A_box))
        A = A_pos if upper else A_neg
        b =np.vstack((b_poly_base,b_box_zero))
    return A, b


def mp_theta_bounds(problem,j, qbounds, ybounds, zbounds):
    '''
    Generate bounds for the MPP parameters which are dictated by the upper-level variable bounds
    '''
    theta_bounds_lo = {}
    theta_bounds_up = {}

    for i in index_set_i(problem):
        theta_bounds_lo[i,j] = 0.0
        theta_bounds_up[i,j] = 0.0
        if any(i == q[0] for q in  index_set_il(problem)):
            for l in [ll for ii,ll,jj in index_set_ilj(problem) if ii == i and jj==j]:   
                theta_bounds_lo[i,j] += qbounds[i, l][0] * ybounds[l, j][0]
                theta_bounds_up[i,j] += qbounds[i, l][1] * ybounds[l, j][1]               
        if any((i,j) == z for z in  index_set_ij(problem)):
            theta_bounds_lo[i,j] += zbounds[i, j][0]
            theta_bounds_up[i,j] += zbounds[i, j][1] 

    t_bounds_lo=np.array(list(theta_bounds_lo.values()))
    t_bounds_up=np.array(list(theta_bounds_up.values()))
    return t_bounds_lo, t_bounds_up

def generate_mpPooling(name,j,k,solve_upper,define_set,ro_,qbounds, ybounds, zbounds):
    '''
    Generate the MPP problem from the given robust lower-level problem

    Arguments:
    :param j: index for product set
    :param k: index for quality set
    :param solve_up: 0 if lower quality violation is examined, 1 for upper quality
    :param define_set: 'box' or 'polyhedron'
    :param ro_: uncertainty set size
    :param qbounds: bounds on variable q
    :param yqbounds: bounds on variable y
    :param zbounds: bounds on variable z

    Returns:
    MPP model object and nominal uncertain parameter
    '''
    problem = pooling_problem_from_data(literature_problem_data(name))
    size_x=len(list(index_set_i(problem))) 
    c_hat,c_nom=nominal_quality(problem,k)
    if solve_upper:
        H=-np.diag(c_hat) #minPu-thetaHx
    else:
        H=np.diag(c_hat) #min thetaHx-Pl  
    A,b=mp_x_constraints(define_set,size_x,ro_,solve_upper)
    #Set bounds on parameters theta
    theta_lo, theta_up= mp_theta_bounds(problem,j, qbounds, ybounds, zbounds)
    size_theta=len(theta_lo)
    #Generate MPP matrices
    A_theta =np.vstack((np.eye(size_theta), -np.eye(size_theta)))#A_theta--> bounds for ULP variables
    b_theta= np.hstack((theta_up, -theta_lo)).reshape(-1, 1) #b_theta--> bounds for ULP variables
    c = np.zeros(size_x).reshape(-1, 1)
    F = np.zeros((A.shape[0], size_theta))
    #Generate MPP model
    mpP = MPLP_Program(A, b, c, H, A_theta, b_theta, F)
    mpP.process_constraints()
    return mpP,c_nom



