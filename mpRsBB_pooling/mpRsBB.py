import pyomo.environ as pe
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from RsBB_pooling.maxg_evals import  prod_quality_cuts,nominal_quality_mp,evaluate_g
import numpy as np
import math
import pandas as pd
import time
from timeit import default_timer as timer
from operator import itemgetter
import logging
import copy
import pickle
import os
#To prevent printing pyomo warnings e.g. SolverStatus infeasible
logging.getLogger('pyomo.core').setLevel(logging.ERROR)




class NodeExhaustion(Exception):
    """Custom exception for node exhaustion termination."""
    def __init__(self, message, z_bf, cpuf, cutsused, nexplored,q_time,l_time,m_time):
        super().__init__(message)
        self.z_bf = z_bf
        self.cpuf = cpuf
        self.cutsused = cutsused
        self.nexplored = nexplored
        self._qp_timer=q_time
        self._lp_timer=l_time
        self._mp_timer=m_time        

class TimeLimit(Exception):
    """Custom exception for time limit."""
    def __init__(self, message, z_bf, cpuf, cutsused, nexplored,q_time,l_time,m_time):
        super().__init__(message)
        self.z_bf = z_bf
        self.cpuf = cpuf
        self.cutsused = cutsused
        self.nexplored = nexplored
        self._qp_timer=q_time
        self._lp_timer=l_time
        self._mp_timer=m_time        



class mpRobustspatialBranchAndBound:
    '''
    The  multi-parametric robust spatial-Branch-and-Bound algorithm for the pooling problems.

    Arguments: 
    instance (str): Pooling problem instance from pooling-network library 
    branch_method (str): Name of branching method ('bisection'/'incumbent')
    QP_solver (str): Name of local QP solver
    LP_solver (str): Name of LP solver
    model_QP (obj): Pyomo model for QP problem
    model_LP (obj): Pyomo model for LP problem
    my_path (str): Local path for results export
    pert_ (scalar): Uncertainty set size
    epsilon_ (scalar): Fathoming tolerance
    unc_set_ (str): Uncertainty set type ('box'/'ellipse'/'polyhedron')
    mp_algo (str): Name of multi-parametric programming algorithm from PPOPT
    cr_path (str): Local path for folder with the results of PPOPT for the obtained critical regions

    Returns: 
    A robust optimal solution for the selected pooling problem
    '''
    def __init__(self,instance,branch_method,QP_solver,LP_solver,model_QP,model_LP,my_path,pert_,epsilon_,unc_set_,rob_tol_,mp_algo,cr_path):
        #The QP model of the pooling problem
        self.QP=model_QP
        #The LP model of the pooling problem
        self.LP=model_LP     
        #Name of pooling problem
        self.name=instance
        #Bound dictionary at each node for q variable
        self._q_bounds={}
        #Bound dictionary at each node for y variable
        self._y_bounds={} 
        #Bound dictionary at each node for z variable
        self._z_bounds={} 
        #LP solution at node 
        self._z_lp={}
        #Best possible solution  
        self._z_bp={} 
        #QP solution at node
        self._z_qp={}
        #Best found solution  
        self._z_bf={} 
        #Gap at node
        self._gap_opt={}
        #Set of waiting nodes
        self._s_w=[0]
        #Set of new nodes
        self._s_n=[0]   
        #Define QP solver
        self._solverQP= pe.SolverFactory(QP_solver)
        #Define LP solver
        self._solverLP=pe.SolverFactory(LP_solver)
        #Define mpP algorithm
        self._mpalgo=mp_algo
        #Define branching method (bisection/incumbent)
        self.method=branch_method
        #Define maximum number or iterations
        self.max_iter=1000
        #Fathoming tolerance
        self.epsilon=epsilon_
        #Robustness tolerance
        self.rob_tol=rob_tol_
        #Branching variable sensitivity
        self.cost_sens=1e-05
        #Time limit [s]
        self.time_lim=3600
        #Monitor parent-child node relation
        self._relation=[]       
        #Define save_path
        self._base_path=my_path        
        #Uncertainty deviation
        self.Psi=1  
        #Uncertainty perturbation
        self.pert=pert_
        #Uncertainty realisations/samples
        self.unc_real={}
        # Uncertainty set type (box/polyhedron)
        self.define_unc=unc_set_
        #Uncertainty violation counter
        self.viol_counter=0
        self.cuts_node={}
        #Initialise the uncertainty realisation
        self.nom_unc_real=True
        #Sigma values for pseudoscore branching
        self.sigma={}
        #eta values for monitoring selected branching variables
        self.eta={}
        #Keys for set of branhcing variables
        self.all_keys=[]
        #RCP iteration counter per node
        self.node_rc_iterations={}
        #Critical regions evaluated by PPOPT
        self._critical_regions={}
        #Path for critical region folder
        self._cr_path=cr_path
        #Timers for QP, LP solves and MPP evaluations
        self._qp_timer={}
        self._lp_timer={}
        self._mp_timer={}

    def load_critical_regions(self):
        '''
        Load critical regions obtained by PPOPT
        '''
        pert=str(self.pert)
        filename='ξ'+pert+'_'+self.name + '_' + self.define_unc + '_' + self._mpalgo+'_critical_regions.pkl'
        full_path = os.path.join(self._cr_path, filename)
        with open(full_path, 'rb') as f:
            self._critical_regions = pickle.load(f)
    
    def statusok(self,res):
        '''
        Evaluate termination condition and solver status for the obtained results
        
        Arguments:
        :param res: Results of the QP, LP or RP problems

        Returns:
        Logic value for result is ok. 1 for ok status, -1 for status not ok
        '''
        if res.solver.status==SolverStatus.ok and (res.solver.termination_condition==TerminationCondition.optimal or res.solver.termination_condition==TerminationCondition.locallyOptimal or res.solver.termination_condition==TerminationCondition.feasible):
            resok=1
        elif res.solver.status==SolverStatus.warning and (res.solver.termination_condition==TerminationCondition.optimal or res.solver.termination_condition==TerminationCondition.locallyOptimal or res.solver.termination_condition==TerminationCondition.feasible):
            resok=1
        else:
            resok=-1
        return resok 
    
    def set_bounds(self,node):  
        '''
        Update variable bounds for (q, y) at the given node. 
        Bounds are updated for both QP and LP models using dictionaries.
        
        Arguments:
        :param node: Given node
        '''
        mQP=self.QP;       mLP=self.LP
        qbound_=self._q_bounds; ybound_=self._y_bounds     
        for i,l in mLP.pooling.q.index_set():
            mQP.pooling.q[(i,l)].setlb(qbound_[node][(i,l)][0]);    mQP.pooling.q[(i,l)].setub(qbound_[node][(i,l)][1])
            mLP.pooling.q[(i,l)].setlb(qbound_[node][(i,l)][0]);    mLP.pooling.q[(i,l)].setub(qbound_[node][(i,l)][1])
            mLP.pooling.q_lo[(i,l)]=qbound_[node][(i,l)][0]    ;    mLP.pooling.q_up[(i,l)]=qbound_[node][(i,l)][1]

        for l,j in mLP.pooling.y.index_set():
            mQP.pooling.y[(l,j)].setlb(ybound_[node][(l,j)][0]);     mQP.pooling.y[(l,j)].setub(ybound_[node][(l,j)][1])
            mLP.pooling.y[(l,j)].setlb(ybound_[node][(l,j)][0]);     mLP.pooling.y[(l,j)].setub(ybound_[node][(l,j)][1])
            mLP.pooling.y_lo[(l,j)]=ybound_[node][(l,j)][0]    ;     mLP.pooling.y_up[(l,j)]=ybound_[node][(l,j)][1]
    
    def set_ps_bounds(self,p_kid,ps_qbound,ps_ybound):
        '''
        Set variable bounds for pseudo-kids generated during the pseudoscore evaluation process for branching variable selection.
        
        Arguments:
        :param p_kid: Node of pseudo-kid
        :param ps_qbound: Bounds of variables q at p_kid
        :param ps_ybound: Bounsd of variables y at p_kid
        '''
        mLP=self.LP
        for i,l in mLP.pooling.q.index_set():
            mLP.pooling.q[(i,l)].setlb(ps_qbound[p_kid][(i,l)][0]);    mLP.pooling.q[(i,l)].setub(ps_qbound[p_kid][(i,l)][1])
            mLP.pooling.q_lo[(i,l)]=ps_qbound[p_kid][(i,l)][0]    ;    mLP.pooling.q_up[(i,l)]=ps_qbound[p_kid][(i,l)][1]

        for l,j in mLP.pooling.y.index_set():
            mLP.pooling.y[(l,j)].setlb(ps_ybound[p_kid][(l,j)][0]);     mLP.pooling.y[(l,j)].setub(ps_ybound[p_kid][(l,j)][1])
            mLP.pooling.y_lo[(l,j)]=ps_ybound[p_kid][(l,j)][0]    ;     mLP.pooling.y_up[(l,j)]=ps_ybound[p_kid][(l,j)][1]

    def save_val(self,m):
        '''
        Save the solution for the optimisation variables of the selected model.
        
        Arguments:
        :param m: Selected model

        Returns:
        Dictionaries for the solutions of the optimisation variables
        '''
        q_val = {(i,l ): pe.value(m.pooling.q[i,l])     for i,l in m.pooling.q.index_set()}
        y_val = {(l, j): pe.value(m.pooling.y[l, j])    for l, j in m.pooling.y.index_set()}
        z_val = {(i, j): pe.value(m.pooling.z[i, j])    for i, j in m.pooling.z.index_set()}
        v_val = {(i,l,j): pe.value(m.pooling.v[i,l,j])  for i,l,j in m.pooling.v.index_set()} 
        return q_val,y_val,z_val,v_val
    
    def initialise(self):
        '''
        Initialise best found and best possible solution. 
        Obtain solution of QP and LP models. For an infeasible model set solution to infinity.
        
        Returns:
        Best found and best possible scalars

        '''
        solver=self._solverQP      
        solverL=self._solverLP
        start_qp=time.process_time()
        res_init_QP=solver.solve(self.QP,tee=False)
        end_qp=time.process_time()
        self._qp_timer[0]=end_qp-start_qp
        start_lp=time.process_time()
        res_init_LP=solverL.solve(self.LP,tee=False)
        end_lp=time.process_time()
        self._lp_timer[0]=end_lp-start_lp
        self._mp_timer[0]=0

        if self.statusok(res_init_QP)==1:
            z_bf=pe.value(self.QP.pooling.cost)    
        else:
            z_bf=float('+inf')
        self._z_qp[0]=z_bf 
        
        if self.statusok(res_init_LP)==1:
            z_bp=pe.value(self.LP.pooling.cost)
            self._z_lp[0]=z_bp
        else:
            z_bp=float('+inf')
        self._z_lp[0]=z_bp 
        return z_bf,z_bp      
    
    def solve_qp(self,_n_c):
        '''
        Solve the QP problem at current node.
        
        Arguments:
        :param _n_c: Current node

        Returns:
        Scalar of the best found solution at the node
        Pyomo result object
        '''
        mQP=self.QP
        self.set_bounds(_n_c)
        solve_num=list(self._qp_timer.keys())[-1]+1
        start_qp=time.process_time()
        resQP=self._solverQP.solve(mQP,tee=False) 
        end_qp=time.process_time()
        self._qp_timer[solve_num]=end_qp-start_qp
        if self.statusok(resQP)==1:
            z_bf_new=pe.value(mQP.pooling.cost) 
        else:
            z_bf_new=float('+inf')

        self._z_qp[_n_c]=z_bf_new

        return    z_bf_new,resQP
    
    def solve_lp(self,_n_c):
        '''
        Solve the LP problem at current node.
        
        Arguments:
        :param _n_c: Current node

        Returns:
        Scalar of the best possible solution at the node
        Pyomo result object
        Dictionaries for the solutions of the optimisation variables
        '''        
        mLP=self.LP
        self.set_bounds(_n_c)
        solve_num=list(self._lp_timer.keys())[-1]+1
        start_lp=time.process_time()        
        resLP=self._solverLP.solve(mLP, load_solutions = False)
        end_lp=time.process_time() 
        self._lp_timer[solve_num]=end_lp-start_lp

        if self.statusok(resLP)==1:
            mLP.solutions.load_from(resLP)
            q_L,y_L,z_L,v_L=self.save_val(mLP)
            z_bp_new=pe.value(mLP.pooling.cost) 
        else:
            z_bp_new=float('+inf')
            q_L=0;y_L=0;z_L=0;v_L=0 # set to zero if I have infeasible LP
        self._z_lp[_n_c]=z_bp_new
        return    z_bp_new,q_L,y_L,z_L,v_L 
    
    def solve_ps_lp(self,p_kid,ps_qbound,ps_ybound):
        '''
        Solve the LP problem at the current pseudo-kid node
        
        Arguments:
        :param p_kid: Node of pseudo-kid
        :param ps_qbound: Bounds of variables q at p_kid
        :param ps_ybound: Bounsd of variables y at p_kid

        Returns:
        Scalar for the LP solution of the pseudo-kid
        '''
        
        mLP=self.LP
        self.set_ps_bounds(p_kid,ps_qbound,ps_ybound)
        resLP=self._solverLP.solve(mLP,load_solutions = False)
        if self.statusok(resLP)==1:
            mLP.solutions.load_from(resLP)
            z_ps_lp=pe.value(mLP.pooling.cost) 
        else:
            z_ps_lp=float('+inf')
        return    z_ps_lp
    
    def update_z_lp(self,viol_new,viol_old,time_begin,z_bf,z_bp,count):
        '''
        Update the LP solution of the waiting nodes based on the augmented sampled uncertainty set.

        
        Arguments:
        :param viol_new: Counter of elements in the sampled uncertainty set after the robustness evaluation
        :param viol_old: Counter of elements in the sampled uncertainty set before the robustness evaluation
        :param time_begin: Timestamp for the start of the RsBB algorithm
        :param z_bf: Best found solution
        :param z_bp: Best possible solution
        :param count: Counter of nodes explored
        '''
        s_w=self._s_w; s_n=self._s_n
        z_lp=self._z_lp; 
        if viol_new>viol_old:     
            for n_w in s_w:
                time_now=timer()
                self.solve_lp(n_w)
                self.time_limit_termination(time_begin,z_bf,z_bp,count)


    def evaluate_ksi(self, j, k, solve_up, theta):
        '''
        Evaluate explicit solution for the generated critical regions

        Arguments:
        :param j: index for product set
        :param k: index for quality set
        :param solve_up: 0 if lower quality violation is examined, 1 for upper quality
        :param theta: theta point evaluated by the upper-level problem solution
        
        Returns:
        Explicit solution of the MPP as a list
        '''
        regions = self._critical_regions[j, k, solve_up]
        tol=1e-2
        x_star=[]
        for i in regions: 
            print('am i here?',i)
            A = regions[i]['A']
            b = regions[i]['b']
            E = regions[i]['E']
            f = regions[i]['f']
            #Evaluate mebmership of theta in critical regions
            if np.all(E @ theta - f < tol):  
                x_star_ar = A @ theta + b #this is an nd array
            else:
                True

        x_star_list=x_star_ar.tolist()
        for m in x_star_list:#flatten the list of lists
            for mm in m:
                x_star.append(mm)    
        #return a flat list    
        return x_star
            
     
    def generate_theta_point(self,j,v_U,z_U):
        '''
        Generate theta point based on the incumbent upper-level solution

        Arguments:
        :param j: index for product set
        :param v_U: Solution of variable v obtained from solving the QP problem
        :param z_U: Solution of variable z obtained from solving the QP problem
        
        Returns:
        Parametric theta point
        '''
        mLP=self.LP
        theta_point={}
        for i in mLP.pooling.input_capacity.index_set():#to be doulbe checked
            print('i is',i)
            theta_point[i,j]=0
            print('thteta point is',theta_point)
            if any(i == q[0] for q in  mLP.pooling.q.index_set()):
                for l in [ll for ii,ll,jj in mLP.pooling.v.index_set() if ii == i and jj==j]:   
                        theta_point[i,j]+=v_U[i,l, j]     
            if any((i,j) == z for z in  mLP.pooling.z.index_set()):           
            # if any(i == z[0] for z in  mLP.pooling.z_index):
                theta_point[i,j]+=z_U[i, j]       

        theta=np.array(list(theta_point.values())).reshape(-1, 1)
        return theta
     
    def evaluate_mp_sol(self, j,k, solve_up,v_U,y_U,z_U,theta_point):
        '''
        Evaluate uncertain parameter value as a function evaluation for the obtained explicit solution

        Arguments:
        :param j: index for product set
        :param k: index for quality set
        :param solve_up: 0 if lower quality violation is examined, 1 for upper quality
        :param v_U: Solution of variable v obtained from solving the QP problem
        :param y_U: Solution of variable y obtained from solving the QP problem
        :param z_U: Solution of variable z obtained from solving the QP problem        
        :param theta: theta point evaluated by the upper-level problem solution
        
        Returns:
        Dictionary of unertain parameter value
        '''
        unc={}
        unc_val={}
        mp_ksi=self.evaluate_ksi(j, k, solve_up,theta_point)
        c_nom=self.unc_real[-1]
        num=0
        for i, kk in c_nom.keys():
            if kk==k:
                unc_val[i,k]=round(c_nom[i,k]*(1+mp_ksi[num]),4)
                num+=1
        solve_num=list(self._mp_timer.keys())[-1]+1
        start_mp=timer()                      
        obj_value=evaluate_g(self.name,j,k,v_U,y_U,z_U,unc_val,solve_up)
        end_mp=timer() 
        self._mp_timer[solve_num]=end_mp-start_mp
        if obj_value>self.rob_tol:
            unc=unc_val                
        return unc    

    def add_cuts_mp(self,unc_new,_n_c,k):
        '''
        Add robust cuts for the new uncertainty samples to the LP and QP problems.
        
        Arguments:
        :param unc_new: New uncertainty samples
        :param _n_c: Current node
        :param k: Quality index
        '''
        mLP=self.LP
        mQP=self.QP
        # If new uncertainty sample does not belong in the sampled uncertainty set
        if unc_new not in self.unc_real.values():
            #Counter for added cuts at current node
            self.cuts_node[self.viol_counter]=_n_c
            #Augment the set of uncertainty samples with the new uncertainty sample
            self.unc_real[self.viol_counter]=unc_new
            self.viol_counter+=1 
            for j,kk in mLP.pooling.product_quality_upper_bound.index_set():
                if kk==k:
                    for rp_ in range(2):
                        expr_ruleL=prod_quality_cuts(self.name,mLP,j,k,unc_new,rp_)
                        mLP.pooling.cuts.add(expr_ruleL)
                        expr_ruleQ=prod_quality_cuts(self.name,mQP,j,k,unc_new,rp_)
                        mQP.pooling.cuts.add(expr_ruleQ)



    def mp_infesibility_test(self,_n_c,v_U,y_U,z_U): # I can have outside the if statement before calling the infeasibility test mUD
        '''
        Iteratively perform function evaluations for the robust lower-level objective functions. Obtain new violating uncertainty samples and generate the corresponding robust cuts.
        
        Arguments:
        :param _n_c: Current node
        :param v_U: Solution of variable v obtained from solving the QP problem
        :param y_U: Solution of variable y obtained from solving the QP problem
        :param z_U: Solution of variable z obtained from solving the QP problem
        '''
        mLP=self.LP

        for rp in range(2):
            solve_up=rp    
            for j,k in mLP.pooling.product_quality_upper_bound.index_set():
                theta_point=self.generate_theta_point(j,v_U,z_U)
                unc_val_new=self.evaluate_mp_sol(j,k, solve_up,v_U,y_U,z_U,theta_point)
                if unc_val_new:
                    self.add_cuts_mp(unc_val_new,_n_c,k)
        
    def initialise_bounds(self):
        '''
        Initialise q, y and z variable bounds
        '''
        mLP=self.LP
        __q_bounds={}
        __y_bounds={}
        __z_bounds={}
        for i,l in mLP.pooling.q.index_set():
            __q_bounds[(i,l)]=(mLP.pooling.q[(i,l)].lb,mLP.pooling.q[(i,l)].ub)
        for l,j in mLP.pooling.y.index_set():
            __y_bounds[(l,j)]=(mLP.pooling.y[(l,j)].lb,mLP.pooling.y[(l,j)].ub)
        for i,j in mLP.pooling.z.index_set():
            __z_bounds[(i,j)]=(mLP.pooling.z[(i,j)].lb,mLP.pooling.z[(i,j)].ub)

        self._q_bounds[0]=__q_bounds
        self._y_bounds[0]=__y_bounds
        self._z_bounds[0]=__z_bounds

      
    def variable_error(self,q_L,y_L,v_L):
        '''
        Maximum violation error evaluation for the choice of branching variable.
        
        Arguments:
        :param q_L: Solution of variable q obtained from solving the LP problem
        :param y_L: Solution of variable y obtained from solving the LP problem
        :param v_L: Solution of variable v obtained from solving the LP problem
        
        Returns:
        Dictionaries of errors for q and y variables
        '''
        mLP=self.LP
        q_er={}
        y_er={}
        #Evaluate approximation error on q variable 
        for i,l in mLP.pooling.q.index_set():
            q_er[i,l]=0
            for j in [jj for ll, jj in mLP.pooling.y.index_set() if ll == l]: 
                q_er[i,l]+=abs(v_L[(i,l,j)]-q_L[i,l]*y_L[l,j]) 
        #Evaluate approximation error on y variable 
        for l,j in mLP.pooling.y.index_set():
            y_er[l,j]=0
            for i in [ii for ii,ll,jj in mLP.pooling.v.index_set() if ll == l and jj==j]:    
                y_er[l,j]+=abs(v_L[(i,l,j)]-q_L[i,l]*y_L[l,j])
        return q_er,y_er

    def pseudo_kids(self,n_c,q_L,y_L):
        '''
        Pseudoscore evaluation for via pseudo-strong branching on all possible child nodes of the current one.
        
        Arguments:
        :param n_c: Current node
        :param q_L: Solution of variable q obtained from solving the LP problem
        :param y_L: Solution of variable y obtained from solving the LP problem

        Returns:
        The key for the selected branching variable and a logic variable that is 0 if y is selected for branching and 1 if q is selected.
        '''
        mLP=self.LP
        p_kid=0
        ps_qbound={}; ps_ybound={}
        delta={}
        lp_impr={} 
        pseudo_cost={}
        bad_kid=[]
        delta['pos'] = {};  delta['neg']  = {}
        lp_impr['pos'] = {};  lp_impr['neg']  = {}

        #Here I am initialising the boundary dictionaries for all kids
        for ps_branch_key in self.all_keys:
            #Initialise pseudoscore list
            sigma=[]
            if ps_branch_key in mLP.pooling.q.index_set():
                branch_q=True
            else:
                branch_q=False
                
            for kids in range(2):
                #Initialise bounds of the pseudokids based on the parent node
                ps_qbound[p_kid]=copy.deepcopy(self._q_bounds[n_c])
                ps_ybound[p_kid]=copy.deepcopy(self._y_bounds[n_c])
                #Select branching point            
                if branch_q:
                    qb = (q_L[ps_branch_key] if self.method != 'bisection' 
                        else ps_qbound[p_kid][ps_branch_key][0] + (ps_qbound[p_kid][ps_branch_key][1] - ps_qbound[p_kid][ps_branch_key][0]) / 2)              
                    yb=0            
                else:
                    yb = (y_L[ps_branch_key] if self.method != 'bisection' 
                        else ps_ybound[p_kid][ps_branch_key][0] + (ps_ybound[p_kid][ps_branch_key][1] - ps_ybound[p_kid][ps_branch_key][0]) / 2)
                    qb=0
                #Evaluate pseudoscores
                sigma=self.pseudo_branch(n_c,p_kid,branch_q,ps_branch_key,kids,qb,yb,ps_qbound,ps_ybound,delta,lp_impr,bad_kid,sigma)
                p_kid+=1
            if sigma:#in case sigma is empty for a kid
                pseudo_cost[ps_branch_key]=max(sigma)#choose the maximum pseudocost from pos and negative   
        pseudo_max_val=max(pseudo_cost.values())
        #Choose branching variable if maximum pseudoscore value is higher than the selected sensitivity
        if pseudo_max_val>=self.cost_sens:
            branch_key = max(pseudo_cost, key=pseudo_cost.get)
        else:
            branch_key=False
        
        if branch_key in mLP.pooling.q.index_set():
            branch_q=True
        else:
            branch_q=False
        return branch_key,branch_q

    def pseudo_branch(self,n_c,p_kid,branch_q,branch_key,kid,qb,yb,ps_qbound,ps_ybound,delta,lp_impr,sigma):
        '''
        Set bounds and evaluate sigma values for pseudoscores of the selected pseudokid.
        
        Arguments:
        :param n_c: Current node
        :param p_kid: Curent pseudokid
        :param branch_q: Logic variable, 0 if y variables are evaluated or 1 if q variabes are evaluated
        :param branch_key: Index key of the evaluated variable 
        :param kid: Logic variable, 0 if negative kid is considered 1 if positive kid is considered 
        :param qb: Branching point for q variable
        :param yb: Branching point for y variable
        :param ps_qbound: Dictionary of bounds on the pseudokid for q variable
        :param ps_ybound: Dictionary of bounds on the pseudokid for y variable
        :param delta: Dictionary for delta values for positive and negative pseudokids
        :param lp_impr: Dictionary for LP improvements for positive and negative pseudokids
        :param bad_kid: List of pseudokis for which pseudoscore cannot be evaluated
        :param sigma: List of pseudoscore values of previously evaluated pseudokids

        Returns:
        Appended pseudoscore list
        '''     
        if branch_q:
            xb=qb ; xbound=ps_qbound[p_kid][branch_key]
        else:
            xb=yb ; xbound=ps_ybound[p_kid][branch_key]
        
        #Set the bound interval of the pseudo kid as a list
        value=list(xbound)
        #Update the lower or upper bound with the selected branching point.
        #If negative kid is considered the upper bound is updated else the lower bound is updated.
        value[abs(kid-1)] =xb 
        xbound= tuple(value)
        #If incumbent solution is at the bound do not consider p_kid for branching    
        if abs(value[1]-value[0])<=10e-5:
            bad_kid.append((branch_key))

        #Update pseudokid bounds
        if branch_q:
            ps_qbound[p_kid][branch_key] = xbound
        else:
            ps_ybound[p_kid][branch_key] = xbound
        
        #Evaluate sigma values
        if branch_key not in  bad_kid:
            sigma=self.pseudo_param(n_c,p_kid,branch_key,branch_q,ps_qbound,ps_ybound,kid,delta,lp_impr,sigma)
        else:
            sigma.append(0)
        return sigma

    def pseudo_param(self,n_c,p_kid,branch_key,branch_q,ps_qbound,ps_ybound,kid,delta,lp_impr,sigma):
        '''
        Evaluate sigma values for pseudoscores of the selected pseudokid.
        
        Arguments:
        :param n_c: Current node
        :param p_kid: Curent pseudokid
        :param branch_q: Logic variable, 0 if y variables are evaluated or 1 if q variabes are evaluated
        :param branch_key: Index key of the evaluated variable 
        :param kid: Logic variable, 0 if negative kid is considered 1 if positive kid is considered 
        :param delta: Dictionary for delta values for positive and negative pseudokids
        :param lp_impr: Dictionary for LP improvements for positive and negative pseudokids
        :param bad_kid: List of pseudokis for which pseudoscore cannot be evaluated
        :param sigma: List of pseudoscore values of previously evaluated pseudokids

        Returns:
        Appended pseudoscore list
        '''
        key = 'neg' if kid == 0 else 'pos' 
        parent_z_lp=self._z_lp[n_c]
        delta[key][branch_key]={}; lp_impr[key][branch_key]={} #need to initialise them
        #Solve the LP problem for the selected pseudokid
        z_ps_lp=self.solve_ps_lp(p_kid,ps_qbound,ps_ybound)     
        if branch_q:
            delta[key][branch_key][p_kid]=ps_qbound[p_kid][branch_key][1]-ps_qbound[p_kid][branch_key][0] # choice of delta
        else:
            delta[key][branch_key][p_kid]=ps_ybound[p_kid][branch_key][1]-ps_ybound[p_kid][branch_key][0]
        
        if not math.isinf(z_ps_lp): 
            lp_impr[key][branch_key][p_kid]=abs(z_ps_lp-parent_z_lp)#minus since I have negative values
            sigma.append(lp_impr[key][branch_key][p_kid]/delta[key][branch_key][p_kid])
        else:
            lp_impr[key][branch_key][p_kid]=float('-inf')
        
        return sigma

    def eta_choice(self):
        '''
        Choice of branching vatiable based on the most frequently selected variable so far. 
        Metric employed if both maximum error and pseudoscore values are below the selected sensitivity

        Returns:
        The key for the selected branching variable and a logic variable that is 0 if y is selected for branching and 1 if q is selected.
        '''
        eta=self.eta
        mLP=self.LP
        branch_key = max(eta, key=eta.get)
        if branch_key in mLP.pooling.q.index_set():
            branch_q=True
        else:
            branch_q=False
        return branch_key,branch_q

    def branching_variable(self,n_c,q_L,y_L,v_L):
        '''
        Select branching variable based on the available metrics.
        
        Arguments:
        :param n_c: Current node
        :param q_L: Solution of variable q obtained from solving the LP problem
        :param y_L: Solution of variable y obtained from solving the LP problem
        :param v_L: Solution of variable v obtained from solving the LP problem
        
        Returns:
        The key for the selected branching variable and a logic variable that is 0 if y is selected for branching and 1 if q is selected.
        Dictionary of eta values.
        '''

        #Evaluate maximum violation error
        q_er,y_er=self.variable_error(q_L,y_L,v_L)
        q_er_max=max(q_er.values())
        y_er_max=max(y_er.values())
        eta_choice=False
        if q_er_max>y_er_max and q_er_max>=self.cost_sens :
            branch_q=True
            branch_key = max(q_er, key=q_er.get)
        elif y_er_max>=q_er_max and y_er_max>=self.cost_sens:
            branch_q=False
            branch_key = max(y_er, key=y_er.get)
        else:#if no branching variable is selected by maximum error, choose via pseudoscores
            branch_key,branch_q=self.pseudo_kids(n_c,q_L,y_L)
            if branch_key:
                True
            else:
                #if no branching variable is selected by pseudoscores, shoose via eta metric
                print('Errors and costs are zero, choose differently',self.eta)
                branch_key,branch_q=self.eta_choice()
                eta_choice=True
        
        #Increase the counter of the eta dictionary for the key of the selected branching variable
        self.eta[branch_key]+=1

        return branch_q, branch_key,eta_choice   
    
    def lower_bound_kid(self,nn,n_c,branch_q,branch_key,q_L,y_L,kid,qb,yb):
        '''
        Evaluate the lower bound of the generated child nodes.
        
        Arguments:
        :param nn: Child node 
        :param n_c: Current node
        :param branch_q: Logic variable, 0 if y variables are evaluated or 1 if q variabes are evaluated
        :param branch_key: Index key of the evaluated variable 
        :param q_L: Solution of variable q obtained from solving the LP problem
        :param y_L: Solution of variable y obtained from solving the LP problem
        :param kid: Logic variable, 0 if negative kid is considered 1 if positive kid is considered 
        :param qb: Branching point for q variable
        :param yb: Branching point for y variable
        '''

        if branch_q:
            x_L=q_L ;   xb=round(qb,10) ; xbound=self._q_bounds[nn][branch_key]
        else:
            x_L=y_L ;   xb=round(yb,10) ; xbound=self._y_bounds[nn][branch_key]

        #Set the bound interval of the pseudo kid as a list
        #value=[xl,xu] of parent 
        value=list(xbound)
        #Update the lower or upper bound with the selected branching point.
        #If negative kid is considered the upper bound is updated else the lower bound is updated.
        value[abs(kid-1)] =xb 
        xbound= tuple(value) 
        if branch_q:
            self._q_bounds[nn][branch_key] = xbound
        else:
            self._y_bounds[nn][branch_key] = xbound
        #Resolve the LP if the incumbent x_L solution is not on the examined child node
        if kid == 0:
            #For negative kid if incumbent solution is lower than the branching point, then set the lower bound value as that of the parent node
            if x_L[branch_key]<=xb:
                self._z_lp[nn]=self._z_lp[n_c]
            else:
                #If not, need to solve the LP a the child node to obtain the lower bound bound value
                self.solve_lp(nn)
        else:
            #For positive kid if incumbent solution is greater than the branching point, then set the lower bound value as that of the parent node
            if x_L[branch_key]>=xb:
                self._z_lp[nn]=self._z_lp[n_c]
            else:
                #If not, need to solve the LP a the child node to obtain the lower bound bound value
                self.solve_lp(nn)            

    def bounding(self,n_c,branch_q,branch_key,q_L,y_L,eta_choice):
        '''
        Generate child nodes.
        
        Arguments:
        :param nn: Child node 
        :param n_c: Current node
        :param branch_q: Logic variable, 0 if y variables are evaluated or 1 if q variabes are evaluated
        :param branch_key: Index key of the evaluated variable 
        :param q_L: Solution of variable q obtained from solving the LP problem
        :param y_L: Solution of variable y obtained from solving the LP problem
        :param eta_choice: Scalar indicating that branching is performed by the eta metric hence only branching by bisection can be performed
        '''
        method=self.method
        if eta_choice:
            method='bisection'
        qbound_=self._q_bounds; ybound_=self._y_bounds
        #Counter for generated child nodes, 0 for negative node and 1 for positive node
        for kids in range(2):
            nn=self._s_n[-1]+1
            qbound_[nn]=copy.deepcopy(qbound_[n_c])
            ybound_[nn]=copy.deepcopy(ybound_[n_c])            
            if branch_q:
                qb = (q_L[branch_key] if method != 'bisection' 
                    else qbound_[n_c][branch_key][0] + (qbound_[n_c][branch_key][1] - qbound_[n_c][branch_key][0]) / 2)              
                yb=0            
            else:
                yb = (y_L[branch_key] if method != 'bisection' 
                    else ybound_[n_c][branch_key][0] + (ybound_[n_c][branch_key][1] - ybound_[n_c][branch_key][0]) / 2)
                qb=0        

            self.lower_bound_kid(nn,n_c,branch_q,branch_key,q_L,y_L,kids,qb,yb)
            self._relation.append([n_c,nn])#append list of parent-child node relation
            self._s_n.append(nn)# append list of generated nodes                 
            self._s_w.append(nn)# append list of waiting nodes 

    def node_exhaustion_termination(self,time_begin,z_bf,z_bp,ncounter):
        '''
        If list of waiting nodes is empty exit algorithm.
        
        Arguments:
        :param time_begin: Timestamp for the start of the RsBB algorithm
        :param z_bf: Best found solution
        :param z_bp: Best possible solution
        :param ncounter: Counter of nodes explored
        '''
        if len(self._s_w)==0:
            time_end=time.process_time()
            elapsed_time=time_end-time_begin
            print('Best found',z_bf,'best possible', z_bp,'time',elapsed_time)
            self.result_export(elapsed_time,'Waiting nodes exhausted',z_bf)
            print('Nodes explored are',ncounter)
            raise NodeExhaustion('FINISH:No nodes in waiting list',z_bf=z_bf,cpuf=elapsed_time,cutsused=self.viol_counter,nexplored=ncounter,q_time=self._qp_timer,l_time=self._lp_timer,m_time=self._mp_timer)

    def time_limit_termination(self,time_begin,z_bf,z_bp,ncounter):
        '''
        Time limit termination function. If elapsed time exceeds imposed time limit exit algorithm.
        
        Arguments:
        :param time_begin: Timestamp for the start of the RsBB algorithm
        :param z_bf: Best found solution
        :param z_bp: Best possible solution
        :param ncounter: Counter of nodes explored
        '''
        time_now=timer()
        elapsed_time=time_now-time_begin
        if elapsed_time>=self.time_lim:
            print('Best found',z_bf,'best possible', z_bp,'time',elapsed_time)
            self.result_export(elapsed_time,'Time limit exceeded',z_bf)
            print('Nodes explored are',ncounter)
            raise TimeLimit('FINISH:Time limit termination',z_bf=z_bf,cpuf=elapsed_time,cutsused=self.viol_counter,nexplored=ncounter,q_time=self._qp_timer,l_time=self._lp_timer,m_time=self._mp_timer)
         
    
    def result_export(self,elapsed_time,termination,_z_bf):
        '''
        Function to export desired results from the algorithm.
        
        Arguments:
        :param elapsed_time: CPU time if waiting nodes exhausted, ellapsed time for time limit termination
        :param termination: Termination condition
        :param z_bf: Best founds solution

        Returns:
        Excel file in the dictated path.
        '''
        mLP=self.LP
        mQP=self.QP
        pert=str(self.pert)
        filename='ξ'+pert+'_'+self.name + '_' + self.define_unc + '_' + self._mpalgo+'_mpP'+'.xlsx'
        df_relation=pd.DataFrame(self._relation)
        df_relation.columns=['Parent','Child']
        df_nodesq = pd.DataFrame.from_dict(self._q_bounds, orient='index')
        nodesq_col= [f'q{tup}' for tup in mLP.pooling.q.index_set()]
        df_nodesq.columns = nodesq_col
        df_nodesq.reset_index(inplace=True)
        df_nodesq.rename(columns={'index': 'node'}, inplace=True)
        df_nodesy = pd.DataFrame.from_dict(self._y_bounds, orient='index')
        nodesy_col=[f'y{tup}' for tup in mLP.pooling.y.index_set()]
        df_nodesy.columns = nodesy_col
        df_nodesy.reset_index(inplace=True)
        df_nodesy.rename(columns={'index': 'node'}, inplace=True)
        df_zlp=pd.DataFrame.from_dict(self._z_lp, orient='index')
        df_zlp.columns=['zLP']
        df_zqp=pd.DataFrame.from_dict(self._z_qp, orient='index')
        df_zqp.columns=['zQP']
        df_zbf=pd.DataFrame.from_dict(self._z_bf, orient='index')
        df_zbf.columns=['zBF']  
        df_zbp=pd.DataFrame.from_dict(self._z_bp, orient='index')
        df_zbp.columns=['zBP']       
        df_out = pd.concat([df_nodesq,df_nodesy,df_zlp,df_zqp,df_zbf,df_zbp],axis=1)
        
        df_time=pd.DataFrame({'Time': [elapsed_time], 'Final solution': [_z_bf],'Termination status':[termination]})
        df_gapOPT=pd.DataFrame(list(self._gap_opt.items()), columns=['Node','Optimality Gap closure'])

        df_cuts=pd.DataFrame.from_dict(self.unc_real)
        df_cuts_node=pd.DataFrame.from_dict(self.cuts_node,orient='index')
        
        df_node_rc=pd.DataFrame.from_dict(self.node_rc_iterations,orient='index')
        
        df_qp_time=pd.DataFrame.from_dict(self._qp_timer,orient='index')
        df_lp_time=pd.DataFrame.from_dict(self._lp_timer,orient='index')
        df_mp_time=pd.DataFrame.from_dict(self._mp_timer,orient='index')
        
        base_path=self._base_path
        path=f'{base_path}{filename}'
        with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
            df_out.to_excel(writer, sheet_name='Bounds',index=False)
            df_relation.to_excel(writer,sheet_name='Relation',index=False)
            df_gapOPT.to_excel(writer,sheet_name='Optimality gap closure',index=True)
            df_time.to_excel(writer,sheet_name='CPU',index=False)
            df_cuts.to_excel(writer, sheet_name='Cuts used')
            df_cuts_node.to_excel(writer, sheet_name='Cuts to node')
            df_node_rc.to_excel(writer, sheet_name='RC iterations to node')
            df_qp_time.to_excel(writer, sheet_name='QP timer')
            df_lp_time.to_excel(writer, sheet_name='LP timer')
            df_mp_time.to_excel(writer, sheet_name='MP timer')
           
    def fathoming(self,z_bf):
        '''
        Fathom waiting nodes.
        
        Arguments:
        :param z_bf: Best found solution
        '''
        s_w=self._s_w
        z_lp=self._z_lp
        s_p=copy.deepcopy(s_w)
        #Evaluate all nodes in waiting list
        for n_p in s_p:
            if z_lp[n_p]*(1-self.epsilon)>=z_bf:
                self._z_qp[n_p]='fathomed'
                s_w.remove(n_p)#remove fathomed nodes from waiting list

    def pseudo_init(self):
        '''
        Initialise dictionaries for pseudoscore evaluations

        '''
        mLP=self.LP
        for i,l in mLP.pooling.q.index_set():
            self.all_keys.append((i,l))
        for l,j in mLP.pooling.y.index_set():
            self.all_keys.append((l,j))
        self.sigma['pos'] = {key: [] for key in self.all_keys}
        self.sigma['neg']  = {key: [] for key in self.all_keys}
        self.eta={key: 0  for key in self.all_keys}

    def run(self):
        '''
        Excecute the mp-RsBB algorihtm

        Returns:
        The best found optimal solution,
        CPU or elapsed time depending on the remination status,
        number of uncertainty samples,
        number of nodes exmplored,
        dictionary of CPU time for solved QP problems,
        dictionary of CPU time for solved LP problems,
        dictionary of CPU time for  MPP function evaluations,
        '''  
        #Set timestamps for the CPU and wallclock timers      
        time_begin=time.process_time()
        time_begin_wall=timer()
        #Load critical regions
        self.load_critical_regions()
        
        #Initialise nominal quality       
        self.unc_real[-1]=nominal_quality_mp(self.name)
        #Initialise pseudocost parameters
        self.pseudo_init() 
        s_w=self._s_w
        z_lp=self._z_lp
        #Initialise best found and best possible solutions
        z_bf,z_bp=self.initialise()
        #Initialise variable bounds
        self.initialise_bounds()
        #Initialise counter for nodes explored
        count=0
        #Flag for robust feasible node
        robust_feas_node=True
        #Flag for having performed a robustness check at current not
        robust_check=False
        #Flag for detecting an infeasible QP solution inside the infeasibility test
        bf_shaky=False
        
        #Loop for maximum allowed iterations 
        for tot_n in range(self.max_iter):
            new_cuts=False
            #Check if termination criteria are met
            self.node_exhaustion_termination(time_begin,z_bf,z_bp,count)
            self.time_limit_termination(time_begin_wall,z_bf,z_bp,count)
            #Initialise list of current nodes
            s_c=[]

            for n_w in s_w:  #for all waiting nodes
                if abs(z_lp[n_w]-z_bp)<=self.cost_sens:#evaluate if lower bound is as good as the best possible
                    s_c.append(n_w) #append those nodes to current nodes

            #For selected current nodes
            for n_c in s_c:
                count+=1 # updet counter for nodes explored
                #Break if set of waiting nodes is empty
                if not s_w:
                    break
                #Remove current form wiating nodes
                s_w.remove(n_c)
                #Solve the QP at current node
                z_bf_new,resQP=self.solve_qp(n_c)
                #if last node was robust infeasible after adding cuts, update z_bf with the next robust feasible node
                if not robust_feas_node and (not math.isinf(z_bf_new)) and robust_check:
                    z_bf=z_bf_new 
                    bf_shaky=True # activate flag for having updated the z_bf by a feasible node

                #Check for time limit termination
                self.time_limit_termination(time_begin_wall,z_bf,z_bp,count)
                #Reset flags
                robust_feas_node=True
                robust_check=False 
                # Perform infeasibility test only to nodes as good as the z_bf
                if z_bf_new<=z_bf :
                    #Set a counter for maximum iterations of the robust cutting planes
                    for loop_infeas in range(self.max_iter):
                        #Save the number of generated uncertainty samples so far
                        viol_counter_old=self.viol_counter
                        #Perform the infeasibility test only for feasible QP nodes
                        if self.statusok(resQP)==1: 
                            q_U,y_U,z_U,v_U=self.save_val(self.QP)
                            self.mp_infesibility_test(n_c,v_U,y_U,z_U)   
                        else:
                            #For infeasible nodes skip
                            robust_feas_node=False
                            break
                        #Save the number of generated uncertainty samples so far
                        viol_counter_new=self.viol_counter
                        #If new samples are generated add new robust cuts to the QP and LP problems                    
                        if viol_counter_new>viol_counter_old:
                            #Activate flags
                            robust_check=True
                            new_cuts=True
                            #Resolve the QP problem with added cuts
                            z_bf_r,resRQP=self.solve_qp(n_c)

                            if self.statusok(resRQP)==1:
                                #Update z_bf_new 
                                z_bf_new=z_bf_r  
                            else:
                                #I have infeasible node after new cuts. Need to break
                                z_bf_new=z_bf_r
                                robust_feas_node=False
                                break
                        else:
                            #No new violations. Continue to branching
                            break
                        #Update counter for robust cutting plane iterations at current node
                        self.node_rc_iterations[n_c]=loop_infeas+1
                    #Check for time limit termination
                    self.time_limit_termination(time_begin_wall,z_bf,z_bp,count)

                #Update z_bf if  I have performed the infeasibility test and the node was robust feasible 
                # OR I have prevously found a robust infesible node but the current one is feasible
                if (robust_check and robust_feas_node) or (bf_shaky and not math.isinf(z_bf_new)):
                    z_bf=z_bf_new 
                    #Reset flag
                    bf_shaky=False
                #The z_bf solution is updated as the minimum between the QP solution and current z_bf               
                z_bf=min(z_bf,z_bf_new)

                #Reset flag
                if not math.isinf(z_bf_new):
                    bf_shaky=False
                
                #Solve LP problem at current node
                z_bp_new, q_L,y_L,z_L,v_L=self.solve_lp(n_c)
                #Check for time limit termination
                self.time_limit_termination(time_begin_wall,z_bf,z_bp,count)
                #Update z_lp values if I have added new cuts
                if robust_check :
                    self.update_z_lp(viol_counter_new,viol_counter_old,time_begin_wall,z_bf,z_bp,count)
                
                #Check for time limit termination
                self.time_limit_termination(time_begin_wall,z_bf,z_bp,count)

                #If LP at current node is feasible, proceed with branching
                if not math.isinf(z_bp_new):
                    branch_q,branch_key,eta_choice=self.branching_variable(n_c,q_L,y_L,v_L)
                    if branch_key:
                        self.bounding(n_c,branch_q,branch_key,q_L,y_L,eta_choice)
                

                #Fathom nodes only if the examined node was robust feasible
                if robust_feas_node and not bf_shaky:
                    self.fathoming(z_bf)  
                
                #Check for time limit termination
                self.time_limit_termination(time_begin_wall,z_bf,z_bp,count)

                #Update best possible solution based on the lowest lower bound
                z_min = [z_lp[n_w] for n_w in s_w]               
                if z_min:
                    z_bp=min(z_min)  


                #Evaluate relative optimality gap 
                self._gap_opt[n_c]=abs(z_bp-z_bf)/abs(z_bp+0.0000001)

                #If new cuts are added I need to break to update the list of current nodes
                if new_cuts:
                    print('I have to exit current node set')
                    break
        #Output for maximum iterations exceeded                            
        time_end=time.process_time()
        elapsed_time=time_end-time_begin
        self.result_export(elapsed_time,'Maximum iterations exceeded',z_bf)
        return (z_bf,elapsed_time,self.viol_counter,count,self._qp_timer,self._lp_timer,self._mp_timer)
