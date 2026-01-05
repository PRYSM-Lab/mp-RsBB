import pyomo.environ as pe
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from ppopt.mp_solvers.solve_mpqp import solve_mpqp, mpqp_algorithm
from call_ppopt import generate_mpPooling
import pandas as pd
from timeit import default_timer as timer
import logging
import pandas as pd
import pickle
import os
#To prevent printing pyomo warnings e.g. SolverStatus infeasible
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

class mpPooling:
    '''
    Class to solve multi-parametric formulation of the robust-lower-level pooling problem.

    Arguments: 
    model_LP (obj): Pyomo model for LP problem
    instance (str): Pooling problem instance from pooling-network library     
    my_path (str): Local path for results export
    pert_ (scalar): Uncertainty set size
    unc_set_ (str): Uncertainty set type ('box'/'ellipse'/'polyhedron')
    mp_algo (str): Name of multi-parametric programming algorithm from PPOPT

    Returns
    Excel and pickle files containing the critical regions of all lowe-level problems.

    '''
    def __init__(self,instance,model_LP,my_path,pert_,unc_set_,mp_algo):
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
        #Define mpP algorithm
        self._mpalgo = getattr(mpqp_algorithm, mp_algo)         
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
        #Initialise the uncertainty realisation
        self.nom_unc_real=True
        #Critical regions evaluated by PPOPT
        self._critical_regions={}
        #MPP solutions
        self._mp_solutions={}
    
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

    def solve_mp(self,modelMP):
        '''
        Solve the MPP problem

        Arguments:
        :param: modelMP: multi-parametric model

        Returns:
        The obtained critical regions, solution object from ppopt and elapsed time
        '''
        start=timer()
        #Call ppopt to solve the examined problem
        mp_solution=solve_mpqp(modelMP,self._mpalgo)
        end=timer()
        elapsed_time=end-start
        #Load critical regions
        critical_regions=mp_solution.critical_regions
        _critical_regions={}
        for i, cr in enumerate(critical_regions):
            region_data = {
                'A': cr.A,
                'b': cr.b,
                'E': cr.E,
                'f': cr.f
            }
            _critical_regions[i] = region_data
        return _critical_regions,mp_solution, elapsed_time   
    
    def mp_generate_regions(self):
        '''
        Generate and solve the MPP problem for all robust lower-level problems of the examined instance

        Returns:
        Filename of saved critical regions, number of critical regions and ellapsed time
        '''
        mLP=self.LP
        self.initialise_bounds()
        qbounds=self._q_bounds
        ybounds=self._y_bounds
        zbounds=self._z_bounds
        #Evaluate both for upper and lower quality constraints
        for rp in range(2):
            solve_up=rp    
            for j,k in mLP.pooling.product_quality_upper_bound.index_set():
                #Generate the MPP
                mpP,c_nom=generate_mpPooling(self.name,j,k,solve_up,self.define_unc,self.pert,qbounds,ybounds,zbounds)
                if self.nom_unc_real==True:#store nominal uncertainty value
                    self.unc_real[-1]=c_nom
                    self.nom_unc_real=False
                #Solve the MPP
                c_region,mp_sol,el_time=self.solve_mp(mpP)
                #Append the critical regions and PPOPT solution objects
                self._critical_regions[j,k,solve_up]=c_region
                self._mp_solutions[j,k,solve_up]=mp_sol
        saved_file,num_cr=self.save_critical_regions()

        return saved_file, num_cr,el_time
    
    def save_critical_regions(self):
        '''
        Export the obtained critical regions
        
        Returns:
        Excel and pickle files containing all MPP solutions for each pooling problem
        '''
        pert=str(self.pert)
        algo=str( self._mpalgo)
        filename='ξ'+pert+'_'+self.name + '_' + self.define_unc + '_' + algo+'_critical_regions.pkl'
        full_path = os.path.join(self._base_path, filename)
        # #Save critical regions to pickle
        with open(full_path, 'wb') as f:
            pickle.dump(self._critical_regions, f)

        #Load generated regions and save critical region parameters to excel
        with open(full_path, 'rb') as f:
            self._critical_regions = pickle.load(f)

        #Save analytical critical regions to excel
        cr_save=self._base_path
        with pd.ExcelWriter(cr_save, engine="xlsxwriter") as writer:
            used_sheet_names = {}

            for (j, k, up), regions in self._critical_regions.items():
                # Create safe and unique sheet name
                base_name = f"{j}_{k}_up{up}"[:25]  # Reserve space for suffix
                sheet_name = base_name
                startrow = 0
                for region_id, region_data in regions.items():

                    # Inside your loop, replace the existing line with:
                    if sheet_name not in used_sheet_names:
                        worksheet = writer.book.add_worksheet(sheet_name)
                        writer.sheets[sheet_name] = worksheet
                        used_sheet_names[sheet_name] = worksheet
                    else:
                        worksheet = used_sheet_names[sheet_name]
                    worksheet.write(startrow, 0, f"Region {region_id}")

                    for key in ["A", "b", "E", "f"]:
                        data = region_data[key]
                        df = pd.DataFrame(data)
                        worksheet.write(startrow + 1, 0, f"Matrix {key}")
                        df.to_excel(writer, sheet_name=sheet_name, startrow=startrow + 2, startcol=0, header=False, index=False)
                        startrow += df.shape[0] + 3  # Leave space between matrices

                    # Extra gap between regions
                    startrow += 2

        return filename,len(self._critical_regions)


