
from typing import Optional
import pyomo.environ as pe
from pyomo.core.base.block import declare_custom_block, BlockData
from pooling_network.cuts import add_valid_cuts
from pooling_network.inequalities import add_all_pooling_inequalities
from pooling_network.network import Network
from dual_pq import (quality_lower_dual_box, quality_upper_dual_box,quality_lower_dual_poly, quality_upper_dual_poly,quality_lamda_dual_poly,
    quality_lower_dual_ell, quality_upper_dual_ell,
    pooling_problem_dual_pq_formulation,
    minimize_cost_objective,
    minimize_flow_cost_objective, quality_lower_nominal, quality_upper_nominal
)


@declare_custom_block(name='PoolingPQFormulationDual')
class PoolingPQFormulationDualData(BlockData):
    def __init__(self, component):
        super().__init__(component)
        self._built = False
        self.pooling_problem: Optional[Network] = None
        self.inequalities = None
        self.pert=None
        self.psi=None



    def set_pooling_problem(self, problem: Network):
        self.pooling_problem = problem

    def rebuild(self, skip_product_quality=False):
        if self.pooling_problem is None:
            raise RuntimeError("Must set pooling_problem")
        
        self._built = True
        pooling_problem_dual_pq_formulation(
            b=self,
            problem=self.pooling_problem,
            skip_product_quality=skip_product_quality,
        )
        self.inequalities = None

    def add_objective(self, use_flow_cost=False):
        assert self._built
        if use_flow_cost:
            return minimize_flow_cost_objective(b=self, problem=self.pooling_problem)
        else:
            return minimize_cost_objective(b=self, problem=self.pooling_problem)

    def add_inequalities(self, add_inequalities=True, add_uxt=True):
        del self.inequalities
        self.inequalities = pe.Block()
        add_all_pooling_inequalities(
            self.inequalities, self, self.pooling_problem,
            add_variables=True,  # always add variables
            add_uxt=add_uxt,
            add_inequalities=add_inequalities
        )

    def add_cuts(self, add_inequalities=False):
        return add_valid_cuts(self.inequalities, self, self.pooling_problem, add_inequalities=add_inequalities)

    def define_uncertainty_parameters(self,psi,pert):
        self.pert=pert
        self.psi=psi

    def robust_uncertainty_lower_constraint(self,set_type):
        assert self._built
        if set_type=='box':
            return quality_lower_dual_box(b=self,problem=self.pooling_problem,_pert=self.pert,_psi=self.psi)
        elif set_type=='ellipse':
            return quality_lower_dual_ell(b=self,problem=self.pooling_problem,_pert=self.pert,omega=self.psi)
        elif set_type=='polyhedron':
            return quality_lower_dual_poly(b=self,problem=self.pooling_problem,_pert=self.pert,_gammak=self.psi)
    
    def robust_uncertainty_upper_constraint(self,set_type):
        assert self._built
        if set_type=='box':
            return quality_upper_dual_box(b=self,problem=self.pooling_problem,_pert=self.pert,_psi=self.psi)
        elif set_type=='ellipse':
            return quality_upper_dual_ell(b=self,problem=self.pooling_problem,_pert=self.pert,omega=self.psi)
        elif set_type=='polyhedron':
            return quality_upper_dual_poly(b=self,problem=self.pooling_problem,_pert=self.pert,_gammak=self.psi) 

    def nominal_quality_lower_constraint(self,set_type):
        assert self._built
        if set_type=='ellipse':
            return quality_lower_nominal(b=self,problem=self.pooling_problem)
    def nominal_quality_upper_constraint(self,set_type):
        assert self._built
        if set_type=='ellipse':
            return quality_upper_nominal(b=self,problem=self.pooling_problem)       
    
    def auxiliary_uncertainty_gamma_constraint(self,set_type):
        assert self._built
        if set_type=='polyhedron':
            return quality_lamda_dual_poly(b=self,problem=self.pooling_problem,_pert=self.pert)
        
                   

    @property
    def flow_input_to_pool_to_output(self):
        return self.v

    @property
    def flow_pool_to_output(self):
        return self.y

    @property
    def flow_input_to_output(self):
        return self.z

    @property
    def fractional_flow_input_to_pool(self):
        return self.q
