import numpy as np
import pyomo.environ as pe

from pooling_network.network import Network
from pooling_network.pooling import (nominal_quality, constant_perturbation,product_quality,
    compute_gamma_ijk,
    compute_gamma_lower_ijk,
    index_set_ij,index_set_ijk,
    index_set_ilj,
    index_set_il,
    index_set_jk,
    index_set_l,
    index_set_lj,
    index_set_j,
    index_set_i
)


def _to_capacity(n):
    if n is None:
        return np.inf
    return n

def minimize_cost_objective(b: pe.Block, problem: Network):
    b.cost = pe.Objective(expr=(
                sum(
                    (problem.nodes[i].cost - problem.nodes[j].cost) * b.v[i, l, j]
                    for (i, l, j) in index_set_ilj(problem)
                ) + sum(
                    (problem.nodes[i].cost - problem.nodes[j].cost) * b.z[i, j]
                    for (i, j) in index_set_ij(problem)
                )
        ))
    return b.cost

def minimize_flow_cost_objective(b: pe.Block, problem: Network):
    b.cost = pe.Objective(expr=(
                sum(
                    (problem.edges[i, l].cost + problem.edges[l, j].cost) * b.v[i, l, j]
                    for (i, l, j) in index_set_ilj(problem)
                ) + sum(
                    problem.edges[i, j].cost * b.z[i, j]
                    for (i, j) in index_set_ij(problem)
                )
        ))
    return b.cost

def quality_lower_dual_box(b: pe.Block, problem: Network, _pert, _psi):
    c_nom = nominal_quality(problem)
    c_pert = constant_perturbation(problem, _pert)
    p_lower, _ = product_quality(problem)

    # Define the constraint for each (j, k) in the index set
    def quality_lower_dual_box_rule(m, j, k):
        expr = 0
        flow = 0
        for l in [ll for ll, jj in index_set_lj(problem) if jj == j]:
            flow += b.y[l, j]
            for i in [ii for ii, ll in index_set_il(problem) if ll == l]:
                expr += c_nom[(i, k)] * b.v[(i, l, j)] + _psi * c_pert[(i, k)] * b.v[(i, l, j)]

        for i in [ii for ii, jj in index_set_ij(problem) if jj == j]:
            flow += b.z[i, j]
            expr += c_nom[(i, k)] * b.z[(i, j)] + _psi * c_pert[(i, k)] * b.z[(i, j)]

        return p_lower[j, k] * flow - expr <= 0

    # Apply the constraint over the index set
    b.quality_lower_dual_box = pe.Constraint(index_set_jk(problem), rule=quality_lower_dual_box_rule)
    return b.quality_lower_dual_box    

def quality_upper_dual_box(b: pe.Block, problem: Network, _pert, _psi):
    c_nom = nominal_quality(problem)
    c_pert = constant_perturbation(problem, _pert)
    _, p_upper = product_quality(problem)

    def quality_upper_dual_box_rule(m, j, k):
        expr = 0
        flow = 0
        for l in [ll for ll, jj in index_set_lj(problem) if jj == j]:
            flow += b.y[l, j]
            for i in [ii for ii, ll in index_set_il(problem) if ll == l]:    
                expr += c_nom[(i, k)] * b.v[(i, l, j)] + _psi * c_pert[(i, k)] * b.v[(i, l, j)]
        for i in [ii for ii, jj in index_set_ij(problem) if jj == j]:
            flow += b.z[i, j]
            expr += c_nom[(i, k)] * b.z[(i, j)] + _psi * c_pert[(i, k)] * b.z[(i, j)]
        return expr <= p_upper[j, k] * flow

    # Create the constraint and assign it to the block
    b.quality_upper_dual_box = pe.Constraint(index_set_jk(problem), rule=quality_upper_dual_box_rule)

    return b.quality_upper_dual_box

def quality_lower_dual_poly(b: pe.Block, problem: Network, _pert, _gammak):
    c_nom = nominal_quality(problem)
    c_pert = constant_perturbation(problem, _pert)
    p_lower, _ = product_quality(problem)
    
    # Define the constraint for each (j, k) in the index set
    def quality_lower_dual_poly_rule(m, j, k):
        expr = 0
        flow = 0
        for l in [ll for ll, jj in index_set_lj(problem) if jj == j]:
            flow += b.y[l, j]
            for i in [ii for ii, ll in index_set_il(problem) if ll == l]:
                expr += c_nom[(i, k)] * b.v[(i, l, j)] 

        for i in [ii for ii, jj in index_set_ij(problem) if jj == j]:
            flow += b.z[i, j]
            expr += c_nom[(i, k)] * b.z[(i, j)] 

        return p_lower[j, k] * flow - expr- _gammak* b.lamda[(j,k)] <= 0

    # Apply the constraint over the index set
    b.quality_lower_dual_poly = pe.Constraint(index_set_jk(problem), rule=quality_lower_dual_poly_rule)
    return b.quality_lower_dual_poly 

def quality_upper_dual_poly(b: pe.Block, problem: Network, _pert, _gammak):
    c_nom = nominal_quality(problem)
    c_pert = constant_perturbation(problem, _pert)
    _, p_upper = product_quality(problem)

    def quality_upper_dual_poly_rule(m, j, k):
        expr = 0
        flow = 0
        for l in [ll for ll, jj in index_set_lj(problem) if jj == j]:
            flow += b.y[l, j]
            for i in [ii for ii, ll in index_set_il(problem) if ll == l]:    
                expr += c_nom[(i, k)] * b.v[(i, l, j)] 
        for i in [ii for ii, jj in index_set_ij(problem) if jj == j]:
            flow += b.z[i, j]
            expr += c_nom[(i, k)] * b.z[(i, j)]
        return expr + _gammak * b.lamda[(j,k)]<= p_upper[j, k] * flow

    # Create the constraint and assign it to the block
    b.quality_upper_dual_poly = pe.Constraint(index_set_jk(problem), rule=quality_upper_dual_poly_rule)

    return b.quality_upper_dual_poly

def quality_lamda_dual_poly(b: pe.Block, problem: Network, _pert):
    c_nom = nominal_quality(problem)
    c_pert = constant_perturbation(problem, _pert)    
    # Initialize the summation expression
    def quality_lamda_dual_poly_rule(m, i, j, k):
        expr = sum(
            c_pert[(i, k)]*b.v[(i, l, j)]
            for l in [ll for ll, jj in index_set_lj(problem) if jj == j]
            if (i, l, j) in b.v  # Include only valid v terms
        )

        # Handle the case where z[(i, j)] is active
        if (i, j) in b.z:
            # Check if both z[(i, j)] and some v[(i, l, j)] are active
            if any((i, l, j) in b.v for l in [ll for ll, jj in index_set_lj(problem) if jj == j]):
                # If both z and v are active, include both in the constraint
                expr += c_pert[(i, k)] * b.z[(i, j)]
            else:
                # If only z is active, include only the z term
                expr = c_pert[(i, k)] * b.z[(i, j)]


        # Final constraint with lamda[j, k] as the upper bound
        return expr <= b.lamda[j, k]
    # Create the constraint and assign it to the block
    b.quality_lamda_dual_poly = pe.Constraint(index_set_ijk(problem), rule=quality_lamda_dual_poly_rule)    

def quality_lower_nominal(b: pe.Block, problem: Network):
    c_nom = nominal_quality(problem)
    p_lower, _ = product_quality(problem)

    # Define the constraint for each (j, k) in the index set
    def quality_lower_nominal_rule(m, j, k):
        expr = 0
        flow = 0

        for l in [ll for ll, jj in index_set_lj(problem) if jj == j]:
            flow += b.y[l, j]
            for i in [ii for ii, ll in index_set_il(problem) if ll == l]:
                expr += c_nom[(i, k)] * b.v[(i, l, j)] 

        for i in [ii for ii, jj in index_set_ij(problem) if jj == j]:
            flow += b.z[i, j]
            expr += c_nom[(i, k)] * b.z[(i, j)] 

        return  expr >= p_lower[j, k] * flow
    

    b.quality_lower_nominal = pe.Constraint(index_set_jk(problem), rule=quality_lower_nominal_rule)
    return b.quality_lower_nominal  

def quality_upper_nominal(b: pe.Block, problem: Network):
    c_nom = nominal_quality(problem)
    _, p_upper = product_quality(problem)

    def quality_upper_nominal_rule(m, j, k):
        expr=0
        flow = 0

        for l in [ll for ll, jj in index_set_lj(problem) if jj == j]:
            flow += b.y[l, j]
            for i in [ii for ii, ll in index_set_il(problem) if ll == l]:    
                expr += c_nom[(i, k)] * b.v[(i, l, j)] 
        for i in [ii for ii, jj in index_set_ij(problem) if jj == j]:
            flow += b.z[i, j]
            expr += c_nom[(i, k)] * b.z[(i, j)] 

        return expr<= p_upper[j, k] * flow

    # Create the constraint and assign it to the block
    b.quality_upper_nominal = pe.Constraint(index_set_jk(problem), rule=quality_upper_nominal_rule)

    return b.quality_upper_nominal


def quality_lower_dual_ell(b: pe.Block, problem: Network, _pert, omega):
    c_nom = nominal_quality(problem)
    c_pert = constant_perturbation(problem, _pert)
    p_lower, _ = product_quality(problem)

    # Define the constraint for each (j, k) in the index set
    def quality_lower_dual_ell_rule(m, j, k):    
        expr1 = 0
        expr2= 0
        power_expr=0
        flow = 0

        for l in [ll for ll, jj in index_set_lj(problem) if jj == j]:
            flow += b.y[l, j]
        
        for i in [ii for ii, jj in index_set_ij(problem) if jj == j]:
            flow += b.z[i, j]        
        
        for i in index_set_i(problem):
            expr2=0
            use_v=True ; use_z=True
            #To facilitate the construction of the new summation
            if (i,j) not in index_set_ij(problem):#
                use_z=False

        
            for l in [ll for ll, jj in index_set_lj(problem) if jj == j]:
                use_v=True
                if (i, l, j) not in index_set_ilj(problem):
                    use_v=False
                
                if use_v:
                    expr1 += c_nom[(i, k)] * b.v[(i, l, j)] 
                    expr2 +=   b.v[(i, l, j)]
            if use_z: 
                expr1 += c_nom[(i, k)] * b.z[(i, j)] 
                expr2 +=   b.z[(i, j)]
            power_expr+=(c_pert[(i,k)]**2)*(expr2**2)
        return  (omega**2)*(power_expr)>= (p_lower[j, k] * flow-expr1)**2   
    

    
    def quality_lower_dual_ell_rule_sqrt(m, j, k):
        expr1 = 0
        expr2= 0
        power_expr=0
        flow = 0

        for l in [ll for ll, jj in index_set_lj(problem) if jj == j]:
            flow += b.y[l, j]
        
        for i in [ii for ii, jj in index_set_ij(problem) if jj == j]:
            flow += b.z[i, j]        
        
        for i in index_set_i(problem):
            expr2=0
            use_v=True ; use_z=True
            #To facilitate the construction of the new summation
            #if not any(i in tup for tup in list(index_set_ij(problem))): 
            if (i,j) not in index_set_ij(problem):
                use_z=False

        
            for l in [ll for ll, jj in index_set_lj(problem) if jj == j]:
                use_v=True
                if (i, l, j) not in index_set_ilj(problem):
                    use_v=False
                
                if use_v:
                    expr1 += c_nom[(i, k)] * b.v[(i, l, j)] 
                    expr2 +=   b.v[(i, l, j)]
            if use_z: # these should be added only once after the l summation is terminated
                expr1 += c_nom[(i, k)] * b.z[(i, j)] 
                expr2 +=   b.z[(i, j)]
            power_expr+=(c_pert[(i,k)]**2)*(expr2**2)

        return  omega*(pe.sqrt(power_expr))>= (p_lower[j, k] * flow-expr1)
    # Apply the constraint over the index set
    b.quality_lower_dual_ell = pe.Constraint(index_set_jk(problem), rule=quality_lower_dual_ell_rule_sqrt)#here I should be simplifying the sqrt one before I need to use the root
    return b.quality_lower_dual_ell  

def quality_upper_dual_ell(b: pe.Block, problem: Network, _pert, omega):
    c_nom = nominal_quality(problem)
    c_pert = constant_perturbation(problem, _pert)
    _, p_upper = product_quality(problem)

    def quality_upper_dual_ell_rule(m, j, k):
        expr1=0
        expr2=0
        power_expr=0
        flow = 0
        use_v=True ; use_z=True

        for l in [ll for ll, jj in index_set_lj(problem) if jj == j]:
            flow += b.y[l, j]
        
        for i in [ii for ii, jj in index_set_ij(problem) if jj == j]:
            flow += b.z[i, j]

        for i in index_set_i(problem):
            expr2=0
            use_v=True ; use_z=True
            if (i,j) not in index_set_ij(problem):
                use_z=False

        
            for l in [ll for ll, jj in index_set_lj(problem) if jj == j]:
                use_v=True
                if (i, l, j) not in index_set_ilj(problem):
                    use_v=False
                if use_v:
                    expr1 += c_nom[(i, k)] * b.v[(i, l, j)] 
                    expr2 +=   b.v[(i, l, j)]
            if use_z: 
                expr1 += c_nom[(i, k)] * b.z[(i, j)] 
                expr2 +=   b.z[(i, j)]
            power_expr+=(c_pert[(i,k)]**2)*(expr2**2)

        return (omega**2)*(power_expr) <= (p_upper[j, k] * flow-expr1)**2
    

def _q_bounds(problem: Network):
    def _bounds(m, i, l):
        edge = problem.edges[i, l]
        _, limit = edge.capacity
        if limit is None:
            return 0, 1.0
        assert not limit > 1.0
        return 0, limit
    return _bounds


def _y_bounds(problem: Network):
    def _bounds(m, l, j):
        edge = problem.edges[l, j]
        _, limit = edge.capacity
        limit = _to_capacity(limit)
        pool_cap = _to_capacity(problem.nodes[l].capacity_upper)
        output_cap = _to_capacity(problem.nodes[j].capacity_upper)
        input_cap = sum(
            _to_capacity(problem.nodes[i.name].capacity_upper) for i in problem.predecessors(l, layer=0)
        )
        return 0, np.min([limit, pool_cap, output_cap, input_cap])
    return _bounds


def _z_bounds(problem: Network):
    def _bounds(m, i, j):
        edge = problem.edges[i, j]
        _, limit = edge.capacity
        limit = _to_capacity(limit)
        input_cap = _to_capacity(problem.nodes[i].capacity_upper)
        output_cap = _to_capacity(problem.nodes[j].capacity_upper)
        return 0, np.min([limit, input_cap, output_cap])
    return _bounds


def pooling_problem_dual_pq_formulation(b: pe.Block, problem: Network, skip_product_quality: bool = False):
    # Scale all flows to [0, 1]
    b.q = pe.Var(index_set_il(problem), bounds=_q_bounds(problem))
    b.v = pe.Var(index_set_ilj(problem), bounds=(0, None))
    b.y = pe.Var(index_set_lj(problem), bounds=_y_bounds(problem))
    b.z = pe.Var(index_set_ij(problem), bounds=_z_bounds(problem))
    b.lamda=pe.Var(index_set_jk(problem), within=pe.Reals)#this may cause my problem to be unbounded ?

    @b.Constraint(index_set_ilj(problem))
    def path_definition(m, i, l, j):
        return m.v[i, l, j] == m.q[i, l] * m.y[l, j]

    @b.Constraint(index_set_l(problem))
    def simplex(m, l):
        return sum(m.q[i, l] for i, l_ in index_set_il(problem) if l_ == l) == 1.0

    @b.Constraint(index_set_lj(problem))
    def reduction_1(m, l, j):
        return sum(
            m.v[i, l, j]
            for i, l_, j_ in index_set_ilj(problem) if l_ == l and j_ == j
        ) == m.y[l, j]

    @b.Constraint(index_set_il(problem))
    def reduction_2(m, i, l):
        pool = problem.nodes[l]
        _, capacity = pool.capacity
        return sum(
            m.v[i, l, j]
            for i_, l_, j in index_set_ilj(problem) if l_ == l and i_ == i
        ) <= m.q[i, l] * capacity

    @b.Constraint(index_set_i(problem))
    def input_capacity(m, i):
        inp = problem.nodes[i]
        (lower, upper) = inp.capacity
        expr = sum(
            m.v[i, l, j] for i_, l, j in index_set_ilj(problem) if i_ == i
        ) + sum(
            m.z[i, j] for i_, j in index_set_ij(problem) if i_ == i
        )
        if lower is None:
            return expr <= upper
        if upper is None:
            return expr >= lower
        return pe.inequality(lower, expr, upper)

    @b.Constraint(index_set_l(problem))
    def pool_capacity(m, l):
        pool = problem.nodes[l]
        _, capacity = pool.capacity
        return sum(
            m.v[i, l, j] for i, l_, j in index_set_ilj(problem) if l_ == l
        ) <= capacity

    @b.Constraint(index_set_j(problem))
    def output_capacity(m, j):
        out = problem.nodes[j]
        (lower, upper) = out.capacity
        expr = sum(
            m.v[i, l, j] for i, l, j_ in index_set_ilj(problem) if j_ == j
        ) + sum(
            m.z[i, j] for i, j_ in index_set_ij(problem) if j_ == j
        )
        if lower is None:
            return expr <= upper
        if upper is None:
            return expr >= lower
        return pe.inequality(lower, expr, upper)

    if not skip_product_quality:
        @b.Constraint(index_set_jk(problem))
        def product_quality_upper_bound(m, j, k):
            out = problem.nodes[j]
            return sum(
                compute_gamma_ijk(problem.nodes[i], out, k) * m.v[i, l, j]
                for i, l, j_ in index_set_ilj(problem) if j_ == j
            ) + sum(
                compute_gamma_ijk(problem.nodes[i], out, k) * m.z[i, j]
                for i, j_ in index_set_ij(problem) if j_ == j
            ) <= 0

        @b.Constraint(index_set_jk(problem))
        def product_quality_lower_bound(m, j, k):
            out = problem.nodes[j]

            if 'quality_lower' not in out.attr:
                return pe.Constraint.Skip
            if out.attr['quality_lower'] is None:
                return pe.Constraint.Skip

            return sum(
                compute_gamma_lower_ijk(problem.nodes[i], out, k) * m.v[i, l, j]
                for i, l, j_ in index_set_ilj(problem) if j_ == j
            ) + sum(
                compute_gamma_lower_ijk(problem.nodes[i], out, k) * m.z[i, j]
                for i, j_ in index_set_ij(problem) if j_ == j
            ) >= 0

