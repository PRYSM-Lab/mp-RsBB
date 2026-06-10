import numpy as np
import pyomo.environ as pe

from pooling_network.network import Network
from pooling_network.pooling import (
    compute_gamma_ijk,
    compute_gamma_lower_ijk,
    index_set_ik,
    index_set_ij,
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

def nominal_quality(problem: Network): 
    c_nom={}
    for i,k in index_set_ik(problem):
        inp=problem.nodes[i]
        c_nom[i,k]=inp.attr['quality'][k]
    return c_nom

def constant_perturbation(problem:Network, pert):
    c_pert={}
    c_nom=nominal_quality(problem)
    for i,k in index_set_ik(problem):
        c_pert[i,k]=c_nom[i,k]*pert
    return c_pert

def ksi_nom(problem: Network): 
    ksi_nom={}
    for i,k in index_set_ik(problem):
        ksi_nom[i,k]=0
    return ksi_nom

def ksi_bounds(problem:Network):
    ksi_lo={}
    ksi_up={}
    for i,k in index_set_ik(problem):
        ksi_lo[i,k]=-1
        ksi_up[i,k]=1
    return ksi_lo, ksi_up


def product_quality(problem: Network):
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

def ksi_bounds_constraint(model, i, k):
    return model.ksi_lo[i, k],model.ksi[i, k],model.ksi_up[i, k]

def pooling_problem_pq_formulationPyros(b: pe.Block, problem: Network, skip_product_quality: bool = False):
    b.q = pe.Var(index_set_il(problem), bounds=_q_bounds(problem))
    b.v = pe.Var(index_set_ilj(problem), bounds=(0, None))
    b.y = pe.Var(index_set_lj(problem), bounds=_y_bounds(problem))
    b.z = pe.Var(index_set_ij(problem), bounds=_z_bounds(problem))
    b.c=pe.Param(index_set_ik(problem),  initialize=nominal_quality(problem), mutable=True)
    b.c_nom=pe.Param(index_set_ik(problem),  initialize=nominal_quality(problem), mutable=True)
    b.c_pert=pe.Param(index_set_ik(problem),  initialize=constant_perturbation(problem,1), mutable=True)
    b.ksi=pe.Param(index_set_ik(problem),  initialize=ksi_nom(problem), mutable=True)
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
            expr=0; flow=0
            for l in [ll for ll, jj in index_set_lj(problem) if jj == j]:
                for i in [ii for ii, ll in index_set_il(problem) if ll == l]:    
                    expr += (m.c_nom[i,k]+m.ksi[i,k]*m.c_pert[i,k])*m.v[i,l, j]
                    flow += m.v[i,l,j]
            for i in [ii for ii, jj in index_set_ij(problem) if jj == j]:
                flow += m.z[i, j]
                expr += (m.c_nom[i,k]+m.ksi[i,k]*m.c_pert[i,k])*m.z[(i, j)]  
            quality=product_quality(problem)[1][j,k]
            cut_expr=expr- quality*flow                          
            return cut_expr<=0

        @b.Constraint(index_set_jk(problem))
        def product_quality_lower_bound(m, j, k):
            out = problem.nodes[j]
            expr=0; flow=0
            if 'quality_lower' not in out.attr:
                return pe.Constraint.Skip
            if out.attr['quality_lower'] is None:
                return pe.Constraint.Skip

            for l in [ll for ll, jj in index_set_lj(problem) if jj == j]:
                for i in [ii for ii, ll in index_set_il(problem) if ll == l]:    
                    expr += (m.c_nom[i,k]+m.ksi[i,k]*m.c_pert[i,k])*m.v[i,l, j]
                    flow += m.v[i,l,j]
            for i in [ii for ii, jj in index_set_ij(problem) if jj == j]:
                flow += m.z[i, j]
                expr += (m.c_nom[i,k]+m.ksi[i,k]*m.c_pert[i,k])*m.z[(i, j)]  
            quality=product_quality(problem)[0][j,k]   
            cut_expr= quality*flow-expr                         
            return cut_expr<=0

