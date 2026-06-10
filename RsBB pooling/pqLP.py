
import numpy as np
import pyomo.environ as pe

from pooling_network.network import Network
from pooling_network.pooling import (
    compute_gamma_ijk,
    compute_gamma_lower_ijk,
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


def q_bounds_dict(b: pe.Block, problem: Network):
    bounds_func = _q_bounds(problem)
    q_lo_={}; q_up_={}
    for i,l in index_set_il(problem):
        q_lo_[(i,l)]=bounds_func(b, i ,l )[0]
        q_up_[(i,l)]=bounds_func(b, i ,l )[1]
    return q_lo_,q_up_

def y_bounds_dict(b: pe.Block, problem: Network):
    bounds_func = _y_bounds(problem)
    y_lo_={}; y_up_={}
    for l,j in index_set_lj(problem):
        y_lo_[(l,j)]=bounds_func(b, l, j )[0]
        y_up_[(l,j)]=bounds_func(b, l, j )[1]
    return y_lo_,y_up_

def pooling_problem_pq_formulationLP(b: pe.Block, problem: Network, skip_product_quality: bool = False):
    # # Scale all flows to [0, 1]
    q_lo_,q_up_=q_bounds_dict(b,problem)
    y_lo_,y_up_=y_bounds_dict(b,problem)
    b.q_lo=pe.Param(index_set_il(problem),domain=pe.NonNegativeReals, initialize=q_lo_, mutable=True)
    b.q_up=pe.Param(index_set_il(problem),domain=pe.NonNegativeReals, initialize=q_up_, mutable=True)
    b.y_lo=pe.Param(index_set_lj(problem),domain=pe.NonNegativeReals, initialize=y_lo_, mutable=True)
    b.y_up=pe.Param(index_set_lj(problem),domain=pe.NonNegativeReals, initialize=y_up_, mutable=True)
    b.q = pe.Var(index_set_il(problem), bounds=_q_bounds(problem))
    b.v = pe.Var(index_set_ilj(problem), bounds=(0, None))
    b.y = pe.Var(index_set_lj(problem), bounds=_y_bounds(problem))
    b.z = pe.Var(index_set_ij(problem), bounds=_z_bounds(problem))

    # @b.Constraint(index_set_ilj(problem))
    # def path_definition(m, i, l, j):
    #     return m.v[i, l, j] == m.q[i, l] * m.y[l, j]
    
    #McCormick relaxations
    @b.Constraint(index_set_ilj(problem))
    def mccormick1_rule(m,i,l,j):
        return m.q_lo[(i,l)]*m.y[(l, j)]+ m.q[(i, l)]*m.y_lo[(l,j)]- m.q_lo[(i,l)]*m.y_lo[(l,j)]- m.v[(i,l,j)]<=0

    @b.Constraint(index_set_ilj(problem))
    def mccormick2_rule(m,i,l,j):
        return m.v[(i,l,j)]- m.q_up[(i,l)]*m.y[(l, j)]-m.q[(i, l)]*m.y_lo[(l,j)]+ m.q_up[(i,l)]*m.y_lo[(l,j)]<=0

    @b.Constraint(index_set_ilj(problem))
    def mccormick3_rule(m,i,l,j):
        return m.q_up[(i,l)]*m.y[(l, j)]+ m.q[(i, l)]*m.y_up[(l,j)]-m.q_up[(i,l)]*m.y_up[(l,j)]- m.v[(i,l,j)]<=0

    @b.Constraint(index_set_ilj(problem))
    def mccormick4_rule(m,i,l,j):
        return  m.v[(i,l,j)]- m.q_lo[(i,l)]*m.y[(l, j)]-m.q[(i, l)]*m.y_up[(l,j)]+ m.q_lo[(i,l)]*m.y_up[(l,j)]<=0

    
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
