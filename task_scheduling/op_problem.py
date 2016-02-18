#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2015, lounick and decabyte
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of task_scheduling nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Orienteering problem solver

Implementation of an integer linear formulation for maximizing the targets visited by a vehicle under cost constraint.
The vehicle has to start and finish at the first point and it is allowed to skip targets.
Described in:
Vansteenwegen, Pieter, Wouter Souffriau, and Dirk Van Oudheusden. "The orienteering problem: A survey."
European Journal of Operational Research 209.1 (2011): 1-10.
"""

from __future__ import division

import numpy as np
from gurobipy import *


def _callback(model, where):
    """Callback function for the solver

    Callback function that adds lazy constraints for the optimisation process. Here it dynamically imposes cardinality
    constraints for the vertices in the solution, ensuring that if a path enters a vertex there must be a path exiting.

    Parameters
    ----------
    model : object
        The gurobi model instance
    where : int
        Gurobi specific callback variable

    Returns
    -------

    """
    if where == GRB.callback.MIPSOL:
        V = set(range(model._n))
        idx_start = model._idxStart
        idx_finish = model._idxFinish

        solmat = np.zeros((model._n, model._n))
        selected = []

        for i in V:
            sol = model.cbGetSolution([model._eVars[i, j] for j in V])
            selected += [(i, j) for j in V if sol[j] > 0.5]

            solmat[i, :] = sol

        if len(selected) <= 1:
            return

        for k in range(len(selected)):
            el = selected[k]
            entry = el[0]
            if idx_start != entry:
                expr1 = quicksum(model._eVars[i, entry] for i in V)
                expr2 = quicksum(model._eVars[entry, j] for j in V)

                model.cbLazy(expr1, GRB.EQUAL, expr2)


def op_solver(cost, profit=None, cost_max=None, idx_start=None, idx_finish=None, **kwargs):
    """Orienteering problem solver instance

    Cost constrained traveling salesman problem solver for a single vehicle using the Gurobi MILP optimiser.

    Parameters
    ----------
    cost : ndarray (n, dims)
        Cost matrix for traveling from point to point. Here is time (seconds) needed to go from points a to b.
    profit : Optional[vector]
        Profit vector for profit of visiting each point.
    cost_max : Optional[double]
        Maximum running time of the mission in seconds.
    idx_start : Optional[int]
        Optional starting point for the tour. If none is provided the first point of the array is chosen.
    idx_finish : Optional[int]
        Optional ending point of the tour. If none is provided the last point of the array is chosen.
    kwargs : Optional[list]
        Optional extra arguments/
    Returns
    -------
    route : list
        The calculated route.
    profit : double
        The profit of the route.
    m : object
        A gurobi model object.
    """

    # extra options
    output_flag = int(kwargs.get('output_flag', 0))
    time_limit = float(kwargs.get('time_limit', 60.0))
    w_coeff = kwargs.get('w_coeff', None)

    # Number of points
    n = cost.shape[0]

    # Check for default values
    if idx_start is None:
        idx_start = 0

    if idx_finish is None:
        idx_finish = n - 1

    if profit is None:
        profit = np.ones(n)

    if cost_max is None:
        cost_max = cost[idx_start, idx_finish]

    # Create the vertices set
    V = set(range(n))

    # Create model variables
    m = Model()
    e_vars = {}
    u_vars = {}

    for i in V:
        for j in V:
            e_vars[i, j] = m.addVar(vtype=GRB.BINARY, name='e_' + str(i) + '_' + str(j))

    m.update()

    for i in V:
        e_vars[i, i].ub = 0
    
    for i in V:
        u_vars[i] = m.addVar(vtype=GRB.INTEGER, name='u_' + str(i))

    m.update()

    # Set objective function (0)
    expr = 0

    # standard formulation
    for i in V:
        for j in V:
            expr += profit[j] * e_vars[i, j]

    # (optional) correlated formulation
    if w_coeff is not None:
        x_expr = []

        for j in V:
            x_j = quicksum(e_vars[i, j] for i in V)
            x_expr.append(x_j)

        for i in V:
            for j in V:                
                if w_coeff[i, j] > 0.1:
                    expr += profit[j] * w_coeff[i, j] * x_expr[i] * (x_expr[i] - x_expr[j])

    m.setObjective(expr, GRB.MAXIMIZE)
    m.update()

    # Constraints

    # Add constraints for the initial and final node (1)
    # None enters the starting point
    m.addConstr(quicksum(e_vars[j, idx_start] for j in V.difference([idx_start])) == 0, "s_entry")
    m.update()

    # None exits the finish point
    m.addConstr(quicksum(e_vars[idx_finish, j] for j in V.difference([idx_finish])) == 0, "f_exit")
    m.update()

    # Always exit the starting point
    m.addConstr(quicksum(e_vars[idx_start, i] for i in V.difference([idx_start])) == 1, "s_exit")
    m.update()

    # Always enter the finish point
    m.addConstr(quicksum(e_vars[i, idx_finish] for i in V.difference([idx_finish])) == 1, "f_entry")
    m.update()

    # From all other points someone may exit
    for i in V.difference([idx_start, idx_finish]):
        m.addConstr(quicksum(e_vars[i, j] for j in V if i != j) <= 1, "v_" + str(i) + "_exit")
    m.update()

    # To all other points someone may enter
    for i in V.difference([idx_start, idx_finish]):
        m.addConstr(quicksum(e_vars[j, i] for j in V if i != j) <= 1, "v_" + str(i) + "_entry")
    m.update()

    # Add cost constraints (3)
    expr = 0

    for i in V:
        for j in V:
            expr += cost[i, j] * e_vars[i, j]

    m.addConstr(expr <= cost_max, "max_energy")
    m.update()

    # Constraint (4)
    for i in V:
        u_vars[i].lb = 0
        u_vars[i].ub = n

    m.update()

    # Add subtour constraint (5)
    for i in V:
        for j in V:
            m.addConstr(
                u_vars[i] - u_vars[j] + 1, GRB.LESS_EQUAL, (n - 1)*(1 - e_vars[i, j]), 
                "sec_" + str(i) + "_" + str(j)
            )

    m.update()

    # Model Variables
    m._n = n
    m._eVars = e_vars
    m._uVars = u_vars
    m._idxStart = idx_start
    m._idxFinish = idx_finish
    m.update()

    m.params.OutputFlag = output_flag
    m.params.TimeLimit = time_limit
    m.params.LazyConstraints = 1
    m.optimize(_callback)

    solution = m.getAttr('X', e_vars)
    selected = [(i, j) for i in V for j in V if solution[i, j] > 0.5]

    # (optinal) show solution info
    if output_flag > 0:
        solmat = np.zeros((n, n), dtype=np.int)
        othmat = np.zeros((n, n))
        ulist = np.zeros(n)
        x_expr = []

        for k, var in m._uVars.iteritems():
            ulist[k] = var.X

        upairs = sorted(range(len(ulist)), key=lambda k: ulist[k])

        for k, v in solution.iteritems():
            solmat[k[0], k[1]] = v

        for j in V:
            x_j = sum(solution[i, j] for i in V)
            x_expr.append(x_j)

        print('')
        print('Solution Matrix:\n %s' % solmat)
        print('Selected Route: %s' % selected)
        print('Total Cost: %s' % sum(cost[s[0], s[1]] for s in selected))
        print('uVars: %s' % ulist)
        print('uVars pairs: %s' % upairs)
        print('')

        if w_coeff is not None:
            for i in V:
                for j in V:
                    othmat[i, j] = profit[j] * w_coeff[i, j] * x_expr[i] * (x_expr[i] - x_expr[j])

            print('X-expression: %s' % x_expr)
            print('C-matrix:\n %s' % othmat)
            print('')


    # calculate output route
    route = []
    next_city = idx_start

    while len(selected) > 0:
        for i in range(len(selected)):
            if selected[i][0] == next_city:
                route.append(next_city)
                next_city = selected[i][1]
                selected.pop(i)
                break

    route.append(next_city)

    return route, m.objVal, m


def main():
    import argparse
    import matplotlib.pyplot as plt
    import task_scheduling.utils as tsu
    
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    # see: https://mkaz.github.io/2014/07/26/python-argparse-cookbook/
    parser = argparse.ArgumentParser(description='Orienteering Problem.')
    parser.add_argument('rows', metavar='M', type=int, default=3, help='number of rows')
    parser.add_argument('cols', metavar='N', type=int, default=4, help='number of columns')
    parser.add_argument('cmax', metavar='C', type=int, default=500, nargs='?', help='maximum cost')

    parser.add_argument('--correlated', '-C', action='store_true', help='correlated formulation')
    parser.add_argument('--verbose', '-v', action='store_true', help='verbose flag')
    args = parser.parse_args()

    if args.verbose:
        output_flag = 1
    else:
        output_flag = 0

    # generate problem
    nodes = tsu.generate_random_grid(args.rows, args.cols)      # tsu.generate_nodes(n=args.nodes)
    cost = tsu.calculate_distances(nodes)

    if args.correlated:
        w_coeff = np.zeros_like(cost)

        # weighting
        d_max = 100
        g_cor = d_max / 3.0
        dist = tsu.calculate_distances(nodes)

        w_coeff = np.exp(-(dist / g_cor)**2)
        # w_coeff = np.clip(-0.01 * dist + 1, 0.0, 1.0)

        # zero-diag
        w_coeff[np.diag_indices(w_coeff.shape[0])] = 0.0
    else:
        w_coeff = None

    solution, objective, _ = tsu.solve_problem(
        op_solver, cost, 
        cost_max=args.cmax, w_coeff=w_coeff,
        output_flag=output_flag, time_limit=30.0
    )

    fig, ax = tsu.plot_problem(nodes, solution, objective)
    plt.show()

if __name__ == '__main__':
    main()