"""
==============================================================================

Title:             Modified Recursive QAOA for Exact Max-Cut Solutions on Bipartite Graphs: Closing the Gap Beyond QAOA Limit
Subtitle:          Recursive QAOA (RQAOA) utilities for weighted Max-Cut / Ising models
Repository:        [GitHub URL]
Version:           1.0.0
Date:              [YYYY-MM-DD]

Authors:           V Vijendran
Affiliations:      Centre for Quantum Technologies, National University of Singapore & A*STAR Quantum Innovation Centre
Contact:           vjqntm@gmail.com

Paper Reference:   arXiv:2408.13207
Corresponding Code: This module provides the core expectation routines for
                    QAOA at depth p=1 and the recursive elimination loop
                    (RQAOA), including a bounded local refinement step.

Description
-----------
This file contains:
  • Fast closed-form p=1 QAOA expectation evaluators for weighted graphs
    (cost for the whole graph and per-edge correlations).
  • A single-variable elimination step used in RQAOA that (anti-)correlates
    the most strongly correlated edge and reduces the instance.
  • A driver that repeats elimination until a target count is reached and
    then solves the residual instance by brute-force as implemented in
    `GraphManager.brute_force()`.

Implementation Notes
--------------------
• Expectation formulae:
  The functions `QAOA_Expectation_Cost` and `QAOA_Expectation_Edges` implement
  the standard p=1 closed-form expectation for weighted Ising/Max-Cut,
  separating contributions into:
      term1 – linear-in-sin parts from single-neighbor paths,
      term2 – quadratic-in-sin parts with products of cos factors and
              triangle/no-triangle structure.
  They support general (possibly signed) edge weights via `adj_mat[u, v]`.

• Performance:
  Both expectation functions are `@jit(nopython=True)` compiled with Numba.

• Angle search region:
  The `restrict_region` flag reduces the gamma/beta box to a physically
  relevant interior informed by the largest |edge weight| to avoid known
  degeneracies/aliasing and to help local optimizers. Otherwise the full
  (0, π) × (0, π) box is used.

How to Cite
-----------
If you use this code in academic work, please cite: arXiv:2408.13207

License
-------
[Choose a license, e.g., MIT / Apache-2.0 / BSD-3-Clause]

==============================================================================
"""

import sys
import numpy as np
from numba import jit
import networkx as nx
from scipy.optimize import brute, minimize
from functools import partial
from numpy import sin, cos

from utils import *


@jit(nopython=True)
def QAOA_Expectation_Cost(edges, adj_mat, angles):
    """
    Compute the (negative) expected objective value at p=1 for a weighted graph.

    This routine evaluates the closed-form p=1 QAOA expectation of the standard
    Ising/Max-Cut Hamiltonian on a graph with (possibly signed) weights.
    It returns the **negative** total cost so that numerical optimizers can
    *minimize* this function directly (equivalently maximizing the original cost).

    Parameters
    ----------
    edges : numpy.ndarray of shape (m, 2)
        List/array of undirected edges (u, v). Each edge must correspond to a
        nonzero weight in `adj_mat[u, v]`. It is assumed that (u, v) ∈ E iff
        adj_mat[u, v] != 0. (Duplicates are not expected.)
    adj_mat : numpy.ndarray of shape (n, n)
        Symmetric adjacency/weight matrix. Entries adj_mat[u, v] may be any real
        weight (including negative for anti-ferromagnetic couplings).
    angles : tuple(float, float)
        (gamma, beta) QAOA parameters for p=1.

    Returns
    -------
    float
        The negative of the total expected cost (so that `minimize` works out
        of the box). If you need the *positive* expected cost, multiply by -1.

    Notes
    -----
    • Internal structure:
        For each edge (u, v), the expectation of Z_u Z_v decomposes into
        a `term1` (∝ sin(4β) sin(γ w_uv) with neighbor-product cos factors)
        and a `term2` (∝ sin^2(2β) times products of cos over "non-triangle"
        edges and triangle-related cos combinations).
    • The final per-edge contribution is 0.5 * w_uv * (1 - ⟨Z_u Z_v⟩),
      which matches the usual cut-value mapping.

    References
    ----------
    • Farhi et al., "A Quantum Approximate Optimization Algorithm", 2014.
    • Crooks, "Performance of the Quantum Approximate Optimization Algorithm
      on the Maximum Cut Problem", 2018.
    """
    gamma, beta = angles
    edge_costs = {}
    
    for u, v in edges:
        # Find neighbors of u and v excluding each other
        # e: neighbors of v except u; d: neighbors of u except v
        eX = np.nonzero(adj_mat[v])[0]
        e = eX[np.where(eX != u)]
        dX = np.nonzero(adj_mat[u])[0]
        d = dX[np.where(dX != v)]
        
        # Common neighbors of u and v (triangles passing through u-v)
        F = np.intersect1d(e, d)
        
        # ----- term1: single-neighbor-path contribution -----
        # product_x cos(γ w_{xv}) + product_y cos(γ w_{uy})
        term1_cos1 = 1
        for x in e:
            term1_cos1 *= cos(gamma * adj_mat[x, v])
        term1_cos2 = 1
        for y in d:
            term1_cos2 *= cos(gamma * adj_mat[u, y])
        term1 = sin(4 * beta) * sin(gamma * adj_mat[u, v]) * (term1_cos1 + term1_cos2)
        
        # Build set E of "non-triangle" edges adjacent to (u, v)
        e_edges_non_triangle = [(x, v) for x in e if x not in F]
        d_edges_non_triangle = [(u, y) for y in d if y not in F]
        E = e_edges_non_triangle + d_edges_non_triangle
        E = list(set(E))  # ensure uniqueness

        # ----- term2: quadratic-in-sin part with cos products and triangle splits -----
        term2 = pow(sin(2 * beta), 2)
        # Product over non-triangle incident edges
        for x, y in E:
            term2 *= cos(gamma * adj_mat[x, y])

        # Triangle terms: split into (u,f)+(v,f) and (u,f)-(v,f)
        triangle_1_terms = 1
        triangle_2_terms = 1
        for f in F:
            triangle_1_terms *= cos(gamma * (adj_mat[u, f] + adj_mat[v, f]))
            triangle_2_terms *= cos(gamma * (adj_mat[u, f] - adj_mat[v, f]))
        term2 *= (triangle_1_terms - triangle_2_terms)
        
        # Expected correlator ⟨Z_u Z_v⟩
        ZuZv = -0.5 * (term1 + term2)

        # Edge contribution to expected cut value: (1 - ⟨Z_u Z_v⟩)/2 times the weight
        edge_costs[(u, v)] = 0.5 * adj_mat[u, v] * (1 - ZuZv)
        
    # Sum contributions across edges and return negative for minimization
    total_cost = 0
    for v in edge_costs.values():
        total_cost += v
    return -total_cost


@jit(nopython=True)
def QAOA_Expectation_Edges(edges, adj_mat, angles):
    """
    Compute p=1 QAOA per-edge correlators ⟨Z_u Z_v⟩ for all edges.

    This is identical to `QAOA_Expectation_Cost` in structure, but returns the
    raw correlators (⟨Z_u Z_v⟩) per edge instead of the combined total cost.
    These values are used by RQAOA to choose which edge to (anti-)correlate.

    Parameters
    ----------
    edges : numpy.ndarray of shape (m, 2)
        List/array of undirected edges (u, v).
    adj_mat : numpy.ndarray of shape (n, n)
        Symmetric adjacency/weight matrix.
    angles : tuple(float, float)
        (gamma, beta) QAOA parameters for p=1.

    Returns
    -------
    dict
        Mapping {(u, v): ⟨Z_u Z_v⟩} over edges provided.

    Notes
    -----
    • The sign and magnitude of ⟨Z_u Z_v⟩ determine which edge is the
      strongest (anti-)correlation candidate for elimination in RQAOA.
    """
    gamma, beta = angles
    edge_costs = {}
    
    for u, v in edges:
        # Find neighbors of u and v excluding each other
        eX = np.nonzero(adj_mat[v])[0]
        e = eX[np.where(eX != u)]
        dX = np.nonzero(adj_mat[u])[0]
        d = dX[np.where(dX != v)]
        
        # Common neighbors (triangles)
        F = np.intersect1d(e, d)
        
        # term1
        term1_cos1 = 1
        for x in e:
            term1_cos1 *= cos(gamma * adj_mat[x, v])
        term1_cos2 = 1
        for y in d:
            term1_cos2 *= cos(gamma * adj_mat[u, y])
        term1 = sin(4 * beta) * sin(gamma * adj_mat[u, v]) * (term1_cos1 + term1_cos2)
        
        # Non-triangle incident edges to (u, v)
        e_edges_non_triangle = [(x, v) for x in e if x not in F]
        d_edges_non_triangle = [(u, y) for y in d if y not in F]
        E = e_edges_non_triangle + d_edges_non_triangle
        E = list(set(E))

        # term2
        term2 = pow(sin(2 * beta), 2)
        for x, y in E:
            term2 *= cos(gamma * adj_mat[x, y])
        triangle_1_terms = 1
        triangle_2_terms = 1
        for f in F:
            triangle_1_terms *= cos(gamma * (adj_mat[u, f] + adj_mat[v, f]))
            triangle_2_terms *= cos(gamma * (adj_mat[u, f] - adj_mat[v, f]))
        term2 *= (triangle_1_terms - triangle_2_terms)
        
        ZuZv = -0.5 * (term1 + term2)
        edge_costs[(u, v)] = ZuZv

    return edge_costs
    

def eliminate_variable(graphmanager: GraphManager, restrict_region):
    """
    Single RQAOA elimination step: optimize angles, score edges, and (anti-)correlate.

    Workflow
    --------
    1) Extract the current reduced instance from `graphmanager`.
    2) Optimize p=1 angles (γ, β) in two stages:
        (a) Coarse grid search via `scipy.optimize.brute` (no finishing step).
        (b) Local refinement with L-BFGS-B in the same box (strictly interior x0).
       If `restrict_region` is True, shrink the domain using the maximum absolute
       edge weight to avoid problematic periodic aliases in γ.
    3) Evaluate per-edge correlators ⟨Z_u Z_v⟩ and pick the edge with the largest
       |correlation|. Its sign decides whether to correlate or anti-correlate.
    4) Log the decision and call the appropriate `graphmanager` method to reduce
       the instance (and advance the iteration counter externally).

    Parameters
    ----------
    graphmanager : GraphManager
        A stateful manager that exposes:
            • reduced_graph : networkx.Graph (weighted)
            • optimal_angles : dict-like, to store angles by iteration
            • log : dict-like, to append iteration logs
            • iter : int, current iteration counter
            • verbose : bool, toggles printing
            • anti_correlate(edge) / correlate(edge) : in-place reduction ops
            • brute_force() : final exact solver on the residual graph
    restrict_region : bool
        If True, set
           γ ∈ (-π / (2 * max|w|), +π / (2 * max|w|)),
           β ∈ [0, π/4],
        otherwise use the full box γ, β ∈ [0, π].

    Returns
    -------
    None
        The graph is reduced in-place via `graphmanager.(anti_)correlate`.

    Notes
    -----
    • The local refinement step is guarded and silently skipped upon failure.
    • The chosen edge is the one with the maximum absolute correlator magnitude.
    """
    # Extract list of edges and dense adjacency for fast JIT routines
    red_edges, adj_mat = extract_properties(graphmanager)

    # Partially apply edges and adj_mat so the optimizer only sees `angles`
    qaoa = partial(QAOA_Expectation_Cost, red_edges, adj_mat)

    # -------------------------
    # Angle search box selection
    # -------------------------
    num_samples = 20
    if restrict_region:
        max_abs_weight = np.abs(adj_mat).max()
        # guard against pathological zero (shouldn't happen if edges exist)
        if max_abs_weight == 0:
            max_abs_weight = 1.0
        range_gamma = (-np.pi / (2 * max_abs_weight), np.pi / (2 * max_abs_weight))
        range_beta = (0, np.pi / 4)
    else:
        range_gamma = (0, np.pi)
        range_beta  = (0, np.pi)

    # 1) Coarse grid search (finish=False prevents extra local search by brute)
    opt_angles = brute(qaoa, (range_gamma, range_beta), Ns=num_samples, finish=False)

    # 2) Local refinement (L-BFGS-B) with interior start point
    bounds = [range_gamma, range_beta]
    x0 = np.array(opt_angles, dtype=float)

    # Push strictly inside bounds to avoid boundary artifacts
    eps = 1e-9
    for k, (lo, hi) in enumerate(bounds):
        x0[k] = min(max(x0[k], lo + eps), hi - eps)

    try:
        res = minimize(
            qaoa,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 20000, "ftol": 1e-12}
        )
        if res.success and np.isfinite(res.fun):
            opt_angles = res.x
    except Exception:
        # If refinement fails for any reason, keep the coarse grid result
        pass

    # Record optimal angles for this iteration
    graphmanager.optimal_angles[graphmanager.iter] = opt_angles

    # Compute (negative) QAOA cost at the optimized angles for logging
    qaoa_cost = QAOA_Expectation_Cost(red_edges, adj_mat, opt_angles)

    # Rank edges by |⟨Z_u Z_v⟩| (strongest correlation/anti-correlation first)
    edge_costs = QAOA_Expectation_Edges(red_edges, adj_mat, opt_angles)
    edge_costs = {k: v for k, v in sorted(edge_costs.items(), key=lambda item: np.abs(item[1]), reverse=True)}

    # Select top edge and determine its sign
    edge, weight = list(edge_costs.items())[0]
    # Small epsilon protects against signed zeros
    sign = int(np.sign(sys.float_info.epsilon + weight))

    # Diagnostics: triangle counts for involved vertices
    num_triangles_u = nx.triangles(graphmanager.reduced_graph)[edge[0]]
    num_triangles_v = nx.triangles(graphmanager.reduced_graph)[edge[1]]

    # Apply the corresponding reduction and log
    if sign < 0:
        msg1 = f"QAOA Cost = {qaoa_cost}. Anti-Correlating Edge {edge} that has maximum absolute weight {weight}."
        msg2 = f"Node {edge[0]} and {edge[1]} were contained in {num_triangles_u} and {num_triangles_v} triangles respectively."
        graphmanager.log[graphmanager.iter] = graphmanager.log[graphmanager.iter] + msg1 + "\n" + msg2 + "\n"
        if graphmanager.verbose:
            print(msg1)
            print(msg2)
        graphmanager.anti_correlate(edge)
    elif sign > 0:
        msg1 = f"QAOA Cost = {qaoa_cost}. Correlating Edge {edge} that has maximum absolute weight {weight}."
        msg2 = f"Node {edge[0]} and {edge[1]} were contained in {num_triangles_u} and {num_triangles_v} triangles respectively."
        graphmanager.log[graphmanager.iter] = graphmanager.log[graphmanager.iter] + msg1 + "\n" + msg2 + "\n"
        if graphmanager.verbose:
            print(msg1)
            print(msg2)
        graphmanager.correlate(edge)
    else:
        # Extremely rare: true zero within numerical tolerance
        error_msg = f"Cannot correlate or anti-correlate edge {edge} for weight {weight}."
        graphmanager.log[graphmanager.iter] = graphmanager.log[graphmanager.iter] + error_msg + "\n"
        if graphmanager.verbose:
            print(error_msg)


def RQAOA(graphmanager: GraphManager, n, restrict_region = False):
    """
    Run RQAOA for `n` elimination steps and then brute-force the residual instance.

    This high-level driver repeats `eliminate_variable` until either `n` steps
    are completed or the reduced graph has no edges left. After the loop, it
    hands the final residual instance to `graphmanager.brute_force()` to obtain
    an exact assignment.

    Parameters
    ----------
    graphmanager : GraphManager
        A stateful problem manager (see `eliminate_variable` docstring for the
        required interface).
    n : int
        Number of elimination iterations to attempt. The loop stops early if
        the reduced graph becomes edgeless.
    restrict_region : bool, optional (default: False)
        If True, uses a more conservative γ/β domain tailored by the largest
        absolute weight (see `eliminate_variable`).

    Returns
    -------
    Any
        Whatever `graphmanager.brute_force()` returns (e.g., best cut value,
        assignment, etc.).

    Side Effects
    ------------
    • Updates `graphmanager.optimal_angles[iter]` at each iteration with the
      best (γ, β) discovered for that reduced instance.
    • Appends human-readable messages to `graphmanager.log[iter]`.
    • Mutates `graphmanager.reduced_graph` via (anti-)correlate operations.

    Notes
    -----
    • Logging/verbosity is controlled by `graphmanager.verbose`.
    • The loop variable `graphmanager.iter` is advanced after each elimination
      step to keep per-iteration artifacts (angles, logs) aligned.
    """
    # Initialize the iteration counter (local display only; the authoritative
    # iteration index is maintained on graphmanager.iter)
    i = 0

    # Repeat elimination steps up to `n`, stopping early if instance is trivial
    while i <= n:
        # If no edges remain, the instance is solved; exit early
        if graphmanager.reduced_graph.number_of_edges() == 0:
            break

        # Status log for the current reduced graph
        out_message = f"Iter {i}: Graph has {graphmanager.reduced_graph.number_of_nodes()} nodes and {graphmanager.reduced_graph.number_of_edges()} edges remaining."
        graphmanager.log[graphmanager.iter] = out_message + "\n"
        if graphmanager.verbose:
            print(out_message)

        # One variable-elimination step (optimize angles, pick edge, reduce)
        eliminate_variable(graphmanager, restrict_region)
        
        # Advance counters
        i += 1
        graphmanager.iter += 1

    # After eliminations, exactly solve the residual instance
    graphmanager.log[graphmanager.iter] = "\nBrute-Forcing\n"
    if graphmanager.verbose:
        print("\nBrute-Forcing")
    
    # Delegate to GraphManager for the final exact solution
    return graphmanager.brute_force()
