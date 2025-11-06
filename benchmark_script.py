"""
==============================================================================

Title:             Modified Recursive QAOA for Exact Max-Cut Solutions on Bipartite Graphs: Closing the Gap Beyond QAOA Limit
Subtitle:          Batch runner for QAOA / RQAOA experiments (parallel)
Repository:        https://github.com/vijeycreative/Modified_RQAOA
Version:           1.0.0
Date:              06/11/2025

Authors:           V Vijendran
Affiliations:      Centre for Quantum Technologies, National University of Singapore & A*STAR Quantum Innovation Centre
Contact:           vjqntm@gmail.com

Paper Reference:   arXiv:2408.13207
Corresponding Code: This script loads pre-generated graphs, evaluates
                    1D p=1 QAOA scans (with/without an "optimal" γ range),
                    runs RQAOA (with/without restricted angle region), and
                    saves per-instance results to disk. Work is parallelized
                    across processes.

Description
-----------
This driver:
  • Enforces single-thread BLAS backends (MKL/OpenBLAS/OMP/numexpr) *before*
    importing NumPy/SciPy to avoid oversubscription when used with
    `ProcessPoolExecutor`.
  • For each task (graph size N, indices p_idx, i), it:
      1) Loads the NetworkX graph from "graphs/G{N}_{p_idx}_{i}.gpickle".
      2) Builds a dense adjacency matrix and finds max |weight|.
      3) Runs a 1D brute-force scan over γ with β fixed at π/8 for:
           - a full γ range (0, π),
           - an "optimal" γ range (-π/(2·max|w|), +π/(2·max|w|)).
      4) Runs RQAOA on copies of the graph with/without restricted region.
      5) Saves a result dict to "sim_results/G{N}_{p_idx}_{i}.pkl".
  • Uses negative sign convention for QAOA cost to align with minimization.

Benchmark Variants
------------------
To run the same benchmark for **weighted graphs**, simply modify two lines:

    filepath = f"{GRAPHS_DIR}/WG{num_vertices}_{p_idx}_{i}.gpickle"
    save_filepath = f"{OUT_DIR}/WG{num_vertices}_{p_idx}_{i}.pkl"

That is, just add an extra `'W'` in front of `'G{num_vertices}'` in both
the input and output filenames. All other code remains identical.

Inputs / Outputs
----------------
Input graphs:    graphs/G{num_vertices}_{p_idx}_{i}.gpickle  (pickle of nx.Graph)
Output results:  sim_results/G{num_vertices}_{p_idx}_{i}.pkl (pickle of dict)

Environment
-----------
Set SIZES, P_INDICES, GRAPHS_DIR, OUT_DIR, and N_CORES as needed.

License
-------
MIT License © 2025 V. Vijendran

==============================================================================
"""

import os

# --- Set BLAS threads BEFORE importing numpy/scipy to avoid oversubscription ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pickle
import numpy as np
from functools import partial
from scipy.optimize import brute
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

from RQAOA import *   # Provides: QAOA_Expectation_Cost, GraphManager, RQAOA, ...
from utils import *   # Provides: graph_to_array, etc.


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
SIZES = (64, 128)           # Graph sizes to run
P_INDICES = range(1, 11)    # Indexing for .gpickle filenames; NOT QAOA depth
GRAPHS_DIR = "graphs"       # Input directory for graphs
OUT_DIR = "sim_results"     # Output directory for results
N_CORES = 22                # Parallel workers for ProcessPoolExecutor

# Ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)


def run_one(num_vertices: int, p_idx: int, i: int) -> str:
    """
    Process a single (num_vertices, p_idx, i) triple and write a result file.

    Workflow
    --------
    1) Load graph "graphs/G{num_vertices}_{p_idx}_{i}.gpickle".
       (For weighted graphs, use "WG{num_vertices}_" instead.)
    2) Build adjacency, compute max |weight| for γ-range scaling.
    3) Define p=1 QAOA objective with β fixed at π/8 and scan γ by brute force:
         • γ ∈ (0, π)
         • γ ∈ (−π/(2·max|w|), +π/(2·max|w|))   [“opt” range]
    4) Run RQAOA on the original graph (no restrict_region) and on a copy
       (restrict_region=True).
    5) Save:
         {
           "QAOA":  -cost_at_best_gamma_full_range,
           "QAOA*": -cost_at_best_gamma_opt_range,
           "RQAOA": rqaoa_cost,
           "RQAOA*": opt_rqaoa_cost
         }
       to "sim_results/G{num_vertices}_{p_idx}_{i}.pkl".
       (For weighted graphs, use "WG{num_vertices}_" instead.)

    Parameters
    ----------
    num_vertices : int
        Number of vertices N used in filename convention.
    p_idx : int
        First index in filename convention; unrelated to QAOA depth here.
    i : int
        Second index in filename convention.

    Returns
    -------
    str
        Path to the written result pickle.

    Notes
    -----
    • The β angle is fixed to π/8 in this scan; only γ is optimized.
    • Sign convention: `QAOA_Expectation_Cost` returns negative total cost so
      that optimizers can minimize it. We negate again for reporting.
    """
    # ------------------------------------------------------------------
    # For weighted graphs: change "G" → "WG" in both filepath and output.
    # ------------------------------------------------------------------
    filepath = f"{GRAPHS_DIR}/G{num_vertices}_{p_idx}_{i}.gpickle"
    with open(filepath, "rb") as f:
        graph = pickle.load(f)

    # RQAOA mutates the graph; keep an independent copy for the 'opt' run
    graph_copy = graph.copy(as_view=False)

    # Build (edges, adj_mat) and compute the scaling factor for γ
    edges, adj_mat = graph_to_array(graph)
    max_abs_weight = float(np.abs(adj_mat).max())

    # Ranges for gamma with and without the 'optimal' scaling
    range_gamma_opt = (-np.pi / (2 * max_abs_weight), np.pi / (2 * max_abs_weight))
    range_gamma = (0, np.pi)

    # Partially bind edges and adj_mat; hold β = π/8
    qaoa = partial(QAOA_Expectation_Cost, edges, adj_mat)

    def qaoa_beta_pi8(gamma):
        # `QAOA_Expectation_Cost` expects (gamma, beta)
        return qaoa((gamma, np.pi / 8))

    # --------------------------
    # 1D brute force over γ only
    # --------------------------
    opt_gamma1 = brute(lambda g: qaoa_beta_pi8(g[0]), ranges=(range_gamma,), Ns=20, finish=False)

    res1 = minimize(
            lambda g: qaoa_beta_pi8(g[0]),
            x0=opt_gamma1,
            method="L-BFGS-B",
            bounds=[range_gamma],
            options={"maxiter": 20000, "ftol": 1e-12}
        )
    opt_gamma11 = res1.x
    qaoa_cost = qaoa_beta_pi8(opt_gamma11[0])

    opt_gamma2 = brute(lambda g: qaoa_beta_pi8(g[0]), ranges=(range_gamma_opt,), Ns=20, finish=False)

    res2 = minimize(
            lambda g: qaoa_beta_pi8(g[0]),
            x0=opt_gamma2,
            method="L-BFGS-B",
            bounds=[range_gamma_opt],
            options={"maxiter": 20000, "ftol": 1e-12}
        )
    opt_gamma21 = res2.x

    opt_qaoa_cost = qaoa_beta_pi8(opt_gamma21[0])

    # --------------------------
    # RQAOA (with / without opt)
    # --------------------------
    gm1 = GraphManager(graph)
    rqaoa_cost, _ = RQAOA(gm1, (num_vertices * 2) - 8)

    gm2 = GraphManager(graph_copy)
    opt_rqaoa_cost, _ = RQAOA(gm2, (num_vertices * 2) - 8, True)

    # Keep previous sign convention for QAOA outputs
    results = {
        "QAOA": -qaoa_cost,
        "QAOA*": -opt_qaoa_cost,
        "RQAOA": rqaoa_cost,
        "RQAOA*": opt_rqaoa_cost
    }

    # ------------------------------------------------------------------
    # For weighted graphs: change "G" → "WG" in the filename below.
    # ------------------------------------------------------------------
    save_filepath = f"{OUT_DIR}/G{num_vertices}_{p_idx}_{i}.pkl"
    with open(save_filepath, "wb") as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    return save_filepath


def main():
    """
    Enumerate tasks and run them in parallel using `ProcessPoolExecutor`.

    Task grid
    ---------
    tasks = [(N, p_idx, i) for N in SIZES for (p_idx, i) in product(P_INDICES, P_INDICES)]

    Behavior
    --------
    • Submits each task to the pool with up to `N_CORES` workers.
    • Prints success/failure per task as results arrive.
    • Use the same structure to benchmark weighted graphs by simply adding
      the 'W' prefix in filenames as shown above.
    """
    tasks = [(nv, p, i) for nv in SIZES for p, i in product(P_INDICES, P_INDICES)]
    print(f"Submitting {len(tasks)} tasks across sizes {SIZES} using {N_CORES} cores...")

    with ProcessPoolExecutor(max_workers=N_CORES) as ex:
        futures = {ex.submit(run_one, nv, p, i): (nv, p, i) for nv, p, i in tasks}
        for fut in as_completed(futures):
            nv, p, i = futures[fut]
            try:
                out = fut.result()
                print(f"Done (N={nv}, p_idx={p}, i={i}) -> {out}")
            except Exception as e:
                print(f"FAILED (N={nv}, p_idx={p}, i={i}): {e}")


if __name__ == "__main__":
    main()
