"""
==============================================================================

Title:             Modified Recursive QAOA for Exact Max-Cut Solutions on Bipartite Graphs: Closing the Gap Beyond QAOA Limit
Subtitle:          Graph utilities and manager for RQAOA (Max-Cut / Ising)
Repository:        https://github.com/vijeycreative/Modified_RQAOA
Version:           1.0.0
Date:              06/11/2025

Authors:           V Vijendran
Affiliations:      Centre for Quantum Technologies, National University of Singapore & A*STAR Quantum Innovation Centre
Contact:           vjqntm@gmail.com

Paper Reference:   arXiv:2408.13207
Corresponding Code: This module provides graph helpers and a stateful
                    GraphManager used by RQAOA to perform variable
                    elimination, maintain mappings, and solve the
                    residual instance exactly via brute force.

Description
-----------
This file contains:
  • `has_edge`: membership test for undirected edges.
  • `graph_to_array`: convert a NetworkX graph to an edge list and a
     dense adjacency matrix.
  • `GraphManager`: stateful helper that tracks the original and reduced
     instances, performs (anti-)correlation eliminations, stores node
     maps and assignments, and computes/returns best solutions.
  • `extract_properties`: export the current reduced instance to the
     (edges, dense adjacency) format used by QAOA evaluators.

License
-------
MIT License © 2025 V. Vijendran

==============================================================================
"""


import sys
import itertools
import numpy as np
import networkx as nx


def has_edge(edge, edge_list):
    """
    Check whether an undirected edge is present in an edge list.

    In an undirected graph, (u, v) and (v, u) refer to the same edge.
    This helper checks membership under that symmetry.

    Parameters
    ----------
    edge : tuple[int, int]
        A candidate edge (u, v).
    edge_list : Collection[tuple[int, int]]
        A collection/list of edges.

    Returns
    -------
    bool
        True if `edge` or its reversed ordering exists in `edge_list`.
    """
    # Accept either orientation to respect undirected semantics
    return (edge in edge_list) or (edge[::-1] in edge_list)


def graph_to_array(graph):
    """
    Convert a NetworkX weighted graph to (edge array, adjacency matrix).

    Parameters
    ----------
    graph : networkx.Graph
        Undirected (weighted) graph. Edge weights are expected to be stored
        under the 'weight' attribute, as commonly used for Max-Cut / Ising.

    Returns
    -------
    tuple
        (edges, adj_mat) where
          • edges : np.ndarray of shape (m, 2)
              Array of edges (u, v) using node integer labels.
          • adj_mat : np.ndarray of shape (n, n)
              Dense adjacency/weight matrix ordered by the sorted node list.

    Notes
    -----
    The node list is sorted to yield a deterministic matrix ordering.
    """
    node_list = list(graph.nodes())
    node_list.sort()

    # Convert to a dense adjacency/weight matrix using a fixed node order
    adj_mat = nx.to_numpy_array(graph, node_list)

    # The edge array uses the graph's native node labels
    return np.array(graph.edges()), adj_mat


class GraphManager:
    """
    Stateful manager used by RQAOA to track reductions and reconstruct solutions.

    Responsibilities
    ----------------
    1) Preserve the original input graph for final evaluation.
    2) Maintain a reduced working graph during variable elimination steps,
       applying (anti-)correlation mappings and updating edges/weights.
    3) Track node-to-node mappings (with ± signs) so eliminated variables
       can be consistently propagated back to the original problem.
    4) Provide brute-force solving of the residual instance and compute the
       resulting original cost.

    Attributes
    ----------
    original_graph : networkx.Graph
        Immutable copy of the input graph for final scoring.
    reduced_graph : networkx.Graph
        Mutable working graph updated after each elimination.
    verbose : bool
        If True, print actions while logging them.
    nodes_vals : dict[int, int]
        Assignment dictionary in {+1, -1} for all nodes (fully populated at end).
    node_maps : dict[int, tuple[int, int]]
        Map from node -> (mapped_node, sign), where sign ∈ {+1, -1}.
        Root nodes map to themselves with sign +1.
    remaining_nodes : list[int]
        Nodes not yet eliminated.
    iter : int
        Iteration counter (advanced externally by the RQAOA driver).
    optimal_angles : dict[int, tuple[float, float]]
        Per-iteration (γ, β) storage for diagnostics/record-keeping.
    log : dict[int, str]
        Per-iteration human-readable log.

    Notes
    -----
    • This class assumes edge weights are stored under 'weight'.
    • Some methods (e.g., eliminate_node) expect node attributes like
      node 'weight' to exist if they are used by your pipeline.
    """

    def __init__(self, graph, verbose=False):
        # Keep a pristine copy for final scoring
        self.original_graph = graph.copy()

        # Working graph that is mutated by (anti-)correlate operations
        self.reduced_graph = graph

        # Controls printing of actions in addition to logging
        self.verbose = verbose

        # Initialize per-node assignments with zeros (unknown)
        self.nodes_vals = {i: 0 for i in range(graph.number_of_nodes())}

        # Initialize identity mappings: each node maps to itself with +1
        self.node_maps = {i: (i, 1) for i in range(graph.number_of_nodes())}

        # Track which nodes remain in the reduced instance
        self.remaining_nodes = [i for i in range(graph.number_of_nodes())]

        # Iteration index and per-iteration artifacts
        self.iter = 0
        self.optimal_angles = {}
        self.log = {}

    def correlate(self, edge):
        """
        Apply correlation u = +v for edge (u, v) and eliminate u from the graph.

        Effect
        ------
        • Set node_maps[u] = (v, +1) and remove u.
        • Rewire edges (w, u) → (w, v) with weight aggregation.
        • Remove zero-weight edges that may arise from cancellations.
        • Log all structural updates.

        Parameters
        ----------
        edge : tuple[int, int]
            Edge (u, v) to correlate; must exist in `self.reduced_graph`.

        Raises
        ------
        AssertionError
            If (u, v) is not present in the reduced graph.
        """
        # Get the vertices u and v (edge must exist)
        u, v = edge
        assert self.reduced_graph.has_edge(u, v), f"Graph does not contain edge ({u},{v})."

        # d: neighbors of u other than v
        d = [w for w in self.reduced_graph[u]]
        d.remove(v)

        # Map u to v with +1 correlation and remove u from active set
        self.node_maps[u] = (v, 1)
        self.remaining_nodes.remove(u)

        rm_msg1 = f"Removing edge ({v}, {u}) with weight {self.reduced_graph[v][u]['weight']} from graph."
        self.log[self.iter] = self.log[self.iter] + rm_msg1 + "\n"
        if self.verbose:
            print(rm_msg1)

        # Remove the (u, v) edge
        self.reduced_graph.remove_edge(v, u)

        # Capture old (w, u) weights before removal
        old_weights = {w: self.reduced_graph[w][u]['weight'] for w in d}

        # Remove all edges incident to u and log them
        for w in d:
            rm_edge_msg = f"Removing edge ({w}, {u}) with weight {self.reduced_graph[w][u]['weight']} from graph."
            self.log[self.iter] = self.log[self.iter] + rm_edge_msg + "\n"
            if self.verbose:
                print(rm_edge_msg)
            self.reduced_graph.remove_edge(w, u)

        # Remove node u from the working graph
        rm_node_msg = f"Removing node {u} from graph."
        self.log[self.iter] = self.log[self.iter] + rm_node_msg + "\n"
        if self.verbose:
            print(rm_node_msg)
        self.reduced_graph.remove_node(u)

        # Build new (w, v) edges using the saved weights
        new_edges = {(w, v): old_weights[w] for w in d}

        # If (w, v) already exists, aggregate weights
        for new_edge in new_edges:
            if self.reduced_graph.has_edge(new_edge[0], new_edge[1]):
                new_edges[new_edge] += self.reduced_graph[new_edge[0]][new_edge[1]]['weight']

        # Commit new/updated edges, dropping exact zeros
        for new_edge, weight in new_edges.items():
            if weight == 0.0:
                rm_new_edge_msg = f"Removing edge {new_edge} with weight {weight} from graph."
                self.log[self.iter] = self.log[self.iter] + rm_new_edge_msg + "\n"
                if self.verbose:
                    print(rm_new_edge_msg)
                self.reduced_graph.remove_edge(new_edge[0], new_edge[1])
            else:
                add_new_edge_msg = f"Adding edge {new_edge} with weight {weight} to graph."
                self.log[self.iter] = self.log[self.iter] + add_new_edge_msg + "\n"
                if self.verbose:
                    print(add_new_edge_msg)
                self.reduced_graph.add_edge(new_edge[0], new_edge[1], weight=weight)

    def anti_correlate(self, edge):
        """
        Apply anti-correlation u = −v for edge (u, v) and eliminate u.

        Same procedure as `correlate`, except new edges pick up a minus sign
        when their weights are transferred from u to v.

        Parameters
        ----------
        edge : tuple[int, int]
            Edge (u, v) to anti-correlate; must exist in `self.reduced_graph`.

        Raises
        ------
        AssertionError
            If (u, v) is not present in the reduced graph.
        """
        u, v = edge
        assert self.reduced_graph.has_edge(u, v), f"Graph does not contain edge ({u},{v})."

        # d: neighbors of u other than v
        d = [w for w in self.reduced_graph[u]]
        d.remove(v)

        # Map u to v with −1 correlation and remove u from active set
        self.node_maps[u] = (v, -1)
        self.remaining_nodes.remove(u)

        rm_msg1 = f"Removing edge ({v}, {u}) with weight {self.reduced_graph[v][u]['weight']} from graph."
        self.log[self.iter] = self.log[self.iter] + rm_msg1 + "\n"
        if self.verbose:
            print(rm_msg1)
        self.reduced_graph.remove_edge(v, u)

        # Save old weights, then remove (w, u) edges
        old_weights = {w: self.reduced_graph[w][u]['weight'] for w in d}
        for w in d:
            rm_edge_msg = f"Removing edge ({w}, {u}) with weight {self.reduced_graph[w][u]['weight']} from graph."
            self.log[self.iter] = self.log[self.iter] + rm_edge_msg + "\n"
            if self.verbose:
                print(rm_edge_msg)
            self.reduced_graph.remove_edge(w, u)

        rm_node_msg = f"Removing node {u} from graph."
        self.log[self.iter] = self.log[self.iter] + rm_node_msg + "\n"
        if self.verbose:
            print(rm_node_msg)
        self.reduced_graph.remove_node(u)

        # New edges inherit a minus sign due to anti-correlation
        new_edges = {(w, v): -old_weights[w] for w in d}

        # Aggregate with existing (w, v), if any
        for new_edge in new_edges:
            if self.reduced_graph.has_edge(new_edge[0], new_edge[1]):
                new_edges[new_edge] += self.reduced_graph[new_edge[0]][new_edge[1]]['weight']

        # Commit edges (drop exact zeros)
        for new_edge, weight in new_edges.items():
            if weight == 0.0:
                rmv_new_edge_msg = f"Removing edge {new_edge} with weight {weight} from graph."
                self.log[self.iter] = self.log[self.iter] + rmv_new_edge_msg + "\n"
                if self.verbose:
                    print(rmv_new_edge_msg)
                self.reduced_graph.remove_edge(new_edge[0], new_edge[1])
            else:
                add_new_edge_msg = f"Adding edge {new_edge} with weight {weight} to graph."
                self.log[self.iter] = self.log[self.iter] + add_new_edge_msg + "\n"
                if self.verbose:
                    print(add_new_edge_msg)
                self.reduced_graph.add_edge(new_edge[0], new_edge[1], weight=weight)

    def eliminate_node(self, node, sign):
        """
        Eliminate a node with a given sign by folding its incident weights into neighbors.

        This routine is independent of a specific (u, v) pair; it removes `node`
        and transfers its incident edge weights to each neighbor's node attribute
        'weight' with the provided sign.

        Parameters
        ----------
        node : int
            Node to eliminate from `self.reduced_graph`.
        sign : int
            +1 or −1 indicating how incident edge weights are accumulated
            into neighbor node weights.

        Side Effects
        ------------
        • Updates neighbor node attributes 'weight'.
        • Removes all incident edges and the node itself.
        • Updates `nodes_vals`, `remaining_nodes`, and the log.
        """
        # Gather current neighbors of `node`
        neighbours = [w for w in self.reduced_graph[node]]

        # Push weighted contributions to neighbors, then remove incident edges
        for neighbour in neighbours:
            self.reduced_graph.nodes[neighbour]['weight'] += sign * self.reduced_graph[node][neighbour]['weight']
            rmv_edge_msg = f"Removing edge {(node, neighbour)} with weight {self.reduced_graph[node][neighbour]['weight']} from graph."
            self.log[self.iter] = self.log[self.iter] + rmv_edge_msg + "\n"
            if self.verbose:
                print(rmv_edge_msg)
            self.reduced_graph.remove_edge(node, neighbour)

        rm_node_msg = f"Removing node {node} with weight {self.reduced_graph.nodes[node]['weight']} from graph."
        self.log[self.iter] = self.log[self.iter] + rm_node_msg + "\n"
        if self.verbose:
            print(rm_node_msg)

        # Record assignment and remove from active set/graph
        self.nodes_vals[node] = sign
        self.remaining_nodes.remove(node)
        self.reduced_graph.remove_node(node)

    def get_root_node(self, node, s):
        """
        Follow node mappings recursively to find the root (non-eliminated) node.

        Parameters
        ----------
        node : int
            Eliminated node whose ultimate root is sought.
        s : int
            Running sign (±1) to track correlation sign flips along the path.

        Returns
        -------
        tuple[int, int]
            (root_node, sign) where root_node ∈ remaining_nodes and
            sign ∈ {+1, −1} is the cumulative correlation sign.

        Notes
        -----
        Eliminations can chain through already eliminated nodes; this function
        resolves such chains and accumulates the resulting sign.
        """
        mapped_tuple = self.node_maps[node]
        mapped_node, sign = mapped_tuple
        sign = sign * s  # accumulate sign

        # If mapped_node is still active, we found the root; otherwise recurse
        if (mapped_node in self.remaining_nodes):
            return mapped_node, sign
        else:
            return self.get_root_node(mapped_node, sign)

    def set_node_values(self, values):
        """
        Assign ±1 values to remaining nodes and propagate to eliminated nodes.

        Parameters
        ----------
        values : list[int]
            A list of ±1 values with length equal to the number of remaining nodes.

        Raises
        ------
        AssertionError
            If `values` has the wrong length or contains entries not in {±1}.
        """
        assert len(values) == len(self.remaining_nodes), "Number of values passed is not equal to the number of remaining nodes."
        for value in values:
            assert value == 1 or value == -1, "Values passed should be either 1 or -1."

        # Set assignments for active (remaining) nodes
        for i, value in enumerate(values):
            node = self.remaining_nodes[i]
            self.nodes_vals[node] = value

        # Propagate assignments to eliminated nodes via node_maps
        for node, mapped_tuple in self.node_maps.items():
            mapped_node, sign = mapped_tuple
            # Skip root nodes (mapped to themselves)
            if node != mapped_node:
                if mapped_node in self.remaining_nodes:
                    self.nodes_vals[node] = sign * self.nodes_vals[mapped_node]
                else:
                    root_node, s = self.get_root_node(mapped_node, sign)
                    self.nodes_vals[node] = s * self.nodes_vals[root_node]

    def compute_cost(self, graph):
        """
        Compute the Max-Cut (Ising) cost for a given graph under current assignments.

        Parameters
        ----------
        graph : networkx.Graph
            Graph whose cut value should be computed. Usually either
            `self.original_graph` or `self.reduced_graph`.

        Returns
        -------
        float
            Sum over edges of 0.5 * w_uv * (1 − x_u x_v), where x_u ∈ {±1}.

        Raises
        ------
        AssertionError
            If any node assignment is not in {±1}.
        """
        # Validate that all assignments have been resolved to ±1
        for value in self.nodes_vals.values():
            assert value == 1 or value == -1, "All nodes should have a value of either 1 or -1."

        total_cost = 0
        for edge in graph.edges():
            total_cost += 0.5 * graph[edge[0]][edge[1]]['weight'] * (1 - self.nodes_vals[edge[0]] * self.nodes_vals[edge[1]])

        return total_cost

    def brute_force(self):
        """
        Exhaustively search ±1 assignments for remaining nodes and return the best cost.

        Returns
        -------
        tuple[float, dict[int, int]]
            (best_cost_on_original_graph, nodes_vals_dictionary)

        Notes
        -----
        • Enumerates all 2^{|remaining_nodes|} assignments; only suitable for
          small residual instances.
        • Logs intermediate best reduced costs; final result is evaluated on
          `self.original_graph` after setting the best assignment.
        """
        num_values = len(self.remaining_nodes)
        assignments = list(map(list, itertools.product([1, -1], repeat=num_values)))

        best_reduced_cost = -sys.maxsize
        best_assignment = assignments[0]

        # Evaluate all assignments on the reduced graph
        for i, assignment in enumerate(assignments):
            self.set_node_values(assignment)
            reduced_cost = self.compute_cost(self.reduced_graph)

            if reduced_cost > best_reduced_cost:
                best_reduced_cost = reduced_cost
                best_assignment = assignment
                bf_msg = f"Best reduced cost found so far is {best_reduced_cost} for assignment {assignment}."
                self.log[self.iter] = self.log[self.iter] + bf_msg + "\n"
                if self.verbose:
                    print(bf_msg)

        # Commit the best assignment and evaluate on the original graph
        self.set_node_values(best_assignment)
        best_cost = self.compute_cost(self.original_graph)

        best_cost_msg = f"Best Cost found for the original problem is {best_cost}."
        self.log[self.iter] = self.log[self.iter] + best_cost_msg + "\n"
        if self.verbose:
            print(best_cost_msg)

        return best_cost, self.nodes_vals
    

def extract_properties(graphmanager: "GraphManager") -> tuple:
    """
    Export the current reduced instance to (edge array, dense adjacency matrix).

    Parameters
    ----------
    graphmanager : GraphManager
        The stateful manager from which the reduced graph and original size
        are obtained.

    Returns
    -------
    tuple
        (red_edges, adj_mat) where
          • red_edges : np.ndarray of shape (m, 2)
              Edge list of the current reduced graph.
          • adj_mat : np.ndarray of shape (N, N)
              Dense adjacency/weight matrix of size equal to the original
              number of nodes, with nonzero entries only for current reduced
              edges.

    Notes
    -----
    • The adjacency matrix is sized to the *original* node count so JIT
      routines can rely on a fixed shape across iterations.
    • Only entries corresponding to edges present in the reduced graph are
      filled (symmetrically); others remain zero.
    """
    red_edges, _ = graph_to_array(graphmanager.reduced_graph)

    # Fix the matrix size to the original problem dimension
    num_nodes = graphmanager.original_graph.number_of_nodes()
    adj_mat = np.zeros((num_nodes, num_nodes))

    # Populate symmetric weights for current reduced edges
    for u, v in red_edges:
        adj_mat[u][v] = graphmanager.reduced_graph[u][v]['weight']
        adj_mat[v][u] = graphmanager.reduced_graph[v][u]['weight']

    return red_edges, adj_mat
