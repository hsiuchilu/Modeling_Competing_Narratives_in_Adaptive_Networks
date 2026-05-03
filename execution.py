"""
Simulation runner for the competing-narrative diffusion model.

This file initializes agents and zealots, repeatedly calls one-cascade diffusion from
model.py, and returns the diffusion and network-structure outcomes used in the paper.
Variable names follow the current manuscript: narrative, adoption/exposure, zealots,
social pressure, network adaptability, and collective attention.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import random

import networkx as nx
import numpy as np

from model import NARRATIVE_A, NARRATIVE_B, diffuse_narrative


def simulate_information_cascade(
    N: int,
    network: nx.Graph,
    steps: int,
    w: float = 0.1,
    alpha: float = 0.05,
    b1: float = 3.5,
    b2: float = 1.0,
    n_zealot_a: int = 0,
    n_zealot_b: int = 0,
    c: float = 0.2,
    h: float = 2.0,
    seed: int | None = None,
    **legacy_kwargs: Any,
) -> Tuple[List[int], List[int], float, Dict[str, float], np.ndarray, List[int], List[int], List[int]]:
    """
    Run repeated cascades of competing-narrative diffusion.

    Parameters
    ----------
    N : int
        Number of agents. The code assumes that network node ids are integers that
        can index the rows of Q, usually 0 to N-1.
    network : networkx.Graph
        Initial social network. This object is updated in place during simulation.
    steps : int
        Number of propagation rounds / cascades.
    w : float, default=0.1
        Network adaptability; controls the overall probability of disconnection and
        reconnection after rejection.
    alpha : float, default=0.05
        Sensitivity to social pressure in the opinion-update equation.
    b1 : float, default=3.5
        Hardness/softness in the adoption function. In the manuscript, this parameter
        operationalizes collective attention.
    b2 : float, default=1.0
        Hardness/softness in the disconnection function.
    n_zealot_a : int, default=0
        Number of zealots assigned to narrative A. Internally, narrative A is coded as 1.
    n_zealot_b : int, default=0
        Number of zealots assigned to narrative B. Internally, narrative B is coded as 0.
    c : float, default=0.2
        Probability that a broken tie is replaced through friend-of-friend closure;
        otherwise, reconnection is random.
    h : float, default=2.0
        Homophily parameter used when selecting a friend-of-friend connection.
    seed : int or None, default=None
        Optional random seed for reproducibility.
    **legacy_kwargs : dict
        Accepts older keyword names ``n_mali_act_1`` and ``n_mali_act_0`` for backward
        compatibility. They are mapped to ``n_zealot_a`` and ``n_zealot_b``.

    Returns
    -------
    accepted_counts : list[int]
        Number of agents accepting/exposed to the focal narrative in each cascade.
        Divide by N to obtain tau.
    narratives : list[int]
        Focal narrative in each cascade. 1 denotes narrative A and 0 denotes narrative B.
    modularity : float
        Modularity of the final network using the final A/B position partition.
    group_assortativity : dict[str, float]
        Numeric assortativity of Delta Q for the whole network and for each final
        position group. Keys are ``'A'``, ``'B'``, ``'1'``, ``'0'``, and ``'all'``.
        The numeric keys are retained for compatibility with earlier scripts.
    final_position : np.ndarray
        Boolean vector where True indicates narrative A and False indicates narrative B.
    degrees : list[int]
        Final node degree counts. These are raw degrees, not NetworkX degree centrality.
    zealots_a : list[int]
        Node ids of zealots assigned to narrative A.
    zealots_b : list[int]
        Node ids of zealots assigned to narrative B.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Backward compatibility with older code that used malicious-actor terminology.
    if "n_mali_act_1" in legacy_kwargs:
        n_zealot_a = legacy_kwargs.pop("n_mali_act_1")
    if "n_mali_act_0" in legacy_kwargs:
        n_zealot_b = legacy_kwargs.pop("n_mali_act_0")
    if legacy_kwargs:
        unknown = ", ".join(legacy_kwargs.keys())
        raise TypeError(f"Unknown keyword argument(s): {unknown}")

    nodes = np.array(list(network.nodes))
    if len(nodes) != N:
        raise ValueError("N must match the number of nodes in network")

    # Initialize each agent's conviction toward narratives B (column 0) and A (column 1).
    Q = np.random.uniform(-1, 1, size=(N, 2))

    # Initial position: True means the agent currently favors narrative A over B.
    initial_position_a = Q[:, NARRATIVE_A] > Q[:, NARRATIVE_B]

    # Select zealots from agents initially leaning toward their assigned narrative.
    # Zealots are fixed-position committed seeders, not necessarily bots or false-content actors.
    a_candidates = nodes[initial_position_a[nodes]]
    b_candidates = nodes[~initial_position_a[nodes]]

    if n_zealot_a > len(a_candidates) or n_zealot_b > len(b_candidates):
        raise ValueError("Requested more zealots than available initial supporters")

    zealots_a = [int(node) for node in np.random.choice(a_candidates, n_zealot_a, replace=False)]
    zealots_b = [int(node) for node in np.random.choice(b_candidates, n_zealot_b, replace=False)]

    # Fix zealots at extreme convictions so they persistently promote their assigned narrative.
    for node in zealots_a:
        Q[node, NARRATIVE_B] = -10
        Q[node, NARRATIVE_A] = 10
    for node in zealots_b:
        Q[node, NARRATIVE_A] = -10
        Q[node, NARRATIVE_B] = 10

    accepted_counts: List[int] = []
    narratives: List[int] = []

    for _ in range(steps):
        Q, network, accepted_count, narrative = diffuse_narrative(
            network=network,
            Q=Q,
            alpha=alpha,
            w=w,
            b1=b1,
            b2=b2,
            zealots_a=zealots_a,
            zealots_b=zealots_b,
            c=c,  # important: use the user-supplied c rather than a hard-coded value
            h=h,
        )
        accepted_counts.append(accepted_count)
        narratives.append(narrative)

    final_position = Q[:, NARRATIVE_A] > Q[:, NARRATIVE_B]
    delta_q = np.tanh(Q[:, NARRATIVE_A]) - np.tanh(Q[:, NARRATIVE_B])

    group_a = set(nodes[final_position[nodes]])
    group_b = set(nodes[~final_position[nodes]])

    modularity = nx.community.modularity(network, [group_b, group_a])
    nx.set_node_attributes(network, {node: float(delta_q[node]) for node in nodes}, "Q_conviction")

    group_assortativity: Dict[str, float] = {
        "B": nx.numeric_assortativity_coefficient(network, "Q_conviction", nodes=list(group_b)),
        "A": nx.numeric_assortativity_coefficient(network, "Q_conviction", nodes=list(group_a)),
        "all": nx.numeric_assortativity_coefficient(network, "Q_conviction"),
    }

    # Retain old numeric keys so existing analysis notebooks do not immediately break.
    group_assortativity["0"] = group_assortativity["B"]
    group_assortativity["1"] = group_assortativity["A"]

    degrees = [network.degree[node] for node in nodes]

    return (
        accepted_counts,
        narratives,
        modularity,
        group_assortativity,
        final_position,
        degrees,
        zealots_a,
        zealots_b,
    )
