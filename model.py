"""
Core functions for the competing-narrative diffusion model.

This module implements one diffusion cascade in an adaptive network. The model is
written in the language used in the current manuscript: agents adopt and propagate
one of two competing narratives; committed seeders are called zealots; and the model
is generic rather than disinformation-specific.

Narrative coding
----------------
The original code used 0/1 internally. To preserve compatibility with earlier
simulation output, this version keeps that coding:

    narrative = 1  -> narrative A
    narrative = 0  -> narrative B

The functions therefore accept integer narrative labels {0, 1}, while comments and
variable names use the manuscript terminology.
"""

from __future__ import annotations

import math
import random
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np


NARRATIVE_A = 1
NARRATIVE_B = 0


def opposite_narrative(narrative: int) -> int:
    """Return the competing narrative label."""
    if narrative not in (NARRATIVE_A, NARRATIVE_B):
        raise ValueError("narrative must be coded as 0 or 1")
    return 1 - narrative


def opinion_climate(spreader: int, Q: np.ndarray, network: nx.Graph, narrative: int) -> float:
    """
    Calculate the local opinion climate surrounding the current spreader.

    The opinion climate is the balance between neighbors who currently support the
    focal narrative and neighbors who support the competing narrative. It follows the
    manuscript's supplemental definition:

        delta_i = (n_supportive - n_opposing) / (n_supportive + n_opposing)

    The raw balance is then transformed through a sigmoid function so that the
    returned social feedback lies approximately in [-1, 1]. This value is used to
    update the spreader's conviction toward the narrative being propagated.

    Parameters
    ----------
    spreader : int
        Node id of the current spreader. The code assumes node ids can index Q.
    Q : np.ndarray, shape (N, 2)
        Opinion-conviction matrix. Q[i, 1] is conviction toward narrative A and
        Q[i, 0] is conviction toward narrative B.
    network : networkx.Graph
        Current adaptive social network.
    narrative : int
        Focal narrative being propagated in this cascade. Use 1 for narrative A
        and 0 for narrative B.

    Returns
    -------
    float
        Social-feedback reward in approximately [-1, 1]. Positive values indicate
        a supportive local climate for the focal narrative; negative values indicate
        an opposing local climate.
    """
    neighbors = list(network.neighbors(spreader))
    if len(neighbors) == 0:
        raw_climate = 0.0
    else:
        competing = opposite_narrative(narrative)
        n_supportive = sum(Q[j, narrative] > Q[j, competing] for j in neighbors)
        n_opposing = sum(Q[j, narrative] < Q[j, competing] for j in neighbors)

        if n_supportive + n_opposing == 0:
            raw_climate = 0.0
        else:
            raw_climate = (n_supportive - n_opposing) / (n_supportive + n_opposing)

    # Manuscript supplement: I_i = l((1 + exp(-k delta_i))^-1 - 1/2), with k=5, l=2.
    return 2 * ((1 + math.exp(-5 * raw_climate)) ** -1 - 0.5)


def adoption_probability(conviction: float, b: float = 3.5) -> float:
    """
    Probability that a receiving agent accepts and propagates the focal narrative.

    The input is the receiver's bounded conviction toward the focal narrative. In the
    simulation, raw Q values are transformed using tanh before entering this function,
    so conviction is typically in [-1, 1]. The parameter b corresponds to b1 in the
    manuscript: the hardness/softness of the softmax-like acceptance function and the
    operationalized level of collective attention.
    """
    return math.exp(b * conviction) / (math.exp(b * 1) + math.exp(b * -1))


def disconnection_probability(
    conviction_distance: float,
    w: float = 0.1,
    b: float = 1.0,
) -> float:
    """
    Probability that a rejecting receiver disconnects from the spreader.

    Parameters
    ----------
    conviction_distance : float
        Absolute difference in bounded conviction, |Delta Q_receiver - Delta Q_spreader|.
        With tanh-transformed Q values, the theoretical range is [0, 4].
    w : float
        Network adaptability. Higher values make rewiring more frequent.
    b : float
        b2 in the manuscript. Higher values make disconnection more sensitive to
        large conviction differences.
    """
    return w * math.exp(b * conviction_distance) / (math.exp(b * 4) + math.exp(b * 0))


def _choose_friend_of_friend(
    node: int,
    network: nx.Graph,
    Q_tanh: np.ndarray,
    h: float = 2.0,
) -> int | None:
    """
    Select a new tie from friends-of-friends using conviction similarity.

    This implements the supplemental reconnection rule: with probability c, an agent
    attempts local closure by connecting to an indirect friend. Candidate probability
    is weighted by similarity in Delta Q raised to the homophily parameter h.
    """
    direct_neighbors = set(network.neighbors(node))
    candidates = set()
    for neighbor in direct_neighbors:
        candidates.update(network.neighbors(neighbor))

    candidates.discard(node)
    candidates.difference_update(direct_neighbors)

    if not candidates:
        return None

    candidates = list(candidates)
    delta_node = Q_tanh[node, NARRATIVE_A] - Q_tanh[node, NARRATIVE_B]
    delta_candidates = np.array(
        [Q_tanh[j, NARRATIVE_A] - Q_tanh[j, NARRATIVE_B] for j in candidates]
    )

    # Similarity term theta_ij = 1 - |Delta Q_i - Delta Q_j| / 4, clipped for safety.
    theta = 1 - np.abs(delta_node - delta_candidates) / 4
    weights = np.clip(theta, 0, None) ** h

    if weights.sum() == 0:
        return random.choice(candidates)

    probabilities = weights / weights.sum()
    return int(np.random.choice(candidates, p=probabilities))


def _choose_random_non_neighbor(node: int, network: nx.Graph) -> int | None:
    """Select a random node that is not already connected to the focal node."""
    excluded = set(network.neighbors(node)) | {node}
    candidates = [n for n in network.nodes if n not in excluded]
    if not candidates:
        return None
    return int(random.choice(candidates))


def diffuse_narrative(
    network: nx.Graph,
    Q: np.ndarray,
    alpha: float,
    w: float,
    b1: float,
    b2: float,
    zealots_a: Sequence[int] | None = None,
    zealots_b: Sequence[int] | None = None,
    c: float = 0.2,
    h: float = 2.0,
) -> Tuple[np.ndarray, nx.Graph, int, int]:
    """
    Run one cascade of competing-narrative diffusion on an adaptive network.

    Algorithmic sequence
    --------------------
    1. Randomly choose the focal narrative, A or B.
    2. Select a starting spreader. If zealots for the focal narrative exist, the
       cascade starts from one of them; otherwise, it starts from an agent whose
       current conviction favors the focal narrative.
    3. The spreader receives local social feedback from the opinion climate and
       updates Q, unless the spreader is a zealot.
    4. Each neighbor decides whether to accept and propagate the focal narrative.
    5. Receivers who reject may disconnect from the spreader based on conviction
       distance and network adaptability.
    6. Broken ties are immediately replaced either through friend-of-friend closure
       with probability c or by random reconnection.
    7. The cascade continues until no new agents accept and propagate the narrative.

    Returns
    -------
    Q : np.ndarray
        Updated conviction matrix.
    network : networkx.Graph
        Updated adaptive network.
    accepted_count : int
        Number of unique agents exposed/accepted in this cascade. This corresponds
        to the numerator of tau before dividing by N.
    narrative : int
        Focal narrative of this cascade. 1 denotes narrative A; 0 denotes narrative B.
    """
    zealots_a = list(zealots_a or [])
    zealots_b = list(zealots_b or [])
    all_zealots = set(zealots_a) | set(zealots_b)

    narrative = int(np.random.randint(2))
    focal_zealots = zealots_a if narrative == NARRATIVE_A else zealots_b

    if focal_zealots:
        spreader = int(np.random.choice(focal_zealots))
    else:
        competing = opposite_narrative(narrative)
        nodes = np.array(list(network.nodes))
        eligible = nodes[Q[nodes, narrative] > Q[nodes, competing]]
        if len(eligible) == 0:
            eligible = nodes
        spreader = int(np.random.choice(eligible))

    # accepted_agents records all agents who have accepted/exposed the focal narrative
    # in this cascade. spread_queue records accepted agents waiting to become spreaders.
    accepted_agents: List[int] = [spreader]
    spread_queue: List[int] = []

    while True:
        # 1. Social feedback updates the spreader's conviction, except for zealots.
        if spreader not in all_zealots:
            reward = opinion_climate(spreader, Q, network, narrative)
            Q[spreader, narrative] = (1 - alpha) * Q[spreader, narrative] + alpha * reward

        Q_tanh = np.tanh(Q)
        neighbors = list(network.neighbors(spreader))

        # 2. Neighbors decide whether to accept and propagate the narrative.
        newly_accepted: List[int] = []
        for receiver in neighbors:
            p_accept = adoption_probability(Q_tanh[receiver, narrative], b=b1)
            if np.random.rand() < p_accept and receiver not in accepted_agents:
                newly_accepted.append(receiver)

        spread_queue.extend(newly_accepted)
        accepted_agents.extend(newly_accepted)

        # 3. Receivers who did not accept may disconnect and immediately form a new tie.
        rejected_receivers = [node for node in neighbors if node not in accepted_agents]
        spreader_delta = Q_tanh[spreader, NARRATIVE_A] - Q_tanh[spreader, NARRATIVE_B]

        for receiver in rejected_receivers:
            receiver_delta = Q_tanh[receiver, NARRATIVE_A] - Q_tanh[receiver, NARRATIVE_B]
            distance = abs(receiver_delta - spreader_delta)
            p_disconnect = disconnection_probability(distance, w=w, b=b2)

            if np.random.rand() < p_disconnect and network.has_edge(spreader, receiver):
                network.remove_edge(spreader, receiver)

                if np.random.rand() < c:
                    new_neighbor = _choose_friend_of_friend(receiver, network, Q_tanh, h=h)
                else:
                    new_neighbor = _choose_random_non_neighbor(receiver, network)

                if new_neighbor is not None and new_neighbor != receiver:
                    network.add_edge(receiver, new_neighbor)

        # 4. Stop when there are no newly accepted agents waiting to spread.
        if not spread_queue:
            break

        spreader = int(random.choice(spread_queue))
        spread_queue.remove(spreader)

    return Q, network, len(set(accepted_agents)), narrative


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

def accept_prop(q: float, b: float = 3.5) -> float:
    """Backward-compatible alias for adoption_probability()."""
    return adoption_probability(q, b=b)


def break_prop(q: float, w: float = 0.1, b: float = 1.0) -> float:
    """Backward-compatible alias for disconnection_probability()."""
    return disconnection_probability(q, w=w, b=b)


def dis_information(
    network: nx.Graph,
    Q: np.ndarray,
    alpha: float,
    w: float,
    b1: float,
    b2: float,
    mali_act_1: Sequence[int] | None = None,
    mali_act_0: Sequence[int] | None = None,
    c: float = 0.2,
    h: float = 2.0,
) -> Tuple[np.ndarray, nx.Graph, int, int]:
    """
    Backward-compatible wrapper for older scripts.

    Older versions used the names ``dis_information`` and ``mali_act_*``. The current
    manuscript uses the more general terms competing narratives and zealots. Here,
    ``mali_act_1`` is treated as zealots for narrative A and ``mali_act_0`` as zealots
    for narrative B.
    """
    return diffuse_narrative(
        network=network,
        Q=Q,
        alpha=alpha,
        w=w,
        b1=b1,
        b2=b2,
        zealots_a=mali_act_1,
        zealots_b=mali_act_0,
        c=c,
        h=h,
    )
