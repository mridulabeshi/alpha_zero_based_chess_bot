"""
PUCT-based Monte Carlo Tree Search, guided by the policy/value network.

Each node corresponds to a chess position. We expand a node by asking the
network for a (policy, value) pair, mask/renormalize the policy over legal
moves, and store one child per legal move. Selection descends the tree
using the PUCT formula until we hit a leaf, then backs the value up.
"""

import math
import numpy as np
import torch
import chess

import config as cfg
from chess_env import board_to_tensor, legal_action_mask, move_to_action, index_to_move, game_result


class Node:
    __slots__ = ("parent", "prior", "children", "visit_count", "value_sum", "to_play")

    def __init__(self, parent, prior: float):
        self.parent = parent
        self.prior = prior
        self.children = {}   # action -> Node
        self.visit_count = 0
        self.value_sum = 0.0
        self.to_play = None  # set on expansion, for debugging/clarity only

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0


def _puct_score(parent: Node, child: Node, c_puct: float) -> float:
    u = c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
    # child.value is from the child's own to-move perspective; the parent
    # wants the negation of that (the opponent's good outcome is bad for us)
    q = -child.value
    return q + u


def _select_child(node: Node, c_puct: float):
    return max(node.children.items(), key=lambda item: _puct_score(node, item[1], c_puct))


@torch.no_grad()
def _evaluate_and_expand(node: Node, board: chess.Board, net, device):
    """Run the network on `board`, expand `node` with its legal children,
    and return the value estimate (from `board`'s side-to-move perspective)."""
    if board.is_game_over():
        return game_result(board)

    tensor = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(device)
    probs, value = net.predict(tensor)
    probs = probs.squeeze(0).cpu().numpy()
    value = float(value.item())

    mask = legal_action_mask(board)
    probs = probs * mask
    total = probs.sum()
    if total > 1e-8:
        probs = probs / total
    else:
        # extremely unlikely (all legal-move probability collapsed to ~0);
        # fall back to a uniform distribution over legal moves
        probs = mask / mask.sum()

    for action in np.flatnonzero(mask):
        node.children[int(action)] = Node(parent=node, prior=float(probs[action]))

    return value


def _add_dirichlet_noise(node: Node):
    actions = list(node.children.keys())
    noise = np.random.dirichlet([cfg.DIRICHLET_ALPHA] * len(actions))
    for a, n in zip(actions, noise):
        child = node.children[a]
        child.prior = (1 - cfg.DIRICHLET_EPSILON) * child.prior + cfg.DIRICHLET_EPSILON * n


def run_mcts(root_board: chess.Board, net, device, num_simulations=cfg.NUM_SIMULATIONS,
             add_noise=True) -> Node:
    root = Node(parent=None, prior=0.0)
    _evaluate_and_expand(root, root_board, net, device)
    if add_noise and root.is_expanded:
        _add_dirichlet_noise(root)

    for _ in range(num_simulations):
        node = root
        board = root_board.copy()
        path = [node]

        # 1. Selection
        while node.is_expanded and not board.is_game_over():
            action, node = _select_child(node, cfg.C_PUCT)
            board.push(index_to_move(board, action))
            path.append(node)

        # 2. Expansion + evaluation
        value = _evaluate_and_expand(node, board, net, device)

        # 3. Backpropagation. `value` is from the perspective of the side to
        # move at the leaf; it flips sign as it propagates up each ply.
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += value
            value = -value

    return root


def get_policy_distribution(root: Node, action_size=cfg.ACTION_SIZE, temperature=1.0) -> np.ndarray:
    """Convert visit counts at the root into a training-target policy pi."""
    pi = np.zeros(action_size, dtype=np.float32)
    actions = list(root.children.keys())
    counts = np.array([root.children[a].visit_count for a in actions], dtype=np.float64)

    if temperature <= 1e-3:
        best = actions[int(np.argmax(counts))]
        pi[best] = 1.0
        return pi

    counts = counts ** (1.0 / temperature)
    counts = counts / counts.sum()
    for a, p in zip(actions, counts):
        pi[a] = p
    return pi


def select_action(root: Node, temperature=1.0) -> int:
    pi = get_policy_distribution(root, temperature=temperature)
    nonzero = np.flatnonzero(pi)
    if temperature <= 1e-3:
        return int(nonzero[0])
    return int(np.random.choice(nonzero, p=pi[nonzero] / pi[nonzero].sum()))
