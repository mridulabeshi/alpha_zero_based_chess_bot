"""
Play a full game of the network against itself using MCTS-guided moves,
collecting (state, policy, outcome) training examples.
"""

import chess
import numpy as np

import config as cfg
from chess_env import board_to_tensor, index_to_move
from mcts import run_mcts, get_policy_distribution, select_action


def play_one_game(net, device, num_simulations=cfg.NUM_SIMULATIONS):
    board = chess.Board()
    history = []  # list of (tensor, pi, side_to_move) waiting for the final result
    ply = 0

    while not board.is_game_over(claim_draw=True) and ply < cfg.MAX_GAME_LENGTH:
        temperature = 1.0 if ply < cfg.TEMPERATURE_MOVES else 1e-3
        root = run_mcts(board, net, device, num_simulations=num_simulations, add_noise=True)

        pi = get_policy_distribution(root, temperature=1.0)  # store the "sharp-ish" full dist
        state = board_to_tensor(board)
        history.append((state, pi, board.turn))

        action = select_action(root, temperature=temperature)
        board.push(index_to_move(board, action))
        ply += 1

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        result_white = 0.0
    else:
        result_white = 1.0 if outcome.winner == chess.WHITE else -1.0

    examples = []
    for state, pi, side_to_move in history:
        z = result_white if side_to_move == chess.WHITE else -result_white
        examples.append((state, pi, z))

    return examples, board.result(claim_draw=True), ply


def generate_self_play_data(net, device, num_games=cfg.GAMES_PER_ITERATION,
                             num_simulations=cfg.NUM_SIMULATIONS, verbose=True):
    all_examples = []
    for g in range(num_games):
        examples, result, plies = play_one_game(net, device, num_simulations)
        all_examples.extend(examples)
        if verbose:
            print(f"  self-play game {g + 1}/{num_games}: result={result} plies={plies} "
                  f"examples={len(examples)}")
    return all_examples
