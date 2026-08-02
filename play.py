"""
Play against a trained checkpoint from the console.

    python play.py --checkpoint checkpoints/model_latest.pt --human-color white
"""

import argparse

import chess
import torch

import config as cfg
from model import build_model, get_device
from mcts import run_mcts, select_action
from chess_env import index_to_move


def bot_move(board, net, device, simulations):
    root = run_mcts(board, net, device, num_simulations=simulations, add_noise=False)
    action = select_action(root, temperature=1e-3)  # greedy at play time
    return index_to_move(board, action)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_latest.pt")
    parser.add_argument("--human-color", choices=["white", "black"], default="white")
    parser.add_argument("--simulations", type=int, default=cfg.NUM_SIMULATIONS)
    args = parser.parse_args()

    device = get_device()
    net = build_model(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    print(f"Loaded checkpoint from iteration {ckpt.get('iteration', '?')}")

    board = chess.Board()
    human_is_white = args.human_color == "white"

    while not board.is_game_over(claim_draw=True):
        print(board)
        print()
        human_turn = (board.turn == chess.WHITE) == human_is_white

        if human_turn:
            move = None
            while move is None:
                try:
                    move_str = input("Your move (UCI, e.g. e2e4): ").strip()
                    candidate = chess.Move.from_uci(move_str)
                    if candidate in board.legal_moves:
                        move = candidate
                    else:
                        print("Illegal move, try again.")
                except ValueError:
                    print("Could not parse move, use UCI format like e2e4 or e7e8q.")
        else:
            print("Bot is thinking...")
            move = bot_move(board, net, device, args.simulations)
            print(f"Bot plays: {move.uci()}")

        board.push(move)

    print(board)
    print("Game over:", board.result(claim_draw=True))


if __name__ == "__main__":
    main()
