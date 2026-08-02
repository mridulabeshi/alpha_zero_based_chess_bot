"""
Main AlphaZero training loop:

    for each iteration:
        1. self-play `GAMES_PER_ITERATION` games with the current network
           (MCTS provides both the move to play and a policy training target)
        2. push the resulting (state, pi, z) examples into a replay buffer
        3. sample minibatches and take gradient steps on
               loss = (z - v)^2  -  pi . log(p)  +  L2 regularization
        4. checkpoint the network

Run with:  python train.py
Resume with: python train.py --resume checkpoints/model_iter_50.pt
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import config as cfg
from model import build_model, get_device
from replay_buffer import ReplayBuffer
from self_play import generate_self_play_data


def compute_loss(policy_logits, value_pred, pi_target, z_target):
    value_loss = F.mse_loss(value_pred, z_target)
    # cross-entropy against a *distribution* target (not a class index)
    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_loss = -(pi_target * log_probs).sum(dim=1).mean()
    return value_loss + policy_loss, value_loss.item(), policy_loss.item()


def train_on_batches(net, optimizer, buffer: ReplayBuffer, device, epochs=cfg.EPOCHS_PER_ITERATION):
    net.train()
    steps_per_epoch = max(1, len(buffer) // cfg.BATCH_SIZE)
    total_loss, total_v, total_p, n = 0.0, 0.0, 0.0, 0

    for _ in range(epochs):
        for _ in range(steps_per_epoch):
            states, pis, zs = buffer.sample(cfg.BATCH_SIZE)
            states = torch.from_numpy(states).to(device)
            pis = torch.from_numpy(pis).to(device)
            zs = torch.from_numpy(zs).to(device)

            policy_logits, value_pred = net(states)
            loss, v_loss, p_loss = compute_loss(policy_logits, value_pred, pis, zs)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.GRAD_CLIP_NORM)
            optimizer.step()

            total_loss += loss.item()
            total_v += v_loss
            total_p += p_loss
            n += 1

    return total_loss / n, total_v / n, total_p / n


def save_checkpoint(net, optimizer, iteration, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "iteration": iteration,
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


def load_checkpoint(net, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    net.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["iteration"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=cfg.NUM_ITERATIONS)
    parser.add_argument("--games-per-iter", type=int, default=cfg.GAMES_PER_ITERATION)
    parser.add_argument("--simulations", type=int, default=cfg.NUM_SIMULATIONS)
    args = parser.parse_args()

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    device = get_device()
    print(f"Using device: {device}")

    net = build_model(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.LR_MILESTONES, gamma=cfg.LR_DECAY)

    start_iteration = 0
    if args.resume:
        start_iteration = load_checkpoint(net, optimizer, args.resume, device) + 1
        print(f"Resumed from {args.resume} at iteration {start_iteration}")

    buffer = ReplayBuffer()

    for iteration in range(start_iteration, args.iterations):
        t0 = time.time()
        print(f"\n=== Iteration {iteration} ===")

        print(f"Self-play: generating {args.games_per_iter} games "
              f"({args.simulations} MCTS sims/move)...")
        examples = generate_self_play_data(
            net, device, num_games=args.games_per_iter, num_simulations=args.simulations
        )
        buffer.add(examples)
        print(f"Buffer size: {len(buffer)}")

        if len(buffer) < cfg.MIN_BUFFER_SIZE_TO_TRAIN:
            print("Buffer below minimum size, skipping training step this iteration.")
            continue

        avg_loss, avg_v, avg_p = train_on_batches(net, optimizer, buffer, device)
        scheduler.step()
        print(f"Loss: {avg_loss:.4f} (value {avg_v:.4f}, policy {avg_p:.4f})  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"time={time.time() - t0:.1f}s")

        ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, f"model_iter_{iteration}.pt")
        save_checkpoint(net, optimizer, iteration, ckpt_path)
        latest_path = os.path.join(cfg.CHECKPOINT_DIR, "model_latest.pt")
        save_checkpoint(net, optimizer, iteration, latest_path)
        print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
