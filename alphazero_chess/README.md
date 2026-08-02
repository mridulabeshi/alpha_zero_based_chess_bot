# AlphaZero-Style Chess Bot

A from-scratch, self-play reinforcement learning chess engine following the
AlphaZero recipe: a residual CNN produces a move-policy and a position
value, Monte Carlo Tree Search (guided by that network) picks moves and
in turn produces improved training targets, and the network is trained
on its own self-play games in a repeating loop.

## Files

| File | Purpose |
|---|---|
| `config.py` | All hyperparameters in one place |
| `chess_env.py` | Board -> 18x8x8 tensor encoding, move <-> 4672-way action encoding, legal move masking |
| `model.py` | Residual tower + policy/value heads (PyTorch) |
| `mcts.py` | PUCT Monte Carlo Tree Search using the network for priors and leaf evaluation |
| `self_play.py` | Plays full games with MCTS, producing `(state, pi, z)` training examples |
| `replay_buffer.py` | Fixed-size buffer of recent self-play examples |
| `train.py` | Main loop: self-play -> buffer -> gradient updates -> checkpoint, repeat |
| `play.py` | Play interactively against a trained checkpoint from the console |

## Quick start

```bash
pip install -r requirements.txt

# Train (Ctrl+C any time; checkpoints are saved every iteration)
python train.py

# Play against your latest checkpoint
python play.py --checkpoint checkpoints/model_latest.pt --human-color white
```

Sensible small-scale defaults are already set in `config.py` (10 residual
blocks, 128 channels, 200 MCTS simulations/move, 25 self-play games per
iteration) so you can see the loop running on a laptop CPU within
minutes. Real strength requires much more compute — see "Scaling up"
below.

## How it fits together

1. **`chess_env.py`** turns a `python-chess` board into the tensors the
   network needs, and turns network output indices back into legal moves.
   Positions are always encoded from the perspective of the side to
   move (using `board.mirror()` for Black), so the network only has to
   learn one perspective. The 4672-way move encoding follows the scheme
   from the AlphaZero paper: for each of the 64 origin squares, 73
   possible "move planes" (56 queen-like directions/distances, 8 knight
   jumps, 9 underpromotions).

2. **`model.py`** is a standard ResNet trunk (`NUM_RES_BLOCKS` residual
   blocks over `NUM_CHANNELS` channels) feeding two heads: a policy head
   producing a distribution over the 4672 actions, and a value head
   producing a scalar in `[-1, 1]` estimating the current side's winning
   chances.

3. **`mcts.py`** runs PUCT search: starting from the root, repeatedly
   select the child maximizing `Q + c_puct * P * sqrt(N_parent)/(1+N_child)`,
   expand leaves using the network's policy (masked to legal moves) as
   priors, and back the network's value estimate up the tree. Dirichlet
   noise is mixed into the root priors during self-play for exploration.

4. **`self_play.py`** plays a full game move by move, running MCTS at
   each ply. The move actually played is sampled from the visit-count
   distribution (temperature 1.0 for the first `TEMPERATURE_MOVES`
   plies, then greedy). The visit-count distribution itself becomes the
   `pi` training target, and the final game outcome (win/loss/draw)
   becomes the `z` target for every position in that game, sign-flipped
   per side to move.

5. **`train.py`** alternates: generate self-play games with the current
   network -> add examples to the replay buffer -> sample minibatches and
   minimize `(z - v)^2 - pi . log(p)` -> checkpoint -> repeat.

## Simplifications vs. the original AlphaZero paper

This is a complete, working implementation, but deliberately compact so
it's readable and runnable on modest hardware. Compared to the paper:

- **Single position, not 8-step history.** DeepMind's input includes the
  last 8 board states (repetition detection, etc.); this implementation
  encodes only the current position plus castling/en-passant/halfmove
  clock. It trains and plays fine, but doesn't see repetition patterns
  directly.
- **No arena / evaluator network.** The paper only replaces the
  "best" network with a freshly trained one after it wins a gauntlet of
  evaluation games. Here every training step's network is used
  immediately for the next round of self-play, which is simpler and
  works reasonably well in practice (this is the approach later
  AlphaZero-inspired projects like Leela Chess Zero simplified toward).
- **Single process, sequential self-play.** No distributed self-play
  workers / parameter server — self-play games run one at a time in the
  main loop. Easy to parallelize with `multiprocessing` if you want more
  throughput.
- **No resignation.** `RESIGN_THRESHOLD` is wired into `config.py` but
  unused; games always play to completion or `MAX_GAME_LENGTH`.

## Scaling up

To move from "runs on a laptop" toward "actually plays strong chess" (the
original paper used thousands of TPUs for days), the main levers, all in
`config.py`, are:

- `NUM_RES_BLOCKS` / `NUM_CHANNELS`: 10/128 -> 19-40 / 256
- `NUM_SIMULATIONS`: 200 -> 800
- `GAMES_PER_ITERATION`: 25 -> hundreds/thousands, ideally parallelized
  across processes/machines
- Add the 8-step history to `chess_env.py`'s encoding
- Add an arena-style evaluation gate in `train.py` before promoting a new
  "best" network
