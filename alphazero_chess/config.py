"""
Central configuration for the AlphaZero-style chess bot.
Tune these to trade off training speed vs. strength.
"""

# ---- Board / action encoding ----
NUM_PLANES = 18          # input feature planes (see chess_env.py)
BOARD_SIZE = 8
ACTION_SIZE = 8 * 8 * 73  # 4672, AlphaZero move encoding

# ---- Neural network ----
NUM_RES_BLOCKS = 10       # use 19-40 for a "real" strength run, 10 for laptops
NUM_CHANNELS = 128        # use 256 for a "real" strength run
VALUE_HEAD_HIDDEN = 256

# ---- MCTS ----
NUM_SIMULATIONS = 200     # simulations per move (AlphaZero paper uses 800)
C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25
TEMPERATURE_MOVES = 15    # sample stochastically for first N plies, then greedy

# ---- Self-play ----
GAMES_PER_ITERATION = 25
MAX_GAME_LENGTH = 300     # plies, avoid infinite games
RESIGN_THRESHOLD = None   # e.g. -0.95 to enable resignation, None disables

# ---- Replay buffer ----
REPLAY_BUFFER_SIZE = 100_000
MIN_BUFFER_SIZE_TO_TRAIN = 500

# ---- Training ----
NUM_ITERATIONS = 1000
BATCH_SIZE = 256
EPOCHS_PER_ITERATION = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LR_MILESTONES = [200, 500]   # iterations at which LR is decayed
LR_DECAY = 0.1
GRAD_CLIP_NORM = 5.0

# ---- Misc ----
CHECKPOINT_DIR = "checkpoints"
DEVICE = "cuda"  # falls back to cpu automatically in model.py if unavailable
SEED = 42
