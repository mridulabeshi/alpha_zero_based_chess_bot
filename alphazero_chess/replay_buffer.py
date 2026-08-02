"""Simple fixed-capacity replay buffer of (state, pi, z) tuples."""

import random
from collections import deque

import numpy as np

import config as cfg


class ReplayBuffer:
    def __init__(self, capacity=cfg.REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def add(self, examples):
        self.buffer.extend(examples)

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, pis, zs = zip(*batch)
        return (
            np.stack(states).astype(np.float32),
            np.stack(pis).astype(np.float32),
            np.array(zs, dtype=np.float32),
        )
