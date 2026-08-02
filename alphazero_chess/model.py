"""
AlphaZero-style neural network: a residual "tower" trunk feeding a
policy head (move probabilities) and a value head (win probability).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


def get_device():
    if cfg.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class AlphaZeroNet(nn.Module):
    def __init__(
        self,
        in_planes=cfg.NUM_PLANES,
        channels=cfg.NUM_CHANNELS,
        num_blocks=cfg.NUM_RES_BLOCKS,
        action_size=cfg.ACTION_SIZE,
        value_hidden=cfg.VALUE_HEAD_HIDDEN,
    ):
        super().__init__()
        self.stem_conv = nn.Conv2d(in_planes, channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)

        self.res_blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(num_blocks)])

        # policy head
        self.policy_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, action_size)

        # value head
        self.value_conv = nn.Conv2d(channels, 3, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(3)
        self.value_fc1 = nn.Linear(3 * 8 * 8, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)

    def forward(self, x):
        x = F.relu(self.stem_bn(self.stem_conv(x)))
        for block in self.res_blocks:
            x = block(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.flatten(1)
        policy_logits = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value.squeeze(-1)

    @torch.no_grad()
    def predict(self, tensor_batch: torch.Tensor):
        """tensor_batch: (N, planes, 8, 8) -> (policy_probs (N,A), value (N,))
        Softmax is applied here; masking of illegal moves happens in mcts.py."""
        self.eval()
        logits, value = self.forward(tensor_batch)
        probs = F.softmax(logits, dim=1)
        return probs, value


def build_model(device=None):
    device = device or get_device()
    model = AlphaZeroNet().to(device)
    return model
