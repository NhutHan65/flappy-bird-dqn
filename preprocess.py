"""
Frame preprocessing for the Flappy Bird DQN.

Supports three modes for ablation study:
  'grayscale'  — resize → grayscale                         → (84, 84)
  'threshold'  — resize → grayscale → binary threshold      → (84, 84)  [original paper]
  'rgb'        — resize → keep RGB, channel-first            → (3, 84, 84)

Data augmentation (applied to training batches in agent.py, not during collection):
  - Random brightness jitter
  - Random Gaussian noise (grayscale/threshold only)
"""

from __future__ import annotations

import cv2
import numpy as np
import config as cfg


def preprocess_frame(frame: np.ndarray, mode: str | None = None) -> np.ndarray:
    """
    Raw pygame frame (W=288, H=512, 3) → processed frame ready for stacking.

    pygame returns (width, height, channels); we transpose to (height, width, channels)
    before passing to cv2.
    """
    mode = mode or cfg.PREPROCESS_MODE

    # pygame surfarray: (W, H, 3) → cv2 convention: (H, W, 3)
    frame = frame.transpose(1, 0, 2)                             # (512, 288, 3)
    frame = cv2.resize(frame, (cfg.FRAME_SIZE, cfg.FRAME_SIZE))  # (84, 84, 3)

    if mode == 'rgb':
        return frame.transpose(2, 0, 1).astype(np.uint8)        # (3, 84, 84)

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)               # (84, 84)

    if mode == 'threshold':
        _, gray = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    return gray.astype(np.uint8)                                 # (84, 84)


def build_initial_state(frame: np.ndarray, mode: str | None = None) -> np.ndarray:
    """Stack N_FRAMES copies of the first frame → (IN_CHANNELS, H, W)."""
    mode = mode or cfg.PREPROCESS_MODE
    processed = preprocess_frame(frame, mode)

    if mode == 'rgb':
        # processed: (3, 84, 84) → stack 4 → (12, 84, 84)
        return np.concatenate([processed] * cfg.N_FRAMES, axis=0)
    else:
        # processed: (84, 84) → stack 4 → (4, 84, 84)
        return np.stack([processed] * cfg.N_FRAMES, axis=0)


def update_state(state: np.ndarray, frame: np.ndarray, mode: str | None = None) -> np.ndarray:
    """Shift frame stack left by one and append the new processed frame."""
    mode = mode or cfg.PREPROCESS_MODE
    processed = preprocess_frame(frame, mode)

    if mode == 'rgb':
        # Drop oldest 3 channels, append new (3, 84, 84)
        return np.concatenate([state[3:], processed], axis=0)
    else:
        # Drop oldest 1 channel, append new (84, 84) as (1, 84, 84)
        return np.concatenate([state[1:], processed[np.newaxis]], axis=0)


def reset_state_after_terminal(next_state: np.ndarray, mode: str | None = None) -> np.ndarray:
    """
    After a terminal step the game has already reset; next_state contains the
    first frame of the new episode mixed with stale frames.  Extract just the
    newest frame and stack it N_FRAMES times for a clean episode start.
    """
    mode = mode or cfg.PREPROCESS_MODE
    cpf = 3 if mode == 'rgb' else 1          # channels per frame
    last_frame = next_state[-cpf:]           # (cpf, 84, 84)
    return np.concatenate([last_frame] * cfg.N_FRAMES, axis=0)
