import torch
import os

# ─── Game ─────────────────────────────────────────────────────────────────────
ACTIONS = 2

# ─── Preprocessing  (ablation: 'grayscale' | 'threshold' | 'rgb') ─────────────
PREPROCESS_MODE = 'grayscale'
FRAME_SIZE      = 84
N_FRAMES        = 4
AUGMENT         = True

# Derived from the two above — will be overridden by CLI args in train.py
IN_CHANNELS = N_FRAMES  # 4 for grayscale/threshold, 3*N_FRAMES=12 for rgb

# ─── Replay buffer ────────────────────────────────────────────────────────────
MEM_SIZE          = 50000
MEM_RETAIN        = 0.1        # protect first 10 % from overwriting (catastrophic forgetting)
REPLAY_START_SIZE = 10000

# Prioritized Experience Replay
PER_ALPHA       = 0.6
PER_BETA_START  = 0.4
PER_BETA_END    = 1.0
PER_BETA_FRAMES = 100000
PER_EPS         = 1e-6

# ─── Training ─────────────────────────────────────────────────────────────────
EPISODES   = 2000
BATCH_SIZE = 32
LR         = 1e-4
GAMMA      = 0.99
GRAD_CLIP  = 10.0

# Epsilon greedy
EPS_START = 0.1
EPS_END   = 0.0001
EPS_DECAY = 4 * MEM_SIZE     # linear decay over this many learning steps

# Soft target-network update coefficient
TARGET_TAU = 0.005

# ─── Reward shaping ───────────────────────────────────────────────────────────
REWARD_STEP  =  0.1
REWARD_PIPE  =  0.0   # removed per requirements (not needed)
REWARD_CRASH = -1.0

# ─── Checkpointing ────────────────────────────────────────────────────────────
CHECKPOINT_DIR = 'checkpoints'
SAVE_EVERY     = 50   # episodes

# ─── WandB ────────────────────────────────────────────────────────────────────
WANDB_PROJECT = 'flappy-bird-dqn'

# ─── Device ───────────────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
