"""
Training script — Double DQN + Dueling + PER on raw pixel frames.

Usage:
    python train.py                         # default (grayscale, WandB on)
    python train.py --mode threshold        # ablation: binary threshold input
    python train.py --mode rgb              # ablation: full colour input
    python train.py --no-wandb              # disable WandB logging
    python train.py --run-name my_run       # custom WandB run name
"""

from __future__ import annotations

import os
import sys as _sys

# Headless pygame: 'dummy' works on Linux; on Windows use 'offscreen' if
# available, otherwise fall back to letting a minimised window appear.
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')   # headless on both Linux and Windows
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import sys
import argparse
import numpy as np

# Always work relative to this file so the script runs from any CWD
_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(_ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'game'))
import config as cfg
from agent import DoubleDQNAgent
from preprocess import build_initial_state, update_state, reset_state_after_terminal

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', default=cfg.PREPROCESS_MODE,
                   choices=['grayscale', 'threshold', 'rgb'],
                   help='Preprocessing mode for ablation study')
    p.add_argument('--no-wandb',     action='store_true')
    p.add_argument('--run-name',     default=None)
    p.add_argument('--episodes',     type=int, default=cfg.EPISODES)
    p.add_argument('--checkpoint-dir', default=None,
                   help='Override checkpoint directory (default: checkpoints/<mode>)')
    return p.parse_args()


def main():
    args = parse_args()

    # Override config from CLI (enables ablation study without touching config.py)
    cfg.PREPROCESS_MODE  = args.mode
    cfg.IN_CHANNELS      = (3 * cfg.N_FRAMES) if args.mode == 'rgb' else cfg.N_FRAMES
    cfg.EPISODES         = args.episodes
    cfg.CHECKPOINT_DIR   = args.checkpoint_dir or os.path.join('checkpoints', args.mode)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    use_wandb = _WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        run_name = args.run_name or f'double-dqn-{args.mode}'
        wandb.init(
            project=cfg.WANDB_PROJECT,
            name=run_name,
            config={k: v for k, v in vars(cfg).items()
                    if not k.startswith('_') and not callable(v)
                    and not isinstance(v, type(cfg))},
        )

    print(f"Device : {cfg.DEVICE}")
    print(f"Mode   : {cfg.PREPROCESS_MODE}  (in_channels={cfg.IN_CHANNELS})")
    print(f"WandB  : {use_wandb}")

    # Import game AFTER SDL env vars are set
    from wrapped_flappy_bird import GameState
    game  = GameState(render=False)
    agent = DoubleDQNAgent()

    # Always resume from the latest checkpoint if one exists
    ckpt = DoubleDQNAgent.latest_checkpoint(cfg.CHECKPOINT_DIR)
    if ckpt:
        agent.load(ckpt)
    else:
        print("No checkpoint found — training from scratch.")

    # Seed the first frame
    do_nothing = np.array([1, 0])
    frame, _, _ = game.frame_step(do_nothing)
    state = build_initial_state(frame, cfg.PREPROCESS_MODE)

    episode        = 0
    episode_reward = 0.0
    global_step    = agent.steps   # pick up from checkpoint

    print(f"\nFilling replay buffer (need {cfg.REPLAY_START_SIZE} samples)...")

    while episode < cfg.EPISODES:
        action_idx = agent.choose_action(state)
        action     = np.array([1, 0]) if action_idx == 0 else np.array([0, 1])

        frame, reward, terminal = game.frame_step(
            action,
            step_reward=cfg.REWARD_STEP,
            pipe_reward=cfg.REWARD_PIPE,
            crash_reward=cfg.REWARD_CRASH,
        )

        next_state = update_state(state, frame, cfg.PREPROCESS_MODE)

        agent.memory.add(state, action_idx, reward, next_state, float(terminal))
        episode_reward += reward
        global_step    += 1

        # Learn once the buffer has enough samples
        loss = q_max = mean_td = None
        if agent.memory.ready:
            loss, mean_td, q_max = agent.learn()

        if terminal:
            episode += 1
            score    = game.last_score

            if episode % 10 == 0 or episode <= 5:
                print(
                    f"Ep {episode:5d}/{cfg.EPISODES} | "
                    f"Score {score:4d} | "
                    f"Reward {episode_reward:8.2f} | "
                    f"eps {agent.epsilon:.5f} | "
                    f"Step {global_step:7d}"
                    + (f" | Loss {loss:.4f}" if loss is not None else "  [filling]")
                )

            if use_wandb:
                log = {
                    'episode':        episode,
                    'score':          score,
                    'episode_reward': episode_reward,
                    'epsilon':        agent.epsilon,
                    'beta':           agent.beta,
                    'global_step':    global_step,
                    'buffer_size':    len(agent.memory),
                }
                if loss is not None:
                    log.update({'loss': loss, 'q_max': q_max, 'mean_td_error': mean_td})
                wandb.log(log, step=episode)

            if episode % cfg.SAVE_EVERY == 0:
                path = os.path.join(cfg.CHECKPOINT_DIR, f'ckpt_ep{episode:06d}.pt')
                agent.save(path)

            episode_reward = 0.0
            # Reset state cleanly for new episode
            state = reset_state_after_terminal(next_state, cfg.PREPROCESS_MODE)
        else:
            state = next_state

    # Final checkpoint
    final_path = os.path.join(cfg.CHECKPOINT_DIR, f'ckpt_final.pt')
    agent.save(final_path)

    if use_wandb:
        wandb.finish()

    print("Training complete.")


if __name__ == '__main__':
    main()
