"""
Evaluation script — always loads the latest (or specified) checkpoint.

Usage:
    python test.py                          # evaluate latest checkpoint, headless
    python test.py --render                 # render game on screen
    python test.py --checkpoint path.pt     # specific checkpoint
    python test.py --episodes 20 --mode threshold
"""

from __future__ import annotations

import os
import sys
import argparse
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(_ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'game'))
import config as cfg
from agent import DoubleDQNAgent
from preprocess import build_initial_state, update_state, reset_state_after_terminal


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--episodes',   type=int, default=10)
    p.add_argument('--render',     action='store_true',
                   help='Show game window (requires a display)')
    p.add_argument('--mode',       default=cfg.PREPROCESS_MODE,
                   choices=['grayscale', 'threshold', 'rgb'])
    return p.parse_args()


def main():
    args = parse_args()

    if not args.render:
        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
        os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

    cfg.PREPROCESS_MODE = args.mode
    cfg.IN_CHANNELS     = (3 * cfg.N_FRAMES) if args.mode == 'rgb' else cfg.N_FRAMES
    cfg.CHECKPOINT_DIR  = os.path.join('checkpoints', args.mode)

    from wrapped_flappy_bird import GameState
    game  = GameState(render=args.render)
    agent = DoubleDQNAgent()

    ckpt_path = args.checkpoint or DoubleDQNAgent.latest_checkpoint(cfg.CHECKPOINT_DIR)
    if not ckpt_path:
        print("No checkpoint found. Run train.py first.")
        sys.exit(1)
    agent.load(ckpt_path)

    # Pure greedy policy during evaluation
    agent.epsilon = 0.0

    scores, rewards = [], []

    for ep in range(1, args.episodes + 1):
        do_nothing = np.array([1, 0])
        frame, _, _ = game.frame_step(do_nothing)
        state       = build_initial_state(frame, cfg.PREPROCESS_MODE)
        ep_reward   = 0.0

        while True:
            action_idx = agent.choose_action(state)
            action     = np.array([1, 0]) if action_idx == 0 else np.array([0, 1])
            frame, reward, terminal = game.frame_step(action)
            next_state  = update_state(state, frame, cfg.PREPROCESS_MODE)
            ep_reward  += reward
            state       = next_state

            if terminal:
                score = game.last_score
                scores.append(score)
                rewards.append(ep_reward)
                print(f"Episode {ep:3d}  |  Score: {score:4d}  |  Reward: {ep_reward:.2f}")
                break

    print(f"\n{'─'*45}")
    print(f"Episodes   : {args.episodes}")
    print(f"Mean score : {np.mean(scores):.2f}")
    print(f"Max score  : {int(np.max(scores))}")
    print(f"Min score  : {int(np.min(scores))}")
    print(f"Mean reward: {np.mean(rewards):.2f}")


if __name__ == '__main__':
    main()
