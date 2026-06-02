"""
Capture gameplay as GIF for a given preprocessing mode.

Usage:
    python capture_gif.py --mode threshold --output images/threshold_demo.gif
    python capture_gif.py --mode grayscale --output images/grayscale_demo.gif
    python capture_gif.py --mode rgb       --output images/rgb_demo.gif
"""

from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import imageio

_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(_ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'game'))

import config as cfg
from agent import DoubleDQNAgent
from preprocess import build_initial_state, update_state, reset_state_after_terminal


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode',     default='grayscale',
                   choices=['grayscale', 'threshold', 'rgb'])
    p.add_argument('--output',   default=None)
    p.add_argument('--episodes', type=int, default=1)
    p.add_argument('--fps',      type=int, default=30)
    p.add_argument('--max-frames', type=int, default=1800,
                   help='Max frames per episode (default 1800 = 60s at 30fps)')
    return p.parse_args()


def main():
    args = parse_args()

    cfg.PREPROCESS_MODE = args.mode
    cfg.IN_CHANNELS     = (3 * cfg.N_FRAMES) if args.mode == 'rgb' else cfg.N_FRAMES
    cfg.CHECKPOINT_DIR  = os.path.join('checkpoints', args.mode)

    output = args.output or f'images/{args.mode}_demo.gif'

    from wrapped_flappy_bird import GameState
    game  = GameState(render=True, visual_mode=args.mode)
    agent = DoubleDQNAgent()

    ckpt = DoubleDQNAgent.latest_checkpoint(cfg.CHECKPOINT_DIR)
    if not ckpt:
        print(f"No checkpoint found in {cfg.CHECKPOINT_DIR}")
        sys.exit(1)
    agent.load(ckpt)
    agent.epsilon = 0.0

    frames = []
    print(f"Capturing {args.episodes} episode(s) in {args.mode} mode...")

    for ep in range(1, args.episodes + 1):
        do_nothing = np.array([1, 0])
        frame, _, _ = game.frame_step(do_nothing)
        state = build_initial_state(frame, cfg.PREPROCESS_MODE)
        frame_count = 0

        while frame_count < args.max_frames:
            # Capture what's on screen (already filtered by visual_mode)
            import pygame
            surface = pygame.display.get_surface()
            raw = pygame.surfarray.array3d(surface)
            # pygame uses (W, H, C) — transpose to (H, W, C) for imageio
            frames.append(raw.transpose(1, 0, 2))

            action_idx = agent.choose_action(state)
            action = np.array([1, 0]) if action_idx == 0 else np.array([0, 1])
            frame, reward, terminal = game.frame_step(action)
            state = update_state(state, frame, cfg.PREPROCESS_MODE)
            frame_count += 1

            if terminal:
                print(f"  Episode {ep} — score: {game.last_score}, frames: {frame_count}")
                break

    print(f"Saving {len(frames)} frames to {output} ...")
    imageio.mimsave(output, frames, fps=args.fps)
    print(f"Done — {output}")


if __name__ == '__main__':
    main()
