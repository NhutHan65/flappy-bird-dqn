"""
visualize_features.py

Extract and visualize CNN feature maps from a trained checkpoint.
Runs the game for a few steps to capture a real mid-game frame,
then passes it through each conv layer and saves a grid image.

Usage:
    python visualize_features.py                        # threshold checkpoint
    python visualize_features.py --mode grayscale
    python visualize_features.py --out docs/images/feature_maps.png
"""
from __future__ import annotations

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import config as cfg
from model import DuelingCNN
from preprocess import preprocess_frame, build_initial_state, update_state
from agent import DoubleDQNAgent


# ── helpers ──────────────────────────────────────────────────────────────────

def latest_checkpoint(ckpt_dir: str) -> str | None:
    pts = sorted(f for f in os.listdir(ckpt_dir) if f.endswith('.pt'))
    return os.path.join(ckpt_dir, pts[-1]) if pts else None


def get_game_frame(mode: str, n_steps: int = 80):
    """Run the game headlessly for n_steps; return a mid-game raw frame."""
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'game'))
    from game.wrapped_flappy_bird import GameState
    game = GameState(render=False)
    action = np.array([1, 0])  # no-flap
    frame, _, _ = game.frame_step(action)
    for i in range(n_steps):
        flap = np.array([0, 1]) if i % 12 == 0 else np.array([1, 0])
        frame, _, _ = game.frame_step(flap)
    return frame


def extract_feature_maps(model: DuelingCNN, state_tensor: torch.Tensor):
    """Forward through each conv layer; return list of (C, H, W) numpy arrays."""
    maps = []
    hooks = []

    def make_hook(storage):
        def hook(_, __, output):
            storage.append(output.squeeze(0).cpu().numpy())
        return hook

    conv_layers = [layer for layer in model.conv if isinstance(layer, torch.nn.Conv2d)]
    for layer in conv_layers:
        h = layer.register_forward_hook(make_hook(maps))
        hooks.append(h)

    with torch.no_grad():
        model(state_tensor)

    for h in hooks:
        h.remove()

    return maps  # list of 3 arrays: [(32,H,W), (64,H,W), (64,H,W)]


def show_maps(ax_row, fmaps: np.ndarray, n_show: int, label: str, cmap: str):
    """Fill a row of axes with feature map thumbnails."""
    for i, ax in enumerate(ax_row):
        if i < n_show:
            fm = fmaps[i]
            # normalise each map independently to [0,1]
            lo, hi = fm.min(), fm.max()
            fm_norm = (fm - lo) / (hi - lo + 1e-8)
            ax.imshow(fm_norm, cmap=cmap, interpolation='nearest')
        ax.axis('off')
    ax_row[0].set_ylabel(label, fontsize=9, labelpad=6, rotation=0,
                         ha='right', va='center', fontweight='bold')


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='threshold',
                        choices=['grayscale', 'threshold', 'rgb'])
    parser.add_argument('--out', default='docs/images/feature_maps.png')
    parser.add_argument('--steps', type=int, default=80,
                        help='game steps before capturing frame')
    args = parser.parse_args()

    # ── load model ──
    cfg.PREPROCESS_MODE = args.mode
    cfg.IN_CHANNELS = cfg.N_FRAMES * (3 if args.mode == 'rgb' else 1)

    ckpt_dir = os.path.join('checkpoints', args.mode)
    ckpt = latest_checkpoint(ckpt_dir)
    if ckpt is None:
        raise FileNotFoundError(f'No checkpoint found in {ckpt_dir}')

    print(f'Loading {ckpt}')
    model = DuelingCNN(cfg.IN_CHANNELS, cfg.ACTIONS).to(cfg.DEVICE)
    data = torch.load(ckpt, map_location=cfg.DEVICE)
    model.load_state_dict(data['policy_net'])
    model.eval()

    # ── get a real game frame ──
    print(f'Running game for {args.steps} steps to capture frame...')
    raw_frame = get_game_frame(args.mode, args.steps)

    # build state
    state = build_initial_state(raw_frame, args.mode)
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(cfg.DEVICE)

    # ── extract feature maps ──
    all_maps = extract_feature_maps(model, state_t)
    layer_labels = ['Conv 1\n32 filters\n8×8 / stride 4',
                    'Conv 2\n64 filters\n4×4 / stride 2',
                    'Conv 3\n64 filters\n3×3 / stride 1']
    n_show = [16, 16, 16]   # how many filters to show per layer
    cmaps  = ['viridis', 'plasma', 'inferno']

    # ── build figure ──
    # Layout: left column = input frame, right area = feature map rows
    fig = plt.figure(figsize=(18, 8), facecolor='#1a1a2e')

    outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 4], wspace=0.04)

    # --- input frame panel ---
    ax_input = fig.add_subplot(outer[0])
    # show first channel of the state stack
    ch0 = state[0] if state.shape[0] >= 1 else state
    lo, hi = ch0.min(), ch0.max()
    ch0_norm = (ch0 - lo) / (hi - lo + 1e-8)
    disp_cmap = 'gray' if args.mode in ('grayscale', 'threshold') else None
    ax_input.imshow(ch0_norm, cmap=disp_cmap, interpolation='nearest')
    ax_input.set_title('Input frame\n(channel 1 of 4)', color='white',
                       fontsize=10, pad=6)
    ax_input.axis('off')
    ax_input.set_facecolor('#1a1a2e')

    # --- feature map rows ---
    inner = gridspec.GridSpecFromSubplotSpec(
        3, max(n_show), subplot_spec=outer[1],
        hspace=0.08, wspace=0.04
    )

    for row_i, (fmaps, label, ns, cmap) in enumerate(
            zip(all_maps, layer_labels, n_show, cmaps)):
        axes_row = [fig.add_subplot(inner[row_i, col]) for col in range(ns)]
        show_maps(axes_row, fmaps, ns, label, cmap)
        for ax in axes_row:
            ax.set_facecolor('#1a1a2e')
        # colour bar strip on the right of each row
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])

    fig.suptitle(
        f'CNN Feature Maps — {args.mode.capitalize()} mode\n'
        'Each cell shows one filter\'s activation on the input frame',
        color='white', fontsize=12, y=1.01
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'Saved → {args.out}')


if __name__ == '__main__':
    main()
