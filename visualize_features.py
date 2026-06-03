"""
visualize_features.py  --  CNN feature map visualisation (3 separate images).

Usage:
    python visualize_features.py                 # threshold (default)
    python visualize_features.py --mode grayscale
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import config as cfg
from model import DuelingCNN
from preprocess import build_initial_state

BG    = '#0f1117'
PANEL = '#1c1f2e'
WHITE = '#e8eaf6'
GREY  = '#6c7a9c'

LAYER_INFO = [
    {'name': 'Conv Layer 1', 'filters': 32, 'kernel': '8x8', 'stride': 4,
     'cmap': 'viridis', 'color': '#4f8ef7'},
    {'name': 'Conv Layer 2', 'filters': 64, 'kernel': '4x4', 'stride': 2,
     'cmap': 'plasma',  'color': '#f7914f'},
    {'name': 'Conv Layer 3', 'filters': 64, 'kernel': '3x3', 'stride': 1,
     'cmap': 'inferno', 'color': '#c44ff7'},
]
N_SHOW = 16


def latest_checkpoint(ckpt_dir):
    pts = sorted(f for f in os.listdir(ckpt_dir) if f.endswith('.pt'))
    return os.path.join(ckpt_dir, pts[-1]) if pts else None


def get_game_frame(mode, n_steps=80):
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'game'))
    from game.wrapped_flappy_bird import GameState
    game = GameState(render=False)
    frame, _, _ = game.frame_step(np.array([1, 0]))
    for i in range(n_steps):
        act = np.array([0, 1]) if i % 12 == 0 else np.array([1, 0])
        frame, _, _ = game.frame_step(act)
    return frame


def extract_feature_maps(model, state_t):
    maps, hooks = [], []
    def make_hook(store):
        def fn(_, __, out): store.append(out.squeeze(0).cpu().numpy())
        return fn
    for layer in model.conv:
        if isinstance(layer, torch.nn.Conv2d):
            hooks.append(layer.register_forward_hook(make_hook(maps)))
    with torch.no_grad():
        model(state_t)
    for h in hooks:
        h.remove()
    return maps


def norm(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def draw_layer(state, fmaps, info, mode, out_path):
    color = info['color']
    fig = plt.figure(figsize=(20, 4.2), facecolor=BG)
    gs = gridspec.GridSpec(
        1, N_SHOW + 3, figure=fig, wspace=0.05,
        width_ratios=[2.8, 2.5, 0.4] + [1] * N_SHOW
    )

    # label panel
    ax_lbl = fig.add_subplot(gs[0, 0])
    ax_lbl.set_facecolor(PANEL)
    ax_lbl.axis('off')
    ax_lbl.axhline(y=0.93, xmin=0.06, xmax=0.94, color=color, linewidth=3)
    ax_lbl.text(0.5, 0.70, info['name'],
                color=color, fontsize=13, fontweight='bold',
                ha='center', va='center', transform=ax_lbl.transAxes)
    details = (f"{info['filters']} filters  |  kernel {info['kernel']}  |  stride {info['stride']}\n"
               f"output: {fmaps.shape[1]}x{fmaps.shape[2]} px per filter")
    ax_lbl.text(0.5, 0.35, details,
                color=GREY, fontsize=9, ha='center', va='center',
                transform=ax_lbl.transAxes, linespacing=1.9)

    # input frame
    ax_inp = fig.add_subplot(gs[0, 1])
    ax_inp.set_facecolor(PANEL)
    inp_cmap = 'gray' if mode in ('grayscale', 'threshold') else None
    ax_inp.imshow(norm(state[0]), cmap=inp_cmap, interpolation='nearest', aspect='auto')
    ax_inp.set_title('Input frame  (channel 1 / 4)', color=GREY, fontsize=8, pad=5)
    ax_inp.axis('off')

    # arrow
    ax_arr = fig.add_subplot(gs[0, 2])
    ax_arr.set_facecolor(BG)
    ax_arr.axis('off')
    ax_arr.annotate('', xy=(0.82, 0.5), xytext=(0.18, 0.5),
                    xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.2))

    # filter cells
    for i in range(N_SHOW):
        ax = fig.add_subplot(gs[0, i + 3])
        ax.set_facecolor(PANEL)
        ax.imshow(norm(fmaps[i]), cmap=info['cmap'],
                  interpolation='nearest', aspect='auto')
        ax.axis('off')
        ax.text(0.5, -0.09, str(i + 1),
                color=GREY, fontsize=7, ha='center', va='top',
                transform=ax.transAxes)

    # colorbar
    sm = plt.cm.ScalarMappable(cmap=info['cmap'])
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes[-1], fraction=0.9, pad=0.04, aspect=22)
    cbar.ax.tick_params(labelcolor=GREY, labelsize=7, length=2)
    cbar.outline.set_edgecolor(GREY)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(['low', 'mid', 'high'])

    fig.text(0.5, -0.03,
             f'Brighter = stronger filter activation  |  '
             f'Filter indices 1-{N_SHOW} shown below each cell  |  '
             f'Mode: {mode}  |  Checkpoint: checkpoints/{mode}/ckpt_final.pt',
             color=GREY, fontsize=7.5, ha='center', va='top')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='threshold',
                        choices=['grayscale', 'threshold', 'rgb'])
    args = parser.parse_args()

    cfg.PREPROCESS_MODE = args.mode
    cfg.IN_CHANNELS = cfg.N_FRAMES * (3 if args.mode == 'rgb' else 1)

    ckpt_dir = os.path.join('checkpoints', args.mode)
    ckpt = latest_checkpoint(ckpt_dir)
    if ckpt is None:
        raise FileNotFoundError(f'No checkpoint in {ckpt_dir}')

    print(f'Loading {ckpt}')
    model = DuelingCNN(cfg.IN_CHANNELS, cfg.ACTIONS).to(cfg.DEVICE)
    data = torch.load(ckpt, map_location=cfg.DEVICE)
    model.load_state_dict(data['policy_net'])
    model.eval()

    print(f'Capturing game frame (80 steps)...')
    raw = get_game_frame(args.mode, 80)
    state = build_initial_state(raw, args.mode)
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(cfg.DEVICE)

    all_maps = extract_feature_maps(model, state_t)

    for i, info in enumerate(LAYER_INFO):
        out = f'docs/images/feature_maps_conv{i+1}.png'
        draw_layer(state, all_maps[i], info, args.mode, out)


if __name__ == '__main__':
    main()
