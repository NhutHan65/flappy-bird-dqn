"""
visualize_features.py  --  CNN feature map visualisation (3 separate images).

Each image: input frame + 16 filter activations. No text inside the image.

Usage:
    python visualize_features.py
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
GREY  = '#6c7a9c'

LAYER_INFO = [
    {'cmap': 'viridis', 'color': '#4f8ef7'},
    {'cmap': 'plasma',  'color': '#f7914f'},
    {'cmap': 'inferno', 'color': '#c44ff7'},
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

    # cols: 1 input frame + 1 narrow arrow + N_SHOW maps + 1 cbar
    col_ratios = [2] + [0.25] + [1] * N_SHOW + [0.15]
    fig, axes = plt.subplots(
        1, len(col_ratios),
        figsize=(22, 3.5),
        facecolor=BG,
        gridspec_kw={'width_ratios': col_ratios, 'wspace': 0.04}
    )

    # input frame
    ax_inp = axes[0]
    ax_inp.set_facecolor(PANEL)
    inp_cmap = 'gray' if mode in ('grayscale', 'threshold') else None
    ax_inp.imshow(norm(state[0]), cmap=inp_cmap,
                  interpolation='nearest', aspect='equal')
    for spine in ax_inp.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(1.5)
    ax_inp.set_xticks([])
    ax_inp.set_yticks([])

    # arrow
    ax_arr = axes[1]
    ax_arr.set_facecolor(BG)
    ax_arr.axis('off')
    ax_arr.annotate('', xy=(0.9, 0.5), xytext=(0.1, 0.5),
                    xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.0))

    # feature map cells
    for i in range(N_SHOW):
        ax = axes[i + 2]
        ax.set_facecolor(PANEL)
        ax.imshow(norm(fmaps[i]), cmap=info['cmap'],
                  interpolation='nearest', aspect='equal')
        ax.axis('off')

    # colorbar using the last axes slot
    ax_cb = axes[-1]
    ax_cb.set_facecolor(BG)
    ax_cb.axis('off')
    sm = plt.cm.ScalarMappable(cmap=info['cmap'])
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_cb, fraction=1.0, pad=0.0, aspect=14)
    cbar.ax.tick_params(labelcolor=GREY, labelsize=7, length=2)
    cbar.outline.set_edgecolor(GREY)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['low', 'high'])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches='tight',
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

    ckpt = latest_checkpoint(os.path.join('checkpoints', args.mode))
    if ckpt is None:
        raise FileNotFoundError(f'No checkpoint in checkpoints/{args.mode}')

    print(f'Loading {ckpt}')
    model = DuelingCNN(cfg.IN_CHANNELS, cfg.ACTIONS).to(cfg.DEVICE)
    data = torch.load(ckpt, map_location=cfg.DEVICE)
    model.load_state_dict(data['policy_net'])
    model.eval()

    print('Capturing game frame...')
    raw = get_game_frame(args.mode)
    state = build_initial_state(raw, args.mode)
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(cfg.DEVICE)

    all_maps = extract_feature_maps(model, state_t)
    for i, info in enumerate(LAYER_INFO):
        draw_layer(state, all_maps[i], info, args.mode,
                   f'docs/images/feature_maps_conv{i+1}.png')


if __name__ == '__main__':
    main()
