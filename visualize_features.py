"""
visualize_features.py  —  CNN feature map visualisation for portfolio.

Usage:
    python visualize_features.py                 # threshold (default)
    python visualize_features.py --mode grayscale
    python visualize_features.py --out docs/images/feature_maps.png
"""
from __future__ import annotations

import os, sys, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

import config as cfg
from model import DuelingCNN
from preprocess import build_initial_state


BG      = '#0f1117'
PANEL   = '#1c1f2e'
ACCENT  = '#4f8ef7'
WHITE   = '#e8eaf6'
GREY    = '#6c7a9c'

LAYER_INFO = [
    {'name': 'Conv Layer 1', 'filters': 32, 'kernel': '8×8', 'stride': 4,
     'cmap': 'viridis',  'color': '#4f8ef7'},
    {'name': 'Conv Layer 2', 'filters': 64, 'kernel': '4×4', 'stride': 2,
     'cmap': 'plasma',   'color': '#f7914f'},
    {'name': 'Conv Layer 3', 'filters': 64, 'kernel': '3×3', 'stride': 1,
     'cmap': 'inferno',  'color': '#c44ff7'},
]
N_SHOW = 16   # filters to display per layer


# ── helpers ──────────────────────────────────────────────────────────────────

def latest_checkpoint(ckpt_dir: str) -> str | None:
    pts = sorted(f for f in os.listdir(ckpt_dir) if f.endswith('.pt'))
    return os.path.join(ckpt_dir, pts[-1]) if pts else None


def get_game_frame(mode: str, n_steps: int = 80) -> np.ndarray:
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'game'))
    from game.wrapped_flappy_bird import GameState
    game  = GameState(render=False)
    frame, _, _ = game.frame_step(np.array([1, 0]))
    for i in range(n_steps):
        act = np.array([0, 1]) if i % 12 == 0 else np.array([1, 0])
        frame, _, _ = game.frame_step(act)
    return frame


def extract_feature_maps(model: DuelingCNN,
                         state_t: torch.Tensor) -> list[np.ndarray]:
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


def norm(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


# ── draw ─────────────────────────────────────────────────────────────────────

def draw(state: np.ndarray, all_maps: list[np.ndarray],
         mode: str, out_path: str) -> None:

    # Figure: 3 rows (one per conv layer) + header row
    # Columns: [label | input | arrow | 16 maps | colorbar]
    n_rows   = 3
    fig_w, fig_h = 20, 11
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)

    # Outer grid: top title strip + 3 content rows
    outer = gridspec.GridSpec(
        n_rows + 1, 1, figure=fig,
        height_ratios=[0.12] + [1] * n_rows,
        hspace=0.08
    )

    # ── title row ──
    ax_title = fig.add_subplot(outer[0])
    ax_title.set_facecolor(BG)
    ax_title.axis('off')
    ax_title.text(0.5, 0.5,
                  f'CNN Feature Maps  —  {mode.capitalize()} Mode',
                  color=WHITE, fontsize=16, fontweight='bold',
                  ha='center', va='center', transform=ax_title.transAxes)

    # ── content rows ──
    for row_i, info in enumerate(LAYER_INFO):
        fmaps = all_maps[row_i]         # (C, H, W)
        color = info['color']

        # inner grid: label | input | 16 maps | cbar
        inner = gridspec.GridSpecFromSubplotSpec(
            1, N_SHOW + 3,
            subplot_spec=outer[row_i + 1],
            wspace=0.05,
            width_ratios=[2.2, 2.2, 0.3] + [1] * N_SHOW
        )

        # ── label panel ──
        ax_lbl = fig.add_subplot(inner[0, 0])
        ax_lbl.set_facecolor(PANEL)
        ax_lbl.axis('off')
        # coloured top border
        ax_lbl.axhline(y=0.97, xmin=0.08, xmax=0.92,
                       color=color, linewidth=3)
        ax_lbl.text(0.5, 0.72, info['name'],
                    color=color, fontsize=10, fontweight='bold',
                    ha='center', va='center', transform=ax_lbl.transAxes)
        details = (f"{info['filters']} filters\n"
                   f"kernel {info['kernel']}\n"
                   f"stride {info['stride']}\n"
                   f"output {fmaps.shape[1]}×{fmaps.shape[2]} px")
        ax_lbl.text(0.5, 0.32, details,
                    color=GREY, fontsize=8, ha='center', va='center',
                    transform=ax_lbl.transAxes, linespacing=1.7)

        # ── input frame ──
        ax_inp = fig.add_subplot(inner[0, 1])
        ax_inp.set_facecolor(PANEL)
        ch0 = state[0]
        inp_cmap = 'gray' if mode in ('grayscale', 'threshold') else None
        ax_inp.imshow(norm(ch0), cmap=inp_cmap,
                      interpolation='nearest', aspect='auto')
        ax_inp.axis('off')
        if row_i == 0:
            ax_inp.set_title('Input frame\n(channel 1 / 4)',
                             color=GREY, fontsize=8, pad=4)

        # ── arrow gap ──
        ax_arr = fig.add_subplot(inner[0, 2])
        ax_arr.set_facecolor(BG)
        ax_arr.axis('off')
        ax_arr.annotate('', xy=(0.85, 0.5), xytext=(0.15, 0.5),
                        xycoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color=color,
                                        lw=2.0))

        # ── feature map cells ──
        for col_i in range(N_SHOW):
            ax = fig.add_subplot(inner[0, col_i + 3])
            ax.set_facecolor(PANEL)
            fm = norm(fmaps[col_i])
            ax.imshow(fm, cmap=info['cmap'],
                      interpolation='nearest', aspect='auto')
            ax.axis('off')
            # filter index
            ax.text(0.5, -0.06, str(col_i + 1),
                    color=GREY, fontsize=6, ha='center', va='top',
                    transform=ax.transAxes)

        # ── colorbar ──
        # attach a thin colorbar to the last map cell
        ax_last = fig.add_subplot(inner[0, N_SHOW + 2])
        ax_last.set_facecolor(BG)
        ax_last.axis('off')
        sm = plt.cm.ScalarMappable(cmap=info['cmap'])
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_last, fraction=1.0, pad=0.0,
                            aspect=18)
        cbar.ax.tick_params(labelcolor=GREY, labelsize=6, length=2)
        cbar.outline.set_edgecolor(GREY)
        cbar.set_ticks([0, 0.5, 1.0])
        cbar.set_ticklabels(['low', 'mid', 'high'])

    # ── footer ──
    fig.text(0.5, 0.01,
             'Brighter = stronger filter activation.  '
             'Filter indices shown below each cell.  '
             f'Checkpoint: checkpoints/{mode}/ckpt_final.pt',
             color=GREY, fontsize=7.5, ha='center', va='bottom')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'Saved: {out_path}')


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',  default='threshold',
                        choices=['grayscale', 'threshold', 'rgb'])
    parser.add_argument('--out',   default='docs/images/feature_maps.png')
    parser.add_argument('--steps', type=int, default=80)
    args = parser.parse_args()

    cfg.PREPROCESS_MODE = args.mode
    cfg.IN_CHANNELS = cfg.N_FRAMES * (3 if args.mode == 'rgb' else 1)

    ckpt_dir = os.path.join('checkpoints', args.mode)
    ckpt = latest_checkpoint(ckpt_dir)
    if ckpt is None:
        raise FileNotFoundError(f'No checkpoint in {ckpt_dir}')

    print(f'Loading {ckpt}')
    model = DuelingCNN(cfg.IN_CHANNELS, cfg.ACTIONS).to(cfg.DEVICE)
    data  = torch.load(ckpt, map_location=cfg.DEVICE)
    model.load_state_dict(data['policy_net'])
    model.eval()

    print(f'Capturing game frame ({args.steps} steps)...')
    raw = get_game_frame(args.mode, args.steps)
    state   = build_initial_state(raw, args.mode)
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(cfg.DEVICE)

    all_maps = extract_feature_maps(model, state_t)
    draw(state, all_maps, args.mode, args.out)


if __name__ == '__main__':
    main()
