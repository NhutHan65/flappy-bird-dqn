"""Recreate network.png — arrows connect at circle edges only."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from PIL import Image
import numpy as np

RED  = '#cc0000'
DARK = '#111111'

W, H = 18, 11
fig, ax = plt.subplots(figsize=(W, H), facecolor='white')
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis('off')

# ── helpers ───────────────────────────────────────────────────────────────────

def edge(cx, cy, rx, ry, toward_x, toward_y):
    """Return the point on ellipse boundary closest to (toward_x, toward_y)."""
    dx = toward_x - cx
    dy = toward_y - cy
    if dx == 0 and dy == 0:
        return cx + rx, cy
    s = 1.0 / np.sqrt((dx / rx) ** 2 + (dy / ry) ** 2)
    return cx + s * dx, cy + s * dy

def circle(x, y, rx, ry, text, fs=11, lw=1.8):
    e = Ellipse((x, y), rx*2, ry*2, facecolor='white',
                edgecolor=DARK, linewidth=lw, zorder=3)
    ax.add_patch(e)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fs, fontweight='bold', color=DARK, zorder=4,
            multialignment='center')
    return x, y, rx, ry   # return for edge calculations

def arr_between(c1, c2, lw=2.0):
    """Arrow from edge of c1 to edge of c2. c1/c2 = (x,y,rx,ry)."""
    x1, y1, rx1, ry1 = c1
    x2, y2, rx2, ry2 = c2
    sx, sy = edge(x1, y1, rx1, ry1, x2, y2)
    ex, ey = edge(x2, y2, rx2, ry2, x1, y1)
    ax.annotate('', xy=(ex, ey), xytext=(sx, sy),
                arrowprops=dict(arrowstyle='->', color=DARK, lw=lw))

def red(x, y, lines, fs=10):
    ax.text(x, y, '\n'.join(lines), ha='center', va='center',
            fontsize=fs, color=RED, linespacing=1.6,
            fontweight='bold', multialignment='center')

# ── layout ────────────────────────────────────────────────────────────────────
TOP_Y  = 7.8
BOT_Y  = 3.8
X_IN   = 3.8;   R_IN   = (1.05, 1.05)
X_C1   = 6.5;   R_C1   = (0.88, 0.88)
X_C2   = 9.8;   R_C2   = (0.78, 0.78)
X_END  = 15.5
X_C3   = 15.5;  R_C3   = (0.78, 0.78)
X_FLAT = 12.6;  R_FLAT = (0.78, 0.78)
X_FC   = 9.8;   R_FC   = (0.82, 0.95)
X_FORK = 7.5
X_V    = 6.5;   Y_V = BOT_Y + 1.6;  R_V = (0.78, 0.65)
X_A    = 6.5;   Y_A = BOT_Y - 1.6;  R_A = (0.78, 0.65)
X_Q    = 4.8;   R_Q   = (0.78, 0.78)
X_OUT  = 2.8;   R_OUT = (0.68, 0.95)

# ── game image ────────────────────────────────────────────────────────────────
IMG_X1, IMG_X2 = 0.35, 2.4
IMG_Y1, IMG_Y2 = 5.5, 9.9
try:
    img = Image.open(r'images\grayscale_demo.gif')
    img.seek(0)
    frame = np.array(img.convert('RGB'))
    ax.imshow(frame, extent=[IMG_X1, IMG_X2, IMG_Y1, IMG_Y2], aspect='auto', zorder=2)
    rect = mpatches.FancyBboxPatch((IMG_X1, IMG_Y1), IMG_X2-IMG_X1, IMG_Y2-IMG_Y1,
        boxstyle='square,pad=0', facecolor='none', edgecolor=DARK, linewidth=2, zorder=3)
    ax.add_patch(rect)
except Exception:
    pass

# ── TOP ROW ───────────────────────────────────────────────────────────────────
# arrow from image to input
img_edge_x = IMG_X2
arr_start = (img_edge_x, TOP_Y, 0.01, 0.01)
ax.annotate('', xy=(X_IN - R_IN[0] - 0.05, TOP_Y),
            xytext=(IMG_X2 + 0.05, TOP_Y),
            arrowprops=dict(arrowstyle='->', color=DARK, lw=2.0))

n_in  = circle(X_IN,  TOP_Y, *R_IN,  '84x84x4',  fs=10.5)
n_c1  = circle(X_C1,  TOP_Y, *R_C1,  '20x20x32', fs=10.5)
n_c2  = circle(X_C2,  TOP_Y, *R_C2,  '9x9x64',   fs=10.5)

arr_between(n_in, n_c1)
arr_between(n_c1, n_c2)

red(X_C1, TOP_Y + R_C1[1] + 1.05, ['CONVOLUTION', '8x8x4x32', 'Stride: 4'])
red(X_C2, TOP_Y + R_C2[1] + 1.05, ['CONVOLUTION', '4x4x32x64', 'Stride: 2'])

# corner arrow right then down
ex, _  = edge(X_C2, TOP_Y, *R_C2, X_END, TOP_Y)
ax.annotate('', xy=(X_END, TOP_Y), xytext=(ex + 0.05, TOP_Y),
            arrowprops=dict(arrowstyle='->', color=DARK, lw=2.0))
ax.annotate('', xy=(X_END, BOT_Y + R_C3[1] + 0.05), xytext=(X_END, TOP_Y),
            arrowprops=dict(arrowstyle='->', color=DARK, lw=2.0))

# ── BOTTOM ROW ────────────────────────────────────────────────────────────────
n_c3   = circle(X_C3,   BOT_Y, *R_C3,   '7x7x64', fs=10.5)
n_flat = circle(X_FLAT, BOT_Y, *R_FLAT, '3136',   fs=10.5)
n_fc   = circle(X_FC,   BOT_Y, *R_FC,   '512\nx\n1', fs=11.5)
n_v    = circle(X_V,    Y_V,   *R_V,    'Value\nV(s)', fs=10)
n_a    = circle(X_A,    Y_A,   *R_A,    'Adv\nA(s,a)', fs=10)
n_q    = circle(X_Q,    BOT_Y, *R_Q,    'Q(s,a)', fs=10.5)
n_out  = circle(X_OUT,  BOT_Y, *R_OUT,  '2\nx\n1', fs=12)

arr_between(n_c3,   n_flat)
arr_between(n_flat, n_fc)

# Fork from FC edge to Value and Advantage
sx_v, sy_v = edge(*n_fc, X_V, Y_V)
ex_v, ey_v = edge(*n_v,  *n_fc[:2])
ax.annotate('', xy=(ex_v, ey_v), xytext=(sx_v, sy_v),
            arrowprops=dict(arrowstyle='->', color=DARK, lw=2.0))

sx_a, sy_a = edge(*n_fc, X_A, Y_A)
ex_a, ey_a = edge(*n_a,  *n_fc[:2])
ax.annotate('', xy=(ex_a, ey_a), xytext=(sx_a, sy_a),
            arrowprops=dict(arrowstyle='->', color=DARK, lw=2.0))

# Combine Value+Adv → Q
sx_qv, sy_qv = edge(*n_v, X_Q, BOT_Y)
ex_qv, ey_qv = edge(*n_q, X_V, Y_V)
ax.annotate('', xy=(ex_qv, ey_qv), xytext=(sx_qv, sy_qv),
            arrowprops=dict(arrowstyle='->', color=DARK, lw=2.0))

sx_qa, sy_qa = edge(*n_a, X_Q, BOT_Y)
ex_qa, ey_qa = edge(*n_q, X_A, Y_A)
ax.annotate('', xy=(ex_qa, ey_qa), xytext=(sx_qa, sy_qa),
            arrowprops=dict(arrowstyle='->', color=DARK, lw=2.0))

arr_between(n_q, n_out)

# ── labels ────────────────────────────────────────────────────────────────────
red(X_FLAT, BOT_Y + R_FLAT[1] + 0.55, ['Flatten'])
red(X_FC,   BOT_Y + R_FC[1]   + 0.65, ['Fully', 'Connected', 'ReLU'])
red(X_Q,    BOT_Y + R_Q[1]    + 0.6,  ['Q = V + A', '- mean(A)'])
red(X_OUT,  BOT_Y + R_OUT[1]  + 0.6,  ['Output'])
red(X_C3,   BOT_Y - R_C3[1]   - 1.1,  ['CONVOLUTION', '3x3x64x64', 'Stride: 1'])
red(X_OUT,  0.65, ['DUELING HEAD'], fs=16)
red(X_C3,   0.65, ['CONVOLUTION'],  fs=16)

plt.tight_layout(pad=0.6)
plt.savefig('images/network.png', dpi=160, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved: images/network.png')
