"""Deterministic figures for lesson 29: pooling and shift-invariance.

Max vs average pooling on a real handwritten digit, the shift-invariance a
2x2 max-pool buys on a real feature map (measured), and how the gradient
flows only to the argmax cell. Numbers reproduced and asserted.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from matplotlib.patches import Rectangle, FancyArrowPatch
from sklearn.datasets import load_digits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "29"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "29"

PAPER = "#fffef9"; INK = "#171915"; MUTED = "#6e726a"; FAINT = "#969990"
GRID = "#deddd4"; LINE = "#c9c8be"; BLUE = "#315f8c"; RED = "#b94a3b"
GREEN = "#38735d"; GOLD = "#a57920"; VIOLET = "#6f5a8f"; WASH = "#f5f3ea"

mpl.rcParams.update({
    "font.family": "PT Sans", "font.size": 12, "axes.titlesize": 15,
    "axes.labelsize": 12, "axes.edgecolor": LINE, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.16, "mathtext.fontset": "dejavuserif",
})


def save(fig, path, *, dpi=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def maxpool(a, k=2):
    h, w = a.shape; oh, ow = h // k, w // k
    return a[:oh * k, :ow * k].reshape(oh, k, ow, k).max((1, 3))


def avgpool(a, k=2):
    h, w = a.shape; oh, ow = h // k, w // k
    return a[:oh * k, :ow * k].reshape(oh, k, ow, k).mean((1, 3))


# ---------------------------- fig 29.1: max vs avg pool on a real digit
def fig_pooling() -> None:
    D = load_digits(); X = D.images / 16.0; y = D.target
    img = X[np.where(y == 3)[0][0]]  # a real handwritten 3
    mp, ap = maxpool(img), avgpool(img)
    assert img.shape == (8, 8) and mp.shape == (4, 4)
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.9))
    for ax, a, title in [(axes[0], img, "цифра  8×8"),
                         (axes[1], mp, "max-пулинг  4×4"),
                         (axes[2], ap, "average-пулинг  4×4")]:
        ax.imshow(a, cmap="magma", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
    # draw the 2x2 window grid on the source
    for i in range(0, 8, 2):
        for j in range(0, 8, 2):
            axes[0].add_patch(Rectangle((j - 0.5, i - 0.5), 2, 2, fc="none", ec=GREEN, lw=1.0))
    fig.text(0.5, -0.02, "каждое окно 2×2 сжимается в одно число: max берёт сильнейший отклик, average — средний",
             ha="center", fontsize=10.5, color=MUTED)
    fig.suptitle("Пулинг сжимает карту, огрубляя положение", y=1.02, fontsize=14)
    fig.tight_layout()
    save(fig, OUT / "pooling.png")
    print(f"pooling: {img.shape} -> {mp.shape}")


# ============================================================ real feature map
def load_gray():
    d = os.path.join(os.path.dirname(mpl.__file__), "mpl-data", "sample_data")
    return mpimg.imread(os.path.join(d, "grace_hopper.jpg")).astype(float) @ [0.299, 0.587, 0.114] / 255.0


KX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float); KY = KX.T


def conv(a, k):
    p = np.pad(a, 1, mode="edge")
    return np.einsum("ijkl,kl->ij", sliding_window_view(p, k.shape), k)


def edge(a):
    return np.hypot(conv(a, KX), conv(a, KY))


def relchange(a, b, crop=4):
    a = a[crop:-crop, crop:-crop]; b = b[crop:-crop, crop:-crop]
    return np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-9)


# ---------------------------- fig 29.2: shift-invariance
def fig_invariance() -> None:
    gray = load_gray()
    e0 = edge(gray); e1 = edge(np.roll(gray, 1, axis=1))
    raw = relchange(e1, e0); pool = relchange(maxpool(e1), maxpool(e0))
    print(f"invariance: unpooled change={raw:.3f}, pooled change={pool:.3f}, {raw/pool:.2f}x")
    assert abs(raw - 0.567) < 0.02 and abs(pool - 0.380) < 0.02
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4))
    axes[0, 0].imshow(e0, cmap="magma", vmax=1.5); axes[0, 0].set_title("карта признаков", fontsize=11.5)
    axes[0, 1].imshow(e1, cmap="magma", vmax=1.5); axes[0, 1].set_title(f"вход сдвинут на 1 пикс.\nизменение {raw*100:.0f}%", fontsize=11.5)
    axes[1, 0].imshow(maxpool(e0), cmap="magma", vmax=1.5); axes[1, 0].set_title("после max-пулинга 2×2", fontsize=11.5)
    axes[1, 1].imshow(maxpool(e1), cmap="magma", vmax=1.5); axes[1, 1].set_title(f"тот же сдвиг\nизменение всего {pool*100:.0f}%", fontsize=11.5)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Max-пулинг делает признак в {raw/pool:.1f} раза устойчивее к сдвигу на пиксель",
                 y=1.0, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "invariance.png")


# ---------------------------- fig 29.3: gradient through max
def fig_gradient() -> None:
    win = np.array([[0.2, 0.9], [0.4, 0.3]])
    am = np.unravel_index(win.argmax(), win.shape)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.8))
    # forward
    a1.imshow(win, cmap="Blues", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(win):
        a1.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=15,
                color=(RED if (i, j) == am else INK), fontweight="bold")
    a1.add_patch(Rectangle((am[1] - 0.5, am[0] - 0.5), 1, 1, fc="none", ec=RED, lw=2.4))
    a1.set_title("прямой ход: max = 0,9", fontsize=12); a1.set_xticks([]); a1.set_yticks([])
    # backward
    g = np.zeros_like(win); g[am] = 1.0
    a2.imshow(g, cmap="Reds", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(g):
        a2.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=15,
                color=(RED if (i, j) == am else MUTED), fontweight="bold")
    a2.set_title("обратный ход: градиент течёт только в максимум", fontsize=12); a2.set_xticks([]); a2.set_yticks([])
    fig.suptitle("Градиент через max попадает лишь в ту клетку, что победила", y=1.02, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "gradient.png")
    print(f"gradient: argmax at {am}")


# ------------------------------------------------ margins
def side_maxavg() -> None:
    fig, ax = plt.subplots(figsize=(4.1, 2.6))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.text(2.5, 5.0, "max", ha="center", fontsize=12, color=RED)
    ax.text(2.5, 3.6, "есть ли\nпризнак", ha="center", fontsize=9, color=MUTED)
    ax.text(7.5, 5.0, "average", ha="center", fontsize=12, color=BLUE)
    ax.text(7.5, 3.6, "сколько\nего", ha="center", fontsize=9, color=MUTED)
    ax.plot([5, 5], [0.5, 5.5], color=LINE, lw=1.0)
    ax.text(5, 1.4, "max ловит яркий\nотклик где угодно\nв окне; average\nразмывает", ha="center", fontsize=8, color=INK)
    ax.set_title("два способа сжать окно", fontsize=10.5)
    save(fig, SIDE / "maxavg.png")


def side_aliasing() -> None:
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.0, 2.8))
    x = np.linspace(0, 12, 400)
    s = np.arange(0.4, 12, 1.7)
    a1.plot(x, np.sin(6 * x), color=LINE, lw=1.0)
    a1.plot(s, np.sin(6 * s), color=RED, lw=1.4, marker="o", markersize=3)
    a1.set_title("редкие отсчёты дают ложный узор", fontsize=9); a1.set_xticks([]); a1.set_yticks([])
    a1.set_ylim(-1.3, 1.3)
    # smooth first: low-frequency signal the sparse samples follow honestly
    a2.plot(x, np.sin(6 * x), color="#e4e3da", lw=0.8)
    a2.plot(x, 0.85 * np.sin(0.9 * x), color=LINE, lw=1.0)
    a2.plot(s, 0.85 * np.sin(0.9 * s), color=GREEN, lw=1.4, marker="o", markersize=3)
    a2.set_title("сгладить сначала — честно", fontsize=9); a2.set_xticks([]); a2.set_yticks([])
    a2.set_ylim(-1.3, 1.3)
    fig.suptitle("aliasing: сгладь перед сжатием", y=1.04, fontsize=10)
    fig.tight_layout()
    save(fig, SIDE / "aliasing.png")


def side_stride() -> None:
    fig, ax = plt.subplots(figsize=(4.1, 2.4))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    ax.text(5, 4.3, "уменьшить карту вдвое", ha="center", fontsize=9.5, color=INK)
    ax.text(2.6, 2.7, "пулинг\n(без весов,\nберёт max)", ha="center", fontsize=8.5, color=RED)
    ax.text(7.4, 2.7, "свёртка с шагом 2\n(учит, что брать)", ha="center", fontsize=8.5, color=BLUE)
    ax.plot([5, 5], [0.6, 3.5], color=LINE, lw=1.0)
    ax.set_title("два пути к даунсэмплингу", fontsize=10.5)
    save(fig, SIDE / "stride.png")


fig_pooling()
fig_invariance()
fig_gradient()
side_maxavg()
side_aliasing()
side_stride()
print("lesson 29 figures written")
