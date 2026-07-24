"""Deterministic figures for lesson 30: from the neocognitron to the CNN.

The LeNet-style architecture as a shrinking-and-widening stack, real feature
maps through a conv/pool pipeline (edges -> blobs), and the parameter budget
(96% of weights live in the dense head; global average pooling replaces it
with a tiny fraction). Numbers reproduced and asserted.
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "30"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "30"

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


def outsize(n, k, p, s):
    return (n + 2 * p - k) // s + 1


# ---------------------------- fig 30.1: architecture as a shrinking stack
def fig_architecture() -> None:
    # (label, spatial, channels)
    stages = [
        ("вход", 32, 1), ("conv 5×5", 28, 6), ("pool", 14, 6),
        ("conv 5×5", 10, 16), ("pool", 5, 16), ("FC", 1, 120), ("выход", 1, 10),
    ]
    assert outsize(32, 5, 0, 1) == 28 and outsize(28, 2, 0, 2) == 14
    assert outsize(14, 5, 0, 1) == 10 and outsize(10, 2, 0, 2) == 5
    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    ax.axis("off"); ax.set_xlim(0, 34); ax.set_ylim(-2, 10)
    x = 1.0
    for k, (lab, sp, ch) in enumerate(stages):
        h = 0.6 + sp / 5.5              # box height ~ spatial size
        w = 0.5 + ch / 22.0            # box width ~ channels
        cy = 3.5
        col = WASH if lab in ("вход", "выход") else ("#eef2f6" if "conv" in lab else ("#f3ece0" if lab == "pool" else "#eef4ef"))
        ax.add_patch(Rectangle((x, cy - h / 2), w, h, fc=col, ec=BLUE if "conv" in lab else (GOLD if lab == "pool" else LINE), lw=1.6))
        ax.text(x + w / 2, cy - h / 2 - 0.5, lab, ha="center", va="top", fontsize=10, color=INK)
        if sp > 1:
            ax.text(x + w / 2, cy + h / 2 + 0.25, f"{sp}×{sp}", ha="center", va="bottom", fontsize=9.5, color=MUTED)
            ax.text(x + w / 2, cy, f"×{ch}", ha="center", va="center", fontsize=9, color=BLUE, rotation=90)
        else:
            ax.text(x + w / 2, cy + h / 2 + 0.25, f"{ch}", ha="center", va="bottom", fontsize=9.5, color=MUTED)
        if k < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.05, cy), (x + w + 0.75, cy),
                                         arrowstyle="-|>", color=MUTED, lw=1.3, mutation_scale=10))
        x += w + 1.3
    ax.text(17, 8.6, "пространство сжимается, каналов становится больше", ha="center", fontsize=12, color=INK)
    ax.text(17, 7.7, "сторона: 32, 28, 14, 10, 5        каналы: 1, 6, 16", ha="center", fontsize=11, color=MUTED)
    ax.set_title("Свёрточная сеть: конус от пикселей к признакам", y=1.0)
    save(fig, OUT / "architecture.png")
    print("architecture: sizes 32->28->14->10->5 ok")


# ============================================================ real feature maps
def load_gray():
    d = os.path.join(os.path.dirname(mpl.__file__), "mpl-data", "sample_data")
    return mpimg.imread(os.path.join(d, "grace_hopper.jpg")).astype(float) @ [0.299, 0.587, 0.114] / 255.0


def conv(a, k):
    p = np.pad(a, ((k.shape[0] // 2,) * 2, (k.shape[1] // 2,) * 2), mode="edge")
    return np.einsum("ijkl,kl->ij", sliding_window_view(p, k.shape), k)


def relu(a):
    return np.maximum(a, 0)


def maxpool(a, k=2):
    h, w = a.shape; oh, ow = h // k, w // k
    return a[:oh * k, :ow * k].reshape(oh, k, ow, k).max((1, 3))


# ---------------------------- fig 30.2: feature maps through the stack
def fig_featuremaps() -> None:
    g = load_gray()
    KX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
    # stage 1: edges
    e = relu(np.hypot(conv(g, KX), conv(g, KX.T)))
    p1 = maxpool(e)
    # stage 2: second-order (blobs / corners) via Laplacian-of-edges
    lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float)
    b = relu(conv(p1, lap))
    p2 = maxpool(b)
    fig, axes = plt.subplots(1, 5, figsize=(11.4, 3.0))
    panels = [(g, "вход\n600×512", "gray"), (p1, "слой 1: края\n300×256", "magma"),
              (p2, "слой 2: пятна и углы\n150×128", "magma"),
              (maxpool(p2), "слой 3\n75×64", "magma"), (maxpool(maxpool(p2)), "слой 4\n37×32", "magma")]
    for ax, (im, title, cmap) in zip(axes, panels):
        ax.imshow(im, cmap=cmap, vmax=None if cmap == "gray" else np.percentile(im, 99) + 1e-6)
        ax.set_title(title, fontsize=10); ax.axis("off")
    print(f"featuremaps: {g.shape} -> {p1.shape} -> {p2.shape}")
    fig.suptitle("Слой за слоем: от пикселей к краям, от краёв к частям", y=1.03, fontsize=14)
    fig.tight_layout()
    save(fig, OUT / "featuremaps.png")


# ---------------------------- fig 30.3: parameter budget
def fig_params() -> None:
    conv_p = 156 + 2416
    fc_p = 48120 + 10164 + 850
    gap_p = 170
    assert conv_p == 2572 and fc_p == 59134
    print(f"params: conv={conv_p} fc={fc_p} ({100*fc_p/(conv_p+fc_p):.0f}% in head), gap={gap_p}")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
    # stacked bar: where the weights live
    a1.bar([0], [conv_p], width=0.5, color=BLUE, label=f"свёртки: {conv_p}")
    a1.bar([0], [fc_p], width=0.5, bottom=[conv_p], color=RED, label=f"полносвязная голова: {fc_p}")
    a1.text(0.32, conv_p + fc_p * 0.6, f"{100*fc_p/(conv_p+fc_p):.0f}% весов —\nв голове", ha="left", fontsize=11, color=RED)
    a1.set_xlim(-0.7, 0.9); a1.set_xticks([]); a1.set_ylabel("число весов"); a1.set_ylim(0, 68000)
    a1.set_title("Где живут веса LeNet", fontsize=12.5)
    a1.legend(loc="center left", frameon=False, fontsize=10)
    a1.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); a1.set_axisbelow(True)
    # comparison: FC head vs GAP head
    a2.bar([0, 1], [fc_p, gap_p], width=0.55, color=[RED, GREEN])
    a2.text(0, fc_p + 2500, f"{fc_p}", ha="center", fontsize=11, color=RED)
    a2.text(1, gap_p + 2500, f"{gap_p}", ha="center", fontsize=11, color=GREEN)
    a2.set_xticks([0, 1]); a2.set_xticklabels(["полносвязная\nголова", "глобальное\nсреднее (GAP)"], fontsize=10)
    a2.set_ylabel("число весов головы")
    a2.set_title(f"GAP убирает голову в {fc_p//gap_p} раз", fontsize=12.5)
    a2.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); a2.set_axisbelow(True)
    fig.suptitle("Свёртки дёшевы по весам; дорога полносвязная голова", y=1.02, fontsize=14)
    fig.tight_layout()
    save(fig, OUT / "params.png")


# ------------------------------------------------ margins
def side_rfgrowth() -> None:
    fig, ax = plt.subplots(figsize=(4.1, 2.7))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    for k, (rf, cy, col) in enumerate([(1, 5.0, LINE), (5, 3.7, BLUE), (14, 2.0, GREEN), (16, 0.6, RED)]):
        ax.text(0.5, cy, f"слой {k}", fontsize=9, color=MUTED, va="center")
        ax.add_patch(Rectangle((3.0, cy - 0.28), rf / 3.6, 0.56, fc="none", ec=col, lw=1.6))
        ax.text(3.1 + rf / 3.6, cy, f"{rf}×{rf}", fontsize=8.5, color=col, va="center")
    ax.text(5, 5.7, "рецептивное поле растёт вглубь", ha="center", fontsize=9, color=INK)
    ax.set_title("одна клетка видит всё больше", fontsize=10.5)
    save(fig, SIDE / "rfgrowth.png")


def side_channels() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    stage = [0, 1, 2, 3, 4]
    space = [32, 28, 14, 10, 5]
    chan = [1, 6, 6, 16, 16]
    ax.plot(stage, space, color=BLUE, lw=1.8, marker="o", markersize=4, label="сторона карты")
    ax2 = ax.twinx()
    ax2.plot(stage, chan, color=RED, lw=1.8, marker="s", markersize=4, label="каналов")
    ax.set_xlabel("слой", fontsize=9); ax.set_ylabel("сторона", color=BLUE, fontsize=9)
    ax2.set_ylabel("каналы", color=RED, fontsize=9)
    ax.set_xticks(stage)
    ax.set_title("пространство ↓, каналы ↑".replace("↓", "вниз").replace("↑", "вверх"), fontsize=10)
    save(fig, SIDE / "channels.png")


def side_neocognitron() -> None:
    fig, ax = plt.subplots(figsize=(4.1, 2.5))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    for cx, lab, col in [(1.6, "S-клетки\n(признак)", BLUE), (5.0, "C-клетки\n(терпят сдвиг)", GOLD), (8.4, "…", MUTED)]:
        ax.add_patch(Rectangle((cx - 0.9, 1.7), 1.8, 1.6, fc=WASH, ec=col, lw=1.4))
        ax.text(cx, 2.5, lab, ha="center", va="center", fontsize=8, color=INK)
    for x0 in (2.6, 6.0):
        ax.add_patch(FancyArrowPatch((x0, 2.5), (x0 + 1.4, 2.5), arrowstyle="-|>", color=MUTED, lw=1.3, mutation_scale=10))
    ax.text(5, 4.3, "неокогнитрон: чередование S и C", ha="center", fontsize=9, color=INK)
    ax.set_title("предок свёрточной сети", fontsize=10.5)
    save(fig, SIDE / "neocognitron.png")


fig_architecture()
fig_featuremaps()
fig_params()
side_rfgrowth()
side_channels()
side_neocognitron()
print("lesson 30 figures written")
