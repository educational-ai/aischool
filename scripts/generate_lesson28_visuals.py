"""Deterministic figures for lesson 28: convolution as a sliding window.

The output-size arithmetic (padding and stride), multi-channel convolution
on a real RGB photograph, and shift-equivariance (move the input, the
response map moves with it). Numbers reproduced and asserted.
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
OUT = ROOT / "public" / "figures" / "lessons" / "28"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "28"

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


def load_rgb():
    d = os.path.join(os.path.dirname(mpl.__file__), "mpl-data", "sample_data")
    return mpimg.imread(os.path.join(d, "grace_hopper.jpg")).astype(float) / 255.0


def outsize(n, k, p, s):
    return (n + 2 * p - k) // s + 1


def conv2(a, kx):
    p = np.pad(a, 1, mode="edge")
    w = sliding_window_view(p, kx.shape)
    return np.einsum("ijkl,kl->ij", w, kx)


KX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
KY = KX.T


# ---------------------------- fig 28.1: output-size arithmetic
def fig_outsize() -> None:
    assert outsize(32, 3, 1, 1) == 32 and outsize(32, 3, 0, 1) == 30 and outsize(32, 3, 1, 2) == 16
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.4))
    cases = [
        ("дополнение $p=1$, шаг $s=1$", 1, 1, 32, "размер сохраняется"),
        ("без дополнения $p=0$", 0, 1, 30, "края теряются"),
        ("шаг $s=2$", 1, 2, 16, "карта вдвое меньше"),
    ]
    for ax, (title, p, s, nout, note) in zip(axes, cases):
        n = 8  # draw an 8x8 toy grid
        # input grid (with padding ring if p)
        for i in range(n):
            for j in range(n):
                ax.add_patch(Rectangle((j, n - 1 - i), 1, 1, fc=WASH, ec=LINE, lw=0.6))
        if p:
            for i in range(-1, n + 1):
                for j in range(-1, n + 1):
                    if i < 0 or i >= n or j < 0 or j >= n:
                        ax.add_patch(Rectangle((j, n - 1 - i), 1, 1, fc="#eef2f6", ec=LINE, lw=0.4, hatch="///"))
        # kernel window at a stride position
        ax.add_patch(Rectangle((1, n - 4), 3, 3, fc="none", ec=RED, lw=2.2))
        if s == 2:
            ax.add_patch(Rectangle((3, n - 4), 3, 3, fc="none", ec=RED, lw=1.2, ls=(0, (3, 2))))
        lo = -1.2 if p else -0.4
        ax.set_xlim(lo, n + 0.4); ax.set_ylim(lo, n + 0.4); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=11.5)
        ax.text(n / 2, lo - 0.3, f"вход 32, выход {nout}\n{note}", ha="center", va="top", fontsize=10.5,
                color=(GREEN if nout == 32 else (RED if nout < 30 else GOLD)))
    fig.suptitle(r"Размер выхода: $n_{\mathrm{out}}=\lfloor (n+2p-k)/s\rfloor+1$   (ядро $k=3$)",
                 y=1.03, fontsize=14.5)
    fig.tight_layout()
    save(fig, OUT / "outsize.png")
    print(f"outsize: p1s1={outsize(32,3,1,1)} p0s1={outsize(32,3,0,1)} s2={outsize(32,3,1,2)}")


# ---------------------------- fig 28.2: multi-channel convolution
def fig_channels() -> None:
    img = load_rgb()
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    # one kernel spanning 3 input channels -> one feature map (edge magnitude summed over channels)
    feat = np.zeros_like(R)
    for ch in (R, G, B):
        feat += np.hypot(conv2(ch, KX), conv2(ch, KY))
    params = 16 * (3 * 3 * 3 + 1)
    assert params == 448
    print(f"channels: params(3->16,k3)={params}")
    fig = plt.figure(figsize=(11.0, 4.6))
    gs = fig.add_gridspec(1, 6, width_ratios=[1.3, 1, 1, 1, 0.25, 1.3])
    ax0 = fig.add_subplot(gs[0]); ax0.imshow(img); ax0.set_title("RGB-вход\n(3 канала)", fontsize=11); ax0.axis("off")
    for k, (ch, name, cmap) in enumerate([(R, "красный", "Reds"), (G, "зелёный", "Greens"), (B, "синий", "Blues")]):
        axc = fig.add_subplot(gs[k + 1]); axc.imshow(ch, cmap=cmap); axc.set_title(name, fontsize=10.5); axc.axis("off")
    axm = fig.add_subplot(gs[5]); axm.imshow(feat, cmap="magma", vmax=2.0)
    axm.set_title("одна карта\nпризнаков", fontsize=11); axm.axis("off")
    axm.annotate("", xy=(-0.14, 0.5), xytext=(-0.30, 0.5), xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6, mutation_scale=13))
    fig.text(0.5, 0.06, "одно ядро 3×3×3 сводит три входных канала в один отклик",
             ha="center", va="center", fontsize=10.5, color=MUTED)
    fig.suptitle("Ядро охватывает все входные каналы сразу и даёт одну карту признаков", y=1.02, fontsize=14)
    save(fig, OUT / "channels.png")


# ---------------------------- fig 28.3: shift equivariance
def fig_equivariance() -> None:
    img = load_rgb()
    gray = img @ [0.299, 0.587, 0.114]
    shift = 60
    shifted = np.roll(gray, shift, axis=1)
    m0 = np.hypot(conv2(gray, KX), conv2(gray, KY))
    m1 = np.hypot(conv2(shifted, KX), conv2(shifted, KY))
    m0s = np.roll(m0, shift, axis=1)
    # equivariance: conv(shift(x)) == shift(conv(x)) away from the wrap seam
    diff = np.abs(m1[:, shift + 5:-5] - m0s[:, shift + 5:-5]).mean()
    print(f"equivariance: mean|conv(shift)-shift(conv)|={diff:.4f}")
    assert diff < 1e-6
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.0))
    axes[0, 0].imshow(gray, cmap="gray"); axes[0, 0].set_title("вход", fontsize=11.5); axes[0, 0].axis("off")
    axes[0, 1].imshow(shifted, cmap="gray"); axes[0, 1].set_title(f"вход, сдвинут на {shift} пикс.", fontsize=11.5); axes[0, 1].axis("off")
    axes[1, 0].imshow(m0, cmap="magma", vmax=1.5); axes[1, 0].set_title("карта откликов", fontsize=11.5); axes[1, 0].axis("off")
    axes[1, 1].imshow(m1, cmap="magma", vmax=1.5); axes[1, 1].set_title("карта откликов — сдвинулась так же", fontsize=11.5); axes[1, 1].axis("off")
    for ax in (axes[0, 1], axes[1, 1]):
        ax.axvline(shift, color=GREEN, lw=1.4, ls=(0, (4, 3)))
    fig.suptitle("Эквивариантность: сдвинь вход — карта откликов сдвинется точно так же", y=1.0, fontsize=14)
    fig.tight_layout()
    save(fig, OUT / "equivariance.png")


# ------------------------------------------------ margins
def side_padding() -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(4.4, 2.5))
    for ax, mode, tag in [(a1, "zero", "нулевая рамка"), (a2, "edge", "повтор края")]:
        n = 5
        for i in range(n):
            for j in range(n):
                ax.add_patch(Rectangle((j, i), 1, 1, fc=WASH, ec=LINE, lw=0.5))
        for i in range(-1, n + 1):
            for j in range(-1, n + 1):
                if i < 0 or i >= n or j < 0 or j >= n:
                    fc = "#ffffff" if mode == "zero" else "#e7edf3"
                    ax.add_patch(Rectangle((j, i), 1, 1, fc=fc, ec=LINE, lw=0.4, hatch="..."))
        ax.set_xlim(-1.2, n + 0.2); ax.set_ylim(-1.2, n + 0.2); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(tag, fontsize=9)
    fig.suptitle("чем заполнить край", y=1.02, fontsize=10.5)
    fig.tight_layout()
    save(fig, SIDE / "padding.png")


def side_stride() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    n = 8
    for i in range(n):
        for j in range(n):
            taken = (i % 2 == 0 and j % 2 == 0)
            ax.add_patch(Rectangle((j, n - 1 - i), 1, 1, fc=(GREEN if taken else WASH),
                                   ec=LINE, lw=0.5, alpha=0.85 if taken else 1))
    ax.set_xlim(-0.3, n + 0.3); ax.set_ylim(-0.3, n + 0.3); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("шаг 2 = брать через клетку", fontsize=10.5)
    save(fig, SIDE / "stride.png")


def side_params() -> None:
    fig, ax = plt.subplots(figsize=(4.1, 2.5))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.text(5, 4.6, "параметры одного слоя", ha="center", fontsize=10, color=INK)
    ax.text(5, 3.2, r"$C_{\mathrm{out}}\,(k_h k_w C_{\mathrm{in}}+1)$", ha="center", fontsize=15, color=BLUE)
    ax.text(5, 1.5, "RGB, 16 ядер 3×3:  16·(27+1) = 448", ha="center", fontsize=10, color=MUTED)
    ax.text(5, 0.6, "не зависит от размера картинки", ha="center", fontsize=9, color=GREEN)
    save(fig, SIDE / "params.png")


fig_outsize()
fig_channels()
fig_equivariance()
side_padding()
side_stride()
side_params()
print("lesson 28 figures written")
