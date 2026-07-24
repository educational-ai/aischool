"""Deterministic figures for lesson 27: how vision finds features.

A receptive field / convolution mechanic, Sobel edge detection on a real
photograph (Grace Hopper, bundled with matplotlib), and a bank of Gabor
filters at four orientations with their response maps. Numbers reproduced.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import fftconvolve
from matplotlib.patches import Rectangle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "27"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "27"

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


def load_gray():
    d = os.path.join(os.path.dirname(mpl.__file__), "mpl-data", "sample_data")
    img = mpimg.imread(os.path.join(d, "grace_hopper.jpg")).astype(float)
    gray = img[..., :3] @ [0.299, 0.587, 0.114]
    return gray / 255.0


GRAY = load_gray()
KX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
KY = KX.T


def conv2(a, k):
    p = np.pad(a, 1, mode="edge")
    w = sliding_window_view(p, k.shape)
    return np.einsum("ijkl,kl->ij", w, k)


def gabor(theta, ksize=15, lam=6, sigma=3, gamma=0.5):
    c = ksize // 2
    y, x = np.mgrid[-c:c + 1, -c:c + 1]
    xt = x * np.cos(theta) + y * np.sin(theta)
    yt = -x * np.sin(theta) + y * np.cos(theta)
    g = np.exp(-(xt ** 2 + gamma ** 2 * yt ** 2) / (2 * sigma ** 2)) * np.cos(2 * np.pi * xt / lam)
    return g - g.mean()


# ---------------------------- fig 27.1: receptive field / convolution
def fig_receptive() -> None:
    fig = plt.figure(figsize=(10.4, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1.2])
    # (a) image with receptive-field window
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(GRAY, cmap="gray", aspect="equal")
    ry, rx, rs = 90, 250, 60
    ax0.add_patch(Rectangle((rx, ry), rs, rs, ec=RED, fc="none", lw=2.2))
    ax0.set_title("рецептивное поле", fontsize=12.5)
    ax0.axis("off")
    # (b) the edge kernel
    axk = fig.add_subplot(gs[1])
    axk.imshow(KX, cmap="RdBu_r", vmin=-2, vmax=2)
    for (i, j), v in np.ndenumerate(KX):
        axk.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=13,
                 color=INK, fontweight="bold")
    axk.set_title("детектор края (ядро)", fontsize=12.5)
    axk.set_xticks([]); axk.set_yticks([])
    # (c) response map (edge magnitude) with same window
    ax2 = fig.add_subplot(gs[2])
    mag = np.hypot(conv2(GRAY, KX), conv2(GRAY, KY))
    ax2.imshow(mag, cmap="magma", aspect="equal")
    ax2.add_patch(Rectangle((rx, ry), rs, rs, ec=GREEN, fc="none", lw=2.2))
    ax2.set_title("карта откликов", fontsize=12.5)
    ax2.axis("off")
    # arrows between panels
    fig.text(0.365, 0.5, "скользит\nпо полю", ha="center", va="center", fontsize=9.5, color=MUTED)
    fig.text(0.635, 0.5, "даёт\nотклик", ha="center", va="center", fontsize=9.5, color=MUTED)
    print(f"receptive: edge mag mean={mag.mean():.3f} max={mag.max():.3f}")
    assert abs(mag.mean() - 0.251) < 0.01
    fig.suptitle("Нейрон смотрит в маленькое окно и отвечает на локальный признак", y=1.02, fontsize=14)
    save(fig, OUT / "receptive.png")


# ---------------------------- fig 27.2: Sobel edges on real photo
def fig_sobel() -> None:
    Gx, Gy = conv2(GRAY, KX), conv2(GRAY, KY)
    mag = np.hypot(Gx, Gy)
    strong = (mag > 0.5).mean()
    print(f"sobel: strong-edge fraction={strong:.3f}")
    assert abs(strong - 0.106) < 0.01
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 5.4))
    a1.imshow(GRAY, cmap="gray"); a1.set_title("фотография (Грейс Хоппер)", fontsize=12.5); a1.axis("off")
    a2.imshow(mag, cmap="magma", vmax=1.5); a2.set_title("карта краёв (детектор Собеля)", fontsize=12.5); a2.axis("off")
    a2.text(0.5, -0.04, f"сильных краёв: {strong*100:.0f}% пикселей",
            transform=a2.transAxes, ha="center", fontsize=10.5, color=MUTED)
    fig.suptitle("Первый шаг зрения — не узнать лицо, а найти границы", y=1.0, fontsize=14)
    fig.tight_layout()
    save(fig, OUT / "sobel.png")


# ---------------------------- fig 27.3: Gabor orientation bank
def fig_gabor() -> None:
    degs = [0, 45, 90, 135]
    resp = {}
    for deg in degs:
        resp[deg] = np.abs(fftconvolve(GRAY, gabor(np.radians(deg)), mode="same"))
    stack = np.stack([resp[d] for d in degs])
    win = stack.argmax(0)
    fracs = [100 * (win == i).mean() for i in range(4)]
    print("gabor wins:", [f"{d}:{f:.1f}%" for d, f in zip(degs, fracs)])
    assert abs(fracs[0] - 30.4) < 1.5
    fig, axes = plt.subplots(2, 4, figsize=(11.2, 5.8),
                             gridspec_kw={"height_ratios": [0.5, 1]})
    for j, deg in enumerate(degs):
        axk = axes[0, j]
        axk.imshow(gabor(np.radians(deg)), cmap="RdBu_r")
        axk.set_title(f"{deg}°", fontsize=13); axk.axis("off")
        axr = axes[1, j]
        axr.imshow(resp[deg], cmap="magma", vmax=np.percentile(resp[deg], 99))
        axr.set_xlabel(f"срабатывает на {fracs[j]:.0f}% поля", fontsize=9.5, color=MUTED)
        axr.set_xticks([]); axr.set_yticks([])
    axes[0, 0].text(-0.55, 0.5, "фильтр", transform=axes[0, 0].transAxes,
                    ha="right", va="center", fontsize=10.5, color=INK, rotation=90)
    axes[1, 0].text(-0.08, 0.5, "отклик", transform=axes[1, 0].transAxes,
                    ha="right", va="center", fontsize=10.5, color=INK, rotation=90)
    fig.suptitle("Банк фильтров ориентаций: каждый ловит края своего наклона", y=1.0, fontsize=14)
    fig.tight_layout()
    save(fig, OUT / "gabor.png")


# ------------------------------------------------ margins
def side_hubel() -> None:
    fig, ax = plt.subplots(figsize=(4.1, 2.8))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    # oriented bar stimulus
    ax.add_patch(Rectangle((1.0, 1.5), 2.4, 2.4, fc=WASH, ec=LINE, lw=1.2))
    ax.plot([1.6, 2.8], [2.1, 3.3], color=INK, lw=3)
    ax.text(2.2, 0.9, "полоска\nпод углом", ha="center", fontsize=8.5, color=MUTED)
    # arrow to a neuron firing
    ax.add_patch(FancyArrowPatch((3.6, 2.7), (6.0, 2.7), arrowstyle="-|>",
                                 color=MUTED, lw=1.6, mutation_scale=12))
    ax.plot([6.6, 6.6, 7.0, 7.0, 7.4, 7.4, 7.8, 7.8, 8.2],
            [2.7, 3.6, 3.6, 2.7, 2.7, 3.6, 3.6, 2.7, 2.7], color=RED, lw=1.4)
    ax.text(7.4, 1.9, "нейрон\nотвечает", ha="center", fontsize=8.5, color=RED)
    ax.set_title("опыт Хьюбела и Визела", fontsize=10.5)
    save(fig, SIDE / "hubel.png")


def side_shared() -> None:
    fig, ax = plt.subplots(figsize=(4.1, 2.6))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    for k in range(3):
        cx = 1.6 + k * 3.0
        ax.add_patch(Rectangle((cx, 3.2), 1.4, 1.4, fc=WASH, ec=BLUE, lw=1.4))
        ax.add_patch(Rectangle((cx - 0.6, 0.7), 2.6, 1.6, fc="none", ec=LINE, lw=1.0, ls=(0, (3, 2))))
        ax.add_patch(FancyArrowPatch((cx + 0.7, 3.1), (cx + 0.7, 2.4), arrowstyle="-|>",
                                     color=MUTED, lw=1.2, mutation_scale=9))
    ax.text(5, 5.4, "одно ядро — во всех местах", ha="center", fontsize=9, color=BLUE)
    ax.text(5, 0.2, "общие веса: мало параметров, инвариантность к сдвигу", ha="center", fontsize=8, color=MUTED)
    ax.set_title("свёртка делит веса", fontsize=10.5)
    save(fig, SIDE / "shared.png")


def side_hierarchy() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.7))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    stages = [("края", 1.4), ("части", 5.0), ("объект", 8.6)]
    for lab, cx in stages:
        ax.add_patch(plt.Circle((cx, 3.2), 0.75, fc=WASH, ec=LINE, lw=1.2))
        ax.text(cx, 3.2, lab, ha="center", va="center", fontsize=9.5, color=INK)
    for x0, x1 in [(2.2, 4.2), (5.8, 7.8)]:
        ax.add_patch(FancyArrowPatch((x0, 3.2), (x1, 3.2), arrowstyle="-|>",
                                     color=MUTED, lw=1.5, mutation_scale=12))
    ax.text(5, 5.0, "простое собирается в сложное", ha="center", fontsize=9, color=MUTED)
    ax.set_title("иерархия признаков", fontsize=10.5)
    save(fig, SIDE / "hierarchy.png")


fig_receptive()
fig_sobel()
fig_gabor()
side_hubel()
side_shared()
side_hierarchy()
print("lesson 27 figures written")
