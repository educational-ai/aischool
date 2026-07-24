"""Deterministic figures for lesson 34: semantic segmentation.

A real image segmented into per-pixel regions by colour, the encoder-decoder
(U-Net) shape with skip connections, and the IoU metric — including how pixel
accuracy lies for small objects. Numbers reproduced and asserted.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "34"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "34"

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


# ---------------------------- fig 34.1: image -> per-pixel segmentation
def fig_segmentation() -> None:
    img = load_rgb()
    small = img[::4, ::4]
    H, Wd, _ = small.shape
    km = KMeans(n_clusters=4, random_state=34, n_init=4).fit(small.reshape(-1, 3))
    labels = km.labels_.reshape(H, Wd)
    seg = km.cluster_centers_[labels]
    fracs = [100 * (labels == k).mean() for k in range(4)]
    print("seg fractions:", [round(f, 1) for f in fracs])
    assert abs(max(fracs) - 55.0) < 3
    # a discrete class map with distinct palette
    palette = np.array([[0.19, 0.37, 0.55], [0.12, 0.12, 0.14], [0.72, 0.43, 0.33], [0.85, 0.78, 0.7]])
    order = np.argsort(km.cluster_centers_.sum(1))
    remap = {old: new for new, old in enumerate(order)}
    classmap = np.vectorize(lambda v: remap[v])(labels)
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 4.2))
    axes[0].imshow(small); axes[0].set_title("вход: пиксели", fontsize=12); axes[0].axis("off")
    axes[1].imshow(seg); axes[1].set_title("сгруппировано по цвету", fontsize=12); axes[1].axis("off")
    axes[2].imshow(palette[classmap]); axes[2].set_title("маска: класс каждого пикселя", fontsize=12); axes[2].axis("off")
    fig.text(0.5, -0.02, "классификация даёт один ярлык на всю картинку; сегментация — свой ярлык каждому пикселю",
             ha="center", fontsize=10.5, color=MUTED)
    fig.suptitle("Семантическая сегментация: каждому пикселю — свой класс", y=1.02, fontsize=14)
    fig.tight_layout()
    save(fig, OUT / "segmentation.png")


# ---------------------------- fig 34.2: encoder-decoder (U-Net)
def fig_encoderdecoder() -> None:
    fig, ax = plt.subplots(figsize=(10.6, 5.0))
    ax.axis("off"); ax.set_xlim(0, 14); ax.set_ylim(0, 8)
    # encoder blocks (shrinking), bottleneck, decoder blocks (growing)
    enc = [(1.0, 5.0, "вход"), (2.6, 4.0, ""), (4.0, 3.0, ""), (5.2, 2.2, "")]
    dec = [(8.8, 2.2, ""), (10.0, 3.0, ""), (11.4, 4.0, ""), (13.0, 5.0, "выход-маска")]
    bott = (7.0, 1.4)
    for x, h, lab in enc:
        ax.add_patch(Rectangle((x - 0.35, 4 - h / 2), 0.7, h, fc="#e7eef4", ec=BLUE, lw=1.6))
        if lab:
            ax.text(x, 4 - h / 2 - 0.4, lab, ha="center", va="top", fontsize=9.5, color=BLUE)
    ax.add_patch(Rectangle((bott[0] - 0.35, 4 - bott[1] / 2), 0.7, bott[1], fc=WASH, ec=INK, lw=1.6))
    ax.text(bott[0], 4 - bott[1] / 2 - 0.4, "бутылочное\nгорло", ha="center", va="top", fontsize=9, color=INK)
    for x, h, lab in dec:
        ax.add_patch(Rectangle((x - 0.35, 4 - h / 2), 0.7, h, fc="#eef4ef", ec=GREEN, lw=1.6))
        if lab:
            ax.text(x, 4 - h / 2 - 0.4, lab, ha="center", va="top", fontsize=9.5, color=GREEN)
    # flow arrows
    xs = [1.0, 2.6, 4.0, 5.2, 7.0, 8.8, 10.0, 11.4, 13.0]
    for a, b in zip(xs[:-1], xs[1:]):
        ax.add_patch(FancyArrowPatch((a + 0.4, 4), (b - 0.4, 4), arrowstyle="-|>", color=MUTED, lw=1.2, mutation_scale=9))
    # skip connections
    for (xa, xb) in [(2.6, 11.4), (4.0, 10.0), (5.2, 8.8)]:
        ax.add_patch(FancyArrowPatch((xa, 4 + 1.6), (xb, 4 + 1.6), connectionstyle="arc3,rad=-0.35",
                                     arrowstyle="-|>", color=RED, lw=1.4, mutation_scale=9, ls="--"))
    ax.text(7.0, 7.3, "короткие пути (skip): возвращают декодеру точные границы", ha="center", fontsize=10, color=RED)
    ax.text(3.0, 0.6, "энкодер: сжимает — «что на картинке»", ha="center", fontsize=9.5, color=BLUE)
    ax.text(11.0, 0.6, "декодер: расширяет — «где именно»", ha="center", fontsize=9.5, color=GREEN)
    ax.set_title("Энкодер—декодер (U-Net): сжать, понять, вернуть разрешение", y=1.0)
    save(fig, OUT / "encoderdecoder.png")
    print("encoderdecoder drawn")


# ---------------------------- fig 34.3: IoU + accuracy lie
def fig_iou() -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})
    # left: two overlapping masks
    a1.add_patch(Circle((0.42, 0.5), 0.30, fc=BLUE, ec="none", alpha=0.45))
    a1.add_patch(Circle((0.60, 0.5), 0.30, fc=RED, ec="none", alpha=0.45))
    a1.text(0.30, 0.5, "истина", ha="center", fontsize=10, color=BLUE)
    a1.text(0.74, 0.5, "предсказано", ha="center", fontsize=10, color=RED)
    a1.text(0.51, 0.5, r"$\cap$", ha="center", va="center", fontsize=16, color=INK)
    a1.text(0.5, 0.12, r"$\mathrm{IoU}=\dfrac{|\text{пересечение}|}{|\text{объединение}|}=\dfrac{80}{120}=0{,}67$",
            ha="center", fontsize=12, color=INK)
    a1.set_xlim(0, 1); a1.set_ylim(0, 1); a1.axis("off")
    a1.set_title("IoU: перекрытие масок", fontsize=12.5)
    # right: accuracy vs IoU for a small object
    fracs = [2, 5, 10, 30]
    acc = [100 - f for f in fracs]
    iou = [0, 0, 0, 0]
    xpos = np.arange(len(fracs))
    a2.bar(xpos - 0.2, acc, 0.4, color=GREEN, label="пиксельная точность")
    a2.bar(xpos + 0.2, iou, 0.4, color=RED, label="IoU объекта")
    for i, (a, f) in enumerate(zip(acc, fracs)):
        a2.text(i - 0.2, a + 2, f"{a}%", ha="center", fontsize=9.5, color=GREEN)
        a2.text(i + 0.2, 3, "0%", ha="center", fontsize=9.5, color=RED)
    a2.set_xticks(xpos); a2.set_xticklabels([f"объект\n{f}%" for f in fracs], fontsize=9.5)
    a2.set_ylim(0, 108); a2.set_ylabel("%")
    a2.set_title("«Всё фон»: точность высока, IoU — ноль", fontsize=12.5)
    a2.legend(loc="center right", frameon=False, fontsize=10)
    a2.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); a2.set_axisbelow(True)
    fig.suptitle("IoU честнее пиксельной точности, особенно для мелких объектов", y=1.02, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "iou.png")
    print("iou drawn")


# ------------------------------------------------ margins
def side_ioudice() -> None:
    fig, ax = plt.subplots(figsize=(4.1, 2.5))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.text(5, 4.9, "две меры перекрытия", ha="center", fontsize=10, color=INK)
    ax.text(5, 3.5, r"$\mathrm{IoU}=\dfrac{|A\cap B|}{|A\cup B|}$", ha="center", fontsize=14, color=BLUE)
    ax.text(5, 1.7, r"$\mathrm{Dice}=\dfrac{2|A\cap B|}{|A|+|B|}=\dfrac{2\,\mathrm{IoU}}{1+\mathrm{IoU}}$",
            ha="center", fontsize=12, color=RED)
    ax.text(5, 0.4, "IoU 0,67 ↔ Dice 0,80".replace("↔", "—"), ha="center", fontsize=9, color=MUTED)
    save(fig, SIDE / "ioudice.png")


def side_smallobject() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    ax.add_patch(Rectangle((0, 0), 1, 1, fc="#eef2f6", ec=LINE, lw=1.0))
    ax.add_patch(Rectangle((0.62, 0.55), 0.12, 0.12, fc=RED, ec="none"))
    ax.text(0.5, -0.12, "объект — крошечная доля кадра", ha="center", fontsize=9, color=MUTED)
    ax.text(0.3, 0.3, "фон 98%", fontsize=9, color=MUTED)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.axis("off")
    ax.set_title("мелкий объект тонет в точности", fontsize=10)
    save(fig, SIDE / "smallobject.png")


def side_spacenet() -> None:
    fig, ax = plt.subplots(figsize=(4.1, 2.5))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    for i in range(4):
        for j in range(6):
            col = BLUE if j < 4 else GREEN
            ax.add_patch(Rectangle((1 + j * 1.2, 1.5 + i * 0.7), 1.1, 0.6, fc=col, ec=PAPER, lw=0.8, alpha=0.7))
    ax.text(3.4, 5.2, "город A — обучение", ha="center", fontsize=8.5, color=BLUE)
    ax.text(7.6, 5.2, "город B — тест", ha="center", fontsize=8.5, color=GREEN)
    ax.text(5, 0.6, "делить по географии, не по тайлам", ha="center", fontsize=9, color=INK)
    ax.set_title("спутник: не смешивай районы", fontsize=10.5)
    save(fig, SIDE / "spacenet.png")


fig_segmentation()
fig_encoderdecoder()
fig_iou()
side_ioudice()
side_smallobject()
side_spacenet()
print("lesson 34 figures written")
