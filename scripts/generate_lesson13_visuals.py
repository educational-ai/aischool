"""Deterministic figures for lesson 13: the McCulloch-Pitts logical neuron.

Pure schematic/geometry figures (no dataset): threshold-element anatomy,
AND/OR/XOR on the input square, the two-layer XOR network, a neural
half-adder, plus margin schematics.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "13"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "13"

PAPER = "#fffef9"
INK = "#171915"
MUTED = "#6e726a"
FAINT = "#969990"
GRID = "#deddd4"
LINE = "#c9c8be"
BLUE = "#315f8c"
RED = "#b94a3b"
GREEN = "#38735d"
GOLD = "#a57920"
VIOLET = "#6f5a8f"
WASH = "#f5f3ea"

mpl.rcParams.update(
    {
        "font.family": "PT Sans",
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "axes.edgecolor": LINE,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.16,
        "mathtext.fontset": "dejavuserif",
    }
)


def save(fig, path, *, dpi=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def arrow(ax, s, e, *, color=MUTED, lw=1.6, rad=0.0):
    ax.add_patch(FancyArrowPatch(s, e, arrowstyle="-|>",
                                 connectionstyle=f"arc3,rad={rad}",
                                 color=color, linewidth=lw, mutation_scale=14,
                                 shrinkA=0, shrinkB=0))


def node(ax, xy, r, label, *, face=WASH, edge=INK, fs=13, tc=INK):
    ax.add_patch(Circle(xy, r, facecolor=face, edgecolor=edge, linewidth=1.6,
                        zorder=4))
    ax.text(xy[0], xy[1], label, ha="center", va="center", fontsize=fs,
            color=tc, zorder=5)


# ------------------------------------------- fig 13.1: neuron anatomy
def fig_neuron() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    node(ax, (1.2, 4.4), 0.5, "$x_1$", face=PAPER, edge=BLUE, tc=BLUE)
    node(ax, (1.2, 1.6), 0.5, "$x_2$", face=PAPER, edge=BLUE, tc=BLUE)
    node(ax, (5.6, 3.0), 0.85, "$\\Sigma$", face=WASH, edge=INK, fs=18)
    arrow(ax, (1.7, 4.3), (4.85, 3.25), color=MUTED)
    arrow(ax, (1.7, 1.7), (4.85, 2.75), color=MUTED)
    ax.text(3.2, 4.15, "$w_1=1$", fontsize=12, color=INK, ha="center")
    ax.text(3.2, 1.75, "$w_2=1$", fontsize=12, color=INK, ha="center")
    # threshold block
    box = FancyBboxPatch((7.1, 2.25), 1.9, 1.5,
                         boxstyle="round,pad=0.05,rounding_size=0.12",
                         facecolor=PAPER, edgecolor=GOLD, linewidth=1.8)
    ax.add_patch(box)
    ax.text(8.05, 3.32, "порог", ha="center", fontsize=11.5, color=GOLD)
    ax.text(8.05, 2.78, r"$z\geq b$ ?", ha="center", fontsize=13, color=INK)
    ax.text(8.05, 2.42, "$b=2$", ha="center", fontsize=10.5, color=MUTED)
    arrow(ax, (6.45, 3.0), (7.05, 3.0), color=MUTED)
    node(ax, (10.4, 3.0), 0.55, "$y$", face=PAPER, edge=GREEN, tc=GREEN)
    arrow(ax, (9.05, 3.0), (9.8, 3.0), color=MUTED)
    ax.text(5.6, 1.75, "сумма $z=w_1x_1+w_2x_2$", ha="center", fontsize=11,
            color=MUTED)
    ax.text(10.4, 1.95, "0 или 1", ha="center", fontsize=10.5, color=MUTED)
    ax.set_title("Пороговый элемент: настройка на вентиль И", fontsize=15,
                 pad=6)
    save(fig, OUT / "neuron-scheme.png")


# ------------------------------ fig 13.2: AND / OR / XOR on the square
def fig_geometry() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.8))
    corners = [(0, 0), (0, 1), (1, 0), (1, 1)]
    specs = [
        ("И", {(1, 1)}, ((-0.15, 1.55), (1.55, -0.15))),
        ("ИЛИ", {(0, 1), (1, 0), (1, 1)}, ((-0.15, 0.55), (0.55, -0.15))),
        ("XOR", {(0, 1), (1, 0)}, None),
    ]
    for ax, (name, ones, line) in zip(axes, specs):
        ax.set_xlim(-0.45, 1.45)
        ax.set_ylim(-0.45, 1.45)
        ax.set_aspect("equal")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title(name, fontsize=14)
        # unit square edges
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=LINE, lw=1.0,
                zorder=1)
        if line:
            (x0, y0), (x1, y1) = line
            ax.plot([x0, x1], [y0, y1], color=BLUE, lw=2.0, zorder=2)
        else:
            # two diagonals to show inseparability
            ax.plot([0, 1], [0, 1], color=RED, lw=1.4, ls=(0, (4, 3)),
                    zorder=2)
            ax.plot([0, 1], [1, 0], color=RED, lw=1.4, ls=(0, (4, 3)),
                    zorder=2)
            ax.text(0.5, -0.36, "диагонали пересекаются:\nпрямой не разделить",
                    ha="center", fontsize=9.5, color=RED)
        for c in corners:
            filled = c in ones
            ax.scatter([c[0]], [c[1]], s=150,
                       color=INK if filled else PAPER,
                       edgecolor=INK, linewidth=1.6, zorder=5)
    fig.suptitle("Нейрон — это прямая: И и ИЛИ разделимы, XOR — нет",
                 y=1.03, fontsize=15)
    fig.tight_layout()
    save(fig, OUT / "geometry-gates.png")


# ---------------------------------- fig 13.3: two-layer XOR network
def fig_xor() -> None:
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.8, 4.4),
                                   gridspec_kw={"width_ratios": [1.7, 1]})
    axl.set_xlim(0, 10)
    axl.set_ylim(0, 6)
    axl.axis("off")
    node(axl, (1.0, 4.4), 0.5, "$x_1$", face=PAPER, edge=BLUE, tc=BLUE)
    node(axl, (1.0, 1.6), 0.5, "$x_2$", face=PAPER, edge=BLUE, tc=BLUE)
    node(axl, (4.6, 4.4), 0.7, "ИЛИ", face=WASH, edge=INK, fs=12)
    node(axl, (4.6, 1.6), 0.7, "НЕ-И", face=WASH, edge=INK, fs=11.5)
    node(axl, (8.2, 3.0), 0.7, "И", face=WASH, edge=GREEN, fs=13, tc=GREEN)
    for a in [(1.5, 4.4), (1.5, 1.6)]:
        arrow(axl, a, (3.95, 4.4), color=MUTED)
        arrow(axl, a, (3.95, 1.6), color=MUTED)
    arrow(axl, (5.3, 4.3), (7.6, 3.2), color=MUTED)
    arrow(axl, (5.3, 1.7), (7.6, 2.8), color=MUTED)
    node(axl, (9.5, 3.0), 0.45, "$y$", face=PAPER, edge=GREEN, tc=GREEN)
    arrow(axl, (8.9, 3.0), (9.05, 3.0), color=MUTED)
    axl.text(4.6, 5.5, "слой 1", ha="center", fontsize=11, color=MUTED,
             style="italic")
    axl.text(8.2, 5.5, "слой 2", ha="center", fontsize=11, color=MUTED,
             style="italic")
    axl.set_title("XOR из трёх нейронов", fontsize=14)
    # truth table
    axr.axis("off")
    axr.set_title("проверка", fontsize=13)
    rows = [("$x_1$", "$x_2$", "XOR"),
            ("0", "0", "0"), ("0", "1", "1"),
            ("1", "0", "1"), ("1", "1", "0")]
    for i, row in enumerate(rows):
        y = 0.82 - i * 0.16
        for j, cell in enumerate(row):
            axr.text(0.2 + j * 0.3, y, cell, ha="center", fontsize=13,
                     color=INK if i else MUTED,
                     weight="bold" if i == 0 else "normal",
                     transform=axr.transAxes)
        if i == 0:
            axr.plot([0.05, 0.95], [y - 0.055, y - 0.055], color=LINE, lw=1,
                     transform=axr.transAxes)
    fig.tight_layout()
    save(fig, OUT / "xor-network.png")


# ---------------------------------- fig 13.4: neural half-adder
def fig_adder() -> None:
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.8, 4.2),
                                   gridspec_kw={"width_ratios": [1.8, 1]})
    axl.set_xlim(0, 10)
    axl.set_ylim(0, 6)
    axl.axis("off")
    node(axl, (1.0, 4.4), 0.5, "$a$", face=PAPER, edge=BLUE, tc=BLUE)
    node(axl, (1.0, 1.6), 0.5, "$b$", face=PAPER, edge=BLUE, tc=BLUE)
    node(axl, (5.2, 4.6), 0.75, "XOR", face=WASH, edge=INK, fs=11.5)
    node(axl, (5.2, 1.4), 0.7, "И", face=WASH, edge=INK, fs=13)
    for a in [(1.5, 4.4), (1.5, 1.6)]:
        arrow(axl, a, (4.5, 4.6), color=MUTED)
        arrow(axl, a, (4.5, 1.4), color=MUTED)
    node(axl, (8.4, 4.6), 0.55, "$s$", face=PAPER, edge=GREEN, tc=GREEN)
    node(axl, (8.4, 1.4), 0.55, "$c$", face=PAPER, edge=RED, tc=RED)
    arrow(axl, (5.95, 4.6), (7.85, 4.6), color=MUTED)
    arrow(axl, (5.9, 1.4), (7.85, 1.4), color=MUTED)
    axl.text(5.2, 5.7, "3 нейрона", ha="center", fontsize=10.5, color=MUTED)
    axl.text(8.4, 3.6, "бит суммы", ha="center", fontsize=10, color=GREEN)
    axl.text(8.4, 0.5, "бит переноса", ha="center", fontsize=10, color=RED)
    axl.set_title("Полусумматор из нейронов", fontsize=14)
    axr.axis("off")
    axr.set_title("сложение битов", fontsize=13)
    rows = [("$a$", "$b$", "$c$", "$s$"),
            ("0", "0", "0", "0"), ("0", "1", "0", "1"),
            ("1", "0", "0", "1"), ("1", "1", "1", "0")]
    for i, row in enumerate(rows):
        y = 0.82 - i * 0.16
        for j, cell in enumerate(row):
            col = INK
            if i and j == 2:
                col = RED
            elif i and j == 3:
                col = GREEN
            axr.text(0.15 + j * 0.24, y, cell, ha="center", fontsize=13,
                     color=col if i else MUTED,
                     weight="bold" if i == 0 else "normal",
                     transform=axr.transAxes)
        if i == 0:
            axr.plot([0.03, 0.97], [y - 0.055, y - 0.055], color=LINE, lw=1,
                     transform=axr.transAxes)
    fig.tight_layout()
    save(fig, OUT / "adder.png")


# ------------------------------------------------ margins
def side_bio() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    # dendrites
    for dy in (5.6, 4.6, 3.6, 2.6):
        ax.plot([0.6, 3.0], [dy, 4.1], color=BLUE, lw=1.4)
        ax.scatter([0.6], [dy], s=18, color=BLUE)
    ax.add_patch(Circle((3.6, 4.1), 0.9, facecolor=WASH, edgecolor=INK,
                        linewidth=1.6))
    ax.text(3.6, 4.1, "тело", ha="center", va="center", fontsize=9.5,
            color=INK)
    ax.plot([4.5, 8.6], [4.1, 4.1], color=INK, lw=2.2)
    ax.text(6.5, 4.5, "аксон", ha="center", fontsize=9.5, color=MUTED)
    for ang in (-0.5, 0, 0.5):
        ax.plot([8.6, 9.5], [4.1, 4.1 + ang], color=GREEN, lw=1.3)
    ax.text(1.4, 6.2, "дендриты — входы", fontsize=9.5, color=BLUE)
    ax.text(5.0, 2.6, "выход дальше по сети", fontsize=9.5, color=GREEN)
    save(fig, SIDE / "bio-neuron.png")


def side_principia() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((1.2, 0.8), 7.0, 8.4,
                                boxstyle="round,pad=0.1,rounding_size=0.15",
                                facecolor=PAPER, edgecolor=LINE, linewidth=1.4))
    ax.text(4.7, 8.4, "*54·43", fontsize=11, color=INK, ha="center")
    ax.text(4.7, 7.2, "$\\vdash .\\; 1 + 1 = 2$", fontsize=13, color=INK,
            ha="center")
    for y in (6.1, 5.4, 4.7, 4.0, 3.3, 2.6):
        ax.plot([1.9, 7.5], [y, y], color=GRID, lw=1.0)
    # margin mark
    ax.plot([8.0, 8.0], [5.2, 6.4], color=RED, lw=2.0)
    ax.text(8.25, 5.8, "?", fontsize=13, color=RED)
    ax.text(4.7, 1.5, "пометка на полях", fontsize=9.5, color=MUTED,
            ha="center")
    save(fig, SIDE / "principia.png")


def side_halfplane() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect("equal")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    # boundary x1+x2=1.5 (AND-ish), shade upper side
    xs = np.array([-0.5, 1.5])
    ys = 1.5 - xs
    ax.plot(xs, ys, color=BLUE, lw=2.0, zorder=3)
    poly = np.array([[1.5, 0.0], [1.5, 1.5], [0.0, 1.5]])
    ax.fill(poly[:, 0], poly[:, 1], color=BLUE, alpha=0.12, zorder=1)
    # weight vector normal to line
    arrow(ax, (0.75, 0.75), (1.05, 1.05), color=RED, lw=2.0)
    ax.text(1.08, 1.12, "$(w_1,w_2)$", fontsize=10.5, color=RED)
    for c, f in [((0, 0), False), ((0, 1), False), ((1, 0), False), ((1, 1), True)]:
        ax.scatter([c[0]], [c[1]], s=120, color=INK if f else PAPER,
                   edgecolor=INK, linewidth=1.5, zorder=5)
    ax.text(1.2, 0.35, "зона\nсрабатывания", fontsize=9, color=BLUE,
            ha="center")
    save(fig, SIDE / "halfplane.png")


fig_neuron()
fig_geometry()
fig_xor()
fig_adder()
side_bio()
side_principia()
side_halfplane()
print("lesson 13 figures written")
