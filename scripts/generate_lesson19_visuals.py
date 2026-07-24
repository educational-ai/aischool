"""Deterministic figures for lesson 19: universal approximation.

Bump from two sigmoids, the real bike hourly-demand curve approximated by
sums of K sigmoid blocks (least-squares readout on spread sigmoid features),
width-vs-depth schematic, plus margins. Errors reproduced.
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "19"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "19"

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


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ------------------------------------------- bike hourly demand
def bike_curve():
    rows = [r for r in csv.DictReader((ROOT / "scripts" / "data" / "bike-sharing-hour.csv").open())]
    d = defaultdict(list)
    for r in rows:
        if r["workingday"] == "1":
            d[int(r["hr"])].append(int(r["cnt"]))
    xs = np.array(sorted(d), float)
    ys = np.array([statistics.mean(d[int(h)]) for h in xs])
    return xs, ys


def approx(xs, ys, K, slope=12.0):
    xn = xs / 23.0
    centers = np.linspace(0, 1, K)
    Phi = np.column_stack([sigmoid(slope * (xn - c)) for c in centers] + [np.ones_like(xn)])
    v, _, _, _ = np.linalg.lstsq(Phi, ys, rcond=None)
    # dense curve
    xd = np.linspace(0, 23, 300)
    xdn = xd / 23.0
    Phid = np.column_stack([sigmoid(slope * (xdn - c)) for c in centers] + [np.ones_like(xdn)])
    pred_dense = Phid @ v
    rmse = np.sqrt(((Phi @ v - ys) ** 2).mean())
    return xd, pred_dense, rmse


# ------------------------------------- fig 19.1: bump from sigmoids
def fig_bump() -> None:
    x = np.linspace(-4, 4, 400)
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.4, 4.0))
    s1 = sigmoid(5 * (x + 1)); s2 = sigmoid(5 * (x - 1))
    axl.plot(x, s1, color=BLUE, lw=2.2, label="$\\sigma(k(x-a))$")
    axl.plot(x, s2, color=GOLD, lw=2.2, label="$\\sigma(k(x-b))$")
    axl.set_title("две сдвинутые ступеньки", fontsize=12.5)
    axl.legend(loc="center left", frameon=False, fontsize=10)
    axr.plot(x, s1 - s2, color=GREEN, lw=2.6)
    axr.fill_between(x, s1 - s2, color=GREEN, alpha=0.12)
    axr.set_title("их разность — горбик", fontsize=12.5)
    for ax in (axl, axr):
        ax.set_xlim(-4, 4); ax.set_ylim(-0.1, 1.15)
        ax.axhline(0, color=GRID, lw=0.8); ax.axvline(0, color=GRID, lw=0.8)
        ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
        ax.set_xlabel("$x$")
    fig.suptitle("Из двух ступенек — горбик", y=1.02, fontsize=15)
    fig.tight_layout()
    save(fig, OUT / "bump-from-sigmoids.png")


# ------------------------------- fig 19.2: bike approx at K=2,4,8,16
def fig_bike() -> None:
    xs, ys = bike_curve()
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.6), sharex=True, sharey=True)
    rmses = {}
    for ax, K in zip(axes.ravel(), [2, 4, 8, 16]):
        ax.scatter(xs, ys, s=28, color=FAINT, alpha=0.7, zorder=3, label="данные")
        xd, pred, rmse = approx(xs, ys, K)
        rmses[K] = rmse
        ax.plot(xd, pred, color=BLUE, lw=2.4, zorder=4, label=f"{K} блоков")
        ax.set_title(f"{K} блоков   (ошибка {rmse:.0f} поездок)", fontsize=12.5)
        ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
        ax.set_xlim(0, 23); ax.set_ylim(-30, 600)
    for ax in axes[1]:
        ax.set_xlabel("час дня")
    for ax in axes[:, 0]:
        ax.set_ylabel("средний спрос")
    print("bike approx rmse:", {k: round(v, 1) for k, v in rmses.items()})
    _, _, r24 = approx(xs, ys, 24)
    print("K=24 rmse", round(r24, 2))
    assert rmses[2] > 100 and rmses[16] < 30
    fig.suptitle("Спрос проката по часам, собранный из сигмоидных блоков",
                 y=1.01, fontsize=15)
    fig.tight_layout()
    save(fig, OUT / "bike-approx.png")


# ------------------------------- fig 19.3: width vs depth
def fig_width_depth() -> None:
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.6, 4.4))
    for ax in (axl, axr):
        ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    # wide shallow: one layer, many nodes
    axl.add_patch(Circle((1.2, 4), 0.28, facecolor=PAPER, edgecolor=BLUE, lw=1.4))
    ny = 11
    for i in range(ny):
        y = 0.7 + i * 0.62
        axl.add_patch(Circle((5, y), 0.2, facecolor=WASH, edgecolor=INK, lw=0.9))
        axl.plot([1.48, 4.8], [4, y], color=LINE, lw=0.3, alpha=0.5)
        axl.plot([5.2, 8.5], [y, 4], color=LINE, lw=0.3, alpha=0.5)
    axl.add_patch(Circle((8.8, 4), 0.28, facecolor=PAPER, edgecolor=GREEN, lw=1.4))
    axl.set_title("широко и мелко: много нейронов", fontsize=12.5)
    axl.text(5, 7.5, str(ny) + " нейронов в 1 слое", ha="center", fontsize=10, color=MUTED)
    # deep narrow: several layers, few nodes
    axr.add_patch(Circle((1.0, 4), 0.28, facecolor=PAPER, edgecolor=BLUE, lw=1.4))
    layers = [(3.0, 3), (5.0, 3), (7.0, 3)]
    prev = [(1.0, 4)]
    for lx, n in layers:
        cur = []
        for j in range(n):
            y = 2.5 + j * 1.4
            axr.add_patch(Circle((lx, y), 0.24, facecolor=WASH, edgecolor=INK, lw=0.9))
            cur.append((lx, y))
        for p in prev:
            for c in cur:
                axr.plot([p[0], c[0]], [p[1], c[1]], color=LINE, lw=0.3, alpha=0.5)
        prev = cur
    axr.add_patch(Circle((9.0, 4), 0.28, facecolor=PAPER, edgecolor=GREEN, lw=1.4))
    for p in prev:
        axr.plot([p[0], 9.0], [p[1], 4], color=LINE, lw=0.3, alpha=0.5)
    axr.set_title("узко и глубоко: мало нейронов", fontsize=12.5)
    axr.text(5, 7.5, "9 нейронов в 3 слоях", ha="center", fontsize=10, color=MUTED)
    fig.suptitle("Одна функция: широко и мелко против узко и глубоко",
                 y=1.02, fontsize=15)
    fig.tight_layout()
    save(fig, OUT / "width-vs-depth.png")


# ------------------------------------------------ margins
def side_exists() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 9); ax.axis("off")
    from matplotlib.patches import Circle as Circ
    ax.add_patch(Circ((3.8, 5.5), 2.4, facecolor=GREEN, alpha=0.18, edgecolor=GREEN, lw=1.5))
    ax.add_patch(Circ((6.0, 5.5), 2.4, facecolor="none", edgecolor=MUTED, lw=1.3, ls=(0, (4, 3))))
    ax.add_patch(Circ((4.9, 3.4), 2.4, facecolor="none", edgecolor=MUTED, lw=1.3, ls=(0, (4, 3))))
    ax.text(2.8, 6.6, "существует", fontsize=10.5, color=GREEN, ha="center")
    ax.text(7.3, 6.6, "обучается", fontsize=10, color=MUTED, ha="center")
    ax.text(4.9, 1.6, "обобщает", fontsize=10, color=MUTED, ha="center")
    ax.text(5, 0.4, "теорема закрашивает только\nлевый круг", ha="center",
            fontsize=9, color=MUTED)
    save(fig, SIDE / "exists-vs-learn.png")


def side_relu() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    x = np.linspace(0, 6, 300)
    target = np.sin(x) + 0.5 * x
    knots = [0.5, 1.5, 2.7, 4.0, 5.2]
    # piecewise-linear interp through target at knots + endpoints
    pts_x = np.array([0] + knots + [6])
    pts_y = np.interp(pts_x, x, target)
    ax.plot(x, target, color=LINE, lw=2.5, label="цель")
    ax.plot(pts_x, pts_y, color=GREEN, lw=2.0, marker="o", markersize=4, label="сумма ReLU")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("ломаная из изломов", fontsize=10)
    save(fig, SIDE / "relu-pieces.png")


def side_bernstein() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    from math import comb
    x = np.linspace(0, 1, 300)
    n = 6
    for k in range(n + 1):
        b = comb(n, k) * x ** k * (1 - x) ** (n - k)
        ax.plot(x, b, color=BLUE, lw=1.2, alpha=0.7)
    ax.set_title("базис Бернштейна ($n=6$)", fontsize=10)
    ax.set_xticks([0, 1]); ax.set_yticks([])
    ax.set_xlabel("$x$", fontsize=9)
    save(fig, SIDE / "bernstein.png")


fig_bump()
fig_bike()
fig_width_depth()
side_exists()
side_relu()
side_bernstein()
print("lesson 19 figures written")
