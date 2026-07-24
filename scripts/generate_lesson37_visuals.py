"""Deterministic figures for lesson 37: matrix multiplication and its cost.

The cubic law measured for real (wall-clock of numpy.matmul on this machine, not a
fake coefficient), the matrix-chain order that changes cost by an order of magnitude
without changing the answer, and the naive-vs-Strassen operation counts with the
crossover where Strassen's extra additions stop mattering. Numbers asserted.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "37"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "37"

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


# ------------------------------------------- fig 37.1: measured cubic law (real timing)
def fig_empirical() -> None:
    rng = np.random.default_rng(37)
    ns = [64, 96, 128, 192, 256, 384, 512, 768, 1024]
    times = []
    for n in ns:
        A = rng.standard_normal((n, n)); B = rng.standard_normal((n, n))
        A @ B  # warm up
        best = min(_time_once(A, B) for _ in range(5))
        times.append(best)
    times = np.array(times); ns_a = np.array(ns, float)
    # fit slope in log-log
    slope, intercept = np.polyfit(np.log(ns_a), np.log(times), 1)
    print(f"measured matmul slope = {slope:.2f} (ideal 3.0)")
    assert 2.2 <= slope <= 3.4, slope
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.loglog(ns_a, times * 1e3, "o", color=BLUE, markersize=7, label="измеренное время numpy.matmul")
    ref = times[0] * (ns_a / ns_a[0]) ** 3
    ax.loglog(ns_a, ref * 1e3, color=RED, lw=1.8, ls=(0, (5, 3)), label="закон $n^3$ (наклон 3)")
    ax.set_xlabel("размер матрицы n"); ax.set_ylabel("время, мс")
    ax.set_title(f"Кубический закон, измеренный на самом деле: наклон {slope:.2f}")
    ax.grid(True, which="both", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.annotate("библиотека обгоняет чистый $n^3$:\nпараллелизм скрадывает часть роста",
                xy=(ns_a[-2], times[-2] * 1e3), xytext=(90, 0.9), fontsize=10, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=LINE, lw=1.0))
    save(fig, OUT / "empirical.png")


def _time_once(A, B):
    t0 = time.perf_counter(); A @ B; return time.perf_counter() - t0


# ------------------------------------------- fig 37.2: matrix chain order changes cost
def fig_chain() -> None:
    # dims p0..p3: A(p0xp1) B(p1xp2) C(p2xp3)
    p = [40, 100, 5, 60]
    ab_c = p[0] * p[1] * p[2] + p[0] * p[2] * p[3]      # (AB)C
    a_bc = p[1] * p[2] * p[3] + p[0] * p[1] * p[3]      # A(BC)
    print(f"chain: (AB)C = {ab_c}, A(BC) = {a_bc}")
    assert ab_c == 32000 and a_bc == 270000
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.6, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    # left: the chain shapes
    ax0.set_aspect("equal"); ax0.axis("off"); ax0.set_xlim(0, 13); ax0.set_ylim(-1, 6)
    shapes = [("A", 0.5, p[0], p[1], BLUE), ("B", 5.2, p[1], p[2], GREEN), ("C", 8.4, p[2], p[3], GOLD)]
    for name, x, rows, cols, col in shapes:
        w = cols / 22.0; h = rows / 22.0
        ax0.add_patch(Rectangle((x, 2.5 - h / 2), w, h, fc=col, ec=col, alpha=0.25, lw=1.5))
        ax0.text(x + w / 2, 2.5 - h / 2 - 0.35, f"{name}\n{rows}×{cols}", ha="center", va="top", fontsize=10, color=col)
    ax0.text(6.5, 5.2, "цепочка A·B·C: ответ один,\nа порядок умножений — на выбор", ha="center", fontsize=11, color=INK)
    ax0.set_title("Три матрицы в цепочке")
    # right: cost bars
    ax1.bar(["(A·B)·C", "A·(B·C)"], [ab_c, a_bc], color=[GREEN, RED], width=0.55)
    for i, v in enumerate([ab_c, a_bc]):
        ax1.text(i, v + 6000, f"{v:,}".replace(",", " "), ha="center", fontsize=12, color=INK)
    ax1.set_ylabel("скалярных умножений")
    ax1.set_ylim(0, a_bc * 1.18)
    ax1.set_title(f"Порядок меняет цену в {a_bc // ab_c} раз")
    ax1.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax1.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "chain.png")


# ------------------------------------------- fig 37.3: naive vs Strassen op counts + crossover
def _naive_ops(n):
    return 2 * n ** 3 - n ** 2


def _strassen_ops(n, cutoff=64):
    if n <= cutoff:
        return _naive_ops(n)
    h = n // 2
    return 7 * _strassen_ops(h, cutoff) + 18 * h * h


def fig_strassen() -> None:
    ks = list(range(6, 13))  # n = 64 .. 4096
    ns = [2 ** k for k in ks]
    naive = [_naive_ops(n) for n in ns]
    stras = [_strassen_ops(n) for n in ns]
    # crossover: smallest n where strassen < naive
    cross = next((n for n in ns if _strassen_ops(n) < _naive_ops(n)), None)
    print(f"strassen crossover at n = {cross}")
    assert cross is not None and cross <= 1024
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.loglog(ns, naive, "o-", color=BLUE, lw=2.0, markersize=6, label="обычный, $\\sim n^3$")
    ax.loglog(ns, stras, "s-", color=RED, lw=2.0, markersize=6, label="Штрассен, $\\sim n^{2{,}81}$")
    ax.axvline(cross, color=GOLD, lw=1.4, ls=(0, (4, 3)))
    ax.annotate(f"перелом при n≈{cross}:\nвыше Штрассен дешевле", xy=(cross, _naive_ops(cross)),
                xytext=(cross * 1.15, _naive_ops(cross) * 0.12), fontsize=10, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=LINE, lw=1.0))
    ax.set_xlabel("размер матрицы n"); ax.set_ylabel("число операций (умножения + сложения)")
    ax.set_title("Семь произведений вместо восьми — но не бесплатно")
    ax.grid(True, which="both", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    save(fig, OUT / "strassen.png")


# ------------------------------------------- margin: convolution as GEMM (im2col)
def side_im2col() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(4.6, 2.0))
    rng = np.random.default_rng(3)
    img = rng.integers(0, 2, (4, 4))
    axes[0].imshow(img, cmap="Blues", vmin=-0.5, vmax=1.5); axes[0].set_title("окна 3×3", fontsize=8.5); axes[0].axis("off")
    # im2col matrix: 4 windows (2x2 output) x 9
    cols = np.array([img[i:i + 3, j:j + 3].ravel() for i in range(2) for j in range(2)])
    axes[1].imshow(cols, cmap="Blues", vmin=-0.5, vmax=1.5, aspect="auto"); axes[1].set_title("im2col 4×9", fontsize=8.5); axes[1].axis("off")
    axes[2].axis("off")
    axes[2].text(0.5, 0.5, "GEMM:\nодно матричное\nумножение", ha="center", va="center", fontsize=9, color=INK)
    fig.suptitle("свёртка становится умножением матриц", y=1.08, fontsize=9)
    fig.tight_layout()
    save(fig, SIDE / "im2col.png")


# ------------------------------------------- margin: blocking / cache tiles
def side_blocking() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    # C = A x B with a highlighted tile
    for (x, lab, col) in [(0.5, "A", BLUE), (4.0, "B", GREEN), (7.5, "C", GOLD)]:
        ax.add_patch(Rectangle((x, 1.5), 2, 3, fc=WASH, ec=col, lw=1.4))
        for t in range(1, 3):
            ax.plot([x, x + 2], [1.5 + t, 1.5 + t], color=LINE, lw=0.6)
            ax.plot([x + t * 2 / 3, x + t * 2 / 3], [1.5, 4.5], color=LINE, lw=0.6)
        ax.text(x + 1, 1.1, lab, ha="center", fontsize=10, color=col)
    ax.add_patch(Rectangle((0.5, 3.5), 2 / 3, 1, fc=BLUE, alpha=0.35, ec=BLUE))
    ax.add_patch(Rectangle((4.0, 2.5), 2 / 3, 1, fc=GREEN, alpha=0.35, ec=GREEN))
    ax.add_patch(Rectangle((7.5, 3.5), 2 / 3, 1, fc=GOLD, alpha=0.35, ec=GOLD))
    ax.text(5, 5.3, "плитка помещается в быстрый кэш\nи переиспользуется", ha="center", fontsize=8.5, color=INK)
    ax.set_title("блочное умножение", fontsize=9.5)
    save(fig, SIDE / "blocking.png")


# ------------------------------------------- margin: loop order access pattern
def side_loops() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(4.2, 2.2))
    for ax, (title, good) in zip(axes, [("порядок ijk", False), ("порядок ikj", True)]):
        ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(0, 5); ax.set_ylim(0, 5)
        ax.add_patch(Rectangle((0.5, 0.5), 4, 4, fc=WASH, ec=LINE))
        col = GREEN if good else RED
        if good:
            for r in range(4):
                ax.add_patch(Rectangle((0.5, 0.5 + r), 4, 1, fc=col, alpha=0.12, ec="none"))
            ax.annotate("", xy=(4.3, 1), xytext=(0.7, 1), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6))
        else:
            for c in range(4):
                ax.add_patch(Rectangle((0.5 + c, 0.5), 1, 4, fc=col, alpha=0.12, ec="none"))
            ax.annotate("", xy=(1, 4.3), xytext=(1, 0.7), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6))
        ax.set_title(title + ("\nпо строкам: дружит с кэшем" if good else "\nпо столбцам: прыжки"), fontsize=8)
    fig.tight_layout()
    save(fig, SIDE / "loops.png")


fig_empirical()
fig_chain()
fig_strassen()
side_im2col()
side_blocking()
side_loops()
print("lesson 37 figures written")
