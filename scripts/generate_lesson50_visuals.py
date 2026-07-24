"""Deterministic figures for lesson 50: polynomial and local features.

A straight line fails on the REAL bimodal bike-rides-by-hour pattern while a basis of
features captures both peaks, high-degree polynomials interpolate the training points but
oscillate and explode outside them, and three basis families (polynomial, radial bells,
cyclic sines) each encode a different assumption. Numbers asserted.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BIKE = ROOT / "scripts" / "data" / "bike-sharing-hour.csv"
OUT = ROOT / "public" / "figures" / "lessons" / "50"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "50"

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


def bike_hourly():
    hour, cnt = [], []
    with open(BIKE) as f:
        for row in csv.DictReader(f):
            hour.append(int(row["hr"])); cnt.append(int(row["cnt"]))
    hour, cnt = np.array(hour), np.array(cnt)
    return np.array([cnt[hour == h].mean() for h in range(24)])


# ---------------------------------------- fig 50.1: line fails, basis captures bimodal (real)
def fig_basis_real() -> None:
    avg = bike_hourly()
    x = np.arange(24)
    c1 = np.polyfit(x, avg, 1); line = np.polyval(c1, x)
    r2_line = 1 - np.sum((avg - line) ** 2) / np.sum((avg - avg.mean()) ** 2)
    xc = x - x.mean()
    c8 = np.polyfit(xc, avg, 8); poly = np.polyval(c8, xc)
    r2_poly = 1 - np.sum((avg - poly) ** 2) / np.sum((avg - avg.mean()) ** 2)
    print(f"basis_real: line R2={r2_line:.2f}, poly8 R2={r2_poly:.2f}")
    assert 0.28 < r2_line < 0.35 and r2_poly > 0.85
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.scatter(x, avg, s=45, color=INK, zorder=5, label="средние поездки по часам")
    ax.plot(x, line, color=BLUE, lw=2.0, ls=(0, (5, 3)), label=f"прямая ($R^2={r2_line:.2f}$)")
    xs = np.linspace(0, 23, 200)
    ax.plot(xs, np.polyval(c8, xs - x.mean()), color=RED, lw=2.4, label=f"базис степени 8 ($R^2={r2_poly:.2f}$)")
    ax.set_xlabel("час суток"); ax.set_ylabel("среднее число поездок")
    ax.set_title("Прямая не ловит два пика, базис признаков — ловит (реальные данные)")
    ax.set_xticks(range(0, 24, 3))
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "basis_real.png")


# ---------------------------------------- fig 50.2: interpolation vs extrapolation (Runge)
def fig_overfit() -> None:
    rng = np.random.default_rng(50)
    xt = np.linspace(-1, 1, 11)
    yt = 1 / (1 + 25 * xt ** 2) + rng.normal(0, 0.03, len(xt))   # Runge function
    xs = np.linspace(-1.35, 1.35, 400)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.axvspan(-1, 1, color=WASH, alpha=0.6)
    ax.scatter(xt, yt, s=45, color=INK, zorder=6, label="обучающие точки")
    for deg, c, lw in [(1, BLUE, 1.8), (3, GREEN, 1.8), (10, RED, 2.2)]:
        co = np.polyfit(xt, yt, deg)
        ax.plot(xs, np.polyval(co, xs), color=c, lw=lw, label=f"степень {deg}")
    ax.axvline(-1, color=MUTED, lw=0.8, ls=(0, (2, 2))); ax.axvline(1, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    ax.set_ylim(-0.6, 1.5)
    ax.text(0, -0.5, "область обучения", ha="center", fontsize=9.5, color=MUTED)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title("Высокая степень проходит через точки, но колеблется и взлетает вне них")
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "overfit.png")
    print("overfit drawn")


# ---------------------------------------- fig 50.3: three basis families
def fig_bases() -> None:
    x = np.linspace(0, 1, 300)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))
    # polynomial
    for k, c in zip(range(4), [BLUE, GREEN, GOLD, RED]):
        axes[0].plot(x, x ** k, color=c, lw=1.8)
    axes[0].set_title("полиномы: глобальная связь", fontsize=11)
    # radial bells
    for c0, c in zip([0.15, 0.4, 0.65, 0.9], [BLUE, GREEN, GOLD, RED]):
        axes[1].plot(x, np.exp(-((x - c0) ** 2) / (2 * 0.09 ** 2)), color=c, lw=1.8)
    axes[1].set_title("радиальные колокола: локальность", fontsize=11)
    # cyclic
    for k, c in zip([1, 2], [BLUE, RED]):
        axes[2].plot(x, np.sin(2 * np.pi * k * x), color=c, lw=1.8, label=f"sin {k}")
        axes[2].plot(x, np.cos(2 * np.pi * k * x), color=c, lw=1.4, ls=(0, (4, 3)))
    axes[2].set_title("синусы: периодичность", fontsize=11)
    for ax in axes:
        ax.set_xlabel("$x$", fontsize=9); ax.set_yticks([])
        ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    fig.suptitle("Каждый базис кодирует своё предположение о форме", y=1.03, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "bases.png")
    print("bases drawn")


# ---------------------------------------- margins
def side_clock() -> None:
    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    ax.set_aspect("equal"); ax.axis("off")
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color=LINE, lw=1.2)
    for h in range(24):
        a = np.pi / 2 - 2 * np.pi * h / 24
        ax.plot(np.cos(a), np.sin(a), "o", color=BLUE, markersize=4)
    for h, col in [(23, RED), (0, RED), (1, RED), (13, GOLD)]:
        a = np.pi / 2 - 2 * np.pi * h / 24
        ax.plot(np.cos(a), np.sin(a), "o", color=col, markersize=8)
        ax.annotate(str(h), (np.cos(a) * 1.2, np.sin(a) * 1.2), ha="center", va="center", fontsize=9, color=col)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_title("час на окружности: 23 и 0 рядом", fontsize=9.5)
    save(fig, SIDE / "clock.png")


def side_scale() -> None:
    x = np.linspace(0, 3, 100)
    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    for k, c in [(2, BLUE), (5, GOLD), (8, RED)]:
        ax.plot(x, x ** k, color=c, lw=1.8, label=f"$x^{{{k}}}$")
    ax.set_yscale("log"); ax.set_xlabel("x", fontsize=9); ax.set_ylabel("значение (лог)", fontsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.set_title("степени взрываются — надо центрировать", fontsize=9)
    save(fig, SIDE / "scale.png")
    print("scale drawn")


def side_interaction() -> None:
    a = np.linspace(0, 50, 50)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(4.4, 2.2), sharey=True)
    for p, c in [(10, BLUE), (20, GREEN), (30, RED)]:
        a0.plot(a, 100 - 4 * p + 2 * a, color=c, lw=1.6)
        a1.plot(a, 100 - 4 * p + 2 * a + 0.1 * p * a, color=c, lw=1.6)
    a0.set_title("без\nвзаимодействия", fontsize=8.5); a1.set_title("с\nвзаимодействием", fontsize=8.5)
    for ax in (a0, a1):
        ax.set_xlabel("реклама", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("параллельно или расходятся", y=1.05, fontsize=9)
    fig.tight_layout()
    save(fig, SIDE / "interaction.png")


fig_basis_real()
fig_overfit()
fig_bases()
side_clock()
side_scale()
side_interaction()
print("lesson 50 figures written")
