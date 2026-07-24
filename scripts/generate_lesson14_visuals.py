"""Deterministic figures for lesson 14: cybernetics, homeostat, Hebb/Oja.

Feedback-loop block diagram, P-controller responses for four gains,
Ashby homeostat schematic, and the Oja neuron finding the principal axis
of real Gapminder-2007 data (log GDP vs life expectancy).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "14"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "14"

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


def box(ax, xy, w, h, label, *, edge=INK, fs=13, face=WASH):
    ax.add_patch(FancyBboxPatch(xy, w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=face, edgecolor=edge, linewidth=1.6))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, label, ha="center", va="center",
            fontsize=fs, color=INK)


# ------------------------------------ P-controller simulation
def simulate(K, alpha=1.0, beta=0.0, r=22.0, x0=10.0, x_out=0.0, n=30):
    xs = [x0]
    x = x0
    for _ in range(n):
        u = K * (r - x)
        x = x + alpha * u - beta * (x - x_out)
        xs.append(x)
    return xs


# ------------------------------------------- fig 14.1: feedback loop
def fig_loop() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    # summing junction
    ax.add_patch(Circle((2.2, 3.4), 0.42, facecolor=PAPER, edgecolor=INK,
                        linewidth=1.6, zorder=4))
    ax.text(2.2, 3.4, "−", ha="center", va="center", fontsize=20, color=INK)
    ax.text(0.5, 3.4, "цель $r$", ha="center", fontsize=12, color=GREEN)
    arrow(ax, (1.15, 3.4), (1.78, 3.4), color=GREEN)
    box(ax, (4.0, 2.8), 2.3, 1.2, "регулятор\n$u=Ke$", edge=BLUE)
    arrow(ax, (2.62, 3.4), (3.95, 3.4), color=MUTED)
    ax.text(3.3, 3.7, "$e$", fontsize=13, color=INK, ha="center")
    box(ax, (8.0, 2.8), 2.3, 1.2, "объект\n(комната)", edge=GOLD)
    arrow(ax, (6.3, 3.4), (7.95, 3.4), color=MUTED)
    ax.text(7.1, 3.7, "$u$", fontsize=13, color=INK, ha="center")
    arrow(ax, (10.3, 3.4), (11.4, 3.4), color=INK)
    ax.text(11.0, 3.7, "$x$", fontsize=13, color=INK, ha="center")
    # feedback line
    ax.plot([10.9, 10.9, 2.2, 2.2], [3.4, 1.2, 1.2, 2.98], color=MUTED,
            lw=1.5)
    ax.add_patch(FancyArrowPatch((2.2, 1.6), (2.2, 2.96), arrowstyle="-|>",
                                 color=MUTED, lw=1.5, mutation_scale=13))
    ax.text(6.4, 0.85, "обратная связь: измерение возвращается ко входу",
            ha="center", fontsize=11, color=MUTED)
    ax.set_title("Контур обратной связи", fontsize=15, pad=4)
    save(fig, OUT / "feedback-loop.png")


# ------------------------------------- fig 14.2: four gain responses
def fig_responses() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    specs = [
        (0.3, BLUE, "$K=0{,}3$: вяло", 20),
        (0.7, GREEN, "$K=0{,}7$: гладко", 20),
        (1.5, GOLD, "$K=1{,}5$: колебания", 20),
        (2.05, RED, "$K=2{,}05$: расходимость", 13),
    ]
    for K, col, lab, n in specs:
        xs = simulate(K, n=n)
        ax.plot(range(len(xs)), xs, color=col, lw=2.0, marker="o",
                markersize=3.5, label=lab)
    ax.axhline(22, color=MUTED, lw=1.2, ls=(0, (5, 4)))
    ax.text(20.3, 22, "цель", fontsize=11, color=MUTED, va="center")
    ax.set_xlim(0, 20)
    ax.set_ylim(2, 43)
    ax.set_xlabel("шаг времени")
    ax.set_ylabel("температура комнаты")
    ax.set_title("Один контур, три судьбы: отклик при разном усилении")
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    save(fig, OUT / "control-responses.png")


# ------------------------------------------- fig 14.3: homeostat
def fig_homeostat() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    pos = {"A": (2.3, 5.6), "B": (7.0, 5.6), "C": (2.3, 2.4), "D": (7.0, 2.4)}
    vals = {"A": 0.35, "B": -0.2, "C": 0.1, "D": -0.45}
    for name, (x, y) in pos.items():
        # allowed zone strip
        ax.add_patch(Rectangle((x - 0.95, y - 0.55), 1.9, 1.1, facecolor=WASH,
                               edgecolor=LINE, linewidth=1.0, zorder=1))
        ax.plot([x - 0.95, x + 0.95], [y, y], color=GRID, lw=0.8, zorder=2)
        # pointer
        v = vals[name]
        ax.add_patch(FancyArrowPatch((x, y), (x + v * 0.8, y + 0.42),
                                     arrowstyle="-|>", color=BLUE, lw=2.0,
                                     mutation_scale=12, zorder=5))
        ax.add_patch(Circle((x, y), 0.09, facecolor=INK, zorder=6))
        ly = y + 0.72 if y > 4 else y - 0.9
        ax.text(x, ly, name, ha="center", fontsize=13, color=INK,
                weight="bold", zorder=7)
    # cross connections
    pairs = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]
    for a, b in pairs:
        (x0, y0), (x1, y1) = pos[a], pos[b]
        ax.annotate("", (x1, y1), (x0, y0),
                    arrowprops=dict(arrowstyle="<|-|>", color=MUTED, lw=1.0,
                                    shrinkA=26, shrinkB=26, alpha=0.55))
    # search switch box
    box(ax, (3.9, 0.15), 2.2, 0.95, "случайный\nпоиск связей", edge=RED,
        fs=10.5, face=PAPER)
    ax.text(5.0, 7.35, "серая полоса — допустимая зона стрелки",
            ha="center", fontsize=10.5, color=MUTED)
    ax.set_title("Гомеостат Эшби: четыре блока и поиск устойчивости",
                 fontsize=14.5, pad=2)
    save(fig, OUT / "homeostat.png")


# ------------------------------------- fig 14.4: Oja on Gapminder
def fig_oja() -> None:
    rows = [r for r in csv.DictReader((ROOT / "scripts" / "data" / "gapminder.csv").open())
            if r["year"] == "2007"]
    x = np.array([math.log10(float(r["gdpPercap"])) for r in rows])
    y = np.array([float(r["lifeExp"]) for r in rows])
    X = np.stack([(x - x.mean()) / x.std(), (y - y.mean()) / y.std()], 1)
    corr = float(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
    rng = np.random.RandomState(0)
    w = np.array([1.0, -0.3])
    w = w / np.linalg.norm(w)
    eta = 0.001
    snap = [w.copy()]
    angs = [math.degrees(math.atan2(w[1], w[0])) % 180]
    for ep in range(60):
        for xi in X[rng.permutation(len(X))]:
            yy = w @ xi
            w = w + eta * yy * (xi - yy * w)
        snap.append(w.copy())
        angs.append(math.degrees(math.atan2(w[1], w[0])) % 180)
    print(f"Oja final angle {angs[-1]:.1f} deg, corr {corr:.3f}")
    assert abs(angs[-1] - 45.0) < 1.0 and abs(corr - 0.809) < 5e-3

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.0, 4.8),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    axl.scatter(X[:, 0], X[:, 1], s=16, color=FAINT, alpha=0.6, linewidth=0,
                zorder=2)
    # rotating weight vectors (subset)
    for i, wv in enumerate(snap[:14]):
        a = 0.25 + 0.75 * i / 13
        axl.add_patch(FancyArrowPatch((0, 0), (wv[0] * 2.6, wv[1] * 2.6),
                                      arrowstyle="-|>", color=MUTED,
                                      lw=1.0, alpha=0.35 + 0.03 * i,
                                      mutation_scale=10, zorder=3))
    # final axis
    wf = snap[-1]
    axl.plot([-wf[0] * 3, wf[0] * 3], [-wf[1] * 3, wf[1] * 3], color=BLUE,
             lw=2.4, zorder=4, label="главная ось, $45°$")
    axl.set_xlim(-3, 3)
    axl.set_ylim(-3, 3)
    axl.set_aspect("equal")
    axl.set_xlabel("$\\log$ ВВП на душу (приведён)")
    axl.set_ylabel("продолж. жизни (приведена)")
    axl.set_title("142 страны и вектор весов Ойи", fontsize=13)
    axl.legend(loc="upper left", frameon=False, fontsize=10.5)
    axl.grid(True, color=GRID, lw=0.6, alpha=0.6)
    axl.set_axisbelow(True)

    axr.plot(range(len(angs)), angs, color=BLUE, lw=2.0)
    axr.axhline(45, color=MUTED, lw=1.2, ls=(0, (5, 4)))
    axr.text(len(angs) * 0.5, 47.5, "$45°$", fontsize=12, color=MUTED)
    axr.set_ylim(0, 140)
    axr.set_xlabel("эпоха обучения")
    axr.set_ylabel("угол вектора весов, °")
    axr.set_title("Сходимость к главной оси", fontsize=13)
    axr.grid(True, color=GRID, lw=0.6, alpha=0.6)
    axr.set_axisbelow(True)
    fig.suptitle("Хеббовский нейрон находит главную ось облака стран",
                 y=1.02, fontsize=15)
    fig.tight_layout()
    save(fig, OUT / "oja-gapminder.png")


# ------------------------------------------------ margins
def side_pupil() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.add_patch(Circle((2.0, 3.5), 1.2, facecolor=WASH, edgecolor=INK, lw=1.5))
    ax.add_patch(Circle((2.0, 3.5), 0.5, facecolor=INK))
    ax.text(2.0, 1.6, "зрачок", ha="center", fontsize=10, color=INK)
    box(ax, (5.6, 2.9), 2.6, 1.2, "меньше света\nна сетчатку", edge=GOLD, fs=10)
    arrow(ax, (3.3, 4.2), (5.55, 3.9), color=GOLD)
    ax.text(4.4, 4.9, "ярче, уже зрачок", fontsize=9.5, color=GOLD, ha="center")
    ax.plot([6.9, 6.9, 2.0, 2.0], [2.85, 1.0, 1.0, 2.25], color=MUTED, lw=1.3)
    ax.add_patch(FancyArrowPatch((2.0, 1.4), (2.0, 2.23), arrowstyle="-|>",
                                 color=MUTED, lw=1.3, mutation_scale=11))
    ax.text(4.5, 0.55, "реакция против отклонения", ha="center", fontsize=9.5,
            color=MUTED)
    save(fig, SIDE / "pupil.png")


def side_howl() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    box(ax, (0.6, 2.9), 2.2, 1.4, "микрофон", edge=BLUE, fs=10.5)
    box(ax, (7.0, 2.9), 2.4, 1.4, "динамик", edge=RED, fs=10.5)
    arrow(ax, (2.9, 4.0), (6.95, 4.0), color=INK)
    ax.text(5.0, 4.45, "усиление", fontsize=10, color=INK, ha="center")
    ax.add_patch(FancyArrowPatch((7.0, 3.1), (2.85, 3.1), arrowstyle="-|>",
                                 color=RED, lw=1.6, mutation_scale=13,
                                 connectionstyle="arc3,rad=0.35"))
    ax.text(5.0, 1.7, "звук снова в микрофон", fontsize=10, color=RED,
            ha="center")
    # growing wave
    xs = np.linspace(0.6, 9.4, 200)
    ax.plot(xs, 5.9 + 0.5 * np.exp((xs - 0.6) * 0.18) *
            np.sin(xs * 3) / np.exp(8.8 * 0.18) * 0.9, color=RED, lw=1.2)
    ax.text(5.0, 6.6, "вой нарастает", ha="center", fontsize=9.5, color=MUTED)
    save(fig, SIDE / "mic-howl.png")


def side_anokhin() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)
    ax.axis("off")
    box(ax, (0.5, 6.6), 3.0, 1.2, "ожидаемый\nрезультат", edge=GREEN, fs=10)
    box(ax, (6.0, 6.6), 3.2, 1.2, "акцептор\n(сверка)", edge=BLUE, fs=10)
    box(ax, (0.5, 1.0), 3.0, 1.2, "действие", edge=GOLD, fs=10.5)
    box(ax, (6.0, 1.0), 3.2, 1.2, "фактический\nисход", edge=INK, fs=10)
    arrow(ax, (3.55, 7.2), (5.95, 7.2), color=MUTED)
    arrow(ax, (2.0, 6.55), (2.0, 2.25), color=GOLD)
    arrow(ax, (3.55, 1.6), (5.95, 1.6), color=MUTED)
    arrow(ax, (7.6, 2.25), (7.6, 6.55), color=BLUE)
    ax.text(7.9, 4.4, "сверка", fontsize=9.5, color=BLUE, rotation=90,
            va="center")
    ax.text(5.0, 0.2, "коррекция по предсказанию", ha="center", fontsize=9.5,
            color=MUTED)
    save(fig, SIDE / "anokhin.png")


fig_loop()
fig_responses()
fig_homeostat()
fig_oja()
side_pupil()
side_howl()
side_anokhin()
print("lesson 14 figures written")
