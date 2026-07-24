"""Deterministic figures for lesson 46: confidence intervals and Fisher information.

The frequentist meaning of coverage (100 repeated samples, about 95 intervals catch the
fixed truth), the Wald interval shown UNCLIPPED so its failure past 0 and 1 is visible next
to the valid Wilson interval, and Fisher information as the curvature of the log-likelihood
that sets the standard error. Numbers asserted.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "public" / "figures" / "lessons" / "46"
SIDE = Path(__file__).resolve().parents[1] / "public" / "figures" / "sidenotes" / "46"

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


# ---------------------------------------- fig 46.1: coverage in repeated sampling
def fig_coverage() -> None:
    p, n, reps = 0.6, 400, 100
    rng = np.random.default_rng(46)
    z = 1.96
    covered = 0
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    for i in range(reps):
        h = rng.binomial(n, p)
        ph = h / n
        se = np.sqrt(ph * (1 - ph) / n)
        lo, hi = ph - z * se, ph + z * se
        hit = lo <= p <= hi
        covered += hit
        c = BLUE if hit else RED
        ax.plot([lo, hi], [i, i], color=c, lw=1.4, alpha=0.85)
        ax.plot(ph, i, ".", color=c, markersize=3)
    ax.axvline(p, color=INK, lw=1.4)
    ax.text(p + 0.004, reps + 1, f"истинная доля $p={p}$", fontsize=10, color=INK)
    print(f"coverage: {covered}/100 intervals cover the truth")
    assert 90 <= covered <= 99
    ax.set_xlabel("доля $\\widehat p$ и её 95% интервал"); ax.set_ylabel("номер опроса")
    ax.set_title(f"100 повторных опросов: {covered} интервалов из 100 накрыли истину")
    ax.set_ylim(-1, reps + 4); ax.set_xlim(0.5, 0.7)
    ax.grid(True, axis="x", color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "coverage.png")


# ---------------------------------------- fig 46.2: Wald (unclipped) vs Wilson at the boundary
def fig_wald_wilson() -> None:
    n = 10
    z = 1.96
    hs = np.arange(0, 11)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for h in hs:
        ph = h / n
        se = np.sqrt(ph * (1 - ph) / n)
        wlo, whi = ph - z * se, ph + z * se           # UNCLIPPED: may exit [0,1]
        # Wilson
        c = (ph + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        wilo, wihi = c - half, c + half
        ax.plot([h - 0.16, h - 0.16], [wlo, whi], color=RED, lw=3, solid_capstyle="round")
        ax.plot([h + 0.16, h + 0.16], [wilo, wihi], color=BLUE, lw=3, solid_capstyle="round")
        ax.plot(h - 0.16, ph, ".", color=INK, markersize=4)
    ax.axhspan(-0.25, 0, color=RED, alpha=0.06)
    ax.axhspan(1, 1.25, color=RED, alpha=0.06)
    ax.axhline(0, color=MUTED, lw=0.8); ax.axhline(1, color=MUTED, lw=0.8)
    ax.text(0.2, -0.18, "интервал Вальда уходит ниже 0 и выше 1 — недопустимо", fontsize=10, color=RED)
    ax.plot([], [], color=RED, lw=3, label="Вальд (подстановка), не обрезан")
    ax.plot([], [], color=BLUE, lw=3, label="Уилсон (score), всегда в $[0,1]$")
    ax.set_xlabel("число успехов $h$ из $n=10$"); ax.set_ylabel("оценка доли $p$")
    ax.set_title("У границы интервал Вальда ломается, интервал Уилсона держится")
    ax.set_ylim(-0.25, 1.25); ax.set_xticks(hs)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    print("wald_wilson drawn (Wald unclipped)")
    save(fig, OUT / "wald_wilson.png")


# ---------------------------------------- fig 46.3: Fisher information = curvature -> se
def fig_fisher() -> None:
    n, h = 40, 24
    phat = h / n
    p = np.linspace(0.3, 0.9, 400)
    ll = h * np.log(p) + (n - h) * np.log(1 - p)
    ll = ll - ll.max()
    I = 1 / (phat * (1 - phat))             # Fisher info per observation
    se = 1 / np.sqrt(n * I)                  # = sqrt(phat(1-phat)/n)
    print(f"fisher: I(p)={I:.2f}, n*I={n*I:.1f}, se={se:.3f}")
    assert abs(se - np.sqrt(phat * (1 - phat) / n)) < 1e-9
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(p, ll, color=GREEN, lw=2.4, label="лог-правдоподобие $\\ell(p)$")
    # quadratic with curvature nI
    ax.plot(p, -0.5 * n * I * (p - phat) ** 2, color=RED, lw=1.6, ls=(0, (5, 3)),
            label="кривизна $= n\\,I(p)$")
    ax.axvline(phat, color=INK, lw=1.0, ls=(0, (2, 2)))
    ax.axvspan(phat - 1.96 * se, phat + 1.96 * se, color=GOLD, alpha=0.14)
    ax.annotate("ширина интервала\n$\\pm1{,}96/\\sqrt{n I}$", xy=(phat + 1.96 * se, -0.5),
                xytext=(phat + 0.09, -2.0), fontsize=10, color=GOLD,
                arrowprops=dict(arrowstyle="->", color=LINE, lw=1.0))
    ax.set_ylim(-6, 0.5); ax.set_xlabel("параметр $p$"); ax.set_ylabel("$\\ell(p)-\\ell(\\widehat p)$")
    ax.set_title("Информация Фишера — кривизна вершины, задающая точность")
    ax.legend(loc="lower center", frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "fisher.png")


# ---------------------------------------- margins
def side_width() -> None:
    ns = np.arange(10, 2001)
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    for z, lab, c in [(1.645, "90%", GREEN), (1.96, "95%", BLUE), (2.576, "99%", RED)]:
        ax.plot(ns, 2 * z * 0.5 / np.sqrt(ns), color=c, lw=1.8, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("объём n", fontsize=9); ax.set_ylabel("ширина интервала", fontsize=9)
    ax.legend(loc="upper right", frameon=False, fontsize=8, title="уровень")
    ax.set_title("уже с ростом n, шире с уровнем", fontsize=9)
    ax.grid(True, which="both", color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "width.png")


def side_bootstrap() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    rng = np.random.default_rng(1)
    base = np.arange(6)
    ax.text(5, 5.5, "bootstrap: пересборка с возвращением", ha="center", fontsize=8.5, color=INK)
    from matplotlib.patches import Rectangle
    for i, v in enumerate(base):
        ax.add_patch(Rectangle((0.6 + i * 1.0, 4), 0.8, 0.8, fc=BLUE, ec=PAPER, alpha=0.7))
    ax.text(0.3, 4.4, "выборка", fontsize=8, color=MUTED, ha="right")
    for r, y in [(0, 2.5), (1, 1.3)]:
        idx = rng.integers(0, 6, 6)
        for i, v in enumerate(idx):
            ax.add_patch(Rectangle((0.6 + i * 1.0, y), 0.8, 0.8, fc=GOLD, ec=PAPER, alpha=0.55))
        ax.text(0.3, y + 0.4, f"копия {r+1}", fontsize=7.5, color=MUTED, ha="right")
    ax.set_title("каждая копия — новая оценка", fontsize=9.5)
    save(fig, SIDE / "bootstrap.png")


def side_prediction() -> None:
    x = np.linspace(0, 10, 100)
    rng = np.random.default_rng(3)
    y = 1 + 0.5 * x
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    ax.plot(x, y, color=INK, lw=1.6)
    ax.fill_between(x, y - 0.4, y + 0.4, color=BLUE, alpha=0.22, label="для среднего")
    ax.fill_between(x, y - 1.4, y + 1.4, color=GOLD, alpha=0.14, label="для нового объекта")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.set_title("два разных интервала", fontsize=9.5)
    save(fig, SIDE / "prediction.png")


fig_coverage()
fig_wald_wilson()
fig_fisher()
side_width()
side_bootstrap()
side_prediction()
print("lesson 46 figures written")
