"""Deterministic figures for lesson 47: Bayesian updating.

Prior times likelihood gives the posterior, shown as PROPER densities (each of area one,
not renormalised to its own peak). The prior strength decides how far the same data move
the belief. And sequential Bayesian updating on the REAL SMS-spam corpus narrows the
posterior of the spam fraction around 0.134 with a credible interval. Numbers asserted.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parents[1]
SPAM = ROOT / "scripts" / "data" / "sms-spam-collection.tsv"
OUT = ROOT / "public" / "figures" / "lessons" / "47"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "47"

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


def spam_counts():
    spam = total = 0
    with open(SPAM) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            total += 1
            if parts[0] == "spam":
                spam += 1
    return spam, total


# ---------------------------------------- fig 47.1: prior x likelihood = posterior (proper densities)
def fig_update() -> None:
    a0, b0 = 2, 2
    h, t = 3, 1
    x = np.linspace(0, 1, 400)
    prior = beta_dist.pdf(x, a0, b0)
    like = beta_dist.pdf(x, h + 1, t + 1)           # likelihood as a normalised density in theta
    post = beta_dist.pdf(x, a0 + h, b0 + t)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(x, prior, color=BLUE, lw=2.2, label=f"prior Beta({a0},{b0})")
    ax.plot(x, like, color=GOLD, lw=2.0, ls=(0, (5, 3)), label="правдоподобие 3/4 (как функция $\\theta$)")
    ax.plot(x, post, color=RED, lw=2.6, label=f"posterior Beta({a0+h},{b0+t})")
    ax.fill_between(x, post, color=RED, alpha=0.10)
    for dist, c, lab in [((a0, b0), BLUE, "prior"), ((a0 + h, b0 + t), RED, "posterior")]:
        m = dist[0] / (dist[0] + dist[1])
        ax.axvline(m, color=c, lw=0.8, ls=(0, (2, 2)))
    print(f"update: prior mean {a0/(a0+b0):.2f}, posterior mean {(a0+h)/(a0+b0+h+t):.3f}")
    assert abs((a0 + h) / (a0 + b0 + h + t) - 0.625) < 1e-9
    ax.set_xlabel("вероятность орла $\\theta$"); ax.set_ylabel("плотность (площадь = 1)")
    ax.set_title("Апостериор = prior × правдоподобие, все как настоящие плотности")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "update.png")


# ---------------------------------------- fig 47.2: weak vs strong prior, same data
def fig_weak_strong() -> None:
    h, t = 3, 1
    x = np.linspace(0, 1, 400)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True)
    for ax, (pa, pb), tit in [(a0, (2, 2), "слабый prior Beta(2,2)"),
                              (a1, (50, 50), "сильный prior Beta(50,50)")]:
        ax.plot(x, beta_dist.pdf(x, pa, pb), color=BLUE, lw=2.0, label="prior")
        ax.plot(x, beta_dist.pdf(x, pa + h, pb + t), color=RED, lw=2.4, label="posterior")
        pm = (pa + h) / (pa + pb + h + t)
        ax.axvline(0.5, color=MUTED, lw=0.8, ls=(0, (2, 2)))
        ax.axvline(pm, color=RED, lw=0.9, ls=(0, (3, 2)))
        ax.annotate(f"среднее {pm:.2f}", xy=(pm, 0.5), xytext=(pm + 0.03, 1.2 if pa == 2 else 4),
                    fontsize=9.5, color=RED)
        ax.set_title(tit, fontsize=12); ax.set_xlabel("$\\theta$")
        ax.legend(loc="upper left", frameon=False, fontsize=9.5)
        ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    a0.set_ylabel("плотность")
    fig.suptitle("Одни данные 3/4: слабый prior сдвигается сильно, сильный — едва", y=1.03, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "weak_strong.png")
    print("weak_strong drawn")


# ---------------------------------------- fig 47.3: sequential updating on real spam
def fig_sequential_real() -> None:
    spam, total = spam_counts()
    phat = spam / total
    rate = phat
    x = np.linspace(0.05, 0.25, 500)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    stages = [(0, "prior Beta(1,1)"), (50, "n=50"), (500, "n=500"), (total, f"n={total} (все)")]
    cols = [MUTED, GOLD, GREEN, BLUE]
    for (n, lab), c in zip(stages, cols):
        h = round(rate * n)
        post = beta_dist.pdf(x, 1 + h, 1 + (n - h))
        ax.plot(x, post, color=c, lw=2.2, label=lab)
    # credible interval on full data
    a, b = 1 + spam, 1 + (total - spam)
    lo, hi = beta_dist.ppf(0.025, a, b), beta_dist.ppf(0.975, a, b)
    print(f"sequential: posterior mean {a/(a+b):.3f}, 95% credible [{lo:.3f},{hi:.3f}]")
    assert abs(a / (a + b) - 0.134) < 0.002
    ax.axvspan(lo, hi, color=BLUE, alpha=0.10)
    ax.annotate(f"95% credible interval\n[{lo:.3f}, {hi:.3f}]", xy=((lo + hi) / 2, 5),
                xytext=(0.17, 20), fontsize=10, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=LINE, lw=1.0))
    ax.set_xlabel("доля спама $\\theta$"); ax.set_ylabel("апостериорная плотность")
    ax.set_title("Байес на реальных SMS: апостериор сужается вокруг 0,134")
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "sequential_real.png")


# ---------------------------------------- margins
def side_odds() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 1.9))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 3)
    from matplotlib.patches import Rectangle, FancyArrowPatch
    for x, lab, col in [(0.4, "prior\nшансы", MUTED), (4.6, "× Bayes\nfactor", GOLD), (9.0, "posterior\nшансы", RED)]:
        ax.add_patch(Rectangle((x, 0.8), 2.5, 1.4, fc=WASH, ec=col, lw=1.5))
        ax.text(x + 1.25, 1.5, lab, ha="center", va="center", fontsize=8.5, color=col)
    ax.add_patch(FancyArrowPatch((3.0, 1.5), (4.5, 1.5), arrowstyle="-|>", mutation_scale=11, color=MUTED, lw=1.3))
    ax.add_patch(FancyArrowPatch((7.2, 1.5), (8.9, 1.5), arrowstyle="-|>", mutation_scale=11, color=MUTED, lw=1.3))
    ax.set_title("обновление в шансах", fontsize=9.5)
    save(fig, SIDE / "odds.png")


def side_credible() -> None:
    x = np.linspace(0, 1, 300)
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    post = beta_dist.pdf(x, 8, 4)
    ax.plot(x, post, color=RED, lw=2.0)
    lo, hi = beta_dist.ppf(0.025, 8, 4), beta_dist.ppf(0.975, 8, 4)
    m = (x >= lo) & (x <= hi)
    ax.fill_between(x[m], post[m], color=RED, alpha=0.18)
    ax.set_xlabel("$\\theta$", fontsize=9); ax.set_yticks([])
    ax.set_title("credible interval: 95% массы", fontsize=9.5)
    save(fig, SIDE / "credible.png")


def side_predictive() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.3))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    ax.text(5, 4.5, "предсказание = среднее прогнозов", ha="center", fontsize=8.5, color=INK)
    ax.text(5, 3.0, r"$p(\tilde y\mid D)=\int p(\tilde y\mid\theta)\,p(\theta\mid D)\,d\theta$",
            ha="center", fontsize=11, color=BLUE)
    ax.text(5, 1.5, "усредняем по всей неопределённости $\\theta$,", ha="center", fontsize=8.5, color=MUTED)
    ax.text(5, 0.9, "а не подставляем одну оценку", ha="center", fontsize=8.5, color=MUTED)
    ax.set_title("апостериорный прогноз", fontsize=9.5)
    save(fig, SIDE / "predictive.png")
    print("predictive drawn")


fig_update()
fig_weak_strong()
fig_sequential_real()
side_odds()
side_credible()
side_predictive()
print("lesson 47 figures written")
