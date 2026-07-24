"""Deterministic figures for lesson 45: maximum likelihood.

The likelihood of the spam fraction estimated on the REAL SMS corpus (sharpening as the
sample grows), the log turning a vanishing product into a visible curve with curvature at
the peak, and the bridge from maximum likelihood to familiar losses (Gaussian NLL = squared
error, Bernoulli NLL = cross-entropy). Numbers asserted.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPAM = ROOT / "scripts" / "data" / "sms-spam-collection.tsv"
OUT = ROOT / "public" / "figures" / "lessons" / "45"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "45"

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


# ---------------------------------------- fig 45.1: likelihood on real spam, sharpening
def fig_likelihood_real() -> None:
    spam, total = spam_counts()
    phat = spam / total
    print(f"likelihood_real: spam {spam}/{total}, MLE p={phat:.3f}")
    assert abs(phat - 0.134) < 0.002
    p = np.linspace(0.001, 0.5, 500)
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for (h, n, c, lab) in [(int(0.134 * 50), 50, GOLD, "n=50"),
                           (int(0.134 * 500), 500, GREEN, "n=500"),
                           (spam, total, BLUE, f"n={total} (все SMS)")]:
        ll = h * np.log(p) + (n - h) * np.log(1 - p)
        ll = ll - ll.max()               # normalise to peak 0
        ax.plot(p, np.exp(ll), color=c, lw=2.2, label=lab)
    ax.axvline(phat, color=RED, lw=1.4, ls=(0, (4, 3)))
    ax.annotate(f"MLE $\\hat p={phat:.3f}$", xy=(phat, 0.5), xytext=(phat + 0.06, 0.7),
                fontsize=11, color=RED, arrowprops=dict(arrowstyle="->", color=LINE, lw=1.0))
    ax.set_xlabel("доля спама $p$"); ax.set_ylabel("правдоподобие (нормировано на пик)")
    ax.set_title("Правдоподобие на реальных SMS: больше данных — острее пик")
    ax.set_xlim(0, 0.4); ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "likelihood_real.png")


# ---------------------------------------- fig 45.2: linear vs log likelihood
def fig_log_vs_lin() -> None:
    h, t = 14, 6
    p = np.linspace(0.001, 0.999, 500)
    L = p ** h * (1 - p) ** t
    ll = h * np.log(p) + t * np.log(1 - p)
    phat = h / (h + t)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.6, 4.4))
    a0.plot(p, L / L.max(), color=BLUE, lw=2.2)
    a0.axvline(phat, color=RED, lw=1.2, ls=(0, (4, 3)))
    a0.set_title("правдоподобие $L(p)$"); a0.set_xlabel("$p$"); a0.set_ylabel("нормировано")
    a0.text(0.06, 0.5, "далёкие значения\nсливаются с нулём", fontsize=9.5, color=MUTED)
    a0.grid(True, color=GRID, lw=0.4, alpha=0.4); a0.set_axisbelow(True)
    a1.plot(p, ll - ll.max(), color=GREEN, lw=2.2, label="$\\ell(p)-\\ell(\\hat p)$")
    # quadratic approx at max
    d2 = -(h / phat ** 2 + t / (1 - phat) ** 2)
    a1.plot(p, 0.5 * d2 * (p - phat) ** 2, color=RED, lw=1.6, ls=(0, (4, 3)), label="парабола 2-го порядка")
    a1.axvline(phat, color=RED, lw=1.0, ls=(0, (2, 2)))
    a1.set_ylim(-12, 1)
    a1.set_title("лог-правдоподобие $\\ell(p)$"); a1.set_xlabel("$p$")
    a1.legend(loc="lower center", frameon=False, fontsize=9.5)
    a1.grid(True, color=GRID, lw=0.4, alpha=0.4); a1.set_axisbelow(True)
    fig.suptitle("Логарифм превращает произведение в сумму и делает разницу видимой", y=1.03, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "log_vs_lin.png")
    print("log_vs_lin drawn")


# ---------------------------------------- fig 45.3: MLE -> familiar losses
def fig_mle_to_loss() -> None:
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.6, 4.4))
    # Gaussian: NLL = squared error
    mus = np.linspace(-2, 4, 300)
    data = np.array([0.5, 1.2, 1.8, 0.9, 1.6])
    nll = np.array([np.sum((data - m) ** 2) for m in mus]) / 2
    a0.plot(mus, nll, color=BLUE, lw=2.2)
    mhat = data.mean()
    a0.axvline(mhat, color=RED, lw=1.2, ls=(0, (4, 3)))
    a0.annotate(f"MLE = среднее = {mhat:.2f}", xy=(mhat, nll.min()), xytext=(mhat + 0.3, nll.min() + 3),
                fontsize=10, color=RED, arrowprops=dict(arrowstyle="->", color=LINE, lw=1.0))
    a0.set_title("нормальный шум: NLL = сумма квадратов")
    a0.set_xlabel("$\\mu$"); a0.set_ylabel("$-\\ell$ (с точностью до const)")
    a0.grid(True, color=GRID, lw=0.4, alpha=0.4); a0.set_axisbelow(True)
    # Bernoulli: NLL = cross-entropy
    ps = np.linspace(0.01, 0.99, 300)
    h, n = 3, 10
    ce = -(h * np.log(ps) + (n - h) * np.log(1 - ps))
    a1.plot(ps, ce, color=GREEN, lw=2.2)
    phat = h / n
    a1.axvline(phat, color=RED, lw=1.2, ls=(0, (4, 3)))
    a1.annotate(f"MLE = доля = {phat:.1f}", xy=(phat, ce.min()), xytext=(phat + 0.1, ce.min() + 4),
                fontsize=10, color=RED, arrowprops=dict(arrowstyle="->", color=LINE, lw=1.0))
    a1.set_title("бинарный ответ: NLL = кросс-энтропия")
    a1.set_xlabel("$p$")
    a1.grid(True, color=GRID, lw=0.4, alpha=0.4); a1.set_axisbelow(True)
    fig.suptitle("Максимум правдоподобия = минимум знакомой функции потерь", y=1.03, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "mle_to_loss.png")
    print("mle_to_loss drawn")


# ---------------------------------------- margins
def side_curvature() -> None:
    p = np.linspace(0.3, 0.95, 400)
    phat = 0.7
    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    for n, c in [(10, GOLD), (40, GREEN), (160, BLUE)]:
        h = phat * n
        ll = h * np.log(p) + (n - h) * np.log(1 - p)
        ax.plot(p, ll - ll.max(), color=c, lw=1.8, label=f"n={n}")
    ax.axvline(phat, color=RED, lw=1.0, ls=(0, (3, 3)))
    ax.set_ylim(-6, 0.5); ax.set_xlabel("$p$", fontsize=9); ax.set_yticks([])
    ax.legend(loc="lower center", frameon=False, fontsize=8)
    ax.set_title("кривизна растёт как n", fontsize=9.5)
    save(fig, SIDE / "curvature.png")


def side_boundary() -> None:
    p = np.linspace(0.001, 1, 400)
    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    L = p ** 10
    ax.plot(p, L / L.max(), color=BLUE, lw=2.0)
    ax.plot(1, 1, "o", color=RED, markersize=7)
    ax.annotate("максимум на границе $p=1$", xy=(1, 1), xytext=(0.28, 0.6), fontsize=9, color=RED,
                arrowprops=dict(arrowstyle="->", color=LINE, lw=1.0))
    ax.set_xlabel("$p$", fontsize=9); ax.set_yticks([])
    ax.set_title("10 успехов из 10: край", fontsize=9.5)
    save(fig, SIDE / "boundary.png")


def side_gaussian() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 2.2))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    ax.text(5, 4.4, "нормаль: MLE двух параметров", ha="center", fontsize=9.5, color=INK)
    ax.text(5, 3.2, r"$\hat\mu=\frac{1}{n}\sum x_i$", ha="center", fontsize=13, color=BLUE)
    ax.text(5, 1.9, r"$\hat\sigma^2=\frac{1}{n}\sum (x_i-\hat\mu)^2$", ha="center", fontsize=13, color=GREEN)
    ax.text(5, 0.7, "делит на n, а не на n−1: смещена", ha="center", fontsize=9, color=RED)
    save(fig, SIDE / "gaussian.png")
    print("gaussian drawn")


fig_likelihood_real()
fig_log_vs_lin()
fig_mle_to_loss()
side_curvature()
side_boundary()
side_gaussian()
print("lesson 45 figures written")
