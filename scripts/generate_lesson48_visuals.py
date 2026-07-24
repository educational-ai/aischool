"""Deterministic figures for lesson 48: conjugate priors and Laplace's rule.

The rule of succession (zero accidents in 20000 trips is not zero risk, but (h+1)/(n+2)),
Laplace add-one smoothing on the REAL SMS-spam corpus (a word never seen in one class gives
a zero probability that wrecks Naive Bayes until smoothed), and Beta-Binomial overdispersion.
Numbers asserted.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_dist, binom

ROOT = Path(__file__).resolve().parents[1]
SPAM = ROOT / "scripts" / "data" / "sms-spam-collection.tsv"
OUT = ROOT / "public" / "figures" / "lessons" / "48"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "48"

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


def spam_word_counts():
    spam = ham = 0
    ws, wh = {}, {}
    with open(SPAM) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            label, text = parts[0], parts[1]
            toks = set(re.findall(r"[a-z]+", text.lower()))
            if label == "spam":
                spam += 1
                for w in toks:
                    ws[w] = ws.get(w, 0) + 1
            else:
                ham += 1
                for w in toks:
                    wh[w] = wh.get(w, 0) + 1
    return spam, ham, ws, wh


# ---------------------------------------- fig 48.1: Laplace rule of succession
def fig_succession() -> None:
    x = np.linspace(0, 0.0004, 400)
    n = 20000
    post = beta_dist.pdf(x, 1, n + 1)   # Beta(1, 20001) for 0 successes
    pnext = 1 / (n + 2)
    print(f"succession: 0/{n} -> P(next)={pnext:.2e}")
    assert abs(pnext - 1 / 20002) < 1e-12
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(x * 1e4, post / post.max(), color=RED, lw=2.4)
    ax.fill_between(x * 1e4, post / post.max(), color=RED, alpha=0.10)
    ax.axvline(pnext * 1e4, color=INK, lw=1.2, ls=(0, (4, 3)))
    ax.annotate(f"$P(\\text{{авария далее}})=\\dfrac{{0+1}}{{20000+2}}\\approx1/20002$",
                xy=(pnext * 1e4, 0.6), xytext=(1.2, 0.75), fontsize=12, color=INK,
                arrowprops=dict(arrowstyle="->", color=LINE, lw=1.0))
    ax.text(1.2, 0.45, "наивная оценка $0/20000=0$ была бы\nсильнее данных: ноль аварий\nне значит нулевой риск",
            fontsize=10.5, color=MUTED)
    ax.set_xlabel("вероятность аварии $\\theta$, $\\times10^{-4}$"); ax.set_ylabel("posterior (нормировано на пик)")
    ax.set_title("Правило Лапласа: посл. распределение при нуле аварий из 20 000")
    ax.set_xlim(0, 4)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "succession.png")


# ---------------------------------------- fig 48.2: Laplace smoothing on real spam words
def fig_smoothing_real() -> None:
    spam, ham, ws, wh = spam_word_counts()
    # pick spam-frequent words with a RANGE of ham counts (incl 0) to show a gradient
    freq = [(w, ws[w], wh.get(w, 0)) for w in ws if ws[w] >= 30]
    chosen, seen_counts = [], set()
    for target in [0, 1, 3, 8, 25]:
        best = min((c for c in freq if c[2] not in seen_counts), key=lambda c: abs(c[2] - target), default=None)
        if best:
            chosen.append(best); seen_counts.add(best[2])
    chosen.sort(key=lambda c: c[2])
    words = [w for w, _, _ in chosen]
    print(f"smoothing: words with ham counts {[c[2] for c in chosen]}: {words}")
    assert len(words) >= 4
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    raw_h = [wh.get(w, 0) / ham for w in words]                 # 0 or tiny
    sm_h = [(wh.get(w, 0) + 1) / (ham + 2) for w in words]      # Laplace add-one
    y = np.arange(len(words))
    ax.barh(y - 0.2, raw_h, height=0.35, color=RED, alpha=0.8, label="сырая $c/n$")
    ax.barh(y + 0.2, sm_h, height=0.35, color=BLUE, alpha=0.8, label="сглажённая $(c+1)/(n+2)$")
    for i, (w, _, hc) in enumerate(chosen):
        ax.text(sm_h[i] + 2e-5, i + 0.2, f"{sm_h[i]:.1e}", va="center", fontsize=9, color=BLUE)
        ax.text(-0.00012, i, f"в ham: {hc}", va="center", ha="right", fontsize=8.5, color=MUTED)
    ax.set_yticks(y); ax.set_yticklabels(words)
    ax.set_xlabel("оценка $P(\\text{слово}\\mid \\text{не спам})$")
    ax.set_title("Сглаживание Лапласа: слово с нулём в классе не обнуляет вероятность")
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.set_xlim(-0.0006, max(sm_h) * 1.5)
    ax.text(0.5, -0.16, "без сглаживания слово с $c=0$ превращает произведение вероятностей в ноль",
            transform=ax.transAxes, ha="center", fontsize=9.5, color=MUTED)
    ax.grid(True, axis="x", color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "smoothing_real.png")


# ---------------------------------------- fig 48.3: Binomial vs Beta-Binomial overdispersion
def fig_overdispersion() -> None:
    n = 50
    k = np.arange(0, n + 1)
    p = 0.2
    bino = binom.pmf(k, n, p)
    a, b = 2, 8   # Beta(2,8) mean 0.2
    from scipy.special import betaln
    from math import comb
    bb = np.array([comb(n, int(kk)) * np.exp(betaln(a + kk, b + n - kk) - betaln(a, b)) for kk in k])
    print(f"overdispersion: binom var {n*p*(1-p):.1f}, beta-binom heavier tails")
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.bar(k - 0.2, bino, width=0.4, color=BLUE, alpha=0.7, label="Binomial (фикс. p=0,2)")
    ax.bar(k + 0.2, bb, width=0.4, color=RED, alpha=0.6, label="Beta-Binomial (p различается)")
    ax.axvline(n * p, color=INK, lw=1.0, ls=(0, (3, 3)))
    ax.set_xlabel("число успехов из 50"); ax.set_ylabel("вероятность")
    ax.set_title("Неопределённый параметр даёт лишний разброс")
    ax.set_xlim(0, 30); ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "overdispersion.png")


# ---------------------------------------- margins
def side_gamma() -> None:
    x = np.linspace(0, 8, 300)
    from scipy.stats import gamma
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    ax.plot(x, gamma.pdf(x, 2, scale=1), color=BLUE, lw=1.8, label="prior Ga(2,1)")
    ax.plot(x, gamma.pdf(x, 2 + 34, scale=1 / (1 + 10)), color=RED, lw=2.0, label="posterior Ga(36,11)")
    ax.set_xlabel("интенсивность $\lambda$", fontsize=9); ax.set_yticks([]); ax.set_xlim(0, 8)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_title("Gamma–Poisson: события и время", fontsize=9)
    save(fig, SIDE / "gamma.png")


def side_shrinkage() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    groups = [(0, 3), (1, 5), (4, 20), (20, 100), (210, 1000)]
    pm = 0.2
    for i, (h, n) in enumerate(groups):
        raw = h / n
        sm = (h + 2) / (n + 10)
        ax.plot([raw, sm], [i, i], color=LINE, lw=1.0)
        ax.plot(raw, i, "o", color=BLUE, markerfacecolor="white", markersize=7)
        ax.plot(sm, i, "o", color=RED, markersize=6)
    ax.axvline(pm, color=INK, lw=1.0, ls=(0, (3, 3)))
    ax.set_yticks(range(5)); ax.set_yticklabels([f"{h}/{n}" for h, n in groups], fontsize=8)
    ax.set_xlabel("доля", fontsize=9); ax.set_xlim(-0.05, 0.5)
    ax.set_title("малые группы стягиваются сильнее", fontsize=9)
    save(fig, SIDE / "shrinkage.png")


def side_normal() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.2))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    ax.text(5, 4.3, "Normal–Normal: среднее по точностям", ha="center", fontsize=8.5, color=INK)
    ax.text(5, 2.8, r"$\hat\mu=\frac{\tau_0\mu_0+n\tau\,\bar x}{\tau_0+n\tau}$", ha="center", fontsize=12, color=BLUE)
    ax.text(5, 1.2, "вес пропорционален точности (1/дисперсия)", ha="center", fontsize=8.5, color=MUTED)
    ax.set_title("сопряжённость для среднего", fontsize=9.5)
    save(fig, SIDE / "normal.png")
    print("normal drawn")


fig_succession()
fig_smoothing_real()
fig_overdispersion()
side_gamma()
side_shrinkage()
side_normal()
print("lesson 48 figures written")
