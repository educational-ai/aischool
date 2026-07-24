"""Deterministic figures for lesson 43: the prosecutor's fallacy.

The confusion of the inverse (P(E|not H) is not P(not H|E)) shown as natural
frequencies in a city of a million, the way "one in a million" becomes a near-certain
event once a whole database is searched, and a REAL multiple-testing experiment where
the largest spurious correlation grows with the number of hypotheses tried. Numbers
asserted; the likelihood ratio uses an explicit sensitivity 0.98 (not 1).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
BIKE = ROOT / "scripts" / "data" / "bike-sharing-hour.csv"
OUT = ROOT / "public" / "figures" / "lessons" / "43"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "43"

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


# ---------------------------------------- fig 43.1: confusion of the inverse
def fig_confusion() -> None:
    # city of a million, random-match p=1e-5 -> ~10 innocent matches + 1 guilty
    N = 1_000_000
    p = 1e-5
    sens = 0.98
    innocent_matches = round((N - 1) * p)
    guilty_match = 1  # guilty is in the city, detected with prob sens ~ 1 here
    total = innocent_matches + guilty_match
    guilt_given_match = guilty_match / total
    print(f"confusion: innocent matches {innocent_matches}, P(guilt|match)={guilt_given_match:.3f}")
    assert innocent_matches == 10 and abs(guilt_given_match - 1 / 11) < 0.01
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 7)
    ax.text(6, 6.6, "Город из миллиона человек, совпадение у 1 из 100 000", ha="center", fontsize=13, color=INK)
    # the confusion of the inverse — stacked so nothing collides
    ax.text(6, 5.65, "эксперт знает:  $P(\\mathrm{совпадение}\\mid \\mathrm{невиновен})=10^{-5}$", ha="center", fontsize=12, color=BLUE)
    ax.text(6, 4.95, "это РАЗНЫЕ числа, их нельзя путать", ha="center", fontsize=10.5, color=MUTED)
    ax.text(6, 4.25, "суду нужно:  $P(\\mathrm{невиновен}\\mid \\mathrm{совпадение})=\\,?$", ha="center", fontsize=12, color=RED)
    # bottom: 11 matches, 1 guilty
    y = 2.5
    for k in range(total):
        c = RED if k == 0 else BLUE
        ax.add_patch(Rectangle((1.2 + k * 0.9, y), 0.7, 0.9, fc=c, ec=PAPER, lw=1, alpha=0.85))
    ax.text(1.55, y + 0.45, "?", ha="center", va="center", color=PAPER, fontsize=13)
    ax.text(6, y - 0.5, "среди 11 совпавших ровно 1 виновен: $P(\\mathrm{виновен}\\mid\\mathrm{совпал})\\approx 1/11$, а не $0{,}99999$",
            ha="center", fontsize=11, color=INK)
    ax.add_patch(Rectangle((1.2, 0.5), 0.5, 0.4, fc=RED, ec="none")); ax.text(1.85, 0.7, "виновный", fontsize=9, color=INK, va="center")
    ax.add_patch(Rectangle((4.2, 0.5), 0.5, 0.4, fc=BLUE, ec="none")); ax.text(4.85, 0.7, "случайные совпадения (10)", fontsize=9, color=INK, va="center")
    save(fig, OUT / "confusion.png")


# ---------------------------------------- fig 43.2: one in a million over a database
def fig_database() -> None:
    p = 1e-6
    Ns = np.logspace(0, 7, 200)
    at_least = 1 - (1 - p) ** Ns
    expected = p * Ns
    print(f"database: at N=1e6, P(>=1)={1-(1-p)**1e6:.3f}")
    assert abs((1 - (1 - p) ** 1e6) - 0.632) < 0.005
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(Ns, at_least, color=RED, lw=2.4, label="$P(\\geq 1$ совпадение$)=1-(1-p)^N$")
    exp_line = np.where(expected <= 1.06, expected, np.nan)
    ax.plot(Ns, exp_line, color=BLUE, lw=1.6, ls=(0, (5, 3)), label="ожидаемое число $Np$")
    ax.axvline(1e6, color=GOLD, lw=1.4, ls=(0, (3, 3)))
    ax.plot(1e6, 0.632, "o", color=RED, markersize=8)
    ax.annotate("«один на миллион» при базе\nв миллион записей = 63%",
                xy=(1e6, 0.632), xytext=(2e3, 0.8), fontsize=11, color=INK,
                arrowprops=dict(arrowstyle="->", color=LINE, lw=1.0))
    ax.set_xscale("log"); ax.set_xlabel("размер базы $N$ (лог-шкала)"); ax.set_ylabel("вероятность / доля")
    ax.set_ylim(0, 1.08)
    ax.set_title("Редкое для одного — обычное для базы ($p=10^{-6}$)")
    ax.legend(loc="center left", frameon=False, fontsize=10)
    ax.grid(True, which="both", color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "database.png")


# ---------------------------------------- fig 43.3: real multiple-testing / look-elsewhere
def fig_multiple() -> None:
    # real target: bike hourly counts; correlate with many RANDOM features (null true)
    cnt = []
    with open(BIKE) as f:
        for row in csv.DictReader(f):
            cnt.append(int(row["cnt"]))
    y = np.array(cnt, float)[:2000]
    y = (y - y.mean()) / y.std()
    rng = np.random.default_rng(43)
    Ns = [1, 3, 10, 30, 100, 300, 1000, 3000]
    max_abs = []
    for N in Ns:
        X = rng.standard_normal((len(y), N))
        X = (X - X.mean(0)) / X.std(0)
        corr = np.abs(X.T @ y / len(y))
        max_abs.append(corr.max())
    max_abs = np.array(max_abs)
    print(f"multiple: max |corr| grows {max_abs.round(3)}")
    assert max_abs[-1] > max_abs[0]
    # expected max of |N(0,1/n)| under null ~ sqrt(2 log N)/sqrt(n)
    exp_max = np.sqrt(2 * np.log(np.maximum(Ns, 2))) / np.sqrt(len(y))
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(Ns, max_abs, "o-", color=RED, lw=2.2, markersize=6, label="наибольшая |корреляция| среди $N$ случайных")
    ax.plot(Ns, exp_max, color=BLUE, lw=1.6, ls=(0, (5, 3)), label="ожидаемый максимум под нулём")
    ax.set_xscale("log")
    ax.set_xlabel("сколько случайных признаков проверено $N$ (лог-шкала)")
    ax.set_ylabel("максимальная |корреляция| с реальной целью")
    ax.set_title("Ищешь среди многих — найдёшь ложное (реальные данные велопроката)")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.annotate("ни один признак не связан с целью,\nно максимум растёт с числом проверок",
                xy=(3000, max_abs[-1]), xytext=(3, max_abs[-1] - 0.01), fontsize=9.5, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=LINE, lw=1.0))
    ax.grid(True, which="both", color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "multiple.png")


# ---------------------------------------- margins
def side_odds() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    priors = np.array([1e-6, 1e-4, 1e-2])
    lr = 1e4
    post_odds = priors / (1 - priors) * lr
    y = [0, 1, 2]
    for i, (pr, po) in enumerate(zip(priors, post_odds)):
        po_prob = po / (1 + po)
        ax.plot([np.log10(pr / (1 - pr)), np.log10(po)], [y[i], y[i]], color=MUTED, lw=1.0)
        ax.plot(np.log10(pr / (1 - pr)), y[i], "o", color=BLUE, markersize=7)
        ax.plot(np.log10(po), y[i], "o", color=RED, markersize=7)
        ax.annotate("", xy=(np.log10(po), y[i]), xytext=(np.log10(pr / (1 - pr)), y[i]),
                    arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=1.4))
        ax.text(np.log10(po) + 0.2, y[i], f"{po_prob:.2f}", fontsize=9, color=RED, va="center")
    ax.set_yticks(y); ax.set_yticklabels(["1:млн", "1:10тыс", "1:100"], fontsize=8)
    ax.set_xlabel("$\\log_{10}$ шансов виновности", fontsize=9)
    ax.set_title("$LR=10^4$ сдвигает, но не задаёт", fontsize=9.5)
    ax.set_ylim(-0.6, 2.6)
    save(fig, SIDE / "odds.png")


def side_poisson() -> None:
    from math import factorial
    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    ks = np.arange(0, 25)
    for lam, c in [(5, BLUE), (10, GOLD)]:
        pmf = np.array([np.exp(-lam) * lam ** k / factorial(k) for k in ks])
        ax.bar(ks + (0 if lam == 5 else 0.0), pmf, color=c, alpha=0.5, width=0.9, label=f"среднее {lam}")
    ax.set_xlabel("число ложных совпадений", fontsize=9); ax.set_yticks([])
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_title("«ожидается 10» — это разброс", fontsize=9.5)
    save(fig, SIDE / "poisson.png")


def side_dependence() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.3))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    # common cause C -> E1, E2
    ax.text(5, 5.4, "общая причина", ha="center", fontsize=9, color=RED)
    from matplotlib.patches import Circle
    ax.add_patch(Circle((5, 4.3), 0.5, fc=WASH, ec=RED, lw=1.6)); ax.text(5, 4.3, "$C$", ha="center", va="center", fontsize=11, color=RED)
    for x in (2.5, 7.5):
        ax.add_patch(Circle((x, 1.6), 0.5, fc=WASH, ec=BLUE, lw=1.6))
        ax.add_patch(FancyArrowPatch((5, 3.8), (x, 2.1), arrowstyle="-|>", mutation_scale=11, color=MUTED, lw=1.3))
    ax.text(2.5, 1.6, "$E_1$", ha="center", va="center", fontsize=11, color=BLUE)
    ax.text(7.5, 1.6, "$E_2$", ha="center", va="center", fontsize=11, color=BLUE)
    ax.text(5, 0.5, "загрязнение образца делает $E_1,E_2$ зависимыми", ha="center", fontsize=8.5, color=INK)
    ax.set_title("нельзя перемножать зависимые улики", fontsize=9.5)
    save(fig, SIDE / "dependence.png")


fig_confusion()
fig_database()
fig_multiple()
side_odds()
side_poisson()
side_dependence()
print("lesson 43 figures written")
