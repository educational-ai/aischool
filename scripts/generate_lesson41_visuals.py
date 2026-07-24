"""Deterministic figures for lesson 41: probability as the language of uncertainty.

The sample space of two dice with an event outlined, a REAL random variable (bike
rides per hour from the bike-sharing dataset) with its mean far from the typical value,
and the law of large numbers watched on that real data. Numbers asserted.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "scripts" / "data" / "bike-sharing-hour.csv"
OUT = ROOT / "public" / "figures" / "lessons" / "41"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "41"

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


def load_bikes():
    cnt, weather = [], []
    with open(DATA) as f:
        for row in csv.DictReader(f):
            cnt.append(int(row["cnt"])); weather.append(int(row["weathersit"]))
    return np.array(cnt), np.array(weather)


# ---------------------------------------- fig 41.1: sample space of two dice
def fig_sample_space() -> None:
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    for i in range(1, 7):
        for j in range(1, 7):
            s = i + j
            shade = plt.cm.YlGnBu((s - 2) / 10 * 0.8 + 0.1)
            ax.add_patch(Rectangle((i - 0.5, j - 0.5), 1, 1, fc=shade, ec=PAPER, lw=1.2))
            ax.text(i, j, str(s), ha="center", va="center", fontsize=11,
                    color=INK if s < 8 else PAPER)
    # event A: i+j >= 10 -> outline those cells
    for i in range(1, 7):
        for j in range(1, 7):
            if i + j >= 10:
                ax.add_patch(Rectangle((i - 0.5, j - 0.5), 1, 1, fill=False, ec=RED, lw=2.6))
    ax.set_xlim(0.5, 6.5); ax.set_ylim(0.5, 6.5); ax.set_aspect("equal")
    ax.set_xticks(range(1, 7)); ax.set_yticks(range(1, 7))
    ax.set_xlabel("первый кубик $i$"); ax.set_ylabel("второй кубик $j$")
    ax.set_title("Пространство исходов: 36 равновероятных пар\nкрасным — событие $i+j\\geq10$ (6 из 36 = $1/6$)", fontsize=13)
    print("sample_space: P(A)=6/36")
    assert sum(1 for i in range(1, 7) for j in range(1, 7) if i + j >= 10) == 6
    save(fig, OUT / "sample_space.png")


# ---------------------------------------- fig 41.2: a real random variable
def fig_real_rv() -> None:
    cnt, _ = load_bikes()
    mean, med, std = cnt.mean(), np.median(cnt), cnt.std()
    print(f"real_rv: mean={mean:.1f} median={med:.0f} std={std:.1f}")
    assert abs(mean - 189.5) < 1 and abs(med - 142) < 1
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.hist(cnt, bins=60, color=BLUE, alpha=0.7, edgecolor=PAPER, linewidth=0.3)
    ax.axvline(mean, color=RED, lw=2.0, label=f"среднее $\\mathbb{{E}}X={mean:.0f}$")
    ax.axvline(med, color=GREEN, lw=2.0, ls=(0, (4, 3)), label=f"медиана ${med:.0f}$")
    ax.annotate("длинный правый хвост:\nсреднее сдвинуто вправо от типичного",
                xy=(500, 400), xytext=(430, 1400), fontsize=10, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=LINE, lw=1.0))
    ax.set_xlabel("число поездок за час"); ax.set_ylabel("сколько часов (частота)")
    ax.set_title("Случайная величина на реальных данных: поездки за случайный час")
    ax.legend(loc="upper right", frameon=False, fontsize=11)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "real_rv.png")


# ---------------------------------------- fig 41.3: law of large numbers on real data
def fig_lln() -> None:
    cnt, _ = load_bikes()
    mu, sd = cnt.mean(), cnt.std()
    rng = np.random.default_rng(41)
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ns = np.arange(1, 3001)
    for c, seed in zip([BLUE, GREEN, GOLD], [1, 2, 3]):
        r = np.random.default_rng(seed)
        sample = cnt[r.integers(0, len(cnt), 3000)]
        run = np.cumsum(sample) / ns
        ax.plot(ns, run, color=c, lw=1.1, alpha=0.8)
    ax.axhline(mu, color=RED, lw=1.8, label=f"истинное среднее {mu:.0f}")
    ax.fill_between(ns, mu - sd / np.sqrt(ns), mu + sd / np.sqrt(ns), color=RED, alpha=0.10,
                    label="коридор $\\pm\\sigma/\\sqrt{n}$")
    ax.set_xscale("log")
    ax.set_xlabel("число наблюдений $n$ (лог-шкала)"); ax.set_ylabel("выборочное среднее")
    ax.set_title("Закон больших чисел: среднее сходится к $\\mathbb{E}X$")
    ax.set_ylim(mu - 130, mu + 130)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.grid(True, which="both", color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    print(f"lln: mu={mu:.1f}, band shrinks as 1/sqrt(n)")
    save(fig, OUT / "lln.png")


# ---------------------------------------- margins
def side_variance() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    xs = np.arange(0, 21)
    dists = [
        ("узкое", np.exp(-((xs - 10) ** 2) / 4), BLUE),
        ("шире", np.exp(-((xs - 10) ** 2) / 20), GREEN),
        ("с хвостами", 0.6 * np.exp(-((xs - 10) ** 2) / 40) + 0.08 * (np.abs(xs - 10) > 8), RED),
    ]
    for lab, p, c in dists:
        p = p / p.sum()
        ax.plot(xs, p, color=c, lw=1.8, marker="o", markersize=3, label=lab)
    ax.axvline(10, color=INK, lw=1.0, ls=(0, (3, 3)))
    ax.set_xlabel("значение", fontsize=9); ax.set_yticks([])
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_title("одно среднее — разный риск", fontsize=9.5)
    save(fig, SIDE / "variance.png")


def side_frequency() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(5.4, 1.9), sharey=True)
    theo = np.array([min(s - 1, 13 - s) for s in range(2, 13)]) / 36
    for ax, n, seed in zip(axes, [20, 200, 20000], [10, 11, 12]):
        r = np.random.default_rng(seed)
        rolls = r.integers(1, 7, (n, 2)).sum(1)
        vals, cnts = np.unique(rolls, return_counts=True)
        freq = np.zeros(11)
        for v, c in zip(vals, cnts):
            freq[v - 2] = c / n
        ax.bar(range(2, 13), freq, color=BLUE, alpha=0.7, width=0.8)
        ax.plot(range(2, 13), theo, "o", color=RED, markersize=3)
        ax.set_title(f"n={n}", fontsize=9); ax.set_xticks([2, 7, 12]); ax.tick_params(labelsize=7)
    fig.suptitle("частоты приближают вероятности", y=1.06, fontsize=9.5)
    fig.tight_layout()
    save(fig, SIDE / "frequency.png")


def side_conditional() -> None:
    cnt, weather = load_bikes()
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    labels = {1: "ясно", 2: "облачно", 3: "дождь/снег"}
    cols = {1: GOLD, 2: BLUE, 3: VIOLET}
    for w in [1, 2, 3]:
        m = weather == w
        ax.hist(cnt[m], bins=40, density=True, histtype="step", color=cols[w], lw=1.6,
                label=f"{labels[w]} ({cnt[m].mean():.0f})")
    ax.set_xlabel("поездок за час", fontsize=9); ax.set_yticks([]); ax.set_xlim(0, 700)
    ax.legend(loc="upper right", frameon=False, fontsize=8, title="погода (среднее)")
    ax.set_title("распределение зависит от условий", fontsize=9.5)
    print(f"conditional: means clear/cloud/rain = {cnt[weather==1].mean():.0f}/{cnt[weather==2].mean():.0f}/{cnt[weather==3].mean():.0f}")
    save(fig, SIDE / "conditional.png")


fig_sample_space()
fig_real_rv()
fig_lln()
side_variance()
side_frequency()
side_conditional()
print("lesson 41 figures written")
