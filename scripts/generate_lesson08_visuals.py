"""Deterministic figures for lesson 08: regression on Bike Sharing.

Train = 2011, test = 2012; the script recomputes every number quoted in the
lesson (MAE ladder, oracle floor, growth correction, quantile cushion).
"""

from __future__ import annotations

import csv
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "08"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "08"
DATA = ROOT / "scripts" / "data" / "bike-sharing-hour.csv"

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


def save(fig: plt.Figure, path: Path, *, dpi: int = 160) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


ROWS = list(csv.DictReader(DATA.open()))
TRAIN = [r for r in ROWS if r["yr"] == "0"]
TEST = [r for r in ROWS if r["yr"] == "1"]


def cnt(r) -> int:
    return int(r["cnt"])


MEAN_TR = st.mean(cnt(r) for r in TRAIN)

BY_H = defaultdict(list)
BY_HW = defaultdict(list)
BY_HWW = defaultdict(list)
for r in TRAIN:
    BY_H[r["hr"]].append(cnt(r))
    BY_HW[(r["hr"], r["workingday"])].append(cnt(r))
    wet = "0" if r["weathersit"] in ("1", "2") else "1"
    BY_HWW[(r["hr"], r["workingday"], wet)].append(cnt(r))
MU_H = {k: st.mean(v) for k, v in BY_H.items()}
MU_HW = {k: st.mean(v) for k, v in BY_HW.items()}
MU_HWW = {k: st.mean(v) for k, v in BY_HWW.items()}


def predict_hw(r) -> float:
    return MU_HW[(r["hr"], r["workingday"])]


MAE_LADDER = [
    ("константа", st.mean(abs(cnt(r) - MEAN_TR) for r in TEST)),
    ("+ час суток", st.mean(abs(cnt(r) - MU_H[r["hr"]]) for r in TEST)),
    ("+ тип дня", st.mean(abs(cnt(r) - predict_hw(r)) for r in TEST)),
    (
        "+ погода",
        st.mean(
            abs(
                cnt(r)
                - MU_HWW.get(
                    (r["hr"], r["workingday"], "0" if r["weathersit"] in ("1", "2") else "1"),
                    predict_hw(r),
                )
            )
            for r in TEST
        ),
    ),
]


# ---------------------------------------------------------------- figure 8.1
def mae_ladder() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    names = [n for n, _ in MAE_LADDER]
    values = [v for _, v in MAE_LADDER]
    colors = [FAINT, BLUE, BLUE, BLUE]
    bars = ax.bar(range(4), values, color=colors, width=0.58, zorder=3)
    for i, (b, v) in enumerate(zip(bars, values)):
        ax.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.1f}".replace(".", ","),
                ha="center", fontsize=12.6, fontweight="bold")
        if i:
            delta = values[i - 1] - v
            ax.annotate(
                f"−{delta:.1f}".replace(".", ","),
                xy=(i - 0.5, (values[i - 1] + v) / 2 + 16),
                ha="center", fontsize=10.6, color=GREEN, fontweight="bold",
            )
    ax.set_xticks(range(4), names)
    ax.set_ylabel("MAE на 2012 годе, велосипедов в час")
    ax.set_ylim(0, 190)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Лесенка моделей «среднее по ячейке»", loc="left", fontweight="bold", pad=12)
    save(fig, OUT / "mae-ladder.png")


# ---------------------------------------------------------------- figure 8.2
def day_curve() -> None:
    day = [r for r in TEST if r["dteday"] == "2012-09-12"]
    day.sort(key=lambda r: int(r["hr"]))
    hours = [int(r["hr"]) for r in day]
    facts = [cnt(r) for r in day]
    preds = [predict_hw(r) for r in day]

    fig, ax = plt.subplots(figsize=(11.6, 5.2))
    ax.plot(hours, preds, color=BLUE, lw=2.2, label="прогноз «час × тип дня» (2011)")
    ax.plot(hours, facts, "o", color=RED, ms=5, label="факт 12.09.2012")
    gap_h = 18
    ax.annotate(
        "недолёт на сотни аренд",
        xy=(gap_h, facts[gap_h]), xytext=(11.2, 830), fontsize=10.6, color=RED,
        arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 1.1, "shrinkB": 5},
    )
    ax.set_xticks(range(0, 24, 3), [f"{h}:00" for h in range(0, 24, 3)])
    ax.set_ylabel("аренды в час")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("Среда 12 сентября 2012 года: форма угадана, размах — нет",
                 loc="left", fontweight="bold", pad=12)
    save(fig, OUT / "day-curve.png")


# ---------------------------------------------------------------- figure 8.3
def temp_hump() -> None:
    bins = defaultdict(list)
    for r in ROWS:
        t = float(r["temp"]) * 47 - 8  # official min-max scaling of the dataset
        bins[int(t // 2) * 2].append(cnt(r))
    xs = sorted(b for b in bins if len(bins[b]) > 30)
    means = [st.mean(bins[b]) for b in xs]
    ses = [st.stdev(bins[b]) / math.sqrt(len(bins[b])) for b in xs]

    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    ax.errorbar([b + 1 for b in xs], means, yerr=[2 * s for s in ses],
                fmt="o-", color=BLUE, lw=1.8, ms=4.5, capsize=3,
                ecolor=FAINT, zorder=3)
    ax.annotate("плато +28…+32 °C", xy=(29, 336),
                xytext=(9, 320), fontsize=10.6, color=INK,
                arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.1, "shrinkB": 6})
    ax.annotate("жара: спрос отступает", xy=(37, means[-1] + 10),
                xytext=(25.5, 130), fontsize=10.6, color=RED,
                arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 1.1, "shrinkB": 6})
    ax.set_xlabel("температура, °C")
    ax.set_ylabel("средний спрос, аренд в час")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Температурный горб: связь важная, но не прямая",
                 loc="left", fontweight="bold", pad=12)
    save(fig, OUT / "temp-hump.png")


# ---------------------------------------------------------------- figure 8.4
def mean_median() -> None:
    sample = [1, 2, 3, 4, 99]
    cs = np.linspace(-10, 60, 500)
    sq = [sum((y - c) ** 2 for y in sample) for c in cs]
    ab = [sum(abs(y - c) for y in sample) for c in cs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.6))
    mean_v = st.mean(sample)
    med_v = st.median(sample)
    ax1.plot(cs, sq, color=BLUE, lw=2)
    ax1.axvline(mean_v, color=RED, lw=1.4, linestyle=(0, (5, 4)))
    ax1.text(mean_v + 1.4, max(sq) * 0.85, f"минимум в среднем\n$c={mean_v:.1f}$".replace(".", "{,}"),
             fontsize=10.4, color=RED, linespacing=1.35)
    ax1.set_title("Сумма квадратов", loc="left", fontweight="bold", fontsize=13)
    ax2.plot(cs, ab, color=BLUE, lw=2)
    ax2.axvline(med_v, color=GREEN, lw=1.4, linestyle=(0, (5, 4)))
    ax2.text(med_v + 1.6, max(ab) * 0.85, f"минимум в медиане\n$c={med_v:g}$",
             fontsize=10.4, color=GREEN, linespacing=1.35)
    ax2.set_title("Сумма модулей", loc="left", fontweight="bold", fontsize=13)
    for ax in (ax1, ax2):
        for y in sample:
            ax.plot(y, 0, "o", color=INK, ms=4, clip_on=False, zorder=5)
        ax.set_xlabel("константа $c$")
        ax.grid(color=GRID, lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.text(0.008, 0.965,
             "Выборка {1, 2, 3, 4, 99}: выброс утащил минимум квадратов к себе, минимум модулей остался с большинством",
             fontsize=11.4, color=MUTED)
    fig.subplots_adjust(top=0.82, wspace=0.18)
    save(fig, OUT / "mean-median.png")


# ---------------------------------------------------------------- figure 8.5
def residual_time() -> None:
    daily = defaultdict(list)
    for r in TEST:
        daily[r["dteday"]].append(cnt(r) - predict_hw(r))
    days = sorted(daily)
    xs = np.arange(len(days))
    means = [st.mean(daily[d]) for d in days]

    fig, ax = plt.subplots(figsize=(12.6, 5.0))
    ax.axhline(0, color=INK, lw=1.2)
    ax.plot(xs, means, "o", ms=3, color=BLUE, alpha=0.55)
    kernel = np.ones(28) / 28
    smooth = np.convolve(means, kernel, mode="valid")
    ax.plot(xs[14 : 14 + len(smooth)], smooth, color=RED, lw=2.2,
            label="скользящее среднее за 28 дней")
    month_pos = [i for i, d in enumerate(days) if d.endswith("-01")]
    ax.set_xticks(month_pos,
                  ["янв", "фев", "мар", "апр", "май", "июн",
                   "июл", "авг", "сен", "окт", "ноя", "дек"])
    ax.set_ylabel("средний остаток дня, велосипедов")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("2012 год: остатки не ходят вокруг нуля — они живут выше него",
                 loc="left", fontweight="bold", pad=12)
    save(fig, OUT / "residual-time.png")


# ---------------------------------------------------------------- figure 8.6
def quantile_cushion() -> None:
    def quant(v, q):
        v = sorted(v)
        i = (len(v) - 1) * q
        lo = int(i)
        hi = min(lo + 1, len(v) - 1)
        return v[lo] + (v[hi] - v[lo]) * (i - lo)

    q10 = {k: quant(v, 0.1) for k, v in BY_HW.items()}
    q50 = {k: quant(v, 0.5) for k, v in BY_HW.items()}
    q90 = {k: quant(v, 0.9) for k, v in BY_HW.items()}

    # Growth factor from the 28 days before the shown day.
    date = "2012-09-12"
    days = sorted({r["dteday"] for r in TEST})
    idx = days.index(date)
    window = set(days[idx - 28 : idx])
    act = sum(cnt(r) for r in TEST if r["dteday"] in window)
    pred = sum(predict_hw(r) for r in TEST if r["dteday"] in window)
    g = act / pred

    day = sorted((r for r in TEST if r["dteday"] == date), key=lambda r: int(r["hr"]))
    hours = [int(r["hr"]) for r in day]
    facts = [cnt(r) for r in day]
    lo = [g * q10[(r["hr"], r["workingday"])] for r in day]
    mid = [g * q50[(r["hr"], r["workingday"])] for r in day]
    hi = [g * q90[(r["hr"], r["workingday"])] for r in day]

    fig, ax = plt.subplots(figsize=(11.6, 5.4))
    ax.fill_between(hours, lo, hi, color=BLUE, alpha=0.16, label="коридор 10–90%")
    ax.plot(hours, mid, color=BLUE, lw=2, label="медианный прогноз")
    ax.plot(hours, facts, "o", color=RED, ms=5, label="факт 12.09.2012")
    ax.set_xticks(range(0, 24, 3), [f"{h}:00" for h in range(0, 24, 3)])
    ax.set_ylabel("аренды в час")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("Коридор квантилей с поправкой роста: неопределённость зависит от часа",
                 loc="left", fontweight="bold", pad=12)
    save(fig, OUT / "quantile-cushion.png")


# ------------------------------------------------------------- margin schemes
def residual_hist() -> None:
    res = [cnt(r) - predict_hw(r) for r in TEST]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.hist(res, bins=np.arange(-400, 640, 25), color=BLUE, alpha=0.85)
    ax.axvline(0, color=INK, lw=1.2)
    ax.text(180, ax.get_ylim()[1] * 0.8 if ax.get_ylim()[1] else 1, "", fontsize=1)
    ax.annotate("тяжёлый правый хвост", xy=(420, 30), xytext=(120, 700),
                fontsize=10, color=RED,
                arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 1.0, "shrinkB": 3})
    ax.set_yticks([])
    ax.set_xlabel("остаток, велосипедов")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    save(fig, SIDE / "residual-hist.png")


def growth() -> None:
    monthly = defaultdict(lambda: defaultdict(list))
    for r in ROWS:
        monthly[r["yr"]][int(r["mnth"])].append(cnt(r))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for yr, color, label in [("0", FAINT, "2011"), ("1", GOLD, "2012")]:
        xs = sorted(monthly[yr])
        ax.plot(xs, [st.mean(monthly[yr][m]) for m in xs], "o-", color=color,
                lw=2, ms=4, label=label)
    ax.set_xticks([1, 4, 7, 10], ["янв", "апр", "июл", "окт"])
    ax.set_ylabel("аренды в час, среднее")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9.6)
    save(fig, SIDE / "growth.png")


def pinball() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    err = np.linspace(-100, 100, 200)
    q = 0.9
    loss = np.where(err >= 0, q * err, -(1 - q) * err)
    ax.plot(err, loss, color=BLUE, lw=2.2)
    ax.text(48, 62, "недолёт:\nкруто ($q$)", fontsize=10, color=RED, linespacing=1.3)
    ax.text(-95, 24, "перелёт:\nполого ($1-q$)", fontsize=10, color=GREEN, linespacing=1.3)
    ax.axvline(0, color=LINE, lw=1)
    ax.set_xlabel("факт минус прогноз")
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    save(fig, SIDE / "pinball.png")


if __name__ == "__main__":
    mae_ladder()
    day_curve()
    temp_hump()
    mean_median()
    residual_time()
    quantile_cushion()
    residual_hist()
    growth()
    pinball()
    print("ladder:", [(n, round(v, 1)) for n, v in MAE_LADDER])
