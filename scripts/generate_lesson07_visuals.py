"""Deterministic figures for lesson 07: classification on SMS Spam Collection.

The script retrains the exact model of the article (same seed, features and
schedule), so every number in the lesson is reproducible from here.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "07"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "07"
DATA = ROOT / "scripts" / "data" / "sms-spam-collection.tsv"

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


# --------------------------------------------------------------- model rerun
def features(text: str) -> list[float]:
    lower = text.lower()
    return [
        1.0,
        len(text) / 100,
        sum(c.isdigit() for c in text) / 10,
        1.0 if "free" in lower else 0.0,
        1.0 if "call" in lower else 0.0,
        1.0 if "txt" in lower or "text" in lower else 0.0,
        1.0 if "www" in lower or "http" in lower else 0.0,
        1.0 if "win" in lower or "prize" in lower or "winner" in lower else 0.0,
    ]


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-max(-30, min(30, z))))


ROWS = [line.rstrip("\n").split("\t", 1) for line in DATA.open(encoding="utf-8")]
random.seed(7)
random.shuffle(ROWS)
TRAIN, TEST = ROWS[:4000], ROWS[4000:]

W = [0.0] * 8
for _ in range(30):
    for label, text in TRAIN:
        x = features(text)
        p = sigmoid(sum(a * b for a, b in zip(W, x)))
        g = p - (1.0 if label == "spam" else 0.0)
        for i in range(8):
            W[i] -= 0.05 * g * x[i]

TEST_SCORES = [
    (label, sum(a * b for a, b in zip(W, features(text))), text)
    for label, text in TEST
]


# ---------------------------------------------------------------- figure 7.1
def feature_plane() -> None:
    fig, ax = plt.subplots(figsize=(12.8, 6.2))
    rng = np.random.default_rng(7)
    for label, marker, color, alpha, size in [
        ("ham", "o", BLUE, 0.35, 9),
        ("spam", "o", RED, 0.6, 12),
    ]:
        xs, ys = [], []
        for lab, text in ROWS:
            if lab != label:
                continue
            xs.append(len(text) + rng.uniform(-1.5, 1.5))
            ys.append(sum(c.isdigit() for c in text) + rng.uniform(-0.4, 0.4))
        ax.scatter(xs, ys, s=size, color=color, alpha=alpha, lw=0,
                   label="переписка" if label == "ham" else "спам")

    # Two-feature boundary: refit a tiny model on (length/100, digits/10).
    w2 = [0.0, 0.0, 0.0]
    for _ in range(30):
        for lab, text in TRAIN:
            x2 = [1.0, len(text) / 100, sum(c.isdigit() for c in text) / 10]
            p = sigmoid(sum(a * b for a, b in zip(w2, x2)))
            g = p - (1.0 if lab == "spam" else 0.0)
            for i in range(3):
                w2[i] -= 0.05 * g * x2[i]
    xs_line = np.linspace(0, 420, 50)
    ys_line = (-(w2[0] + w2[1] * xs_line / 100) / w2[2]) * 10
    mask = (ys_line >= -1) & (ys_line <= 42)
    ax.plot(xs_line[mask], ys_line[mask], color=INK, lw=2.0, zorder=5,
            label="граница двух признаков")

    ax.set_xlim(0, 420)
    ax.set_ylim(-1, 42)
    ax.set_xlabel("длина сообщения, символов")
    ax.set_ylabel("цифр в сообщении")
    ax.grid(color=GRID, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("5574 настоящих SMS: длина, цифры и линейная граница",
                 loc="left", fontweight="bold", pad=12)
    save(fig, OUT / "feature-plane.png")


# ---------------------------------------------------------------- figure 7.2
def score_histograms() -> None:
    fig, ax = plt.subplots(figsize=(12.8, 5.4))
    ham = [s for lab, s, _ in TEST_SCORES if lab == "ham"]
    spam = [s for lab, s, _ in TEST_SCORES if lab == "spam"]
    bins = np.arange(-8, 13, 0.75)
    ax.hist(ham, bins=bins, color=BLUE, alpha=0.75, label="переписка")
    ax.hist(spam, bins=bins, color=RED, alpha=0.75, label="спам")
    ax.set_yscale("log")
    ax.axvline(0, color=INK, lw=1.8, linestyle=(0, (6, 4)))
    ax.text(0.15, 700, "порог $\\hat p=0{,}5$\n(счёт 0)", fontsize=10.6, va="top")
    bbox = {"boxstyle": "round,pad=0.32", "facecolor": PAPER, "edgecolor": LINE, "alpha": 0.94}
    ax.annotate("хвост спама левее порога:\nбудущие пропуски",
                xy=(-1.6, 4), xytext=(-7.6, 40), fontsize=10.2, color=RED,
                linespacing=1.35, bbox=bbox, zorder=6,
                arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 1.1, "shrinkB": 5})
    ax.annotate("хвост переписки правее:\nложные тревоги",
                xy=(1.8, 3), xytext=(4.4, 160), fontsize=10.2, color=BLUE,
                linespacing=1.35, bbox=bbox, zorder=6,
                arrowprops={"arrowstyle": "-|>", "color": BLUE, "lw": 1.1, "shrinkB": 5})
    ax.set_xlabel("счёт фильтра $s$")
    ax.set_ylabel("сообщений (лог-шкала)")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("Два распределения счёта на 1574 отложенных сообщениях",
                 loc="left", fontweight="bold", pad=12)
    save(fig, OUT / "score-histograms.png")


# ---------------------------------------------------------------- figure 7.3
def confusion_matrix() -> None:
    tp = sum(1 for lab, s, _ in TEST_SCORES if lab == "spam" and sigmoid(s) >= 0.5)
    fn = sum(1 for lab, s, _ in TEST_SCORES if lab == "spam" and sigmoid(s) < 0.5)
    fp = sum(1 for lab, s, _ in TEST_SCORES if lab == "ham" and sigmoid(s) >= 0.5)
    tn = sum(1 for lab, s, _ in TEST_SCORES if lab == "ham" and sigmoid(s) < 0.5)

    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    ax.set_xlim(0, 9.6)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    cells = [
        (2.2, 3.3, str(tp), "пойманный спам (TP)", GREEN, WASH),
        (5.7, 3.3, str(fn), "пропуски (FN):\nдойдут до глаз", RED, PAPER),
        (2.2, 0.7, str(fp), "ложные тревоги (FP):\nсамые дорогие", RED, PAPER),
        (5.7, 0.7, str(tn), "тихая норма (TN)", GREEN, WASH),
    ]
    for x, y, num, label, color, face in cells:
        rect = mpl.patches.FancyBboxPatch(
            (x, y), 3.1, 2.1, boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=face, edgecolor=color, linewidth=1.8)
        ax.add_patch(rect)
        ax.text(x + 1.55, y + 1.42, num, ha="center", fontsize=25,
                fontweight="bold", color=color)
        ax.text(x + 1.55, y + 0.62, label, ha="center", fontsize=10.6,
                color=MUTED, linespacing=1.35)

    ax.text(3.75, 5.75, "предсказано: спам", ha="center", fontsize=12.6,
            fontweight="bold")
    ax.text(7.25, 5.75, "предсказано: переписка", ha="center", fontsize=12.6,
            fontweight="bold")
    ax.text(1.35, 4.35, "на деле\nспам", ha="center", fontsize=12.2,
            fontweight="bold", linespacing=1.35)
    ax.text(1.35, 1.75, "на деле\nпереписка", ha="center", fontsize=12.2,
            fontweight="bold", linespacing=1.35)
    fig.text(0.05, 0.94, "Матрица ошибок при пороге 0,5: четыре судьбы 1574 сообщений",
             fontsize=15.5, fontweight="bold")
    save(fig, OUT / "confusion-matrix.png")


# ---------------------------------------------------------------- figure 7.4
def pr_curve() -> None:
    fig, ax = plt.subplots(figsize=(9.8, 6.6))
    spam_total = sum(1 for lab, _, _ in TEST_SCORES if lab == "spam")
    precisions, recalls = [], []
    marks = {}
    for thr in np.arange(0.02, 0.995, 0.005):
        tp = sum(1 for lab, s, _ in TEST_SCORES if lab == "spam" and sigmoid(s) >= thr)
        fp = sum(1 for lab, s, _ in TEST_SCORES if lab == "ham" and sigmoid(s) >= thr)
        if tp + fp == 0:
            continue
        precisions.append(tp / (tp + fp))
        recalls.append(tp / spam_total)
        for target in (0.3, 0.5, 0.7, 0.9):
            if abs(thr - target) < 0.0026:
                marks[target] = (tp / spam_total, tp / (tp + fp))
    ax.plot(recalls, precisions, color=BLUE, lw=2.2, zorder=3)
    offsets = {0.3: (12, -18), 0.5: (12, -4), 0.7: (12, 10), 0.9: (-14, 12)}
    for thr, (r, p) in marks.items():
        ax.plot(r, p, "o", color=RED, ms=6, zorder=4)
        thr_str = f"{thr:.1f}".replace(".", ",")
        ax.annotate(f"порог {thr_str}", xy=(r, p), xytext=offsets[thr],
                    textcoords="offset points", fontsize=10.4, color=RED)
    ax.set_xlabel("полнота: доля пойманного спама")
    ax.set_ylabel("точность: доля правды в криках «спам»")
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.5, 1.02)
    ax.grid(color=GRID, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Кривая «точность против полноты» нашего фильтра",
                 loc="left", fontweight="bold", pad=12)
    save(fig, OUT / "pr-curve.png")


# ---------------------------------------------------------------- figure 7.5
def calibration() -> None:
    bins = [[0, 0, 0.0] for _ in range(5)]
    for lab, s, _ in TEST_SCORES:
        p = sigmoid(s)
        i = min(4, int(p * 5))
        bins[i][0] += 1
        bins[i][1] += lab == "spam"
        bins[i][2] += p

    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    centers = [0.1 + 0.2 * i for i in range(5)]
    fractions = [k / n if n else 0 for n, k, _ in bins]
    ax.plot([0, 1], [0, 1], color=FAINT, lw=1.4, linestyle=(0, (5, 4)),
            label="идеальная калибровка")
    ax.bar(centers, fractions, width=0.16, color=BLUE, alpha=0.85, zorder=3,
           label="фактическая доля спама")
    for c, f, (n, _, _) in zip(centers, fractions, bins):
        f_str = f"{f:.2f}".replace(".", ",")
        ax.text(c, f + 0.03, f_str, ha="center", fontsize=10.6, fontweight="bold")
        ax.text(c, -0.075, f"n={n}", ha="center", fontsize=9.8, color=MUTED)
    ax.set_xlabel("предсказанная вероятность спама (корзины)")
    ax.set_ylabel("фактическая доля спама в корзине")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.11, 1.1)
    ax.set_xticks(centers, ["0–0,2", "0,2–0,4", "0,4–0,6", "0,6–0,8", "0,8–1"])
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("Диаграмма надёжности: обещанное против сбывшегося",
                 loc="left", fontweight="bold", pad=12)
    save(fig, OUT / "calibration.png")


# ---------------------------------------------------------------- figure 7.6
def refusal_zone() -> None:
    fig, ax = plt.subplots(figsize=(12.8, 5.2))
    ham = [s for lab, s, _ in TEST_SCORES if lab == "ham"]
    spam = [s for lab, s, _ in TEST_SCORES if lab == "spam"]
    lo, hi = math.log(0.3 / 0.7), math.log(0.9 / 0.1)
    bins = np.arange(-8, 13, 0.75)
    ax.hist(ham, bins=bins, color=BLUE, alpha=0.7)
    ax.hist(spam, bins=bins, color=RED, alpha=0.7)
    ax.set_yscale("log")
    ax.axvspan(lo, hi, color=GOLD, alpha=0.22, zorder=0)
    ax.axvline(lo, color=GOLD, lw=1.6)
    ax.axvline(hi, color=GOLD, lw=1.6)

    in_zone = sum(1 for _, s, _ in TEST_SCORES if lo <= s <= hi)
    err_in = sum(
        1 for lab, s, _ in TEST_SCORES
        if lo <= s <= hi and ((s >= 0 and lab == "ham") or (s < 0 and lab == "spam"))
    )
    ax.text((lo + hi) / 2, 700,
            f"зона отказа\n{in_zone} сообщения, {err_in} ошибок порога 0,5",
            ha="center", va="top", fontsize=10.8, color=INK, linespacing=1.4)
    ax.text(-7.6, 700, "уверенная переписка:\nрешаем автоматически",
            fontsize=10.2, color=BLUE, va="top", linespacing=1.35)
    ax.text(12.4, 700, "уверенный спам:\nрешаем автоматически",
            fontsize=10.2, color=RED, va="top", ha="right", linespacing=1.35)
    ax.set_xlabel("счёт фильтра $s$; границы зоны — $\\hat p=0{,}3$ и $\\hat p=0{,}9$")
    ax.set_ylabel("сообщений (лог-шкала)")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Зона отказа: два процента потока, треть всех ошибок",
                 loc="left", fontweight="bold", pad=12)
    save(fig, OUT / "refusal-zone.png")


# ------------------------------------------------------------- margin schemes
def length_hist() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True)
    for ax, lab, color, title in [
        (axes[0], "ham", BLUE, "переписка"),
        (axes[1], "spam", RED, "спам"),
    ]:
        lengths = [len(t) for l, t in ROWS if l == lab]
        ax.hist(lengths, bins=np.arange(0, 320, 10), color=color, alpha=0.85)
        ax.set_yticks([])
        ax.set_title(title, loc="left", fontsize=10.8, pad=4, color=color)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
    axes[1].axvline(160, color=INK, lw=1.2, linestyle=(0, (4, 3)))
    axes[1].text(163, axes[1].get_ylim()[1] * 0.75, "лимит SMS", fontsize=9.4)
    axes[1].set_xlabel("длина, символов")
    fig.subplots_adjust(hspace=0.45)
    save(fig, SIDE / "length-hist.png")


def weight_ladder() -> None:
    names = ["фон", "длина", "цифры", "free", "call", "txt", "www", "win"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    order = np.argsort(W)
    colors = [RED if W[i] < 0 else GREEN for i in order]
    ax.barh(range(8), [W[i] for i in order], color=colors, height=0.62)
    ax.set_yticks(range(8), [names[i] for i in order], fontsize=10.5)
    ax.axvline(0, color=LINE, lw=1)
    for pos, i in enumerate(order):
        v = W[i]
        v_str = f"{v:+.2f}".replace(".", ",").replace("+", "")
        ax.text(v + (0.18 if v >= 0 else -0.18), pos, v_str,
                va="center", ha="left" if v >= 0 else "right", fontsize=9.8)
    ax.set_xlim(-6.2, 7.6)
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    save(fig, SIDE / "weight-ladder.png")


def spam_drift() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    spam = np.array([s for lab, s, _ in TEST_SCORES if lab == "spam"])
    bins = np.arange(-6, 13, 0.9)
    ax.hist(spam, bins=bins, color=RED, alpha=0.75, label="спам сегодня")
    ax.hist(spam - 3.2, bins=bins, color=GOLD, alpha=0.6, label="после адаптации")
    ax.axvline(0, color=INK, lw=1.5, linestyle=(0, (5, 4)))
    ax.text(0.25, ax.get_ylim()[1] * 0.93, "порог", fontsize=9.8)
    ax.legend(loc="upper right", frameon=False, fontsize=9.6)
    ax.set_yticks([])
    ax.set_xlabel("счёт фильтра")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    save(fig, SIDE / "spam-drift.png")


if __name__ == "__main__":
    feature_plane()
    score_histograms()
    confusion_matrix()
    pr_curve()
    calibration()
    refusal_zone()
    length_hist()
    weight_ladder()
    spam_drift()
    print("lesson 07 figures written; weights:", [round(v, 2) for v in W])
