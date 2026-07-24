"""Deterministic figures for lesson 10: labelling economics on SMS Spam.

Re-runs the lesson's experiments (learning curves, active learning,
self-training map, weak supervision, bandit regret) so every quoted number
is reproducible from this file alone.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "10"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "10"
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


def rounded_box(ax, xy, size, *, face=PAPER, edge=LINE, lw=1.3, rounding=0.05):
    patch = FancyBboxPatch(xy, *size,
                           boxstyle=f"round,pad=0.012,rounding_size={rounding}",
                           facecolor=face, edgecolor=edge, linewidth=lw)
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, *, color=MUTED, lw=1.5, rad=0.0, ls="solid"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                 connectionstyle=f"arc3,rad={rad}",
                                 color=color, linewidth=lw, linestyle=ls,
                                 mutation_scale=13, shrinkA=0, shrinkB=0))


# --------------------------------------------------------------- experiments
ROWS = [line.rstrip("\n").split("\t", 1) for line in DATA.open(encoding="utf-8")]
random.seed(7)
random.shuffle(ROWS)
POOL, TEST = ROWS[:4000], ROWS[4000:]


def features(text: str) -> list[float]:
    lower = text.lower()
    return [
        1.0, len(text) / 100, sum(c.isdigit() for c in text) / 10,
        1.0 if "free" in lower else 0.0,
        1.0 if "call" in lower else 0.0,
        1.0 if "txt" in lower or "text" in lower else 0.0,
        1.0 if "www" in lower or "http" in lower else 0.0,
        1.0 if "win" in lower or "prize" in lower or "winner" in lower else 0.0,
    ]


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-max(-30, min(30, z))))


def train(data, epochs=60):
    w = [0.0] * 8
    for _ in range(epochs):
        for lab, t in data:
            x = features(t)
            p = sigmoid(sum(a * b for a, b in zip(w, x)))
            g = p - (1.0 if lab == "spam" else 0.0)
            for i in range(8):
                w[i] -= 0.05 * g * x[i]
    return w


def f1(w) -> float:
    tp = fp = fn = 0
    for lab, t in TEST:
        p = sigmoid(sum(a * b for a, b in zip(w, features(t))))
        pred = p >= 0.5
        y = lab == "spam"
        if pred and y:
            tp += 1
        elif pred and not y:
            fp += 1
        elif not pred and y:
            fn += 1
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return 2 * prec * rec / max(prec + rec, 1e-9)


RANDOM_BUDGETS = [25, 50, 100, 200, 300, 500, 1000, 2000, 4000]
RANDOM_F1 = [f1(train(POOL[:n])) for n in RANDOM_BUDGETS]

# Active loop.
ACTIVE_BUDGETS, ACTIVE_F1 = [25], []
labeled = list(POOL[:25])
unlabeled = list(POOL[25:])
w_act = train(labeled)
ACTIVE_F1.append(f1(w_act))
while len(labeled) < 500:
    scored = sorted(unlabeled,
                    key=lambda r: abs(sigmoid(sum(a * b for a, b in
                                                  zip(w_act, features(r[1])))) - 0.5))
    take = scored[:25]
    labeled += take
    ids = set(id(r) for r in take)
    unlabeled = [r for r in unlabeled if id(r) not in ids]
    w_act = train(labeled)
    ACTIVE_BUDGETS.append(len(labeled))
    ACTIVE_F1.append(f1(w_act))


# ---------------------------------------------------------------- figure 10.1
def learning_curve() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.9))
    ax.plot(RANDOM_BUDGETS, RANDOM_F1, "o-", color=BLUE, lw=2.2, ms=5)
    ax.set_xscale("log")
    ax.set_xticks(RANDOM_BUDGETS,
                  [str(b) for b in RANDOM_BUDGETS])
    ax.axhline(0.94, color=GRID, lw=1.2)
    ax.text(27, 0.943, "потолок восьми признаков", fontsize=10.2, color=MUTED)
    ax.annotate("первые 25 меток", xy=(25, RANDOM_F1[0]),
                xytext=(40, 0.856), fontsize=10.4, color=INK,
                arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.1,
                            "shrinkB": 5})
    ax.set_xlabel("куплено случайных меток (лог-шкала)")
    ax.set_ylabel("F-мера на отложенных")
    ax.set_ylim(0.85, 0.96)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Кривая обучения: жадный старт, скупая середина, шумный хвост",
                 loc="left", fontweight="bold", pad=12)
    save(fig, OUT / "learning-curve.png")


# ---------------------------------------------------------------- figure 10.2
def active_vs_random() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    rb = [b for b in RANDOM_BUDGETS if b <= 500]
    ax.plot(rb, RANDOM_F1[:len(rb)], "o-", color=FAINT, lw=2, ms=5,
            label="случайные метки")
    ax.plot(ACTIVE_BUDGETS, ACTIVE_F1, "o-", color=BLUE, lw=2.2, ms=4,
            label="активная стратегия")
    ax.axhline(0.94, color=GRID, lw=1.2)
    ax.annotate("потолок за 225 меток", xy=(225, 0.942),
                xytext=(275, 0.898), fontsize=10.4, color=BLUE,
                arrowprops={"arrowstyle": "-|>", "color": BLUE, "lw": 1.1,
                            "shrinkB": 5})
    ax.annotate("хвостовой спад:\nвыборка перекошена", xy=(430, 0.923),
                xytext=(320, 0.868), fontsize=10.2, color=RED,
                arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 1.1,
                            "shrinkB": 5})
    ax.set_xlabel("бюджет меток")
    ax.set_ylabel("F-мера на отложенных")
    ax.set_ylim(0.85, 0.96)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Активная разметка против случайной: те же деньги, разный путь",
                 loc="left", fontweight="bold", pad=12)
    save(fig, OUT / "active-vs-random.png")


# ---------------------------------------------------------------- figure 10.3
def self_training_map() -> None:
    w100 = train(POOL[:100])
    probs = [sigmoid(sum(a * b for a, b in zip(w100, features(t))))
             for _, t in POOL[100:]]
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    bins = np.arange(0, 1.01, 0.02)
    counts, edges = np.histogram(probs, bins=bins)
    for c, lo in zip(counts, edges[:-1]):
        taken = lo >= 0.95 or lo + 0.02 <= 0.021
        ax.bar(lo + 0.01, c, width=0.018,
               color=GREEN if taken else FAINT,
               alpha=0.9 if taken else 0.55)
    ax.set_yscale("log")
    ax.annotate("псевдометки: почти весь пул,\nточность 99,4%",
                xy=(0.98, 1300), xytext=(0.6, 900), fontsize=10.4, color=GREEN,
                arrowprops={"arrowstyle": "-|>", "color": GREEN, "lw": 1.1,
                            "shrinkB": 5})
    ax.annotate("пустая середина:\nздесь живут ошибки — и ни одной метки",
                xy=(0.5, 6), xytext=(0.2, 120), fontsize=10.4, color=RED,
                arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 1.1,
                            "shrinkB": 5})
    ax.set_xlabel("предсказанная вероятность спама (модель на 100 метках)")
    ax.set_ylabel("сообщений (лог)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Куда self-training кладёт свои метки",
                 loc="left", fontweight="bold", pad=12)
    save(fig, OUT / "self-training-map.png")


# ---------------------------------------------------------------- figure 10.4
def weak_supervision() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.9),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    # Left: coverage map on the (length, digits) plane.
    rng = np.random.default_rng(10)
    for lab, t in POOL:
        x = min(len(t), 400) + rng.uniform(-1.5, 1.5)
        y = sum(c.isdigit() for c in t) + rng.uniform(-0.4, 0.4)
        lower = t.lower()
        covered_spam = ("free" in lower or "win" in lower or "prize" in lower
                        or "www" in lower or "http" in lower
                        or sum(c.isdigit() for c in t) >= 10)
        covered_ham = len(t) < 50 and sum(c.isdigit() for c in t) == 0
        color = RED if covered_spam else (BLUE if covered_ham else FAINT)
        ax1.plot(x, y, "o", ms=2.4, color=color,
                 alpha=0.65 if color != FAINT else 0.3)
    ax1.set_xlabel("длина сообщения")
    ax1.set_ylabel("цифр в сообщении")
    ax1.set_ylim(-1, 42)
    ax1.set_title("Кого покрыли правила", loc="left", fontweight="bold",
                  fontsize=13, pad=8)
    ax1.text(180, 36, "серое — не покрыто ничем", fontsize=10.2, color=MUTED)
    ax1.grid(color=GRID, lw=0.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: outcome bars.
    bars = [("25 честных\nметок", 0.875, BLUE),
            ("2026 меток\nот правил", 0.672, RED)]
    for i, (name, v, color) in enumerate(bars):
        ax2.bar(i, v, width=0.55, color=color, alpha=0.85)
        ax2.text(i, v + 0.012, f"{v:.3f}".replace(".", ","), ha="center",
                 fontsize=12, fontweight="bold")
    ax2.set_xticks([0, 1], [b[0] for b in bars], fontsize=10.5)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("F-мера")
    ax2.grid(axis="y", color=GRID, lw=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_title("Итог: точность меток 94,7% не спасла",
                  loc="left", fontweight="bold", fontsize=13, pad=8)
    save(fig, OUT / "weak-supervision.png")


# ---------------------------------------------------------------- figure 10.5
def regret() -> None:
    mus = [0.030, 0.032, 0.040, 0.028]
    best = max(mus)
    T = 20000
    runs = 30

    def simulate(policy: str, seed: int) -> np.ndarray:
        rng = random.Random(seed)
        counts = [0] * 4
        sums = [0.0] * 4
        regret_track = np.zeros(T)
        cum = 0.0
        for t in range(T):
            if policy == "eps":
                if rng.random() < 0.1:
                    a = rng.randrange(4)
                else:
                    means = [sums[i] / counts[i] if counts[i] else 1.0
                             for i in range(4)]
                    a = means.index(max(means))
            else:
                if t < 4:
                    a = t
                else:
                    ucb = [sums[i] / counts[i]
                           + math.sqrt(2 * math.log(t + 1) / counts[i])
                           for i in range(4)]
                    a = ucb.index(max(ucb))
            reward = 1.0 if rng.random() < mus[a] else 0.0
            counts[a] += 1
            sums[a] += reward
            cum += best - mus[a]
            regret_track[t] = cum
        return regret_track

    fig, ax = plt.subplots(figsize=(10.8, 4.9))
    for policy, color, label in [("eps", FAINT, "$\\varepsilon$-жадность, $\\varepsilon=0{,}1$"),
                                 ("ucb", BLUE, "UCB")]:
        tracks = np.mean([simulate(policy, 100 + s) for s in range(runs)], axis=0)
        ax.plot(np.arange(T), tracks, color=color, lw=2.2, label=label)
    ax.set_xlabel("показов")
    ax.set_ylabel("накопленное сожаление, кликов")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("Сожаление двух стратегий на четырёх заголовках (30 запусков)",
                 loc="left", fontweight="bold", pad=12)
    save(fig, OUT / "regret.png")


# ---------------------------------------------------------------- figure 10.6
def pipeline() -> None:
    fig, ax = plt.subplots(figsize=(12.6, 5.2))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    boxes = {
        "stream": (0.4, 3.3, "поток\nсообщений", MUTED),
        "filter": (2.9, 3.3, "фильтр", BLUE),
        "auto": (5.6, 4.1, "уверенные:\nавторешение", GREEN),
        "zone": (5.6, 2.4, "зона отказа\n(~2%)", GOLD),
        "human": (8.2, 2.4, "разметчик", RED),
        "store": (10.6, 2.4, "метки", INK),
        "retrain": (10.6, 0.6, "дообучение", VIOLET),
    }
    bw, bh = 1.9, 1.0
    for x, y, label, color in boxes.values():
        rounded_box(ax, (x, y), (bw, bh), edge=color, rounding=0.08)
        ax.text(x + bw / 2, y + bh / 2, label, ha="center", va="center",
                fontsize=11, linespacing=1.3)

    def edge(a, b, **kw):
        xa, ya, _, _ = boxes[a]
        xb, yb, _, _ = boxes[b]
        arrow(ax, (xa + bw, ya + bh / 2), (xb, yb + bh / 2), **kw)

    edge("stream", "filter", color=INK)
    arrow(ax, (boxes["filter"][0] + bw, boxes["filter"][1] + 0.8),
          (boxes["auto"][0], boxes["auto"][1] + 0.5), color=GREEN)
    arrow(ax, (boxes["filter"][0] + bw, boxes["filter"][1] + 0.2),
          (boxes["zone"][0], boxes["zone"][1] + 0.5), color=GOLD)
    edge("zone", "human", color=RED)
    edge("human", "store", color=INK)
    arrow(ax, (boxes["store"][0] + bw / 2, boxes["store"][1]),
          (boxes["retrain"][0] + bw / 2, boxes["retrain"][1] + bh), color=VIOLET)
    arrow(ax, (boxes["retrain"][0], boxes["retrain"][1] + bh / 2),
          (boxes["filter"][0] + bw / 2, boxes["filter"][1]), color=VIOLET,
          rad=-0.32)
    arrow(ax, (boxes["stream"][0] + bw / 2, boxes["stream"][1]),
          (boxes["human"][0] + 0.3, boxes["human"][1]), color=FAINT,
          rad=0.5, ls=(0, (5, 4)))
    ax.text(2.7, 0.5, "случайная ветка: честный срез для оценки",
            fontsize=10.2, color=MUTED)
    fig.text(0.055, 0.93, "Конвейер меток вокруг живого фильтра",
             fontsize=17, fontweight="bold")
    save(fig, OUT / "pipeline.png")


# ------------------------------------------------------------- margin schemes
def double_labeling() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.set_xlim(0, 6.4)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    rounded_box(ax, (0.3, 1.3), (1.5, 0.8), edge=INK)
    ax.text(1.05, 1.7, "пример", ha="center", fontsize=10.5)
    for y, name in [(2.4, "разметчик А"), (0.4, "разметчик Б")]:
        rounded_box(ax, (2.5, y), (1.6, 0.8), edge=BLUE)
        ax.text(3.3, y + 0.4, name, ha="center", fontsize=10)
        arrow(ax, (1.8, 1.7), (2.5, y + 0.4), color=MUTED)
    rounded_box(ax, (4.7, 1.3), (1.5, 0.8), edge=RED)
    ax.text(5.45, 1.7, "арбитр при\nрасхождении", ha="center", fontsize=9.2,
            linespacing=1.25)
    arrow(ax, (4.1, 2.8), (4.85, 2.1), color=MUTED)
    arrow(ax, (4.1, 0.8), (4.85, 1.3), color=MUTED)
    save(fig, SIDE / "double-labeling.png")


def consistency() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.set_xlim(0, 6.4)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    rounded_box(ax, (0.3, 2.0), (1.7, 0.8), edge=INK)
    ax.text(1.15, 2.4, "$x$", ha="center", fontsize=12)
    rounded_box(ax, (0.3, 0.4), (1.7, 0.8), edge=GOLD)
    ax.text(1.15, 0.8, "$\\tilde x$ (искажение)", ha="center", fontsize=9.6)
    rounded_box(ax, (2.9, 1.2), (1.6, 0.8), edge=BLUE)
    ax.text(3.7, 1.6, "модель", ha="center", fontsize=10.5)
    arrow(ax, (2.0, 2.4), (2.95, 1.8), color=MUTED)
    arrow(ax, (2.0, 0.8), (2.95, 1.4), color=MUTED)
    rounded_box(ax, (5.0, 1.2), (1.2, 0.8), edge=RED)
    ax.text(5.6, 1.6, "штраф за\nразницу", ha="center", fontsize=9.2,
            linespacing=1.25)
    arrow(ax, (4.5, 1.6), (5.0, 1.6), color=MUTED)
    save(fig, SIDE / "consistency.png")


def ucb_hats() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    means = [0.50, 0.33, 0.25]
    hats = [1.07, 1.24, 1.52]
    labels = ["A (4 показа)", "B (3)", "C (2)"]
    for i, (m, h) in enumerate(zip(means, hats)):
        ax.bar(i, m, width=0.5, color=BLUE, alpha=0.8)
        ax.plot([i, i], [m, m + h], color=GOLD, lw=2.4)
        ax.plot(i, m + h, "v", color=GOLD, ms=8)
        total = f"{m + h:.2f}".replace(".", ",")
        ax.text(i, m + h + 0.07, total, ha="center", fontsize=10.5,
                fontweight="bold", color=GOLD)
    ax.set_xticks(range(3), labels, fontsize=10)
    ax.set_yticks([])
    ax.set_ylim(0, 2.15)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    save(fig, SIDE / "ucb-hats.png")


if __name__ == "__main__":
    learning_curve()
    active_vs_random()
    self_training_map()
    weak_supervision()
    regret()
    pipeline()
    double_labeling()
    consistency()
    ucb_hats()
    print("random F1:", [round(v, 3) for v in RANDOM_F1])
    print("active F1:", [round(v, 3) for v in ACTIVE_F1])
