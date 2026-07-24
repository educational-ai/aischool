"""Deterministic figures for lesson 32: honest evaluation (train/val/test).

The overfitting curve (train accuracy climbs to 100% while test plateaus,
the gap grows), the three data splits with the forbidden "tune on test"
arrow, and the learning curve. All on real handwritten digits. Numbers
reproduced and asserted.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "32"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "32"

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


def data():
    D = load_digits(); X = D.data / 16.0; y = D.target
    return train_test_split(X, y, test_size=0.3, random_state=32, stratify=y)


# ---------------------------- fig 32.1: overfitting curve
def fig_overfitting() -> None:
    Xtr, Xte, ytr, yte = data()
    depths = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20]
    tr, te = [], []
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
        tr.append(clf.score(Xtr, ytr)); te.append(clf.score(Xte, yte))
    tr, te = np.array(tr), np.array(te)
    best = int(np.argmax(te))
    print(f"overfit: max train={tr.max():.3f}, max test={te.max():.3f}, gap@deep={tr[-1]-te[-1]:.3f}")
    assert tr[-1] > 0.99 and abs(te[-1] - 0.82) < 0.03
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(depths, tr, color=BLUE, lw=2.2, marker="o", markersize=4, label="на обучении (train)")
    ax.plot(depths, te, color=RED, lw=2.2, marker="s", markersize=4, label="на тесте (test)")
    ax.fill_between(depths, te, tr, color=RED, alpha=0.07)
    # gap annotation at the deep end
    ax.annotate("", xy=(depths[-1], tr[-1]), xytext=(depths[-1], te[-1]),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.2))
    ax.text(depths[-1] - 0.4, (tr[-1] + te[-1]) / 2, f"разрыв\n{(tr[-1]-te[-1])*100:.0f}%",
            ha="right", va="center", fontsize=10.5, color=MUTED)
    ax.axvline(depths[best], color=GREEN, lw=1.2, ls=(0, (4, 3)))
    ax.text(depths[best] + 0.3, 0.45, "лучшая\nсложность", color=GREEN, fontsize=10)
    ax.text(2, 0.30, "недообучение:\nмодель слишком проста", fontsize=9.5, color=MUTED)
    ax.text(13, 0.30, "переобучение:\nзаучила обучающую", fontsize=9.5, color=RED, ha="center")
    ax.set_xlabel("сложность модели (глубина дерева)")
    ax.set_ylabel("точность")
    ax.set_ylim(0.1, 1.05)
    ax.set_title("Дерево заучивает обучающую выборку, но не обобщает")
    ax.legend(loc="center right", frameon=False, fontsize=11)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "overfitting.png")


# ---------------------------- fig 32.2: three splits + forbidden arrow
def fig_splits() -> None:
    fig, ax = plt.subplots(figsize=(10.0, 4.4))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 6)
    parts = [("train", 0.5, 5.0, BLUE, "модель учится\n(подбор весов)"),
             ("validation", 6.0, 3.0, GOLD, "настройка\n(выбор модели, порогов)"),
             ("test", 9.6, 1.6, GREEN, "один раз в конце\n(честная оценка)")]
    for lab, x, w, col, desc in parts:
        ax.add_patch(Rectangle((x, 3.2), w, 1.4, fc=WASH, ec=col, lw=2.0))
        ax.text(x + w / 2, 3.9, lab, ha="center", va="center", fontsize=12, color=col, fontweight="bold")
        ax.text(x + w / 2, 2.4, desc, ha="center", va="top", fontsize=9, color=MUTED)
    # forbidden arrow: tuning on test
    ax.add_patch(FancyArrowPatch((10.4, 4.9), (3.0, 4.9), connectionstyle="arc3,rad=0.35",
                                 arrowstyle="-|>", color=RED, lw=1.8, mutation_scale=13))
    ax.text(6.7, 5.75, "подсмотреть в тест и подстроиться — запрещено (утечка)",
            ha="center", fontsize=10.5, color=RED)
    ax.plot([6.4, 7.0], [5.35, 5.05], color=RED, lw=2.2)  # a small "no" slash over the arrow
    ax.plot([6.4, 7.0], [5.05, 5.35], color=RED, lw=2.2)
    ax.text(6, 0.7, "Данные делят по будущей встрече: тест — это те, кого модель ещё не видела.",
            ha="center", fontsize=10, color=INK)
    ax.set_title("Три роли данных: учить, настраивать, судить", y=1.0)
    save(fig, OUT / "splits.png")
    print("splits drawn")


# ---------------------------- fig 32.3: learning curve
def fig_learning() -> None:
    Xtr, Xte, ytr, yte = data()
    ns = [40, 62, 90, 125, 180, 251, 400, 628, 900, 1257]
    te = []
    for n in ns:
        clf = LogisticRegression(max_iter=2000).fit(Xtr[:n], ytr[:n])
        te.append(clf.score(Xte, yte))
    te = np.array(te)
    print(f"learning: n=62 acc={te[1]:.3f}, n=1257 acc={te[-1]:.3f}")
    assert te[-1] > te[0] and abs(te[-1] - 0.957) < 0.02
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(ns, te, color=GREEN, lw=2.2, marker="o", markersize=4)
    ax.fill_between(ns, 0.8, te, color=GREEN, alpha=0.06)
    ax.set_xlabel("размер обучающей выборки")
    ax.set_ylabel("точность на тесте")
    ax.set_ylim(0.8, 1.0)
    ax.set_title("Больше данных — лучше обобщение")
    ax.text(700, 0.9, "кривая растёт и\nвыходит на плато:\nданные важнее\nхитрой модели",
            fontsize=10, color=MUTED)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "learning.png")


# ------------------------------------------------ margins
def side_group() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.text(5, 5.4, "строки одного пациента", ha="center", fontsize=9.5, color=INK)
    # wrong: split rows across
    for i, col in enumerate([BLUE, GREEN, BLUE, GREEN]):
        ax.add_patch(Rectangle((1 + i * 0.7, 3.4), 0.6, 0.6, fc=col, ec=LINE, lw=0.6))
    ax.text(2.4, 2.9, "врозь — утечка", ha="center", fontsize=8, color=RED)
    for i in range(4):
        ax.add_patch(Rectangle((6 + i * 0.7, 3.4), 0.6, 0.6, fc=BLUE, ec=LINE, lw=0.6))
    ax.text(7.05, 2.9, "вместе — честно", ha="center", fontsize=8, color=GREEN)
    ax.set_title("группу — целиком в одну часть", fontsize=10.5)
    save(fig, SIDE / "group.png")


def side_time() -> None:
    fig, ax = plt.subplots(figsize=(4.1, 2.4))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4)
    ax.add_patch(FancyArrowPatch((0.5, 1.8), (9.5, 1.8), arrowstyle="-|>", color=MUTED, lw=1.4, mutation_scale=12))
    ax.text(9.5, 1.2, "время", fontsize=9, color=MUTED, ha="right")
    ax.add_patch(Rectangle((0.6, 2.1), 5.5, 0.9, fc="#e7eef4", ec=BLUE, lw=1.4))
    ax.text(3.35, 2.55, "train (прошлое)", ha="center", fontsize=9, color=BLUE)
    ax.add_patch(Rectangle((6.3, 2.1), 3.1, 0.9, fc="#eef4ef", ec=GREEN, lw=1.4))
    ax.text(7.85, 2.55, "test (будущее)", ha="center", fontsize=9, color=GREEN)
    ax.set_title("время нельзя перемешивать", fontsize=10.5)
    save(fig, SIDE / "time.png")


def side_doubledescent() -> None:
    fig, ax = plt.subplots(figsize=(4.1, 2.6))
    x = np.linspace(0, 10, 400)
    interp = 5.5
    # classic U (first descent + rise) blended with a peak at interpolation, then second descent
    classic = 0.35 + 0.9 * (x - 2.3) ** 2 / 9        # U with min near x=2.3
    peak = 1.1 * np.exp(-((x - interp) ** 2) / 0.7)    # sharp spike at interpolation
    second = np.where(x > interp, 0.6 * (1 - np.exp(-(x - interp) / 1.6)), 0)  # descent after
    y = np.clip(classic, 0, 1.2) + peak - second
    ax.plot(x, y, color=VIOLET, lw=2.0)
    ax.axvline(interp, color=LINE, lw=1.0, ls=(0, (3, 2)))
    ax.text(interp + 0.2, y.max() * 0.9, "порог\nинтерполяции", fontsize=8, color=MUTED)
    ax.annotate("классика", xy=(2.3, 0.4), xytext=(0.4, 1.1), fontsize=8, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=LINE, lw=0.8))
    ax.annotate("второй спуск", xy=(9, y[-1]), xytext=(6.2, 0.3), fontsize=8, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=LINE, lw=0.8))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(0, y.max() * 1.15)
    ax.set_xlabel("сложность модели", fontsize=9); ax.set_ylabel("ошибка на тесте", fontsize=9)
    ax.set_title("двойной спуск", fontsize=10.5)
    save(fig, SIDE / "doubledescent.png")


fig_overfitting()
fig_splits()
fig_learning()
side_group()
side_time()
side_doubledescent()
print("lesson 32 figures written")
