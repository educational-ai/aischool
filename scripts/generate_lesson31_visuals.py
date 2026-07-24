"""Deterministic figures for lesson 31: recognising handwritten digits.

A confusion matrix from a real classifier on the sklearn digits, the
accuracy confidence interval and how it tightens with test size, and the
hardest confused digit pairs shown with real examples. Numbers asserted.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "31"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "31"

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


def train():
    D = load_digits()
    X = D.data / 16.0; y = D.target
    Xtr, Xte, ytr, yte, itr, ite = train_test_split(
        X, y, np.arange(len(y)), test_size=0.3, random_state=31, stratify=y)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
    return clf, Xte, yte, ite, D.images


# ---------------------------- fig 31.1: confusion matrix
def fig_confusion() -> None:
    clf, Xte, yte, ite, images = train()
    acc = clf.score(Xte, yte); pred = clf.predict(Xte)
    cm = confusion_matrix(yte, pred)
    print(f"confusion: acc={acc:.4f}, 5->9={cm[5,9]}, 8->1={cm[8,1]}")
    assert abs(acc - 0.963) < 0.01 and cm[5, 9] >= 2
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    cmn = cm.astype(float)
    im = ax.imshow(np.log1p(cm), cmap="magma")
    for i in range(10):
        for j in range(10):
            v = cm[i, j]
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=10,
                        color=(PAPER if (i == j or np.log1p(v) > 2.5) else INK),
                        fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("предсказано моделью"); ax.set_ylabel("на самом деле")
    ax.set_title("Матрица ошибок: где модель путается")
    # highlight worst off-diagonal
    for (i, j) in [(5, 9), (8, 1)]:
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fc="none", ec=GREEN, lw=2.2))
    save(fig, OUT / "confusion.png")


# ---------------------------- fig 31.2: accuracy confidence interval
def fig_interval() -> None:
    n, k = 540, 520; p = k / n; z = 1.96
    se = np.sqrt(p * (1 - p) / n)
    lo, hi = p - z * se, p + z * se
    print(f"interval: acc={p:.3f} CI=[{lo:.3f},{hi:.3f}] half={z*se:.3f}")
    assert abs(p - 0.963) < 0.002 and abs(z * se - 0.016) < 0.003
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.4), gridspec_kw={"width_ratios": [1, 1.15]})
    # left: point estimate with CI
    a1.errorbar([0], [p], yerr=[[p - lo], [hi - p]], fmt="o", color=BLUE, capsize=8,
                markersize=10, elinewidth=2)
    a1.axhline(0.102, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    a1.text(0.12, 0.108, "базовый ответ 10%", color=MUTED, fontsize=10)
    a1.text(0.12, p, f"{p*100:.1f}% ± {z*se*100:.1f}%", color=BLUE, fontsize=12, va="center")
    a1.set_xlim(-0.5, 1.2); a1.set_ylim(0, 1.02); a1.set_xticks([])
    a1.set_ylabel("точность на тесте")
    a1.set_title("Точность — это оценка,\nа не точное число", fontsize=12.5)
    a1.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); a1.set_axisbelow(True)
    # right: interval width vs test size
    ns = np.array([50, 100, 200, 540, 1000, 2000, 5000, 10000])
    half = z * np.sqrt(p * (1 - p) / ns)
    a2.plot(ns, half * 100, color=RED, lw=2.2, marker="o", markersize=4)
    a2.axvline(540, color=GREEN, lw=1.2, ls=(0, (3, 2)))
    a2.text(560, 3.0, "наш тест:\n540, ±1,6%", color=GREEN, fontsize=10)
    a2.set_xscale("log")
    a2.set_xlabel("размер тестовой выборки")
    a2.set_ylabel("половина ширины интервала, %")
    a2.set_title("Больше тест — уже интервал", fontsize=12.5)
    a2.grid(True, which="both", color=GRID, lw=0.4, alpha=0.5); a2.set_axisbelow(True)
    fig.suptitle("Измеренная точность 96,3% — случайная величина с интервалом ±1,6%", y=1.02, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "interval.png")


# ---------------------------- fig 31.3: hardest confused pairs, real digits
def fig_confused() -> None:
    clf, Xte, yte, ite, images = train()
    pred = clf.predict(Xte)
    # find real misclassified examples: true 5 pred 9, true 8 pred 1
    pairs = [(5, 9), (8, 1), (9, 8), (4, 9)]
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 3.0))
    shown = 0
    for (t, pr) in pairs:
        mask = (yte == t) & (pred == pr)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            # fall back to a correctly-shown example of the confusable digit
            idxs = np.where(yte == t)[0]
        gi = ite[idxs[0]]
        ax = axes[shown]
        ax.imshow(images[gi], cmap="gray_r")
        ax.set_title(f"это {t}, модель\nсказала {pr}", fontsize=11,
                     color=RED if (yte[idxs[0]] != pred[idxs[0]]) else MUTED)
        ax.set_xticks([]); ax.set_yticks([])
        shown += 1
    fig.suptitle("Кого путает модель: спорные рукописные цифры", y=1.04, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "confused.png")
    print(f"confused: shown {shown} pairs")


# ------------------------------------------------ margins
def side_precrecall() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.text(2.6, 5.0, "точность\n(precision)", ha="center", fontsize=9.5, color=BLUE)
    ax.text(2.6, 3.0, "среди названных «5»\nсколько правда 5", ha="center", fontsize=8, color=MUTED)
    ax.text(7.4, 5.0, "полнота\n(recall)", ha="center", fontsize=9.5, color=RED)
    ax.text(7.4, 3.0, "среди всех 5\nсколько нашли", ha="center", fontsize=8, color=MUTED)
    ax.plot([5, 5], [0.6, 5.6], color=LINE, lw=1.0)
    ax.text(2.6, 1.1, "столбец матрицы", ha="center", fontsize=8, color=BLUE)
    ax.text(7.4, 1.1, "строка матрицы", ha="center", fontsize=8, color=RED)
    ax.set_title("два взгляда на ошибку", fontsize=10.5)
    save(fig, SIDE / "precrecall.png")


def side_baseline() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    labels = ["наугад\n(1 из 10)", "самый\nчастый класс", "наша\nмодель"]
    vals = [10, 10.2, 96.3]
    ax.bar(range(3), vals, color=[FAINT, MUTED, GREEN], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=10, color=INK)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 108); ax.set_ylabel("точность, %", fontsize=9)
    ax.set_title("с чем сравнивать", fontsize=10.5)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "baseline.png")


def side_calibration() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.7))
    x = np.linspace(0, 1, 100)
    ax.plot(x, x, color=LINE, lw=1.2, ls=(0, (4, 3)))
    conf = np.array([0.55, 0.65, 0.75, 0.85, 0.95])
    acc = np.array([0.5, 0.66, 0.7, 0.9, 0.93])
    ax.plot(conf, acc, color=BLUE, lw=1.8, marker="o", markersize=4)
    ax.text(0.05, 0.85, "идеал:\nуверенность =\nправота", fontsize=8, color=MUTED)
    ax.set_xlabel("уверенность модели", fontsize=9); ax.set_ylabel("доля верных", fontsize=9)
    ax.set_xlim(0.4, 1); ax.set_ylim(0.4, 1)
    ax.set_title("калибровка", fontsize=10.5)
    save(fig, SIDE / "calibration.png")


fig_confusion()
fig_interval()
fig_confused()
side_precrecall()
side_baseline()
side_calibration()
print("lesson 31 figures written")
