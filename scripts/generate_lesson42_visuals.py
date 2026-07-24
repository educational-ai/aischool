"""Deterministic figures for lesson 42: conditional probability and Bayes' rule.

Conditional probability as renormalization, the base-rate fallacy shown as a natural-
frequency array of 10 000 people (a 90%/95% test gives only 15% PPV at 1% prevalence),
and Bayes updating on the REAL SMS-spam corpus (posterior odds move word by word).
Numbers reproduced and asserted.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
SPAM = ROOT / "scripts" / "data" / "sms-spam-collection.tsv"
OUT = ROOT / "public" / "figures" / "lessons" / "42"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "42"

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


def spam_stats():
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


def lr(word, spam, ham, ws, wh):
    ps = (ws.get(word, 0) + 1) / (spam + 2)
    ph = (wh.get(word, 0) + 1) / (ham + 2)
    return ps / ph


# ---------------------------------------- fig 42.1: conditional as renormalization
def fig_renorm() -> None:
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.2, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    a0.set_aspect("equal"); a0.set_xlim(0, 10); a0.set_ylim(0, 8); a0.axis("off")
    a0.add_patch(Rectangle((0.3, 0.3), 9.4, 7.4, fc=WASH, ec=LINE, lw=1.4))
    a0.text(5, 7.9, "всё пространство исходов, площадь = 1", ha="center", fontsize=11, color=MUTED)
    a0.add_patch(Rectangle((1.0, 1.0), 4.5, 5.5, fc=BLUE, ec=BLUE, alpha=0.12, lw=1.5))
    a0.text(1.4, 6.0, "$A$", fontsize=15, color=BLUE)
    a0.add_patch(Rectangle((3.5, 2.0), 5.2, 4.2, fc=GOLD, ec=GOLD, alpha=0.14, lw=1.5))
    a0.text(8.2, 5.6, "$B$", fontsize=15, color=GOLD)
    a0.add_patch(Rectangle((3.5, 2.0), 2.0, 4.2, fc=GREEN, ec=GREEN, alpha=0.42, lw=0))
    a0.text(4.5, 4.0, "$A\\cap B$", fontsize=12, color=INK, ha="center")
    a0.set_title("совместные события")
    # right: B renormalized
    a1.set_aspect("equal"); a1.set_xlim(0, 6); a1.set_ylim(0, 8); a1.axis("off")
    a1.add_patch(Rectangle((0.5, 1.0), 5.0, 6.0, fc=GOLD, ec=GOLD, alpha=0.14, lw=1.5))
    a1.add_patch(Rectangle((0.5, 1.0), 1.92, 6.0, fc=GREEN, ec=GREEN, alpha=0.42, lw=0))
    a1.text(3.0, 7.4, "теперь всё пространство — это $B$", ha="center", fontsize=11, color=MUTED)
    a1.text(1.46, 4.0, "$A\\cap B$", fontsize=11, color=INK, ha="center")
    a1.text(3.0, 0.3, "$P(A\\mid B)=\\dfrac{P(A\\cap B)}{P(B)}$", ha="center", fontsize=13, color=INK)
    a1.set_title("условие перенормирует")
    # arrow between
    fig.tight_layout()
    save(fig, OUT / "renorm.png")
    print("renorm drawn")


# ---------------------------------------- fig 42.2: natural frequencies (base-rate fallacy)
def fig_natural_freq() -> None:
    prev, sens, spec = 0.01, 0.90, 0.95
    N = 10000
    sick = int(N * prev); healthy = N - sick
    tp = round(sick * sens); fn = sick - tp
    fp = round(healthy * (1 - spec)); tn = healthy - fp
    ppv = tp / (tp + fp)
    print(f"natural_freq: TP={tp} FP={fp} PPV={ppv:.3f}")
    assert tp == 90 and fp == 495 and abs(ppv - 0.1538) < 0.001
    # 100x100 grid, each cell a person; color by category
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    grid = np.zeros((100, 100), int)  # 0 healthy-neg, 1 healthy-pos(FP), 2 sick-pos(TP), 3 sick-neg(FN)
    order = [2] * tp + [3] * fn + [1] * fp + [0] * tn
    # place sick block first (top-left) so it is visible
    idx = 0
    cells = []
    for r in range(100):
        for c in range(100):
            cells.append((r, c))
    for k, (r, c) in enumerate(cells):
        grid[r, c] = order[k]
    cmap = {0: "#cdd7e0", 1: RED, 2: GREEN, 3: GOLD}   # healthy-neg light blue-grey so grid reads full
    for r in range(100):
        for c in range(100):
            ax.add_patch(Rectangle((c, 99 - r), 1, 1, fc=cmap[grid[r, c]], ec=PAPER, lw=0.05))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("10 000 человек: болезнь 1%, тест 90% / 95%")
    # legend
    labels = [(GREEN, f"больны и «+»  (истинно): {tp}"), (RED, f"здоровы, но «+» (ложно): {fp}"),
              (GOLD, f"больны, но «−»: {fn}"), (WASH, f"здоровы и «−»: {tn}")]
    for i, (c, lab) in enumerate(labels):
        ax.add_patch(Rectangle((2, -8 - i * 6), 4, 4, fc=c, ec=LINE, lw=0.6, clip_on=False))
        ax.text(8, -6 - i * 6, lab, fontsize=10, color=INK, va="center")
    ax.text(58, -14, f"после «+»: болезнь лишь у\n$90/(90+495)=15{{,}}4\\%$", fontsize=12, color=RED, va="center")
    save(fig, OUT / "natural_freq.png")


# ---------------------------------------- fig 42.3: Bayes updating on real spam
def fig_spam_update() -> None:
    spam, ham, ws, wh = spam_stats()
    prior = spam / (spam + ham)
    print(f"spam_update: prior spam {prior:.3f}")
    assert abs(prior - 0.134) < 0.002
    spam_words = ["free", "win", "txt", "claim"]
    ham_words = ["sorry", "love", "home", "lol"]
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for words, col, name in [(spam_words, RED, "спам-слова"), (ham_words, GREEN, "обычные слова")]:
        odds = prior / (1 - prior)
        probs = [prior]
        for w in words:
            odds *= lr(w, spam, ham, ws, wh)
            probs.append(odds / (1 + odds))
        ax.plot(range(len(probs)), probs, "o-", color=col, lw=2.2, markersize=7, label=name)
        for k, w in enumerate(words):
            lrv = lr(w, spam, ham, ws, wh)
            ax.annotate(f'«{w}»\n×{lrv:.0f}' if lrv >= 1 else f'«{w}»\n×{lrv:.2f}',
                        (k + 1, probs[k + 1]), fontsize=8.5, color=col, ha="center",
                        va="bottom", xytext=(0, 10), textcoords="offset points")
    ax.axhline(prior, color=MUTED, lw=1.0, ls=(0, (3, 3)))
    ax.text(0.05, prior + 0.03, f"начальная вероятность спама {prior:.2f}", fontsize=9.5, color=MUTED)
    ax.set_xlabel("сколько слов прочитано"); ax.set_ylabel("вероятность спама $P(\\text{спам}\\mid \\text{слова})$")
    ax.set_ylim(-0.05, 1.08); ax.set_xticks(range(5))
    ax.set_title("Байес на настоящих SMS: каждое слово двигает вероятность")
    ax.legend(loc="center right", frameon=False, fontsize=11)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "spam_update.png")


# ---------------------------------------- margins
def side_ppv() -> None:
    prev = np.linspace(0.001, 0.2, 200)
    sens = 0.90
    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    for spec, c in [(0.90, RED), (0.95, GOLD), (0.99, GREEN)]:
        ppv = sens * prev / (sens * prev + (1 - spec) * (1 - prev))
        ax.plot(prev * 100, ppv, color=c, lw=1.8, label=f"спец. {spec:.0%}")
    ax.set_xlabel("распространённость, %", fontsize=9); ax.set_ylabel("$P(D\\mid+)$", fontsize=9)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_title("чем реже болезнь, тем слабее «+»", fontsize=9)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "ppv.png")


def side_spam_words() -> None:
    spam, ham, ws, wh = spam_stats()
    words = ["claim", "txt", "win", "free", "call", "lol", "sorry", "love", "home"]
    lrs = [(w, lr(w, spam, ham, ws, wh)) for w in words]
    lrs.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ys = range(len(lrs))
    logs = [np.log10(v) for _, v in lrs]
    cols = [GREEN if v < 0 else RED for v in logs]
    ax.barh(list(ys), logs, color=cols, alpha=0.8)
    ax.set_yticks(list(ys)); ax.set_yticklabels([w for w, _ in lrs], fontsize=8)
    ax.axvline(0, color=INK, lw=1.0)
    ax.set_xlabel("$\\log_{10}$ отношения правдоподобий", fontsize=8.5)
    ax.set_title("слова-улики (реальные SMS)", fontsize=9)
    save(fig, SIDE / "spam_words.png")
    print("spam_words drawn")


def side_odds() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 1.9))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 3)
    for x, lab, col in [(0.4, "шансы\nдо", MUTED), (4.6, "×  правдо-\nподобие", GOLD), (9.0, "шансы\nпосле", RED)]:
        ax.add_patch(Rectangle((x, 0.9), 2.4, 1.3, fc=WASH, ec=col, lw=1.5))
        ax.text(x + 1.2, 1.55, lab, ha="center", va="center", fontsize=9, color=col)
    ax.add_patch(FancyArrowPatch((2.9, 1.55), (4.5, 1.55), arrowstyle="-|>", mutation_scale=11, color=MUTED, lw=1.3))
    ax.add_patch(FancyArrowPatch((7.1, 1.55), (8.9, 1.55), arrowstyle="-|>", mutation_scale=11, color=MUTED, lw=1.3))
    ax.set_title("Байес в шансах: наблюдение — множитель", fontsize=9.5)
    save(fig, SIDE / "odds.png")


fig_renorm()
fig_natural_freq()
fig_spam_update()
side_ppv()
side_spam_words()
side_odds()
print("lesson 42 figures written")
