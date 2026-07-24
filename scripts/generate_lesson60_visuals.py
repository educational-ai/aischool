"""Deterministic figures for lesson 60: the curse of dimensionality.

Everything quoted in the prose is computed here and asserted:
  * the exponential price of a grid, m^d;
  * the volume of the unit ball inside the cube, V_d / 2^d, and the outer shell;
  * loss of distance contrast for iid coordinates, and how REAL structure (sklearn digits)
    keeps a contrast that column-shuffled digits lose;
  * k-NN on the REAL breast-cancer data degrading as noise coordinates are appended,
    and recovering when selection happens inside the cross-validation pipeline;
  * PCA on REAL digits: how many components carry 80/90/95/99 % of the variance;
  * random projection (Johnson-Lindenstrauss) distortion of pairwise distances on digits;
  * the Hughes peaking effect measured on real data with a small training sample.

Figures -> public/figures/lessons/60, margins -> public/figures/sidenotes/60,
numbers -> scripts/data/lesson60_facts.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from scipy.special import gammaln
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "60"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "60"
FACTS = ROOT / "scripts" / "data" / "lesson60_facts.json"

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

facts: dict[str, float | int | str] = {}


def save(fig, path, *, dpi=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def ball_volume(d):
    return float(np.exp(d / 2 * np.log(np.pi) - gammaln(d / 2 + 1)))


# ---------------------------------------------------------------- fig 60.1: the grid
def fig_cells() -> None:
    rng = np.random.default_rng(60)
    n = 200
    fig = plt.figure(figsize=(11.0, 3.6))
    ax0 = fig.add_subplot(1, 3, 1)
    x = rng.random(n)
    ax0.scatter(x, rng.normal(0, 0.06, n), s=12, color=BLUE, alpha=0.75)
    for c in np.linspace(0, 1, 11):
        ax0.axvline(c, color=GRID, lw=0.8)
    ax0.set_ylim(-0.5, 0.5); ax0.set_yticks([])
    ax0.set_title("$d=1$: 10 клеток, ~20 точек в каждой", fontsize=11)

    ax1 = fig.add_subplot(1, 3, 2)
    p = rng.random((n, 2))
    ax1.scatter(p[:, 0], p[:, 1], s=12, color=BLUE, alpha=0.75)
    for c in np.linspace(0, 1, 11):
        ax1.axvline(c, color=GRID, lw=0.8); ax1.axhline(c, color=GRID, lw=0.8)
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_aspect("equal")
    counts = np.zeros((10, 10), dtype=int)
    for a, b in p:
        counts[min(9, int(a * 10)), min(9, int(b * 10))] += 1
    empty = int((counts == 0).sum())
    facts["grid_empty_cells_2d"] = empty
    ax1.set_title(f"$d=2$: 100 клеток, {empty} пустых", fontsize=11)

    ax2 = fig.add_subplot(1, 3, 3)
    ds = np.arange(1, 9)
    ax2.bar(ds, 10.0 ** ds, color=VIOLET, alpha=0.85, width=0.62)
    ax2.axhline(1797, color=RED, lw=1.6, ls=(0, (5, 3)))
    ax2.text(1.1, 2600, "1797 цифр из sklearn", color=RED, fontsize=9)
    ax2.set_yscale("log"); ax2.set_xlabel("размерность $d$")
    ax2.set_ylabel("число клеток $10^d$")
    ax2.set_title("цена одного и того же разрешения", fontsize=11)
    ax2.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax2.set_axisbelow(True)

    fig.suptitle("Равное разрешение по каждой оси стоит экспоненциально", y=1.04, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "cells.png")
    assert empty >= 5, empty
    print("cells:", empty, "empty cells in 2D")


# ---------------------------------------------------------------- fig 60.2: ball in cube
def fig_ball_cube() -> None:
    ds = np.arange(1, 51)
    frac = np.array([ball_volume(int(d)) / 2.0 ** int(d) for d in ds])
    for d in (2, 3, 10, 20, 50):
        facts[f"ball_frac_d{d}"] = ball_volume(d) / 2.0 ** d
    facts["ball_vol_d5"] = ball_volume(5)
    facts["ball_vol_d20"] = ball_volume(20)
    facts["ball_argmax"] = int(np.argmax([ball_volume(int(d)) for d in ds]) + 1)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    a0.plot(ds, [ball_volume(int(d)) for d in ds], color=BLUE, lw=2.2)
    a0.scatter([facts["ball_argmax"]], [ball_volume(facts["ball_argmax"])], s=45, color=RED, zorder=5)
    a0.annotate(f"максимум при $d={facts['ball_argmax']}$", (facts["ball_argmax"], ball_volume(facts["ball_argmax"])),
                xytext=(12, 4.6), fontsize=10, color=RED)
    a0.set_xlabel("размерность $d$"); a0.set_ylabel("объём единичного шара $V_d$")
    a0.set_title("объём шара сам стремится к нулю", fontsize=12)
    a0.grid(True, color=GRID, lw=0.4, alpha=0.5); a0.set_axisbelow(True)

    a1.semilogy(ds, frac, color=RED, lw=2.2)
    for d in (2, 3, 10, 20):
        a1.scatter([d], [frac[d - 1]], s=38, color=INK, zorder=5)
        mant, expo = f"{frac[d-1]:.2e}".split("e")
        a1.annotate(f"$d={d}$: ${mant}\\cdot10^{{{int(expo)}}}$", (d, frac[d - 1]),
                    xytext=(d + 1.5, frac[d - 1] * 3), fontsize=9, color=MUTED)
    a1.set_xlabel("размерность $d$"); a1.set_ylabel("$V_d/2^d$ (лог. ось)")
    a1.set_title("доля вписанного шара в кубе", fontsize=12)
    a1.grid(True, color=GRID, lw=0.4, alpha=0.5); a1.set_axisbelow(True)
    fig.suptitle("Шар исчезает внутри куба: почти весь объём — у углов", y=1.02, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "ball_cube.png")
    assert abs(facts["ball_frac_d2"] - np.pi / 4) < 1e-12
    assert facts["ball_argmax"] == 5
    assert facts["ball_frac_d20"] < 1e-7
    # every mantissa quoted in the prose, pinned to its third significant digit
    assert abs(facts["ball_frac_d10"] / 1e-3 - 2.49) < 0.005
    assert abs(facts["ball_frac_d20"] / 1e-8 - 2.46) < 0.005
    assert abs(facts["ball_frac_d50"] / 1e-28 - 1.54) < 0.005
    # caption: one hit in ~40.6 million, NOT twenty million
    facts["ball_odds_d20"] = float(1 / facts["ball_frac_d20"])
    assert abs(facts["ball_odds_d20"] / 1e7 - 4.06) < 0.005
    print("ball:", facts["ball_argmax"], facts["ball_frac_d20"])


# ---------------------------------------------------------------- fig 60.3: distance contrast
def contrast_of(X, rng, reps=200):
    idx = rng.choice(len(X), reps, replace=False)
    vals = []
    for i in idx:
        D = np.linalg.norm(X - X[i], axis=1)
        D = D[D > 0]
        vals.append((D.max() - D.min()) / D.min())
    return float(np.mean(vals))


def fig_contrast() -> None:
    rng = np.random.default_rng(60)
    dims = [2, 20, 200, 2000]
    stats = {}
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.0, 4.3))
    for d, c in zip(dims, [BLUE, GREEN, GOLD, RED]):
        X = rng.normal(size=(500, d)); q = rng.normal(size=d)
        D = np.linalg.norm(X - q, axis=1)
        stats[d] = {"min": float(D.min()), "max": float(D.max()),
                    "contrast": float((D.max() - D.min()) / D.min())}
        a0.hist(D / D.mean(), bins=40, histtype="step", lw=2.0, color=c,
                density=True, label=f"$d={d}$")
    a0.set_xlabel("расстояние, делённое на своё среднее")
    a0.set_ylabel("плотность")
    a0.set_title("гистограммы сжимаются к единице", fontsize=12)
    a0.legend(frameon=False, fontsize=10)
    a0.grid(True, color=GRID, lw=0.4, alpha=0.5); a0.set_axisbelow(True)

    for d in dims:
        facts[f"contrast_gauss_d{d}"] = stats[d]["contrast"]

    digits = load_digits().data
    c_real = contrast_of(digits, np.random.default_rng(1))
    sh = digits.copy()
    r2 = np.random.default_rng(2)
    for j in range(sh.shape[1]):
        sh[:, j] = r2.permutation(sh[:, j])
    c_shuf = contrast_of(sh, np.random.default_rng(1))
    facts["contrast_digits"] = c_real
    facts["contrast_digits_shuffled"] = c_shuf

    xs = np.array(dims, dtype=float)
    a1.loglog(xs, [stats[d]["contrast"] for d in dims], "o-", color=RED, lw=2.0,
              label="независимые координаты")
    a1.scatter([64], [c_real], s=70, color=GREEN, zorder=6)
    a1.annotate(f"реальные цифры 8×8:\nконтраст {c_real:.2f}", (64, c_real),
                xytext=(70, 5.0), fontsize=9.5, color=GREEN)
    a1.scatter([64], [c_shuf], s=70, color=VIOLET, zorder=6)
    a1.annotate(f"те же цифры, столбцы\nперемешаны: {c_shuf:.2f}", (64, c_shuf),
                xytext=(3, 0.45), fontsize=9.5, color=VIOLET)
    a1.set_xlabel("размерность $d$ (лог)")
    a1.set_ylabel(r"контраст $(\max-\min)/\min$ (лог)")
    a1.set_title("структура возвращает контраст", fontsize=12)
    a1.grid(True, which="both", color=GRID, lw=0.4, alpha=0.5); a1.set_axisbelow(True)
    fig.suptitle("Чем больше независимых координат, тем меньше разница между близким и далёким",
                 y=1.02, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "distance_contrast.png")

    # each contrast quoted in the prose, pinned to the printed precision
    assert abs(stats[2]["contrast"] - 25.54) < 0.005
    assert abs(stats[20]["contrast"] - 1.89) < 0.005
    assert abs(stats[200]["contrast"] - 0.366) < 0.0005
    assert abs(stats[2000]["contrast"] - 0.087) < 0.0005
    assert abs(c_real - 3.22) < 0.005 and abs(c_shuf - 0.90) < 5e-3
    # prose says the contrast drops "three and a half times"
    assert abs(c_real / c_shuf - 3.5647) < 5e-05
    # prose says the far neighbour is 26.5x the near one (contrast + 1)
    assert abs((stats[2]["contrast"] + 1) - 26.54) < 0.005
    print("contrast:", {d: round(stats[d]["contrast"], 3) for d in dims},
          "digits", round(c_real, 3), "shuffled", round(c_shuf, 3))


# ---------------------------------------------------------------- fig 60.4: k-NN drowns in noise
def fig_knn_noise() -> None:
    bc = load_breast_cancer()
    X, y = bc.data, bc.target
    facts["bc_n"] = int(X.shape[0]); facts["bc_d"] = int(X.shape[1])
    noise_counts = [0, 10, 50, 100, 200, 500, 1000]
    accs = []
    for m in noise_counts:
        r = np.random.default_rng(7)
        Z = np.hstack([X, r.normal(size=(X.shape[0], m))]) if m else X
        pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(15))
        accs.append(float(cross_val_score(pipe, Z, y, cv=5).mean()))
    r = np.random.default_rng(7)
    Z = np.hstack([X, r.normal(size=(X.shape[0], 1000))])
    pipe_sel = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=30),
                             KNeighborsClassifier(15))
    acc_sel = float(cross_val_score(pipe_sel, Z, y, cv=5).mean())

    # how many pure-noise columns look "significant" on the full table
    Zs = StandardScaler().fit_transform(Z)
    picked = SelectKBest(f_classif, k=30).fit(Zs, y).get_support()
    noise_picked = int(picked[X.shape[1]:].sum())

    facts["knn_acc_clean"] = accs[0]
    facts["knn_acc_1000"] = accs[-1]
    facts["knn_acc_100"] = accs[noise_counts.index(100)]
    facts["knn_acc_select"] = acc_sel
    facts["knn_noise_picked"] = noise_picked

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(noise_counts, [a * 100 for a in accs], "o-", color=RED, lw=2.4,
            label="k-NN по всем координатам")
    ax.axhline(accs[0] * 100, color=GREEN, lw=1.4, ls=(0, (5, 3)))
    ax.text(2, accs[0] * 100 + 0.35, f"30 честных признаков: {accs[0]*100:.1f}%",
            color=GREEN, fontsize=10)
    ax.scatter([1000], [acc_sel * 100], s=90, color=BLUE, zorder=6, marker="s")
    ax.annotate(f"отбор 30 признаков\nвнутри пайплайна: {acc_sel*100:.1f}%",
                (1000, acc_sel * 100), xytext=(430, 92.6), fontsize=10, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.1))
    ax.set_xlabel("добавлено чисто шумовых координат")
    ax.set_ylabel("точность 5-fold CV, %")
    ax.set_title("Реальные данные о раке груди: шумовые оси топят соседей")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False, fontsize=10)
    save(fig, OUT / "knn_noise.png")

    # the four percentages quoted in prose and in the caption
    assert abs(accs[0] * 100 - 96.1) < 0.05
    assert abs(facts["knn_acc_100"] * 100 - 93.2) < 0.05
    assert abs(accs[-1] * 100 - 84.4) < 0.05
    assert abs(acc_sel * 100 - 94.9) < 0.05
    # prose: "almost twelve percentage points" lost
    assert 11.5 < (accs[0] - accs[-1]) * 100 < 12.0
    assert noise_picked == 5
    print("knn:", [round(a, 4) for a in accs], "sel", round(acc_sel, 4),
          "noise picked", noise_picked)


# ---------------------------------------------------------------- fig 60.5: effective dimension
def fig_pca_digits() -> None:
    digits = load_digits()
    X = digits.data
    p = PCA().fit(X)
    cum = np.cumsum(p.explained_variance_ratio_)
    ks = {t: int(np.searchsorted(cum, t) + 1) for t in (0.8, 0.9, 0.95, 0.99)}
    for t, k in ks.items():
        facts[f"pca_k_{int(t*100)}"] = k
    facts["pca_var_2"] = float(cum[1] * 100)
    facts["pca_var_10"] = float(cum[9] * 100)

    fig = plt.figure(figsize=(11.0, 4.4))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(np.arange(1, 65), cum * 100, color=BLUE, lw=2.4)
    for t, c in zip((0.8, 0.9, 0.95, 0.99), (GREEN, GOLD, RED, VIOLET)):
        k = ks[t]
        ax.axhline(t * 100, color=c, lw=0.9, ls=(0, (3, 3)))
        ax.scatter([k], [cum[k - 1] * 100], s=42, color=c, zorder=5)
        ax.annotate(f"{int(t*100)}%: k={k}", (k, cum[k - 1] * 100),
                    xytext=(k + 3.0, t * 100 - 12), fontsize=9.5, color=c)
    ax.set_xlabel("число главных компонент $k$")
    ax.set_ylabel("накопленная объяснённая дисперсия, %")
    ax.set_title("64 пикселя, но не 64 независимых направления", fontsize=12)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)

    # reconstructions of one real digit
    img = X[17]
    axes = [fig.add_subplot(2, 4, 3), fig.add_subplot(2, 4, 4),
            fig.add_subplot(2, 4, 7), fig.add_subplot(2, 4, 8)]
    for a, k in zip(axes, [2, 10, ks[0.95], 64]):
        pk = PCA(n_components=k).fit(X)
        rec = pk.inverse_transform(pk.transform(img.reshape(1, -1)))[0]
        a.imshow(rec.reshape(8, 8), cmap="bone_r")
        a.set_xticks([]); a.set_yticks([])
        a.set_title(f"$k={k}$", fontsize=10)
    fig.suptitle("Эффективная размерность реальных цифр много меньше исходной",
                 y=1.02, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "pca_digits.png")
    assert ks[0.8] == 13 and ks[0.9] == 21 and ks[0.95] == 29 and ks[0.99] == 41
    assert abs(facts["pca_var_2"] - 28.5) < 0.05
    assert abs(facts["pca_var_10"] - 73.8) < 0.05
    print("pca:", ks, round(facts["pca_var_10"], 1))


# ---------------------------------------------------------------- fig 60.6: random projection
def fig_projection() -> None:
    X = load_digits().data
    rng = np.random.default_rng(60)
    sub = X[rng.choice(len(X), 300, replace=False)]
    d0 = pdist(sub)
    facts["jl_pairs"] = int(len(d0))
    ks = [2, 4, 8, 16, 32, 64, 128, 256]
    med, p90 = [], []
    for k in ks:
        mm, pp = [], []
        for rep in range(5):
            R = np.random.default_rng(100 + rep).normal(size=(sub.shape[1], k)) / np.sqrt(k)
            e = np.abs(pdist(sub @ R) / d0 - 1)
            mm.append(np.median(e)); pp.append(np.quantile(e, 0.9))
        med.append(float(np.mean(mm))); p90.append(float(np.mean(pp)))
    facts["jl_med_k32"] = med[ks.index(32)]
    facts["jl_med_k8"] = med[ks.index(8)]
    facts["jl_p90_k32"] = p90[ks.index(32)]
    facts["jl_theory_k"] = float(8 * np.log(300) / 0.2 ** 2)

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.loglog(ks, med, "o-", color=BLUE, lw=2.2, label="медианное искажение")
    ax.loglog(ks, p90, "s--", color=RED, lw=2.0, label="90-й процентиль")
    ax.loglog(ks, 0.7 / np.sqrt(np.array(ks, dtype=float)), color=MUTED, lw=1.2,
              ls=(0, (2, 3)), label=r"ориентир $0{,}7/\sqrt{k}$")
    ax.set_xlabel("размерность проекции $k$")
    ax.set_ylabel("относительное искажение расстояний")
    ax.set_title(f"Случайная проекция {facts['jl_pairs']} пар реальных цифр")
    ax.grid(True, which="both", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10)
    save(fig, OUT / "random_projection.png")
    assert abs(facts["jl_med_k32"] * 100 - 8.9) < 0.05
    assert abs(facts["jl_med_k8"] * 100 - 17.7) < 0.05
    assert abs(facts["jl_p90_k32"] * 100 - 21.5) < 0.05
    assert abs(facts["jl_theory_k"] - 1141) < 1.0
    assert facts["jl_pairs"] == 44850
    print("jl:", [round(m, 4) for m in med])


# ---------------------------------------------------------------- margins
def side_corners() -> None:
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(plt.Rectangle((-1, -1), 2, 2, facecolor=WASH, edgecolor=LINE, lw=1.2))
    th = np.linspace(0, 2 * np.pi, 300)
    ax.fill(np.cos(th), np.sin(th), color=BLUE, alpha=0.30)
    ax.plot(np.cos(th), np.sin(th), color=BLUE, lw=1.4)
    for sx, sy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        ax.annotate("угол", (sx * 0.86, sy * 0.86), ha="center", va="center",
                    fontsize=8, color=RED)
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.35, 1.35)
    ax.set_title(r"$d=2$: круг = $\pi/4\approx0{,}785$ квадрата", fontsize=9)
    save(fig, SIDE / "corners.png")


def side_shell() -> None:
    ds = np.arange(1, 101)
    y = 1 - 0.9 ** ds
    facts["shell_d10"] = float(1 - 0.9 ** 10)
    facts["shell_d20"] = float(1 - 0.9 ** 20)
    facts["shell_d50"] = float(1 - 0.9 ** 50)
    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    ax.plot(ds, y * 100, color=RED, lw=2.0)
    for d in (10, 20, 50):
        ax.scatter([d], [(1 - 0.9 ** d) * 100], s=26, color=INK, zorder=5)
    ax.set_xlabel("$d$", fontsize=9)
    ax.set_ylabel("% объёма", fontsize=9)
    ax.set_title("доля объёма шара\nво внешних 10% радиуса", fontsize=9)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "shell.png")
    assert abs(facts["shell_d10"] * 100 - 65.1) < 0.05
    assert abs(facts["shell_d20"] * 100 - 87.8) < 0.05
    assert abs(facts["shell_d50"] * 100 - 99.5) < 0.05


def side_hamming() -> None:
    from math import comb
    fig, ax = plt.subplots(figsize=(4.1, 2.5))
    for d, c in [(10, BLUE), (100, GOLD), (1000, RED)]:
        ks = np.arange(0, d + 1)
        pmf = np.array([comb(d, int(k)) for k in ks], dtype=float)
        pmf = pmf / pmf.sum()
        ax.plot(ks / d, pmf * d, color=c, lw=1.8, label=f"$d={d}$")
    facts["hamming_sd_d1000"] = float(np.sqrt(1000) / 2)
    facts["hamming_rel_lo"] = 0.5 - 3 * float(np.sqrt(1000) / 2) / 1000
    facts["hamming_rel_hi"] = 0.5 + 3 * float(np.sqrt(1000) / 2) / 1000
    ax.set_xlim(0.2, 0.8); ax.set_xlabel("доля различий", fontsize=9)
    ax.set_yticks([]); ax.legend(frameon=False, fontsize=8)
    ax.set_title("расстояние Хэмминга\nсжимается к 0,5", fontsize=9)
    save(fig, SIDE / "hamming.png")
    assert abs(facts["hamming_rel_lo"] - 0.4526) < 5e-05
    assert abs(facts["hamming_rel_hi"] - 0.5474) < 5e-05


def side_peaking() -> None:
    bc = load_breast_cancer()
    X, y = bc.data, bc.target
    rng = np.random.default_rng(11)
    Xa = np.hstack([X, rng.normal(size=(X.shape[0], 200))])
    ps = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 230]
    accs = []
    for p in ps:
        sc = []
        for rep in range(40):
            r = np.random.default_rng(500 + rep)
            idx = r.permutation(len(X)); tr, te = idx[:40], idx[40:]
            F, _ = f_classif(Xa[tr], y[tr])
            order = np.argsort(-np.nan_to_num(F))[:p]
            sca = StandardScaler().fit(Xa[tr][:, order])
            m = KNeighborsClassifier(5).fit(sca.transform(Xa[tr][:, order]), y[tr])
            sc.append(m.score(sca.transform(Xa[te][:, order]), y[te]))
        accs.append(float(np.mean(sc)))
    best = int(np.argmax(accs))
    facts["peak_p"] = ps[best]
    facts["peak_acc"] = accs[best]
    facts["peak_acc_last"] = accs[-1]
    facts["peak_acc_first"] = accs[0]
    fig, ax = plt.subplots(figsize=(4.1, 2.6))
    ax.semilogx(ps, [a * 100 for a in accs], "o-", color=VIOLET, lw=1.8, ms=4)
    ax.scatter([ps[best]], [accs[best] * 100], s=45, color=RED, zorder=6)
    ax.set_xlabel("число признаков", fontsize=9)
    ax.set_ylabel("точность, %", fontsize=9)
    ax.set_title(f"пик Хьюза: обучение по 40 больным,\nмаксимум при {ps[best]} признаках", fontsize=9)
    ax.grid(True, which="both", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "peaking.png")
    assert ps[best] == 20 and ps[-1] == 230
    assert abs(accs[best] * 100 - 92.3) < 0.05
    assert abs(accs[0] * 100 - 89.3) < 0.05
    assert abs(accs[-1] * 100 - 85.3) < 0.05
    print("peaking:", list(zip(ps, [round(a, 4) for a in accs])))


# ---------------------------------------------------------------- pure arithmetic quoted in prose
def checks() -> None:
    # grid exercise: d = 12, m = 5
    cells = 5 ** 12
    facts["ex_cells_5_12"] = cells
    facts["ex_fill_1e6"] = 1e6 / cells
    facts["ex_needed_20"] = 20 * cells
    assert cells == 244140625
    assert abs(facts["ex_fill_1e6"] - 0.0041) < 5e-05
    assert abs(facts["ex_needed_20"] / 1e9 - 4.88) < 0.005

    # corner vs face distance
    facts["corner_ratio_d20"] = float(np.sqrt(20))
    facts["corners_count_d20"] = 2 ** 20
    assert abs(facts["corner_ratio_d20"] - 4.47) < 0.005
    assert facts["corners_count_d20"] == 1048576
    facts["corner_share_d2"] = 1 - np.pi / 4
    assert abs(facts["corner_share_d2"] - 0.215) < 0.0005

    # radius of the ball taking half of the cube
    def half_radius(dim):
        return 2 * (1 / (2 * ball_volume(dim))) ** (1 / dim)
    for dim in (2, 10, 50):
        facts[f"half_radius_d{dim}"] = float(half_radius(dim))
    assert abs(facts["half_radius_d2"] - 0.80) < 0.005
    assert abs(facts["half_radius_d10"] - 1.70) < 0.005
    assert abs(facts["half_radius_d50"] - 3.55) < 0.005
    # asymptotics r ~ sqrt(d)/2, computed in logs to avoid underflow of V_d
    def half_radius_log(dim):
        log_v = dim / 2 * np.log(np.pi) - gammaln(dim / 2 + 1)
        return 2 * np.exp((-np.log(2) - log_v) / dim)
    assert abs(half_radius_log(2000) / np.sqrt(2000) - 0.5) < 0.05

    # ball fractions quoted with three digits
    assert abs(np.pi / 4 - 0.785) < 0.0005 and abs(np.pi / 6 - 0.524) < 5e-4
    assert abs(ball_volume(3) - 4.19) < 0.005 and abs(ball_volume(5) - 5.264) < 5e-4
    assert abs(ball_volume(20) - 0.0258) < 5e-05

    # noise share exercise: 1/(1+m) < 0.1  =>  m > 9
    assert 1 / (1 + 9) == 0.1 and 1 / (1 + 10) < 0.1
    facts["noise_share_m1000"] = 1 / 1001

    # Hamming
    assert abs(np.sqrt(10) / 2 - 1.58) < 0.005
    assert abs(np.sqrt(1000) / 2 - 15.81) < 0.005
    facts["hamming_noise_contrib"] = 980 * 0.5      # expected differing bits among the 980
    assert abs(facts["hamming_noise_contrib"] - 490) < 1e-9

    # genomics: rank deficit
    facts["genome_zero_eigen"] = 20000 - 499
    assert facts["genome_zero_eigen"] == 19501
    facts["genome_false_positives"] = 0.05 * 20000
    assert facts["genome_false_positives"] == 1000

    # boolean functions on the cube
    assert 2 ** 20 == 1048576
    print("checks: all prose arithmetic verified")


checks()
fig_cells()
fig_ball_cube()
fig_contrast()
fig_knn_noise()
fig_pca_digits()
fig_projection()
side_corners()
side_shell()
side_hamming()
side_peaking()

FACTS.write_text(json.dumps(facts, ensure_ascii=False, indent=1, sort_keys=True))
print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in facts.items()},
                 ensure_ascii=False, indent=1, sort_keys=True))
print("lesson 60 figures written")
