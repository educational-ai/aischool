"""Deterministic figures for lesson 38: orthogonality and PCA.

Projected variance as a function of the projection angle (max along the long axis),
the covariance ellipse with its two orthogonal principal axes, and the scree plot
with cumulative explained variance and eigen-digits. On real iris and digits data.
Numbers reproduced and asserted.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch
from sklearn.datasets import load_iris, load_digits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "38"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "38"

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


def iris_2d():
    I = load_iris()
    X = I.data[:, [2, 3]].astype(float)   # petal length, petal width (cm)
    Xc = X - X.mean(0)
    return Xc, I.target


def cov_eig(Xc):
    S = Xc.T @ Xc / (len(Xc) - 1)
    w, V = np.linalg.eigh(S)
    order = np.argsort(w)[::-1]
    return S, w[order], V[:, order]


# ---------------------------------------- fig 38.1: projected variance vs angle
def fig_rotation() -> None:
    Xc, y = iris_2d()
    S, w, V = cov_eig(Xc)
    angs = np.linspace(0, np.pi, 181)
    var = np.array([np.array([np.cos(a), np.sin(a)]) @ S @ np.array([np.cos(a), np.sin(a)]) for a in angs])
    a_best = angs[np.argmax(var)]
    print(f"rotation: max var {var.max():.4f} at {np.degrees(a_best):.1f} deg, lambda1 {w[0]:.4f}")
    assert var.max() <= w[0] + 1e-9 and w[0] - var.max() < 1e-2
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.6, 5.0))
    # cloud + best axis
    ax0.scatter(Xc[:, 0], Xc[:, 1], s=14, color=BLUE, alpha=0.6)
    d = V[:, 0] * 3.2
    ax0.add_patch(FancyArrowPatch(-d, d, arrowstyle="-", color=RED, lw=2.4))
    ax0.add_patch(FancyArrowPatch((0, 0), V[:, 0] * 2.4, arrowstyle="-|>", mutation_scale=15, color=RED, lw=2.4, zorder=5))
    ax0.annotate("первая компонента", xy=(V[0, 0] * 1.6, V[1, 0] * 1.6), xytext=(-0.4, 1.15),
                 color=RED, fontsize=11, ha="center",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    ax0.set_aspect("equal"); ax0.axhline(0, color=GRID, lw=0.8); ax0.axvline(0, color=GRID, lw=0.8)
    ax0.set_xlabel("длина лепестка − среднее, см"); ax0.set_ylabel("ширина лепестка − среднее, см")
    ax0.set_title("Облако ирисов и ось проекции")
    # variance vs angle
    ax1.plot(np.degrees(angs), var, color=BLUE, lw=2.4)
    ax1.axvline(np.degrees(a_best), color=RED, lw=1.4, ls=(0, (4, 3)))
    ax1.plot(np.degrees(a_best), var.max(), "o", color=RED, markersize=7)
    ax1.annotate(f"максимум = $\\lambda_1$ = {w[0]:.2f}", xy=(np.degrees(a_best), var.max()),
                 xytext=(np.degrees(a_best) + 20, var.max() * 0.92), fontsize=10, color=MUTED,
                 arrowprops=dict(arrowstyle="->", color=LINE, lw=1.0))
    ax1.set_xlabel("угол оси проекции, °"); ax1.set_ylabel("дисперсия проекции $u^\\top S u$")
    ax1.set_title("Дисперсия максимальна вдоль длинной оси")
    ax1.grid(True, color=GRID, lw=0.4, alpha=0.5); ax1.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "rotation.png")


# ---------------------------------------- fig 38.2: covariance ellipse + orthogonal axes
def fig_ellipse() -> None:
    Xc, y = iris_2d()
    S, w, V = cov_eig(Xc)
    ang = np.degrees(np.arctan2(V[1, 0], V[0, 0]))
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    cmap = [BLUE, GREEN, GOLD]
    for c in range(3):
        m = y == c
        ax.scatter(Xc[m, 0], Xc[m, 1], s=16, color=cmap[c], alpha=0.55, label=load_iris().target_names[c])
    for k in (2.0,):
        e = Ellipse((0, 0), 2 * k * np.sqrt(w[0]), 2 * k * np.sqrt(w[1]), angle=ang,
                    fill=False, ec=MUTED, lw=1.4, ls=(0, (4, 3)))
        ax.add_patch(e)
    label_xy = [(-2.6, -0.7), (0.9, 0.9)]
    for i, col in enumerate([RED, VIOLET]):
        d = V[:, i] * 2 * np.sqrt(w[i])
        ax.add_patch(FancyArrowPatch((0, 0), d, arrowstyle="-|>", mutation_scale=16, color=col, lw=2.6, zorder=6))
        ax.annotate(f"$\\lambda_{i+1}={w[i]:.2f}$", xy=d, xytext=label_xy[i], color=col, fontsize=12,
                    arrowprops=dict(arrowstyle="->", color=col, lw=0.9))
    ax.set_aspect("equal"); ax.axhline(0, color=GRID, lw=0.8); ax.axvline(0, color=GRID, lw=0.8)
    ax.set_xlabel("длина лепестка − среднее, см"); ax.set_ylabel("ширина лепестка − среднее, см")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_title("Главные оси — ортогональные собственные векторы ковариации")
    print(f"ellipse: lambda {w.round(3)}, PC1 angle {ang:.1f} deg, orthogonal check {V[:,0]@V[:,1]:.1e}")
    assert abs(V[:, 0] @ V[:, 1]) < 1e-9
    save(fig, OUT / "ellipse.png")


# ---------------------------------------- fig 38.3: scree + cumulative + reconstruction (digits)
def fig_scree() -> None:
    D = load_digits(); X = D.data / 16.0
    Xc = X - X.mean(0)
    S = Xc.T @ Xc / (len(Xc) - 1)
    w = np.sort(np.linalg.eigvalsh(S))[::-1]
    w = np.clip(w, 0, None)
    cum = np.cumsum(w) / w.sum()
    m80 = int(np.argmax(cum >= 0.8) + 1)
    print(f"scree: top eigenvalue {w[0]:.3f}, components for 80% variance = {m80}")
    assert 10 <= m80 <= 16
    fig = plt.figure(figsize=(11.0, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1])
    axL = fig.add_subplot(gs[0, 0])
    ks = np.arange(1, 41)
    axL.plot(ks, w[:40], color=RED, lw=2.0, marker="o", markersize=3, label="собственные значения (спектр)")
    axL.set_yscale("log"); axL.set_xlabel("номер компоненты"); axL.set_ylabel("дисперсия $\\lambda_j$", color=RED)
    axR2 = axL.twinx()
    axR2.plot(ks, cum[:40] * 100, color=BLUE, lw=2.0, ls=(0, (4, 3)), label="накопленная дисперсия")
    axR2.set_ylabel("накоплено, %", color=BLUE); axR2.set_ylim(0, 105)
    axR2.axhline(80, color=GRID, lw=0.8)
    axR2.annotate(f"80% при {m80} компонентах", xy=(m80, 80), xytext=(m80 + 4, 55), fontsize=9.5, color=MUTED,
                  arrowprops=dict(arrowstyle="->", color=LINE, lw=1.0))
    axL.set_title("Спектр компонент и накопленная дисперсия")
    axL.grid(True, color=GRID, lw=0.4, alpha=0.4); axL.set_axisbelow(True)
    # reconstruction strip
    from sklearn.decomposition import PCA
    i = np.where(D.target == 4)[0][0]
    axR = fig.add_subplot(gs[0, 1]); axR.axis("off")
    sub = gs[0, 1].subgridspec(1, 4)
    for j, k in enumerate([2, 8, 16, 40]):
        p = PCA(n_components=k).fit(Xc)
        rec = p.inverse_transform(p.transform(Xc[i:i + 1]))[0] + X.mean(0)
        a = fig.add_subplot(sub[0, j])
        a.imshow(np.clip(rec, 0, 1).reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
        a.set_title(f"m={k}", fontsize=10); a.axis("off")
    axR.set_title("Реконструкция цифры по m компонентам", fontsize=12, pad=18)
    save(fig, OUT / "scree.png")


# ---------------------------------------- margins
def side_center() -> None:
    rng = np.random.default_rng(38)
    cloud = rng.multivariate_normal([6, 5], [[0.5, 0.4], [0.4, 0.5]], 80)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(4.4, 2.3))
    for ax, cen, tit in [(a0, False, "без центрирования"), (a1, True, "с центрированием")]:
        pts = cloud - cloud.mean(0) if cen else cloud
        ax.scatter(pts[:, 0], pts[:, 1], s=8, color=BLUE, alpha=0.5)
        S = (pts - pts.mean(0)).T @ (pts - pts.mean(0)) if cen else pts.T @ pts
        w, V = np.linalg.eigh(S); v = V[:, np.argmax(w)]
        c = pts.mean(0) if cen else np.zeros(2)
        ax.add_patch(FancyArrowPatch(c - v * 2.5, c + v * 2.5, arrowstyle="-|>", mutation_scale=10, color=RED, lw=1.8))
        ax.plot(0, 0, "x", color=INK, markersize=7)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(tit, fontsize=8.5)
    fig.suptitle("центрировать обязательно", y=1.06, fontsize=9.5)
    fig.tight_layout()
    save(fig, SIDE / "center.png")


def side_scale() -> None:
    rng = np.random.default_rng(7)
    z = rng.standard_normal(120)
    x1 = z + rng.standard_normal(120) * 0.3            # small units
    x2 = 60 * (z + rng.standard_normal(120) * 0.3)     # large units (e.g. рубли)
    x1c, x2c = x1 - x1.mean(), x2 - x2.mean()
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(4.6, 2.4))
    # covariance PCA: raw units, true scale -> x2 dominates, axis nearly vertical
    a0.scatter(x1c, x2c, s=7, color=GREEN, alpha=0.5)
    S = np.c_[x1c, x2c].T @ np.c_[x1c, x2c]; w, V = np.linalg.eigh(S); v = V[:, np.argmax(w)]
    a0.add_patch(FancyArrowPatch(-v * 190, v * 190, arrowstyle="-|>", mutation_scale=10, color=RED, lw=1.9))
    a0.set_xlim(-150, 150); a0.set_ylim(-210, 210); a0.set_aspect("equal"); a0.set_xticks([]); a0.set_yticks([])
    a0.set_title("по ковариации\nось идёт за крупным признаком", fontsize=8)
    # correlation PCA: standardized -> round cloud, axis along diagonal
    xs, ys = x1c / x1.std(), x2c / x2.std()
    a1.scatter(xs, ys, s=7, color=GREEN, alpha=0.5)
    S2 = np.c_[xs, ys].T @ np.c_[xs, ys]; w2, V2 = np.linalg.eigh(S2); v2 = V2[:, np.argmax(w2)]
    a1.add_patch(FancyArrowPatch(-v2 * 2.6, v2 * 2.6, arrowstyle="-|>", mutation_scale=10, color=RED, lw=1.9))
    a1.set_xlim(-3, 3); a1.set_ylim(-3, 3); a1.set_aspect("equal"); a1.set_xticks([]); a1.set_yticks([])
    a1.set_title("по корреляции\nось по связи признаков", fontsize=8)
    fig.suptitle("масштаб меняет первую ось", y=1.05, fontsize=9.5)
    fig.tight_layout()
    save(fig, SIDE / "scale.png")


def side_eigendigits() -> None:
    D = load_digits(); X = D.data / 16.0
    from sklearn.decomposition import PCA
    p = PCA(n_components=6).fit(X - X.mean(0))
    fig, axes = plt.subplots(1, 6, figsize=(4.6, 1.1))
    for j, ax in enumerate(axes):
        comp = p.components_[j].reshape(8, 8)
        ax.imshow(comp, cmap="RdBu", vmin=-0.4, vmax=0.4); ax.set_title(f"{j+1}", fontsize=8); ax.axis("off")
    fig.suptitle("первые «собственные цифры» (главные направления)", y=1.15, fontsize=8.5)
    fig.tight_layout()
    save(fig, SIDE / "eigendigits.png")


fig_rotation()
fig_ellipse()
fig_scree()
side_center()
side_scale()
side_eigendigits()
print("lesson 38 figures written")
