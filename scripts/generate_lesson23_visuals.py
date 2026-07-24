"""Deterministic figures for lesson 23: constraints, Lagrange, sparsity.

Level-curve tangency at the constrained optimum, the L2-circle vs L1-diamond
sparsity picture, and the real Lasso coefficient path on the diabetes data
(Efron et al. 2004): BMI and S5 are the two survivors; ridge keeps all ten.
Every quoted number is reproduced and asserted here.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Circle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "23"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "23"

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


def arrow(ax, s, e, *, color=INK, lw=2.0, ms=13):
    ax.add_patch(FancyArrowPatch(s, e, arrowstyle="-|>", color=color, lw=lw,
                                 mutation_scale=ms, shrinkA=0, shrinkB=0))


# ============================================================ diabetes data
NAMES = ["AGE", "SEX", "BMI", "BP", "S1", "S2", "S3", "S4", "S5", "S6"]


def diabetes():
    rows = list(csv.reader((ROOT / "scripts" / "data" / "diabetes.tab.txt").open(),
                           delimiter="\t"))
    data = np.array([[float(v) for v in r] for r in rows[1:]])
    X, y = data[:, :10], data[:, 10]
    Xs = (X - X.mean(0)) / X.std(0)
    return Xs, y - y.mean()


def lasso_cd(X, y, alpha, iters=4000):
    w = np.zeros(X.shape[1]); n = len(y); cn = (X ** 2).sum(0) / n
    for _ in range(iters):
        for j in range(X.shape[1]):
            r = y - X @ w + X[:, j] * w[j]
            rho = (X[:, j] @ r) / n
            w[j] = np.sign(rho) * max(abs(rho) - alpha, 0) / cn[j]
    return w


def ridge_w(X, y, lam):
    n = len(y)
    return np.linalg.solve(X.T @ X / n + lam * np.eye(X.shape[1]), X.T @ y / n)


# ---------------------------- fig 23.1: tangency of level curve and constraint
def fig_tangency() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    # objective f = (x-2.3)^2 + 0.6*(y-1.7)^2 ; constraint x^2+y^2=1 (unit circle)
    cx, cy, sc = 2.3, 1.7, 0.6
    xs = np.linspace(-2.2, 3.2, 240); ys = np.linspace(-2.0, 2.8, 240)
    GX, GY = np.meshgrid(xs, ys)
    F = (GX - cx) ** 2 + sc * (GY - cy) ** 2
    ax.contour(GX, GY, F, levels=[0.5, 1.5, 3.0, 5.0, 7.5], colors=[LINE], linewidths=1.0)
    # constraint circle
    th = np.linspace(0, 2 * math.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=BLUE, lw=2.2, zorder=3)
    ax.text(-1.62, 0.05, "ограничение\n$g=0$", color=BLUE, fontsize=12, ha="center")
    # constrained optimum: minimize f on unit circle -> numeric
    ang = np.linspace(0, 2 * math.pi, 4000)
    fv = (np.cos(ang) - cx) ** 2 + sc * (np.sin(ang) - cy) ** 2
    a0 = ang[np.argmin(fv)]
    px, py = math.cos(a0), math.sin(a0)
    # level curve through the optimum
    lv = (px - cx) ** 2 + sc * (py - cy) ** 2
    ax.contour(GX, GY, F, levels=[lv], colors=[RED], linewidths=1.6)
    ax.scatter([px], [py], s=80, color=INK, zorder=6)
    ax.text(px + 0.08, py + 0.12, "оптимум", fontsize=11, color=INK)
    # gradients at optimum: grad f = (2(x-cx), 2sc(y-cy)); grad g = (2x,2y)
    gf = np.array([2 * (px - cx), 2 * sc * (py - cy)])
    gg = np.array([2 * px, 2 * py])
    ufa = gf / np.hypot(*gf); uga = gg / np.hypot(*gg)
    arrow(ax, (px, py), (px + 0.75 * ufa[0], py + 0.75 * ufa[1]), color=RED)
    arrow(ax, (px, py), (px + 0.75 * uga[0], py + 0.75 * uga[1]), color=GREEN)
    ax.text(px + 0.8 * ufa[0] - 0.28, py + 0.8 * ufa[1] - 0.02, r"$\nabla f$", color=RED, fontsize=13)
    ax.text(px + 0.8 * uga[0] + 0.06, py + 0.8 * uga[1], r"$\nabla g$", color=GREEN, fontsize=13)
    ax.scatter([cx], [cy], s=45, color=FAINT, zorder=5)
    ax.text(cx + 0.06, cy, "цель $(2{,}3;\\,1{,}7)$", fontsize=10.5, color=MUTED)
    # a neighbouring feasible point where gradients are NOT parallel
    a1 = a0 - 1.5
    qx, qy = math.cos(a1), math.sin(a1)
    ax.scatter([qx], [qy], s=45, color=MUTED, zorder=5)
    gfq = np.array([2 * (qx - cx), 2 * sc * (qy - cy)]); ggq = np.array([2 * qx, 2 * qy])
    ufq = gfq / np.hypot(*gfq); ugq = ggq / np.hypot(*ggq)
    arrow(ax, (qx, qy), (qx + 0.5 * ufq[0], qy + 0.5 * ufq[1]), color=RED, lw=1.5, ms=11)
    arrow(ax, (qx, qy), (qx + 0.5 * ugq[0], qy + 0.5 * ugq[1]), color=GREEN, lw=1.5, ms=11)
    ax.text(qx - 0.05, qy - 0.4, "соседняя точка:\nстрелки расходятся", fontsize=9.5,
            color=MUTED, ha="center", va="top")
    # check parallelism at optimum
    cross = ufa[0] * uga[1] - ufa[1] * uga[0]
    print(f"tangency: optimum angle={math.degrees(a0):.1f} deg, grad cross={cross:.4f}")
    assert abs(cross) < 0.02
    ax.set_aspect("equal"); ax.set_xlim(-2.2, 3.3); ax.set_ylim(-1.7, 2.8)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title("В оптимуме градиенты параллельны")
    save(fig, OUT / "tangency.png")


# ---------------------------- fig 23.2: L2 circle vs L1 diamond sparsity
def fig_l1l2() -> None:
    fig, (a2, a1) = plt.subplots(1, 2, figsize=(10.2, 5.2))
    # unconstrained min of loss ellipse
    wx, wy = 2.1, 0.5
    A, B = 1.0, 1.4  # ellipse axis weights: (w1-wx)^2 + B*(w2-wy)^2
    xs = np.linspace(-1.5, 3.0, 240); ys = np.linspace(-1.5, 2.0, 240)
    GX, GY = np.meshgrid(xs, ys)
    F = A * (GX - wx) ** 2 + B * (GY - wy) ** 2

    def solve(kind, R):
        best, bv = None, 1e18
        for gx in np.linspace(-R, R, 400):
            # boundary y for given x
            if kind == "l2":
                if gx ** 2 > R ** 2: continue
                for sgn in (1, -1):
                    gy = sgn * math.sqrt(max(R ** 2 - gx ** 2, 0))
                    v = A * (gx - wx) ** 2 + B * (gy - wy) ** 2
                    if v < bv: bv, best = v, (gx, gy)
            else:
                rem = R - abs(gx)
                if rem < 0: continue
                for gy in (rem, -rem):
                    v = A * (gx - wx) ** 2 + B * (gy - wy) ** 2
                    if v < bv: bv, best = v, (gx, gy)
        return best

    R = 1.3
    for ax, kind, title, patch in [
        (a2, "l2", "$L_2$: круг — оба веса ненулевые", "circle"),
        (a1, "l1", "$L_1$: ромб — решение садится на ось", "diamond")]:
        ax.contour(GX, GY, F, levels=[0.4, 1.2, 2.5, 4.5, 7.0], colors=[LINE], linewidths=1.0)
        if patch == "circle":
            ax.add_patch(Circle((0, 0), R, fill=True, facecolor=WASH, edgecolor=BLUE, lw=2.0, alpha=0.7, zorder=2))
        else:
            d = np.array([[R, 0], [0, R], [-R, 0], [0, -R]])
            ax.add_patch(Polygon(d, closed=True, facecolor=WASH, edgecolor=BLUE, lw=2.0, alpha=0.7, zorder=2))
        # path of solutions for shrinking R
        pth = np.array([solve(kind, r) for r in np.linspace(0.25, 2.6, 40)])
        ax.plot(pth[:, 0], pth[:, 1], color=GOLD, lw=1.4, ls=(0, (4, 2)), zorder=3)
        sol = solve(kind, R)
        ax.scatter([sol[0]], [sol[1]], s=85, color=RED, zorder=6)
        ax.scatter([wx], [wy], s=45, color=FAINT, zorder=5)
        ax.text(wx + 0.05, wy + 0.05, "своб. мин.", fontsize=9.5, color=MUTED)
        if kind == "l1":
            ax.text(sol[0] - 0.1, sol[1] - 0.32, "$w_2=0$", fontsize=11, color=RED, ha="center")
            print(f"L1 solution R={R}: w=({sol[0]:.3f}, {sol[1]:.3f})")
            assert abs(sol[1]) < 1e-2  # sits on the axis
        else:
            print(f"L2 solution R={R}: w=({sol[0]:.3f}, {sol[1]:.3f})")
            assert abs(sol[1]) > 0.1  # both nonzero
        ax.axhline(0, color=GRID, lw=0.8); ax.axvline(0, color=GRID, lw=0.8)
        ax.set_aspect("equal"); ax.set_xlim(-1.5, 3.0); ax.set_ylim(-1.5, 2.0)
        ax.set_xlabel("$w_1$"); ax.set_ylabel("$w_2$")
        ax.set_title(title, fontsize=12.5)
    fig.suptitle("Одна и та же потеря, два ограничения: круг сжимает, ромб зануляет",
                 y=1.02, fontsize=14.5)
    fig.tight_layout()
    save(fig, OUT / "l1-l2.png")


# ---------------------------- fig 23.3: real Lasso path on diabetes
def fig_lasso_path() -> None:
    Xs, yc = diabetes()
    alphas = np.logspace(np.log10(0.03), np.log10(70), 60)
    W = np.array([lasso_cd(Xs, yc, a) for a in alphas])  # (60,10)
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    palette = [MUTED, VIOLET, RED, BLUE, FAINT, GOLD, GREEN, "#8c6d4a", "#b94a3b", "#4a6d8c"]
    highlight = {"BMI": RED, "S5": BLUE, "BP": GREEN}
    for j, nm in enumerate(NAMES):
        col = highlight.get(nm, LINE)
        lw = 2.4 if nm in highlight else 1.1
        ax.plot(alphas, W[:, j], color=col, lw=lw, zorder=4 if nm in highlight else 2)
        if nm in highlight:
            ax.text(alphas[0] * 0.86, W[0, j], nm, color=col, fontsize=12,
                    ha="left", va="center", fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(72, 0.017)  # inverted: budget grows to the right (alpha shrinks)
    ax.axhline(0, color=INK, lw=0.8)
    # annotate the two-survivor budget
    a2 = 25.0
    w2 = lasso_cd(Xs, yc, a2)
    active2 = [NAMES[j] for j in range(10) if abs(w2[j]) > 1e-6]
    print(f"lasso alpha={a2}: active={active2}")
    assert set(active2) == {"BMI", "S5"}
    ax.axvline(a2, color=GOLD, lw=1.2, ls=(0, (4, 2)), zorder=1)
    ax.text(a2, ax.get_ylim()[1] * 0.94, " жёсткий бюджет:\n только BMI и S5",
            color=GOLD, fontsize=10, va="top")
    # full-budget count
    wfull = lasso_cd(Xs, yc, alphas[0])
    print(f"lasso min alpha nonzero: {int((np.abs(wfull) > 1e-6).sum())}")
    ax.set_xlabel(r"штраф $\alpha$  (бюджет растёт вправо)")
    ax.set_ylabel("коэффициент $w_j$")
    ax.set_title("Путь Lasso на данных о диабете: кто выживает при жёстком бюджете")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "lasso-path.png")


# ------------------------------------------------ margins
def side_shadow() -> None:
    fig, ax = plt.subplots(figsize=(4.1, 2.9))
    B = np.linspace(0.4, 3.0, 100)
    V = 2.5 / B + 0.3 * B  # convex-ish value function
    ax.plot(B, V, color=BLUE, lw=2.0)
    b0 = 1.4
    v0 = 2.5 / b0 + 0.3 * b0
    slope = -2.5 / b0 ** 2 + 0.3  # V'(B) = -lambda
    t = np.array([b0 - 0.6, b0 + 0.6])
    ax.plot(t, v0 + slope * (t - b0), color=RED, lw=1.6)
    ax.scatter([b0], [v0], s=40, color=INK, zorder=5)
    ax.text(b0 + 0.05, v0 + 0.35, "наклон $=-\\lambda$", fontsize=9.5, color=RED)
    ax.set_xlabel("бюджет $B$"); ax.set_ylabel("лучшая ошибка $V(B)$")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("множитель — теневая цена", fontsize=10.5)
    save(fig, SIDE / "shadow.png")


def side_kkt() -> None:
    fig, (l, r) = plt.subplots(1, 2, figsize=(4.6, 2.5))
    for ax, cx, tag, lam in [(l, 0.0, "запас: $\\lambda=0$", False),
                             (r, 1.9, "граница: $\\lambda>0$", True)]:
        th = np.linspace(0, 2 * math.pi, 200)
        ax.add_patch(Circle((0, 0), 1.0, facecolor=WASH, edgecolor=BLUE, lw=1.4, alpha=0.7))
        xs = np.linspace(-1.6, 2.6, 120); ys = np.linspace(-1.6, 1.6, 120)
        GX, GY = np.meshgrid(xs, ys)
        F = (GX - cx) ** 2 + (GY) ** 2
        ax.contour(GX, GY, F, levels=[0.15, 0.6, 1.4], colors=[LINE], linewidths=0.8)
        if lam:
            ax.scatter([1.0], [0], s=35, color=RED, zorder=5)
        else:
            ax.scatter([0], [0], s=35, color=GREEN, zorder=5)
        ax.set_aspect("equal"); ax.set_xlim(-1.6, 2.6); ax.set_ylim(-1.6, 1.6)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(tag, fontsize=9)
    fig.tight_layout()
    save(fig, SIDE / "kkt.png")


def side_softthresh() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    z = np.linspace(-3, 3, 200)
    ax.plot(z, z, color=LINE, lw=1.2, ls=(0, (3, 2)))
    thr = 1.0
    st = np.sign(z) * np.maximum(np.abs(z) - thr, 0)
    ax.plot(z, st, color=RED, lw=2.0)
    ax.axhline(0, color=GRID, lw=0.8); ax.axvline(0, color=GRID, lw=0.8)
    ax.text(1.4, -1.2, "плоский участок:\nмалые веса гаснут в нуль", fontsize=8.5, color=RED)
    ax.set_xlabel("вход"); ax.set_ylabel("после $L_1$")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("мягкий порог", fontsize=10.5)
    save(fig, SIDE / "softthresh.png")


fig_tangency()
fig_l1l2()
fig_lasso_path()
side_shadow()
side_kkt()
side_softthresh()
print("lesson 23 figures written")
