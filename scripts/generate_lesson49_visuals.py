"""Deterministic figures for lesson 49: linear regression from normal noise.

A least-squares line fitted to REAL data (bike rides against temperature) with the
residuals drawn, least squares as an orthogonal projection onto the column space, and
four residual-diagnostic panels that one summary RMSE would hide. Numbers asserted.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

ROOT = Path(__file__).resolve().parents[1]
BIKE = ROOT / "scripts" / "data" / "bike-sharing-hour.csv"
OUT = ROOT / "public" / "figures" / "lessons" / "49"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "49"

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


def bike_temp_rides():
    temp, cnt = [], []
    with open(BIKE) as f:
        for row in csv.DictReader(f):
            temp.append(float(row["temp"]) * 41)   # denormalise to Celsius
            cnt.append(int(row["cnt"]))
    return np.array(temp), np.array(cnt)


# ---------------------------------------- fig 49.1: least-squares line on real data
def fig_regression_real() -> None:
    x, y = bike_temp_rides()
    w1, w0 = np.polyfit(x, y, 1)
    yhat = w0 + w1 * x
    ss_res = np.sum((y - yhat) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"regression: rides = {w0:.0f} + {w1:.1f}*temp, R2={r2:.3f}")
    assert w1 > 0 and 0.1 < r2 < 0.3
    rng = np.random.default_rng(49)
    idx = rng.choice(len(x), 500, replace=False)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.scatter(x[idx], y[idx], s=12, color=BLUE, alpha=0.35, edgecolors="none")
    xs = np.array([x.min(), x.max()])
    lbl = f"$\\hat y={w0:.0f}{w1:+.0f}\\,x$" if abs(w0) >= 1 else f"$\\hat y={w1:.0f}\\,x$"
    ax.plot(xs, w0 + w1 * xs, color=RED, lw=2.6, label=lbl)
    # a few residual segments
    for i in idx[:40]:
        ax.plot([x[i], x[i]], [y[i], w0 + w1 * x[i]], color=MUTED, lw=0.5, alpha=0.5)
    ax.set_xlabel("температура, °C"); ax.set_ylabel("число поездок за час")
    ax.set_title(f"Линия наименьших квадратов на реальных данных ($R^2={r2:.2f}$)")
    ax.legend(loc="upper left", frameon=False, fontsize=12)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "regression_real.png")


# ---------------------------------------- fig 49.2: least squares as orthogonal projection
def fig_projection() -> None:
    from matplotlib.patches import Polygon
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1, 11); ax.set_ylim(-1, 9)
    # column-space plane as a 2D parallelogram (perspective)
    plane = np.array([[0.5, 1.0], [8.5, 1.0], [10.0, 3.0], [2.0, 3.0]])
    ax.add_patch(Polygon(plane, closed=True, fc=BLUE, ec=LINE, alpha=0.12, lw=1.2))
    ax.text(8.6, 1.3, "столбцовое пространство $X$\n(все прогнозы $Xw$)", fontsize=10.5, color=MUTED)
    origin = np.array([3.0, 2.0])
    yhat = np.array([6.6, 2.4])       # foot of perpendicular, inside plane
    y = np.array([6.6 + 0.9, 2.4 + 4.2])   # point y above the plane
    # basis columns
    ax.add_patch(FancyArrowPatch(origin, origin + np.array([2.6, 0.15]), arrowstyle="-|>", mutation_scale=13, color=MUTED, lw=1.6))
    ax.add_patch(FancyArrowPatch(origin, origin + np.array([0.9, 0.9]), arrowstyle="-|>", mutation_scale=13, color=MUTED, lw=1.6))
    ax.text(origin[0] + 2.7, origin[1] - 0.05, "$x_1$", fontsize=11, color=MUTED)
    ax.text(origin[0] + 0.6, origin[1] + 1.0, "$x_2$", fontsize=11, color=MUTED)
    # y, yhat, residual
    ax.add_patch(FancyArrowPatch(origin, y, arrowstyle="-|>", mutation_scale=16, color=INK, lw=2.4))
    ax.add_patch(FancyArrowPatch(origin, yhat, arrowstyle="-|>", mutation_scale=16, color=BLUE, lw=2.4))
    ax.add_patch(FancyArrowPatch(yhat, y, arrowstyle="-|>", mutation_scale=16, color=RED, lw=2.4))
    # right-angle marker at yhat
    d1 = (origin - yhat); d1 = d1 / np.linalg.norm(d1) * 0.5
    d2 = (y - yhat); d2 = d2 / np.linalg.norm(d2) * 0.5
    corner = yhat + d1 + d2
    ax.plot([yhat[0] + d1[0], corner[0]], [yhat[1] + d1[1], corner[1]], color=RED, lw=1.0)
    ax.plot([yhat[0] + d2[0], corner[0]], [yhat[1] + d2[1], corner[1]], color=RED, lw=1.0)
    ax.text(y[0] + 0.15, y[1], "$y$ (наблюдения)", fontsize=13, color=INK)
    ax.text(yhat[0] + 0.15, yhat[1] - 0.5, "$\\hat y=Xw$ (проекция)", fontsize=12, color=BLUE)
    ax.text((yhat[0] + y[0]) / 2 + 0.2, (yhat[1] + y[1]) / 2, "$r=y-\\hat y$", fontsize=12, color=RED)
    ax.set_title("Наименьшие квадраты — ортогональная проекция: $X^\\top r=0$", fontsize=14)
    save(fig, OUT / "projection.png")
    print("projection drawn")


# ---------------------------------------- fig 49.3: four residual diagnostics
def fig_diagnostics() -> None:
    rng = np.random.default_rng(3)
    n = 200
    yhat = np.linspace(0, 10, n)
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.4))
    # A: good
    axes[0, 0].scatter(yhat, rng.normal(0, 1, n), s=10, color=GREEN, alpha=0.6)
    axes[0, 0].set_title("A. случайное облако — модель в порядке", fontsize=10.5)
    # B: curvature
    axes[0, 1].scatter(yhat, 0.15 * (yhat - 5) ** 2 - 1 + rng.normal(0, 0.5, n), s=10, color=GOLD, alpha=0.6)
    axes[0, 1].set_title("B. дуга — пропущена нелинейность", fontsize=10.5)
    # C: fan
    axes[1, 0].scatter(yhat, rng.normal(0, 0.15 + 0.35 * yhat, n), s=10, color=RED, alpha=0.6)
    axes[1, 0].set_title("C. веер — дисперсия растёт", fontsize=10.5)
    # D: time wave
    t = np.linspace(0, 4 * np.pi, n)
    axes[1, 1].scatter(range(n), 1.5 * np.sin(t) + rng.normal(0, 0.4, n), s=10,
                       c=range(n), cmap="viridis", alpha=0.7)
    axes[1, 1].set_title("D. волна во времени — зависимость", fontsize=10.5)
    for ax in axes.ravel():
        ax.axhline(0, color=INK, lw=0.8, ls=(0, (3, 3)))
        ax.set_xlabel("прогноз / время", fontsize=9); ax.set_ylabel("остаток", fontsize=9)
        ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    fig.suptitle("Один RMSE не различает эти четыре поломки", y=1.02, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "diagnostics.png")
    print("diagnostics drawn")


# ---------------------------------------- margins
def side_leverage() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    x = np.array([-1, 0, 1.0]); y = np.array([-1, 0, 1.0])
    ax.scatter(x, y, s=40, color=BLUE, zorder=5)
    xs = np.array([-1.5, 10.5])
    ax.plot(xs, xs, color=BLUE, lw=1.6, label="без выброса")
    x2 = np.append(x, 10); y2 = np.append(y, 0)
    w = np.sum(x2 * y2) / np.sum(x2 ** 2)
    ax.scatter(10, 0, s=55, color=RED, zorder=5)
    ax.plot(xs, w * xs, color=RED, lw=1.6, ls=(0, (5, 3)), label="с выбросом")
    ax.set_xlabel("x", fontsize=9); ax.set_ylabel("y", fontsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.set_title("одна далёкая точка кладёт прямую", fontsize=9.5)
    save(fig, SIDE / "leverage.png")


def side_band() -> None:
    rng = np.random.default_rng(5)
    x = rng.uniform(2, 8, 40); y = 1 + 0.5 * x + rng.normal(0, 0.6, 40)
    xs = np.linspace(0, 10, 100)
    w1, w0 = np.polyfit(x, y, 1)
    xm = x.mean(); sxx = np.sum((x - xm) ** 2); s = 0.6
    se = s * np.sqrt(1 / len(x) + (xs - xm) ** 2 / sxx)
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    ax.scatter(x, y, s=12, color=BLUE, alpha=0.5)
    ax.plot(xs, w0 + w1 * xs, color=INK, lw=1.6)
    ax.fill_between(xs, w0 + w1 * xs - 2 * se, w0 + w1 * xs + 2 * se, color=RED, alpha=0.15)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("интервал шире вдали от данных", fontsize=9.5)
    save(fig, SIDE / "band.png")


def side_noise_loss() -> None:
    e = np.linspace(-3, 3, 200)
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    ax.plot(e, e ** 2, color=BLUE, lw=2.0, label="нормальный: квадрат")
    ax.plot(e, np.abs(e) * 2, color=RED, lw=2.0, ls=(0, (5, 3)), label="Лаплас: модуль")
    ax.set_xlabel("остаток e", fontsize=9); ax.set_yticks([])
    ax.legend(loc="upper center", frameon=False, fontsize=8)
    ax.set_title("модель шума задаёт потерю", fontsize=9.5)
    save(fig, SIDE / "noise.png")
    print("noise drawn")


fig_regression_real()
fig_projection()
fig_diagnostics()
side_leverage()
side_band()
side_noise_loss()
print("lesson 49 figures written")
