"""Deterministic figures for lesson 24: gradient descent and SGD.

Full-batch, mini-batch and single-sample descent on the real Bike Sharing
MSE bowl (17 379 rows); loss against examples processed; and the heart of
SGD — how well a random batch's gradient points downhill as a function of
batch size. Every quoted number is reproduced and asserted.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "24"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "24"

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


# ============================================================ bike data
def bike():
    rows = list(csv.DictReader((ROOT / "scripts" / "data" / "bike-sharing-hour.csv").open()))
    temp = np.array([float(r["temp"]) for r in rows])
    cnt = np.array([float(r["cnt"]) for r in rows])
    x = (temp - temp.mean()) / temp.std()
    y = (cnt - cnt.mean()) / cnt.std()
    return x, y


X, Y = bike()
N = len(X)
OPT_W1 = float(np.corrcoef(X, Y)[0, 1])
FLOOR = 1 - OPT_W1 ** 2
print(f"N={N}  opt w1={OPT_W1:.4f}  floor loss={FLOOR:.4f}")
assert abs(OPT_W1 - 0.4048) < 1e-3 and abs(FLOOR - 0.8362) < 1e-3


def loss(w0, w1):
    r = w0 + w1 * X - Y
    return float(np.mean(r * r))


def grad(w0, w1, idx):
    xi, yi = X[idx], Y[idx]
    r = w0 + w1 * xi - yi
    return 2 * r.mean(), 2 * (r * xi).mean()


def descend(kind, eta, epochs, bs, seed):
    """Return path of (w0,w1) and list of (examples_seen, loss)."""
    r = np.random.default_rng(seed)
    w0, w1 = 1.5, -0.6
    path = [(w0, w1)]
    curve = [(1, loss(w0, w1))]
    seen = 0
    for _ in range(epochs):
        if kind == "full":
            g = grad(w0, w1, np.arange(N)); w0 -= eta * g[0]; w1 -= eta * g[1]
            seen += N; path.append((w0, w1)); curve.append((seen, loss(w0, w1)))
        else:
            perm = r.permutation(N)
            for s in range(0, N - bs + 1, bs):
                idx = perm[s:s + bs]
                g = grad(w0, w1, idx); w0 -= eta * g[0]; w1 -= eta * g[1]
                seen += bs
                if (s // bs) % 6 == 0:
                    path.append((w0, w1)); curve.append((seen, loss(w0, w1)))
    return np.array(path), np.array(curve)


# ---------------------------- fig 24.1: three descent paths on the bowl
def fig_paths() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    w0g = np.linspace(-0.7, 1.7, 140); w1g = np.linspace(-0.8, 1.0, 140)
    G0, G1 = np.meshgrid(w0g, w1g)
    Z = np.zeros_like(G0)
    for i in range(G0.shape[0]):
        for j in range(G0.shape[1]):
            Z[i, j] = loss(G0[i, j], G1[i, j])
    ax.contour(G0, G1, Z, levels=12, colors=[LINE], linewidths=0.9)

    full, _ = descend("full", 0.3, 40, None, 0)
    mb, _ = descend("mb", 0.2, 1, 64, 2)
    sgd, _ = descend("sgd", 0.05, 1, 1, 1)
    ax.plot(sgd[::3, 0], sgd[::3, 1], color=GOLD, lw=0.7, alpha=0.5, zorder=2)
    ax.plot(mb[:, 0], mb[:, 1], color=BLUE, lw=1.4, alpha=0.95, zorder=3)
    ax.plot(full[:, 0], full[:, 1], color=INK, lw=2.0, marker="o", markersize=3, zorder=4)
    ax.scatter([1.5], [-0.6], s=70, color=RED, zorder=6)
    ax.text(1.5, -0.72, "старт", color=RED, fontsize=10.5, ha="center")
    ax.scatter([0], [OPT_W1], s=90, color=GREEN, zorder=7, edgecolor=PAPER, linewidth=1.2)
    ax.annotate("минимум", xy=(0, OPT_W1), xytext=(-0.55, -0.35),
                color=GREEN, fontsize=11, zorder=7,
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=1.0))
    # legend proxies
    ax.plot([], [], color=INK, lw=2.0, label="полный градиент (гладко)")
    ax.plot([], [], color=BLUE, lw=1.6, label="мини-батч 64 (умеренно)")
    ax.plot([], [], color=GOLD, lw=1.2, label="SGD по одному (шумно)")
    ax.legend(loc="upper left", frameon=False, fontsize=10.5)
    ax.set_xlabel("сдвиг $w_0$"); ax.set_ylabel("наклон $w_1$")
    ax.set_title("Три спуска по одной чаше потерь")
    save(fig, OUT / "paths.png")
    print(f"paths: full end w1={full[-1,1]:.3f}, mb end w1={mb[-1,1]:.3f}, sgd end w1={sgd[-1,1]:.3f}")


# ---------------------------- fig 24.2: loss vs examples processed
def _ema(curve, alpha=0.04):
    out = curve.copy().astype(float)
    s = out[0, 1]
    for i in range(len(out)):
        s = alpha * out[i, 1] + (1 - alpha) * s
        out[i, 1] = s
    return out


def fig_efficiency() -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    _, cf = descend("full", 0.3, 200, None, 0)
    _, cm = descend("mb", 0.2, 3, 64, 2)
    _, cs = descend("sgd", 0.05, 2, 1, 1)
    ax.axhline(FLOOR, color=GREEN, lw=1.2, ls=(0, (5, 3)))
    ax.text(1.4, FLOOR + 0.02, "минимально достижимая потеря", color=GREEN, fontsize=10)
    es, em = _ema(cs), _ema(cm)
    ax.plot(es[:, 0], es[:, 1], color=GOLD, lw=1.8, label="SGD по одному (сглажено)")
    ax.plot(em[:, 0], em[:, 1], color=BLUE, lw=1.8, label="мини-батч 64 (сглажено)")
    ax.plot(cf[:, 0], cf[:, 1], color=INK, lw=2.0, marker="o", markersize=2.5, label="полный градиент")
    ax.axvline(N, color=MUTED, lw=1.0, ls=(0, (2, 2)))
    ax.text(N * 0.9, 1.36, "один проход\nпо данным", color=MUTED, fontsize=9.5, ha="right")
    ax.set_xscale("log")
    ax.set_xlim(1, 3e5)
    ax.set_ylim(0.80, 1.42)
    ax.set_xlabel("обработано примеров (журнал)")
    ax.set_ylabel("потеря на всех данных")
    ax.set_title("Кто быстрее спускается на единицу вычислений")
    ax.legend(loc="center right", frameon=False, fontsize=10.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "efficiency.png")
    # headline: loss after exactly one epoch (raw iterate)
    lf = cf[cf[:, 0] <= N][-1, 1]
    lm = cm[cm[:, 0] <= N][-1, 1]
    print(f"after 1 epoch: full={lf:.3f}, mb={lm:.3f} (floor {FLOOR:.3f})")
    assert lm < 0.9 and lf > 1.2  # minibatch near floor, full-batch barely moved


# ---------------------------- fig 24.3: gradient alignment vs batch size
def fig_alignment() -> None:
    w0t, w1t = 0.8, 0.1
    gt = np.array(grad(w0t, w1t, np.arange(N)))
    gtn = gt / np.linalg.norm(gt)
    r = np.random.default_rng(7)
    sizes = [1, 8, 64, 512]
    means, downhill = [], []
    for bs in sizes:
        coss = []
        for _ in range(600):
            idx = r.integers(0, N, size=bs)
            g = np.array(grad(w0t, w1t, idx))
            coss.append(float(g @ gtn / np.linalg.norm(g)))
        coss = np.array(coss)
        means.append(coss.mean()); downhill.append((coss > 0).mean())
    print("alignment means:", [round(m, 3) for m in means], "downhill:", [round(d, 2) for d in downhill])
    assert means[0] > 0.4 and means[2] > 0.95 and downhill[0] > 0.7
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    xs = np.arange(len(sizes))
    bars = ax.bar(xs, means, width=0.55, color=[GOLD, BLUE, GREEN, INK], zorder=3)
    for i, (m, d) in enumerate(zip(means, downhill)):
        ax.text(i, m + 0.02, f"{m:.2f}".replace(".", ","), ha="center", fontsize=11, color=INK)
        ax.text(i, 0.04, f"вниз\n{int(round(d*100))}%", ha="center", fontsize=9.5, color=PAPER, va="bottom")
    ax.axhline(1.0, color=GREEN, lw=1.0, ls=(0, (4, 3)))
    ax.text(-0.42, 1.03, "истинный градиент по всем 17 379 строкам", color=GREEN, fontsize=9.5, ha="left")
    ax.set_xticks(xs); ax.set_xticklabels([f"батч {s}" for s in sizes])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("средняя близость к истинному\nнаправлению (косинус)")
    ax.set_title("Даже один пример чаще ведёт вниз, чем вверх")
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "alignment.png")


# ------------------------------------------------ margins
def side_schedule() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    epochs = 3
    for tag, decay, col in [("постоянный шаг", False, RED), ("убывающий шаг", True, GREEN)]:
        r = np.random.default_rng(3); w0, w1 = 1.5, -0.6; trace = []
        step = 0
        for _ in range(epochs):
            for i in r.permutation(N):
                eta = 0.06 if not decay else 0.3 / (1 + step * 0.004)
                g = grad(w0, w1, np.array([i])); w0 -= eta * g[0]; w1 -= eta * g[1]; step += 1
                if step % 60 == 0:
                    trace.append(w1)
        ax.plot(trace, color=col, lw=1.0, alpha=0.9, label=tag)
    ax.axhline(OPT_W1, color=MUTED, lw=1.0, ls=(0, (3, 2)))
    ax.text(len(trace) * 0.02, OPT_W1 + 0.08, "минимум", color=MUTED, fontsize=8.5)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    ax.set_ylim(-0.1, 1.05)
    ax.set_xticks([]); ax.set_ylabel("наклон $w_1$", fontsize=9)
    ax.set_title("шаг надо гасить", fontsize=10.5)
    save(fig, SIDE / "schedule.png")


def side_noise() -> None:
    fig, ax = plt.subplots(figsize=(4.1, 2.7))
    U = lambda t: 0.5 * (t ** 2 - 1) ** 2 - 0.28 * t  # shallow left well, deep right well
    t = np.linspace(-2.2, 2.3, 400)
    ax.plot(t, U(t), color=LINE, lw=1.8)
    tl = np.linspace(-1.6, -0.2, 200); left = tl[np.argmin(U(tl))]
    tr = np.linspace(0.2, 1.6, 200); right = tr[np.argmin(U(tr))]
    ax.scatter([left], [U(left)], s=45, color=RED, zorder=5)
    ax.scatter([right], [U(right)], s=45, color=GREEN, zorder=5)
    ax.add_patch(FancyArrowPatch((left + 0.05, U(left) + 0.12), (right - 0.15, U(right) + 0.22),
                                 connectionstyle="arc3,rad=-0.4",
                                 arrowstyle="-|>", color=GOLD, lw=1.6, mutation_scale=12))
    ax.text(-1.35, U(0) + 0.12, "шум SGD выбивает\nиз мелкой ямы", fontsize=8.5, color=GOLD, ha="left")
    ax.text(right, U(right) - 0.28, "лучший\nминимум", fontsize=8, color=GREEN, ha="center", va="top")
    ax.set_ylim(U(right) - 0.55, U(0) + 0.45)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("шум помогает искать", fontsize=10.5)
    save(fig, SIDE / "noise.png")


def side_minibatch() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    rng = np.random.default_rng(5)
    pts = rng.uniform([0.4, 0.6], [9.6, 5.4], size=(120, 2))
    ax.scatter(pts[:, 0], pts[:, 1], s=10, color=FAINT, zorder=2)
    pick = rng.choice(120, 8, replace=False)
    ax.scatter(pts[pick, 0], pts[pick, 1], s=42, color=BLUE, zorder=4, edgecolor=PAPER, linewidth=0.6)
    ax.text(5, 5.7, "весь набор данных", fontsize=9.5, color=MUTED, ha="center")
    ax.text(5, 0.15, "случайная горсть — мини-батч", fontsize=9, color=BLUE, ha="center")
    save(fig, SIDE / "minibatch.png")


fig_paths()
fig_efficiency()
fig_alignment()
side_schedule()
side_noise()
side_minibatch()
print("lesson 24 figures written")
