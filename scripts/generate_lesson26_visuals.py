"""Deterministic figures for lesson 26: saddles, games, adversarial attacks.

A robustness curve (how many correct iris predictions an FGSM attack breaks
as the budget grows), one worked adversarial example that flips at eps=0.25,
and the saddle geometry with gradient descent-ascent (bilinear game spirals
out, the x^2-y^2 saddle converges). Every quoted number is reproduced.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "26"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "26"

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


# ============================================================ iris classifier
FEATURES = ["длина\nчашелистика", "ширина\nчашелистика", "длина\nлепестка", "ширина\nлепестка"]


def train_iris():
    rows = [r for r in csv.reader((ROOT / "scripts" / "data" / "iris.data").open()) if r and len(r) == 5]
    X = np.array([[float(v) for v in r[:4]] for r in rows])
    labels = [r[4] for r in rows]
    classes = sorted(set(labels)); y = np.array([classes.index(l) for l in labels])
    mu, sd = X.mean(0), X.std(0); Xs = (X - mu) / sd
    W = np.zeros((3, 4)); b = np.zeros(3)

    def softmax(Z):
        Z = Z - Z.max(1, keepdims=True); E = np.exp(Z); return E / E.sum(1, keepdims=True)

    for _ in range(2000):
        P = softmax(Xs @ W.T + b); G = P.copy(); G[np.arange(len(y)), y] -= 1
        W -= 0.1 * (G.T @ Xs) / len(y); b -= 0.1 * G.mean(0)
    return Xs, y, W, b, mu, sd, classes


def softmax1(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()


def gradx(x, t, W, b):
    p = softmax1(x @ W.T + b); g = p.copy(); g[t] -= 1
    return g @ W


# ---------------------------- fig 26.1: robustness curve
def fig_robustness() -> None:
    Xs, y, W, b, mu, sd, classes = train_iris()
    pred = (Xs @ W.T + b).argmax(1)
    acc = (pred == y).mean()
    correct = np.where(pred == y)[0]
    print(f"iris acc={acc:.3f}, correct={len(correct)}")
    assert abs(acc - 0.973) < 0.01
    epss = np.linspace(0, 1.1, 23)
    frac = []
    for eps in epss:
        flips = 0
        for i in correct:
            xadv = Xs[i] + eps * np.sign(gradx(Xs[i], y[i], W, b))
            if (xadv @ W.T + b).argmax() != y[i]:
                flips += 1
        frac.append(100 * flips / len(correct))
    frac = np.array(frac)
    # check a couple of points
    def at(e): return frac[np.argmin(np.abs(epss - e))]
    print(f"eps0.3->{at(0.3):.0f}%  eps0.5->{at(0.5):.0f}%  eps0.8->{at(0.8):.0f}%")
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(epss, frac, color=RED, lw=2.4, marker="o", markersize=4)
    ax.fill_between(epss, 0, frac, color=RED, alpha=0.07)
    ax.axhline(0, color=LINE, lw=0.8)
    for e in [0.3, 0.5, 0.8]:
        v = at(e)
        ax.plot([e, e], [0, v], color=MUTED, lw=0.8, ls=(0, (2, 2)))
        ax.text(e, v + 3, f"{v:.0f}%", ha="center", fontsize=10.5, color=RED)
    ax.set_xlim(0, 1.1); ax.set_ylim(0, 105)
    ax.set_xlabel("бюджет искажения $\\varepsilon$  (в долях стандартного отклонения)")
    ax.set_ylabel("сломано верных ответов, %")
    ax.set_title("Классификатор точен на 97 %, но хрупок к искажению входа")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "robustness.png")


# ---------------------------- fig 26.2: one adversarial example
def fig_example() -> None:
    Xs, y, W, b, mu, sd, classes = train_iris()
    i, eps = 87, 0.25
    x = Xs[i]; g = np.sign(gradx(x, y[i], W, b)); xadv = x + eps * g
    p0 = softmax1(x @ W.T + b); padv = softmax1(xadv @ W.T + b)
    print(f"example {i}: true {classes[y[i]]} p={p0[y[i]]:.3f} -> adv predicts {classes[padv.argmax()]} p={padv.max():.3f}")
    assert padv.argmax() != y[i] and p0[y[i]] > 0.9
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 5.0), gridspec_kw={"width_ratios": [1.4, 1]})
    xp = np.arange(4); wbar = 0.38
    a1.bar(xp - wbar / 2, x, wbar, color=BLUE, label="исходный цветок")
    a1.bar(xp + wbar / 2, xadv, wbar, color=RED, label="после атаки ($\\varepsilon$=0,25)")
    a1.axhline(0, color=LINE, lw=0.8)
    a1.set_xticks(xp); a1.set_xticklabels(FEATURES, fontsize=9.5)
    a1.set_ylabel("признак (стандартизован)")
    a1.set_title("Крошечный сдвиг четырёх измерений", fontsize=12.5)
    a1.legend(loc="lower right", frameon=False, fontsize=10)
    a1.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); a1.set_axisbelow(True)
    # prediction bars
    a2.bar(np.arange(3) - wbar / 2, p0, wbar, color=BLUE, label="до")
    a2.bar(np.arange(3) + wbar / 2, padv, wbar, color=RED, label="после")
    a2.set_xticks(np.arange(3)); a2.set_xticklabels([c.replace("Iris-", "") for c in classes], fontsize=9.5)
    a2.set_ylim(0, 1); a2.set_ylabel("уверенность модели")
    a2.set_title("Ответ перевернулся", fontsize=12.5)
    a2.legend(loc="upper center", frameon=False, fontsize=10)
    a2.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); a2.set_axisbelow(True)
    pert_cm = eps * g * sd
    a1.text(0.5, -0.30, "сдвиг в сантиметрах: " +
            ", ".join(f"{v:+.2f}".replace(".", ",") for v in pert_cm),
            transform=a1.transAxes, ha="center", fontsize=9.5, color=MUTED)
    fig.suptitle("Один цветок ириса: незаметное искажение — уверенная ошибка", y=1.02, fontsize=14.5)
    fig.tight_layout()
    save(fig, OUT / "example.png")


# ---------------------------- fig 26.3: saddle and GDA
def fig_saddle() -> None:
    fig = plt.figure(figsize=(10.6, 4.8))
    # left: saddle surface x^2 - y^2
    axl = fig.add_subplot(1, 2, 1, projection="3d")
    u = np.linspace(-1.5, 1.5, 40); v = np.linspace(-1.5, 1.5, 40)
    U, V = np.meshgrid(u, v); Z = U ** 2 - V ** 2
    axl.plot_surface(U, V, Z, cmap="coolwarm", alpha=0.7, linewidth=0, rstride=2, cstride=2)
    axl.scatter([0], [0], [0], color=INK, s=40)
    axl.set_title("седло $F=x^2-y^2$", fontsize=12.5)
    axl.set_xticks([]); axl.set_yticks([]); axl.set_zticks([])
    axl.view_init(elev=26, azim=-52)
    # right: GDA trajectories on the two games (contours + paths)
    axr = fig.add_subplot(1, 2, 2)
    lr = 0.1
    # bilinear F=xy: min_x max_y -> x -= lr*y, y += lr*x  (spirals out)
    x, yv = 1.0, 0.6; bil = [(x, yv)]
    for _ in range(90):
        gx, gy = yv, x; x -= lr * gx; yv += lr * gy; bil.append((x, yv))
    bil = np.array(bil)
    # F=x^2-y^2: x -= lr*2x, y += lr*(-2y) (converges)
    x, yv = 1.2, 1.0; sad = [(x, yv)]
    for _ in range(90):
        x -= lr * 2 * x; yv += lr * (-2 * yv); sad.append((x, yv))
    sad = np.array(sad)
    print(f"GDA bilinear radius {np.hypot(*bil[0]):.2f}->{np.hypot(*bil[-1]):.2f}; saddle end ({sad[-1,0]:.3f},{sad[-1,1]:.3f})")
    assert np.hypot(*bil[-1]) > np.hypot(*bil[0]) and np.hypot(*sad[-1]) < 0.05
    axr.axhline(0, color=LINE, lw=0.8); axr.axvline(0, color=LINE, lw=0.8)
    axr.plot(bil[:, 0], bil[:, 1], color=RED, lw=1.8, label="игра $F=xy$: кружит и расходится")
    axr.plot(sad[:, 0], sad[:, 1], color=GREEN, lw=1.8, marker="o", markersize=2.5, label="седло $F=x^2-y^2$: сходится")
    axr.scatter([0], [0], color=INK, s=45, zorder=5)
    axr.text(0.06, 0.08, "равновесие", fontsize=10, color=INK)
    axr.set_xlim(-3.4, 3.4); axr.set_ylim(-3.4, 3.4); axr.set_aspect("equal")
    axr.set_xlabel("ход защитника $x$"); axr.set_ylabel("ход атакующего $y$")
    axr.set_title("спуск-подъём: сходится или кружит", fontsize=12.5)
    axr.legend(loc="upper left", frameon=False, fontsize=9.5)
    axr.grid(True, color=GRID, lw=0.4, alpha=0.4); axr.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "saddle.png")


# ------------------------------------------------ margins
def side_minimax() -> None:
    fig, ax = plt.subplots(figsize=(4.1, 2.8))
    x = np.linspace(-1.4, 1.4, 100)
    for yv, col in [(-0.8, LINE), (0.0, MUTED), (0.8, LINE)]:
        ax.plot(x, x ** 2 - yv ** 2 + 0.0 * x, color=col, lw=1.2)
    ax.plot(x, x ** 2, color=BLUE, lw=2.0)
    ax.axvline(0, color=GRID, lw=0.8)
    ax.scatter([0], [0], color=INK, s=35, zorder=5)
    ax.text(0.1, 0.15, "минимакс", fontsize=9, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("минимакс фон Неймана", fontsize=10.5)
    save(fig, SIDE / "minimax.png")


def side_gan() -> None:
    fig, ax = plt.subplots(figsize=(4.1, 2.5))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    ax.add_patch(plt.Rectangle((0.6, 1.6), 2.6, 1.8, fc=WASH, ec=BLUE, lw=1.4))
    ax.text(1.9, 2.5, "генератор\n(подделывает)", ha="center", fontsize=8.5, color=BLUE)
    ax.add_patch(plt.Rectangle((6.8, 1.6), 2.6, 1.8, fc=WASH, ec=RED, lw=1.4))
    ax.text(8.1, 2.5, "дискриминатор\n(различает)", ha="center", fontsize=8.5, color=RED)
    ax.annotate("", xy=(6.7, 2.7), xytext=(3.3, 2.7),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4, mutation_scale=12))
    ax.annotate("", xy=(3.3, 2.1), xytext=(6.7, 2.1),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4, mutation_scale=12))
    ax.text(5, 3.9, "две сети играют друг против друга", ha="center", fontsize=8.5, color=MUTED)
    ax.set_title("состязательные сети", fontsize=10.5)
    save(fig, SIDE / "gan.png")


def side_pursuit() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    t = np.linspace(0, 1, 30)
    ev = np.array([1 + 7 * t, 5 - 1.5 * np.sin(3 * t)]).T
    pu = np.array([1 + 6.5 * t, 1 + 2.5 * t]).T
    ax.plot(ev[:, 0], ev[:, 1], color=BLUE, lw=1.6)
    ax.plot(pu[:, 0], pu[:, 1], color=RED, lw=1.6)
    ax.scatter(*ev[-1], color=BLUE, s=35); ax.text(ev[-1, 0] - 0.3, ev[-1, 1] + 0.5, "убегающий", fontsize=8, color=BLUE, ha="right")
    ax.scatter(*pu[-1], color=RED, s=35); ax.text(pu[-1, 0] - 0.3, pu[-1, 1] - 0.7, "преследователь", fontsize=8, color=RED, ha="right")
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 6.3)
    ax.set_title("дифференциальная игра", fontsize=10.5)
    save(fig, SIDE / "pursuit.png")


fig_robustness()
fig_example()
fig_saddle()
side_minimax()
side_gan()
side_pursuit()
print("lesson 26 figures written")
