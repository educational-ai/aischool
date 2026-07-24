"""Deterministic figures for lesson 09: clustering on Gapminder 2007.

Implements k-means, silhouette, stability and average-linkage agglomeration
from scratch so every number quoted in the lesson is reproduced here.
"""

from __future__ import annotations

import csv
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "09"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "09"
DATA = ROOT / "scripts" / "data" / "gapminder.csv"

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


ROWS = [r for r in csv.DictReader(DATA.open()) if r["year"] == "2007"]
GDP = np.array([float(r["gdpPercap"]) for r in ROWS])
LIFE = np.array([float(r["lifeExp"]) for r in ROWS])
NAMES = [r["country"] for r in ROWS]

Z_GDP = (GDP - GDP.mean()) / GDP.std(ddof=1)
Z_LIFE = (LIFE - LIFE.mean()) / LIFE.std(ddof=1)
Z = np.stack([Z_GDP, Z_LIFE], axis=1)
RAW = np.stack([GDP, LIFE], axis=1)


def kmeans(points: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    cents = points[rng.sample(range(len(points)), k)].copy()
    for _ in range(60):
        d2 = ((points[:, None, :] - cents[None, :, :]) ** 2).sum(-1)
        labels = d2.argmin(1)
        for j in range(k):
            if (labels == j).any():
                cents[j] = points[labels == j].mean(0)
    return labels


def silhouette(points: np.ndarray, labels: np.ndarray) -> float:
    dist = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1))
    vals = []
    for i in range(len(points)):
        own = labels == labels[i]
        a = dist[i, own].sum() / max(own.sum() - 1, 1)
        b = min(
            dist[i, labels == c].mean()
            for c in set(labels.tolist())
            if c != labels[i]
        )
        vals.append((b - a) / max(a, b))
    return float(np.mean(vals))


def inertia(points: np.ndarray, labels: np.ndarray) -> float:
    total = 0.0
    for j in set(labels.tolist()):
        cluster = points[labels == j]
        total += ((cluster - cluster.mean(0)) ** 2).sum()
    return total


def agreement(l1: np.ndarray, l2: np.ndarray) -> float:
    same1 = l1[:, None] == l1[None, :]
    same2 = l2[:, None] == l2[None, :]
    mask = ~np.eye(len(l1), dtype=bool)
    return float((same1 == same2)[mask].mean())


LAB3 = kmeans(Z, 3, 9)
# Order clusters: 0 = shortest lives, 2 = longest.
order = np.argsort([LIFE[LAB3 == j].mean() for j in range(3)])
rank = {order[i]: i for i in range(3)}
LAB3 = np.array([rank[v] for v in LAB3])
CLUSTER_COLORS = [RED, GOLD, GREEN]

RU = {
    "Botswana": "Ботсвана", "Equatorial Guinea": "Экв. Гвинея",
    "South Africa": "ЮАР", "Cuba": "Куба", "Costa Rica": "Коста-Рика",
    "Norway": "Норвегия", "United States": "США", "Japan": "Япония",
    "Korea, Rep.": "Юж. Корея", "China": "Китай", "India": "Индия",
    "Brazil": "Бразилия", "Poland": "Польша", "Turkey": "Турция",
    "Ethiopia": "Эфиопия", "Nigeria": "Нигерия", "Afghanistan": "Афганистан",
    "Germany": "Германия", "Mexico": "Мексика", "Vietnam": "Вьетнам",
    "Ireland": "Ирландия", "Argentina": "Аргентина",
}


# ---------------------------------------------------------------- figure 9.1
def raw_vs_scaled() -> None:
    lab_raw = kmeans(RAW, 3, 9)
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.6))
    for ax, labels, title in [
        (axes[0], lab_raw, "Сырые единицы: режет только доход"),
        (axes[1], LAB3, "После стандартизации: обе оси в деле"),
    ]:
        for j in set(labels.tolist()):
            mask = labels == j
            color = CLUSTER_COLORS[j % 3]
            ax.scatter(GDP[mask] / 1000, LIFE[mask], s=26, color=color,
                       alpha=0.8, lw=0)
        ax.set_xlabel("ВВП на душу, тыс. долларов")
        ax.grid(color=GRID, lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=13, pad=10)
    axes[0].set_ylabel("продолжительность жизни, лет")
    axes[0].annotate("один «бедный» кластер:\nжизнь от 39,6 до 78,8 лет",
                     xy=(6, 55), xytext=(16, 46), fontsize=10.2, color=INK,
                     bbox={"boxstyle": "round,pad=0.3", "facecolor": PAPER,
                           "edgecolor": LINE, "alpha": 0.94},
                     arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.1,
                                 "shrinkB": 4})
    save(fig, OUT / "raw-vs-scaled.png")


# ---------------------------------------------------------------- figure 9.2
def k_choice() -> None:
    ks = range(2, 7)
    js, sils = [], []
    for k in ks:
        best = None
        for seed in range(5):
            lab = kmeans(Z, k, seed)
            val = inertia(Z, lab)
            if best is None or val < best[0]:
                best = (val, lab)
        js.append(best[0])
        sils.append(silhouette(Z, best[1]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.8))
    ax1.plot(list(ks), js, "o-", color=BLUE, lw=2)
    ax1.set_xlabel("число кластеров $K$")
    ax1.set_ylabel("$J(K)$")
    ax1.set_title("Локоть: экономия выдыхается после трёх",
                  loc="left", fontweight="bold", fontsize=13, pad=10)
    ax2.plot(list(ks), sils, "o-", color=GREEN, lw=2)
    best_k = list(ks)[int(np.argmax(sils))]
    ax2.plot(best_k, max(sils), "o", color=RED, ms=10, mfc="none", mew=2)
    sil_str = f"{max(sils):.3f}".replace(".", ",")
    ax2.annotate(f"максимум {sil_str}", xy=(best_k, max(sils)),
                 xytext=(best_k + 0.5, max(sils) - 0.008), fontsize=10.4, color=RED)
    ax2.set_xlabel("число кластеров $K$")
    ax2.set_ylabel("средний силуэт")
    ax2.set_title("Силуэт: умеет и падать",
                  loc="left", fontweight="bold", fontsize=13, pad=10)
    for ax in (ax1, ax2):
        ax.grid(color=GRID, lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks(list(ks))
    save(fig, OUT / "k-choice.png")


# ---------------------------------------------------------------- figure 9.3
def stability_curve() -> None:
    ks = range(2, 7)
    ags = []
    for k in ks:
        labs = [kmeans(Z, k, s) for s in (1, 2, 3, 4, 5)]
        vals = [agreement(labs[a], labs[b])
                for a in range(5) for b in range(a + 1, 5)]
        ags.append(st.mean(vals))
    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    ax.plot(list(ks), ags, "o-", color=BLUE, lw=2.2, ms=6)
    ax.axhline(1.0, color=GRID, lw=1)
    ax.annotate("две равноценные двойки:\nразрез по доходу или по жизни",
                xy=(2, ags[0]), xytext=(2.4, ags[0] - 0.03), fontsize=10.4,
                color=RED,
                arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 1.1,
                            "shrinkB": 5})
    ax.annotate("пять запусков — один мир", xy=(3, ags[1]),
                xytext=(3.3, 0.955), fontsize=10.4, color=GREEN,
                arrowprops={"arrowstyle": "-|>", "color": GREEN, "lw": 1.1,
                            "shrinkB": 5})
    ax.set_xticks(list(ks))
    ax.set_ylim(0.7, 1.03)
    ax.set_xlabel("число кластеров $K$")
    ax.set_ylabel("согласие запусков")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Среднее попарное согласие пяти случайных стартов",
                 loc="left", fontweight="bold", pad=12)
    save(fig, OUT / "stability-curve.png")


# ---------------------------------------------------------------- figure 9.4
def named_clusters() -> None:
    fig, ax = plt.subplots(figsize=(12.4, 6.4))
    for j in range(3):
        mask = LAB3 == j
        ax.scatter(GDP[mask], LIFE[mask], s=30, color=CLUSTER_COLORS[j],
                   alpha=0.8, lw=0)
    ax.set_xscale("log")
    ax.set_xticks([500, 1000, 2000, 5000, 10000, 20000, 40000],
                  ["0,5", "1", "2", "5", "10", "20", "40"])
    ax.set_xlabel("ВВП на душу, тыс. долларов (лог-шкала)")
    ax.set_ylabel("продолжительность жизни, лет")

    bbox = {"boxstyle": "round,pad=0.3", "facecolor": PAPER,
            "edgecolor": LINE, "alpha": 0.94}
    marks = {
        "Botswana": (-10, -34), "Equatorial Guinea": (12, -46),
        "South Africa": (-72, -22), "Cuba": (-30, 26), "Costa Rica": (16, 20),
    }
    for name, (dx, dy) in marks.items():
        i = NAMES.index(name)
        ax.plot(GDP[i], LIFE[i], "o", ms=11, mfc="none", mec=INK, mew=1.4)
        ax.annotate(RU[name], xy=(GDP[i], LIFE[i]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=10.2, color=INK,
                    bbox=bbox, zorder=6,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.9,
                                "shrinkB": 5})
    ax.grid(color=GRID, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Три кластера 2007 года и их диссиденты",
                 loc="left", fontweight="bold", pad=12)
    save(fig, OUT / "named-clusters.png")


# ---------------------------------------------------------------- figure 9.5
def null_silhouette() -> None:
    vals = []
    for trial in range(20):
        rng = random.Random(100 + trial)
        pts = np.array([[rng.random(), rng.random()] for _ in range(142)])
        lab = kmeans(pts, 3, 7)
        vals.append(silhouette(pts, lab))
    real = silhouette(Z, LAB3)

    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.hist(vals, bins=np.arange(0.34, 0.46, 0.01), color=FAINT, alpha=0.9,
            label="20 шумовых миров")
    ax.axvline(real, color=RED, lw=2.4)
    real_str = f"{real:.3f}".replace(".", ",")
    ax.text(real - 0.004, 4.6, f"настоящие страны: {real_str}",
            fontsize=11.2, color=RED, ha="right", fontweight="bold")
    ax.set_xlabel("средний силуэт при $K=3$")
    ax.set_ylabel("число запусков")
    ax.set_xlim(0.33, 0.63)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", frameon=False)
    ax.set_title("Шум тоже даёт «кластеры» — но не такие отчётливые",
                 loc="left", fontweight="bold", pad=12)
    save(fig, OUT / "null-silhouette.png")


# ---------------------------------------------------------------- figure 9.6
def dendrogram() -> None:
    chosen = ["Norway", "Ireland", "United States", "Japan", "Germany",
              "Korea, Rep.", "Poland", "Argentina", "Costa Rica", "Cuba",
              "Mexico", "Turkey", "Brazil", "China", "Vietnam", "India",
              "South Africa", "Botswana", "Nigeria", "Afghanistan"]
    idx = [NAMES.index(c) for c in chosen]
    pts = Z[idx]
    n = len(idx)

    clusters = [[i] for i in range(n)]
    heights = []
    merges = []
    active = list(range(n))
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    next_id = n
    dist = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))

    def cdist(a: int, b: int) -> float:
        ma, mb = members[a], members[b]
        return float(np.mean([dist[i, j] for i in ma for j in mb]))

    node_pos: dict[int, float] = {}
    node_h: dict[int, float] = {i: 0.0 for i in range(n)}
    children: dict[int, tuple[int, int]] = {}

    while len(active) > 1:
        best = None
        for ai in range(len(active)):
            for bi in range(ai + 1, len(active)):
                d = cdist(active[ai], active[bi])
                if best is None or d < best[0]:
                    best = (d, active[ai], active[bi])
        d, a, b = best
        members[next_id] = members[a] + members[b]
        children[next_id] = (a, b)
        node_h[next_id] = d
        active = [x for x in active if x not in (a, b)] + [next_id]
        next_id += 1
    root = active[0]

    # Leaf ordering by recursive walk.
    order: list[int] = []

    def walk(node: int) -> None:
        if node < n:
            order.append(node)
            return
        a, b = children[node]
        walk(a)
        walk(b)

    walk(root)
    for pos, leaf in enumerate(order):
        node_pos[leaf] = pos

    def position(node: int) -> float:
        if node in node_pos:
            return node_pos[node]
        a, b = children[node]
        node_pos[node] = (position(a) + position(b)) / 2
        return node_pos[node]

    fig, ax = plt.subplots(figsize=(12.8, 6.4))
    for node in range(n, next_id):
        a, b = children[node]
        xa, xb = position(a), position(b)
        ha, hb, h = node_h[a], node_h[b], node_h[node]
        ax.plot([xa, xa], [ha, h], color=BLUE, lw=1.6)
        ax.plot([xb, xb], [hb, h], color=BLUE, lw=1.6)
        ax.plot([xa, xb], [h, h], color=BLUE, lw=1.6)

    cut = 1.9
    ax.axhline(cut, color=GOLD, lw=1.6, linestyle=(0, (6, 4)))
    ax.text(n - 0.4, cut + 0.06, "разрез на три группы", fontsize=10.4,
            color=GOLD, ha="right")

    lab20 = LAB3[idx]
    for leaf in range(n):
        ax.text(node_pos[leaf], -0.12, RU[chosen[leaf]], rotation=60,
                ha="right", va="top", fontsize=9.6,
                color=CLUSTER_COLORS[lab20[leaf]])
    ax.set_xticks([])
    ax.set_ylabel("расстояние слияния (сигмы)")
    ax.set_ylim(-1.6, node_h[root] * 1.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_title("Дендрограмма двадцати стран (среднее попарное расстояние)",
                 loc="left", fontweight="bold", pad=12)
    save(fig, OUT / "dendrogram.png")


# ------------------------------------------------------------- margin schemes
def lloyd_steps() -> None:
    rng = np.random.default_rng(4)
    a = rng.normal([1.2, 1.2], 0.35, (12, 2))
    b = rng.normal([3.0, 2.2], 0.35, (12, 2))
    pts = np.vstack([a, b])
    c1, c2 = np.array([1.0, 2.6]), np.array([3.2, 1.0])

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2))
    lab = ((pts - c1) ** 2).sum(1) > ((pts - c2) ** 2).sum(1)
    for ax, title in [(axes[0], "назначение"), (axes[1], "пересчёт")]:
        for m, color in [(~lab, BLUE), (lab, GOLD)]:
            ax.plot(pts[m, 0], pts[m, 1], "o", ms=3.6, color=color, alpha=0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10.6, loc="left", pad=4)
        for spine in ax.spines.values():
            spine.set_color(LINE)
    axes[0].plot(*c1, "X", ms=10, color=BLUE, mec=INK)
    axes[0].plot(*c2, "X", ms=10, color=GOLD, mec=INK)
    n1, n2 = pts[~lab].mean(0), pts[lab].mean(0)
    axes[1].plot(*n1, "X", ms=10, color=BLUE, mec=INK)
    axes[1].plot(*n2, "X", ms=10, color=GOLD, mec=INK)
    for ax, старт, конец in [(axes[1], c1, n1), (axes[1], c2, n2)]:
        ax.annotate("", xy=конец, xytext=старт,
                    arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.2})
    save(fig, SIDE / "lloyd-steps.png")


def silhouette_scheme() -> None:
    rng = np.random.default_rng(5)
    own = rng.normal([1.1, 1.0], 0.3, (10, 2))
    other = rng.normal([3.0, 1.4], 0.3, (10, 2))
    p = np.array([1.9, 1.15])

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(own[:, 0], own[:, 1], "o", ms=4, color=BLUE, alpha=0.8)
    ax.plot(other[:, 0], other[:, 1], "o", ms=4, color=GOLD, alpha=0.8)
    ax.plot(*p, "o", ms=8, color=RED, zorder=5)
    for q in own[:4]:
        ax.plot([p[0], q[0]], [p[1], q[1]], color=BLUE, lw=0.9, alpha=0.7)
    for q in other[:4]:
        ax.plot([p[0], q[0]], [p[1], q[1]], color=GOLD, lw=0.9, alpha=0.7)
    ax.text(1.15, 1.62, "$a_i$: до своих", fontsize=10.6, color=BLUE)
    ax.text(2.55, 0.62, "$b_i$: до чужих", fontsize=10.6, color=GOLD)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(LINE)
    save(fig, SIDE / "silhouette-scheme.png")


def shapes() -> None:
    rng = np.random.default_rng(6)
    theta = rng.uniform(0, 2 * math.pi, 140)
    ring = np.stack([1.5 * np.cos(theta), 1.5 * np.sin(theta)], 1) + rng.normal(0, 0.08, (140, 2))
    core = rng.normal(0, 0.25, (60, 2))

    t = rng.uniform(0, math.pi, 90)
    moon1 = np.stack([np.cos(t), np.sin(t)], 1) + rng.normal(0, 0.07, (90, 2))
    moon2 = np.stack([1 - np.cos(t), 0.35 - np.sin(t)], 1) + rng.normal(0, 0.07, (90, 2))

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2))
    axes[0].plot(ring[:, 0], ring[:, 1], "o", ms=2.6, color=BLUE, alpha=0.75)
    axes[0].plot(core[:, 0], core[:, 1], "o", ms=2.6, color=GOLD, alpha=0.75)
    axes[1].plot(moon1[:, 0], moon1[:, 1], "o", ms=2.6, color=BLUE, alpha=0.75)
    axes[1].plot(moon2[:, 0], moon2[:, 1], "o", ms=2.6, color=GOLD, alpha=0.75)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_color(LINE)
    save(fig, SIDE / "shapes.png")


if __name__ == "__main__":
    raw_vs_scaled()
    k_choice()
    stability_curve()
    named_clusters()
    null_silhouette()
    dendrogram()
    lloyd_steps()
    silhouette_scheme()
    shapes()
    print("silhouette k3:", round(silhouette(Z, LAB3), 3))
