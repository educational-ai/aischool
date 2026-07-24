"""Deterministic figures for lesson 06: five modes of learning.

Every number is computed from the real Bike Sharing extract in
scripts/data/bike-sharing-hour.csv (Capital Bikeshare, 2011-2012, UCI).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "06"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "06"
DATA = ROOT / "scripts" / "data" / "bike-sharing-hour.csv"

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


def rounded_box(ax, xy, size, *, face=PAPER, edge=LINE, linewidth=1.2, rounding=0.018, zorder=2):
    patch = FancyBboxPatch(
        xy, *size,
        boxstyle=f"round,pad=0.012,rounding_size={rounding}",
        facecolor=face, edgecolor=edge, linewidth=linewidth, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, *, color=MUTED, width=1.4, rad=0.0, mutation=13,
          linestyle="solid", zorder=3, style="-|>"):
    patch = FancyArrowPatch(
        start, end, arrowstyle=style, connectionstyle=f"arc3,rad={rad}",
        color=color, linewidth=width, linestyle=linestyle,
        mutation_scale=mutation, shrinkA=0, shrinkB=0, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


# ------------------------------------------------------------------ data prep
ROWS = list(csv.DictReader(DATA.open()))
CNT = np.array([int(r["cnt"]) for r in ROWS])

day_profile: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(24))
day_meta: dict[str, tuple[str, str]] = {}
for r in ROWS:
    day_profile[r["dteday"]][int(r["hr"])] += int(r["cnt"])
    day_meta[r["dteday"]] = (r["workingday"], r["season"])

hour_work = defaultdict(list)
hour_free = defaultdict(list)
for r in ROWS:
    (hour_work if r["workingday"] == "1" else hour_free)[int(r["hr"])].append(int(r["cnt"]))
WORK_PROFILE = np.array([np.mean(hour_work[h]) for h in range(24)])
FREE_PROFILE = np.array([np.mean(hour_free[h]) for h in range(24)])

# Day share profiles for clustering (days with meaningful volume).
SHARE = {d: p / p.sum() for d, p in day_profile.items() if p.sum() > 200}
DAY_KEYS = sorted(SHARE)


def kmeans(points: dict[str, np.ndarray], init_keys: list[str], iters: int = 30):
    centers = [points[k].copy() for k in init_keys]
    groups: dict[int, list[str]] = {}
    for _ in range(iters):
        groups = defaultdict(list)
        for d, v in points.items():
            j = int(np.argmin([np.sum((v - c) ** 2) for c in centers]))
            groups[j].append(d)
        for j in range(len(centers)):
            centers[j] = np.mean([points[d] for d in groups[j]], axis=0)
    return centers, groups


CENTERS, GROUPS = kmeans(SHARE, [DAY_KEYS[10], DAY_KEYS[15], DAY_KEYS[100]])
MID = next(j for j in range(3) if int(np.argmax(CENTERS[j])) in (12, 13, 14))
MORN = next(j for j in range(3) if int(np.argmax(CENTERS[j])) == 8)
EVE = next(j for j in range(3) if j not in (MID, MORN))
VIOLATORS = sorted(d for d in GROUPS[MID] if day_meta[d][0] == "1")


# ---------------------------------------------------------------- figure 6.1
def weekly_profiles() -> None:
    fig, ax = plt.subplots(figsize=(12.8, 5.4))
    hours = np.arange(24)
    ax.plot(hours, WORK_PROFILE, color=BLUE, lw=2.2, marker="o", ms=4, label="рабочий день")
    ax.plot(hours, FREE_PROFILE, color=GOLD, lw=2.2, marker="s", ms=4, label="выходной")

    for h, v, txt, dy in [(8, WORK_PROFILE[8], "8:00 — 477", 26), (17, WORK_PROFILE[17], "17:00 — 525", 26),
                          (13, FREE_PROFILE[13], "13:00 — 373", -34)]:
        ax.annotate(txt, xy=(h, v), xytext=(0, dy), textcoords="offset points",
                    ha="center", fontsize=10.6, color=INK,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.9, "shrinkB": 3})

    ax.set_xticks(range(0, 24, 3), [f"{h}:00" for h in range(0, 24, 3)])
    ax.set_ylabel("аренды в час, среднее за 2011–2012")
    ax.set_ylim(0, 620)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("Два ритма одного города: работа и прогулка", loc="left", fontweight="bold", pad=12)
    save(fig, OUT / "weekly-profiles.png")


# ---------------------------------------------------------------- figure 6.2
def cnt_threshold() -> None:
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(12.8, 5.2),
                                  gridspec_kw={"width_ratios": [2.5, 1.0], "wspace": 0.22})
    ax.hist(CNT, bins=60, color=BLUE, alpha=0.85)
    ax.axvline(80, color=RED, lw=1.8)
    ax.text(92, ax.get_ylim()[1] * 0.93 if ax.get_ylim()[1] else 1, "", fontsize=1)
    ax.annotate("порог 80", xy=(80, 2350), xytext=(180, 2350), fontsize=11.4, color=RED,
                va="center", arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 1.2})
    ax.annotate("хвост до 977 аренд", xy=(920, 60), xytext=(560, 700), fontsize=10.6, color=MUTED,
                va="center", arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.0, "shrinkB": 2})
    ax.set_xlabel("аренды за час")
    ax.set_ylabel("число часов")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("До порога: всё распределение", loc="left", fontweight="bold", pad=10)

    low = float(np.mean(CNT < 80)) * 100
    high = 100 - low
    bars = axr.bar([0, 1], [low, high], color=[FAINT, RED], width=0.62)
    for b, v in zip(bars, [low, high]):
        axr.text(b.get_x() + b.get_width() / 2, v + 1.6, f"{v:.1f}%".replace(".", ","),
                 ha="center", fontsize=11.6, fontweight="bold")
    axr.set_xticks([0, 1], ["низкая\n(<80)", "высокая\n(≥80)"])
    axr.set_ylim(0, 78)
    axr.set_ylabel("доля часов, %")
    axr.grid(axis="y", color=GRID, lw=0.7)
    axr.spines["top"].set_visible(False)
    axr.spines["right"].set_visible(False)
    axr.set_title("После порога", loc="left", fontweight="bold", pad=10)
    save(fig, OUT / "cnt-threshold.png")


# ---------------------------------------------------------------- figure 6.3
def day_clusters() -> None:
    fig, (ax, axs) = plt.subplots(1, 2, figsize=(14.2, 5.6),
                                  gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.2})
    hours = np.arange(24)
    styles = [
        (MORN, BLUE, f"утренний пик · {len(GROUPS[MORN])} дней, 99% рабочие"),
        (EVE, GREEN, f"вечерний пик · {len(GROUPS[EVE])} дней, 100% рабочие"),
        (MID, GOLD, f"дневной горб · {len(GROUPS[MID])} дней, 3% рабочие"),
    ]
    for j, color, label in styles:
        ax.plot(hours, CENTERS[j] * 100, color=color, lw=2.2, label=label)
    ax.set_xticks(range(0, 24, 3), [f"{h}" for h in range(0, 24, 3)])
    ax.set_xlabel("час суток")
    ax.set_ylabel("доля суточных аренд, %")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 14.6)
    ax.legend(loc="upper left", frameon=False, fontsize=10.2)
    ax.set_title("Средние профили трёх кластеров (без меток)", loc="left", fontweight="bold", pad=10)

    # Scatter: morning share vs midday share, violators ringed.
    cluster_of = {d: j for j in range(3) for d in GROUPS[j]}
    color_map = {MORN: BLUE, EVE: GREEN, MID: GOLD}
    for d in DAY_KEYS:
        v = SHARE[d]
        axs.plot(v[8] * 100, v[13] * 100, "o", ms=3.2,
                 color=color_map[cluster_of[d]], alpha=0.55, zorder=2)
    for d in VIOLATORS:
        v = SHARE[d]
        axs.plot(v[8] * 100, v[13] * 100, "o", ms=9, mfc="none", mec=RED, mew=1.6, zorder=4)
    bbox = {"boxstyle": "round,pad=0.32", "facecolor": PAPER, "edgecolor": LINE, "alpha": 0.92}
    sandy = SHARE["2012-10-30"]
    axs.annotate("30.10.2012:\nураган Сэнди", xy=(sandy[8] * 100, sandy[13] * 100),
                 xytext=(34, 30), textcoords="offset points", fontsize=10.2, color=RED,
                 linespacing=1.3, bbox=bbox, zorder=6,
                 arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 1.1, "shrinkB": 6})
    bf = SHARE["2011-11-25"]
    axs.annotate("пятницы после Дня\nблагодарения, канун\nНового года…",
                 xy=(bf[8] * 100, bf[13] * 100), xytext=(78, -72), textcoords="offset points",
                 fontsize=10.2, color=RED, linespacing=1.3, bbox=bbox, zorder=6,
                 arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 1.1, "shrinkB": 6})
    axs.set_xlabel("доля аренд в 8:00, %")
    axs.set_ylabel("доля аренд в 13:00, %")
    axs.grid(color=GRID, lw=0.6)
    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)
    axs.set_title("730 дней; кольцами — рабочие дни в «выходном» кластере",
                  loc="left", fontweight="bold", fontsize=12.6, pad=10)
    save(fig, OUT / "day-clusters.png")


# ---------------------------------------------------------------- figure 6.4
def label_propagation() -> None:
    # Real day features; ground truth = rhythm type found in fig 6.3
    # (weekend-shaped midday hump vs workday-shaped commute peaks).
    midday_days = set(GROUPS[MID])
    feats, truth, keys = [], [], []
    for d in DAY_KEYS:
        v = SHARE[d]
        feats.append([v[8] * 100, v[13] * 100])
        truth.append(0 if d in midday_days else 1)
        keys.append(d)
    X = np.array(feats)
    y = np.array(truth)
    rng = np.random.default_rng(6)

    def propagate(seed_idx, seed_labels, radius=0.62):
        labels = -np.ones(len(X), dtype=int)
        labels[seed_idx] = seed_labels
        for _ in range(40):
            changed = False
            unl = np.where(labels < 0)[0]
            lab = np.where(labels >= 0)[0]
            if not len(unl):
                break
            d2 = ((X[unl, None, :] - X[None, lab, :]) ** 2).sum(-1)
            nearest = d2.min(1)
            order = np.argsort(nearest)
            for oi in order[: max(6, len(unl) // 8)]:
                if nearest[oi] < radius ** 2:
                    labels[unl[oi]] = labels[lab[np.argmin(d2[oi])]]
                    changed = True
            if not changed:
                break
        return labels

    seed_idx = rng.choice(len(X), size=20, replace=False)
    clean = propagate(seed_idx, y[seed_idx])

    # Poisoned run: flip the seed closest to the class boundary.
    boundary_seed = min(
        seed_idx,
        key=lambda i: abs(X[i, 0] - 6.5),
    )
    poisoned_labels = y[seed_idx].copy()
    poisoned_labels[np.where(seed_idx == boundary_seed)[0][0]] ^= 1
    dirty = propagate(seed_idx, poisoned_labels)

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8), sharex=True, sharey=True)
    for ax, labels, title in [
        (axes[0], clean, "Двадцать честных меток: облака подписаны верно"),
        (axes[1], dirty, "Одна ошибка среди тех же двадцати: захват чужого облака"),
    ]:
        err = 0
        for i in range(len(X)):
            if labels[i] < 0:
                ax.plot(*X[i], "o", ms=3, color=FAINT, alpha=0.5, zorder=1)
            else:
                ok = labels[i] == y[i]
                err += not ok
                ax.plot(*X[i], "o", ms=3.6,
                        color=BLUE if labels[i] == 1 else GOLD,
                        alpha=0.75, zorder=2)
                if not ok:
                    ax.plot(*X[i], "o", ms=8.6, mfc="none", mec=RED, mew=1.3, zorder=4)
        for i in seed_idx:
            ax.plot(*X[i], "s", ms=7, mfc="none",
                    mec=INK, mew=1.5, zorder=5)
        covered = int((labels >= 0).sum())
        ax.text(0.02, 0.03,
                f"псевдометок: {covered - 20}, ошибок: {err}",
                transform=ax.transAxes, fontsize=11.2,
                color=GREEN if err < 8 else RED, fontweight="bold")
        ax.set_xlabel("доля аренд в 8:00, %")
        ax.grid(color=GRID, lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=12.8, pad=10)
    if (dirty >= 0).any():
        axes[1].plot(*X[boundary_seed], "s", ms=9, mfc=RED, mec=INK, mew=1.4, zorder=6)
        axes[1].annotate("подсаженная\nошибка", xy=(X[boundary_seed, 0], X[boundary_seed, 1]),
                         xytext=(-58, 26), textcoords="offset points", fontsize=10.2,
                         color=RED, linespacing=1.3,
                         arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 1.1, "shrinkB": 5})
    axes[0].set_ylabel("доля аренд в 13:00, %")
    fig.text(0.008, 0.965,
             "Квадраты — настоящие метки; цвет — присвоенная метка; красные кольца — ошибочные псевдометки",
             fontsize=11.4, color=MUTED)
    fig.subplots_adjust(top=0.86, wspace=0.1)
    save(fig, OUT / "label-propagation.png")


# ---------------------------------------------------------------- figure 6.5
def return_discount() -> None:
    rewards = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 5.0])
    gammas = [1.0, 0.72, 0.5]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.9), sharey=True)
    for ax, g in zip(axes, gammas):
        contrib = np.array([g ** k * rewards[k] for k in range(6)])
        total = contrib.sum()
        colors = [RED if c < 0 else (GREEN if c > 0 else FAINT) for c in contrib]
        ax.bar(range(1, 7), contrib, color=colors, width=0.62, zorder=3)
        ax.axhline(0, color=LINE, lw=1)
        for k, c in enumerate(contrib, start=1):
            if abs(c) > 1e-9:
                ax.text(k, c + (0.16 if c > 0 else -0.16),
                        f"{c:+.2f}".replace(".", ",").replace("+", ""),
                        ha="center", va="bottom" if c > 0 else "top", fontsize=10.2)
        g_str = f"{g:g}".replace(".", "{,}")
        total_str = f"{total:+.2f}".replace(".", "{,}").replace("+", "")
        ax.set_title(f"$\\gamma = {g_str}$:  $G_1 = {total_str}$",
                     loc="left", fontsize=13.4, pad=10)
        ax.set_xlabel("шаг цепочки")
        ax.set_xticks(range(1, 7))
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("вклад шага в возврат $G_1$")
    axes[0].set_ylim(-1.35, 5.4)
    fig.text(0.008, 0.965,
             "Награды цепочки: [0, 0, −1, 0, 0, 5]. Затухание переставляет победителя спора «близкий штраф против далёкой награды».",
             fontsize=11.4, color=MUTED)
    fig.subplots_adjust(top=0.8, wspace=0.08)
    save(fig, OUT / "return-discount.png")


# ---------------------------------------------------------------- figure 6.6
def signal_map() -> None:
    fig, ax = plt.subplots(figsize=(12.8, 6.6))
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    target = (10.3, 3.05)
    rounded_box(ax, (target[0] - 1.25, target[1] - 0.62), (2.5, 1.24), face=WASH, edge=INK,
                linewidth=1.6, rounding=0.1)
    ax.text(target[0], target[1] + 0.13, "параметры модели", ha="center", va="center",
            fontsize=13.2, fontweight="bold")
    ax.text(target[0], target[1] - 0.3, r"$\theta$", ha="center", va="center", fontsize=14)

    sources = [
        (5.55, "готовый класс", "классификация · платит разметчик", BLUE, "solid"),
        (4.3, "готовое число", "регрессия · часто пишут приборы", GREEN, "solid"),
        (3.05, "геометрия входов", "без учителя · бесплатно, но расстояние выбираем мы", VIOLET, "solid"),
        (1.8, "редкая метка + псевдометки", "полуобучение · доверие к своим догадкам", GOLD, "dashed"),
        (0.55, "награда за цепочку", "подкрепление · лазейки в спецификации", RED, "dashed"),
    ]
    for y, title, sub, color, ls in sources:
        rounded_box(ax, (0.4, y - 0.028), (4.4, 0.96), face=PAPER, edge=color, linewidth=1.4,
                    rounding=0.08)
        ax.text(0.62, y + 0.6, title, ha="left", va="center", fontsize=12.4,
                fontweight="bold", color=color)
        ax.text(0.62, y + 0.22, sub, ha="left", va="center", fontsize=9.8, color=MUTED)
        start = (4.86, y + 0.45)
        offset = max(-0.48, min(0.48, (y - 2.6) * 0.18))
        end = (target[0] - 1.32, target[1] + offset)
        arrow(ax, start, end, color=color, width=1.6, rad=-0.06,
              linestyle="solid" if ls == "solid" else (0, (5, 4)), mutation=14)

    ax.text(7.4, 5.9, "пунктир: сигнал частично изготавливает сама система —\nнужен внешний контроль",
            fontsize=10.4, color=MUTED, ha="center", linespacing=1.4)
    fig.text(0.055, 0.945, "Пять источников одного и того же: сигнала для параметров",
             fontsize=17, fontweight="bold")
    save(fig, OUT / "signal-map.png")


# ------------------------------------------------------------- margin schemes
def normalization_changes_clusters() -> None:
    # Same days, clustered into TWO groups by share profiles vs absolute ones.
    absolute = {d: day_profile[d] for d in DAY_KEYS}
    _, share2 = kmeans(SHARE, [DAY_KEYS[10], DAY_KEYS[100]])
    _, abs_groups = kmeans(absolute, [DAY_KEYS[10], DAY_KEYS[100]])

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True)
    idx = {d: i for i, d in enumerate(DAY_KEYS)}
    colors3 = [BLUE, GOLD, GREEN]

    share_cluster_of = {d: j for j in range(2) for d in share2[j]}
    abs_cluster_of = {d: j for j in range(2) for d in abs_groups[j]}
    for ax, mapping, title in [
        (axes[0], share_cluster_of, "по долям часов: полосы будней и выходных"),
        (axes[1], abs_cluster_of, "по числам аренд: главным стал сезонный объём"),
    ]:
        for d in DAY_KEYS:
            ax.axvspan(idx[d] - 0.5, idx[d] + 0.5, color=colors3[mapping[d]], lw=0)
        ax.set_yticks([])
        ax.set_ylim(0, 1)
        ax.set_title(title, loc="left", fontsize=10.4, pad=6)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[1].set_xticks(
        [idx["2011-01-01"], idx["2011-07-01"], idx["2012-01-01"], idx["2012-07-02"], idx["2012-12-30"]],
        ["янв 11", "июл 11", "янв 12", "июл 12", "дек 12"],
    )
    fig.subplots_adjust(hspace=0.5)
    save(fig, SIDE / "normalization-changes-clusters.png")


def mask_task() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    hours = np.arange(24)
    profile = WORK_PROFILE.copy()
    ax.plot(hours, profile, color=BLUE, lw=1.9, marker="o", ms=3.4, zorder=2)
    ax.plot(14, profile[14], "o", ms=13, mfc=PAPER, mec=RED, mew=1.8, zorder=4)
    ax.text(14, profile[14], "?", ha="center", va="center", fontsize=13, color=RED,
            fontweight="bold", zorder=5)
    ax.annotate("скрыли час — получили пару\n«вопрос и ответ» бесплатно",
                xy=(14, profile[14]), xytext=(2.2, 500), fontsize=10.2, color=RED,
                linespacing=1.35,
                arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 1.1, "shrinkB": 8})
    ax.set_xticks(range(0, 24, 6), [f"{h}:00" for h in range(0, 24, 6)])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    save(fig, SIDE / "mask-task.png")


def weather_ladder() -> None:
    by_w = defaultdict(list)
    for r in ROWS:
        by_w[r["weathersit"]].append(int(r["cnt"]))
    means = [np.mean(by_w[k]) for k in ["1", "2", "3", "4"]]
    labels = ["ясно", "облачно", "осадки", "ливень,\nгроза"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    colors = [GOLD, FAINT, BLUE, VIOLET]
    bars = ax.bar(range(4), means, color=colors, width=0.6)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v + 5, f"{v:.0f}", ha="center", fontsize=11,
                fontweight="bold")
    ax.set_xticks(range(4), labels, fontsize=10)
    ax.set_ylabel("аренды в час, среднее")
    ax.set_ylim(0, 235)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, SIDE / "weather-ladder.png")


if __name__ == "__main__":
    weekly_profiles()
    cnt_threshold()
    day_clusters()
    label_propagation()
    return_discount()
    signal_map()
    normalization_changes_clusters()
    mask_task()
    weather_ladder()
    print("lesson 06 figures written")
