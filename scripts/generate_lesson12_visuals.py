"""Deterministic figures for lesson 12: dataset passport practicum on UCI Adult.

Recomputes every number quoted in the lesson (shares of '?', top-coding
spikes, weighted vs unweighted >50K share, sex/age income gaps, train/test
label mismatch) from scripts/data/adult.data and adult.test.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "12"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "12"

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

COLS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital",
        "occupation", "relationship", "race", "sex", "capital_gain",
        "capital_loss", "hours", "country", "income"]


def save(fig: plt.Figure, path: Path, *, dpi: int = 160) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def load(name: str, skip_header: bool = False):
    rows = []
    for line in (ROOT / "scripts" / "data" / name).open(encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("|"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 15:
            rows.append(dict(zip(COLS, parts)))
    return rows


train = load("adult.data")
test = load("adult.test")
assert len(train) == 32561 and len(test) == 16281

pos = sum(1 for r in train if r["income"] == ">50K")
share = pos / len(train)
w = np.array([int(r["fnlwgt"]) for r in train], float)
wshare = w[[r["income"] == ">50K" for r in train]].sum() / w.sum()
q_wc = sum(1 for r in train if r["workclass"] == "?")
q_oc = sum(1 for r in train if r["occupation"] == "?")
q_ct = sum(1 for r in train if r["country"] == "?")
both = sum(1 for r in train if r["workclass"] == "?" and r["occupation"] == "?")
miss = [r for r in train if r["workclass"] == "?"]
full = [r for r in train if r["workclass"] != "?"]
miss_pos = sum(1 for r in miss if r["income"] == ">50K") / len(miss)
full_pos = sum(1 for r in full if r["income"] == ">50K") / len(full)
miss_age = np.mean([int(r["age"]) for r in miss])
full_age = np.mean([int(r["age"]) for r in full])
age90 = sum(1 for r in train if r["age"] == "90")
h99 = sum(1 for r in train if r["hours"] == "99")
cg99999 = sum(1 for r in train if r["capital_gain"] == "99999")
test_pos = sum(1 for r in test if r["income"].rstrip(".") == ">50K") / len(test)
men = [r for r in train if r["sex"] == "Male"]
women = [r for r in train if r["sex"] == "Female"]
men_pos = sum(1 for r in men if r["income"] == ">50K") / len(men)
women_pos = sum(1 for r in women if r["income"] == ">50K") / len(women)

print(f">50K share {share:.4f}, weighted {wshare:.4f}")
print(f"? shares: workclass {q_wc/len(train):.4f} occupation {q_oc/len(train):.4f} country {q_ct/len(train):.4f}, both {both}")
print(f"missing group: >50K {miss_pos:.3f} vs {full_pos:.3f}; age {miss_age:.1f} vs {full_age:.1f}")
print(f"spikes: age90 {age90}, hours99 {h99}, cg99999 {cg99999}")
print(f"test >50K (clean) {test_pos:.4f}; men {men_pos:.3f} women {women_pos:.3f}")
assert abs(share - 0.2408) < 5e-4 and abs(wshare - 0.2386) < 5e-4
assert (q_wc, q_oc, q_ct, both) == (1836, 1843, 583, 1836)
assert (age90, h99, cg99999) == (43, 85, 159)
assert abs(test_pos - 0.2362) < 5e-4
assert abs(men_pos - 0.306) < 5e-3 and abs(women_pos - 0.109) < 5e-3


# --------------------------------------- fig 12.1: missing pattern portrait
def fig_missing() -> None:
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.4, 4.3),
                                   gridspec_kw={"width_ratios": [1.1, 1]})
    labels = ["workclass", "occupation", "country"]
    vals = [q_wc / len(train) * 100, q_oc / len(train) * 100,
            q_ct / len(train) * 100]
    bars = axl.barh([2, 1, 0], vals, height=0.55,
                    color=[RED, RED, GOLD], alpha=0.85)
    axl.set_yticks([2, 1, 0])
    axl.set_yticklabels(labels, fontsize=12)
    for y, v in zip([2, 1, 0], vals):
        axl.text(v + 0.12, y, f"{v:.2f}".replace(".", ",") + " %",
                 va="center", fontsize=11.5, color=INK)
    axl.set_xlim(0, 7.4)
    axl.set_xlabel("доля «?» среди 32 561 анкеты")
    axl.set_title("пропуски по столбцам", fontsize=12.5)
    axl.text(2.9, 1.5, "1836 общих анкет,\nразница — семь человек",
             fontsize=10.5, color=MUTED, va="center")
    axl.grid(True, axis="x", color=GRID, lw=0.6)
    axl.set_axisbelow(True)

    x = np.arange(2)
    axr.bar(x - 0.18, [miss_pos * 100, miss_age], width=0, alpha=0)  # spacing
    b1 = axr.bar([0, 1], [miss_pos * 100, full_pos * 100], width=0.5,
                 color=[VIOLET, BLUE], alpha=0.85)
    for xi, v in zip([0, 1], [miss_pos * 100, full_pos * 100]):
        axr.text(xi, v + 0.6, f"{v:.1f}".replace(".", ",") + " %",
                 ha="center", fontsize=11.5, color=INK)
    axr.set_xticks([0, 1])
    axr.set_xticklabels(["с «?»\n(41 год в среднем)", "без пропусков\n(38 лет)"],
                        fontsize=11)
    axr.set_ylim(0, 30)
    axr.set_ylabel("доля >50K, %")
    axr.set_title("портрет «молчунов»", fontsize=12.5)
    axr.grid(True, axis="y", color=GRID, lw=0.6)
    axr.set_axisbelow(True)
    fig.suptitle("Карта пропусков: один отказ, два столбца, разные люди",
                 y=1.02, fontsize=15)
    fig.tight_layout()
    save(fig, OUT / "missing-pattern.png")


# ------------------------------------------- fig 12.2: top-coding spikes
def fig_spikes() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 4.0))
    specs = [
        ("age", [int(r["age"]) for r in train], 90, range(15, 95, 3),
         "возраст, лет", "43 анкеты на «90»,\nна 89 — ноль"),
        ("hours", [int(r["hours"]) for r in train], 99, range(0, 103, 3),
         "часов в неделю", "85 анкет ровно 99"),
        ("capital_gain", [int(r["capital_gain"]) for r in train if r["capital_gain"] != "0"],
         99999, range(0, 104000, 4000), "прирост капитала, $ (без нулей)",
         "159 анкет ровно\n99 999"),
    ]
    for ax, (name, vals, cap, bins, xlabel, note) in zip(axes, specs):
        arr = np.array(vals)
        bins = np.array(list(bins))
        counts, edges = np.histogram(arr, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        colors = [RED if edges[i] <= cap < edges[i + 1] else BLUE
                  for i in range(len(counts))]
        ax.bar(centers, counts, width=(edges[1] - edges[0]) * 0.9,
               color=colors, alpha=0.85)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.grid(True, axis="y", color=GRID, lw=0.6, which="both", alpha=0.6)
        ax.set_axisbelow(True)
        ax.text(0.97, 0.95, note, transform=ax.transAxes, fontsize=10,
                color=RED, ha="right", va="top")
    axes[0].set_ylabel("анкет (лог. шкала)")
    fig.suptitle("Три спайка цензуры: 90 лет, 99 часов, 99 999 долларов",
                 y=1.03, fontsize=15)
    fig.tight_layout()
    save(fig, OUT / "topcoding-spikes.png")


# ------------------------------------ fig 12.3: train/test label mismatch
def fig_labels() -> None:
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.4, 4.2))
    raw = Counter(r["income"] for r in train)
    raw_t = Counter(r["income"] for r in test)
    names = ["<=50K", ">50K", "<=50K.", ">50K."]
    vals = [raw.get(n, 0) + raw_t.get(n, 0) for n in names]
    cols = [BLUE, RED, VIOLET, GOLD]
    axl.bar(range(4), vals, color=cols, alpha=0.85, width=0.62)
    axl.set_xticks(range(4))
    axl.set_xticklabels(["«<=50K»", "«>50K»", "«<=50K.»", "«>50K.»"],
                        fontsize=11)
    for i, v in enumerate(vals):
        axl.text(i, v + 500, f"{v:,}".replace(",", " "), ha="center",
                 fontsize=10.5, color=INK)
    axl.set_title("до чистки: четыре «класса»", fontsize=12.5)
    axl.set_ylabel("строк")
    axl.set_ylim(0, 27500)
    axl.grid(True, axis="y", color=GRID, lw=0.6)
    axl.set_axisbelow(True)

    tr = share * 100
    te = test_pos * 100
    axr.bar([0, 1], [tr, te], width=0.5, color=[BLUE, GREEN], alpha=0.85)
    for xi, v in zip([0, 1], [tr, te]):
        axr.text(xi, v + 0.4, f"{v:.2f}".replace(".", ",") + " %",
                 ha="center", fontsize=11.5, color=INK)
    axr.set_xticks([0, 1])
    axr.set_xticklabels(["train\n32 561 строка", "test\n16 281 строка"],
                        fontsize=11)
    axr.set_ylabel("доля >50K, %")
    axr.set_ylim(0, 30)
    axr.set_title("после чистки: два класса, доли сходятся", fontsize=12.5)
    axr.grid(True, axis="y", color=GRID, lw=0.6)
    axr.set_axisbelow(True)
    fig.suptitle("Одна метка, четыре написания: до и после чистки",
                 y=1.02, fontsize=15)
    fig.tight_layout()
    save(fig, OUT / "train-test-labels.png")


# -------------------------------------------- fig 12.4: income gap bars
def fig_gap() -> None:
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.4, 4.2),
                                   gridspec_kw={"width_ratios": [0.75, 1.25]})
    axl.bar([0, 1], [men_pos * 100, women_pos * 100], width=0.5,
            color=[BLUE, RED], alpha=0.85)
    for xi, v in zip([0, 1], [men_pos * 100, women_pos * 100]):
        axl.text(xi, v + 0.7, f"{v:.1f}".replace(".", ",") + " %",
                 ha="center", fontsize=11.5, color=INK)
    axl.set_xticks([0, 1])
    axl.set_xticklabels(["мужчины", "женщины"], fontsize=11.5)
    axl.set_ylabel("доля >50K, %")
    axl.set_ylim(0, 36)
    axl.set_title("по полу", fontsize=12.5)
    axl.grid(True, axis="y", color=GRID, lw=0.6)
    axl.set_axisbelow(True)

    groups = [(17, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 90)]
    shares_g = []
    for lo, hi in groups:
        sel = [r for r in train if lo <= int(r["age"]) <= hi]
        shares_g.append(100 * sum(1 for r in sel if r["income"] == ">50K") / len(sel))
    print("age-group >50K shares:", [round(v, 1) for v in shares_g])
    axr.bar(range(6), shares_g, width=0.6, color=GREEN, alpha=0.85)
    axr.set_xticks(range(6))
    axr.set_xticklabels(["17–24", "25–34", "35–44", "45–54", "55–64", "65+"],
                        fontsize=10.5)
    for xi, v in enumerate(shares_g):
        axr.text(xi, v + 0.7, f"{v:.0f}", ha="center", fontsize=10.5,
                 color=INK)
    axr.set_xlabel("возраст, лет")
    axr.set_ylim(0, 42)
    axr.set_title("по возрасту: горб опыта", fontsize=12.5)
    axr.grid(True, axis="y", color=GRID, lw=0.6)
    axr.set_axisbelow(True)
    fig.suptitle("Разрыв, который модель выучит как правило", y=1.02,
                 fontsize=15)
    fig.tight_layout()
    save(fig, OUT / "income-gap.png")


# -------------------------------------------- fig 12.5: Adult timeline
def fig_timeline() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    ax.set_xlim(1992, 2027)
    ax.set_ylim(-1.6, 2.1)
    ax.axis("off")
    ax.axhline(0, color=INK, lw=1.6)
    events = [
        (1994, "перепись CPS:\nанкеты собраны", BLUE, 1),
        (1996, "Adult опубликован\nв UCI", GREEN, -1),
        (2018, "«datasheets for\ndatasets»: манифест\nпаспортов", GOLD, 1),
        (2021, "«Retiring Adult»:\nотставка и замена", RED, -1),
    ]
    for year, label, color, side in events:
        ax.plot([year], [0], marker="o", markersize=9, color=color,
                markeredgecolor=PAPER, markeredgewidth=1.2, zorder=5)
        y = 0.55 * side
        ax.plot([year, year], [0, y], color=color, lw=1.2)
        ax.text(year, y + 0.16 * side, label, ha="center",
                va="bottom" if side > 0 else "top", fontsize=10.5,
                color=color)
        ax.text(year, -0.03, "", ha="center")
    for year in (1994, 1996, 2018, 2021):
        ax.text(year, 0.13 if year in (1996, 2021) else -0.24, str(year),
                ha="center", fontsize=10, color=MUTED)
    ax.annotate("четверть века службы почти без паспорта",
                (2007, 0), (2007, 1.55), fontsize=11, color=MUTED,
                ha="center",
                arrowprops=dict(arrowstyle="-[, widthB=9.2, lengthB=0.4",
                                color=LINE, lw=1.2))
    ax.set_title("Три жизни одной таблицы: перепись, бенчмарк, отставка",
                 fontsize=15, pad=18)
    save(fig, OUT / "adult-timeline.png")


# ---------------------------------------- fig 12.6: passport skeleton
def fig_skeleton() -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    blocks = [
        ("источник", "перепись CPS,\n1994, метод. БЛС", 0.4),
        ("единица", "анкета × вес\nfnlwgt", 3.2),
        ("схема", "15 столбцов,\n2 — дубли", 6.0),
        ("пропуски", "«?»: 5,6 %,\nне случайны", 0.4),
        ("цензура", "90 лет, 99 ч,\n$99 999", 3.2),
        ("рамка", "США, 1994,\nдомохозяйства", 6.0),
        ("срок", "порог = $ 1994,\n≈ $107 тыс. ныне", 0.4),
        ("применения", "исследование — да,\nскоринг людей — нет", 3.2),
    ]
    rows_y = [4.35, 2.45, 0.55]
    group_titles = [("что это", GREEN), ("чего не видно", RED),
                    ("что можно", BLUE)]
    for gi, (gt, gc) in enumerate(group_titles):
        y = rows_y[gi]
        ax.text(9.05, y + 0.55, gt, fontsize=13, color=gc, va="center",
                style="italic")
        row_blocks = blocks[gi * 3:(gi + 1) * 3]
        for title, sub, x in row_blocks:
            box = FancyBboxPatch((x, y), 2.45, 1.14,
                                 boxstyle="round,pad=0.06,rounding_size=0.09",
                                 facecolor=WASH, edgecolor=gc, linewidth=1.4)
            ax.add_patch(box)
            ax.text(x + 1.22, y + 0.86, title, ha="center", fontsize=12,
                    color=INK, weight="bold")
            ax.text(x + 1.22, y + 0.38, sub, ha="center", fontsize=9.3,
                    color=MUTED)
    for y0, y1 in [(rows_y[0], rows_y[1]), (rows_y[1], rows_y[2])]:
        ax.add_patch(FancyArrowPatch((8.65, y0 + 0.5), (8.65, y1 + 0.75),
                                     arrowstyle="-|>", color=MUTED, lw=1.4,
                                     connectionstyle="arc3,rad=-0.25",
                                     mutation_scale=13))
    ax.set_title("Скелет паспорта: восемь разделов в три блока (пример — Adult)",
                 fontsize=15, pad=14)
    save(fig, OUT / "passport-skeleton.png")


# ----------------------------------------------- margins
def side_fnlwgt() -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    # table row
    ax.add_patch(plt.Rectangle((0.4, 5.2), 4.6, 1.0, facecolor=WASH,
                               edgecolor=INK, lw=1.2))
    ax.text(2.7, 5.7, "анкета №1", ha="center", fontsize=10, color=INK)
    ax.text(7.6, 5.7, "fnlwgt = 189 778", ha="center", fontsize=10.5,
            color=BLUE)
    # crowd
    rng = np.random.default_rng(7)
    for i in range(60):
        x = 0.8 + 8.6 * rng.random()
        y = 0.6 + 3.0 * rng.random()
        ax.plot([x], [y], marker="o", markersize=4.2, color=MUTED,
                alpha=0.55)
    ax.annotate("", (5.0, 3.9), (2.6, 5.15),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.4,
                                connectionstyle="arc3,rad=-0.2"))
    ax.text(5.0, 0.15, "одна строка отвечает за тысячи людей",
            ha="center", fontsize=9.5, color=MUTED)
    save(fig, SIDE / "fnlwgt-scheme.png")


def side_missing_overlap() -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.6, 3.6), 7.4, 0.8, facecolor=RED,
                               alpha=0.8))
    ax.add_patch(plt.Rectangle((0.6, 2.4), 7.43, 0.8, facecolor=RED,
                               alpha=0.55))
    ax.add_patch(plt.Rectangle((0.6, 1.2), 2.35, 0.8, facecolor=GOLD,
                               alpha=0.8))
    ax.text(8.25, 4.0, "workclass: 1836", fontsize=9.5, va="center",
            color=INK)
    ax.text(8.28, 2.8, "occupation: 1843", fontsize=9.5, va="center",
            color=INK)
    ax.text(3.2, 1.6, "country: 583", fontsize=9.5, va="center", color=INK)
    ax.plot([0.6, 0.6], [1.0, 4.6], color=INK, lw=1.4)
    ax.text(5.0, 0.35, "первые две полосы — один и тот же отказ",
            ha="center", fontsize=9.5, color=MUTED)
    save(fig, SIDE / "missing-overlap.png")


def side_rose() -> None:
    fig = plt.figure(figsize=(4.0, 3.6))
    ax = fig.add_subplot(projection="polar")
    rng = np.random.default_rng(3)
    n = 12
    theta = np.linspace(0, 2 * math.pi, n, endpoint=False)
    r1 = 0.55 + 0.75 * rng.random(n)
    r2 = r1 * (0.35 + 0.3 * rng.random(n))
    width = 2 * math.pi / n * 0.98
    ax.bar(theta, r1, width=width, color=BLUE, alpha=0.5, edgecolor=PAPER)
    ax.bar(theta, r2, width=width, color=RED, alpha=0.75, edgecolor=PAPER)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_color(LINE)
    fig.text(0.5, 0.03, "площадь сектора = число смертей за месяц",
             ha="center", fontsize=9.5, color=MUTED)
    save(fig, SIDE / "nightingale-rose.png")


fig_missing()
fig_spikes()
fig_labels()
fig_gap()
fig_timeline()
fig_skeleton()
side_fnlwgt()
side_missing_overlap()
side_rose()
print("lesson 12 figures written")
