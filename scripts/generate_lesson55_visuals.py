"""Deterministic figures for lesson 55: experiments, randomization, p-value, power.

Every number quoted in the lesson is computed here and asserted. The null distribution,
peeking, subgroup hunting and self-selection are demonstrated on the REAL bike-sharing
data (17379 hourly records): random A/A splits of real records, not invented noise.
Power/sample-size curves are analytic; the click-rate example is an explicitly modelled
one with a fixed seed.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BIKE = ROOT / "scripts" / "data" / "bike-sharing-hour.csv"
OUT = ROOT / "public" / "figures" / "lessons" / "55"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "55"

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

FACTS: dict[str, float] = {}


def save(fig, path, *, dpi=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def load_bike():
    cnt, hr, wd = [], [], []
    with open(BIKE) as f:
        for row in csv.DictReader(f):
            cnt.append(int(row["cnt"])); hr.append(int(row["hr"])); wd.append(int(row["weekday"]))
    return np.array(cnt, float), np.array(hr), np.array(wd)


CNT, HR, WD = load_bike()


def welch_z(a, b):
    return (b.mean() - a.mean()) / math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))


# ---------------------------------------------------------------- fig 55.1: A/A on real data
def fig_aa_real() -> None:
    rng = np.random.default_rng(55)
    n = len(CNT)
    zs = []
    for _ in range(4000):
        m = rng.permutation(n)
        a, b = CNT[m[: n // 2]], CNT[m[n // 2:]]
        zs.append(welch_z(a, b))
    zs = np.array(zs)
    share = float(np.mean(np.abs(zs) > 1.959964))
    FACTS.update(bike_n=n, bike_mean=float(CNT.mean()), bike_median=float(np.median(CNT)),
                 bike_sd=float(CNT.std(ddof=1)), aa_share=share, aa_maxabs=float(np.abs(zs).max()),
                 aa_sd=float(zs.std(ddof=1)))
    print(f"AA: n={n} mean={CNT.mean():.2f} median={np.median(CNT):.0f} sd={CNT.std(ddof=1):.2f} "
          f"share|z|>1.96={share:.4f} sd(z)={zs.std(ddof=1):.3f} max|z|={np.abs(zs).max():.2f}")
    assert n == 17379
    assert abs(CNT.mean() - 189.46) < 0.005 and abs(np.median(CNT) - 142) < 1e-9
    assert 0.040 <= share <= 0.062
    assert 0.95 <= zs.std(ddof=1) <= 1.05

    # one concrete split, quoted in the text
    m = np.random.default_rng(2024).permutation(n)
    a, b = CNT[m[: n // 2]], CNT[m[n // 2:]]
    one_diff = float(b.mean() - a.mean()); one_z = welch_z(a, b)
    one_p = 2 * (1 - phi(abs(one_z)))
    FACTS.update(one_diff=one_diff, one_z=one_z, one_p=one_p,
                 one_a=float(a.mean()), one_b=float(b.mean()))
    print(f"one split: A={a.mean():.2f} B={b.mean():.2f} diff={one_diff:.2f} z={one_z:.3f} p={one_p:.3f}")
    assert abs(one_diff) < 12 and abs(one_z) < 3

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.hist(zs, bins=60, density=True, color=BLUE, alpha=0.32, edgecolor=PAPER, lw=0.4,
            label="4000 случайных A/A-разбиений реальных данных")
    xs = np.linspace(-4.2, 4.2, 400)
    ax.plot(xs, np.exp(-xs ** 2 / 2) / math.sqrt(2 * math.pi), color=INK, lw=2.0,
            label="стандартный нормальный закон")
    for s in (-1.959964, 1.959964):
        ax.axvline(s, color=RED, lw=1.4, ls=(0, (4, 3)))
    tail = np.linspace(1.959964, 4.2, 120)
    ax.fill_between(tail, 0, np.exp(-tail ** 2 / 2) / math.sqrt(2 * math.pi), color=RED, alpha=0.22)
    ax.fill_between(-tail, 0, np.exp(-tail ** 2 / 2) / math.sqrt(2 * math.pi), color=RED, alpha=0.22)
    ax.axvline(one_z, color=GOLD, lw=2.2)
    ax.annotate(f"одно разбиение: $z={one_z:.2f}$", xy=(one_z, 0.30), xytext=(one_z + 0.55, 0.36),
                color=GOLD, fontsize=10.5,
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.2))
    ax.text(2.55, 0.09, f"хвосты: {share * 100:.1f}%\nразбиений", color=RED, fontsize=10.5, ha="left")
    ax.set_xlabel("$z=(\\bar y_B-\\bar y_A)/\\mathrm{se}$"); ax.set_ylabel("плотность")
    ax.set_title("A/A-тест на реальных данных: эффекта нет, разница есть всегда")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "aa_real.png")


# ---------------------------------------------------------------- fig 55.2: power & sample size
def power_two_prop(p0, delta, n, alpha=0.05):
    za = 1.959964 if abs(alpha - 0.05) < 1e-12 else -math.sqrt(2) * 0  # only 0.05 used
    p1 = p0 + delta
    se = math.sqrt((p0 * (1 - p0) + p1 * (1 - p1)) / n)
    return (1 - phi(za - delta / se)) + phi(-za - delta / se)


def n_for_power(p0, delta, power=0.80, alpha=0.05):
    za, zb = 1.959964, 0.8416212
    p1 = p0 + delta
    return (za + zb) ** 2 * (p0 * (1 - p0) + p1 * (1 - p1)) / delta ** 2


def fig_power() -> None:
    p0 = 0.08
    ns = np.logspace(2.6, 5.9, 400)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    marks = {}
    for delta, c, lab in [(0.005, BLUE, "0,5 п.п."), (0.010, GREEN, "1,0 п.п."), (0.020, GOLD, "2,0 п.п.")]:
        pw = [power_two_prop(p0, delta, n) for n in ns]
        ax.plot(ns, pw, color=c, lw=2.2, label=f"эффект {lab}")
        n80 = n_for_power(p0, delta)
        marks[delta] = n80
        ax.plot([n80], [0.80], "o", color=c, ms=7, zorder=6)
    ax.axhline(0.80, color=RED, lw=1.2, ls=(0, (4, 3)))
    ax.text(4e2, 0.82, "мощность 80%", color=RED, fontsize=10.5)
    ax.set_xscale("log")
    ax.set_xlabel("объём одной группы $n$ (лог. шкала)")
    ax.set_ylabel("мощность")
    ax.set_title("Чем меньше эффект, тем дороже его увидеть (база 8%, $\\alpha=0{,}05$)")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    for delta, c in [(0.005, BLUE), (0.010, GREEN), (0.020, GOLD)]:
        ax.annotate(f"$n\\approx{marks[delta]:,.0f}$".replace(",", "\\,"),
                    xy=(marks[delta], 0.80), xytext=(marks[delta] * 1.15, 0.62),
                    color=c, fontsize=10)
    save(fig, OUT / "power.png")

    n05 = n_for_power(p0, 0.005); n10 = n_for_power(p0, 0.010); n20 = n_for_power(p0, 0.020)
    pw10k = power_two_prop(p0, 0.005, 10000); pw100k = power_two_prop(p0, 0.005, 100000)
    FACTS.update(n80_05=n05, n80_10=n10, n80_20=n20, pw10k=pw10k, pw100k=pw100k)
    print(f"power: n80(0.5pp)={n05:.0f} n80(1pp)={n10:.0f} n80(2pp)={n20:.0f} "
          f"power@10k={pw10k:.3f} power@100k={pw100k:.3f}")
    assert 47400 < n05 < 47700 and 12100 < n10 < 12300 and 3150 < n20 < 3250
    assert 0.245 < pw10k < 0.255 and 0.975 < pw100k < 0.99
    assert 3.7 < n05 / n10 < 4.0  # вдвое меньший эффект — почти вчетверо дороже


# ---------------------------------------------------------------- fig 55.3: peeking
def fig_peeking() -> None:
    rng = np.random.default_rng(555)
    n = len(CNT)
    looks = 20                # проверка после каждого пакета
    batch = (n // 2) // looks
    trials = 2000
    crossed_any, crossed_end, first_cross = 0, 0, []
    paths = []
    for t in range(trials):
        m = rng.permutation(n)
        a_all, b_all = CNT[m[: n // 2]], CNT[m[n // 2: n // 2 * 2]]
        zs = []
        for k in range(1, looks + 1):
            a, b = a_all[: k * batch], b_all[: k * batch]
            zs.append(welch_z(a, b))
        zs = np.array(zs)
        hit = np.abs(zs) > 1.959964
        if hit.any():
            crossed_any += 1
            first_cross.append(int(np.argmax(hit)) + 1)
        if hit[-1]:
            crossed_end += 1
        if t < 60:
            paths.append(zs)
    rate_peek = crossed_any / trials
    rate_fixed = crossed_end / trials
    FACTS.update(peek_rate=rate_peek, fixed_rate=rate_fixed, peek_looks=looks,
                 peek_batch=batch, peek_ratio=rate_peek / rate_fixed)
    print(f"peeking: {looks} looks -> false discovery {rate_peek:.3f} vs fixed {rate_fixed:.3f} "
          f"(ratio {rate_peek / rate_fixed:.1f}), batch={batch}")
    assert 0.15 <= rate_peek <= 0.32
    assert 0.035 <= rate_fixed <= 0.065
    assert rate_peek / rate_fixed > 3

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ks = np.arange(1, looks + 1) * batch
    for zs in paths:
        crossed = np.abs(zs).max() > 1.959964
        ax.plot(ks, zs, color=RED if crossed else FAINT, lw=1.4 if crossed else 0.8,
                alpha=0.85 if crossed else 0.5)
    for s in (-1.959964, 1.959964):
        ax.axhline(s, color=INK, lw=1.6, ls=(0, (5, 3)))
    ax.axhline(0, color=LINE, lw=0.8)
    ax.text(ks[-1], 2.15, "порог $|z|=1{,}96$", ha="right", color=INK, fontsize=10.5)
    ax.set_xlabel("накопленный объём одной группы")
    ax.set_ylabel("$z$ по накопленным данным")
    ax.set_title(f"Подглядывание: {rate_peek * 100:.0f}% нулевых экспериментов хоть раз пересекают порог")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    ax.text(ks[0], -3.4, "красные — те, что дали бы «значимый» результат при остановке в момент пересечения",
            color=RED, fontsize=10)
    ax.set_ylim(-4, 4)
    save(fig, OUT / "peeking.png")


# ---------------------------------------------------------------- fig 55.4: multiplicity
def fig_multiplicity() -> None:
    ms = np.arange(1, 41)
    fam = 1 - 0.95 ** ms
    bonf = 1 - (1 - 0.05 / ms) ** ms
    FACTS.update(fam20=float(1 - 0.95 ** 20), fam5=float(1 - 0.95 ** 5),
                 fam12=float(1 - 0.95 ** 12), bonf20=0.05 / 20)
    print(f"multiplicity: m=5 {1 - 0.95 ** 5:.3f}, m=12 {1 - 0.95 ** 12:.3f}, m=20 {1 - 0.95 ** 20:.4f}")
    assert abs((1 - 0.95 ** 20) - 0.6415) < 5e-05 and abs((1 - 0.95 ** 12) - 0.4596) < 0.001

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(ms, fam, color=RED, lw=2.4, label="без поправки: $1-0{,}95^m$")
    ax.plot(ms, bonf, color=BLUE, lw=2.2, label="поправка Бонферрони: порог $\\alpha/m$")
    ax.axhline(0.05, color=MUTED, lw=1.0, ls=(0, (4, 3)))
    ax.plot([20], [1 - 0.95 ** 20], "o", color=RED, ms=8)
    ax.annotate(f"20 метрик: {(1 - 0.95 ** 20) * 100:.1f}%", xy=(20, 1 - 0.95 ** 20),
                xytext=(21, 0.52), color=RED, fontsize=11,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.text(30, 0.075, "заявленные 5%", color=MUTED, fontsize=10.5)
    ax.set_xlabel("число одновременно проверяемых метрик $m$")
    ax.set_ylabel("вероятность хотя бы одной ложной тревоги")
    ax.set_title("Каждая лишняя метрика — ещё один билет в лотерее ложных открытий")
    ax.set_ylim(0, 1)
    ax.legend(loc="center right", frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "multiplicity.png")


# ---------------------------------------------------------------- fig 55.5: subgroup hunting (real)
def fig_subgroups() -> None:
    rng = np.random.default_rng(7)
    trials = 1000
    max_abs, any_hit, hit_overall = [], 0, 0
    example = None
    for t in range(trials):
        m = rng.permutation(len(CNT))
        idx_a, idx_b = m[: len(CNT) // 2], m[len(CNT) // 2:]
        z_all = welch_z(CNT[idx_a], CNT[idx_b])
        hit_overall += abs(z_all) > 1.959964
        zs = []
        for h in range(24):
            a = CNT[idx_a][HR[idx_a] == h]; b = CNT[idx_b][HR[idx_b] == h]
            zs.append(welch_z(a, b))
        zs = np.array(zs)
        max_abs.append(np.abs(zs).max())
        any_hit += np.abs(zs).max() > 1.959964
        if t == 0:
            example = (z_all, zs)
    rate_sub = any_hit / trials
    rate_all = hit_overall / trials
    med_max = float(np.median(max_abs))
    FACTS.update(sub_rate=rate_sub, sub_overall=rate_all, sub_medmax=med_max,
                 sub_ex_all=float(example[0]), sub_ex_max=float(np.abs(example[1]).max()),
                 sub_ex_hour=int(np.argmax(np.abs(example[1]))))
    print(f"subgroups: any-of-24 {rate_sub:.3f}, overall {rate_all:.3f}, median max|z|={med_max:.2f}; "
          f"example overall z={example[0]:.2f}, best hour {np.argmax(np.abs(example[1]))} z={example[1][np.argmax(np.abs(example[1]))]:.2f}")
    assert 0.50 <= rate_sub <= 0.85 and 0.03 <= rate_all <= 0.07
    assert 2.0 <= med_max <= 3.2

    z_all, zs = example
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    colors = [RED if abs(z) > 1.959964 else BLUE for z in zs]
    ax.bar(np.arange(24), zs, color=colors, alpha=0.85, edgecolor=PAPER, lw=0.6)
    for s in (-1.959964, 1.959964):
        ax.axhline(s, color=INK, lw=1.4, ls=(0, (5, 3)))
    ax.axhline(z_all, color=GOLD, lw=2.2)
    ax.text(0.2, z_all + 0.12, f"общий эффект: $z={z_all:.2f}$", color=GOLD, fontsize=11)
    ax.set_xlabel("сегмент: час суток"); ax.set_ylabel("$z$ внутри сегмента")
    ax.set_title("Одно A/A-разбиение, 24 сегмента: «победитель» находится почти всегда")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4, axis="y"); ax.set_axisbelow(True)
    ax.text(12, -3.2, f"в {rate_sub * 100:.0f}% нулевых экспериментов хотя бы один сегмент «значим»",
            ha="center", color=MUTED, fontsize=10.5)
    ax.set_ylim(-3.6, 3.6)
    save(fig, OUT / "subgroups.png")


# ---------------------------------------------------------------- fig 55.6: self-selection
def fig_selection() -> None:
    rng = np.random.default_rng(41)
    true_mean = float(CNT.mean())
    # добровольный опрос: активные часы отвечают охотнее (вероятность растёт с загрузкой)
    w = 0.02 + 0.98 * (CNT / CNT.max()) ** 0.7
    w = w / w.sum()
    sizes = np.unique(np.round(np.logspace(1.7, 3.9, 22)).astype(int))
    means, los, his = [], [], []
    for s in sizes:
        est = []
        for _ in range(300):
            pick = rng.choice(len(CNT), size=s, replace=True, p=w)
            est.append(CNT[pick].mean())
        est = np.array(est)
        means.append(est.mean()); los.append(np.percentile(est, 2.5)); his.append(np.percentile(est, 97.5))
    means, los, his = map(np.array, (means, los, his))
    biased_limit = float(np.sum(w * CNT))
    bias = biased_limit - true_mean
    FACTS.update(sel_true=true_mean, sel_limit=biased_limit, sel_bias=bias,
                 sel_bias_pct=100 * bias / true_mean,
                 sel_ci_small=float(his[0] - los[0]), sel_ci_big=float(his[-1] - los[-1]),
                 sel_n_small=int(sizes[0]), sel_n_big=int(sizes[-1]))
    print(f"selection: true={true_mean:.2f} limit={biased_limit:.2f} bias={bias:.2f} "
          f"({100 * bias / true_mean:.1f}%), CI width {his[0] - los[0]:.1f} -> {his[-1] - los[-1]:.1f}")
    assert bias > 80
    assert his[-1] - los[-1] < 0.35 * (his[0] - los[0])
    assert los[-1] > true_mean  # доверительный интервал уже не накрывает истину

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.fill_between(sizes, los, his, color=RED, alpha=0.18, label="95% разброс оценки по добровольцам")
    ax.plot(sizes, means, color=RED, lw=2.4, label="добровольный опрос")
    ax.axhline(true_mean, color=GREEN, lw=2.2, label=f"истинное среднее по всем записям = {true_mean:.0f}")
    ax.axhline(biased_limit, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax.text(sizes[0], biased_limit + 6, f"предел смещённой оценки = {biased_limit:.0f}",
            color=MUTED, fontsize=10.5)
    ax.annotate("", xy=(sizes[-1] * 0.85, true_mean), xytext=(sizes[-1] * 0.85, biased_limit),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))
    ax.text(sizes[-1] * 0.80, (true_mean + biased_limit) / 2, f"смещение {bias:.0f}",
            ha="right", color=INK, fontsize=11)
    ax.set_xscale("log")
    ax.set_xlabel("число ответивших $n$ (лог. шкала)"); ax.set_ylabel("оценка среднего числа поездок")
    ax.set_title("Рост выборки сужает интервал вокруг неправильного числа")
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "selection.png")


# ---------------------------------------------------------------- sidenote: MDE vs n
def side_mde() -> None:
    p0 = 0.08
    ns = np.logspace(2.6, 6.0, 300)
    za, zb = 1.959964, 0.8416212
    mde = (za + zb) * np.sqrt(2 * p0 * (1 - p0) / ns)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(ns, mde * 100, color=BLUE, lw=2.2)
    for n, c in [(1000, GOLD), (10000, GREEN), (100000, RED)]:
        m = (za + zb) * math.sqrt(2 * p0 * (1 - p0) / n) * 100
        ax.plot([n], [m], "o", color=c, ms=6)
        ax.annotate(f"{m:.2f} п.п.", xy=(n, m), xytext=(n * 1.2, m * 1.15), color=c, fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$n$ на группу", fontsize=10); ax.set_ylabel("MDE, п.п.", fontsize=10)
    ax.set_title("Минимальный различимый эффект", fontsize=11)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4, which="both"); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, SIDE / "mde.png")
    vals = {n: (za + zb) * math.sqrt(2 * p0 * (1 - p0) / n) * 100 for n in (1000, 10000, 100000)}
    FACTS.update(mde1k=vals[1000], mde10k=vals[10000], mde100k=vals[100000])
    print("MDE: " + ", ".join(f"n={n}: {v:.2f} п.п." for n, v in vals.items()))
    assert abs(vals[1000] - 3.40) < 0.005 and abs(vals[10000] - 1.07) < 0.02 and abs(vals[100000] - 0.34) < 0.02


# ---------------------------------------------------------------- sidenote: unit of randomization
def side_unit() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 3.0))
    rng = np.random.default_rng(3)
    for ax, title, per_user in [(axes[0], "по показу", False), (axes[1], "по пользователю", True)]:
        for u in range(4):
            col = [BLUE, RED][rng.integers(0, 2)] if per_user else None
            for k in range(5):
                c = col if per_user else [BLUE, RED][rng.integers(0, 2)]
                ax.add_patch(plt.Rectangle((k, -u), 0.8, 0.8, color=c, alpha=0.85))
            ax.text(-0.35, -u + 0.4, f"{u + 1}", ha="right", va="center", color=MUTED, fontsize=9)
        ax.set_xlim(-1.1, 5.1); ax.set_ylim(-3.4, 1.2)
        ax.set_title(title, fontsize=10.5); ax.axis("off")
    fig.suptitle("Единица рандомизации", fontsize=11.5, y=0.99)
    fig.text(0.5, 0.02, "строка — один человек, клетка — один показ", ha="center",
             color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    save(fig, SIDE / "unit.png")


# ---------------------------------------------------------------- sidenote: intervals and decision
def side_intervals() -> None:
    mde = 0.5
    rows = [("A: точно, но мелко", 0.08, 0.11, GOLD),
            ("B: неопределённо", 0.7, 0.9, VIOLET),
            ("C: уверенная польза", 1.4, 0.35, GREEN)]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for i, (name, est, half, c) in enumerate(rows):
        y = -i
        ax.plot([est - half, est + half], [y, y], color=c, lw=3.0, solid_capstyle="round")
        ax.plot([est], [y], "o", color=c, ms=7)
        ax.text(-2.4, y + 0.28, name, color=INK, fontsize=9.5)
    ax.axvline(0, color=INK, lw=1.2)
    ax.axvline(mde, color=RED, lw=1.2, ls=(0, (4, 3)))
    ax.text(mde + 0.06, -2.6, "порог пользы", color=RED, fontsize=9, rotation=90, va="bottom")
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.8, 0.9)
    ax.set_yticks([]); ax.set_xlabel("эффект, п.п.", fontsize=10)
    ax.set_title("Интервал и решение", fontsize=11)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4, axis="x"); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, SIDE / "intervals.png")


# ---------------------------------------------------------------- модельный пример с кнопкой
def button_example() -> None:
    nA = nB = 10000
    xA, xB = 780, 830
    pA, pB = xA / nA, xB / nB
    d = pB - pA
    se = math.sqrt(pA * (1 - pA) / nA + pB * (1 - pB) / nB)
    z = d / se
    p = 2 * (1 - phi(abs(z)))
    lo, hi = d - 1.959964 * se, d + 1.959964 * se
    se10 = math.sqrt(pA * (1 - pA) / 100000 + pB * (1 - pB) / 100000)
    FACTS.update(btn_d=d, btn_se=se, btn_z=z, btn_p=p, btn_lo=lo, btn_hi=hi,
                 btn_se10=se10, btn_z10=d / se10,
                 btn_clicks=10_000_000 * d)
    print(f"button: d={d:.4f} se={se:.5f} z={z:.3f} p={p:.3f} CI=[{lo:.5f},{hi:.5f}] "
          f"se(100k)={se10:.5f} z(100k)={d / se10:.2f}")
    assert abs(se - 0.0038474) < 5e-08 and abs(z - 1.29957) < 5e-06 and abs(p - 0.193749) < 5e-07
    assert abs(lo + 0.00254) < 1e-4 and abs(hi - 0.01254) < 1e-4
    assert abs(se10 - 0.0012167) < 5e-08 and abs(d / se10 - 4.1096) < 1e-3

    # a design-effect number for the cluster problem
    de = 1 + (25 - 1) * 0.2
    FACTS.update(design_effect=de, eff_n=500 / de)
    assert abs(de - 5.8) < 0.05
    print(f"design effect={de}, effective n={500 / de:.1f}")


def exercise_numbers() -> None:
    """Числа, которые произносятся в разборах инлайн-упражнений и в прозе."""
    from scipy.stats import norm  # локально: используется только здесь

    z_a, z_b = norm.ppf(0.975), norm.ppf(0.8)
    # «Пять с половиной процентов»: se доли ложных тревог по B=4000 повторам
    se_share = math.sqrt(0.05 * 0.95 / 4000)
    FACTS["aa_share_se"] = se_share
    FACTS["aa_share_dev"] = (FACTS["aa_share"] - 0.05) / se_share
    # «Хватит ли школы»: 60% -> 68%, alpha=0.05, power=0.8
    var_sum = 0.6 * 0.4 + 0.68 * 0.32
    FACTS["school_z2"] = (z_a + z_b) ** 2
    FACTS["school_var"] = var_sum
    FACTS["school_n"] = (z_a + z_b) ** 2 * var_sum / 0.08 ** 2
    FACTS["school_total"] = 2 * FACTS["school_n"]
    FACTS["school_mde"] = (z_a + z_b) * math.sqrt(2 * 0.64 * 0.36 / 200) * 100
    # «Порог и цена»: Бонферрони на 12 метрик
    FACTS["bonf12"] = 0.05 / 12
    FACTS["bonf12_z"] = norm.ppf(1 - FACTS["bonf12"] / 2)
    FACTS["bonf12_z2"] = (FACTS["bonf12_z"] + z_b) ** 2
    FACTS["bonf12_cost"] = FACTS["bonf12_z2"] / FACTS["school_z2"]
    # «Ночной провал»: порог Бонферрони на 24 сегмента
    FACTS["bonf24"] = 0.05 / 24
    # «Двенадцать тысяч ответов»
    FACTS["poll_se"] = math.sqrt(0.78 * 0.22 / 12000)
    FACTS["poll_lo"] = 78 - 100 * 1.96 * FACTS["poll_se"]
    FACTS["poll_hi"] = 78 + 100 * 1.96 * FACTS["poll_se"]
    FACTS["poll_true"] = 0.12 * 0.78 + 0.88 * 0.5
    FACTS["poll_gap"] = 100 * (0.78 - FACTS["poll_true"])
    FACTS["poll_silent"] = 100000 - 12000
    # Кластеры
    FACTS["de_sqrt"] = math.sqrt(FACTS["design_effect"])
    FACTS["eff_n_alt"] = 500 / (1 + 9 * 0.2)
    # Подглядывание: во сколько раз вырос уровень
    FACTS["peek_ratio"] = FACTS["peek_rate"] / FACTS["fixed_rate"]


def verify_quoted() -> None:
    """Каждое число, которое произносится в тексте урока, проверяется здесь."""
    exercise_numbers()
    f = FACTS
    close = lambda key, val, tol: abs(f[key] - val) <= tol  # noqa: E731
    assert f["bike_n"] == 17379
    assert close("bike_mean", 189.46, 0.005)
    assert close("bike_median", 142, 0)
    assert close("bike_sd", 181.39, 0.005)
    assert close("aa_share", 0.0548, 0.0002)      # 5,5 %
    assert close("aa_sd", 1.008, 0.001)
    assert close("one_a", 187.57, 0.005) and close("one_b", 191.36, 0.005)
    assert close("one_diff", 3.79, 0.005) and close("one_z", 1.38, 0.005)
    assert close("one_p", 0.168, 0.0005)
    assert close("n80_05", 47525, 1) and close("n80_10", 12205, 1) and close("n80_20", 3210, 1)
    assert close("pw10k", 0.250, 0.0005) and close("pw100k", 0.982, 0.0005)
    assert close("peek_rate", 0.2485, 1e-9) and close("fixed_rate", 0.043, 1e-9)
    assert f["peek_batch"] == 434 and f["peek_looks"] == 20
    assert close("fam5", 0.226, 0.0005) and close("fam12", 0.460, 0.0005)
    assert close("fam20", 0.6415, 0.0001) and close("bonf20", 0.0025, 1e-9)
    assert close("sub_rate", 0.713, 0.0005) and close("sub_overall", 0.059, 0.0005)
    assert close("sub_medmax", 2.19, 0.005)
    assert close("sub_ex_all", 1.00, 0.005) and close("sub_ex_max", 2.26, 0.005)
    assert f["sub_ex_hour"] == 0
    assert close("sel_true", 189.46, 0.005) and close("sel_limit", 310.73, 0.005)
    assert close("sel_bias", 121.26, 0.005) and close("sel_bias_pct", 64.0, 0.05)
    assert close("sel_ci_small", 115.7, 0.05) and close("sel_ci_big", 9.3, 0.05)
    assert f["sel_n_small"] == 50 and f["sel_n_big"] == 7943
    assert close("mde1k", 3.40, 0.005) and close("mde10k", 1.07, 0.005) and close("mde100k", 0.34, 0.005)
    assert close("btn_d", 0.005, 1e-9) and close("btn_se", 0.00385, 5e-6)
    assert close("btn_z", 1.30, 0.005) and close("btn_p", 0.194, 0.0005)
    assert close("btn_lo", -0.00254, 5e-6) and close("btn_hi", 0.01254, 5e-6)
    assert close("btn_se10", 0.00122, 5e-6) and close("btn_z10", 4.11, 0.005)
    assert close("btn_clicks", 50000, 1)
    assert close("design_effect", 5.8, 1e-9) and close("eff_n", 86.2, 0.05)
    # числа из разборов упражнений
    assert close("aa_share_se", 0.0034, 5e-5) and close("aa_share_dev", 1.4, 0.05)
    assert close("school_z2", 7.85, 0.005) and close("school_var", 0.4576, 1e-9)
    assert close("school_n", 561, 1) and close("school_total", 1122, 2)
    assert 13.0 <= f["school_mde"] <= 14.0
    assert close("bonf12", 0.00417, 5e-6) and close("bonf12_z", 2.87, 0.005)
    assert close("bonf12_z2", 13.74, 0.005) and close("bonf12_cost", 1.75, 0.01)
    assert round(f["bonf24"], 3) == 0.002  # в тексте округлено до 0,002
    assert close("poll_se", 0.0038, 5e-5)
    assert close("poll_lo", 77.3, 0.05) and close("poll_hi", 78.7, 0.05)
    assert close("poll_true", 0.534, 0.0005) and close("poll_gap", 25, 0.5)
    assert f["poll_silent"] == 88000
    assert close("de_sqrt", 2.4, 0.01) and close("eff_n_alt", 179, 0.5)
    assert close("peek_ratio", 5.78, 0.05)
    print("verify_quoted: все числа урока совпали с расчётом")


def main() -> None:
    fig_aa_real()
    fig_power()
    fig_peeking()
    fig_multiplicity()
    fig_subgroups()
    fig_selection()
    side_mde()
    side_unit()
    side_intervals()
    button_example()
    verify_quoted()
    print("\n--- FACTS ---")
    for k, v in FACTS.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
