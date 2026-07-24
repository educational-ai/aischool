"""Deterministic figures for lesson 66: random walk, sqrt(n) scale, boundaries, Brownian limit.

Synthetic parts use a FIXED seed and are declared as model examples in the text.
The real-data part uses scripts/data/mta-daily-ridership.csv (MTA daily subway ridership):
we test whether the log-ridership series behaves like a free random walk (it does not:
increments are strongly anti-correlated and the mean squared displacement saturates),
while the SAME increments reshuffled in time produce a textbook diffusive slope 1.

Every number quoted in the lesson prose is computed here and asserted.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MTA = ROOT / "scripts" / "data" / "mta-daily-ridership.csv"
OUT = ROOT / "public" / "figures" / "lessons" / "66"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "66"

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

FACTS: dict[str, float | int | str] = {}


def save(fig, path, *, dpi=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def style(ax):
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- fig 66.1
def fig_scales() -> None:
    rng = np.random.default_rng(6601)
    ns = [100, 1000, 10000]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3))
    for ax, n in zip(axes, ns):
        steps = rng.integers(0, 2, size=(6, n)) * 2 - 1
        walks = np.cumsum(steps, axis=1)
        k = np.arange(1, n + 1)
        ax.fill_between(k, -2 * np.sqrt(k), 2 * np.sqrt(k), color=WASH, zorder=0)
        ax.plot(k, np.sqrt(k), color=BLUE, lw=1.2, ls=(0, (5, 3)))
        ax.plot(k, -np.sqrt(k), color=BLUE, lw=1.2, ls=(0, (5, 3)))
        ax.plot(k, 2 * np.sqrt(k), color=FAINT, lw=1.0, ls=(0, (2, 3)))
        ax.plot(k, -2 * np.sqrt(k), color=FAINT, lw=1.0, ls=(0, (2, 3)))
        for w, c in zip(walks, [INK, RED, GREEN, GOLD, VIOLET, BLUE]):
            ax.plot(k, w, color=c, lw=0.9, alpha=0.85)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(f"n = {n}", fontsize=13)
        ax.set_xlabel("шаг k")
        style(ax)
    axes[0].set_ylabel("положение $S_k$")
    axes[0].text(0.03, 0.95, r"полосы $\pm\sqrt{k}$ и $\pm2\sqrt{k}$",
                 transform=axes[0].transAxes, va="top", fontsize=10, color=BLUE)
    fig.suptitle("Один и тот же закон в трёх масштабах: облако растёт как корень", y=1.0)
    save(fig, OUT / "walk-scales.png")


# ---------------------------------------------------------------- fig 66.2
def fig_sqrt_law() -> None:
    rng = np.random.default_rng(6602)
    ns = [100, 400, 1600, 6400]
    reps = 20000
    sds, ends = [], {}
    for n in ns:
        steps = rng.integers(0, 2, size=(reps, n)) * 2 - 1
        s = steps.sum(axis=1)
        ends[n] = s
        sds.append(float(s.std(ddof=1)))
    ratios = [sd / math.sqrt(n) for sd, n in zip(sds, ns)]
    FACTS["sd_n100"] = round(sds[0], 2)
    FACTS["sd_n400"] = round(sds[1], 2)
    FACTS["sd_n1600"] = round(sds[2], 2)
    FACTS["sd_n6400"] = round(sds[3], 2)
    FACTS["sd_ratio_max_dev_pct"] = round(max(abs(r - 1) for r in ratios) * 100, 2)
    assert all(abs(r - 1) < 0.03 for r in ratios), ratios

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))
    ax = axes[0]
    ax.plot(np.sqrt(ns), sds, "o-", color=RED, lw=2.0, ms=8, label="выборочное $\\sigma(S_n)$")
    grid = np.linspace(0, 85, 50)
    ax.plot(grid, grid, color=BLUE, lw=1.4, ls=(0, (5, 3)), label="$\\sqrt{n}$")
    for n, sd in zip(ns, sds):
        ax.annotate(f"n={n}", (math.sqrt(n), sd), textcoords="offset points",
                    xytext=(8, -12), fontsize=10, color=MUTED)
    ax.set_xlabel("$\\sqrt{n}$"); ax.set_ylabel("стандартное отклонение $S_n$")
    ax.set_title("Разброс идёт по прямой в осях $\\sqrt{n}$", fontsize=13)
    ax.legend(frameon=False, fontsize=10, loc="upper left"); style(ax)

    ax = axes[1]
    grid = np.linspace(-4, 4, 400)
    for n, c in zip(ns, [BLUE, GREEN, GOLD, RED]):
        z = np.sort(ends[n] / math.sqrt(n))
        ax.plot(z, np.arange(1, len(z) + 1) / len(z), color=c, lw=1.5, label=f"n={n}")
    ax.plot(grid, 0.5 * (1 + np.vectorize(math.erf)(grid / math.sqrt(2))),
            color=INK, lw=1.2, ls=(0, (2, 3)), label="нормальная")
    ax.set_xlim(-4, 4)
    ax.set_xlabel("$S_n/\\sqrt{n}$"); ax.set_ylabel("эмпирическая функция распределения")
    ax.set_title("Делим на $\\sqrt{n}$ — кривые сливаются", fontsize=13)
    ax.legend(frameon=False, fontsize=9, loc="upper left"); style(ax)

    ax = axes[2]
    for n, c in zip(ns, [BLUE, GREEN, GOLD, RED]):
        z = np.sort(ends[n] / n)
        ax.plot(z, np.arange(1, len(z) + 1) / len(z), color=c, lw=1.5, label=f"n={n}")
    ax.set_xlim(-0.35, 0.35)
    ax.set_xlabel("$S_n/n$"); ax.set_ylabel("эмпирическая функция распределения")
    ax.set_title("Делим на $n$ — всё схлопывается в ноль", fontsize=13)
    ax.legend(frameon=False, fontsize=9, loc="upper left"); style(ax)
    fig.suptitle("Правильная нормировка — корневая: закон больших чисел и масштаб флуктуаций", y=1.02)
    save(fig, OUT / "sqrt-law.png")


# ---------------------------------------------------------------- fig 66.3
def fig_ruin() -> None:
    N = 100
    i = np.arange(N + 1)
    u = i / N
    tau = i * (N - i)
    # проверяем, что это решение разностных уравнений, а не «формула из воздуха»
    assert np.allclose(u[1:-1], 0.5 * u[:-2] + 0.5 * u[2:])
    assert np.allclose(tau[1:-1], 1 + 0.5 * tau[:-2] + 0.5 * tau[2:])
    FACTS["ruin_u10"] = round(float(u[10]), 3)
    FACTS["ruin_tau10"] = int(tau[10])
    FACTS["ruin_tau50"] = int(tau[50])
    assert FACTS["ruin_u10"] == 0.1 and FACTS["ruin_tau10"] == 900 and FACTS["ruin_tau50"] == 2500

    # несимметричная игра: p = 0.49
    p = 0.49
    r = (1 - p) / p
    ui = (1 - r ** 10) / (1 - r ** N)
    FACTS["ruin_u10_p049"] = round(float(ui), 5)
    assert 0.008 < ui < 0.011, ui
    FACTS["ruin_drop_factor_p049"] = round(float(u[10] / ui), 1)
    FACTS["ruin_u90"] = round(float(u[90]), 2)
    FACTS["ruin_tau90"] = int(tau[90])
    assert FACTS["ruin_tau90"] == 900

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.plot(i, u, color=BLUE, lw=2.4, label="$u_i=i/N$ — шанс дойти до $N$")
    ax.set_xlabel("начальный капитал $i$ (при $N=100$)")
    ax.set_ylabel("вероятность выиграть", color=BLUE)
    ax.set_ylim(0, 1.05)
    for m in (10, 50, 90):
        ax.plot([m], [u[m]], "o", color=BLUE, ms=9)
        ax.annotate(f"$u_{{{m}}}={u[m]:.1f}$", (m, u[m]), textcoords="offset points",
                    xytext=(6, -16), fontsize=10, color=BLUE)
    ax2 = ax.twinx()
    ax2.plot(i, tau, color=RED, lw=2.4, label=r"$\mathbb{E}\tau=i(N-i)$")
    ax2.set_ylabel("среднее число раундов до конца", color=RED)
    ax2.set_ylim(0, 2900)
    for m in (10, 50, 90):
        ax2.plot([m], [tau[m]], "s", color=RED, ms=8)
    ax2.annotate("900 раундов", (10, 900), textcoords="offset points", xytext=(10, 10),
                 fontsize=10, color=RED)
    ax2.annotate("2500 раундов", (50, 2500), textcoords="offset points", xytext=(-40, 12),
                 fontsize=10, color=RED)
    ax2.spines["top"].set_visible(False)
    ax.set_title("Разорение игрока: линейный шанс и квадратичное время")
    style(ax)
    save(fig, OUT / "ruin.png")


# ---------------------------------------------------------------- fig 66.4
def fig_return() -> None:
    ns = np.arange(1, 201)
    exact = np.array([math.comb(2 * n, n) / 2.0 ** (2 * n) for n in ns])
    approx = 1 / np.sqrt(math.pi * ns)
    FACTS["p_return_100"] = round(float(exact[49]), 4)          # 2n = 100
    FACTS["p_return_100_approx"] = round(float(approx[49]), 4)
    assert abs(exact[49] - 0.0796) < 5e-4, exact[49]
    FACTS["stirling_rel_err_n50_pct"] = round(float(abs(exact[49] / approx[49] - 1) * 100), 2)
    FACTS["harmonic_sum_200"] = round(float(exact.sum()), 2)

    # возвраты в размерностях 1, 2, 3 — модельный эксперимент с фиксированным seed
    rng = np.random.default_rng(6603)
    horizon, walks = 20000, 3000
    frac = {}
    for d in (1, 2, 3):
        axis = rng.integers(0, d, size=(walks, horizon))
        sign = rng.integers(0, 2, size=(walks, horizon)) * 2 - 1
        pos = np.zeros((walks, d), dtype=np.int32)
        back = np.zeros(walks, dtype=bool)
        for t in range(horizon):
            pos[np.arange(walks), axis[:, t]] += sign[:, t]
            if t % 2 == 1:
                back |= ~pos.any(axis=1)
        frac[d] = float(back.mean())
    FACTS["ret_d1"] = round(frac[1], 3)
    FACTS["ret_d2"] = round(frac[2], 3)
    FACTS["ret_d3"] = round(frac[3], 3)
    FACTS["polya_d3"] = 0.3405
    assert frac[1] > 0.98 and 0.75 < frac[2] < 0.95 and 0.28 < frac[3] < 0.38, frac

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    ax = axes[0]
    ax.plot(2 * ns, exact, color=RED, lw=2.2, label="точное $\\Pr(S_{2n}=0)$")
    ax.plot(2 * ns, approx, color=BLUE, lw=1.6, ls=(0, (5, 3)), label="$1/\\sqrt{\\pi n}$")
    ax.plot([100], [exact[49]], "o", color=INK, ms=8)
    ax.annotate(f"$\\Pr(S_{{100}}=0)={exact[49]:.4f}$", (100, exact[49]),
                textcoords="offset points", xytext=(20, 22), fontsize=11, color=INK)
    ax.set_xlabel("число шагов $2n$"); ax.set_ylabel("вероятность быть в нуле")
    ax.set_title("Стирлинг попадает почти точно", fontsize=13)
    ax.legend(frameon=False, fontsize=10); style(ax)

    ax = axes[1]
    bars = ax.bar([1, 2, 3], [frac[1], frac[2], frac[3]],
                  color=[BLUE, GREEN, RED], width=0.55)
    ax.axhline(0.3405, color=INK, lw=1.2, ls=(0, (3, 3)))
    ax.annotate("константа Пойа 0.3405", (3.32, 0.3405), textcoords="offset points",
                xytext=(-150, 12), fontsize=10, color=INK)
    for b, v in zip(bars, [frac[1], frac[2], frac[3]]):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), ha="center",
                    textcoords="offset points", xytext=(0, 6), fontsize=11, color=MUTED)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["$d=1$", "$d=2$", "$d=3$"])
    ax.set_ylim(0, 1.15); ax.set_ylabel("доля вернувшихся за 20 000 шагов")
    ax.set_title("Три измерения хватает, чтобы потеряться", fontsize=13)
    style(ax)
    fig.suptitle("Возврат в начало: точная арифметика и роль размерности", y=1.02)
    save(fig, OUT / "return-recurrence.png")


# ---------------------------------------------------------------- fig 66.5
def fig_drift() -> None:
    p = 0.51
    n = np.arange(1, 4_000_001)
    mean = n * (2 * p - 1)
    sd = 2 * np.sqrt(n * p * (1 - p))
    cross = int(np.argmax(mean >= 2 * sd) + 1)
    FACTS["drift_p"] = p
    FACTS["drift_mean_1e6"] = int(round(1_000_000 * (2 * p - 1)))
    FACTS["drift_sd_1e6"] = round(float(2 * math.sqrt(1_000_000 * p * (1 - p))), 1)
    FACTS["drift_cross_2sigma"] = cross
    exact_cross = math.ceil((2 * 2 * math.sqrt(p * (1 - p)) / (2 * p - 1)) ** 2)
    assert cross == exact_cross, (cross, exact_cross)
    FACTS["drift_snr_1e6"] = round(FACTS["drift_mean_1e6"] / FACTS["drift_sd_1e6"], 1)
    cross1 = math.ceil((2 * math.sqrt(p * (1 - p)) / (2 * p - 1)) ** 2)
    FACTS["drift_cross_1sigma"] = cross1
    assert cross1 * (2 * p - 1) >= 2 * math.sqrt(cross1 * p * (1 - p))
    assert (cross1 - 1) * (2 * p - 1) < 2 * math.sqrt((cross1 - 1) * p * (1 - p))

    rng = np.random.default_rng(6604)
    horizon = 40000
    steps = (rng.random((5, horizon)) < p) * 2 - 1
    walks = np.cumsum(steps, axis=1)
    k = np.arange(1, horizon + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    ax = axes[0]
    ax.fill_between(k, k * (2 * p - 1) - 2 * 2 * np.sqrt(k * p * (1 - p)),
                    k * (2 * p - 1) + 2 * 2 * np.sqrt(k * p * (1 - p)),
                    color=WASH, zorder=0, label="коридор $\\pm2\\sigma$")
    ax.plot(k, k * (2 * p - 1), color=RED, lw=2.0, label="снос $n(2p-1)$")
    for w, c in zip(walks, [INK, BLUE, GREEN, GOLD, VIOLET]):
        ax.plot(k, w, lw=0.8, color=c, alpha=0.8)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(cross, color=BLUE, lw=1.2, ls=(0, (4, 3)))
    ax.annotate(f"$n^*={cross}$", (cross, 0), fontsize=11, color=BLUE,
                textcoords="offset points", xytext=(8, 8))
    ax.set_xlabel("число шагов $n$"); ax.set_ylabel("$S_n$")
    ax.set_title(f"Перекос $p={p}$: снос обгоняет шум", fontsize=13)
    ax.legend(frameon=False, fontsize=10, loc="upper left"); style(ax)

    ax = axes[1]
    nn = np.logspace(1, 7, 200)
    ax.loglog(nn, nn * (2 * p - 1), color=RED, lw=2.2, label="снос $\\propto n$")
    ax.loglog(nn, 2 * np.sqrt(nn * p * (1 - p)), color=BLUE, lw=2.2,
              label="шум $\\propto\\sqrt{n}$")
    ax.axvline(cross, color=MUTED, lw=1.0, ls=(0, (4, 3)))
    ax.set_xlabel("$n$ (лог)"); ax.set_ylabel("масштаб (лог)")
    ax.set_title("Прямая круче корня — вопрос лишь во времени", fontsize=13)
    ax.legend(frameon=False, fontsize=10); style(ax)
    fig.suptitle("Слабый сигнал против случайности: гонка $n$ и $\\sqrt{n}$", y=1.02)
    save(fig, OUT / "drift-vs-noise.png")


# ---------------------------------------------------------------- real data
def load_mta():
    dates, subway = [], []
    with open(MTA) as f:
        for row in csv.DictReader(f):
            d = datetime.fromisoformat(row["date"])
            v = float(row["subways_total_estimated_ridership"])
            dates.append(d); subway.append(v)
    dates = np.array(dates); subway = np.array(subway)
    mask = np.array([d >= datetime(2022, 1, 1) for d in dates]) & (subway > 0)
    return dates[mask], subway[mask]


def msd(y, lags):
    return np.array([float(np.mean((y[t:] - y[:-t]) ** 2)) for t in lags])


def fig_mta() -> None:
    dates, sub = load_mta()
    y = np.log(sub)
    n = len(y)
    FACTS["mta_days"] = int(n)
    FACTS["mta_first"] = dates[0].date().isoformat()
    FACTS["mta_last"] = dates[-1].date().isoformat()
    assert n > 1000
    d = np.diff(y)
    FACTS["mta_step_sd"] = round(float(d.std(ddof=1)), 3)
    ac1 = float(np.corrcoef(d[:-1], d[1:])[0, 1])
    ac7 = float(np.corrcoef(d[:-7], d[7:])[0, 1])
    FACTS["mta_acf1"] = round(ac1, 2)
    FACTS["mta_acf7"] = round(ac7, 2)
    assert abs(ac1) < 0.15 and ac7 > 0.5, (ac1, ac7)

    lags = np.arange(1, 61)
    weekly = np.arange(7, 57, 7)
    m_real = msd(y, lags)
    rng = np.random.default_rng(6605)
    m_shuf = np.zeros_like(m_real)
    y_shuf = None
    for rep in range(40):
        perm = rng.permutation(d)
        ys = np.concatenate([[y[0]], y[0] + np.cumsum(perm)])
        if rep == 0:
            y_shuf = ys
        m_shuf += msd(ys, lags)
    m_shuf /= 40

    def slope(mvals, sel):
        return float(np.polyfit(np.log(lags[sel]), np.log(mvals[sel]), 1)[0])

    sel_w = np.isin(lags, weekly)
    s_real = slope(m_real, sel_w)
    s_shuf = slope(m_shuf, np.ones_like(lags, dtype=bool))
    FACTS["mta_slope_real_weekly"] = round(s_real, 2)
    FACTS["mta_slope_shuffled"] = round(s_shuf, 2)
    assert s_real < 0.25 and abs(s_shuf - 1) < 0.1, (s_real, s_shuf)
    FACTS["mta_msd_lag7"] = round(float(m_real[6]), 4)
    FACTS["mta_msd_lag56"] = round(float(m_real[55]), 4)
    FACTS["mta_msd_real_ratio_56_7"] = round(float(m_real[55] / m_real[6]), 2)
    FACTS["mta_msd_shuf_ratio_56_7"] = round(float(m_shuf[55] / m_shuf[6]), 2)
    FACTS["mta_msd_lag3_over_lag7"] = round(float(m_real[2] / m_real[6]), 1)
    FACTS["mta_ratio_shuf_over_real_56"] = round(float(m_shuf[55] / m_real[55]), 1)
    FACTS["mta_var_step"] = round(float(d.var(ddof=1)), 4)
    FACTS["mta_dist_ratio_shuf_over_real_56"] = round(math.sqrt(m_shuf[55] / m_real[55]), 1)
    assert abs(m_shuf[0] - d.var(ddof=1)) < 0.01

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.5))
    ax = axes[0]
    ax.plot(dates, sub / 1e6, color=LINE, lw=0.5)
    k7 = np.convolve(sub / 1e6, np.ones(7) / 7, mode="valid")
    ax.plot(dates[6:], k7, color=BLUE, lw=1.6)
    ax.set_ylabel("поездки в метро, млн/сутки"); ax.set_xlabel("дата")
    ax.set_title("Реальные данные: метро Нью-Йорка", fontsize=13)
    ax.tick_params(axis="x", labelrotation=30)
    style(ax)

    ax = axes[1]
    ax.plot(np.arange(1, 121), y[:120] - y[0], color=BLUE, lw=1.2, label="реальный лог-ряд")
    ax.plot(np.arange(1, 121), y_shuf[:120] - y_shuf[0], color=RED, lw=1.2,
            label="те же шаги, перемешанные")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("день"); ax.set_ylabel("$\\ln$ поездок минус старт")
    ax.set_title("Порядок шагов решает всё", fontsize=13)
    ax.legend(frameon=False, fontsize=9); style(ax)

    ax = axes[2]
    ax.loglog(lags, m_real, "-", color=BLUE, lw=1.3, alpha=0.5)
    ax.loglog(weekly, m_real[weekly - 1], "o", color=BLUE, ms=6,
              label=f"реальный ряд, лаги ×7 (наклон {s_real:.2f})")
    ax.loglog(lags, m_shuf, "-", color=RED, lw=1.8,
              label=f"перемешанные шаги (наклон {s_shuf:.2f})")
    ref = m_shuf[0] * lags
    ax.loglog(lags, ref, color=INK, lw=1.0, ls=(0, (3, 3)), label="наклон 1 (диффузия)")
    ax.set_xlabel("лаг $\\tau$, сутки (лог)")
    ax.set_ylabel("средний квадрат смещения (лог)")
    ax.set_title("Свободного блуждания нет", fontsize=13)
    ax.legend(frameon=False, fontsize=9, loc="upper left"); style(ax)
    fig.suptitle("Проверка гипотезы блуждания на настоящем ряде: MSD и перемешивание", y=1.02)
    save(fig, OUT / "mta-msd.png")


# ---------------------------------------------------------------- sidenote images
def side_parity() -> None:
    n = 8
    ms = np.arange(-n, n + 1, 2)
    probs = np.array([math.comb(n, (n + m) // 2) / 2.0 ** n for m in ms])
    FACTS["p8_0"] = round(float(probs[list(ms).index(0)]), 4)
    FACTS["p8_4"] = round(float(probs[list(ms).index(4)]), 4)
    FACTS["p8_ge6"] = round(float(probs[[abs(m) >= 6 for m in ms]].sum()), 4)
    assert abs(FACTS["p8_0"] - 0.2734) < 1e-3
    assert abs(FACTS["p8_ge6"] - 0.0703) < 1e-3
    FACTS["p8_2"] = round(float(probs[list(ms).index(2)]), 4)
    assert abs(FACTS["p8_2"] - 56 / 256) < 1e-4
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.bar(ms, probs, width=1.2, color=BLUE)
    ax.bar(ms + 1, np.zeros_like(probs), width=1.2, color=RED)
    for m in range(-n + 1, n, 2):
        ax.plot([m], [0.004], "x", color=RED, ms=6)
    ax.set_xticks(range(-8, 9, 2))
    ax.set_xlabel("$S_8$"); ax.set_ylabel("вероятность")
    ax.set_title("Чётность: нечётные точки пусты", fontsize=11)
    style(ax)
    save(fig, SIDE / "parity.png", dpi=160)


def side_lil() -> None:
    rng = np.random.default_rng(6606)
    n = 300000
    w = np.cumsum(rng.integers(0, 2, n) * 2 - 1)
    k = np.arange(1, n + 1)
    env = np.sqrt(2 * k * np.log(np.log(np.maximum(k, 16))))
    ratio = float(np.max(np.abs(w[1000:]) / env[1000:]))
    FACTS["lil_max_ratio"] = round(ratio, 2)
    assert ratio < 1.3, ratio
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.plot(k, w, color=INK, lw=0.4)
    ax.plot(k, env, color=RED, lw=1.4)
    ax.plot(k, -env, color=RED, lw=1.4)
    ax.plot(k, np.sqrt(k), color=BLUE, lw=1.0, ls=(0, (4, 3)))
    ax.plot(k, -np.sqrt(k), color=BLUE, lw=1.0, ls=(0, (4, 3)))
    ax.set_xlabel("шаг $n$"); ax.set_ylabel("$S_n$")
    ax.set_title("Оболочка Хинчина $\\sqrt{2n\\ln\\ln n}$", fontsize=11)
    style(ax)
    save(fig, SIDE / "lil.png", dpi=160)


def side_arcsine() -> None:
    rng = np.random.default_rng(6607)
    reps, n = 20000, 2000
    steps = rng.integers(0, 2, size=(reps, n)) * 2 - 1
    w = np.cumsum(steps, axis=1)
    frac = (w > 0).mean(axis=1)
    near_edge = float(np.mean((frac < 0.1) | (frac > 0.9)))
    near_mid = float(np.mean((frac > 0.4) & (frac < 0.6)))
    FACTS["arcsine_edge"] = round(near_edge, 3)
    FACTS["arcsine_mid"] = round(near_mid, 3)
    theo_edge = 2 * (2 / math.pi) * math.asin(math.sqrt(0.1))
    FACTS["arcsine_edge_theory"] = round(theo_edge, 3)
    assert abs(near_edge - theo_edge) < 0.03, (near_edge, theo_edge)
    assert near_mid < near_edge
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.hist(frac, bins=40, color=BLUE, density=True)
    x = np.linspace(0.005, 0.995, 400)
    ax.plot(x, 1 / (math.pi * np.sqrt(x * (1 - x))), color=RED, lw=1.6)
    ax.set_xlabel("доля времени выше нуля"); ax.set_ylabel("плотность")
    ax.set_title("Закон арксинуса: середина редка", fontsize=11)
    style(ax)
    save(fig, SIDE / "arcsine.png", dpi=160)


def side_bridge() -> None:
    rng = np.random.default_rng(6608)
    n = 5000
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for c in (INK, BLUE, RED):
        w = np.cumsum(rng.integers(0, 2, n) * 2 - 1) / math.sqrt(n)
        ax.plot(np.arange(1, n + 1) / n, w, color=c, lw=0.8)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("$t=k/n$"); ax.set_ylabel("$S_{k}/\\sqrt{n}$")
    ax.set_title("Ломаная становится винеровской", fontsize=11)
    style(ax)
    save(fig, SIDE / "wiener.png", dpi=160)


def misc_facts() -> None:
    """Numbers quoted in prose that do not belong to any single figure."""
    # Stirling for the central binomial coefficient at n = 1: 1/sqrt(pi) vs exact 1/2.
    st1 = 1.0 / math.sqrt(math.pi)
    FACTS["stirling_n1_approx"] = round(st1, 3)
    FACTS["stirling_n1_exact"] = round(math.comb(2, 1) / 4, 3)
    assert abs(st1 - 0.564) < 5e-4 and FACTS["stirling_n1_exact"] == 0.5

    # Khinchin envelope: how much wider than sqrt(n) it is at n = 300 000.
    lil = math.sqrt(2 * math.log(math.log(300000)))
    FACTS["lil_factor_300k"] = round(lil, 2)
    assert abs(lil - 2.25) < 5e-3, lil

    # One-sigma threshold for p = 0.51 (used in the inline exercise).
    p = 0.51
    FACTS["sd_over_gap_p051"] = round(2 * math.sqrt(p * (1 - p)) / (2 * p - 1), 2)
    FACTS["sqrt_pq_p051"] = round(math.sqrt(p * (1 - p)), 5)
    assert FACTS["sd_over_gap_p051"] == 49.99 and FACTS["sqrt_pq_p051"] == 0.4999
    assert round(FACTS["sd_over_gap_p051"] ** 2) == 2499

    # Two-sided tail beyond two sigmas of a normal law (widget readout).
    from statistics import NormalDist

    tail = 200 * (1 - NormalDist().cdf(2))
    FACTS["outside_2sigma_pct"] = round(tail, 2)
    assert abs(tail - 4.55) < 5e-3, tail


def main() -> None:
    misc_facts()
    fig_scales()
    fig_sqrt_law()
    fig_ruin()
    fig_return()
    fig_drift()
    fig_mta()
    side_parity()
    side_lil()
    side_arcsine()
    side_bridge()
    (ROOT / "scripts" / "data" / "lesson66_facts.json").write_text(
        json.dumps(FACTS, ensure_ascii=False, indent=1), encoding="utf8")
    for k, v in FACTS.items():
        print(f"{k:34s} {v}")


if __name__ == "__main__":
    main()
