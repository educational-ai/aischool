"""Deterministic figures for lesson 58: empirical risk vs true risk.

Real data: scripts/data/bike-sharing-hour.csv (17379 hours, 2011-2012).
Task: binary "busy hour" (cnt above the global median), logistic regression on
hour/weather features, 0/1 loss -> bounded in [0,1], so Hoeffding applies.

Everything quoted in the lesson text is computed here and asserted.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
BIKE = ROOT / "scripts" / "data" / "bike-sharing-hour.csv"
OUT = ROOT / "public" / "figures" / "lessons" / "58"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "58"

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


# ------------------------------------------------------------------ data
def load_bike():
    rows = []
    with open(BIKE) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    n = len(rows)
    hr = np.array([int(r["hr"]) for r in rows])
    temp = np.array([float(r["temp"]) for r in rows])
    hum = np.array([float(r["hum"]) for r in rows])
    wind = np.array([float(r["windspeed"]) for r in rows])
    work = np.array([int(r["workingday"]) for r in rows], float)
    weather = np.array([int(r["weathersit"]) for r in rows], float)
    yr = np.array([int(r["yr"]) for r in rows])
    day = np.array([r["dteday"] for r in rows])
    cnt = np.array([int(r["cnt"]) for r in rows], float)
    ang = 2 * np.pi * hr / 24.0
    X = np.column_stack([
        np.sin(ang), np.cos(ang), np.sin(2 * ang), np.cos(2 * ang),
        temp, hum, wind, work, weather / 4.0,
    ])
    med = float(np.median(cnt))
    y = (cnt > med).astype(int)
    return dict(X=X, y=y, cnt=cnt, yr=yr, day=day, hr=hr, n=n, med=med)


D = load_bike()
print(f"rows={D['n']}, median cnt={D['med']}, share busy={D['y'].mean():.3f}")
assert D["n"] == 17379
FACTS["rows"] = D["n"]
FACTS["median_cnt"] = D["med"]

rng_global = np.random.default_rng(58)
perm = rng_global.permutation(D["n"])
train_idx = perm[:4000]
pool_idx = perm[4000:]          # "весь будущий мир" для приближения истинного риска

base = LogisticRegression(max_iter=2000, C=1.0)
base.fit(D["X"][train_idx], D["y"][train_idx])
pool_err = (base.predict(D["X"][pool_idx]) != D["y"][pool_idx]).astype(float)
R_true = float(pool_err.mean())
print(f"true risk R(f) ~= {R_true:.4f} on pool of {len(pool_idx)}")
assert 0.10 < R_true < 0.30
FACTS["R_true"] = R_true
FACTS["pool"] = len(pool_idx)
FACTS["train_fixed"] = len(train_idx)


# ------------------------------------------- fig 58.1: spread of the estimate
def fig_sampling():
    rng = np.random.default_rng(581)
    sizes = [50, 200, 2000]
    colors = [GOLD, BLUE, GREEN]
    reps = 4000
    stats = {}
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for n, c in zip(sizes, colors):
        est = np.array([pool_err[rng.integers(0, len(pool_err), n)].mean()
                        for _ in range(reps)])
        sd_emp = float(est.std(ddof=1))
        sd_thy = math.sqrt(R_true * (1 - R_true) / n)
        stats[n] = (float(est.mean()), sd_emp, sd_thy)
        print(f"  n={n}: mean={est.mean():.4f}, sd={sd_emp:.4f}, theory={sd_thy:.4f}")
        assert abs(est.mean() - R_true) < 0.01
        # наблюдённая ширина расходится с теоретической меньше чем на 2 % — это и
        # заявлено в подписи к рис. 58.1
        rel = abs(sd_emp - sd_thy) / sd_thy
        assert rel < 0.02, (n, sd_emp, sd_thy, rel)
        FACTS[f"sd_emp{n}"] = sd_emp
        FACTS[f"sd_rel{n}"] = rel
        xs = np.linspace(R_true - 0.11, R_true + 0.11, 600)
        dens = np.exp(-0.5 * ((xs - R_true) / sd_thy) ** 2) / (sd_thy * math.sqrt(2 * math.pi))
        ax.plot(xs, dens, color=c, lw=2.2, label=f"$n={n}$: se $\\approx${sd_thy:.3f}")
        ax.hist(est, bins=40, density=True, color=c, alpha=0.20)
    ax.axvline(R_true, color=RED, lw=2.2)
    ax.text(R_true + 0.003, ax.get_ylim()[1] * 0.93,
            f"истинный риск $R(f)={R_true:.3f}$", color=RED, fontsize=11)
    ax.set_xlabel("оценка $\\widehat R_n(f)$ по одной тестовой выборке")
    ax.set_ylabel("плотность")
    ax.set_title("Одна и та же функция, разные тестовые выборки")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "sampling.png")
    FACTS["se50"] = stats[50][2]; FACTS["se200"] = stats[200][2]; FACTS["se2000"] = stats[2000][2]
    lo = R_true - 1.96 * stats[200][2]; hi = R_true + 1.96 * stats[200][2]
    FACTS["ci200_lo"] = lo; FACTS["ci200_hi"] = hi
    print(f"  95% интервал при n=200: [{lo:.3f}, {hi:.3f}]")
    return stats


STATS = fig_sampling()


# ------------------------------------------- candidates: real family of models
def build_candidates(seed=582, K=120):
    """K реальных моделей: логистические регрессии на случайных подмножествах
    признаков и с разной силой регуляризации, обученные на непересекающихся
    кусках обучающей части."""
    rng = np.random.default_rng(seed)
    X, y = D["X"], D["y"]
    tr = perm[:6000]
    cands = []
    for k in range(K):
        cols = rng.choice(X.shape[1], size=rng.integers(4, X.shape[1] + 1), replace=False)
        Cv = float(10 ** rng.uniform(-2.2, 1.0))
        sub = rng.choice(tr, size=1200, replace=False)
        m = LogisticRegression(max_iter=2000, C=Cv)
        m.fit(X[sub][:, cols], y[sub])
        pred_pool = m.predict(X[pool_idx][:, cols])
        cands.append((pred_pool != D["y"][pool_idx]).astype(float))
    return np.array(cands)   # K x |pool| матрица потерь


CAND = build_candidates()
R_cand = CAND.mean(axis=1)
print(f"candidates: K={len(R_cand)}, best true risk={R_cand.min():.4f}, "
      f"median={np.median(R_cand):.4f}, worst={R_cand.max():.4f}")
FACTS["K"] = int(len(R_cand))
FACTS["cand_best"] = float(R_cand.min())
FACTS["cand_med"] = float(np.median(R_cand))
FACTS["cand_worst"] = float(R_cand.max())
assert R_cand.min() < R_cand.max()


# ------------------------------------------- fig 58.2: winner's curse
def fig_winner():
    rng = np.random.default_rng(583)
    n_val = 200
    idx = rng.integers(0, CAND.shape[1], n_val)
    val = CAND[:, idx].mean(axis=1)
    win = int(np.argmin(val))
    opt = R_cand[win] - val[win]
    print(f"winner: val={val[win]:.4f}, true={R_cand[win]:.4f}, optimism={opt:.4f}, "
          f"best true={R_cand.min():.4f}")
    assert opt > 0.01
    FACTS["win_val"] = float(val[win]); FACTS["win_true"] = float(R_cand[win])
    FACTS["win_opt"] = float(opt); FACTS["n_val"] = n_val

    # средний оптимизм как функция K, усреднение по повторениям
    reps = 400
    Ks = [1, 2, 5, 10, 20, 40, 80, 120]
    mean_opt = []
    for K in Ks:
        acc = []
        for _ in range(reps):
            sub = rng.choice(len(R_cand), size=K, replace=False)
            ii = rng.integers(0, CAND.shape[1], n_val)
            v = CAND[np.ix_(sub, ii)].mean(axis=1)
            w = int(np.argmin(v))
            acc.append(R_cand[sub][w] - v[w])
        mean_opt.append(float(np.mean(acc)))
    print("  mean optimism by K:", [f"{m:.4f}" for m in mean_opt])
    assert mean_opt[-1] > mean_opt[0] + 0.01
    FACTS["opt_K1"] = mean_opt[0]; FACTS["opt_K10"] = mean_opt[Ks.index(10)]
    FACTS["opt_K120"] = mean_opt[-1]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    ax = axes[0]
    order = np.argsort(R_cand)
    xs = np.arange(len(R_cand))
    ax.vlines(xs, np.minimum(R_cand[order], val[order]), np.maximum(R_cand[order], val[order]),
              color=LINE, lw=0.9)
    ax.scatter(xs, R_cand[order], s=16, color=MUTED, label="истинный риск кандидата")
    ax.scatter(xs, val[order], s=16, color=BLUE, alpha=0.75, label=f"оценка по val ($n={n_val}$)")
    wpos = int(np.where(order == win)[0][0])
    ax.scatter([wpos], [val[win]], s=90, color=RED, zorder=6, label="выбранный минимум оценки")
    ax.scatter([wpos], [R_cand[win]], s=90, facecolors="none", edgecolors=RED, lw=1.8, zorder=6)
    ax.annotate("", xy=(wpos, R_cand[win]), xytext=(wpos, val[win]),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))
    ax.text(wpos + 4, (val[win] + R_cand[win]) / 2, f"оптимизм {opt:.3f}", color=RED, fontsize=10.5)
    ax.set_xlabel("кандидаты, упорядоченные по истинному риску")
    ax.set_ylabel("доля ошибок")
    ax.set_title("Побеждает тот, кому повезло", fontsize=13)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)

    ax = axes[1]
    ax.plot(Ks, mean_opt, color=RED, lw=2.4, marker="o", ms=5)
    ax.axhline(0, color=MUTED, lw=0.9, ls=(0, (3, 3)))
    ax.set_xscale("log")
    ax.set_xlabel("число просмотренных кандидатов $K$ (лог-шкала)")
    ax.set_ylabel("средний оптимизм $R-\\widehat R_{val}$")
    ax.set_title("Цена перебора растёт с числом попыток", fontsize=13)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "winner.png")


fig_winner()


# ------------------------------------------- fig 58.3: split design
def fig_splits():
    X, y, day, yr = D["X"], D["y"], D["day"], D["yr"]
    days = np.unique(day)
    rng = np.random.default_rng(584)

    def fit_eval(tr, te):
        m = LogisticRegression(max_iter=2000, C=1.0)
        m.fit(X[tr], y[tr])
        return float((m.predict(X[te]) != y[te]).mean())

    # (1) случайный по строкам
    p = rng.permutation(D["n"]); cut = int(0.7 * D["n"])
    e_rand = fit_eval(p[:cut], p[cut:])
    # (2) по дням (целые сутки не пересекаются)
    dperm = rng.permutation(len(days)); dtr = set(days[dperm[: int(0.7 * len(days))]])
    mask = np.array([d in dtr for d in day])
    e_day = fit_eval(np.where(mask)[0], np.where(~mask)[0])
    # (3) хронологический: учим на 2011, проверяем на 2012
    e_time = fit_eval(np.where(yr == 0)[0], np.where(yr == 1)[0])
    print(f"splits: random={e_rand:.4f}, by-day={e_day:.4f}, chronological={e_time:.4f}")
    assert e_time > e_rand
    FACTS["e_rand"] = e_rand; FACTS["e_day"] = e_day; FACTS["e_time"] = e_time
    FACTS["gap_time"] = e_time - e_rand

    # доля busy в 2011 и 2012 — сдвиг метки
    s11 = float(y[yr == 0].mean()); s12 = float(y[yr == 1].mean())
    c11 = float(D["cnt"][yr == 0].mean()); c12 = float(D["cnt"][yr == 1].mean())
    print(f"  busy share 2011={s11:.3f}, 2012={s12:.3f}; mean cnt {c11:.1f} -> {c12:.1f}")
    FACTS["share11"] = s11; FACTS["share12"] = s12
    FACTS["cnt11"] = c11; FACTS["cnt12"] = c12
    FACTS["growth"] = c12 / c11
    assert s12 > s11

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.2))
    names = ["случайно по строкам", "целыми сутками", "хронологически: 2011, затем 2012"]
    errs = [e_rand, e_day, e_time]
    cols = [BLUE, GREEN, RED]
    quest = ["ещё один час\nиз знакомой недели", "новый день\nв том же сезоне", "следующий год\nсервиса"]
    for ax, nm, er, c, q in zip(axes, names, errs, cols, quest):
        ax.bar([0], [er], width=0.5, color=c, alpha=0.85)
        ax.set_xlim(-0.6, 0.6); ax.set_xticks([])
        ax.set_ylim(0, max(errs) * 1.45)
        ax.text(0, er + max(errs) * 0.05, f"{er:.3f}", ha="center", fontsize=13, color=INK)
        ax.set_title(nm, fontsize=12)
        ax.text(0, -max(errs) * 0.28, q, ha="center", fontsize=10.5, color=MUTED)
        ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    axes[0].set_ylabel("доля ошибок на test")
    fig.suptitle("Одна таблица, три разбиения — три разных вопроса о будущем", y=1.02, fontsize=14)
    fig.tight_layout()
    save(fig, OUT / "splits.png")


fig_splits()


# ------------------------------------------- fig 58.4: Hoeffding
def fig_hoeffding():
    delta = 0.05
    n02 = math.ceil(math.log(2 / delta) / (2 * 0.02 ** 2))
    n01 = math.ceil(math.log(2 / delta) / (2 * 0.01 ** 2))
    print(f"Hoeffding: n(eps=0.02)={n02}, n(eps=0.01)={n01}, ratio={n01/n02:.2f}")
    assert n02 == 4612 and n01 == 18445
    FACTS["n_eps02"] = n02; FACTS["n_eps01"] = n01
    # нормальная («планировочная») формула при R = 0,15 — с ней сравнивается 4612
    n_norm = math.ceil(1.96 ** 2 * 0.15 * 0.85 / 0.02 ** 2)
    print(f"  нормальная формула при R=0.15: n={n_norm}, отношение {n02/n_norm:.2f}")
    assert n_norm == 1225 and 3.5 < n02 / n_norm < 4.0
    FACTS["n_normal_015"] = n_norm
    FACTS["hoef_over_normal"] = n02 / n_norm

    rng = np.random.default_rng(585)
    n = 500
    reps = 20000
    est = np.array([pool_err[rng.integers(0, len(pool_err), n)].mean() for _ in range(reps)])
    eps = np.linspace(0.005, 0.09, 60)
    emp = np.array([float(np.mean(np.abs(est - R_true) >= e)) for e in eps])
    bound = 2 * np.exp(-2 * n * eps ** 2)
    clt = 2 * (1 - 0.5 * (1 + np.vectorize(math.erf)(eps / (math.sqrt(2 * R_true * (1 - R_true) / n)))))
    assert np.all(emp <= bound + 1e-9)
    e_at = 0.03
    b_at = 2 * math.exp(-2 * n * e_at ** 2)
    emp_at = float(np.mean(np.abs(est - R_true) >= e_at))
    print(f"  n=500, eps=0.03: bound={b_at:.4f}, empirical={emp_at:.4f}")
    FACTS["hoef_bound_003"] = b_at; FACTS["hoef_emp_003"] = emp_at

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    ax = axes[0]
    ax.semilogy(eps, bound, color=RED, lw=2.3, label="граница Хёффдинга $2e^{-2n\\varepsilon^2}$")
    ax.semilogy(eps, np.maximum(clt, 1e-6), color=GOLD, lw=1.8, ls=(0, (5, 3)),
                label="нормальное приближение")
    ax.semilogy(eps, np.maximum(emp, 1 / reps), color=BLUE, lw=2.0, label="наблюдённая частота")
    ax.set_xlabel("$\\varepsilon$"); ax.set_ylabel("$P(|\\widehat R-R|\\geq\\varepsilon)$")
    ax.set_title(f"Граница честна, но щедра ($n={n}$)", fontsize=13)
    ax.legend(loc="lower left", frameon=False, fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)

    ax = axes[1]
    ee = np.linspace(0.005, 0.06, 300)
    need = np.log(2 / delta) / (2 * ee ** 2)
    ax.plot(ee, need, color=BLUE, lw=2.4)
    for e0, nn, c in [(0.02, n02, GREEN), (0.01, n01, RED)]:
        ax.plot([e0, e0], [0, nn], color=c, lw=1.4, ls=(0, (3, 3)))
        ax.plot([0, e0], [nn, nn], color=c, lw=1.4, ls=(0, (3, 3)))
        ax.text(e0 + 0.001, nn * 1.02, f"$\\varepsilon={e0}$: $n={nn}$", color=c, fontsize=10.5)
    ax.set_xlim(0.005, 0.06); ax.set_ylim(0, 25000)
    ax.set_xlabel("требуемая точность $\\varepsilon$")
    ax.set_ylabel("нужный объём test при $\\delta=0{,}05$")
    ax.set_title("Точность вдвое дороже вчетверо", fontsize=13)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "hoeffding.png")


fig_hoeffding()


# ------------------------------------------- fig 58.5: shift and reweighting
def fig_shift():
    X, y, yr, hr = D["X"], D["y"], D["yr"], D["hr"]
    i11 = np.where(yr == 0)[0]; i12 = np.where(yr == 1)[0]
    rng = np.random.default_rng(586)
    p = rng.permutation(i11)
    tr, ho = p[: int(0.7 * len(p))], p[int(0.7 * len(p)):]
    m = LogisticRegression(max_iter=2000, C=1.0); m.fit(X[tr], y[tr])
    e_hold = float((m.predict(X[ho]) != y[ho]).mean())
    e_next = float((m.predict(X[i12]) != y[i12]).mean())
    print(f"shift: holdout 2011={e_hold:.4f}, next year 2012={e_next:.4f}")
    assert e_next > e_hold
    FACTS["e_hold11"] = e_hold; FACTS["e_next12"] = e_next
    FACTS["shift_gap"] = e_next - e_hold

    # эффективный размер выборки при перевзвешивании: смещённый по часам "старый" набор
    night_rate = 0.15
    w_hour = np.where((hr >= 7) & (hr <= 19), 1.0, night_rate)   # старый сбор ночью реже
    night_ratio = 1.0 / night_rate
    assert abs(night_ratio - 6.67) < 0.01
    FACTS["night_ratio"] = night_ratio
    sel = rng.random(D["n"]) < w_hour * 0.5
    old = np.where(sel)[0]
    p_new = np.ones(24) / 24
    cnt_old = np.array([np.sum(hr[old] == h) for h in range(24)], float)
    p_old = cnt_old / cnt_old.sum()
    w = p_new[hr[old]] / p_old[hr[old]]
    ess = float(w.sum() ** 2 / np.sum(w ** 2))
    print(f"  reweighting: |old|={len(old)}, ESS={ess:.0f} ({100*ess/len(old):.0f}%), max w={w.max():.2f}")
    assert ess < 0.75 * len(old)
    FACTS["old_n"] = int(len(old)); FACTS["ess"] = ess
    FACTS["ess_pct"] = 100 * ess / len(old); FACTS["w_max"] = float(w.max())

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    ax = axes[0]
    ax.bar([0, 1], [e_hold, e_next], width=0.5, color=[BLUE, RED], alpha=0.85)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["отложенная часть\n2011 года", "весь 2012 год"])
    ax.set_ylabel("доля ошибок")
    for i, v in enumerate([e_hold, e_next]):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=13, color=INK)
    ax.set_ylim(0, max(e_hold, e_next) * 1.35)
    ax.set_title("Тот же test-протокол, другой год", fontsize=13)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)

    ax = axes[1]
    ax.bar(np.arange(24) - 0.2, p_old, width=0.4, color=MUTED, label="старый сбор $p_{old}$")
    ax.bar(np.arange(24) + 0.2, p_new, width=0.4, color=GREEN, label="целевое $p_{new}$")
    ax2 = ax.twinx()
    ax2.plot(np.arange(24), p_new / p_old, color=RED, lw=2.0, marker="o", ms=3.5,
             label="вес $w(x)$")
    ax2.set_ylabel("вес $w$", color=RED); ax2.tick_params(axis="y", colors=RED)
    ax.set_xlabel("час суток"); ax.set_ylabel("доля наблюдений")
    ax.set_title(f"Перевзвешивание: ESS {ess/len(old)*100:.0f}% от строк", fontsize=13)
    ax.legend(loc="upper center", frameon=False, fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "shift.png")


fig_shift()


# ------------------------------------------- sidenote 1: duplicated rows
def side_duplicate():
    n = 200
    se_true = math.sqrt(R_true * (1 - R_true) / n)
    se_fake = math.sqrt(R_true * (1 - R_true) / (2 * n))
    print(f"duplicate: se(n=200)={se_true:.4f}, fake se(2n)={se_fake:.4f}, ratio={se_true/se_fake:.3f}")
    assert abs(se_true / se_fake - math.sqrt(2)) < 1e-9
    FACTS["se_dup_true"] = se_true; FACTS["se_dup_fake"] = se_fake
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    xs = np.linspace(R_true - 0.09, R_true + 0.09, 400)
    for se, c, lab in [(se_true, BLUE, "честные 200 строк"), (se_fake, RED, "«400» из дублей")]:
        ax.plot(xs, np.exp(-0.5 * ((xs - R_true) / se) ** 2), color=c, lw=2.0, label=lab)
    ax.axvline(R_true, color=MUTED, lw=1.0, ls=(0, (3, 3)))
    ax.set_yticks([]); ax.set_xlabel("оценка риска")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "duplicate.png")


side_duplicate()


# ------------------------------------------- sidenote 2: leakage of feature selection
def side_leakage():
    """Модельный пример с фиксированным seed: у признаков нет связи с меткой."""
    rng = np.random.default_rng(587)
    n, p, k = 120, 4000, 20
    Z = rng.normal(size=(n, p))
    lab = rng.integers(0, 2, n)

    def cv_err(Zs, ys):
        errs = []
        folds = np.array_split(np.arange(len(ys)), 5)
        for f in folds:
            m = np.ones(len(ys), bool); m[f] = False
            mo = LogisticRegression(max_iter=1000)
            mo.fit(Zs[m], ys[m])
            errs.append(float((mo.predict(Zs[~m]) != ys[~m]).mean()))
        return float(np.mean(errs))

    cor = np.array([abs(np.corrcoef(Z[:, j], lab)[0, 1]) for j in range(p)])
    top = np.argsort(-cor)[:k]
    err_leak = cv_err(Z[:, top], lab)

    honest = []
    folds = np.array_split(np.arange(n), 5)
    for f in folds:
        m = np.ones(n, bool); m[f] = False
        c = np.array([abs(np.corrcoef(Z[m, j], lab[m])[0, 1]) for j in range(p)])
        t = np.argsort(-c)[:k]
        mo = LogisticRegression(max_iter=1000); mo.fit(Z[m][:, t], lab[m])
        honest.append(float((mo.predict(Z[~m][:, t]) != lab[~m]).mean()))
    err_honest = float(np.mean(honest))
    print(f"leakage: selection outside CV err={err_leak:.3f}, selection inside CV err={err_honest:.3f}")
    assert err_leak < 0.30 and err_honest > 0.40
    FACTS["leak_err"] = err_leak; FACTS["honest_err"] = err_honest
    FACTS["leak_p"] = p; FACTS["leak_n"] = n; FACTS["leak_k"] = k

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.bar([0, 1], [err_leak, err_honest], width=0.5, color=[RED, GREEN], alpha=0.85)
    ax.axhline(0.5, color=MUTED, lw=1.2, ls=(0, (3, 3)))
    ax.text(1.35, 0.505, "монетка", fontsize=9, color=MUTED, ha="right")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["отбор до CV", "отбор внутри CV"], fontsize=9.5)
    ax.set_ylabel("ошибка CV", fontsize=10); ax.set_ylim(0, 0.62)
    for i, v in enumerate([err_leak, err_honest]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=11)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "leakage.png")


side_leakage()


# ------------------------------------------- sidenote 3: mean hides the tail
def side_tail():
    """Реальные абсолютные ошибки регрессии числа поездок: среднее и хвост."""
    X, cnt = D["X"], D["cnt"]
    rng = np.random.default_rng(588)
    p = rng.permutation(D["n"]); cut = int(0.7 * D["n"])
    tr, te = p[:cut], p[cut:]
    A = np.column_stack([np.ones(len(tr)), X[tr]])
    w, *_ = np.linalg.lstsq(A, cnt[tr], rcond=None)
    pred = np.column_stack([np.ones(len(te)), X[te]]) @ w
    err = np.abs(cnt[te] - pred)
    mae = float(err.mean()); med = float(np.median(err))
    q90 = float(np.quantile(err, 0.90)); q99 = float(np.quantile(err, 0.99))
    print(f"tail: MAE={mae:.1f}, median={med:.1f}, q90={q90:.1f}, q99={q99:.1f}")
    assert q99 > 3 * mae
    FACTS["mae"] = mae; FACTS["med_err"] = med; FACTS["q90"] = q90; FACTS["q99"] = q99

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.hist(err, bins=60, range=(0, 500), color=BLUE, alpha=0.55)
    for v, c, lab in [(mae, RED, f"среднее {mae:.0f}"), (med, GREEN, f"медиана {med:.0f}"),
                      (q99, GOLD, f"99% квантиль {q99:.0f}")]:
        ax.axvline(v, color=c, lw=1.8)
        ax.text(v + 6, ax.get_ylim()[1] * (0.9 if c != GOLD else 0.6), lab, color=c, fontsize=9)
    ax.set_xlabel("|ошибка| прогноза числа поездок", fontsize=10)
    ax.set_yticks([])
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "tail.png")


side_tail()

(ROOT / "scripts" / "data" / "lesson58_facts.json").write_text(
    json.dumps(FACTS, ensure_ascii=False, indent=1), encoding="utf-8")
print("\n=== FACTS ===")
for k, v in FACTS.items():
    print(f"{k}: {v}")
print("OK")
