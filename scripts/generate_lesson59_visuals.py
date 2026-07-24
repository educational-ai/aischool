"""Deterministic figures for lesson 59: double descent and the interpolation threshold.

Real data only:
  * bike-sharing-hour.csv  -> honest bias-variance-noise decomposition (the "true"
    function is the two-year mean of rides per hour; a training sample is one real
    observation per hour, so the noise term is measured, not invented);
  * sklearn load_digits    -> random-ReLU-feature ridgeless regression, the classical
    double-descent curve with a peak exactly at p = n, its spectral explanation, the
    ridge damping, the sample-wise version and the label-noise dependence.

Every number quoted in the lesson text is computed here and asserted.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

ROOT = Path(__file__).resolve().parents[1]
BIKE = ROOT / "scripts" / "data" / "bike-sharing-hour.csv"
OUT = ROOT / "public" / "figures" / "lessons" / "59"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "59"

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


# ============================================================ real bike data
def bike_by_hour() -> list[np.ndarray]:
    buckets: list[list[int]] = [[] for _ in range(24)]
    with open(BIKE) as f:
        for row in csv.DictReader(f):
            buckets[int(row["hr"])].append(int(row["cnt"]))
    return [np.array(b, dtype=float) for b in buckets]


# ---------------------------------------------- fig 59.1: bias / variance / noise (real)
def fig_bias_variance():
    buckets = bike_by_hour()
    truth = np.array([b.mean() for b in buckets])
    noise = float(np.mean([b.var() for b in buckets]))
    rng = np.random.default_rng(59)
    degrees = list(range(0, 13))
    trials = 400
    x = np.arange(24, dtype=float)
    xc = (x - x.mean()) / 12.0
    bias2, var, tot = [], [], []
    preds_by_deg = {}
    for d in degrees:
        P = np.zeros((trials, 24))
        for t in range(trials):
            idx = [rng.integers(len(b)) for b in buckets]
            y = np.array([buckets[h][idx[h]] for h in range(24)], dtype=float)
            co = np.polyfit(xc, y, d)
            P[t] = np.polyval(co, xc)
        preds_by_deg[d] = P
        m = P.mean(axis=0)
        bias2.append(float(np.mean((m - truth) ** 2)))
        var.append(float(np.mean(P.var(axis=0))))
        tot.append(bias2[-1] + var[-1])
    bias2 = np.array(bias2); var = np.array(var); tot = np.array(tot)
    best = int(degrees[int(np.argmin(tot))])
    print("bias2 :", np.round(bias2, 0))
    print("var   :", np.round(var, 0))
    print("total :", np.round(tot, 0), "best degree", best, "noise", round(noise))
    # monotonicity checks
    assert bias2[0] > bias2[-1], "bias must fall with flexibility"
    assert var[-1] > var[0] * 3, "variance must grow with flexibility"
    assert 3 <= best <= 8
    FACTS.update(
        bv_noise=noise, bv_best_degree=best,
        bv_bias0=float(bias2[0]), bv_bias_best=float(bias2[best]),
        bv_var0=float(var[0]), bv_var_best=float(var[best]), bv_var12=float(var[-1]),
        bv_tot_best=float(tot[best]), bv_tot12=float(tot[-1]), bv_tot0=float(tot[0]),
        bv_tot3=float(tot[3]),
        bv_ratio=float(var[-1] / var[best]),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5))
    ax = axes[0]
    ax.plot(degrees, bias2, color=BLUE, lw=2.2, marker="o", ms=4, label="bias$^2$")
    ax.plot(degrees, var, color=RED, lw=2.2, marker="o", ms=4, label="variance")
    ax.axhline(noise, color=MUTED, lw=1.6, ls=(0, (5, 3)), label="шум (измеренный)")
    ax.plot(degrees, tot, color=INK, lw=2.6, label="bias$^2$ + variance (устранимая часть)")
    ax.axvline(best, color=GOLD, lw=1.2, ls=(0, (2, 2)))
    ax.text(best + 0.25, tot.min() * 0.55, f"минимум:\nстепень {best}", color=GOLD, fontsize=10)
    ax.set_yscale("log")
    ax.set_xlabel("степень полинома"); ax.set_ylabel("вклад в MSE, поездок$^2$")
    ax.set_title("Разложение на реальном велопрокате", fontsize=13)
    ax.legend(frameon=False, fontsize=9.5, loc="lower left")
    ax = axes[1]
    for d, c in [(1, BLUE), (best, GREEN), (12, RED)]:
        P = preds_by_deg[d]
        lo, hi = np.percentile(P, [5, 95], axis=0)
        ax.fill_between(x, lo, hi, color=c, alpha=0.14)
        ax.plot(x, P.mean(axis=0), color=c, lw=2.0, label=f"степень {d}")
    ax.plot(x, truth, color=INK, lw=2.4, ls=(0, (4, 2)), label="истинное среднее")
    ax.set_xlabel("час суток"); ax.set_ylabel("поездок в час")
    ax.set_title("Средняя кривая и коридор 5–95%", fontsize=13)
    ax.set_xticks(range(0, 24, 3))
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    for a in axes:
        a.grid(True, color=GRID, lw=0.4, alpha=0.4); a.set_axisbelow(True)
    fig.suptitle("Классическая картина: bias падает, variance растёт, шум неустраним", y=1.02, fontsize=14.5)
    fig.tight_layout()
    save(fig, OUT / "bias_variance.png")


# ============================================================ digits + random ReLU features
def digits_split(n_train=100, n_test=1200, seed=59, noise=0.0):
    X, y = load_digits(return_X_y=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    X = X[perm] / 16.0; y = y[perm]
    Xtr, ytr = X[:n_train], y[:n_train].copy()
    Xte, yte = X[n_train:n_train + n_test], y[n_train:n_train + n_test]
    if noise > 0:
        k = int(round(noise * n_train))
        flip = rng.choice(n_train, size=k, replace=False)
        ytr[flip] = rng.integers(0, 10, size=k)
    mu = Xtr.mean(axis=0)
    return Xtr - mu, ytr, Xte - mu, yte


def rff_error(Xtr, ytr, Xte, yte, p, seed, lam=0.0, want_extra=False):
    rng = np.random.default_rng(1000 + seed)
    d = Xtr.shape[1]
    W = rng.normal(0, 1.0 / np.sqrt(d), size=(d, p))
    b = rng.uniform(-0.3, 0.3, size=p)
    Ztr = np.maximum(0.0, Xtr @ W + b)
    Zte = np.maximum(0.0, Xte @ W + b)
    Y = np.eye(10)[ytr]
    if lam > 0:
        n, pp = Ztr.shape
        if pp <= n:
            A = Ztr.T @ Ztr + lam * np.eye(pp)
            Wt = np.linalg.solve(A, Ztr.T @ Y)
        else:
            A = Ztr @ Ztr.T + lam * np.eye(n)
            Wt = Ztr.T @ np.linalg.solve(A, Y)
    else:
        Wt = np.linalg.pinv(Ztr, rcond=1e-12) @ Y
    tr = float(np.mean(np.argmax(Ztr @ Wt, axis=1) != ytr))
    te = float(np.mean(np.argmax(Zte @ Wt, axis=1) != yte))
    if not want_extra:
        return tr, te
    sv = np.linalg.svd(Ztr, compute_uv=False)
    return tr, te, float(np.linalg.norm(Wt)), float(sv.min()), float(sv.max()), sv


PGRID = [5, 10, 20, 30, 50, 70, 85, 95, 100, 105, 115, 130, 160, 200, 300, 500, 800, 1200, 2000]
SEEDS = list(range(8))
N_TRAIN = 100


def sweep(noise=0.0, lam=0.0, extra=False):
    tr = np.zeros(len(PGRID)); te = np.zeros(len(PGRID))
    nrm = np.zeros(len(PGRID)); smin = np.zeros(len(PGRID)); cond = np.zeros(len(PGRID))
    for s in SEEDS:
        Xtr, ytr, Xte, yte = digits_split(N_TRAIN, seed=59 + s, noise=noise)
        for i, p in enumerate(PGRID):
            if extra:
                a, b_, nn, s0, s1, _ = rff_error(Xtr, ytr, Xte, yte, p, s, lam, True)
                nrm[i] += nn; smin[i] += s0; cond[i] += s1 / max(s0, 1e-12)
            else:
                a, b_ = rff_error(Xtr, ytr, Xte, yte, p, s, lam)
            tr[i] += a; te[i] += b_
    k = len(SEEDS)
    return tr / k, te / k, nrm / k, smin / k, cond / k


# ---------------------------------------------- fig 59.2: the double descent curve (real)
CACHE: dict[str, object] = {}


def fig_double_descent():
    tr, te, nrm, smin, cond = sweep(noise=0.20, extra=True)
    CACHE["tr"], CACHE["te"], CACHE["nrm"], CACHE["smin"], CACHE["cond"] = tr, te, nrm, smin, cond
    P = np.array(PGRID)
    i_peak = int(np.argmax(te[(P >= 70) & (P <= 160)]) + np.where(P >= 70)[0][0])
    left = te[P < 70]; i_left = int(np.argmin(left))
    i_right = len(P) - 1
    print("p    :", P)
    print("train:", np.round(tr * 100, 1))
    print("test :", np.round(te * 100, 1))
    print("peak at p =", P[i_peak], "->", round(te[i_peak] * 100, 1))
    print("first min p =", P[i_left], "->", round(left[i_left] * 100, 1))
    print("final p =", P[i_right], "->", round(te[i_right] * 100, 1))
    assert P[i_peak] == N_TRAIN, "peak must sit exactly at the interpolation threshold"
    assert te[i_peak] > left[i_left] + 0.03, "peak must exceed the first minimum"
    assert te[i_right] < left[i_left] - 0.02, "second descent must beat the first minimum"
    i100 = int(np.where(P == 100)[0][0])
    assert tr[i100] < 1e-9, "training error must vanish at p = n"
    FACTS.update(
        dd_n=N_TRAIN, dd_noise=20,
        dd_first_min_p=int(P[i_left]), dd_first_min=float(left[i_left]),
        dd_peak_p=int(P[i_peak]), dd_peak=float(te[i_peak]),
        dd_final_p=int(P[i_right]), dd_final=float(te[i_right]),
        dd_p2000_train=float(tr[i_right]),
        dd_p30=float(te[int(np.where(P == 30)[0][0])]),
    )
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.axvspan(85, 130, color=WASH, alpha=0.9)
    ax.plot(P, te * 100, color=RED, lw=2.6, marker="o", ms=4.5, label="ошибка на тесте")
    ax.plot(P, tr * 100, color=BLUE, lw=2.2, marker="o", ms=4, label="ошибка на обучении")
    ax.axvline(N_TRAIN, color=GOLD, lw=1.4, ls=(0, (3, 3)))
    ax.set_xscale("log")
    ax.text(N_TRAIN * 1.25, 6, "порог\nинтерполяции\n$p=n=100$", color=GOLD, fontsize=9.5)
    ax.annotate(f"пик {te[i_peak]*100:.0f}%", xy=(P[i_peak], te[i_peak] * 100),
                xytext=(P[i_peak] * 2.4, te[i_peak] * 100 - 3), color=RED, fontsize=10.5,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.annotate(f"первый минимум {left[i_left]*100:.0f}%", xy=(P[i_left], left[i_left] * 100),
                xytext=(P[i_left] * 0.34, left[i_left] * 100 + 12), color=GREEN, fontsize=10.5,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))
    ax.annotate(f"второй спуск {te[i_right]*100:.0f}%", xy=(P[i_right], te[i_right] * 100),
                xytext=(P[i_right] * 0.16, te[i_right] * 100 - 14), color=VIOLET, fontsize=10.5,
                arrowprops=dict(arrowstyle="->", color=VIOLET, lw=1.2))
    ax.set_xlabel("число случайных признаков $p$ (логарифмическая ось)")
    ax.set_ylabel("доля ошибок, %")
    ax.set_title(f"Двойной спуск на реальных digits: $n={N_TRAIN}$, 20% меток испорчено")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "double_descent.png")


# ---------------------------------------------- fig 59.3: spectrum explains the peak
def fig_spectrum():
    P = np.array(PGRID)
    nrm, smin, cond = CACHE["nrm"], CACHE["smin"], CACHE["cond"]
    i100 = int(np.where(P == 100)[0][0])
    i30 = int(np.where(P == 30)[0][0])
    i2000 = len(P) - 1
    print("norm :", np.round(nrm, 2))
    print("smin :", np.round(smin, 5))
    print("cond :", np.round(cond, 1))
    assert smin[i100] < smin[i30] / 20 and smin[i100] < smin[i2000] / 20
    assert nrm[i100] > nrm[i30] * 3 and nrm[i100] > nrm[i2000] * 3
    FACTS.update(
        sp_smin30=float(smin[i30]), sp_smin100=float(smin[i100]), sp_smin2000=float(smin[i2000]),
        sp_norm30=float(nrm[i30]), sp_norm100=float(nrm[i100]), sp_norm2000=float(nrm[i2000]),
        sp_cond100=float(cond[i100]), sp_cond2000=float(cond[i2000]),
        sp_norm_drop=float(nrm[i100] / nrm[i2000]),
    )
    Xtr, ytr, Xte, yte = digits_split(N_TRAIN, seed=59, noise=0.20)
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2))
    for ax, p, c in zip(axes[:2], [30, 100], [GREEN, RED]):
        *_ , sv = rff_error(Xtr, ytr, Xte, yte, p, 0, 0.0, True)
        ax.semilogy(np.arange(1, len(sv) + 1), sv, color=c, lw=2.0, marker="o", ms=3)
        ax.set_title(f"спектр $Z$ при $p={p}$", fontsize=12)
        ax.set_xlabel("номер сингулярного числа"); ax.set_ylabel(r"$\sigma_j$")
        ax.set_ylim(1e-4, 1e2)
    ax = axes[2]
    ax.plot(P, nrm, color=VIOLET, lw=2.4, marker="o", ms=4)
    ax.axvline(N_TRAIN, color=GOLD, lw=1.4, ls=(0, (3, 3)))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$p$"); ax.set_ylabel(r"норма решения $\|W\|_F$")
    ax.set_title("Норма минимальной длины взлетает у порога", fontsize=12)
    for a in axes:
        a.grid(True, color=GRID, lw=0.4, alpha=0.4); a.set_axisbelow(True)
    fig.suptitle("Пик — не мистика: у порога появляется почти нулевое сингулярное число", y=1.03, fontsize=14)
    fig.tight_layout()
    save(fig, OUT / "spectrum.png")


# ---------------------------------------------- fig 59.4: ridge kills the peak
def fig_ridge():
    P = np.array(PGRID)
    te0 = CACHE["te"]
    curves = {0.0: te0}
    for lam in (0.01, 1.0):
        _, te, *_ = sweep(noise=0.20, lam=lam)
        curves[lam] = te
    i100 = int(np.where(P == 100)[0][0])
    for lam, te in curves.items():
        print("lam", lam, np.round(te * 100, 1))
    peak0 = te0[i100]
    peak1 = curves[1.0][i100]
    assert peak1 < peak0 - 0.10, "ridge must flatten the peak"
    best_ridge = float(curves[1.0].min())
    assert best_ridge <= te0.min() + 0.02
    FACTS.update(
        rg_peak0=float(peak0), rg_peak_small=float(curves[0.01][i100]), rg_peak_big=float(peak1),
        rg_best=best_ridge, rg_best_p=int(P[int(np.argmin(curves[1.0]))]),
        rg_ridgeless_best=float(te0.min()),
    )
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    for (lam, te), c in zip(curves.items(), [RED, GOLD, BLUE]):
        lbl = "без регуляризации" if lam == 0 else f"ridge $\\lambda={lam}$"
        ax.plot(P, te * 100, color=c, lw=2.4, marker="o", ms=4, label=lbl)
    ax.axvline(N_TRAIN, color=GOLD, lw=1.2, ls=(0, (3, 3)))
    ax.set_xscale("log")
    ax.set_xlabel("число случайных признаков $p$"); ax.set_ylabel("ошибка на тесте, %")
    ax.set_title("Ridge срезает пик: катастрофа у порога была узкой и лечится")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "ridge.png")


# ---------------------------------------------- fig 59.5: sample-wise + label noise
def fig_samplewise_and_noise():
    # sample-wise: p fixed, n varies
    p_fixed = 300
    ns = [40, 80, 150, 220, 270, 300, 330, 400, 600, 900]
    tew = np.zeros(len(ns))
    for s in range(6):
        for i, n in enumerate(ns):
            Xtr, ytr, Xte, yte = digits_split(n, n_test=800, seed=59 + s, noise=0.20)
            _, te = rff_error(Xtr, ytr, Xte, yte, p_fixed, s)
            tew[i] += te
    tew /= 6
    i_p = ns.index(p_fixed)
    print("sample-wise:", ns, np.round(tew * 100, 1))
    assert tew[i_p] == tew.max(), "more data must hurt exactly at n = p"
    assert tew[-1] < tew[ns.index(150)], "with enough data the curve recovers"
    FACTS.update(
        sw_p=p_fixed, sw_n_peak=p_fixed, sw_peak=float(tew[i_p]),
        sw_n150=float(tew[ns.index(150)]), sw_n900=float(tew[-1]),
        sw_n80=float(tew[ns.index(80)]),
    )
    # label-noise dependence of the peak
    P = np.array(PGRID)
    i100 = int(np.where(P == 100)[0][0])
    levels = [0.0, 0.10, 0.20, 0.40]
    peaks, mins, curves = [], [], {}
    for lv in levels:
        _, te, *_ = sweep(noise=lv)
        curves[lv] = te
        peaks.append(float(te[i100])); mins.append(float(te[P < 70].min()))
    finals = [float(curves[lv][-1]) for lv in levels]
    gains = np.array(mins) - np.array(finals)
    print("noise levels", levels)
    print("peak       ", np.round(np.array(peaks) * 100, 1))
    print("first min  ", np.round(np.array(mins) * 100, 1))
    print("final p2000", np.round(np.array(finals) * 100, 1))
    print("gain       ", np.round(gains * 100, 1))
    assert np.all(np.diff(np.array(mins)) > 0), "classical optimum must degrade with label noise"
    assert min(peaks) > 0.75, "the peak reaches chance level at every noise level"
    assert np.all(np.diff(np.array(finals)) > 0), "the interpolating model degrades with label noise too"
    assert np.all(gains > 0.04), "over-parameterisation beats the classical optimum at every noise level"
    assert gains[0] > gains[3], "but the benefit shrinks as labels get dirtier"
    FACTS.update(
        ln_peak0=peaks[0], ln_peak40=peaks[3],
        ln_min0=mins[0], ln_min10=mins[1], ln_min20=mins[2], ln_min40=mins[3],
        ln_final0=finals[0], ln_final10=finals[1], ln_final20=finals[2], ln_final40=finals[3],
        ln_gain0=float(gains[0]), ln_gain40=float(gains[3]),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    ax = axes[0]
    ax.plot(ns, tew * 100, color=RED, lw=2.4, marker="o", ms=4.5)
    ax.axvline(p_fixed, color=GOLD, lw=1.4, ls=(0, (3, 3)))
    ax.text(p_fixed * 1.05, tew.max() * 100 - 2, "$n=p=300$", color=GOLD, fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel("размер обучающей выборки $n$"); ax.set_ylabel("ошибка на тесте, %")
    ax.set_title("Больше данных — временно хуже", fontsize=13)
    ax = axes[1]
    for lv, c in zip(levels, [BLUE, GREEN, GOLD, RED]):
        ax.plot(P, curves[lv] * 100, color=c, lw=2.2, marker="o", ms=3.5,
                label=f"шум меток {int(lv*100)}%")
    ax.axvline(N_TRAIN, color=GOLD, lw=1.2, ls=(0, (3, 3)))
    ax.set_xscale("log")
    ax.set_xlabel("число признаков $p$"); ax.set_ylabel("ошибка на тесте, %")
    ax.set_title("Шум меток портит всю кривую, но пик стоит у порога", fontsize=13)
    ax.legend(frameon=False, fontsize=9.5)
    for a in axes:
        a.grid(True, color=GRID, lw=0.4, alpha=0.4); a.set_axisbelow(True)
    fig.suptitle("Двойной спуск виден и по числу объектов, и при любой чистоте разметки", y=1.03, fontsize=14)
    fig.tight_layout()
    save(fig, OUT / "samplewise_noise.png")


# ---------------------------------------------- sidenote: minimum norm geometry
def side_minnorm():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
    ax = axes[0]
    t = np.linspace(-0.6, 1.6, 50)
    ax.plot(t, 1 - t, color=BLUE, lw=2.2)
    th = np.linspace(0, 2 * np.pi, 200)
    r = np.sqrt(0.5)
    ax.plot(r * np.cos(th), r * np.sin(th), color=GRID, lw=1.4)
    ax.plot([0, 0.5], [0, 0.5], color=GREEN, lw=2.0)
    ax.scatter([0.5], [0.5], s=60, color=GREEN, zorder=5)
    ax.scatter([1, 0], [0, 1], s=35, color=MUTED, zorder=4)
    ax.text(0.56, 0.5, "(0,5; 0,5)", color=GREEN, fontsize=9)
    ax.set_title("исходные координаты", fontsize=11)
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(-0.6, 1.6)
    ax = axes[1]
    # after rescaling the second feature by 10: w1 + 10 v2 = 1
    ax.plot(t, (1 - t) / 10, color=BLUE, lw=2.2)
    r2 = 1 / np.sqrt(101)
    ax.plot(r2 * np.cos(th), r2 * np.sin(th), color=GRID, lw=1.4)
    ax.plot([0, 1 / 101], [0, 10 / 101], color=RED, lw=2.0)
    ax.scatter([1 / 101], [10 / 101], s=60, color=RED, zorder=5)
    ax.text(0.12, 0.1, "(0,010; 0,099)", color=RED, fontsize=9)
    ax.set_title("второй признак умножен на 10", fontsize=11)
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(-0.6, 1.6)
    for a in axes:
        a.axhline(0, color=LINE, lw=0.8); a.axvline(0, color=LINE, lw=0.8)
        a.set_aspect("equal"); a.grid(True, color=GRID, lw=0.4, alpha=0.4); a.set_axisbelow(True)
        a.set_xticks([0, 1]); a.set_yticks([0, 1])
    fig.tight_layout()
    save(fig, SIDE / "minnorm.png")
    # numbers used in the text
    w = np.array([1 / 101, 10 / 101])
    print("rescaled min-norm:", np.round(w, 4), "prediction on (1,0):", round(w[0], 4))
    FACTS.update(mn_w1=float(w[0]), mn_w2=float(w[1]), mn_norm=float(np.linalg.norm(w)),
                 mn_norm_plain=float(np.linalg.norm([0.5, 0.5])))


# ---------------------------------------------- sidenote: epoch-wise double descent
def side_epochwise():
    Xtr, ytr, Xte, yte = digits_split(N_TRAIN, seed=59, noise=0.30)
    rng = np.random.default_rng(7)
    p = 400
    d = Xtr.shape[1]
    Wr = rng.normal(0, 1 / np.sqrt(d), size=(d, p)); b = rng.uniform(-0.3, 0.3, p)
    Ztr = np.maximum(0, Xtr @ Wr + b); Zte = np.maximum(0, Xte @ Wr + b)
    Y = np.eye(10)[ytr]
    Wt = np.zeros((p, 10))
    lr = 1.0 / (np.linalg.norm(Ztr, 2) ** 2)
    iters, te_hist, tr_hist = [], [], []
    for it in range(1, 60001):
        G = Ztr.T @ (Ztr @ Wt - Y)
        Wt -= lr * G
        if it in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 40000, 60000):
            iters.append(it)
            te_hist.append(float(np.mean(np.argmax(Zte @ Wt, 1) != yte)))
            tr_hist.append(float(np.mean(np.argmax(Ztr @ Wt, 1) != ytr)))
    te_hist = np.array(te_hist); tr_hist = np.array(tr_hist)
    print("epoch-wise iters:", iters)
    print("epoch-wise test :", np.round(te_hist * 100, 1))
    i_best_early = int(np.argmin(te_hist[:8]))
    i_worst = int(np.argmax(te_hist[i_best_early:]) + i_best_early)
    assert te_hist[i_worst] > te_hist[i_best_early] + 0.01, "test error must degrade after the early minimum"
    assert tr_hist[-1] < 0.02
    FACTS.update(ew_early_it=int(iters[i_best_early]), ew_early=float(te_hist[i_best_early]),
                 ew_worst_it=int(iters[i_worst]), ew_worst=float(te_hist[i_worst]),
                 ew_final=float(te_hist[-1]), ew_train_final=float(tr_hist[-1]))
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.semilogx(iters, te_hist * 100, color=RED, lw=2.2, marker="o", ms=3.5, label="тест")
    ax.semilogx(iters, tr_hist * 100, color=BLUE, lw=2.0, marker="o", ms=3.5, label="обучение")
    ax.set_xlabel("шаг градиентного спуска", fontsize=9)
    ax.set_ylabel("ошибка, %", fontsize=9)
    ax.set_title("вдоль времени обучения", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, SIDE / "epochwise.png")


# ---------------------------------------------- sidenote: memorising random labels
def side_random_labels():
    X, y = load_digits(return_X_y=True)
    rng = np.random.default_rng(3)
    perm = rng.permutation(len(y))
    X = X[perm] / 16.0; y = y[perm]
    n = 200
    Xtr = X[:n] - X[:n].mean(0)
    Xte = X[n:n + 800] - X[:n].mean(0)
    yte = y[n:n + 800]
    y_true = y[:n]
    y_rand = rng.integers(0, 10, n)
    res = {}
    for name, yy in (("настоящие метки", y_true), ("случайные метки", y_rand)):
        trs, tes = [], []
        for p in PGRID:
            tr, te = rff_error(Xtr, yy, Xte, yte, p, 4)
            trs.append(tr); tes.append(te)
        res[name] = (np.array(trs), np.array(tes))
    tr_rand = res["случайные метки"][0]
    te_rand = res["случайные метки"][1]
    te_true = res["настоящие метки"][1]
    print("random labels train:", np.round(tr_rand * 100, 1))
    print("random labels test :", np.round(te_rand * 100, 1))
    assert tr_rand[-1] < 1e-9, "the model must memorise random labels exactly"
    assert te_rand[-1] > 0.80, "and generalise no better than chance"
    assert te_true[-1] < 0.10
    FACTS.update(rl_n=n, rl_train_rand=float(tr_rand[-1]), rl_test_rand=float(te_rand[-1]),
                 rl_test_true=float(te_true[-1]), rl_p=int(PGRID[-1]))
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.semilogx(PGRID, res["настоящие метки"][1] * 100, color=BLUE, lw=2.2, label="тест, настоящие")
    ax.semilogx(PGRID, te_rand * 100, color=RED, lw=2.2, label="тест, случайные")
    ax.semilogx(PGRID, tr_rand * 100, color=RED, lw=1.6, ls=(0, (3, 2)), label="обучение, случайные")
    ax.axhline(90, color=MUTED, lw=1.0, ls=(0, (2, 2)))
    ax.text(8, 92, "угадывание", color=MUTED, fontsize=8)
    ax.set_xlabel("$p$", fontsize=9); ax.set_ylabel("ошибка, %", fontsize=9)
    ax.set_title("память ≠ понимание", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, SIDE / "random_labels.png")


# ---------------------------------------------- sidenote: ridge filter factors
def side_filter():
    sig = np.logspace(-3, 1, 300)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.loglog(sig, 1 / sig, color=RED, lw=2.2, label=r"$1/\sigma$ (МНК)")
    for lam, c in ((0.01, GOLD), (1.0, BLUE)):
        ax.loglog(sig, sig / (sig ** 2 + lam), color=c, lw=2.0, label=fr"$\lambda={lam}$")
    ax.set_xlabel(r"$\sigma_j$", fontsize=9); ax.set_ylabel("усиление", fontsize=9)
    ax.set_title("ridge ограничивает усиление", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, SIDE / "filter.png")
    s = 0.01; lam = 0.01
    print("gain at sigma=0.01: ols", 1 / s, "ridge", s / (s ** 2 + lam))
    FACTS.update(ft_ols_gain=1 / s, ft_ridge_gain=s / (s ** 2 + lam),
                 ft_ratio=(1 / s) / (s / (s ** 2 + lam)))


def main():
    fig_bias_variance()
    fig_double_descent()
    fig_spectrum()
    fig_ridge()
    fig_samplewise_and_noise()
    side_minnorm()
    side_epochwise()
    side_random_labels()
    side_filter()
    out = ROOT / "scripts" / "data" / "lesson59_facts.json"
    out.write_text(json.dumps({k: round(v, 6) for k, v in FACTS.items()},
                              ensure_ascii=False, indent=2), encoding="utf8")
    print("\n=== FACTS ===")
    for k, v in FACTS.items():
        print(f"{k:22s} {v}")
    print("OK")


if __name__ == "__main__":
    main()
