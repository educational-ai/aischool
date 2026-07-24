"""Deterministic figures for lesson 56: the uncertainty map.

Three sources of doubt (aleatoric / epistemic / distributional) on the REAL
breast-cancer dataset and the REAL bike-sharing hourly file: variance
decomposition from a bootstrap ensemble, reliability diagram before and after
temperature scaling, risk-coverage curve for selective prediction, an
out-of-support split where every marginal looks normal, and split-conformal
coverage that holds under exchangeability and breaks under shift.
Every number quoted in the lesson is computed and asserted here.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
BIKE = ROOT / "scripts" / "data" / "bike-sharing-hour.csv"
OUT = ROOT / "public" / "figures" / "lessons" / "56"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "56"

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


def fact(name, value):
    FACTS[name] = float(value)
    return value


# ---------------------------------------------------------------- data
DATA = load_breast_cancer()
X_ALL = DATA.data
Y_ALL = (DATA.target == 0).astype(int)          # 1 = злокачественная
N_ALL = len(Y_ALL)
BASE_RATE = Y_ALL.mean()
fact("n_all", N_ALL)
fact("n_features", X_ALL.shape[1])
fact("base_rate", BASE_RATE)
assert N_ALL == 569 and X_ALL.shape[1] == 30
assert abs(BASE_RATE - 0.3726) < 0.001

RNG = np.random.default_rng(56)
perm = RNG.permutation(N_ALL)
idx_tr = perm[:60]
idx_cal = perm[60:220]
idx_te = perm[220:]
fact("n_train", len(idx_tr)); fact("n_cal", len(idx_cal)); fact("n_test", len(idx_te))

MU = X_ALL[idx_tr].mean(0)
SD = X_ALL[idx_tr].std(0) + 1e-9


def z(X):
    return (X - MU) / SD


def fit(Xtr, ytr, C=1e6):
    m = LogisticRegression(C=C, max_iter=20000)
    m.fit(Xtr, ytr)
    return m


MODEL = fit(z(X_ALL[idx_tr]), Y_ALL[idx_tr])
p_cal = MODEL.predict_proba(z(X_ALL[idx_cal]))[:, 1]
p_te = MODEL.predict_proba(z(X_ALL[idx_te]))[:, 1]
y_cal = Y_ALL[idx_cal]
y_te = Y_ALL[idx_te]

ACC = fact("acc", (np.round(p_te) == y_te).mean())
AUC = fact("auc", roc_auc_score(y_te, p_te))
BRIER = fact("brier", brier_score_loss(y_te, p_te))
assert ACC > 0.9 and AUC > 0.95


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1]) if i else (p <= edges[1])
        if m.sum() == 0:
            continue
        tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return tot


# temperature scaling on the calibration split
def nll(T, p, y):
    lo = np.log(np.clip(p, 1e-12, 1 - 1e-12) / np.clip(1 - p, 1e-12, 1))
    q = 1 / (1 + np.exp(-lo / T))
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q)).mean()


Ts = np.linspace(0.3, 8.0, 3000)
T_STAR = fact("T", Ts[int(np.argmin([nll(t, p_cal, y_cal) for t in Ts]))])


def temper(p, T):
    lo = np.log(np.clip(p, 1e-12, 1 - 1e-12) / np.clip(1 - p, 1e-12, 1))
    return 1 / (1 + np.exp(-lo / T))


p_te_T = temper(p_te, T_STAR)
ECE0 = fact("ece_raw", ece(p_te, y_te))
ECE1 = fact("ece_cal", ece(p_te_T, y_te))
BRIER1 = fact("brier_cal", brier_score_loss(y_te, p_te_T))
AUC1 = fact("auc_cal", roc_auc_score(y_te, p_te_T))
assert T_STAR > 1.5, T_STAR
assert ECE1 < ECE0 * 0.8, (ECE0, ECE1)
assert abs(AUC1 - AUC) < 1e-9        # монотонное преобразование не меняет ранжирование
print(f"acc={ACC:.4f} auc={AUC:.4f} brier={BRIER:.4f} T={T_STAR:.2f} "
      f"ECE {ECE0:.4f} -> {ECE1:.4f}, brier -> {BRIER1:.4f}")

# один конкретный пациент и верхняя корзина прогнозов
HI = p_te > 0.9
fact("hi_n", int(HI.sum()))
fact("hi_meanp", p_te[HI].mean())
fact("hi_freq", y_te[HI].mean())
HI2 = p_te_T > 0.9
fact("hi_n_cal", int(HI2.sum()))
fact("hi_freq_cal", y_te[HI2].mean())
I_PAT = int(np.argmin(np.abs(p_te - 0.07)))
P_PAT = fact("patient_p", p_te[I_PAT])
P_PAT_T = fact("patient_p_cal", p_te_T[I_PAT])
fact("patient_y", y_te[I_PAT])
assert P_PAT < 0.1 < P_PAT_T
assert abs(p_te[HI].mean() - 0.996) < 0.002 and abs(y_te[HI].mean() - 0.917) < 0.002
print(f"patient: p={P_PAT:.4f} -> {P_PAT_T:.4f} (y={y_te[I_PAT]}); "
      f"bin>0.9: n={int(HI.sum())} meanp={p_te[HI].mean():.4f} freq={y_te[HI].mean():.4f}; "
      f"after T: n={int(HI2.sum())} freq={y_te[HI2].mean():.4f}")


# ------------------------------------------------- fig 1: three causes of doubt
def fig_three_causes():
    f1, f2 = 0, 1                       # mean radius, mean texture
    x1, x2 = X_ALL[:, f1], X_ALL[:, f2]
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.9))

    # (a) aleatoric: dense but overlapping classes in one informative feature
    ax = axes[0]
    fsm = 4  # mean smoothness
    v = X_ALL[:, fsm]
    for cls, col, lab in [(0, BLUE, "доброкач."), (1, RED, "злокач.")]:
        ax.hist(v[Y_ALL == cls], bins=26, color=col, alpha=0.55, label=lab)
    ov = fact("overlap_smooth", roc_auc_score(Y_ALL, v))
    ax.set_title("aleatoric:\nклассы перекрываются", fontsize=11.5)
    ax.set_xlabel("гладкость (mean smoothness)", fontsize=9)
    ax.legend(frameon=False, fontsize=8.5)
    ax.text(0.03, 0.72, f"данных много (569),\nно AUC этого признака {ov:.2f}",
            transform=ax.transAxes, fontsize=8.5, color=MUTED)

    # (b) epistemic: few points, many lines agree
    ax = axes[1]
    sub = RNG.choice(N_ALL, 12, replace=False)
    ax.scatter(x1[sub], x2[sub], s=42, c=[RED if t else BLUE for t in Y_ALL[sub]], zorder=5)
    xs = np.linspace(x1.min(), x1.max(), 50)
    slopes = []
    for b in range(14):
        bs = RNG.choice(sub, len(sub), replace=True)
        if len(np.unique(Y_ALL[bs])) < 2:
            continue
        m = LogisticRegression(C=1.0, max_iter=5000).fit(
            np.c_[x1[bs], x2[bs]], Y_ALL[bs])
        w = m.coef_[0]
        if abs(w[1]) < 1e-6:
            continue
        ax.plot(xs, -(m.intercept_[0] + w[0] * xs) / w[1], color=MUTED, lw=1.0, alpha=0.75)
        slopes.append(-w[0] / w[1])
    ax.set_ylim(x2.min() - 1, x2.max() + 1)
    # в уроке названы «12 точек — 14 границ»: проверяем, что все 14 действительно нарисованы
    fact("epi_lines", len(slopes))
    fact("epi_points", len(sub))
    assert len(slopes) == 14 and len(sub) == 12, (len(slopes), len(sub))
    ax.set_title("epistemic:\nмного границ согласны с данными", fontsize=11.5)
    ax.set_xlabel("радиус", fontsize=9); ax.set_ylabel("текстура", fontsize=9)
    ax.text(0.03, 0.05, "12 точек — 14 разных границ", transform=ax.transAxes,
            fontsize=8.5, color=MUTED)

    # (c) distributional: joint region never seen
    ax = axes[2]
    band = (x2 > 12 + 0.9 * (x1 - 8)) & (x2 < 12 + 0.9 * (x1 - 8) + 9)
    ax.scatter(x1[band], x2[band], s=14, color=LINE, label="обучающая область")
    ax.scatter(x1[~band], x2[~band], s=18, color=GOLD, label="новые сочетания")
    ax.set_title("distributional:\nсочетание вне обучения", fontsize=11.5)
    ax.set_xlabel("радиус", fontsize=9); ax.set_ylabel("текстура", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fact("band_share", band.mean())

    for a in axes:
        a.grid(True, color=GRID, lw=0.4, alpha=0.4); a.set_axisbelow(True)
    fig.suptitle("Одна ширина — три разных диагноза", y=1.04, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "three_causes.png")
    print(f"three_causes: AUC(smoothness)={ov:.3f}, band share={band.mean():.3f}")


# ------------------------------- fig 2: variance split on REAL bike sharing
def bike():
    t, c = [], []
    with open(BIKE) as f:
        for row in csv.DictReader(f):
            t.append(float(row["temp"])); c.append(float(row["cnt"]))
    return np.array(t), np.array(c)


def fig_variance_split():
    temp, cnt = bike()
    fact("bike_rows", len(cnt))
    assert len(cnt) == 17379
    rng = np.random.default_rng(560)
    # редкие холодные часы: намеренно оставляем мало точек при temp<0.25
    keep = np.where(temp >= 0.25, True, rng.random(len(temp)) < 0.02)
    tt, cc = temp[keep], cnt[keep]
    n_cold = fact("bike_cold", int((tt < 0.25).sum()))
    n_warm = fact("bike_warm", int((tt >= 0.25).sum()))

    def design(x):
        return np.c_[np.ones_like(x), x, x ** 2, x ** 3]

    grid = np.linspace(0.05, 1.0, 120)
    G = design(grid)
    B = 200
    preds = np.zeros((B, len(grid)))
    for b in range(B):
        s = rng.integers(0, len(tt), len(tt))
        w, *_ = np.linalg.lstsq(design(tt[s]), cc[s], rcond=None)
        preds[b] = G @ w
    w0, *_ = np.linalg.lstsq(design(tt), cc, rcond=None)
    mean = G @ w0
    epi_sd = preds.std(0)
    resid = cc - design(tt) @ w0
    ale_sd = fact("bike_ale_sd", resid.std())
    # локальный шум: sd остатков в скользящих корзинах по температуре
    edges = np.linspace(0.05, 1.0, 13)
    ale_x, ale_y = [], []
    for i in range(12):
        m = (tt >= edges[i]) & (tt < edges[i + 1])
        if m.sum() >= 15:
            ale_x.append(0.5 * (edges[i] + edges[i + 1])); ale_y.append(resid[m].std())
    ale_x, ale_y = np.array(ale_x), np.array(ale_y)
    ale_cold = fact("bike_ale_cold", ale_y[np.argmin(abs(ale_x - 0.10))])
    ale_warm = fact("bike_ale_warm", ale_y[np.argmin(abs(ale_x - 0.60))])
    epi_cold = fact("bike_epi_cold", epi_sd[np.argmin(abs(grid - 0.10))])
    epi_warm = fact("bike_epi_warm", epi_sd[np.argmin(abs(grid - 0.60))])
    ratio = fact("bike_epi_ratio", epi_cold / epi_warm)
    assert ratio > 3, ratio
    # «данных в 465 раз больше» — отношение тёплых часов к холодным
    n_ratio = fact("bike_n_ratio", n_warm / n_cold)
    assert abs(n_ratio - 465) < 1, n_ratio
    tot_cold = fact("bike_tot_cold", np.hypot(ale_cold, epi_cold))
    tot_warm = fact("bike_tot_warm", np.hypot(ale_warm, epi_warm))
    fact("bike_epi_share_cold", epi_cold ** 2 / (epi_cold ** 2 + ale_cold ** 2))
    fact("bike_epi_share_warm", epi_warm ** 2 / (epi_warm ** 2 + ale_warm ** 2))

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.8, 5.8), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.scatter(tt, cc, s=3, color=LINE, alpha=0.35)
    ax.fill_between(grid, mean - np.hypot(ale_sd, epi_sd), mean + np.hypot(ale_sd, epi_sd),
                    color=BLUE, alpha=0.12, label="полный разброс (шум + модель)")
    ax.fill_between(grid, mean - epi_sd, mean + epi_sd, color=RED, alpha=0.35,
                    label="epistemic: разброс кривых")
    ax.plot(grid, mean, color=INK, lw=2.0, label="кубический базис по всем данным")
    ax.axvspan(0.05, 0.25, color=WASH, alpha=0.8, zorder=0)
    ax.text(0.075, cc.max() * 0.60, f"мало данных:\n{int(n_cold)} часов", fontsize=9, color=GOLD)
    ax.text(0.45, cc.max() * 0.06, f"данных много: {int(n_warm)} часов", fontsize=9, color=MUTED)
    ax.set_ylabel("поездок в час"); ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Шум остаётся, незнание модели тает там, где есть данные")
    ax2.plot(grid, epi_sd, color=RED, lw=2.0, label="epistemic sd (разброс кривых)")
    ax2.plot(ale_x, ale_y, color=BLUE, lw=1.8, ls=(0, (5, 3)), label="aleatoric sd (остатки в корзине)")
    ax2.set_yscale("log")
    ax2.set_xlabel("нормированная температура"); ax2.set_ylabel("sd (лог)")
    ax2.legend(frameon=False, fontsize=8.5, ncol=2, loc="lower center")
    for a in (ax, ax2):
        a.grid(True, color=GRID, lw=0.4, alpha=0.4); a.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "variance_split.png")
    print(f"variance: ale_sd={ale_sd:.1f} (cold {ale_cold:.1f}, warm {ale_warm:.1f}), "
          f"epi cold={epi_cold:.1f} warm={epi_warm:.1f} "
          f"ratio={ratio:.1f}, total cold={tot_cold:.1f} warm={tot_warm:.1f}")


# --------------------------------------------- fig 3: reliability + temperature
def reliability_points(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    xs, ys, ns = [], [], []
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1]) if i else (p <= edges[1])
        if m.sum() == 0:
            continue
        xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def fig_reliability():
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.6, 6.4), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.2, ls=(0, (4, 3)), label="идеальная калибровка")
    for p, col, lab in [(p_te, RED, f"как есть, ECE={ECE0:.3f}"),
                        (p_te_T, GREEN, f"после T={T_STAR:.2f}, ECE={ECE1:.3f}")]:
        xs, ys, ns = reliability_points(p, y_te)
        big = ns >= 10
        ax.plot(xs[big], ys[big], "-o", color=col, lw=2.0, markersize=6, label=lab)
        ax.scatter(xs[~big], ys[~big], s=26, color=col, alpha=0.35, marker="o")
        ax.scatter(xs[big], ys[big], s=6 + 2.0 * ns[big], color=col, alpha=0.25)
    ax.text(0.34, 0.93, "полупрозрачные точки — bins,\nв которых меньше 10 объектов",
            fontsize=8.5, color=MUTED)
    ax.set_ylabel("наблюдаемая доля класса")
    ax.set_title("Reliability diagram: где числа расходятся с частотами")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.text(0.55, 0.12, "ниже диагонали —\nмодель самоуверенна", fontsize=9, color=RED)
    ax2.hist(p_te, bins=np.linspace(0, 1, 11), color=LINE, edgecolor=PAPER)
    ax2.set_xlabel("предсказанная вероятность"); ax2.set_ylabel("объектов")
    hi = fact("share_extreme", ((p_te > 0.99) | (p_te < 0.01)).mean())
    ax2.text(0.30, ax2.get_ylim()[1] * 0.6,
             f"{hi*100:.0f}% прогнозов вжаты в края", fontsize=9, color=MUTED)
    for a in (ax, ax2):
        a.grid(True, color=GRID, lw=0.4, alpha=0.4); a.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "reliability.png")
    print(f"reliability: extreme share={hi:.3f}")


# ------------------------------------------------- fig 4: risk-coverage curve
def risk_coverage(score, y, p):
    order = np.argsort(-score)
    err = (np.round(p) != y).astype(float)[order]
    cov = np.arange(1, len(y) + 1) / len(y)
    risk = np.cumsum(err) / np.arange(1, len(y) + 1)
    return cov, risk


def fig_risk_coverage():
    conf = np.maximum(p_te, 1 - p_te)
    cov, risk = risk_coverage(conf, y_te, p_te)
    rng = np.random.default_rng(5600)
    rnd = np.zeros_like(risk)
    for _ in range(300):
        c2, r2 = risk_coverage(rng.random(len(y_te)), y_te, p_te)
        rnd += r2
    rnd /= 300

    def at(c):
        i = int(np.argmin(abs(cov - c)))
        return risk[i]

    r100 = fact("risk100", at(1.0)); r80 = fact("risk80", at(0.8))
    r60 = fact("risk60", at(0.6)); r50 = fact("risk50", at(0.5))
    rnd80 = fact("rnd80", rnd[int(np.argmin(abs(cov - 0.8)))])
    assert r80 < r100 and r60 <= r80
    # coverage at target selective risk 2%
    ok = np.where(risk <= 0.02)[0]
    cov_at_2 = fact("cov_at_2", cov[ok.max()])

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(cov * 100, risk * 100, color=BLUE, lw=2.4, label="отказ по уверенности модели")
    ax.plot(cov * 100, rnd * 100, color=MUTED, lw=1.8, ls=(0, (5, 3)), label="случайный отказ")
    ax.scatter([80], [r80 * 100], s=70, color=RED, zorder=6)
    ax.annotate(f"80% автоматически,\nошибка {r80*100:.1f}%", (80, r80 * 100),
                textcoords="offset points", xytext=(-16, 34), fontsize=9.5, color=RED)
    ax.scatter([100], [r100 * 100], s=60, color=INK, zorder=6)
    ax.annotate(f"без отказа: {r100*100:.1f}%", (100, r100 * 100),
                textcoords="offset points", xytext=(-120, 8), fontsize=9.5, color=INK)
    ax.set_xlabel("coverage: доля решённого автоматически, %")
    ax.set_ylabel("selective risk: ошибка на принятых, %")
    ax.set_title("Отказ работает, только если уверенность информативна")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "risk_coverage.png")
    print(f"risk-coverage: 100%={r100:.4f} 80%={r80:.4f} 60%={r60:.4f} 50%={r50:.4f} "
          f"random80={rnd80:.4f}, coverage at 2% risk={cov_at_2:.3f}")


# ------------------------------------------- fig 5: out-of-support combination
def fig_ood():
    f1, f2 = 0, 1
    x1, x2 = X_ALL[:, f1], X_ALL[:, f2]
    band = (x2 > 12 + 0.9 * (x1 - 8)) & (x2 < 12 + 0.9 * (x1 - 8) + 9)
    tr, new = np.where(band)[0], np.where(~band)[0]
    # выберем «нового пациента»: обычные значения по каждой оси, редкое сочетание
    lo1, hi1 = np.percentile(x1[tr], [10, 90]); lo2, hi2 = np.percentile(x2[tr], [10, 90])
    inside = new[(x1[new] > lo1) & (x1[new] < hi1) & (x2[new] > lo2) & (x2[new] < hi2)]
    S = np.cov(np.c_[x1[tr], x2[tr]].T)
    Si = np.linalg.inv(S)
    m = np.array([x1[tr].mean(), x2[tr].mean()])

    def maha(px, py):
        d = np.array([px, py]) - m
        return float(np.sqrt(d @ Si @ d))

    md = np.array([maha(x1[i], x2[i]) for i in inside])
    star = inside[int(np.argmax(md))]
    d_star = fact("maha_star", md.max())
    d_tr = np.array([maha(x1[i], x2[i]) for i in tr])
    d_tr_med = fact("maha_train_median", np.median(d_tr))
    pct = fact("maha_pct", (d_tr < d_star).mean())
    p1 = fact("pct_axis1", (x1[tr] < x1[star]).mean())
    p2 = fact("pct_axis2", (x2[tr] < x2[star]).mean())
    assert d_star > 2.5 and 0.05 < p1 < 0.95 and 0.05 < p2 < 0.95, (d_star, p1, p2)

    # domain classifier train vs "production"
    lab = np.r_[np.zeros(len(tr)), np.ones(len(new))]
    feat = np.r_[np.c_[x1[tr], x2[tr]], np.c_[x1[new], x2[new]]]
    dm = LogisticRegression(max_iter=5000).fit(feat, lab)
    dom_auc = fact("domain_auc", roc_auc_score(lab, dm.predict_proba(feat)[:, 1]))
    assert dom_auc > 0.6

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    ax.scatter(x1[tr], x2[tr], s=16, color=BLUE, alpha=0.55, label="обучающая популяция")
    ax.scatter(x1[star], x2[star], s=150, marker="*", color=RED, zorder=8, label="новый пациент")
    ax.axvline(x1[star], color=GOLD, lw=1.0, ls=(0, (3, 3)))
    ax.axhline(x2[star], color=GOLD, lw=1.0, ls=(0, (3, 3)))
    ax.set_xlabel("радиус"); ax.set_ylabel("текстура")
    ax.set_title(f"каждая координата обычна,\nсочетание — нет ($d_M={d_star:.1f}$)", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax2.hist(d_tr, bins=24, color=LINE, edgecolor=PAPER, label="обучающие: $d_M$")
    ax2.axvline(d_star, color=RED, lw=2.2, label=f"новый пациент: {d_star:.1f}")
    ax2.set_xlabel("расстояние Махаланобиса до центра обучения")
    ax2.set_ylabel("объектов")
    ax2.set_title(f"он дальше {pct*100:.0f}% обучающих точек", fontsize=12)
    ax2.legend(frameon=False, fontsize=9)
    for a in (ax, ax2):
        a.grid(True, color=GRID, lw=0.4, alpha=0.4); a.set_axisbelow(True)
    fig.suptitle("Одномерные диапазоны не видят выхода из области поддержки", y=1.02, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "ood_support.png")
    print(f"ood: percentiles {p1:.2f}/{p2:.2f}, maha={d_star:.2f} (median {d_tr_med:.2f}, "
          f"beyond {pct*100:.1f}%), domain AUC={dom_auc:.3f}")


# ------------------------------------------------ fig 6: conformal coverage
def fig_conformal():
    alpha = 0.05
    P_cal = np.c_[1 - p_cal, p_cal]
    P_te = np.c_[1 - p_te, p_te]
    s_cal = 1 - P_cal[np.arange(len(y_cal)), y_cal]
    n = len(s_cal)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    q = fact("conf_q", np.sort(s_cal)[k - 1])
    sets = P_te >= 1 - q
    cover = fact("conf_cov", sets[np.arange(len(y_te)), y_te].mean())
    size = fact("conf_size", sets.sum(1).mean())
    frac2 = fact("conf_two", (sets.sum(1) == 2).mean())
    frac0 = fact("conf_zero", (sets.sum(1) == 0).mean())
    assert 0.85 < cover < 0.99, cover

    # то же правило под сдвигом: «другой прибор» завышает первые десять признаков на 30%
    X_shift = X_ALL[idx_te].copy()
    X_shift[:, :10] *= 1.3
    p_shift = MODEL.predict_proba(z(X_shift))[:, 1]
    sets_shift = np.c_[1 - p_shift, p_shift] >= 1 - q
    cov_shift = fact("conf_cov_shift", sets_shift[np.arange(len(y_te)), y_te].mean())
    acc_shift = fact("acc_shift", (np.round(p_shift) == y_te).mean())
    assert cov_shift < 0.9 and acc_shift < ACC
    naive = fact("naive_cov", (np.maximum(p_te, 1 - p_te) >= 0.9).mean())
    naive_hit = fact("naive_hit", (np.round(p_te) == y_te)[np.maximum(p_te, 1 - p_te) >= 0.9].mean())

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.4))
    ax.hist(s_cal, bins=24, color=LINE, edgecolor=PAPER)
    ax.axvline(q, color=RED, lw=2.2)
    ax.text(q, ax.get_ylim()[1] * 0.75, f"  $q={q:.3f}$\n  ({(1-alpha)*100:.0f}-й процентиль)", fontsize=9.5, color=RED)
    ax.set_xlabel("нонконформность $s=1-\\hat p(\\text{истинный класс})$")
    ax.set_ylabel("калибровочных объектов")
    ax.set_title("Порог берут из калибровочной выборки", fontsize=12)
    labels = ["покрытие\nна test", "покрытие\nпосле сдвига", "обещано"]
    vals = [cover, cov_shift, 1 - alpha]
    cols = [GREEN, RED, MUTED]
    ax2.bar(labels, [v * 100 for v in vals], color=cols, width=0.55)
    ax2.axhline((1 - alpha) * 100, color=INK, lw=1.2, ls=(0, (4, 3)))
    for i, v in enumerate(vals):
        ax2.text(i, v * 100 + 1.5, f"{v*100:.1f}%", ha="center", fontsize=10, color=cols[i])
    ax2.set_ylim(0, 108); ax2.set_ylabel("доля покрытых, %")
    ax2.set_title(f"средний размер множества {size:.2f}", fontsize=12)
    for a in (ax, ax2):
        a.grid(True, color=GRID, lw=0.4, alpha=0.4); a.set_axisbelow(True)
    fig.suptitle("Конформный прогноз: гарантия обмена, а не веры в модель", y=1.03, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "conformal.png")
    print(f"conformal: q={q:.3f} cover={cover:.3f} size={size:.2f} two={frac2:.3f} "
          f"empty={frac0:.3f} shift={cov_shift:.3f}; naive share>=0.9: {naive:.3f} hit {naive_hit:.3f}")


# ----------------------------------------------------------- sidenote figures
def side_softmax_far():
    rng = np.random.default_rng(5601)
    x = np.r_[rng.normal(-2, 0.6, 40), rng.normal(2, 0.6, 40)]
    y = np.r_[np.zeros(40), np.ones(40)]
    m = LogisticRegression(C=0.05, max_iter=5000).fit(x[:, None], y)
    xs = np.linspace(-14, 14, 400)
    p = m.predict_proba(xs[:, None])[:, 1]
    p12 = fact("softmax_at12", m.predict_proba([[12.0]])[0, 1])
    fact("softmax_at4", m.predict_proba([[4.0]])[0, 1])
    assert p12 > 0.999
    fig, ax = plt.subplots(figsize=(4.1, 2.5))
    ax.axvspan(-4, 4, color=WASH)
    ax.plot(xs, p, color=BLUE, lw=2.0)
    ax.scatter(x, y, s=9, color=INK, alpha=0.5)
    ax.axvline(12, color=RED, lw=1.4, ls=(0, (3, 3)))
    ax.text(3.6, 0.42, f"вдали от данных\np={p12:.5f}", fontsize=8, color=RED)
    ax.set_xlabel("признак", fontsize=8.5); ax.set_ylabel("$\\hat p$", fontsize=8.5)
    ax.set_title("softmax уверен и там, где не был", fontsize=9)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "softmax_far.png")
    print(f"softmax far: p(12)={p12:.5f}")


def side_epistemic_shrink():
    rng = np.random.default_rng(5602)
    sizes = [30, 60, 120, 240, 360]
    epi, ale = [], []
    pool = np.setdiff1d(np.arange(N_ALL), idx_te)
    for nsz in sizes:
        ps = []
        for b in range(60):
            s = rng.choice(pool, nsz, replace=True)
            if len(np.unique(Y_ALL[s])) < 2:
                continue
            mu, sd = X_ALL[s].mean(0), X_ALL[s].std(0) + 1e-9
            mm = LogisticRegression(C=1.0, max_iter=5000).fit((X_ALL[s] - mu) / sd, Y_ALL[s])
            ps.append(mm.predict_proba((X_ALL[idx_te] - mu) / sd)[:, 1])
        P = np.array(ps)
        epi.append(P.var(0).mean())
        ale.append((P * (1 - P)).mean())
    e30 = fact("epi_n30", epi[0]); e360 = fact("epi_n360", epi[-1])
    a30 = fact("ale_n30", ale[0]); a360 = fact("ale_n360", ale[-1])
    fact("epi_drop", e30 / e360)
    assert e30 > 3 * e360, (e30, e360)
    fig, ax = plt.subplots(figsize=(4.1, 2.5))
    ax.plot(sizes, epi, "-o", color=RED, lw=1.8, markersize=4, label="epistemic")
    ax.plot(sizes, ale, "-o", color=BLUE, lw=1.8, markersize=4, label="aleatoric")
    ax.set_xlabel("объём обучения n", fontsize=8.5); ax.set_ylabel("средняя дисперсия", fontsize=8.5)
    ax.set_title("данные лечат только одну часть", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "epistemic_shrink.png")
    print(f"epistemic shrink: epi {e30:.4f}->{e360:.4f}, ale {a30:.4f}->{a360:.4f}")


def side_base_rate():
    """Calibration breaks when the base rate changes, ranking survives."""
    rng = np.random.default_rng(5603)
    pos = np.where(y_te == 1)[0]; neg = np.where(y_te == 0)[0]
    keep = np.r_[rng.choice(pos, max(3, len(pos) // 5), replace=False), neg]
    p2, y2 = p_te_T[keep], y_te[keep]
    br1 = fact("br_before", y_te.mean()); br2 = fact("br_after", y2.mean())
    ece1 = fact("ece_shift_before", ece(p_te_T, y_te))
    ece2 = fact("ece_shift_after", ece(p2, y2))
    auc2 = fact("auc_shift", roc_auc_score(y2, p2))
    assert ece2 > ece1
    fig, ax = plt.subplots(figsize=(4.1, 2.5))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.0, ls=(0, (4, 3)))
    for p, y, col, lab in [(p_te_T, y_te, GREEN, f"исходная база {br1*100:.0f}%"),
                           (p2, y2, RED, f"редкая болезнь {br2*100:.0f}%")]:
        xs, ys, ns = reliability_points(p, y, bins=6)
        ax.plot(xs, ys, "-o", color=col, lw=1.6, markersize=4, label=lab)
    ax.set_xlabel("$\\hat p$", fontsize=8.5); ax.set_ylabel("частота", fontsize=8.5)
    ax.set_title("сменилась база — сломалась калибровка", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "base_rate.png")
    print(f"base rate: {br1:.3f}->{br2:.3f}, ECE {ece1:.3f}->{ece2:.3f}, AUC after {auc2:.3f}")


def side_cost():
    """Expected cost as a function of the reject threshold."""
    conf = np.maximum(p_te, 1 - p_te)
    ths = np.linspace(0.5, 1.0, 200)
    C_ERR, C_MAN = 1000.0, 50.0
    cost = []
    for t in ths:
        auto = conf >= t
        errs = ((np.round(p_te) != y_te) & auto).sum()
        cost.append((errs * C_ERR + (~auto).sum() * C_MAN) / len(y_te))
    cost = np.array(cost)
    i = int(np.argmin(cost))
    t_opt = fact("cost_t", ths[i]); c_opt = fact("cost_min", cost[i])
    c_all = fact("cost_auto_all", cost[0])
    c_none = fact("cost_manual_all", C_MAN)
    assert c_opt < min(c_all, c_none)
    fig, ax = plt.subplots(figsize=(4.1, 2.5))
    ax.plot(ths, cost, color=BLUE, lw=2.0)
    ax.scatter([t_opt], [c_opt], s=45, color=RED, zorder=6)
    ax.axhline(c_none, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax.text(0.52, c_none + 1, "всё людям", fontsize=7.5, color=MUTED)
    ax.text(t_opt - 0.22, c_opt + 6, f"порог {t_opt:.2f}\n{c_opt:.0f} ₽/случай", fontsize=7.5, color=RED)
    ax.set_xlabel("порог передачи человеку", fontsize=8.5)
    ax.set_ylabel("₽ на случай", fontsize=8.5)
    ax.set_title("цена ошибки выбирает порог", fontsize=9)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "cost_threshold.png")
    print(f"cost: t*={t_opt:.3f}, min={c_opt:.1f}, all-auto={c_all:.1f}, all-manual={c_none:.1f}")


fig_three_causes()
fig_variance_split()
fig_reliability()
fig_risk_coverage()
fig_ood()
fig_conformal()
side_softmax_far()
side_epistemic_shrink()
side_base_rate()
side_cost()

(ROOT / "scripts" / "data").mkdir(parents=True, exist_ok=True)
with open(ROOT / "scripts" / "data" / "lesson56_facts.json", "w") as f:
    json.dump(FACTS, f, ensure_ascii=False, indent=1, sort_keys=True)
print("\n--- facts ---")
for k in sorted(FACTS):
    print(f"{k:22s} {FACTS[k]:.5f}")
print("lesson 56 figures written")
