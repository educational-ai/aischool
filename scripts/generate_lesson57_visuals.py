"""Deterministic figures for lesson 57: loss functions and the price of a decision.

Every number quoted in the lesson text is computed here and asserted.

Real data:
  * scripts/data/sms-spam-collection.tsv  — cost-sensitive threshold, calibration, AUC
  * scripts/data/bike-sharing-hour.csv    — mean / median / 0.9-quantile of real demand
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SMS = ROOT / "scripts" / "data" / "sms-spam-collection.tsv"
BIKE = ROOT / "scripts" / "data" / "bike-sharing-hour.csv"
OUT = ROOT / "public" / "figures" / "lessons" / "57"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "57"

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


def note(key, value):
    FACTS[key] = float(value)
    return value


# ============================================================ fig 57.1 — two risks, one probability
C_FN, C_FP = 10000.0, 20.0


def fig_threshold():
    thr = C_FP / (C_FP + C_FN)
    note("alarm_threshold", thr)
    assert abs(thr - 0.001996007984) < 5e-13, thr
    note("alarm_ratio_to_half", 0.5 / thr)
    assert abs(FACTS["alarm_ratio_to_half"] - 250.5) < 0.05
    # цена самоуверенности в кросс-энтропии (натуральный логарифм)
    note("logloss_p001", float(-np.log(0.001)))
    note("logloss_p04", float(-np.log(0.4)))
    note("logloss_conf_ratio", FACTS["logloss_p001"] / FACTS["logloss_p04"])
    assert abs(FACTS["logloss_p001"] - 6.91) < 0.005
    assert abs(FACTS["logloss_p04"] - 0.92) < 0.005
    assert 7.4 < FACTS["logloss_conf_ratio"] < 7.6
    p = np.linspace(0, 0.02, 400)
    r0, r1 = C_FN * p, C_FP * (1 - p)
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    ax.plot(p, r0, color=RED, lw=2.4, label=r"$R(a=0\mid x)=10000\,p$  (не эвакуировать)")
    ax.plot(p, r1, color=BLUE, lw=2.4, label=r"$R(a=1\mid x)=20\,(1-p)$  (эвакуировать)")
    ax.plot(p, np.minimum(r0, r1), color=INK, lw=3.6, alpha=0.20,
            label="нижняя огибающая — байесовское решение")
    ax.axvline(thr, color=GOLD, lw=1.6, ls=(0, (5, 3)))
    ax.annotate("порог $p^*=" + f"{thr:.4f}".replace(".", "{,}") + "$", xy=(thr, C_FN * thr), xytext=(thr + 0.0022, 34),
                color=GOLD, fontsize=11,
                arrowprops=dict(arrowstyle="-", color=GOLD, lw=1.0))
    ax.fill_between(p, 0, 42, where=p < thr, color=BLUE, alpha=0.05)
    ax.text(0.0007, 38, "ждать", color=BLUE, fontsize=11, ha="center")
    ax.text(0.011, 38, "эвакуировать", color=RED, fontsize=11, ha="center")
    ax.set_xlim(0, 0.02); ax.set_ylim(0, 42)
    ax.set_xlabel("прогноз $p=P(y=1\\mid x)$"); ax.set_ylabel("ожидаемая потеря, у.е.")
    ax.set_title("Одна вероятность, два действия: порог задаёт цена, а не число 0,5")
    ax.legend(loc="center right", frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "threshold.png")


# ============================================================ fig 57.2 — surrogates
def fig_surrogates():
    m = np.linspace(-2.5, 3.0, 600)
    zero_one = (m <= 0).astype(float)
    logi = np.log(1 + np.exp(-m)) / np.log(2)
    hinge = np.maximum(0.0, 1.0 - m)
    expo = np.exp(-m)
    for name, arr in (("logistic", logi), ("hinge", hinge), ("exp", expo)):
        v0 = arr[np.argmin(np.abs(m))]
        assert abs(v0 - 1.0) < 0.05, (name, v0)
    note("logistic_at_m2", np.log(1 + np.exp(-2)) / np.log(2))
    note("logistic_at_m2_nat", float(np.log(1 + np.exp(-2))))
    note("logistic_at_m01_nat", float(np.log(1 + np.exp(-0.1))))
    note("logistic_at_m10_nat", float(np.log(1 + np.exp(-10.0))))
    note("exp_at_m_minus2", float(np.exp(2)))
    assert abs(FACTS["logistic_at_m2_nat"] - 0.127) < 0.0005
    assert abs(FACTS["logistic_at_m01_nat"] - 0.644) < 0.0005
    assert abs(FACTS["logistic_at_m10_nat"] - 0.000045) < 5e-07
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    ax.plot(m, zero_one, color=INK, lw=2.6, label=r"$0/1$: $\mathbf{1}\{m\leq 0\}$")
    ax.plot(m, logi, color=BLUE, lw=2.2, label=r"logistic: $\log_2(1+e^{-m})$")
    ax.plot(m, hinge, color=GREEN, lw=2.2, label=r"hinge: $\max(0,1-m)$")
    ax.plot(m, expo, color=RED, lw=2.2, label=r"exponential: $e^{-m}$")
    ax.axvline(0, color=LINE, lw=1.0)
    ax.axvspan(-2.5, 0, color=RED, alpha=0.045)
    ax.text(-1.25, 4.4, "ошибка", color=RED, fontsize=11, ha="center")
    ax.text(1.5, 1.55, "верно, с запасом", color=GREEN, fontsize=11, ha="center")
    ax.set_xlim(-2.5, 3); ax.set_ylim(0, 5)
    ax.set_xlabel("правильный отступ $m=y\\,s(x)$"); ax.set_ylabel("потеря")
    ax.set_title("Ступенька не даёт градиента, суррогаты — дают")
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "surrogates.png")


# ============================================================ real SMS pipeline
def load_sms():
    y, txt = [], []
    with open(SMS, encoding="utf-8", errors="replace") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 2:
                continue
            y.append(1 if row[0].strip() == "spam" else 0)
            txt.append(row[1])
    return np.array(y), txt


def sms_model():
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    y, txt = load_sms()
    note("sms_n", len(y)); note("sms_spam", int(y.sum()))
    note("sms_spam_share", y.mean())
    xtr, xte, ytr, yte = train_test_split(txt, y, test_size=0.3, random_state=0, stratify=y)
    vec = CountVectorizer(lowercase=True, min_df=2, binary=True)
    Atr = vec.fit_transform(xtr); Ate = vec.transform(xte)
    clf = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear")
    clf.fit(Atr, ytr)
    p = clf.predict_proba(Ate)[:, 1]
    return np.asarray(yte), p


def cost_curve(y, p, c_fp, c_fn, grid):
    out = np.empty_like(grid)
    for i, t in enumerate(grid):
        a = (p >= t).astype(int)
        fp = int(((a == 1) & (y == 0)).sum())
        fn = int(((a == 0) & (y == 1)).sum())
        out[i] = c_fp * fp + c_fn * fn
    return out


def fig_sms_cost(y, p):
    grid = np.linspace(0.001, 0.999, 999)
    n = len(y)
    note("sms_test_n", n); note("sms_test_spam", int(y.sum()))
    acc_all_ham = 1 - y.mean()
    note("sms_all_ham_acc", acc_all_ham)
    acc = np.array([( (p >= t).astype(int) == y).mean() for t in grid])
    best_acc_t = grid[int(np.argmax(acc))]
    note("sms_best_acc", acc.max()); note("sms_acc_at_half", ((p >= 0.5).astype(int) == y).mean())

    curves = {}
    for label, (cfp, cfn) in (("1:1", (1.0, 1.0)), ("20:1", (20.0, 1.0)), ("1:20", (1.0, 20.0))):
        c = cost_curve(y, p, cfp, cfn, grid)
        t_star = grid[int(np.argmin(c))]
        curves[label] = (c, t_star, c.min(), cfp, cfn)
        note(f"sms_t_{label.replace(':','_')}", t_star)
        note(f"sms_cost_{label.replace(':','_')}", c.min())
    a05 = (p >= 0.5).astype(int)
    note("sms_fp_at_half", int(((a05 == 1) & (y == 0)).sum()))
    note("sms_fn_at_half", int(((a05 == 0) & (y == 1)).sum()))
    for label, (cfp, cfn) in (("1_1", (1.0, 1.0)), ("20_1", (20.0, 1.0)), ("1_20", (1.0, 20.0))):
        note(f"sms_cost_{label}_at_half", float(cost_curve(y, p, cfp, cfn, np.array([0.5]))[0]))
    note("sms_overpay_20_1", FACTS["sms_cost_20_1_at_half"] - FACTS["sms_cost_20_1"])
    note("sms_overpay_1_20", FACTS["sms_cost_1_20_at_half"] - FACTS["sms_cost_1_20"])
    note("sms_ratio_1_20", FACTS["sms_cost_1_20_at_half"] / FACTS["sms_cost_1_20"])
    assert curves["20:1"][1] > curves["1:1"][1] > curves["1:20"][1], curves

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    cols = {"1:20": GREEN, "1:1": BLUE, "20:1": RED}
    names = {"1:20": "пропуск дороже: $c_{FP}{:}c_{FN}=1{:}20$",
             "1:1": "симметрично: $1{:}1$",
             "20:1": "блокировка дороже: $20{:}1$"}
    for label in ("1:20", "1:1", "20:1"):
        c, t_star, cmin, _, _ = curves[label]
        ax.plot(grid, c, color=cols[label], lw=2.2, label=names[label])
        ax.plot([t_star], [cmin], "o", color=cols[label], ms=8, zorder=6)
        dx, dy, ha = (0.035, -72, "left") if label == "1:20" else (0.0, 55, "center")
        ax.annotate("$t^*=" + f"{t_star:.2f}".replace(".", "{,}") + "$",
                    xy=(t_star, cmin), xytext=(t_star + dx, cmin + dy),
                    color=cols[label], fontsize=10, ha=ha)
    ax.axvline(0.5, color=GOLD, lw=1.4, ls=(0, (5, 3)))
    ax.text(0.5, 430, " привычные 0,5", color=GOLD, fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 750)
    ax.set_xlabel("порог $t$ по вероятности спама")
    ax.set_ylabel("суммарная стоимость ошибок на тесте, у.е.")
    ax.set_title("Реальный SMS-спам: три прайс-листа — три разных оптимальных порога")
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "sms_cost.png")


# ============================================================ fig 57.4 — calibration vs ranking
def auc(y, s):
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float)
    ss = s[order]
    i = 0
    ranks = np.arange(1, len(s) + 1, dtype=float)
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        r[order[i:j + 1]] = ranks[i:j + 1].mean()
        i = j + 1
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)


def fig_calibration(y, p):
    q = p ** 3  # монотонное искажение: порядок тот же, числа другие
    a1, a2 = auc(y, p), auc(y, q)
    note("sms_auc", a1)
    assert abs(a1 - a2) < 1e-12, (a1, a2)
    edges = np.linspace(0, 1, 11)
    mids, obs_p, obs_q = [], [], []

    def ece_of(s_):
        tot = 0.0
        for k in range(10):
            mk = (s_ >= edges[k]) & (s_ < edges[k + 1] + (1e-9 if k == 9 else 0))
            if mk.sum() == 0:
                continue
            tot += mk.sum() / len(s_) * abs(y[mk].mean() - s_[mk].mean())
        return tot

    for k in range(10):
        m1 = (p >= edges[k]) & (p < edges[k + 1])
        m2 = (q >= edges[k]) & (q < edges[k + 1])
        mids.append((edges[k] + edges[k + 1]) / 2)
        obs_p.append(y[m1].mean() if m1.sum() >= 5 else np.nan)
        obs_q.append(y[m2].mean() if m2.sum() >= 5 else np.nan)
    mids = np.array(mids); obs_p = np.array(obs_p); obs_q = np.array(obs_q)
    ok = ~np.isnan(obs_p); okq = ~np.isnan(obs_q)
    note("sms_ece_raw", ece_of(p))
    note("sms_ece_warp", ece_of(q))
    assert FACTS["sms_ece_warp"] > FACTS["sms_ece_raw"]
    # решения по цене 20:1 при пороге, посчитанном для честных вероятностей
    t = 20.0 / 21.0
    cost_p = float(cost_curve(y, p, 20.0, 1.0, np.array([t]))[0])
    cost_q = float(cost_curve(y, q, 20.0, 1.0, np.array([t]))[0])
    note("sms_theory_thr", t)
    note("sms_cost_calibrated", cost_p); note("sms_cost_warped", cost_q)
    assert cost_q > cost_p, (cost_p, cost_q)
    note("sms_cost_warp_ratio", cost_q / cost_p)
    assert abs(FACTS["sms_cost_warp_ratio"] - 1.5) < 0.05

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.2, ls=(0, (4, 3)), label="идеальная калибровка")
    ax.plot(mids[ok], obs_p[ok], "o-", color=BLUE, lw=2.0, ms=6, label="честные вероятности")
    ax.plot(mids[okq], obs_q[okq], "s-", color=RED, lw=2.0, ms=6, label="искажённые $p^3$")
    ax.set_xlabel("заявленная вероятность"); ax.set_ylabel("доля спама в корзине")
    ax.set_title("Надёжность", fontsize=13)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)

    ax = axes[1]
    ax.bar([0, 1], [a1, a2], color=[BLUE, RED], width=0.5)
    ax.bar([2.4, 3.4], [cost_p / 400, cost_q / 400], color=[BLUE, RED], width=0.5)
    ax.set_xticks([0, 1, 2.4, 3.4])
    ax.set_xticklabels(["AUC\nчестн.", "AUC\nискаж.", "цена\nчестн.", "цена\nискаж."], fontsize=9)
    for x, v in ((0, a1), (1, a2)):
        ax.text(x, v + 0.03, f"{v:.4f}", ha="center", color=INK, fontsize=9)
    for x, v in ((2.4, cost_p), (3.4, cost_q)):
        ax.text(x, v / 400 + 0.03, f"{v:.0f}", ha="center", color=INK, fontsize=9)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("AUC (слева) и стоимость / 400 (справа)")
    ax.set_title("Порядок тот же, счёт другой", fontsize=13)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    fig.suptitle("Монотонное искажение сохраняет ранжирование и разрушает решение по цене",
                 fontsize=14, y=1.02)
    save(fig, OUT / "calibration.png")


# ============================================================ fig 57.5 — real demand quantiles
def bike_slice():
    y = []
    with open(BIKE) as f:
        for row in csv.DictReader(f):
            if int(row["hr"]) == 8 and int(row["workingday"]) == 1:
                y.append(int(row["cnt"]))
    return np.array(y, float)


def pinball(y, q, tau):
    u = y - q
    return np.mean(np.where(u >= 0, tau * u, (tau - 1) * u))


def fig_quantiles():
    y = bike_slice()
    note("bike_n", len(y))
    mean = y.mean(); med = float(np.median(y)); q90 = float(np.quantile(y, 0.9))
    note("bike_mean", mean); note("bike_median", med); note("bike_q90", q90)
    note("bike_max", y.max()); note("bike_min", y.min())
    grid = np.linspace(y.min(), y.max(), 900)
    sq = np.array([np.mean((y - g) ** 2) for g in grid])
    ab = np.array([np.mean(np.abs(y - g)) for g in grid])
    p9 = np.array([pinball(y, g, 0.9) for g in grid])
    g_sq = grid[int(np.argmin(sq))]; g_ab = grid[int(np.argmin(ab))]; g_p9 = grid[int(np.argmin(p9))]
    note("bike_argmin_sq", g_sq); note("bike_argmin_abs", g_ab); note("bike_argmin_p90", g_p9)
    assert abs(g_sq - mean) < 2.0, (g_sq, mean)
    assert abs(g_ab - med) < 3.0, (g_ab, med)
    assert abs(g_p9 - q90) < 8.0, (g_p9, q90)
    note("bike_share_below_mean", float((y <= mean).mean()))
    # газетчик: недостача стоит 9, излишек 1
    def news(q):
        return float(np.mean(9.0 * np.maximum(y - q, 0) + 1.0 * np.maximum(q - y, 0)))
    note("news_cost_mean", news(mean))
    note("news_cost_median", news(med))
    note("news_cost_q90", news(q90))
    ng = np.linspace(y.min(), y.max(), 900)
    nc = np.array([news(g) for g in ng])
    note("news_argmin", ng[int(np.argmin(nc))])
    note("news_saving_vs_mean", news(mean) - nc.min())
    assert abs(FACTS["news_argmin"] - q90) < 8.0
    note("bike_share_below_q90", float((y <= q90).mean()))
    note("bike_gap_q90_mean", q90 - mean)
    note("news_ratio_mean_to_q90", news(mean) / news(q90))
    assert abs(FACTS["bike_gap_q90_mean"] - 242.0) < 0.05
    assert abs(FACTS["news_ratio_mean_to_q90"] - 2.7) < 0.05

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.12})
    ax = axes[0]
    ax.hist(y, bins=40, color=WASH, edgecolor=LINE, lw=0.6)
    for v, c, lab, dx, fy in ((mean, BLUE, f"среднее {mean:.0f}", 34, 0.97),
                              (med, GREEN, f"медиана {med:.0f}", -22, 0.97),
                              (q90, RED, f"квантиль 0,9 — {q90:.0f}", 9, 0.97)):
        ax.axvline(v, color=c, lw=2.0)
        ax.annotate(lab, xy=(v, ax.get_ylim()[1] * fy), xytext=(v + dx, ax.get_ylim()[1] * fy),
                    color=c, fontsize=10, rotation=90, va="top", ha="left",
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.8, shrinkA=0, shrinkB=2)
                    if abs(dx) > 12 else None)
    ax.set_ylabel("число дней")
    ax.set_title("Реальный спрос: 8 утра рабочего дня, два года велопроката")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)

    ax = axes[1]
    ax.plot(grid, sq / sq.max(), color=BLUE, lw=2.2, label="квадрат (нормирован)")
    ax.plot(grid, ab / ab.max(), color=GREEN, lw=2.2, label="модуль (нормирован)")
    ax.plot(grid, p9 / p9.max(), color=RED, lw=2.2, label=r"pinball $\tau=0{,}9$ (нормирован)")
    for g, c in ((g_sq, BLUE), (g_ab, GREEN), (g_p9, RED)):
        ax.axvline(g, color=c, lw=1.0, ls=(0, (4, 3)))
    ax.set_xlabel("константный прогноз $q$, поездок в час")
    ax.set_ylabel("средняя потеря")
    ax.legend(loc="upper center", frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "quantiles.png")


# ============================================================ fig 57.6 — proper vs improper
def fig_proper():
    p = 0.7
    q = np.linspace(0.001, 0.999, 999)
    logl = -(p * np.log(q) + (1 - p) * np.log(1 - q))
    brier = p * (1 - q) ** 2 + (1 - p) * q ** 2
    absl = p * (1 - q) + (1 - p) * q
    note("proper_p", p)
    note("proper_log_min_q", q[int(np.argmin(logl))])
    note("proper_brier_min_q", q[int(np.argmin(brier))])
    note("proper_abs_min_q", q[int(np.argmin(absl))])
    note("proper_log_at_truth", -(p * np.log(p) + (1 - p) * np.log(1 - p)))
    note("proper_brier_at_truth", p * (1 - p))
    assert abs(q[int(np.argmin(logl))] - p) < 0.002
    assert abs(q[int(np.argmin(brier))] - p) < 0.002
    assert q[int(np.argmin(absl))] > 0.99

    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    ax.plot(q, logl, color=BLUE, lw=2.4, label="log-loss (proper)")
    ax.plot(q, brier, color=GREEN, lw=2.4, label="Brier (proper)")
    ax.plot(q, absl, color=RED, lw=2.4, label=r"средний модуль $|q-y|$ (не proper)")
    ax.axvline(p, color=GOLD, lw=1.6, ls=(0, (5, 3)))
    ax.text(p, 1.28, " истина $p=0{,}7$", color=GOLD, fontsize=11)
    for arr, c in ((logl, BLUE), (brier, GREEN)):
        i = int(np.argmin(arr)); ax.plot([q[i]], [arr[i]], "o", color=c, ms=8, zorder=6)
    ax.plot([q[-1]], [absl[-1]], "o", color=RED, ms=8, zorder=6)
    ax.set_ylim(0, 1.4); ax.set_xlim(0, 1)
    ax.set_xlabel("объявленная вероятность $q$")
    ax.set_ylabel("ожидаемая потеря при истинном $p=0{,}7$")
    ax.set_title("Честность выгодна не всякой метрике")
    ax.legend(loc="upper center", frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "proper.png")


# ============================================================ sidenote images
def side_matrix():
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.15, 0.18), 0.7, 0.52, fill=True, color=WASH, ec=LINE))
    ax.plot([0.5, 0.5], [0.18, 0.70], color=LINE, lw=1.0)
    ax.plot([0.15, 0.85], [0.44, 0.44], color=LINE, lw=1.0)
    ax.text(0.325, 0.79, "$y=0$", ha="center", color=MUTED, fontsize=11)
    ax.text(0.675, 0.79, "$y=1$", ha="center", color=MUTED, fontsize=11)
    ax.text(0.09, 0.57, "$a=0$", ha="right", color=MUTED, fontsize=11)
    ax.text(0.09, 0.29, "$a=1$", ha="right", color=MUTED, fontsize=11)
    ax.text(0.325, 0.55, "0", ha="center", color=GREEN, fontsize=15)
    ax.text(0.675, 0.55, "$c_{FN}$", ha="center", color=RED, fontsize=15)
    ax.text(0.325, 0.27, "$c_{FP}$", ha="center", color=RED, fontsize=15)
    ax.text(0.675, 0.27, "0", ha="center", color=GREEN, fontsize=15)
    ax.text(0.5, 0.04, r"$p^*=\dfrac{c_{FP}}{c_{FP}+c_{FN}}$", ha="center", color=INK, fontsize=13)
    ax.set_xlim(0, 1); ax.set_ylim(-0.02, 0.95)
    save(fig, SIDE / "cost_matrix.png")


def side_pinball():
    u = np.linspace(-3, 3, 400)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for tau, c, lab in ((0.5, GREEN, r"$\tau=0{,}5$"), (0.9, RED, r"$\tau=0{,}9$")):
        ax.plot(u, np.where(u >= 0, tau * u, (tau - 1) * u), color=c, lw=2.2, label=lab)
    ax.axvline(0, color=LINE, lw=1.0)
    ax.set_xlabel("$u=y-q$", fontsize=10); ax.set_ylabel("потеря", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper center")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)
    save(fig, SIDE / "pinball.png")


def side_step():
    m = np.linspace(-2, 2, 800)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.step(m, (m <= 0).astype(float), where="post", color=INK, lw=2.2)
    ax.plot(m, np.log(1 + np.exp(-m)) / np.log(2), color=BLUE, lw=2.0)
    for x0 in (-1.4, 1.2):
        g = -1 / (np.log(2) * (1 + np.exp(x0)))
        y0 = np.log(1 + np.exp(-x0)) / np.log(2)
        ax.annotate("", xy=(x0 + 0.55, y0 + 0.55 * g), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4))
        ax.annotate("", xy=(x0 + 0.55, (x0 <= 0) * 1.0), xytext=(x0, (x0 <= 0) * 1.0),
                    arrowprops=dict(arrowstyle="->", color=FAINT, lw=1.2))
    ax.text(-0.55, 1.62, "градиент ступеньки = 0", color=MUTED, fontsize=9)
    ax.set_xlabel("$m=y\\,s$", fontsize=10); ax.set_ylabel("потеря", fontsize=10)
    ax.set_ylim(-0.15, 3.0)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)
    save(fig, SIDE / "step_gradient.png")


def side_imbalance():
    # реальный SMS: доля спама и «accuracy ничего не делающего фильтра»
    y, _ = load_sms()
    share = y.mean()
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.bar([0, 1], [1 - share, share], color=[WASH, RED], edgecolor=LINE, width=0.55)
    ax.text(0, 1 - share + 0.03, f"{(1-share)*100:.1f}%", ha="center", fontsize=10, color=INK)
    ax.text(1, share + 0.03, f"{share*100:.1f}%", ha="center", fontsize=10, color=RED)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["не спам", "спам"], fontsize=10)
    ax.set_ylim(0, 1.05); ax.set_ylabel("доля сообщений", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "imbalance.png")


def main():
    fig_threshold()
    fig_surrogates()
    y, p = sms_model()
    fig_sms_cost(y, p)
    fig_calibration(y, p)
    fig_quantiles()
    fig_proper()
    side_matrix(); side_pinball(); side_step(); side_imbalance()
    for k in sorted(FACTS):
        print(f"{k:28s} {FACTS[k]:.6f}")
    (ROOT / "scripts" / "data" / "lesson57_facts.json").write_text(
        json.dumps(FACTS, indent=1, ensure_ascii=False), encoding="utf-8")
    print("OK")


if __name__ == "__main__":
    main()
