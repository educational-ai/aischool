"""Deterministic figures for lesson 52: Bayesian linear regression.

Real data: bike-sharing-hour.csv (evening peak, working days of 2011) gives a small,
honestly noisy regression problem; sklearn's diabetes set is used for the ridge/posterior
mean identity. One calibration experiment is explicitly synthetic with a fixed seed.
Every number quoted in the lesson is computed here and asserted.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes

ROOT = Path(__file__).resolve().parents[1]
BIKE = ROOT / "scripts" / "data" / "bike-sharing-hour.csv"
OUT = ROOT / "public" / "figures" / "lessons" / "52"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "52"

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


def record(name, value):
    FACTS[name] = float(value)
    return value


# ------------------------------------------------------------------ data
def bike_evening():
    """Evening peak (17:00) of working days, 2011: temperature and ride count."""
    temp, atemp, cnt = [], [], []
    with open(BIKE) as f:
        for row in csv.DictReader(f):
            if row["hr"] == "17" and row["workingday"] == "1" and row["yr"] == "0":
                temp.append(float(row["temp"]))
                atemp.append(float(row["atemp"]))
                cnt.append(float(row["cnt"]))
    return np.array(temp), np.array(atemp), np.array(cnt)


TEMP, ATEMP, CNT = bike_evening()
N_ALL = len(CNT)
record("n_evening_days", N_ALL)
record("temp_mean", TEMP.mean())
record("cnt_mean", CNT.mean())

# reference fit on the whole subset -> noise scale sigma
Xall = np.column_stack([np.ones(N_ALL), TEMP - TEMP.mean()])
w_all, *_ = np.linalg.lstsq(Xall, CNT, rcond=None)
resid_all = CNT - Xall @ w_all
SIGMA = float(np.sqrt(resid_all @ resid_all / (N_ALL - 2)))
record("sigma_full", SIGMA)
record("slope_full", w_all[1])
record("intercept_full", w_all[0])

# small sample: six days
CENTER = 0.5  # середина шкалы temp
rng = np.random.default_rng(0)
IDX = np.sort(rng.choice(N_ALL, size=6, replace=False))
XS = TEMP[IDX]
YS = CNT[IDX]
XC = XS - CENTER
X6 = np.column_stack([np.ones(6), XC])
w_ols6, *_ = np.linalg.lstsq(X6, YS, rcond=None)
record("slope_ols6", w_ols6[1])
record("intercept_ols6", w_ols6[0])

TAU = 300.0  # prior sd for both coefficients, in rides
S0 = TAU ** 2 * np.eye(2)


def posterior(X, y, sigma, S0inv, m0=None):
    if m0 is None:
        m0 = np.zeros(X.shape[1])
    Sn_inv = S0inv + X.T @ X / sigma ** 2
    Sn = np.linalg.inv(Sn_inv)
    mn = Sn @ (S0inv @ m0 + X.T @ y / sigma ** 2)
    return mn, Sn


S0INV = np.linalg.inv(S0)
M6, SN6 = posterior(X6, YS, SIGMA, S0INV)
record("slope_post6", M6[1])
record("intercept_post6", M6[0])
record("slope_post6_sd", np.sqrt(SN6[1, 1]))
record("intercept_post6_sd", np.sqrt(SN6[0, 0]))
record("shrink_ratio", M6[1] / w_ols6[1])
assert abs(M6[1]) < abs(w_ols6[1])


def band(xgrid, mn, Sn, sigma):
    Phi = np.column_stack([np.ones(len(xgrid)), xgrid - CENTER])
    mean = Phi @ mn
    var_f = np.einsum("ij,jk,ik->i", Phi, Sn, Phi)
    return mean, np.sqrt(var_f), np.sqrt(var_f + sigma ** 2)


# ---------------------------------------------- fig 52.1 posterior lines
def fig_posterior_lines() -> None:
    xg = np.linspace(0.0, 1.0, 300)
    mean, sd_f, sd_y = band(xg, M6, SN6, SIGMA)
    i_c = int(np.argmin(np.abs(xg - CENTER)))
    i_e = 0
    record("sd_f_center", sd_f[i_c])
    record("sd_f_cold_edge", sd_f[i_e])
    record("sd_f_hot_edge", sd_f[-1])
    record("halfwidth_center", 1.96 * sd_f[i_c])
    record("halfwidth_edge", 1.96 * sd_f[i_e])
    record("fan_ratio", sd_f[i_e] / sd_f[i_c])
    record("gap_ols_full", abs(w_ols6[1] - w_all[1]))
    record("gap_post_full", abs(M6[1] - w_all[1]))
    assert sd_f[i_e] > 2 * sd_f[i_c]
    assert abs(M6[1] - w_all[1]) < abs(w_ols6[1] - w_all[1])

    draws = np.random.default_rng(520).multivariate_normal(M6, SN6, size=40)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.axvspan(XS.min(), XS.max(), color=WASH, alpha=0.75, zorder=0)
    for w in draws:
        ax.plot(xg, w[0] + w[1] * (xg - CENTER), color=BLUE, lw=0.9, alpha=0.22, zorder=1)
    ax.fill_between(xg, mean - 1.96 * sd_f, mean + 1.96 * sd_f, color=BLUE, alpha=0.16,
                    lw=0, zorder=2, label="95% для среднего $x^\\top w$")
    ax.plot(xg, mean, color=BLUE, lw=2.6, zorder=4, label="posterior mean")
    ols = w_ols6[0] + w_ols6[1] * (xg - CENTER)
    ax.plot(xg, ols, color=RED, lw=2.0, ls=(0, (5, 3)), zorder=4, label="МНК по шести точкам")
    ax.scatter(XS, YS, s=60, color=INK, zorder=6, label="шесть реальных вечеров")
    ax.scatter(TEMP, CNT, s=12, color=FAINT, alpha=0.55, zorder=1)
    ax.set_xlabel("температура (нормированная, признак temp)")
    ax.set_ylabel("поездок в 17:00")
    ax.set_title("Не одна линия, а веер линий, согласованных с данными")
    ax.set_xlim(0, 1); ax.set_ylim(0, 900)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "posterior_lines.png")


# ---------------------------------------------- fig 52.2 weight space
def gauss_grid(mu, cov, g1, g2):
    G1, G2 = np.meshgrid(g1, g2)
    d = np.stack([G1 - mu[0], G2 - mu[1]], axis=-1)
    P = np.linalg.inv(cov)
    q = np.einsum("...i,ij,...j->...", d, P, d)
    return G1, G2, np.exp(-0.5 * q)


def fig_weight_space() -> None:
    n = 25
    idx = np.sort(np.random.default_rng(7).choice(N_ALL, size=n, replace=False))
    t = (TEMP[idx] - TEMP.mean()) / TEMP.std()
    a = (ATEMP[idx] - ATEMP.mean()) / ATEMP.std()
    y = CNT[idx] - CNT.mean()
    r = float(np.corrcoef(TEMP, ATEMP)[0, 1])
    record("corr_temp_atemp", r)
    assert r > 0.97
    X = np.column_stack([t, a])
    tau2 = 300.0 ** 2
    S0i = np.eye(2) / tau2
    like_cov = np.linalg.inv(X.T @ X / SIGMA ** 2)
    w_ml = like_cov @ (X.T @ y / SIGMA ** 2)
    mn, Sn = posterior(X, y, SIGMA, S0i)
    ev_l = np.linalg.eigvalsh(like_cov)
    ev_p = np.linalg.eigvalsh(Sn)
    record("like_axis_ratio", np.sqrt(ev_l[1] / ev_l[0]))
    record("post_axis_ratio", np.sqrt(ev_p[1] / ev_p[0]))
    u_sum = np.array([1, 1]) / np.sqrt(2)
    u_dif = np.array([1, -1]) / np.sqrt(2)
    record("sd_sum", np.sqrt(u_sum @ Sn @ u_sum))
    record("sd_diff", np.sqrt(u_dif @ Sn @ u_dif))
    record("sd_ratio_diff_sum", np.sqrt(u_dif @ Sn @ u_dif) / np.sqrt(u_sum @ Sn @ u_sum))
    record("mn_sum", mn.sum())
    assert (u_dif @ Sn @ u_dif) > 5 * (u_sum @ Sn @ u_sum)

    g = np.linspace(-700, 700, 320)
    panels = [("prior: круг", np.zeros(2), tau2 * np.eye(2), GREEN),
              ("likelihood: гребень", w_ml, like_cov, RED),
              ("posterior: произведение", mn, Sn, BLUE)]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.9), sharex=True, sharey=True)
    for ax, (title, mu, cov, col) in zip(axes, panels):
        G1, G2, Z = gauss_grid(mu, cov, g, g)
        ax.contourf(G1, G2, Z, levels=[0.05, 0.25, 0.6, 1.0], colors=[col + "22", col + "44", col + "77"])
        ax.contour(G1, G2, Z, levels=[0.05, 0.25, 0.6], colors=col, linewidths=1.1)
        ax.plot([-700, 700], [700, -700], color=MUTED, lw=0.7, ls=(0, (3, 3)))
        ax.plot(mu[0], mu[1], "o", color=INK, ms=5)
        ax.set_title(title, fontsize=11.5)
        ax.set_xlabel("$w_{\\mathrm{temp}}$")
        ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    axes[0].set_ylabel("$w_{\\mathrm{atemp}}$")
    fig.suptitle("Два почти одинаковых признака: данные знают сумму, но не разность", y=1.04, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "weight_space.png")


# ---------------------------------------------- fig 52.3 ridge identity
def fig_ridge() -> None:
    Xd, yd = load_diabetes(return_X_y=True)
    Xd = (Xd - Xd.mean(0)) / Xd.std(0)
    yd = yd - yd.mean()
    n, p = Xd.shape
    record("diabetes_n", n)
    record("diabetes_p", p)
    sigma = float(np.std(yd - Xd @ np.linalg.lstsq(Xd, yd, rcond=None)[0], ddof=p))
    record("diabetes_sigma", sigma)
    taus = np.logspace(-1.0, 2.4, 60)
    paths = []
    worst = 0.0
    for tau in taus:
        mn, _ = posterior(Xd, yd, sigma, np.eye(p) / tau ** 2)
        lam = sigma ** 2 / tau ** 2
        ridge = np.linalg.solve(Xd.T @ Xd + lam * np.eye(p), Xd.T @ yd)
        worst = max(worst, float(np.max(np.abs(mn - ridge))))
        paths.append(mn)
    paths = np.array(paths)
    record("ridge_max_gap", worst)
    assert worst < 1e-8
    lam_mid = sigma ** 2 / 10.0 ** 2
    record("lambda_at_tau10", lam_mid)
    mn10, Sn10 = posterior(Xd, yd, sigma, np.eye(p) / 100.0)
    ols = np.linalg.lstsq(Xd, yd, rcond=None)[0]
    record("bmi_ols", ols[2])
    record("bmi_post_tau10", mn10[2])
    record("bmi_post_sd", np.sqrt(Sn10[2, 2]))
    record("s1_ols", ols[4])
    record("s1_post_tau10", mn10[4])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.3), gridspec_kw={"width_ratios": [1.35, 1]})
    cols = [BLUE, RED, GREEN, GOLD, VIOLET, INK, MUTED, "#7a5c3a", "#3f6f7a", "#8c3a6b"]
    for j in range(p):
        ax.plot(taus, paths[:, j], color=cols[j % len(cols)], lw=1.6)
    ax.set_xscale("log")
    ax.axvline(10.0, color=MUTED, lw=0.9, ls=(0, (3, 3)))
    ax.text(10.6, ax.get_ylim()[1] * 0.86, "$\\tau=10$", fontsize=10, color=MUTED)
    ax.set_xlabel("ширина prior $\\tau$ (лог)"); ax.set_ylabel("posterior mean коэффициента")
    ax.set_title("Ширина prior = сила ridge", fontsize=13)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)

    order = np.argsort(-np.abs(ols))[:6]
    names = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
    ypos = np.arange(len(order))
    ax2.barh(ypos + 0.18, ols[order], height=0.34, color=RED, alpha=0.75, label="МНК")
    ax2.barh(ypos - 0.18, mn10[order], height=0.34, color=BLUE, alpha=0.85, label="posterior mean, $\\tau=10$")
    ax2.errorbar(mn10[order], ypos - 0.18, xerr=1.96 * np.sqrt(np.diag(Sn10)[order]),
                 fmt="none", ecolor=INK, elinewidth=1.0, capsize=3)
    ax2.set_yticks(ypos); ax2.set_yticklabels([names[j] for j in order], fontsize=10)
    ax2.axvline(0, color=MUTED, lw=0.8)
    ax2.set_xlabel("коэффициент")
    ax2.set_title("Стягивание и его цена", fontsize=13)
    ax2.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax2.grid(True, axis="x", color=GRID, lw=0.4, alpha=0.4); ax2.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "ridge_identity.png")


# ---------------------------------------------- fig 52.4 two layers
def fig_two_layers() -> None:
    xg = np.linspace(0.0, 1.0, 300)
    mean, sd_f, sd_y = band(xg, M6, SN6, SIGMA)
    i_c = int(np.argmin(np.abs(xg - CENTER)))
    i_e = 0
    record("share_noise_center", SIGMA ** 2 / (SIGMA ** 2 + sd_f[i_c] ** 2))
    record("share_noise_edge", SIGMA ** 2 / (SIGMA ** 2 + sd_f[i_e] ** 2))
    i_hot = int(np.argmin(np.abs(xg - 0.9)))
    record("mean_at_hot", mean[i_hot])
    record("pred_lo_hot", mean[i_hot] - 1.96 * sd_y[i_hot])
    record("pred_hi_hot", mean[i_hot] + 1.96 * sd_y[i_hot])
    record("mean_at_center", mean[i_c])
    record("pred_halfwidth_center", 1.96 * sd_y[i_c])
    record("noise_only_halfwidth", 1.96 * SIGMA)
    record("shrink_of_center_band_pct", 100 * (1 - 1.96 * SIGMA / (1.96 * sd_y[i_c])))
    record("xs_mean", XS.mean())
    assert 6.5 < FACTS["shrink_of_center_band_pct"] < 8.0
    record("pred_halfwidth_edge", 1.96 * sd_y[i_e])
    assert FACTS["share_noise_center"] > 0.8 > FACTS["share_noise_edge"]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.8, 6.4), sharex=True,
                                  gridspec_kw={"height_ratios": [1.7, 1]})
    ax.fill_between(xg, mean - 1.96 * sd_y, mean + 1.96 * sd_y, color=GOLD, alpha=0.20, lw=0,
                    label="95% для нового вечера (с шумом $\\sigma^2$)")
    ax.fill_between(xg, mean - 1.96 * sd_f, mean + 1.96 * sd_f, color=BLUE, alpha=0.28, lw=0,
                    label="95% для среднего $x^\\top w$")
    ax.plot(xg, mean, color=BLUE, lw=2.4)
    ax.scatter(XS, YS, s=60, color=INK, zorder=6, label="шесть вечеров")
    ax.set_ylabel("поездок в 17:00"); ax.set_ylim(-100, 1000)
    ax.set_title("Две причины ширины: шум мира и незнание весов")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)

    ax2.fill_between(xg, 0, SIGMA ** 2 * np.ones_like(xg), color=GOLD, alpha=0.35, lw=0,
                     label="$\\sigma^2$ — шум объекта")
    ax2.fill_between(xg, SIGMA ** 2, SIGMA ** 2 + sd_f ** 2, color=BLUE, alpha=0.35, lw=0,
                     label="$x^\\top S_n x$ — незнание весов")
    for x0 in XS:
        ax2.axvline(x0, color=MUTED, lw=0.6, alpha=0.5)
    ax2.set_xlabel("температура (признак temp)"); ax2.set_ylabel("дисперсия")
    ax2.legend(loc="upper center", frameon=False, fontsize=9.5, ncol=2)
    ax2.grid(True, color=GRID, lw=0.4, alpha=0.4); ax2.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "two_layers.png")


# ---------------------------------------------- fig 52.5 calibration
def fig_calibration() -> None:
    """Explicitly synthetic, fixed seed: does the interval cover what it promises?"""
    rng = np.random.default_rng(1952)
    n, reps = 5, 4000
    sigma_true, w_true = 2.0, np.array([1.0, 2.0])
    tau = 5.0
    hit_known = hit_plug = hit_t = 0
    width_plug = []
    width_t = []
    from scipy.stats import t as student
    for _ in range(reps):
        x = rng.uniform(-2, 2, n)
        X = np.column_stack([np.ones(n), x])
        y = X @ w_true + rng.normal(0, sigma_true, n)
        xs = rng.uniform(-2, 2)
        phi = np.array([1.0, xs])
        ynew = phi @ w_true + rng.normal(0, sigma_true)
        # (a) sigma known
        mn, Sn = posterior(X, y, sigma_true, np.eye(2) / tau ** 2)
        sd = np.sqrt(phi @ Sn @ phi + sigma_true ** 2)
        hit_known += abs(ynew - phi @ mn) <= 1.96 * sd
        # (b) plug-in sigma-hat, same normal quantile
        w_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        rss = float(np.sum((y - X @ w_hat) ** 2))
        s2 = rss / (n - 2)
        mnp, Snp = posterior(X, y, np.sqrt(s2), np.eye(2) / tau ** 2)
        sdp = np.sqrt(phi @ Snp @ phi + s2)
        hit_plug += abs(ynew - phi @ mnp) <= 1.96 * sdp
        width_plug.append(2 * 1.96 * sdp)
        # (c) Student quantile with n-2 degrees of freedom
        q = float(student.ppf(0.975, n - 2))
        hit_t += abs(ynew - phi @ mnp) <= q * sdp
        width_t.append(2 * q * sdp)
    cov_known = hit_known / reps
    cov_plug = hit_plug / reps
    cov_t = hit_t / reps
    record("cov_known", cov_known)
    record("cov_plug", cov_plug)
    record("cov_student", cov_t)
    record("width_plug", float(np.median(width_plug)))
    record("width_student", float(np.median(width_t)))
    record("student_q3", float(student.ppf(0.975, 3)))
    record("calib_reps", reps)
    record("calib_n", n)
    assert cov_plug < 0.93 < cov_t
    assert 0.93 < cov_known < 0.97

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    labels = ["$\\sigma$ известна", "подставили $\\widehat\\sigma$,\nквантиль 1,96",
              "подставили $\\widehat\\sigma$,\nквантиль Стьюдента"]
    vals = [cov_known, cov_plug, cov_t]
    cols = [BLUE, RED, GREEN]
    bars = ax.bar(labels, vals, color=cols, alpha=0.85, width=0.55)
    ax.axhline(0.95, color=INK, lw=1.2, ls=(0, (4, 3)))
    ax.text(2.45, 0.952, "обещано 0,95", fontsize=10, color=INK, ha="right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=11.5, color=INK)
    ax.set_ylim(0.80, 1.0); ax.set_ylabel("доля покрытий")
    ax.set_title(f"Что обещает интервал и что он делает: {reps} повторов, n={n} (модельный эксперимент)",
                 fontsize=13)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "calibration.png")


# ---------------------------------------------- fig 52.6 design of experiment
def fig_design() -> None:
    xg = np.linspace(0.0, 1.0, 301)
    Phi = np.column_stack([np.ones(len(xg)), xg - CENTER])
    var_f = np.einsum("ij,jk,ik->i", Phi, SN6, Phi)
    trace_after = []
    det_after = []
    for phi in Phi:
        Sn_new = np.linalg.inv(np.linalg.inv(SN6) + np.outer(phi, phi) / SIGMA ** 2)
        trace_after.append(np.trace(Sn_new))
        det_after.append(np.linalg.det(Sn_new))
    trace_after = np.array(trace_after); det_after = np.array(det_after)
    i_best = int(np.argmin(det_after))
    record("design_best_x", xg[i_best])
    record("det_before", np.linalg.det(SN6))
    record("det_after_best", det_after[i_best])
    record("det_drop_pct", 100 * (1 - det_after[i_best] / np.linalg.det(SN6)))
    i_mid = int(np.argmin(np.abs(xg - CENTER)))
    record("det_after_mid", det_after[i_mid])
    record("det_drop_mid_pct", 100 * (1 - det_after[i_mid] / np.linalg.det(SN6)))
    record("sd_f_at_best", np.sqrt(var_f[i_best]))
    record("sd_f_at_mid", np.sqrt(var_f[i_mid]))
    assert det_after[i_best] < det_after[i_mid]
    assert xg[i_best] > 0.9 or xg[i_best] < 0.1
    record("design_worst_x", xg[int(np.argmax(det_after))])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax.plot(xg, np.sqrt(var_f), color=BLUE, lw=2.4)
    ax.axvspan(XS.min(), XS.max(), color=WASH, alpha=0.8, zorder=0)
    for x0 in XS:
        ax.plot(x0, 0, "o", color=INK, ms=6, clip_on=False)
    ax.set_xlabel("кандидат $x$"); ax.set_ylabel("$\\sqrt{x^\\top S_n x}$, поездок")
    ax.set_title("Где модель меньше всего уверена", fontsize=13)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)

    ax2.plot(xg, det_after / np.linalg.det(SN6), color=RED, lw=2.4)
    ax2.axvline(xg[i_best], color=GREEN, lw=1.2, ls=(0, (3, 3)))
    ax2.text(0.04, 0.10, "D-оптимальная точка", fontsize=9.5, color=GREEN,
             ha="left", transform=ax2.transAxes)
    ax2.axvspan(XS.min(), XS.max(), color=WASH, alpha=0.8, zorder=0)
    ax2.set_xlabel("куда поставить седьмое измерение")
    ax2.set_ylabel("$\\det S_{n+1}/\\det S_n$")
    ax2.set_title("Сколько незнания снимет новый вечер", fontsize=13)
    ax2.grid(True, color=GRID, lw=0.4, alpha=0.4); ax2.set_axisbelow(True)
    fig.suptitle("Планирование эксперимента: критерий указывает на края", y=1.03, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "design.png")


# ---------------------------------------------- sidenote margins
def side_precision() -> None:
    """1D worked example: prior N(0,1), one observation x=2,y=6, sigma^2=4."""
    w = np.linspace(-2, 5, 400)
    prior = np.exp(-0.5 * w ** 2)
    like = np.exp(-((6 - 2 * w) ** 2) / (2 * 4))
    Sn = 1 / (1 + 4 / 4)
    mn = Sn * (2 * 6 / 4)
    record("toy_Sn", Sn); record("toy_mn", mn)
    assert abs(Sn - 0.5) < 1e-12 and abs(mn - 1.5) < 1e-12
    post = np.exp(-0.5 * (w - mn) ** 2 / Sn)
    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    ax.plot(w, prior / prior.max(), color=GREEN, lw=1.7, label="prior")
    ax.plot(w, like / like.max(), color=RED, lw=1.7, ls=(0, (4, 3)), label="likelihood")
    ax.plot(w, post / post.max(), color=BLUE, lw=2.2, label="posterior")
    ax.axvline(3.0, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    ax.text(3.05, 0.25, "МНК = 3", fontsize=8, color=MUTED)
    ax.set_xlabel("$w$", fontsize=9); ax.set_yticks([])
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("точности складываются: 1 + 1 = 2", fontsize=9)
    save(fig, SIDE / "precision_add.png")


def side_shrink() -> None:
    n = np.arange(1, 61)
    for sn2 in (1.0,):
        pass
    lam = 4.0  # sigma^2/tau^2 in units of x^2 sums
    factor = n / (n + lam)
    record("shrink_n5", float(5 / (5 + lam)))
    record("shrink_n50", float(50 / (50 + lam)))
    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    ax.plot(n, factor, color=BLUE, lw=2.0)
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    ax.plot(5, 5 / (5 + lam), "o", color=RED, ms=6)
    ax.text(7, 5 / (5 + lam) - 0.02, f"n=5: {5/(5+lam):.2f}", fontsize=8, color=RED)
    ax.set_xlabel("число наблюдений", fontsize=9)
    ax.set_ylabel("доля от МНК", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_title("prior уступает данным", fontsize=9)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "shrink_n.png")


def side_student() -> None:
    from scipy.stats import norm, t as student
    z = np.linspace(-6, 6, 500)
    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    ax.plot(z, norm.pdf(z), color=BLUE, lw=1.9, label="норм.")
    ax.plot(z, student.pdf(z, 3), color=RED, lw=1.9, ls=(0, (4, 3)), label="$t_3$")
    ax.axvline(1.96, color=BLUE, lw=0.8, ls=(0, (2, 2)))
    ax.axvline(student.ppf(0.975, 3), color=RED, lw=0.8, ls=(0, (2, 2)))
    ax.text(2.0, 0.30, "1,96", fontsize=8, color=BLUE)
    ax.text(3.3, 0.20, f"{student.ppf(0.975,3):.2f}", fontsize=8, color=RED)
    ax.set_yticks([]); ax.set_xlabel("$z$", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("незнание $\\sigma$ утяжеляет хвосты", fontsize=9)
    save(fig, SIDE / "student_tails.png")


def side_corr() -> None:
    r = float(np.corrcoef(TEMP, ATEMP)[0, 1])
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    ax.scatter(TEMP, ATEMP, s=12, color=BLUE, alpha=0.6)
    ax.set_xlabel("temp", fontsize=9); ax.set_ylabel("atemp", fontsize=9)
    ax.set_title(f"temp и atemp: $r={r:.3f}$", fontsize=9)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "corr_temp_atemp.png")


fig_posterior_lines()
fig_weight_space()
fig_ridge()
fig_two_layers()
fig_calibration()
fig_design()
side_precision()
side_shrink()
side_student()
side_corr()

(ROOT / "scripts" / "data").mkdir(parents=True, exist_ok=True)
print("six points:", [(round(a, 3), int(b)) for a, b in zip(XS, YS)])
print(json.dumps({k: round(v, 6) for k, v in FACTS.items()}, ensure_ascii=False, indent=1))
print("lesson 52 figures written")
