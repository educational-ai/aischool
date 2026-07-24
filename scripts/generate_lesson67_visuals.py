"""Deterministic figures for lesson 67: MCMC / Metropolis-Hastings.

Everything quoted in the prose is computed here and asserted:
  * random-walk Metropolis on a bimodal target (trace, histogram, mode crossings);
  * three proposal scales -> acceptance rate, autocorrelation, tau_int, N_eff;
  * REAL data check with a known answer: SMS Spam Collection, Beta-Binomial posterior
    (exact mean vs MCMC mean);
  * REAL data Bayesian logistic regression on sklearn load_breast_cancer:
    isotropic vs preconditioned proposal, N_eff per gradient-free evaluation;
  * the "pretty histogram lie": two chains, each stuck in its own mode, split-Rhat;
  * MCSE coverage: naive iid interval vs N_eff-corrected interval;
  * sidenotes: acceptance vs sigma curve, rejections-are-steps histogram,
    Sobol LP-tau quasi-random vs pseudo-random points.

Run: python3 scripts/generate_lesson67_visuals.py
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
OUT = ROOT / "public" / "figures" / "lessons" / "67"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "67"

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


# ---------------------------------------------------------------- machinery
def log_target_bimodal(x, w=0.6, m=2.5, s=0.7):
    """Unnormalised log density of 0.6 N(-m,s^2) + 0.4 N(+m,s^2)."""
    a = w * np.exp(-((x + m) ** 2) / (2 * s * s))
    b = (1 - w) * np.exp(-((x - m) ** 2) / (2 * s * s))
    return np.log(a + b)


def rw_metropolis(logpdf, x0, sigma, n, seed, dim=1, cov_chol=None):
    """Random-walk Metropolis. Returns (chain, acceptance rate)."""
    rng = np.random.default_rng(seed)
    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    lp = logpdf(x if dim > 1 else x[0])
    chain = np.empty((n, dim))
    acc = 0
    for t in range(n):
        z = rng.standard_normal(dim)
        step = sigma * (z if cov_chol is None else cov_chol @ z)
        y = x + step
        lq = logpdf(y if dim > 1 else y[0])
        if np.log(rng.random()) < lq - lp:
            x, lp = y, lq
            acc += 1
        chain[t] = x
    return (chain if dim > 1 else chain[:, 0]), acc / n


def autocorr(x, maxlag):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    denom = np.dot(x, x)
    return np.array([np.dot(x[: n - k], x[k:]) / denom for k in range(maxlag + 1)])


def tau_geyer(x, maxlag=2000):
    """Geyer initial positive sequence estimator of the integrated autocorr time."""
    maxlag = min(maxlag, len(x) // 4)
    rho = autocorr(x, maxlag)
    tau = 1.0
    m = 1
    while 2 * m < maxlag:
        pair = rho[2 * m - 1] + rho[2 * m]
        if pair <= 0:
            break
        tau += 2 * pair
        m += 1
    return max(tau, 1.0)


def split_rhat(chains):
    """split-R-hat over a list of equal-length 1-D chains."""
    halves = []
    for c in chains:
        h = len(c) // 2
        halves.append(np.asarray(c[:h]))
        halves.append(np.asarray(c[h: 2 * h]))
    m = len(halves)
    n = len(halves[0])
    means = np.array([h.mean() for h in halves])
    variances = np.array([h.var(ddof=1) for h in halves])
    W = variances.mean()
    B = n * means.var(ddof=1)
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


# ---------------------------------------------------------------- fig 67.1
def fig_walk():
    xs = np.linspace(-6, 6, 800)
    dens = np.exp(log_target_bimodal(xs))
    Z = np.trapezoid(dens, xs)
    chain, acc = rw_metropolis(log_target_bimodal, 0.0, 1.2, 120000, seed=6701)
    warm = 20000
    post = chain[warm:]
    # mode crossings: sign change after visiting |x| >= 1.5
    side = np.where(post >= 1.5, 1, np.where(post <= -1.5, -1, 0))
    seen = side[side != 0]
    crossings = int(np.sum(seen[1:] != seen[:-1]))
    frac_left = float(np.mean(post < 0))
    tau = tau_geyer(post)
    neff = len(post) / tau
    exact_left = float(np.trapezoid(dens[xs <= 0], xs[xs <= 0]) / Z)
    print(f"   exact left mass = {exact_left:.4f}, gap = {exact_left - frac_left:.4f}, "
          f"1 crossing per {len(post)/crossings:.0f} steps")
    assert abs(exact_left - 0.6) < 0.05
    FACTS.update(walk_exact_left=exact_left, walk_gap=exact_left - frac_left,
                 walk_per_cross=len(post) / crossings)
    FACTS.update(walk_acc=acc, walk_cross=crossings, walk_left=frac_left,
                 walk_tau=tau, walk_neff=neff, walk_mean=float(post.mean()))
    print(f"fig1 walk: acc={acc:.3f} crossings={crossings} P(x<0)={frac_left:.3f} "
          f"tau={tau:.1f} Neff={neff:.0f} mean={post.mean():.3f}")
    assert 0.50 < acc < 0.62, acc
    assert 300 < crossings < 700, crossings
    assert 0.55 < frac_left < 0.62, frac_left
    naive_se = float(np.sqrt(exact_left * (1 - exact_left) / len(post)))
    honest_se = float(np.sqrt(exact_left * (1 - exact_left) / neff))
    print(f"   naive se (per draw) = {naive_se:.4f}, honest se (per Neff) = {honest_se:.4f}, "
          f"ratio = {honest_se / naive_se:.1f}")
    assert abs(naive_se - 0.0015) < 5e-05, naive_se
    assert abs(honest_se - 0.021) < 0.0005, honest_se
    assert (exact_left - frac_left) < 1.5 * honest_se
    FACTS.update(walk_naive_se=naive_se, walk_honest_se=honest_se)

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.2),
                             gridspec_kw={"height_ratios": [1.0, 1.1, 1.0]})
    ax = axes[0]
    ax.plot(xs, dens, color=INK, lw=2.0)
    ax.fill_between(xs, 0, dens, color=WASH)
    ax.set_title("Ненормированная цель $\\tilde{\\pi}(x)$: два холма", fontsize=13)
    ax.set_yticks([]); ax.set_xlim(-6, 6)

    ax = axes[1]
    ax.plot(np.arange(400), chain[:400], color=BLUE, lw=1.0)
    ax.axhline(-2.5, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.axhline(2.5, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.set_title(f"Первые 400 состояний, $\\sigma=1{{,}}2$, доля принятия {acc:.2f}", fontsize=13)
    ax.set_xlabel("такт $t$"); ax.set_ylabel("$x_t$"); ax.set_ylim(-6, 6)

    ax = axes[2]
    ax.hist(post, bins=80, range=(-6, 6), density=True, color=GOLD, alpha=0.55,
            label=f"гистограмма цепи, $N={len(post)}$")
    ax.plot(xs, dens / Z, color=RED, lw=2.2, label="нормированная $\\pi(x)$")
    ax.set_title(f"После прогрева: доля $x<0$ равна {frac_left:.3f}"
                 f" при {crossings} переходах между модами", fontsize=13)
    ax.set_xlabel("$x$"); ax.set_yticks([]); ax.set_xlim(-6, 6)
    ax.legend(frameon=False, fontsize=10)
    for a in axes:
        a.grid(True, color=GRID, lw=0.4, alpha=0.4); a.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "metropolis_walk.png")


# ---------------------------------------------------------------- fig 67.2
def fig_stepsize():
    """One-dimensional standard normal target: three proposal scales."""
    def lp(x):
        return -0.5 * x * x

    sigmas = [0.05, 2.4, 25.0]
    res = []
    for j, s in enumerate(sigmas):
        ch, acc = rw_metropolis(lp, 5.0, s, 60000, seed=6710 + j)
        post = ch[10000:]
        tau = tau_geyer(post)
        res.append({"sigma": s, "acc": acc, "chain": post, "tau": tau,
                    "neff": len(post) / tau, "rho": autocorr(post, 60)})
    for r in res:
        print(f"fig2 sigma={r['sigma']}: acc={r['acc']:.3f} tau={r['tau']:.1f} "
              f"Neff={r['neff']:.0f} mean={r['chain'].mean():.3f}")
    FACTS.update({f"step{i}_{k}": float(res[i][k]) for i in range(3)
                  for k in ("sigma", "acc", "tau", "neff")})
    assert res[0]["acc"] > 0.95 and res[2]["acc"] < 0.10
    assert res[1]["neff"] > 20 * res[0]["neff"] and res[1]["neff"] > 5 * res[2]["neff"]

    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.2))
    titles = ["слишком мелкий шаг", "рабочий масштаб", "слишком крупный шаг"]
    for i, r in enumerate(res):
        ax = axes[0, i]
        ax.plot(np.arange(1200), r["chain"][:1200], color=[RED, GREEN, VIOLET][i], lw=0.8)
        ax.set_title(f"{titles[i]}\n$\\sigma={r['sigma']:g}$, принято {r['acc']*100:.0f}%",
                     fontsize=11.5)
        ax.set_ylim(-4.2, 4.2); ax.set_xlabel("такт")
        if i == 0:
            ax.set_ylabel("$x_t$")
        ax = axes[1, i]
        ax.bar(np.arange(61), r["rho"], color=[RED, GREEN, VIOLET][i], width=0.9)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_ylim(-0.15, 1.05); ax.set_xlabel("лаг $k$")
        ax.set_title(f"$\\tau_{{int}}={r['tau']:.0f}$, $N_{{eff}}={r['neff']:.0f}$", fontsize=11.5)
        if i == 0:
            ax.set_ylabel("$\\rho_k$")
    for ax in axes.ravel():
        ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    fig.suptitle("Цель $N(0,1)$, 50 000 состояний после прогрева: "
                 "высокая доля принятия не значит хорошее перемешивание", y=1.02, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "stepsize.png")


# ---------------------------------------------------------------- fig 67.3 (REAL sms)
def load_sms():
    y = []
    with open(SMS, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "\t" not in line:
                continue
            label = line.split("\t", 1)[0].strip()
            y.append(1 if label == "spam" else 0)
    return np.array(y)


def fig_known_answer():
    y = load_sms()
    n, k = len(y), int(y.sum())
    a, b = 1 + k, 1 + n - k          # Beta(1,1) prior
    exact_mean = a / (a + b)
    exact_sd = np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    print(f"fig3 sms: n={n} spam={k} share={k/n:.5f} exact_mean={exact_mean:.5f} sd={exact_sd:.5f}")
    assert n == 5574 and k == 747

    def lp(p):
        if p <= 0 or p >= 1:
            return -np.inf
        return k * np.log(p) + (n - k) * np.log1p(-p)

    chain, acc = rw_metropolis(lp, 0.5, 0.02, 60000, seed=6720)
    post = chain[10000:]
    tau = tau_geyer(post)
    neff = len(post) / tau
    mc_mean = float(post.mean())
    mc_sd = float(post.std(ddof=1))
    err = abs(mc_mean - exact_mean)
    mcse = mc_sd / np.sqrt(neff)
    lo, hi = np.quantile(post, [0.025, 0.975])
    elo = exact_mean - 1.96 * exact_sd
    ehi = exact_mean + 1.96 * exact_sd
    print(f"   mcmc: acc={acc:.3f} mean={mc_mean:.5f} sd={mc_sd:.5f} tau={tau:.1f} "
          f"Neff={neff:.0f} err={err:.6f} mcse={mcse:.6f} CI=[{lo:.4f},{hi:.4f}] "
          f"exactCI=[{elo:.4f},{ehi:.4f}]")
    FACTS.update(sms_n=n, sms_k=k, sms_share=k / n, sms_exact=exact_mean,
                 sms_exact_sd=exact_sd, sms_acc=acc, sms_mean=mc_mean,
                 sms_tau=tau, sms_neff=neff, sms_err=err, sms_mcse=mcse,
                 sms_lo=float(lo), sms_hi=float(hi))
    assert err < 3 * mcse, (err, mcse)
    assert abs(mc_mean - 0.134) < 0.0005
    assert 0.124 < lo < 0.127 and 0.142 < hi < 0.145

    grid = np.linspace(0.118, 0.152, 500)
    from math import lgamma
    logbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
    pdf = np.exp((a - 1) * np.log(grid) + (b - 1) * np.log1p(-grid) - logbeta)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ax.hist(post, bins=70, density=True, color=BLUE, alpha=0.45,
            label="гистограмма цепи")
    ax.plot(grid, pdf, color=RED, lw=2.3, label="точная Beta$(748,\\,4828)$")
    ax.axvline(exact_mean, color=INK, lw=1.2, ls=(0, (3, 3)))
    ax.set_title(f"Реальный SMS-спам: цепь против известного ответа\n"
                 f"точное среднее {exact_mean:.4f}, среднее цепи {mc_mean:.4f}", fontsize=12.5)
    ax.set_xlabel("доля спама $p$"); ax.set_yticks([])
    ax.legend(frameon=False, fontsize=10)
    ax = axes[1]
    run = np.cumsum(post) / np.arange(1, len(post) + 1)
    ax.plot(np.arange(1, len(post) + 1), run, color=BLUE, lw=1.2, label="бегущее среднее")
    ax.axhline(exact_mean, color=RED, lw=1.6, label=f"точное {exact_mean:.4f}")
    ax.set_xscale("log"); ax.set_ylim(0.125, 0.145)
    ax.set_xlabel("число учтённых состояний $N$"); ax.set_ylabel("$\\hat{I}_N$")
    ax.set_title(f"Сходимость среднего: $\\tau_{{int}}={tau:.0f}$, "
                 f"$N_{{eff}}\\approx{neff:.0f}$", fontsize=12.5)
    ax.legend(frameon=False, fontsize=10)
    for a2 in axes:
        a2.grid(True, color=GRID, lw=0.4, alpha=0.4); a2.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "known_answer.png")


# ---------------------------------------------------------------- fig 67.4 (REAL breast cancer)
def fig_geometry():
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    names = list(d.feature_names)
    i1, i2 = names.index("mean radius"), names.index("mean perimeter")
    X = d.data[:, [i1, i2]]
    y = (d.target == 0).astype(float)      # 1 = malignant
    Xs = (X - X.mean(0)) / X.std(0)
    corr = float(np.corrcoef(Xs[:, 0], Xs[:, 1])[0, 1])
    n_obs = len(y)
    print(f"fig4 bc: n={n_obs} malignant={int(y.sum())} corr(radius,perimeter)={corr:.4f}")
    assert n_obs == 569 and int(y.sum()) == 212
    assert corr > 0.99

    A = np.column_stack([np.ones(n_obs), Xs])
    s2 = 4.0                                   # prior N(0, 2^2)

    def lp(beta):
        z = A @ beta
        # stable log-likelihood
        ll = np.sum(y * z - np.logaddexp(0.0, z))
        return ll - np.dot(beta, beta) / (2 * s2)

    # Laplace approximation of the posterior covariance (Newton, deterministic)
    beta = np.zeros(3)
    for _ in range(60):
        z = A @ beta
        p = 1 / (1 + np.exp(-z))
        grad = A.T @ (y - p) - beta / s2
        Wd = p * (1 - p)
        H = A.T @ (A * Wd[:, None]) + np.eye(3) / s2
        beta = beta + np.linalg.solve(H, grad)
    Sigma = np.linalg.inv(H)
    L = np.linalg.cholesky(Sigma)
    kappa = float(np.linalg.cond(Sigma))
    post_corr = float(Sigma[1, 2] / np.sqrt(Sigma[1, 1] * Sigma[2, 2]))
    print(f"   MAP={np.round(beta, 3)} cond(Sigma)={kappa:.1f} posterior corr={post_corr:.4f}")
    assert kappa > 100 and post_corr < -0.95
    axis_ratio = float(np.sqrt(kappa))
    print(f"   axis ratio of posterior ellipse = {axis_ratio:.1f}")
    FACTS.update(bc_axis_ratio=axis_ratio)
    FACTS.update(bc_post_corr=post_corr, bc_map1=float(beta[1]), bc_map2=float(beta[2]))

    N = 60000
    warm = 20000
    iso, acc_i = rw_metropolis(lp, [0.0, 0.0, 0.0], 0.30, N, seed=6731, dim=3)
    pre, acc_p = rw_metropolis(lp, [0.0, 0.0, 0.0], 1.20, N, seed=6732, dim=3, cov_chol=L)
    iso_p, pre_p = iso[warm:], pre[warm:]
    neff_i = [len(iso_p) / tau_geyer(iso_p[:, j]) for j in range(3)]
    neff_p = [len(pre_p) / tau_geyer(pre_p[:, j]) for j in range(3)]
    gain = min(neff_p) / min(neff_i)
    print(f"   isotropic acc={acc_i:.3f} Neff={[round(v) for v in neff_i]}")
    print(f"   preconditioned acc={acc_p:.3f} Neff={[round(v) for v in neff_p]}")
    print(f"   gain(min Neff) = {gain:.1f}x")
    assert gain > 5, gain

    m1 = float(pre_p[:, 1].mean()); m2 = float(pre_p[:, 2].mean())
    q1 = np.quantile(pre_p[:, 1], [0.025, 0.975])
    q2 = np.quantile(pre_p[:, 2], [0.025, 0.975])
    print(f"   beta_radius mean={m1:.2f} CI=[{q1[0]:.2f},{q1[1]:.2f}]")
    print(f"   beta_perimeter mean={m2:.2f} CI=[{q2[0]:.2f},{q2[1]:.2f}]")
    sumq = np.quantile(pre_p[:, 1] + pre_p[:, 2], [0.025, 0.975])
    sum_mean = float((pre_p[:, 1] + pre_p[:, 2]).mean())
    print(f"   sum mean={sum_mean:.2f} CI=[{sumq[0]:.2f},{sumq[1]:.2f}]")
    width_single = float(q1[1] - q1[0])
    width_sum = float(sumq[1] - sumq[0])
    assert width_sum < width_single
    FACTS.update(bc_n=n_obs, bc_mal=int(y.sum()), bc_corr=corr, bc_kappa=kappa,
                 bc_acc_iso=acc_i, bc_acc_pre=acc_p,
                 bc_neff_iso=min(neff_i), bc_neff_pre=min(neff_p), bc_gain=gain,
                 bc_m1=m1, bc_m2=m2, bc_sum=sum_mean,
                 bc_w_single=width_single, bc_w_sum=width_sum)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.0))
    lo1, hi1 = np.quantile(pre_p[:, 1], [0.001, 0.999])
    lo2, hi2 = np.quantile(pre_p[:, 2], [0.001, 0.999])
    for ax, ch, col, ttl, acc, ne in [
        (axes[0], iso_p, RED, "изотропное предложение", acc_i, min(neff_i)),
        (axes[1], pre_p, BLUE, "предложение по ковариации posterior", acc_p, min(neff_p)),
    ]:
        ax.hexbin(pre_p[:, 1], pre_p[:, 2], gridsize=45, cmap="Greys", mincnt=1,
                  alpha=0.35, extent=(lo1, hi1, lo2, hi2), linewidths=0)
        ax.plot(ch[:2500, 1], ch[:2500, 2], color=col, lw=0.6, alpha=0.85)
        ax.set_xlim(lo1, hi1); ax.set_ylim(lo2, hi2)
        ax.set_xlabel("$\\beta$ при mean radius")
        ax.set_ylabel("$\\beta$ при mean perimeter")
        ax.set_title(f"{ttl}\nпринято {acc*100:.0f}%, $N_{{eff}}\\approx{ne:.0f}$", fontsize=12)
        ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    fig.suptitle("Реальные данные: два почти дублирующих признака дают узкую долину "
                 f"(корреляция {corr:.3f})", y=1.02, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "geometry.png")


# ---------------------------------------------------------------- fig 67.5 multimodal trap
def fig_trap():
    def lp(x):
        return log_target_bimodal(x, w=0.5, m=6.0, s=0.8)

    chains = []
    for j in range(4):
        start = -6.0 if j < 2 else 6.0
        ch, _ = rw_metropolis(lp, start, 0.8, 40000, seed=6740 + j)
        chains.append(ch[10000:])
    crossings = []
    for c in chains:
        side = np.where(c >= 3, 1, np.where(c <= -3, -1, 0))
        seen = side[side != 0]
        crossings.append(int(np.sum(seen[1:] != seen[:-1])))
    pooled = np.concatenate(chains)
    frac = float(np.mean(pooled < 0))
    rhat = split_rhat(chains)
    print(f"fig5 trap: crossings={crossings} pooled P(x<0)={frac:.3f} split-Rhat={rhat:.2f}")
    assert max(crossings) == 0
    assert 0.45 < frac < 0.55
    assert rhat > 3
    FACTS.update(trap_frac=frac, trap_rhat=rhat, trap_cross=max(crossings))

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    cols = [BLUE, GREEN, RED, GOLD]
    for c, col in zip(chains, cols):
        ax.plot(np.arange(1500), c[:1500], color=col, lw=0.8, alpha=0.9)
    ax.set_xlabel("такт"); ax.set_ylabel("$x_t$"); ax.set_ylim(-9, 9)
    ax.set_title(f"Четыре цепи, ноль переходов, split-$\\hat{{R}}={rhat:.2f}$", fontsize=12.5)
    ax = axes[1]
    ax.hist(pooled, bins=80, density=True, color=VIOLET, alpha=0.55)
    ax.set_title(f"Объединённая гистограмма: доля $x<0$ равна {frac:.3f}\n"
                 "выглядит идеально — и лжёт", fontsize=12.5)
    ax.set_xlabel("$x$"); ax.set_yticks([])
    for a in axes:
        a.grid(True, color=GRID, lw=0.4, alpha=0.4); a.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "trap.png")


# ---------------------------------------------------------------- fig 67.6 MCSE coverage
def fig_coverage():
    def lp(x):
        return -0.5 * x * x

    R = 200
    N = 4000
    naive_hit = 0
    corr_hit = 0
    widths_n, widths_c = [], []
    for r in range(R):
        ch, _ = rw_metropolis(lp, 0.0, 0.6, N + 1000, seed=6760 + r)
        post = ch[1000:]
        mean = post.mean()
        sd = post.std(ddof=1)
        tau = tau_geyer(post)
        neff = len(post) / tau
        w_n = 1.96 * sd / np.sqrt(len(post))
        w_c = 1.96 * sd / np.sqrt(neff)
        widths_n.append(w_n); widths_c.append(w_c)
        naive_hit += abs(mean - 0.0) <= w_n
        corr_hit += abs(mean - 0.0) <= w_c
    cn, cc = naive_hit / R, corr_hit / R
    ratio = float(np.mean(widths_c) / np.mean(widths_n))
    print(f"fig6 coverage: naive={cn:.3f} corrected={cc:.3f} width ratio={ratio:.1f}")
    assert cn < 0.5 and cc > 0.85
    FACTS.update(cov_naive=cn, cov_corr=cc, cov_ratio=ratio, cov_R=R, cov_N=N)

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.bar([0, 1], [cn, cc], width=0.45, color=[RED, GREEN])
    ax.axhline(0.95, color=INK, lw=1.4, ls=(0, (4, 3)))
    ax.text(1.42, 0.955, "номинал 0,95", color=INK, fontsize=10.5, ha="right")
    for i, v in enumerate([cn, cc]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=13, color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"$\\bar f\\pm1{{,}}96\\,s/\\sqrt{{N}}$\n(как будто выборка независима)",
                        f"$\\bar f\\pm1{{,}}96\\,s/\\sqrt{{N_{{eff}}}}$\n(поправка на автокорреляцию)"])
    ax.set_ylim(0, 1.08); ax.set_ylabel("доля попаданий в 200 запусках")
    ax.set_title(f"Покрытие интервала для среднего $N(0,1)$, "
                 f"$N={N}$ состояний на запуск", fontsize=13)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "coverage.png")


# ---------------------------------------------------------------- sidenote images
def side_acceptance():
    def lp(x):
        return -0.5 * x * x

    sigmas = np.array([0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 2.4, 4.0, 8.0, 16.0, 32.0])
    accs, neffs = [], []
    for j, s in enumerate(sigmas):
        ch, a = rw_metropolis(lp, 0.0, s, 20000, seed=6780 + j)
        post = ch[4000:]
        accs.append(a)
        neffs.append(len(post) / tau_geyer(post))
    best = int(np.argmax(neffs))
    print(f"side1: best sigma={sigmas[best]:g} acc={accs[best]:.3f} Neff={neffs[best]:.0f}")
    assert 0.2 < accs[best] < 0.6
    FACTS.update(side_best_sigma=float(sigmas[best]), side_best_acc=accs[best])

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.semilogx(sigmas, accs, "o-", color=BLUE, lw=1.8, ms=5, label="доля принятия")
    ax2 = ax.twinx()
    ax2.semilogx(sigmas, np.array(neffs) / 16000, "s--", color=RED, lw=1.6, ms=4,
                 label="$N_{eff}/N$")
    ax.axvline(sigmas[best], color=MUTED, lw=1.0, ls=(0, (3, 3)))
    ax.set_xlabel("$\\sigma$ предложения"); ax.set_ylabel("доля принятия", color=BLUE)
    ax2.set_ylabel("$N_{eff}/N$", color=RED)
    ax.set_title("Принимать почти всё — не цель", fontsize=12)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "acceptance.png")


def side_repeats():
    """Dropping rejected steps biases the histogram towards the wide mode."""
    def lp(x):
        a = 0.5 * np.exp(-((x + 3) ** 2) / (2 * 0.35 ** 2)) / 0.35
        b = 0.5 * np.exp(-((x - 3) ** 2) / (2 * 1.8 ** 2)) / 1.8
        return np.log(a + b)

    chain, acc = rw_metropolis(lp, 0.0, 2.0, 200000, seed=6790)
    post = chain[20000:]
    moved = np.concatenate([[True], post[1:] != post[:-1]])
    accepted_only = post[moved]
    xs = np.linspace(-6, 9, 800)
    dens = np.exp(lp(xs))
    Z = np.trapezoid(dens, xs)
    full_left = float(np.mean(post < 0))
    acc_left = float(np.mean(accepted_only < 0))
    print(f"side2: acc={acc:.3f} P(x<0) full={full_left:.3f} accepted-only={acc_left:.3f}")
    assert abs(full_left - 0.5) < 0.05
    assert acc_left < 0.35
    FACTS.update(rep_full=full_left, rep_acc=acc_left, rep_acc_rate=acc)

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.hist(post, bins=70, range=(-6, 9), density=True, color=BLUE, alpha=0.45,
            label=f"все такты: {full_left:.2f}")
    ax.hist(accepted_only, bins=70, range=(-6, 9), density=True, histtype="step",
            color=RED, lw=1.8, label=f"только принятые: {acc_left:.2f}")
    ax.plot(xs, dens / Z, color=INK, lw=1.4, ls=(0, (4, 3)), label="истина: 0,50")
    ax.set_yticks([]); ax.set_xlabel("$x$")
    ax.set_title("Отказ — тоже такт", fontsize=12)
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "repeats.png")


def side_sobol():
    """Sobol LP-tau points vs pseudo-random points: discrepancy of the unit square."""
    from scipy.stats import qmc
    m = 256
    sob = qmc.Sobol(d=2, scramble=False).random(m)
    rng = np.random.default_rng(6799)
    prn = rng.random((m, 2))

    def worst_cell(pts, g=8):
        h, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=g, range=[[0, 1], [0, 1]])
        return int(h.min()), int(h.max())

    smin, smax = worst_cell(sob)
    pmin, pmax = worst_cell(prn)
    print(f"side3 sobol: cells 8x8 (expected 4) sobol=[{smin},{smax}] random=[{pmin},{pmax}]")
    assert smin == smax == 4
    assert pmin < 4 < pmax
    FACTS.update(sob_min=smin, sob_max=smax, rnd_min=pmin, rnd_max=pmax)

    fig, axes = plt.subplots(1, 2, figsize=(5.6, 3.0))
    for ax, pts, col, ttl in [(axes[0], sob, BLUE, f"ЛП$\\tau$ Соболя\n{smin}–{smax} в клетке"),
                              (axes[1], prn, RED, f"псевдослучайные\n{pmin}–{pmax} в клетке")]:
        ax.scatter(pts[:, 0], pts[:, 1], s=5, color=col)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(ttl, fontsize=10)
        for g in np.linspace(0, 1, 9):
            ax.axhline(g, color=GRID, lw=0.5); ax.axvline(g, color=GRID, lw=0.5)
    fig.tight_layout()
    save(fig, SIDE / "sobol.png")


def side_burnin():
    def lp(x):
        return -0.5 * (x - 0.0) ** 2

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    starts = [30.0, 12.0, -20.0]
    reach = []
    for j, s0 in enumerate(starts):
        ch, _ = rw_metropolis(lp, s0, 0.25, 6000, seed=6795 + j)
        ax.plot(ch[:2500], lw=0.9, color=[RED, GOLD, BLUE][j])
        idx = int(np.argmax(np.abs(ch) < 3))
        reach.append(idx)
    print(f"side4 burn-in: steps to reach |x|<3 = {reach}")
    assert max(reach) > 250
    FACTS.update(burn_max=max(reach), burn_min=min(reach))
    ax.axhspan(-3, 3, color=WASH)
    ax.set_xlabel("такт"); ax.set_ylabel("$x_t$")
    ax.set_title(f"Прогрев: до {max(reach)} тактов\nтолько чтобы дойти", fontsize=11.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "burnin.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SIDE.mkdir(parents=True, exist_ok=True)
    fig_walk()
    fig_stepsize()
    fig_known_answer()
    fig_geometry()
    fig_trap()
    fig_coverage()
    side_acceptance()
    side_repeats()
    side_sobol()
    side_burnin()
    (ROOT / "scripts" / "data" / "lesson67_facts.json").write_text(
        json.dumps({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in FACTS.items()}, ensure_ascii=False, indent=1))
    print("\nOK: all lesson 67 figures written and asserted")


if __name__ == "__main__":
    main()
