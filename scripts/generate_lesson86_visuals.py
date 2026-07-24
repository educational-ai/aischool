"""Deterministic figures for lesson 86: diffusion as reversed noising.

Real data: sklearn `load_digits` (1797 images 8x8, offline in sklearn).

Two denoisers with a CLOSED-FORM optimal epsilon-prediction are used instead of a
trained U-Net, so every number in the lesson is exact and reproducible:
  * empirical mixture of 1797 delta-functions  -> exact posterior E[x0|xt] (memorises);
  * mixture of 10 class-Gaussians fitted to digits -> exact score, generates new samples,
    supports classifier-free guidance.
Every quoted number is computed here and asserted.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "86"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "86"

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


def fact(key, value):
    FACTS[key] = float(value)
    return value


# ----------------------------------------------------------------- schedules
T = 1000


def linear_schedule(T=T):
    beta = np.linspace(1e-4, 0.02, T)
    ab = np.cumprod(1.0 - beta)
    return beta, np.concatenate([[1.0], ab])          # ab[0] = 1


def cosine_schedule(T=T, s=0.008):
    t = np.arange(T + 1) / T
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    ab = f / f[0]
    beta = np.clip(1 - ab[1:] / ab[:-1], 0, 0.999)
    ab2 = np.concatenate([[1.0], np.cumprod(1 - beta)])
    return beta, ab2


BETA_L, AB_L = linear_schedule()
BETA_C, AB_C = cosine_schedule()


def snr(ab):
    with np.errstate(divide="ignore"):
        return ab / (1 - ab)


def cross_step(ab):
    """first t with SNR_t < 1, i.e. noise power exceeds signal power."""
    s = snr(ab[1:])
    return int(np.argmax(s < 1.0)) + 1


# ----------------------------------------------------------------- data
DIG = load_digits()
X_ALL = DIG.data / 8.0 - 1.0            # to [-1, 1]
Y_ALL = DIG.target
N_ALL, DIM = X_ALL.shape
rng_split = np.random.default_rng(8600)
perm = rng_split.permutation(N_ALL)
TR, TE = perm[:1400], perm[1400:]
XTR, YTR = X_ALL[TR], Y_ALL[TR]
XTE, YTE = X_ALL[TE], Y_ALL[TE]


def img(v):
    return (v.reshape(8, 8) + 1) / 2


# ----------------------------------------------------------------- class-Gaussian model
RIDGE = 0.05


class GaussMix:
    """Mixture of 10 class Gaussians with a shared (pooled) covariance.

    Shared covariance makes the classifier-free guidance analysis exact: the guided
    mean is mu_u + w (mu_c - mu_u) and the guided covariance shrinks with w.
    """

    def __init__(self, X, y, ridge=RIDGE):
        self.classes = np.arange(10)
        mus, pis = [], []
        S = np.zeros((DIM, DIM))
        for c in self.classes:
            Z = X[y == c]
            m = Z.mean(0)
            mus.append(m); pis.append(len(Z) / len(X))
            S += (Z - m).T @ (Z - m)
        S = S / len(X) + ridge * np.eye(DIM)
        lam, U = np.linalg.eigh(S)
        self.mu = np.array(mus); self.pi = np.array(pis)
        self.eig = np.tile(lam, (10, 1)); self.vec = np.tile(U, (10, 1, 1))

    def _comp(self, c, x, ab):
        """log-density and score of N(sqrt(ab) mu_c, ab S_c + (1-ab) I) at rows x."""
        d = x - np.sqrt(ab) * self.mu[c]
        lam = ab * self.eig[c] + (1 - ab)
        p = d @ self.vec[c]                      # rotate to eigenbasis
        quad = (p ** 2 / lam).sum(1)
        logp = -0.5 * quad - 0.5 * np.log(lam).sum() - 0.5 * DIM * np.log(2 * np.pi)
        score = -(p / lam) @ self.vec[c].T
        return logp, score

    def score(self, x, ab, cls=None):
        x = np.atleast_2d(x)
        comps = self.classes if cls is None else [cls]
        L = np.zeros((len(comps), len(x))); S = np.zeros((len(comps), len(x), DIM))
        for i, c in enumerate(comps):
            lp, sc = self._comp(c, x, ab)
            L[i] = lp + (np.log(self.pi[c]) if cls is None else 0.0)
            S[i] = sc
        L -= L.max(0, keepdims=True)
        W = np.exp(L); W /= W.sum(0, keepdims=True)
        return (W[:, :, None] * S).sum(0)


GM = GaussMix(XTR, YTR)


def eps_from_score(score, ab):
    return -np.sqrt(1 - ab) * score


def sample_chain(model, n, steps, ab_tab, rng, cls=None, w=1.0, stochastic=True,
                 record=False, x_init=None):
    """Ancestral (or DDIM-deterministic) reverse chain on a subsequence of steps."""
    ts = np.unique(np.round(np.linspace(T, 1, steps)).astype(int))[::-1]
    x = rng.standard_normal((n, DIM)) if x_init is None else x_init.copy()
    traj = [x.copy()] if record else None
    for i, t in enumerate(ts):
        ab_t = ab_tab[t]
        s_next = ts[i + 1] if i + 1 < len(ts) else 0
        ab_s = ab_tab[s_next]
        if cls is None or w == 1.0:
            sc = model.score(x, ab_t, cls=cls)
            eps = eps_from_score(sc, ab_t)
        else:
            eu = eps_from_score(model.score(x, ab_t, cls=None), ab_t)
            ec = eps_from_score(model.score(x, ab_t, cls=cls), ab_t)
            eps = eu + w * (ec - eu)
        x0h = (x - np.sqrt(1 - ab_t) * eps) / np.sqrt(ab_t)
        x0h = np.clip(x0h, -1.2, 1.2)
        if stochastic and s_next > 0:
            sig = np.sqrt((1 - ab_s) / (1 - ab_t) * (1 - ab_t / ab_s))
            dirn = np.sqrt(max(1 - ab_s - sig ** 2, 0.0))
            x = np.sqrt(ab_s) * x0h + dirn * eps + sig * rng.standard_normal((n, DIM))
        else:
            x = np.sqrt(ab_s) * x0h + np.sqrt(1 - ab_s) * eps
        if record:
            traj.append(x.copy())
    return (x, np.array(traj), ts) if record else x


# ----------------------------------------------------------------- fig 1
def fig_forward():
    ab = AB_C
    x0 = XTR[np.where(YTR == 3)[0][0]]
    shots = [0, 100, 250, 450, 700, 1000]
    rng = np.random.default_rng(8601)
    eps = rng.standard_normal(DIM)
    fig = plt.figure(figsize=(9.6, 6.4))
    gs = fig.add_gridspec(2, len(shots), height_ratios=[1.0, 1.5], hspace=0.35)
    for k, t in enumerate(shots):
        a = ab[t]
        xt = np.sqrt(a) * x0 + np.sqrt(1 - a) * eps
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(img(xt), cmap="gray", vmin=-0.4, vmax=1.4)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"$t={t}$\nSNR={a/(1-a):.2f}" if t else f"$t=0$\nоригинал",
                     fontsize=10, color=INK)
    ax = fig.add_subplot(gs[1, :])
    tt = np.arange(1, T + 1)
    ax.plot(tt, snr(AB_L[1:]), color=BLUE, lw=2.2, label="linear")
    ax.plot(tt, snr(AB_C[1:]), color=RED, lw=2.2, label="cosine")
    ax.axhline(1.0, color=MUTED, lw=0.9, ls=(0, (3, 3)))
    ax.set_yscale("log"); ax.set_ylim(1e-4, 1e4)
    for t in shots[1:]:
        ax.axvline(t, color=LINE, lw=0.7)
    cl, cc = cross_step(AB_L), cross_step(AB_C)
    ax.plot([cl], [1.0], "o", color=BLUE, ms=7); ax.plot([cc], [1.0], "o", color=RED, ms=7)
    ax.annotate(f"SNR=1 при t={cl}", (cl, 1.0), textcoords="offset points",
                xytext=(8, 26), color=BLUE, fontsize=10)
    ax.annotate(f"SNR=1 при t={cc}", (cc, 1.0), textcoords="offset points",
                xytext=(6, -30), color=RED, fontsize=10)
    ax.set_xlabel("шаг $t$"); ax.set_ylabel(r"$\mathrm{SNR}_t=\bar\alpha_t/(1-\bar\alpha_t)$")
    ax.set_title("Расписание решает, на каких задачах учится denoiser", fontsize=13)
    ax.legend(frameon=False, fontsize=11, loc="upper right")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "forward-schedule.png")
    fact("cross_linear", cl); fact("cross_cosine", cc)
    fact("abar_T_linear", AB_L[T]); fact("abar_T_cosine", AB_C[T])
    fact("frac_dead_linear", np.mean(snr(AB_L[1:]) < 0.01))
    fact("frac_dead_cosine", np.mean(snr(AB_C[1:]) < 0.01))
    assert cl < cc, (cl, cc)
    assert AB_L[T] < 5e-5 and AB_C[T] < 5e-5


# ----------------------------------------------------------------- fig 2
def posterior_weights(xt, ab, X):
    d2 = ((xt[None, :] - np.sqrt(ab) * X) ** 2).sum(1)
    lg = -d2 / (2 * (1 - ab))
    lg -= lg.max()
    w = np.exp(lg)
    return w / w.sum()


def fig_posterior():
    ab = AB_C
    rng = np.random.default_rng(8602)
    shots = [50, 200, 400, 600, 800]
    x0 = XTR[np.where(YTR == 3)[0][1]]
    eps = rng.standard_normal(DIM)
    neff_shot = {}
    fig = plt.figure(figsize=(10.2, 6.6))
    gs = fig.add_gridspec(3, len(shots), height_ratios=[1, 1, 1.7], hspace=0.35)
    for k, t in enumerate(shots):
        a = ab[t]
        xt = np.sqrt(a) * x0 + np.sqrt(1 - a) * eps
        w = posterior_weights(xt, a, XTR)
        x0h = w @ XTR
        neff = float(np.exp(-(w * np.log(w + 1e-300)).sum()))
        neff_shot[t] = neff
        a1 = fig.add_subplot(gs[0, k]); a1.imshow(img(xt), cmap="gray", vmin=-0.4, vmax=1.4)
        a1.set_xticks([]); a1.set_yticks([]); a1.set_title(f"$x_t$, $t={t}$", fontsize=10)
        a2 = fig.add_subplot(gs[1, k]); a2.imshow(img(x0h), cmap="gray", vmin=-0.4, vmax=1.4)
        a2.set_xticks([]); a2.set_yticks([])
        a2.set_xlabel(f"$n_{{\\mathrm{{eff}}}}={neff:.0f}$", fontsize=10, color=MUTED)
        if k == 0:
            a2.set_ylabel(r"$\widehat x_0$", fontsize=11)
    ts = np.arange(20, 1001, 20)
    curve = []
    for t in ts:
        a = ab[t]
        vals = []
        for j in range(24):
            src = XTR[rng.integers(len(XTR))]
            xt = np.sqrt(a) * src + np.sqrt(1 - a) * rng.standard_normal(DIM)
            w = posterior_weights(xt, a, XTR)
            vals.append(np.exp(-(w * np.log(w + 1e-300)).sum()))
        curve.append(np.mean(vals))
    curve = np.array(curve)
    ax = fig.add_subplot(gs[2, :])
    ax.plot(ts, curve, color=VIOLET, lw=2.4)
    ax.axhline(len(XTR), color=MUTED, lw=0.9, ls=(0, (3, 3)))
    ax.text(30, len(XTR) * 0.62, f"вся выборка, {len(XTR)}", color=MUTED, fontsize=10)
    ax.set_yscale("log"); ax.set_xlabel("шаг $t$")
    ax.set_ylabel("$n_{\\mathrm{eff}}$ — сколько картинок\nреально усредняется")
    ax.set_title("Оптимальный одношаговый ответ — среднее по всё большему числу картинок",
                 fontsize=13)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "posterior-average.png")
    for t, v in neff_shot.items():
        fact(f"neff_t{t}", v)
    fact("neff_curve_t100", curve[np.searchsorted(ts, 100)])
    fact("neff_curve_t1000", curve[-1])
    assert neff_shot[50] < 1.5 < neff_shot[800]
    assert curve[-1] > 800


# ----------------------------------------------------------------- fig 3 (2D score field)
def fig_score_field():
    """PCA-2D of real digits 0 and 1: forward densities and reverse score field."""
    mask = np.isin(YTR, [0, 1])
    Z = XTR[mask]; lab = YTR[mask]
    Zc = Z - Z.mean(0)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    P = Zc @ Vt[:2].T
    P = P / P.std(0)                              # unit-variance 2D real data
    ts = [0, 200, 500, 900]
    ab = AB_C
    g = np.linspace(-3.4, 3.4, 160)
    GX, GY = np.meshgrid(g, g)
    pts = np.stack([GX.ravel(), GY.ravel()], 1)
    fig, axes = plt.subplots(1, len(ts), figsize=(12.4, 3.5))
    seps = []
    for k, t in enumerate(ts):
        a = ab[t]
        d2 = ((pts[:, None, :] - np.sqrt(a) * P[None, :, :]) ** 2).sum(2)
        var = max(1 - a, 1e-6)
        lg = -d2 / (2 * var)
        m = lg.max(1, keepdims=True)
        wgt = np.exp(lg - m)
        dens = (m.ravel() + np.log(wgt.sum(1)))
        ax = axes[k]
        if t > 0:
            ax.imshow(dens.reshape(GX.shape), extent=[-3.4, 3.4, -3.4, 3.4], origin="lower",
                      cmap="bone_r", alpha=0.85, aspect="auto")
        if t == 0:
            ax.scatter(P[lab == 0, 0], P[lab == 0, 1], s=6, color=BLUE)
            ax.scatter(P[lab == 1, 0], P[lab == 1, 1], s=6, color=RED)
        else:
            wn = wgt / wgt.sum(1, keepdims=True)
            sc = ((np.sqrt(a) * P[None, :, :] - pts[:, None, :]) / var * wn[:, :, None]).sum(1)
            step = 12
            idx = np.arange(len(pts)).reshape(GX.shape)[::step, ::step].ravel()
            ax.quiver(pts[idx, 0], pts[idx, 1], sc[idx, 0], sc[idx, 1], color=GREEN,
                      width=0.005, scale=28 / max(var, 0.08))
        mu0 = P[lab == 0].mean(0); mu1 = P[lab == 1].mean(0)
        seps.append(float(np.sqrt(a) * np.linalg.norm(mu0 - mu1) / np.sqrt(a + var)))
        ax.set_title(f"$t={t}$" + ("  реальные данные" if t == 0 else
                     f"  $\\bar\\alpha={a:.2f}$, разделимость {seps[-1]:.2f}"), fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-3.4, 3.4); ax.set_ylim(-3.4, 3.4)
    fig.suptitle("Прямой процесс стирает моды, обратный — тянет к остаткам плотности", fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "score-field.png")
    fact("sep_t0", seps[0]); fact("sep_t200", seps[1])
    fact("sep_t500", seps[2]); fact("sep_t900", seps[3])
    assert seps[0] > seps[2] > seps[3]


# ----------------------------------------------------------------- fig 4 (who locks first)
def fig_lockin():
    rng = np.random.default_rng(8604)
    n, steps = 64, 200
    x, traj, ts = sample_chain(GM, n, steps, AB_C, rng, record=True)
    Xc = XTR - XTR.mean(0)
    lam, V = np.linalg.eigh(np.cov(Xc.T))
    order = np.argsort(lam)[::-1]
    lam, V = lam[order], V[:, order]
    comps = [0, 1, 4, 15, 40]
    lock = {}
    tgrid = np.concatenate([[T], ts])
    for c in comps:
        pj = (traj - XTR.mean(0)) @ V[:, c]                # (steps+1, n)
        fin = pj[-1]
        sd = np.sqrt(lam[c])
        dev = np.abs(pj - fin[None, :]) / sd
        st = []
        for i in range(n):
            ok = np.where(dev[:, i] < 0.5)[0]
            k = len(dev) - 1
            for j in ok:
                if np.all(dev[j:, i] < 0.5):
                    k = j; break
            st.append(tgrid[min(k, len(tgrid) - 1)])
        lock[c] = float(np.median(st))
    theo = {c: cross_step_lambda(lam[c]) for c in comps}
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    for c, col in zip(comps, [BLUE, GREEN, GOLD, VIOLET, RED]):
        pj = (traj - XTR.mean(0)) @ V[:, c]
        ax.plot(tgrid, pj[:, 0] / np.sqrt(lam[c]), color=col, lw=1.9,
                label=f"компонента {c + 1} ($\\lambda={lam[c]:.2f}$)")
    ax.invert_xaxis()
    ax.set_xlabel("шаг $t$ (обратный ход: справа налево)")
    ax.set_ylabel("координата / с.к.о.")
    ax.set_ylim(-5.5, 5.5)
    ax.set_title("Крупные направления фиксируются раньше мелких", fontsize=13)
    ax.legend(frameon=False, fontsize=9.0, loc="lower right", ncol=2)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax2 = axes[1]
    xs = np.arange(len(comps))
    ax2.bar(xs - 0.19, [lock[c] for c in comps], width=0.36, color=BLUE, label="замер по траекториям")
    ax2.bar(xs + 0.19, [theo[c] for c in comps], width=0.36, color=GOLD, label=r"теория: $\bar\alpha\lambda=1-\bar\alpha$")
    ax2.set_xticks(xs); ax2.set_xticklabels([f"№{c + 1}" for c in comps])
    ax2.set_ylabel("шаг $t$, на котором координата села")
    ax2.set_title("Порядок рождения деталей предсказуем", fontsize=13)
    ax2.legend(frameon=False, fontsize=9.5)
    ax2.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax2.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "lock-in.png")
    fact("lam1", lam[0]); fact("lam41", lam[40])
    fact("lock_c1", lock[0]); fact("lock_c41", lock[40])
    fact("theo_c1", theo[0]); fact("theo_c41", theo[40])
    assert lock[0] > lock[40] and theo[0] > theo[40]


def cross_step_lambda(lmb, ab=None):
    """t at which per-direction SNR = ab*lmb/(1-ab) drops below 1."""
    ab = AB_C if ab is None else ab
    s = ab[1:] * lmb / (1 - ab[1:])
    idx = np.argmax(s < 1.0)
    return float(idx + 1)


# ----------------------------------------------------------------- fig 5 (steps)
def fig_steps():
    rng = np.random.default_rng(8605)
    n = 96
    x_init = rng.standard_normal((n, DIM))
    ref = sample_chain(GM, n, 500, AB_C, np.random.default_rng(1), stochastic=False,
                       x_init=x_init)
    Ks = [3, 5, 10, 25, 50, 100, 200]
    dist, secs = [], []
    for K in Ks:
        t0 = time.perf_counter()
        xs = sample_chain(GM, n, K, AB_C, np.random.default_rng(1), stochastic=False,
                          x_init=x_init)
        secs.append(time.perf_counter() - t0)
        dist.append(float(np.mean(np.linalg.norm(xs - ref, axis=1))))
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    ax = axes[0]
    ax.plot(Ks, dist, "o-", color=RED, lw=2.2, ms=6)
    ax.set_xscale("log"); ax.set_xlabel("число шагов $K$ (DDIM, тот же $x_T$)")
    ax.set_ylabel("расстояние до эталона в 500 шагов")
    ax.set_title("Ускорение стоит точности — но не линейно", fontsize=13)
    for K, d in zip(Ks, dist):
        ax.annotate(f"{d:.2f}", (K, d), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9, color=MUTED)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax2 = axes[1]
    ax2.plot(Ks, np.array(secs) * 1000, "o-", color=BLUE, lw=2.2, ms=6)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("число шагов $K$"); ax2.set_ylabel("время генерации 96 образцов, мс")
    ax2.set_title("Цена растёт строго линейно по шагам", fontsize=13)
    ax2.grid(True, color=GRID, lw=0.4, alpha=0.5); ax2.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "steps-cost.png")
    for K, d in zip(Ks, dist):
        fact(f"dist_K{K}", d)
    fact("ratio_K10_K100", dist[Ks.index(10)] / dist[Ks.index(100)])
    assert dist[0] > dist[-1]
    assert dist[Ks.index(50)] < 0.5 * dist[Ks.index(5)]


# ----------------------------------------------------------------- fig 6 (guidance)
def fig_guidance():
    clf = LogisticRegression(max_iter=4000, C=1.0).fit(XTR, YTR)
    acc = float(clf.score(XTE, YTE))
    ws = [0.0, 1.0, 2.0, 4.0, 8.0]
    assert ws[0] == 0.0
    target = 3
    align, diver = [], []
    grids = {}
    for w in ws:
        rng = np.random.default_rng(8606)
        xs = sample_chain(GM, 120, 120, AB_C, rng, cls=target, w=max(w, 1e-9))
        pred = clf.predict(np.clip(xs, -1, 1))
        align.append(float(np.mean(pred == target)))
        d = np.linalg.norm(xs[:, None, :] - xs[None, :, :], axis=2)
        iu = np.triu_indices(len(xs), 1)
        diver.append(float(d[iu].mean()))
        grids[w] = xs[:6]
    fig = plt.figure(figsize=(11.6, 6.4))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.35, 1], hspace=0.3)
    ax = fig.add_subplot(gs[0, :3])
    ax.plot(ws, align, "o-", color=BLUE, lw=2.2, ms=6, label="доля попаданий в класс")
    ax.set_xlabel("сила guidance $w$"); ax.set_ylabel("соответствие условию", color=BLUE)
    ax.set_ylim(0, 1.05)
    axb = ax.twinx()
    axb.plot(ws, diver, "s--", color=RED, lw=2.2, ms=6)
    axb.set_ylabel("среднее попарное расстояние", color=RED)
    axb.spines["right"].set_color(LINE)
    ax.set_title("Точнее по условию — беднее по разнообразию", fontsize=13)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax2 = fig.add_subplot(gs[0, 3:])
    ax2.plot(diver, align, "-", color=MUTED, lw=1.2)
    ax2.scatter(diver, align, s=60, c=[BLUE, GREEN, GOLD, VIOLET, RED], zorder=5)
    for w, d, a in zip(ws, diver, align):
        ax2.annotate(f"w={w:g}", (d, a), textcoords="offset points", xytext=(6, -12),
                     fontsize=10, color=MUTED)
    ax2.set_xlabel("разнообразие"); ax2.set_ylabel("соответствие")
    ax2.set_title("Кривая компромисса, а не шкала «качества»", fontsize=13)
    ax2.grid(True, color=GRID, lw=0.4, alpha=0.5); ax2.set_axisbelow(True)
    inner = gs[1, :].subgridspec(2, 6, hspace=0.05, wspace=0.05)
    for r, w in enumerate([1.0, 8.0]):
        for k in range(6):
            a = fig.add_subplot(inner[r, k])
            a.imshow(img(np.clip(grids[w][k], -1, 1)), cmap="gray", vmin=-0.2, vmax=1.2)
            a.set_xticks([]); a.set_yticks([])
            if k == 0:
                a.set_ylabel(f"w={w:g}", fontsize=10, color=MUTED)
    save(fig, OUT / "guidance-tradeoff.png")
    fact("clf_acc", acc)
    for w, a, d in zip(ws, align, diver):
        fact(f"align_w{w:g}", a); fact(f"diver_w{w:g}", d)
    fact("align_peak", max(align)); fact("w_peak", ws[int(np.argmax(align))])
    assert acc > 0.93
    assert align[ws.index(2.0)] > 0.9 > align[ws.index(0.0)]
    assert all(diver[i + 1] < diver[i] for i in range(len(ws) - 1))
    assert align[ws.index(8.0)] < max(align)


# ----------------------------------------------------------------- sidenote images
def side_coeffs():
    ab = AB_C
    t = np.arange(T + 1)
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.plot(t, np.sqrt(ab), color=BLUE, lw=2.0, label=r"$\sqrt{\bar\alpha_t}$ — сигнал")
    ax.plot(t, np.sqrt(1 - ab), color=RED, lw=2.0, label=r"$\sqrt{1-\bar\alpha_t}$ — шум")
    i = int(np.argmin(np.abs(ab - 0.5)))
    ax.plot([i], [np.sqrt(ab[i])], "o", color=INK, ms=6)
    ax.annotate(f"поровну при t={i}", (i, np.sqrt(ab[i])), textcoords="offset points",
                xytext=(-8, 18), fontsize=9, color=MUTED, ha="right")
    ax.set_xlabel("шаг $t$", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, SIDE / "coefficients.png")
    fact("half_step", i)


def side_onejump():
    """One MSE jump from pure noise = the average digit; the chain gives a sample."""
    rng = np.random.default_rng(8607)
    a = AB_C[T]
    xt = rng.standard_normal((1, DIM))
    w = posterior_weights(xt[0], a, XTR)
    jump = w @ XTR
    mean_img = XTR.mean(0)
    chain = sample_chain(GM, 1, 200, AB_C, np.random.default_rng(11))[0]
    fig, axes = plt.subplots(1, 3, figsize=(4.6, 2.1))
    for ax, v, ttl in zip(axes, [mean_img, jump, chain],
                          ["среднее\nпо выборке", "один\nпрыжок", "цепочка\n200 шагов"]):
        ax.imshow(img(np.clip(v, -1, 1)), cmap="gray", vmin=-0.2, vmax=1.2)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl, fontsize=9)
    fig.tight_layout()
    save(fig, SIDE / "one-jump.png")
    dev = float(np.linalg.norm(jump - mean_img))
    rough_j = float(np.abs(np.diff(jump.reshape(8, 8), axis=1)).mean())
    rough_c = float(np.abs(np.diff(chain.reshape(8, 8), axis=1)).mean())
    fact("jump_to_mean", dev); fact("rough_jump", rough_j); fact("rough_chain", rough_c)
    fact("rough_real", float(np.abs(np.diff(XTR.reshape(-1, 8, 8), axis=2)).mean()))
    assert dev < 0.01 and rough_c > 1.4 * rough_j


def side_memorization():
    """Exact empirical score memorises; the limited-capacity model does not."""
    rng = np.random.default_rng(8608)
    n, K = 40, 120
    ts = np.unique(np.round(np.linspace(T, 1, K)).astype(int))[::-1]
    x = rng.standard_normal((n, DIM))
    for i, t in enumerate(ts):
        a = AB_C[t]
        s_next = ts[i + 1] if i + 1 < len(ts) else 0
        ab_s = AB_C[s_next]
        d2 = ((x[:, None, :] - np.sqrt(a) * XTR[None, :, :]) ** 2).sum(2)
        lg = -d2 / (2 * (1 - a)); lg -= lg.max(1, keepdims=True)
        wgt = np.exp(lg); wgt /= wgt.sum(1, keepdims=True)
        x0h = wgt @ XTR
        eps = (x - np.sqrt(a) * x0h) / np.sqrt(1 - a)
        if s_next > 0:
            sig = np.sqrt((1 - ab_s) / (1 - a) * (1 - a / ab_s))
            x = np.sqrt(ab_s) * x0h + np.sqrt(max(1 - ab_s - sig ** 2, 0)) * eps \
                + sig * rng.standard_normal((n, DIM))
        else:
            x = x0h
    d_emp = np.linalg.norm(x[:, None, :] - XTR[None, :, :], axis=2).min(1)
    xg = sample_chain(GM, n, K, AB_C, np.random.default_rng(8609))
    d_gm = np.linalg.norm(xg[:, None, :] - XTR[None, :, :], axis=2).min(1)
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    bins = np.linspace(0, max(d_gm.max(), 1.0) * 1.05, 26)
    ax.hist(d_emp, bins=bins, color=RED, alpha=0.75, label="точный score выборки")
    ax.hist(d_gm, bins=bins, color=BLUE, alpha=0.65, label="модель с ограниченной ёмкостью")
    ax.set_xlabel("расстояние до ближайшей обучающей картинки", fontsize=9)
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, SIDE / "memorization.png")
    fact("nn_emp", float(d_emp.mean())); fact("nn_gm", float(d_gm.mean()))
    assert d_emp.mean() < 0.2 * d_gm.mean()


# ----------------------------------------------------------------- misc facts
def extra_facts():
    ab = AB_C
    rng = np.random.default_rng(8610)
    # correlation between x_t and x_0: rho = sqrt(ab) s / sqrt(ab s^2 + 1 - ab)
    s2 = float(XTR.var())
    for t in [100, 300, 600, 900]:
        a = ab[t]
        idx = rng.choice(len(XTR), 400, replace=False)
        x0 = XTR[idx]
        xt = np.sqrt(a) * x0 + np.sqrt(1 - a) * rng.standard_normal(x0.shape)
        r = float(np.corrcoef(xt.ravel(), x0.ravel())[0, 1])
        rho = float(np.sqrt(a * s2) / np.sqrt(a * s2 + 1 - a))
        fact(f"corr_t{t}", r); fact(f"rho_t{t}", rho)
        fact(f"sqrt_ab_t{t}", float(np.sqrt(a)))
        assert abs(r - rho) < 0.03, (t, r, rho)
    fact("data_var", float(XTR.var()))
    fact("n_train", len(XTR)); fact("n_test", len(XTE)); fact("dim", DIM)
    fact("n_all", N_ALL)


def main():
    fig_forward()
    fig_posterior()
    fig_score_field()
    fig_lockin()
    fig_steps()
    fig_guidance()
    side_coeffs()
    side_onejump()
    side_memorization()
    extra_facts()
    (ROOT / "scripts" / "data").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "scripts" / "data" / "lesson86_facts.json", "w") as f:
        json.dump({k: round(v, 6) for k, v in FACTS.items()}, f, ensure_ascii=False, indent=1)
    for k in sorted(FACTS):
        print(f"{k:22s} {FACTS[k]:.4f}")
    print("lesson 86 figures OK")


if __name__ == "__main__":
    main()
