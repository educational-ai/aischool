"""Deterministic figures for lesson 82: GAN as a two-player game.

Real data:
  * scripts/data/bike-sharing-hour.csv  -- one-dimensional real distribution of hourly
    rides; two candidate "generators" (normal and lognormal fits) let us compute the
    optimal discriminator D* and the Jensen-Shannon divergence honestly;
  * sklearn load_digits -> PCA(2)      -- a real ten-mode two-dimensional distribution
    on which a small numpy GAN is actually trained (non-saturating loss, Adam).

Everything quoted in the lesson text is computed here and asserted.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

ROOT = Path(__file__).resolve().parents[1]
BIKE = ROOT / "scripts" / "data" / "bike-sharing-hour.csv"
OUT = ROOT / "public" / "figures" / "lessons" / "82"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "82"

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


def fact(key, value, fmt="{:.4f}"):
    FACTS[key] = float(value)
    print(f"  {key} = " + fmt.format(value))
    return value


def save(fig, path, *, dpi=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ------------------------------------------------------------------ real 1D data
def bike_counts() -> np.ndarray:
    vals = []
    with open(BIKE) as f:
        for row in csv.DictReader(f):
            vals.append(int(row["cnt"]))
    return np.array(vals, dtype=float)


def js_divergence(p, q, dx):
    """Jensen-Shannon divergence between two densities on a grid, in bits."""
    p = np.clip(p, 1e-12, None); q = np.clip(q, 1e-12, None)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m)) * dx
    kl_qm = np.sum(q * np.log2(q / m)) * dx
    return 0.5 * (kl_pm + kl_qm)


# ---------------------------------------------------- fig 82.1: optimal discriminator
def fig_optimal_d():
    print("fig 82.1 optimal discriminator on real bike data")
    cnt = bike_counts()
    n = len(cnt)
    fact("bike_n", n, "{:.0f}")
    fact("bike_mean", cnt.mean(), "{:.1f}")
    fact("bike_median", float(np.median(cnt)), "{:.1f}")
    fact("bike_std", cnt.std(ddof=1), "{:.1f}")

    grid = np.linspace(0, 800, 1601)
    dx = grid[1] - grid[0]

    # p_data: smoothed histogram (kernel width 12 rides) of the real sample
    h = 12.0
    counts, edges = np.histogram(cnt, bins=np.arange(0, 1001, 4))
    centers = 0.5 * (edges[:-1] + edges[1:])
    dens = counts / (counts.sum() * 4.0)
    kern = np.exp(-0.5 * ((grid[:, None] - centers[None, :]) / h) ** 2)
    kern /= (np.sqrt(2 * np.pi) * h)
    p_data = kern @ (dens * 4.0)
    p_data /= np.trapezoid(p_data, grid)

    # generator A: normal fit (moment matching)
    mu, sg = cnt.mean(), cnt.std(ddof=1)
    p_norm = np.exp(-0.5 * ((grid - mu) / sg) ** 2) / (np.sqrt(2 * np.pi) * sg)
    # generator B: lognormal fit on positive counts
    pos = cnt[cnt > 0]
    lm, ls = np.log(pos).mean(), np.log(pos).std(ddof=1)
    g = np.clip(grid, 1e-6, None)
    p_logn = np.exp(-0.5 * ((np.log(g) - lm) / ls) ** 2) / (g * np.sqrt(2 * np.pi) * ls)
    p_logn[grid <= 0] = 0.0

    js_norm = fact("js_norm_bits", js_divergence(p_data, p_norm, dx))
    js_logn = fact("js_logn_bits", js_divergence(p_data, p_logn, dx))
    assert js_norm > js_logn > 0
    v_norm = fact("V_norm", -2 * np.log(2) + 2 * js_norm * np.log(2))
    v_logn = fact("V_logn", -2 * np.log(2) + 2 * js_logn * np.log(2))
    assert v_norm > v_logn > -2 * np.log(2)

    d_norm = p_data / (p_data + p_norm + 1e-15)
    d_logn = p_data / (p_data + p_logn + 1e-15)
    i50 = int(np.argmin(np.abs(grid - 50)))
    i400 = int(np.argmin(np.abs(grid - 400)))
    fact("D_norm_at50", d_norm[i50], "{:.3f}")
    fact("D_norm_at400", d_norm[i400], "{:.3f}")
    fact("D_logn_at50", d_logn[i50], "{:.3f}")
    assert d_norm[i50] > 0.6 and d_norm[i400] < 0.45

    # the normal "generator" puts mass on impossible negative counts
    from math import erf
    p_neg = 0.5 * (1 + erf((0 - mu) / (sg * np.sqrt(2))))
    fact("norm_neg_mass", p_neg, "{:.3f}")
    assert 0.10 < p_neg < 0.20

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    ax = axes[0]
    ax.fill_between(grid, p_data, color=BLUE, alpha=0.20, lw=0)
    ax.plot(grid, p_data, color=BLUE, lw=2.2, label="$p_{data}$: реальный прокат")
    ax.plot(grid, p_norm, color=RED, lw=2.0, ls=(0, (5, 3)),
            label=f"$p_g$: нормальный (JS = {js_norm:.3f} бит)")
    ax.plot(grid, p_logn, color=GREEN, lw=2.0,
            label=f"$p_g$: логнормальный (JS = {js_logn:.3f} бит)")
    ax.set_xlabel("поездок за час"); ax.set_ylabel("плотность")
    ax.set_title("Два генератора против одних данных")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)

    ax = axes[1]
    ax.axhline(0.5, color=MUTED, lw=1.0, ls=":")
    ax.plot(grid, d_norm, color=RED, lw=2.2, label="$D^*$ против нормального")
    ax.plot(grid, d_logn, color=GREEN, lw=2.2, label="$D^*$ против логнормального")
    ax.scatter([50, 400], [d_norm[i50], d_norm[i400]], s=46, color=INK, zorder=5)
    ax.annotate(f"$D^*(50)={d_norm[i50]:.2f}$", (50, d_norm[i50]), textcoords="offset points",
                xytext=(14, -4), fontsize=10, color=INK)
    ax.annotate(f"$D^*(400)={d_norm[i400]:.2f}$", (400, d_norm[i400]), textcoords="offset points",
                xytext=(10, 10), fontsize=10, color=INK)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("поездок за час"); ax.set_ylabel("$D^*(x)$")
    ax.set_title("Оптимальный судья читает отношение плотностей")
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "optimal_d.png")


# ---------------------------------------------------- fig 82.2: saturation of the loss
def fig_saturation():
    print("fig 82.2 saturating vs non-saturating loss")
    d = np.linspace(0.001, 0.999, 999)
    l_sat = np.log(1 - d)
    l_ns = -np.log(d)
    g_sat = 1.0 / (1 - d)      # |d/dD log(1-D)|
    g_ns = 1.0 / d             # |d/dD (-log D)|
    d0 = 0.05
    fact("grad_sat_at005", 1 / (1 - d0), "{:.4f}")
    fact("grad_ns_at005", 1 / d0, "{:.4f}")
    ratio = fact("grad_ratio_at005", (1 / d0) / (1 / (1 - d0)), "{:.2f}")
    assert abs(ratio - 19.0) < 0.05
    # gradient through the sigmoid: D = sigma(s)
    s = np.linspace(-8, 8, 400)
    ds = 1 / (1 + np.exp(-s))
    gs_sat = ds                # |d/ds log(1-sigma(s))| = sigma(s)
    gs_ns = 1 - ds             # |d/ds (-log sigma(s))| = 1 - sigma(s)
    s0 = -4.0
    sig0 = 1 / (1 + np.exp(-s0))
    fact("logit_m4_D", sig0, "{:.4f}")
    fact("logit_m4_grad_sat", sig0, "{:.4f}")
    fact("logit_m4_grad_ns", 1 - sig0, "{:.4f}")
    fact("logit_m4_ratio", (1 - sig0) / sig0, "{:.1f}")
    assert (1 - sig0) / sig0 > 50

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))
    ax = axes[0]
    ax.plot(d, l_sat, color=RED, lw=2.2, label=r"$\log(1-D)$ — насыщающий")
    ax.plot(d, l_ns, color=GREEN, lw=2.2, label=r"$-\log D$ — non-saturating")
    ax.set_ylim(-6, 7); ax.set_xlabel("$D(G(z))$"); ax.set_ylabel("потеря генератора")
    ax.set_title("Две цели с одним и тем же желаемым концом")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)

    ax = axes[1]
    ax.plot(s, gs_sat, color=RED, lw=2.2, label="насыщающий")
    ax.plot(s, gs_ns, color=GREEN, lw=2.2, label="non-saturating")
    ax.axvline(s0, color=MUTED, lw=1.0, ls=":")
    ax.annotate(f"logit $=-4$: {sig0:.3f} против {1-sig0:.3f}", (s0, 0.5),
                textcoords="offset points", xytext=(10, 30), fontsize=10, color=INK)
    ax.set_xlabel("logit судьи на подделке"); ax.set_ylabel("модуль градиента по logit")
    ax.set_title("Кто ещё чувствует уклон, когда подделка очевидна")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "saturation.png")


# ---------------------------------------------------- fig 82.3: rotation in a bilinear game
def fig_rotation():
    print("fig 82.3 bilinear game rotation")

    def run(eta, steps=100, alternating=False):
        x, y = 1.0, 0.0
        xs, ys = [x], [y]
        for _ in range(steps):
            if alternating:
                x = x - eta * y
                y = y + eta * x
            else:
                x, y = x - eta * y, y + eta * x
            xs.append(x); ys.append(y)
        return np.array(xs), np.array(ys)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4))
    ax = axes[0]
    for eta, col in ((0.01, GREEN), (0.1, BLUE)):
        xs, ys = run(eta)
        r = float(np.hypot(xs[-1], ys[-1]))
        theory = (1 + eta ** 2) ** 50
        assert abs(r - theory) < 1e-6 * max(1.0, theory)
        fact(f"radius100_eta{str(eta).replace('.', '')}", r, "{:.4f}")
        ax.plot(xs, ys, color=col, lw=1.8, label=fr"$\eta={eta}$: $r_{{100}}={r:.2f}$")
        ax.scatter([xs[-1]], [ys[-1]], color=col, s=40, zorder=5)
    ax.scatter([0], [0], color=INK, s=50, zorder=6)
    ax.annotate("седловая точка $(0,0)$", (0, 0), textcoords="offset points", xytext=(8, -18),
                fontsize=10, color=INK)
    ax.scatter([1], [0], color=INK, s=28, zorder=6)
    ax.annotate("старт $(1,0)$", (1, 0), textcoords="offset points", xytext=(-6, 10),
                fontsize=10, color=MUTED)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ (минимизирует)"); ax.set_ylabel("$y$ (максимизирует)")
    ax.set_title("Одновременный шаг: не спуск, а раскручивание", fontsize=14)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)

    ax = axes[1]
    steps = np.arange(101)
    for eta, col in ((0.01, GREEN), (0.1, BLUE), (0.5, RED)):
        xs, ys = run(eta)
        r = np.hypot(xs, ys)
        fact(f"radius100_eta{str(eta).replace('.', '')}", float(r[-1]), "{:.4f}")
        ax.semilogy(steps, r, color=col, lw=2.0, label=fr"одновременный, $\eta={eta}$")
        xa, ya = run(eta, alternating=True)
        ra = np.hypot(xa, ya)
        fact(f"alt_radius100_eta{str(eta).replace('.', '')}", float(ra[-1]), "{:.4f}")
        ax.semilogy(steps, ra, color=col, lw=1.6, ls=(0, (4, 3)),
                    label=fr"поочерёдный, $\eta={eta}$")
    ax.set_xlabel("шаг"); ax.set_ylabel("расстояние до равновесия (лог. ось)")
    ax.set_title("Порядок ходов решает судьбу траектории", fontsize=14)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper left")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4, which="both"); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "rotation.png")
    assert FACTS["alt_radius100_eta01"] < 1.05
    assert FACTS["radius100_eta05"] > 1000


# ---------------------------------------------------- fig 82.5: JS saturates, W1 does not
def fig_js_vs_w1():
    print("fig 82.5 JS vs Wasserstein for a shifted narrow distribution")
    thetas = np.linspace(0, 3, 301)
    sigma = 0.1
    grid = np.linspace(-2, 5, 3001)
    dx = grid[1] - grid[0]
    p = np.exp(-0.5 * (grid / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
    js, w1 = [], []
    for th in thetas:
        q = np.exp(-0.5 * ((grid - th) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
        js.append(js_divergence(p, q, dx))
        w1.append(th)
    js = np.array(js); w1 = np.array(w1)
    i1 = int(np.argmin(np.abs(thetas - 1.0)))
    i2 = int(np.argmin(np.abs(thetas - 2.0)))
    fact("js_theta1_bits", js[i1], "{:.4f}")
    fact("js_theta2_bits", js[i2], "{:.4f}")
    fact("js_slope_theta1", (js[i1 + 1] - js[i1 - 1]) / (thetas[i1 + 1] - thetas[i1 - 1]), "{:.2e}")
    assert js[i1] > 0.99 and js[i2] > 0.99
    assert abs(FACTS["js_slope_theta1"]) < 1e-4
    fact("w1_theta1", w1[i1], "{:.2f}")
    fact("w1_theta2", w1[i2], "{:.2f}")

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(thetas, js, color=RED, lw=2.4, label="JS-дивергенция (бит)")
    ax.plot(thetas, w1, color=BLUE, lw=2.4, label=r"Расстояние $W_1=|\theta|$")
    ax.axhline(1.0, color=MUTED, lw=1.0, ls=":")
    ax.annotate("плато $1$ бит: градиент по $\\theta$ равен нулю", (1.6, 1.03),
                fontsize=10, color=RED)
    ax.set_xlabel(r"сдвиг генератора $\theta$"); ax.set_ylabel("значение меры")
    ax.set_title("Когда носители не пересекаются, JS перестаёт что-либо подсказывать")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    ax.set_ylim(0, 3.2)
    save(fig, OUT / "js_vs_w1.png")


# ---------------------------------------------------- the actual GAN on real 2D data
def digits_2d():
    """Real ten-mode 2D distribution: sklearn digits projected on two discriminant axes."""
    dg = load_digits()
    X = LinearDiscriminantAnalysis(n_components=2).fit(dg.data, dg.target).transform(dg.data)
    X = (X - X.mean(0)) / X.std(0)
    return X.astype(np.float64), dg.target


class Net:
    """Small MLP, LeakyReLU(0.2) hidden, linear output, Adam."""

    def __init__(self, sizes, rng, lr):
        self.W, self.b = [], []
        for a, b in zip(sizes[:-1], sizes[1:]):
            self.W.append(rng.normal(0, np.sqrt(2.0 / a), (a, b)))
            self.b.append(np.zeros(b))
        self.lr = lr
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(x) for x in self.b]
        self.vb = [np.zeros_like(x) for x in self.b]
        self.t = 0

    def forward(self, x):
        self.a = [x]
        h = x
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = h @ w + b
            if i < len(self.W) - 1:
                h = np.where(z > 0, z, 0.2 * z)
            else:
                h = z
            self.a.append(h)
        return h

    def backward(self, gout):
        gW = [None] * len(self.W); gb = [None] * len(self.b)
        g = gout
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = self.a[i].T @ g
            gb[i] = g.sum(0)
            g = g @ self.W[i].T
            if i > 0:
                z = self.a[i]
                g = g * np.where(z > 0, 1.0, 0.2)
        return gW, gb, g

    def step(self, gW, gb, b1=0.5, b2=0.999, eps=1e-8):
        self.t += 1
        for i in range(len(self.W)):
            self.mW[i] = b1 * self.mW[i] + (1 - b1) * gW[i]
            self.vW[i] = b2 * self.vW[i] + (1 - b2) * gW[i] ** 2
            self.mb[i] = b1 * self.mb[i] + (1 - b1) * gb[i]
            self.vb[i] = b2 * self.vb[i] + (1 - b2) * gb[i] ** 2
            mw = self.mW[i] / (1 - b1 ** self.t); vw = self.vW[i] / (1 - b2 ** self.t)
            mb = self.mb[i] / (1 - b1 ** self.t); vb = self.vb[i] / (1 - b2 ** self.t)
            self.W[i] -= self.lr * mw / (np.sqrt(vw) + eps)
            self.b[i] -= self.lr * mb / (np.sqrt(vb) + eps)


def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-np.clip(s, -60, 60)))


def train_gan(X, *, seed, lr_g=2e-4, lr_d=2e-4, steps=6000, batch=128, zdim=8,
              snapshots=(), noise_seed=777):
    rng = np.random.default_rng(seed)
    G = Net([zdim, 64, 64, 2], rng, lr_g)
    D = Net([2, 64, 64, 1], rng, lr_d)
    n = len(X)
    hist = {"step": [], "d_loss": [], "g_loss": [], "cov": []}
    snaps = {}
    srng = np.random.default_rng(noise_seed)
    fixed_z = srng.normal(size=(2000, zdim))
    for t in range(1, steps + 1):
        real = X[rng.integers(0, n, batch)]
        z = rng.normal(size=(batch, zdim))
        fake = G.forward(z)
        # discriminator
        sr = D.forward(real).ravel(); pr = sigmoid(sr)
        gr = ((pr - 1.0) / batch)[:, None]
        gWr, gbr, _ = D.backward(gr)
        sf = D.forward(fake).ravel(); pf = sigmoid(sf)
        gf = ((pf - 0.0) / batch)[:, None]
        gWf, gbf, _ = D.backward(gf)
        D.step([a + b for a, b in zip(gWr, gWf)], [a + b for a, b in zip(gbr, gbf)])
        d_loss = float(-np.mean(np.log(pr + 1e-12)) - np.mean(np.log(1 - pf + 1e-12)))
        # generator, non-saturating
        z = rng.normal(size=(batch, zdim))
        fake = G.forward(z)
        sf = D.forward(fake).ravel(); pf = sigmoid(sf)
        gs = ((pf - 1.0) / batch)[:, None]
        _, _, gx = D.backward(gs)
        gWg, gbg, _ = G.backward(gx)
        G.step(gWg, gbg)
        g_loss = float(-np.mean(np.log(pf + 1e-12)))
        if t % 100 == 0 or t == 1:
            samp = G.forward(fixed_z)
            hist["step"].append(t); hist["d_loss"].append(d_loss); hist["g_loss"].append(g_loss)
            hist["cov"].append(samp)
        if t in snapshots:
            snaps[t] = G.forward(fixed_z).copy()
    return G, hist, snaps, fixed_z


def mode_stats(samples, centers, radius):
    """Share of samples inside each mode's radius; a mode counts as covered at >=1%."""
    d = np.linalg.norm(samples[:, None, :] - centers[None, :, :], axis=2)
    near = d.argmin(1); dm = d.min(1)
    inside = dm <= radius
    share = np.array([np.mean(inside & (near == k)) for k in range(len(centers))])
    return share, float(np.mean(~inside))


def prec_recall(real, fake, k=5):
    """Kynkaanniemi-style improved precision/recall with k-NN manifolds."""
    def radii(A):
        d = np.linalg.norm(A[:, None, :] - A[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        return np.sort(d, axis=1)[:, k - 1]
    rr = radii(real); rf = radii(fake)
    d_fr = np.linalg.norm(fake[:, None, :] - real[None, :, :], axis=2)
    precision = float(np.mean(np.any(d_fr <= rr[None, :], axis=1)))
    recall = float(np.mean(np.any(d_fr.T <= rf[None, :], axis=1)))
    return precision, recall


def fig_gan_real():
    print("fig 82.4 + 82.6: GAN actually trained on real digits (LDA-2D)")
    X, y = digits_2d()
    fact("digits_n", len(X), "{:.0f}")
    centers = np.array([X[y == k].mean(0) for k in range(10)])
    intra = np.concatenate([np.linalg.norm(X[y == k] - centers[k], axis=1) for k in range(10)])
    radius = float(np.median(intra))
    fact("mode_radius", radius, "{:.3f}")

    snaps = (200, 1500, 6000)
    # run A: latent space of dimension 8 -- enough room for ten modes
    _, histA, snapA, _ = train_gan(X, seed=82, zdim=8, steps=6000, snapshots=snaps)
    # run B: latent space of dimension 1 -- a curve cannot cover ten islands at once
    _, histB, snapB, _ = train_gan(X, seed=82, zdim=1, steps=6000, snapshots=snaps)

    covA = np.array([(mode_stats(s, centers, radius)[0] >= 0.01).sum() for s in histA["cov"]])
    covB = np.array([(mode_stats(s, centers, radius)[0] >= 0.01).sum() for s in histB["cov"]])
    steps = np.array(histA["step"])
    fact("covA_final", covA[-1], "{:.0f}")
    fact("covA_min_after1000", covA[steps >= 1000].min(), "{:.0f}")
    fact("covB_max", covB.max(), "{:.0f}")
    fact("covB_min", covB.min(), "{:.0f}")
    fact("covB_final", covB[-1], "{:.0f}")
    fact("covB_mean", covB.mean(), "{:.1f}")
    assert covA[-1] == 10 and covB.max() < 10
    for t in snaps:
        shA, outA = mode_stats(snapA[t], centers, radius)
        shB, outB = mode_stats(snapB[t], centers, radius)
        fact(f"covA_step{t}", int((shA >= 0.01).sum()), "{:.0f}")
        fact(f"covB_step{t}", int((shB >= 0.01).sum()), "{:.0f}")
        fact(f"outB_step{t}", outB, "{:.3f}")

    # how many modes are lost and later regained: forgetting, not monotone progress
    MB = np.array([mode_stats(s, centers, radius)[0] for s in histB["cov"]]).T
    onB = MB >= 0.01
    revived = 0
    for k in range(10):
        seq = onB[k]
        for t in range(1, len(seq) - 1):
            if seq[t - 1] and not seq[t] and seq[t + 1:].any():
                revived += 1
                break
    fact("modes_lost_and_regained", revived, "{:.0f}")
    fact("covB_drops", int(np.sum(np.diff(covB) < 0)), "{:.0f}")
    assert revived >= 3

    real_eval = X[np.random.default_rng(5).integers(0, len(X), 600)]
    fakeA = snapA[6000][:600]
    fakeB = snapB[6000][:600]
    pA, rA = prec_recall(real_eval, fakeA)
    pB, rB = prec_recall(real_eval, fakeB)
    fact("prec_A", pA, "{:.3f}"); fact("rec_A", rA, "{:.3f}")
    fact("prec_B", pB, "{:.3f}"); fact("rec_B", rB, "{:.3f}")
    assert rB < rA

    # ---- figure 82.4: three snapshots of the collapsing run + coverage heatmap
    fig = plt.figure(figsize=(12.4, 7.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.42, wspace=0.16)
    for j, t in enumerate(snaps):
        ax = fig.add_subplot(gs[0, j])
        ax.scatter(X[:, 0], X[:, 1], s=6, color=BLUE, alpha=0.16, lw=0)
        S = snapB[t]
        ax.scatter(S[:800, 0], S[:800, 1], s=7, color=RED, alpha=0.6, lw=0)
        ax.scatter(centers[:, 0], centers[:, 1], s=30, color=INK, marker="x")
        sh, out = mode_stats(snapB[t], centers, radius)
        ax.set_title(f"шаг {t}: мод покрыто {int((sh>=0.01).sum())} из 10", fontsize=13)
        ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"вне мод: {out*100:.0f}% выборки", fontsize=10)
    ax = fig.add_subplot(gs[1, :])
    im = ax.imshow(MB, aspect="auto", origin="lower", cmap="magma_r",
                   extent=[0, 6000, -0.5, 9.5], vmin=0, vmax=0.35)
    ax.set_yticks(range(10)); ax.set_yticklabels([str(k) for k in range(10)], fontsize=9)
    ax.set_xlabel("шаг обучения"); ax.set_ylabel("мода (цифра)")
    ax.set_title("Доля выборки в каждой моде: режимы гаснут и возвращаются", fontsize=13)
    cb = fig.colorbar(im, ax=ax, pad=0.01); cb.outline.set_edgecolor(LINE)
    cb.ax.tick_params(labelsize=9)
    save(fig, OUT / "gan_modes.png")

    # ---- figure 82.6: fidelity and coverage are two different axes
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.4))
    for ax, (S, nm, pr, rc) in zip(axes[:2],
                                   [(fakeA, "латент $z\\in\\mathbb{R}^8$", pA, rA),
                                    (fakeB, "латент $z\\in\\mathbb{R}^1$", pB, rB)]):
        ax.scatter(X[:, 0], X[:, 1], s=7, color=BLUE, alpha=0.18, lw=0, label="реальные")
        ax.scatter(S[:, 0], S[:, 1], s=9, color=RED, alpha=0.6, lw=0, label="сгенерированные")
        ax.set_title(f"{nm}\nprecision {pr:.2f}, recall {rc:.2f}", fontsize=12)
        ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax = axes[2]
    ax.bar([0, 1], [pA, pB], width=0.34, color=GREEN, label="precision (реализм)")
    ax.bar([0.36, 1.36], [rA, rB], width=0.34, color=GOLD, label="recall (покрытие)")
    for xx, vv in [(0, pA), (1, pB), (0.36, rA), (1.36, rB)]:
        ax.annotate(f"{vv:.2f}", (xx, vv), ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks([0.18, 1.18]); ax.set_xticklabels(["$z$ из $\\mathbb{R}^8$", "$z$ из $\\mathbb{R}^1$"])
    ax.set_ylim(0, 1.15); ax.set_ylabel("доля")
    ax.set_title("Реализм почти не падает, покрытие рушится", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "precision_recall.png")

    # ---- sidenote: loss curves are not a scoreboard
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(histB["step"], histB["d_loss"], color=BLUE, lw=1.3, label="потеря судьи")
    ax.plot(histB["step"], histB["g_loss"], color=RED, lw=1.3, label="потеря генератора")
    ax.set_xlabel("шаг"); ax.set_ylabel("потеря")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    ax.set_title("Кривые потерь ничего не решают", fontsize=12)
    fig.tight_layout()
    save(fig, SIDE / "loss_curves.png")
    fact("d_loss_final", histB["d_loss"][-1], "{:.3f}")
    fact("g_loss_final", histB["g_loss"][-1], "{:.3f}")
    corr = float(np.corrcoef(np.array(histB["g_loss"]), covB)[0, 1])
    fact("corr_gloss_cov", corr, "{:.3f}")
    assert abs(corr) < 0.8

    # ---- sidenote: who disappeared from the sample
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    real_share = np.array([np.mean(y == k) for k in range(10)])
    gen_share = mode_stats(snapB[6000], centers, radius)[0]
    gen_share = gen_share / max(gen_share.sum(), 1e-9)
    ax.bar(np.arange(10) - 0.19, real_share, width=0.38, color=BLUE, label="реальные")
    ax.bar(np.arange(10) + 0.19, gen_share, width=0.38, color=RED, label="генератор")
    ax.set_xticks(range(10)); ax.set_xlabel("мода (цифра)"); ax.set_ylabel("доля")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    ax.set_title("Кто пропал из выборки", fontsize=12)
    fig.tight_layout()
    save(fig, SIDE / "mode_shares.png")
    fact("gen_share_max", gen_share.max(), "{:.3f}")
    fact("gen_share_argmax", int(gen_share.argmax()), "{:.0f}")
    fact("gen_modes_below_1pct", int(np.sum(gen_share < 0.01)), "{:.0f}")
    fact("real_share_min", real_share.min(), "{:.3f}")

    # ---- sidenote: memorization test
    rng = np.random.default_rng(11)
    idx = rng.permutation(len(X))
    tr, ho = X[idx[:1200]], X[idx[1200:]]
    S = snapA[6000][:800]
    d_tr = np.linalg.norm(S[:, None, :] - tr[None, :, :], axis=2).min(1)
    d_ho = np.linalg.norm(S[:, None, :] - ho[None, :, :], axis=2).min(1)
    fact("nn_train_median", float(np.median(d_tr)), "{:.3f}")
    fact("nn_hold_median", float(np.median(d_ho)), "{:.3f}")
    fact("nn_ratio", float(np.median(d_ho) / np.median(d_tr)), "{:.2f}")
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    bins = np.linspace(0, float(max(d_ho.max(), d_tr.max())), 26)
    ax.hist(d_tr, bins=bins, color=BLUE, alpha=0.55, label="до обучающих")
    ax.hist(d_ho, bins=bins, color=GOLD, alpha=0.55, label="до отложенных")
    ax.set_xlabel("расстояние до ближайшего реального"); ax.set_ylabel("число выборок")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    ax.set_title("Тест на запоминание", fontsize=12)
    fig.tight_layout()
    save(fig, SIDE / "memorization.png")


def widget_facts():
    """Numbers behind the interactive lab (model example: two modes, hand-set)."""
    print("widget model example (gan-game-lab)")
    N, xmin, xmax = 481, -7.0, 7.0
    x = np.linspace(xmin, xmax, N); dx = x[1] - x[0]
    npdf = lambda t, m, s: np.exp(-0.5 * ((t - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    p = 0.6 * npdf(x, -2, 0.5) + 0.4 * npdf(x, 2, 0.5)
    eps = 1e-7

    def gen(mu, sg, w2):
        return (1 - w2) * npdf(x, mu, sg) + w2 * npdf(x, 2, sg)

    def js(q):
        a = np.maximum(p, eps); b = np.maximum(q, eps); m = 0.5 * (a + b)
        return float(np.sum(0.5 * (a * np.log2(a / m) + b * np.log2(b / m))) * dx)

    def w1(q):
        return float(np.sum(np.abs(np.cumsum(p) * dx - np.cumsum(q) * dx)) * dx)

    def frac(base, other):
        tau = 0.02 * base.max()
        return float(np.sum(other[base >= tau]) * dx / (np.sum(other) * dx))

    def gloss(mu, sg, w2, kind):
        q = np.maximum(gen(mu, sg, w2), eps)
        d = np.maximum(p, eps) / (np.maximum(p, eps) + q)
        v = -np.log(d) if kind == "ns" else np.log(np.maximum(1 - d, eps))
        return float(np.sum(v * q) * dx)

    def grad(mu, sg, w2, kind, h=0.02):
        return abs((gloss(mu + h, sg, w2, kind) - gloss(mu - h, sg, w2, kind)) / (2 * h))

    q_far = gen(4.5, 0.5, 0.0)
    fact("wid_js_far", js(q_far), "{:.3f}")
    fact("wid_w1_far", w1(q_far), "{:.2f}")
    fact("wid_grad_ns_far", grad(4.5, 0.5, 0.0, "ns"), "{:.2f}")
    fact("wid_grad_sat_far", grad(4.5, 0.5, 0.0, "sat"), "{:.3f}")
    fact("wid_grad_ratio_far", grad(4.5, 0.5, 0.0, "ns") / grad(4.5, 0.5, 0.0, "sat"), "{:.0f}")
    assert js(q_far) > 0.98 and grad(4.5, 0.5, 0.0, "ns") > 50 * grad(4.5, 0.5, 0.0, "sat")

    q_col = gen(-2.0, 0.5, 0.0)
    fact("wid_js_collapse", js(q_col), "{:.3f}")
    fact("wid_prec_collapse", frac(p, q_col), "{:.2f}")
    fact("wid_rec_collapse", frac(q_col, p), "{:.2f}")
    assert frac(p, q_col) > 0.95 and abs(frac(q_col, p) - 0.6) < 0.05

    q_wide = gen(0.0, 2.5, 0.0)
    fact("wid_prec_wide", frac(p, q_wide), "{:.2f}")
    fact("wid_rec_wide", frac(q_wide, p), "{:.2f}")
    assert frac(p, q_wide) < 0.7 and frac(q_wide, p) > 0.95

    q_ok = gen(-2.0, 0.5, 0.4)
    fact("wid_js_match", js(q_ok), "{:.4f}")
    assert js(q_ok) < 1e-3


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SIDE.mkdir(parents=True, exist_ok=True)
    fig_optimal_d()
    fig_saturation()
    fig_rotation()
    fig_js_vs_w1()
    fig_gan_real()
    widget_facts()
    (ROOT / "scripts" / "data" / "lesson82_facts.json").write_text(
        json.dumps(FACTS, ensure_ascii=False, indent=1), encoding="utf8")
    print("\nOK: figures written to", OUT, "and", SIDE)


if __name__ == "__main__":
    main()
