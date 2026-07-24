"""Deterministic figures for lesson 85: variational autoencoder (VAE).

Everything is trained from scratch in numpy on the REAL sklearn `load_digits`
dataset (1797 handwritten 8x8 digits, UCI/NIST preprocessed). No fake numbers:
every value quoted in the lesson is computed here and asserted.

Outputs
    public/figures/lessons/85/*.png       (6 figures)
    public/figures/sidenotes/85/*.png     (4 marginal images)
    scripts/data/lesson85_facts.json      (all asserted numbers)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "85"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "85"
FACTS = ROOT / "scripts" / "data" / "lesson85_facts.json"

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

F: dict[str, float | int | str] = {}


def save(fig, path, *, dpi=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ----------------------------------------------------------------- data
def digits_split():
    d = load_digits()
    X = d.data / 16.0                      # brightness in [0,1]
    y = d.target
    rng = np.random.default_rng(8501)
    idx = rng.permutation(len(X))
    ntr = 1400
    return X[idx[:ntr]], y[idx[:ntr]], X[idx[ntr:]], y[idx[ntr:]]


# ----------------------------------------------------------------- VAE in numpy
def sigmoid(t):
    return 1.0 / (1.0 + np.exp(-np.clip(t, -30, 30)))


def init_params(rng, D=64, H=128, L=2):
    def w(a, b):
        return rng.normal(0, np.sqrt(2.0 / (a + b)), (a, b))
    return {
        "W1": w(D, H), "b1": np.zeros(H),
        "W2": w(H, L), "b2": np.zeros(L),
        "W3": w(H, L), "b3": np.zeros(L) - 1.0,   # start with modest sigma
        "W4": w(L, H), "b4": np.zeros(H),
        "W5": w(H, D), "b5": np.zeros(D),
    }


def decode(p, Z):
    h2 = np.maximum(Z @ p["W4"] + p["b4"], 0)
    logits = h2 @ p["W5"] + p["b5"]
    return h2, logits


def decode_grid(p, coords):
    _, logits = decode(p, coords)
    return sigmoid(logits)


def bce(X, logits):
    """Sum over pixels, mean over objects (nats)."""
    xh = sigmoid(logits)
    eps = 1e-7
    return -np.sum(X * np.log(xh + eps) + (1 - X) * np.log(1 - xh + eps), axis=1)


def kl_terms(mu, logvar):
    return 0.5 * (mu ** 2 + np.exp(logvar) - logvar - 1.0)


def train_vae(Xtr, beta, seed, latent=2, hidden=128, epochs=400, batch=128, lr=2e-3):
    rng = np.random.default_rng(seed)
    p = init_params(rng, Xtr.shape[1], hidden, latent)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v) for k, v in p.items()}
    t = 0
    n = len(Xtr)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            xb = Xtr[order[s:s + batch]]
            B = len(xb)
            h1 = np.maximum(xb @ p["W1"] + p["b1"], 0)
            mu = h1 @ p["W2"] + p["b2"]
            logvar = np.clip(h1 @ p["W3"] + p["b3"], -12, 12)
            sig = np.exp(0.5 * logvar)
            eps = rng.standard_normal(mu.shape)
            z = mu + sig * eps
            h2 = np.maximum(z @ p["W4"] + p["b4"], 0)
            logits = h2 @ p["W5"] + p["b5"]
            xh = sigmoid(logits)

            g = {}
            dlogits = (xh - xb) / B
            g["W5"] = h2.T @ dlogits; g["b5"] = dlogits.sum(0)
            dh2 = (dlogits @ p["W5"].T) * (h2 > 0)
            g["W4"] = z.T @ dh2; g["b4"] = dh2.sum(0)
            dz = dh2 @ p["W4"].T
            dmu = dz + beta * mu / B
            dlogvar = dz * eps * 0.5 * sig + beta * 0.5 * (np.exp(logvar) - 1.0) / B
            g["W2"] = h1.T @ dmu; g["b2"] = dmu.sum(0)
            g["W3"] = h1.T @ dlogvar; g["b3"] = dlogvar.sum(0)
            dh1 = (dmu @ p["W2"].T + dlogvar @ p["W3"].T) * (h1 > 0)
            g["W1"] = xb.T @ dh1; g["b1"] = dh1.sum(0)

            t += 1
            for k in p:
                m[k] = 0.9 * m[k] + 0.1 * g[k]
                v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
                mh = m[k] / (1 - 0.9 ** t); vh = v[k] / (1 - 0.999 ** t)
                p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
    return p


def evaluate(p, X, rng_seed=99):
    h1 = np.maximum(X @ p["W1"] + p["b1"], 0)
    mu = h1 @ p["W2"] + p["b2"]
    logvar = np.clip(h1 @ p["W3"] + p["b3"], -12, 12)
    _, logits_mu = decode(p, mu)
    rng = np.random.default_rng(rng_seed)
    z = mu + np.exp(0.5 * logvar) * rng.standard_normal(mu.shape)
    _, logits_z = decode(p, z)
    return {
        "mu": mu, "logvar": logvar,
        "bce_mu": float(bce(X, logits_mu).mean()),
        "bce_z": float(bce(X, logits_z).mean()),
        "kl": float(kl_terms(mu, logvar).sum(1).mean()),
        "kl_per_dim": kl_terms(mu, logvar).mean(0),
    }


def probe_accuracy(mu_tr, y_tr, mu_te, y_te):
    clf = LogisticRegression(C=1.0, max_iter=5000)
    clf.fit(mu_tr, y_tr)
    return float(clf.score(mu_te, y_te))


# ----------------------------------------------------------------- experiments
Xtr, ytr, Xte, yte = digits_split()
F["n_train"] = int(len(Xtr)); F["n_test"] = int(len(Xte))
F["n_total"] = int(len(Xtr) + len(Xte)); F["n_pixels"] = 64

print("training latent-2 models ...")
MODELS2 = {b: train_vae(Xtr, b, seed=8510 + int(b * 10), latent=2) for b in (0.0, 1.0)}
EV2 = {b: evaluate(m, Xte) for b, m in MODELS2.items()}
EV2TR = {b: evaluate(m, Xtr) for b, m in MODELS2.items()}

F["bce_ae_l2"] = round(EV2[0.0]["bce_mu"], 1)
F["bce_vae_l2"] = round(EV2[1.0]["bce_mu"], 1)
F["bce_ae_l2_train"] = round(EV2TR[0.0]["bce_mu"], 1)
F["bce_vae_l2_train"] = round(EV2TR[1.0]["bce_mu"], 1)
F["kl_vae_l2"] = round(EV2[1.0]["kl"], 2)
F["kl_ae_l2"] = round(EV2[0.0]["kl"], 1)
assert F["bce_ae_l2_train"] <= F["bce_vae_l2_train"]
assert F["kl_ae_l2"] > 20 * F["kl_vae_l2"]
print("  AE bce", F["bce_ae_l2"], "VAE bce", F["bce_vae_l2"],
      "| KL", F["kl_ae_l2"], F["kl_vae_l2"])

mu_ae = EV2TR[0.0]["mu"]; mu_vae = EV2TR[1.0]["mu"]


def coverage(mu):
    """Share of codes inside the prior's central disk ||z|| <= 2."""
    return float(np.mean(np.linalg.norm(mu, axis=1) <= 2.0))


F["cover_ae"] = round(coverage(mu_ae) * 100, 1)
F["cover_vae"] = round(coverage(mu_vae) * 100, 1)
F["span_ae"] = round(float(np.abs(mu_ae).max()), 1)
F["span_vae"] = round(float(np.abs(mu_vae).max()), 1)
F["prior_disk_mass"] = round(float(1 - np.exp(-2.0)) * 100, 1)   # P(||z||<=2), d=2
print("  coverage AE", F["cover_ae"], "% VAE", F["cover_vae"],
      "% spans", F["span_ae"], F["span_vae"])
assert F["cover_ae"] < 5 < 70 < F["cover_vae"]
assert F["span_ae"] > 3 * F["span_vae"]

F["probe_ae"] = round(probe_accuracy(mu_ae, ytr, EV2[0.0]["mu"], yte) * 100, 1)
F["probe_vae"] = round(probe_accuracy(mu_vae, ytr, EV2[1.0]["mu"], yte) * 100, 1)
print("  probe(2D) AE", F["probe_ae"], "VAE", F["probe_vae"])


# ---------------------------------------- fig 85.1: latent map
def fig_latent_map():
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.4))
    panels = [
        (mu_ae, f"$\\beta=0$: масштаб кода ничем не закреплён\nв круге $\\|z\\|\\leq2$ лежит {F['cover_ae']:.1f}% кодов"),
        (mu_vae, f"$\\beta=1$: KL держит облако у нуля\nв круге $\\|z\\|\\leq2$ лежит {F['cover_vae']:.0f}% кодов"),
    ]
    for ax, (M, title) in zip(axes, panels):
        lim = float(np.abs(M).max()) * 1.1
        sc = ax.scatter(M[:, 0], M[:, 1], c=ytr, cmap="tab10", s=7, alpha=0.75, lw=0)
        th = np.linspace(0, 2 * np.pi, 300)
        for r, a in [(1, 0.5), (2, 0.9)]:
            ax.plot(r * np.cos(th), r * np.sin(th), color=INK, lw=1.5, alpha=a, ls=(0, (4, 3)))
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title(title, fontsize=12.5)
        ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$")
        ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
        ax.set_aspect("equal")
    axes[0].annotate("окружности $\\|z\\|=1$ и $2$ — почти точка",
                     xy=(0, 2), xytext=(0.05, 0.6 * axes[0].get_ylim()[1]),
                     fontsize=10, color=INK,
                     arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1))
    cb = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02, ticks=range(10))
    cb.set_label("цифра", color=MUTED)
    fig.suptitle("Одни и те же 1400 цифр в двумерном коде: два разных масштаба", y=0.99)
    save(fig, OUT / "latent_map.png")


fig_latent_map()


# ---------------------------------------- gradient estimators
def gradient_variance():
    """d/dmu E_{z~N(mu,1)}[z^2]; true value 2*mu. Two unbiased estimators."""
    rng = np.random.default_rng(8520)
    mu0 = 1.0
    e = rng.standard_normal(20000)
    z = mu0 + e
    return 2 * z, z ** 2 * (z - mu0), 2 * mu0


PATH, SCORE, TRUEG = gradient_variance()
F["var_path"] = round(float(PATH.var()), 2)
F["var_score"] = round(float(SCORE.var()), 2)
F["var_ratio"] = round(float(SCORE.var() / PATH.var()), 1)
F["grad_mean_path"] = round(float(PATH.mean()), 3)
F["grad_mean_score"] = round(float(SCORE.mean()), 3)
assert abs(PATH.mean() - TRUEG) < 0.05 and abs(SCORE.mean() - TRUEG) < 0.2
assert F["var_ratio"] > 3
print("  grad var: path", F["var_path"], "score", F["var_score"], "ratio", F["var_ratio"])


def fig_reparam():
    fig = plt.figure(figsize=(11.4, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.22)
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ax.set_xlim(0, 10.2); ax.set_ylim(0, 6.4)

    def box(x, y, w, h, text, color=INK, fc=PAPER):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=color, lw=1.4, zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11.5, zorder=4)

    def arrow(x1, y1, x2, y2, color=INK, ls="-", lw=1.4):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls))

    box(0.1, 2.4, 1.2, 1.0, "$x$")
    box(1.9, 2.4, 1.5, 1.0, "encoder\n$\\phi$", BLUE)
    box(4.0, 3.7, 1.3, 0.9, "$\\mu$", BLUE)
    box(4.0, 1.2, 1.3, 0.9, "$\\sigma$", BLUE)
    box(3.6, 5.2, 2.2, 0.8, "$\\varepsilon\\sim\\mathcal{N}(0,I)$", GOLD, WASH)
    box(6.2, 2.4, 1.9, 1.0, "$z=\\mu+\\sigma\\varepsilon$", GREEN)
    box(8.7, 2.4, 1.4, 1.0, "decoder\n$\\theta$", VIOLET)
    arrow(1.3, 2.9, 1.9, 2.9)
    arrow(3.4, 3.1, 4.0, 4.1); arrow(3.4, 2.7, 4.0, 1.8)
    arrow(5.3, 4.1, 6.2, 3.2); arrow(5.3, 1.7, 6.2, 2.6)
    arrow(4.7, 5.2, 6.6, 3.4, GOLD, (0, (4, 3)))
    arrow(8.1, 2.9, 8.7, 2.9)
    arrow(8.7, 2.2, 7.0, 2.2, RED, "-", 2.0)
    arrow(6.4, 2.5, 5.2, 1.6, RED, "-", 2.0)
    arrow(6.4, 3.3, 5.2, 4.2, RED, "-", 2.0)
    ax.text(7.1, 1.75, "градиент", color=RED, fontsize=10.5)
    ax.text(4.7, 6.1, "случайность вынесена в отдельный вход", color=GOLD,
            fontsize=10.5, ha="center")
    ax.set_title("Репараметризация: путь градиента обходит жребий", fontsize=13)

    ax2 = fig.add_subplot(gs[0, 1])
    bins = np.linspace(-8, 12, 70)
    ax2.hist(SCORE, bins=bins, color=RED, alpha=0.45,
             label=f"score-function, Var$={F['var_score']:.1f}$")
    ax2.hist(PATH, bins=bins, color=GREEN, alpha=0.65,
             label=f"репараметризация, Var$={F['var_path']:.1f}$")
    ax2.axvline(TRUEG, color=INK, lw=1.6, ls=(0, (4, 3)))
    ax2.set_ylim(0, 1500)
    ax2.text(TRUEG + 0.5, 1180, "истинный\nградиент $=2$", fontsize=10, color=INK)
    ax2.annotate("пик у нуля обрезан", xy=(0.15, 1450), xytext=(-7.5, 900),
                 fontsize=9.5, color=RED,
                 arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.0))
    ax2.set_title(f"Разброс двух несмещённых оценок\nодного градиента: в {F['var_ratio']:.0f} раз",
                  fontsize=12.5)
    ax2.set_xlabel("значение оценки"); ax2.set_ylabel("частота")
    ax2.legend(frameon=False, fontsize=9.5, loc="center right")
    ax2.grid(True, color=GRID, lw=0.4, alpha=0.4); ax2.set_axisbelow(True)
    save(fig, OUT / "reparam.png")


fig_reparam()


# ---------------------------------------- rate-distortion sweep (latent 8)
print("training latent-8 sweep ...")
BETAS = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 8.0]
SWEEP = {}
for b in BETAS:
    p = train_vae(Xtr, b, seed=8530, latent=8, epochs=300)
    ev = evaluate(p, Xte)
    evtr = evaluate(p, Xtr)
    act = int(np.sum(ev["mu"].var(0) > 0.01))
    acc = probe_accuracy(evtr["mu"], ytr, ev["mu"], yte)
    SWEEP[b] = {"bce": ev["bce_mu"], "kl": ev["kl"], "active": act, "acc": acc,
                "kl_per_dim": np.sort(ev["kl_per_dim"])[::-1], "params": p}
    print(f"  beta={b:5}: bce={ev['bce_mu']:7.2f} kl={ev['kl']:7.2f} "
          f"active={act} probe={acc:.3f}")

for tag, b in [("0", 0.0), ("025", 0.25), ("05", 0.5), ("1", 1.0),
               ("2", 2.0), ("4", 4.0), ("8", 8.0)]:
    F[f"rd_beta{tag}_bce"] = round(SWEEP[b]["bce"], 1)
    F[f"rd_beta{tag}_kl"] = round(SWEEP[b]["kl"], 2)
    F[f"rd_beta{tag}_active"] = SWEEP[b]["active"]
    F[f"rd_beta{tag}_acc"] = round(SWEEP[b]["acc"] * 100, 1)
F["kl_drop_ratio"] = round(SWEEP[0.0]["kl"] / SWEEP[0.25]["kl"], 1)
F["bce_cost_025"] = round(SWEEP[0.25]["bce"] - SWEEP[0.0]["bce"], 2)
assert SWEEP[0.0]["bce"] < SWEEP[1.0]["bce"] < SWEEP[8.0]["bce"]
assert SWEEP[8.0]["kl"] < 0.01 and SWEEP[8.0]["active"] == 0
assert SWEEP[8.0]["acc"] < 0.3 < SWEEP[1.0]["acc"]
assert SWEEP[0.0]["kl"] > 100 and SWEEP[0.25]["kl"] < 15

pbar = np.clip(Xtr.mean(0), 1e-4, 1 - 1e-4)
const_logits = np.log(pbar / (1 - pbar))[None, :] * np.ones((len(Xte), 1))
F["baseline_bce"] = round(float(bce(Xte, const_logits).mean()), 1)
print("  constant-mean baseline bce", F["baseline_bce"])
assert F["baseline_bce"] >= F["rd_beta8_bce"] - 0.5


def fig_rate_distortion():
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9))
    ax = axes[0]
    ks = [SWEEP[b]["kl"] for b in BETAS]
    ds = [SWEEP[b]["bce"] for b in BETAS]
    ax.plot(ks, ds, color=BLUE, lw=1.8, zorder=2)
    ax.scatter(ks, ds, s=60, color=BLUE, zorder=3)
    for b, k, d in zip(BETAS, ks, ds):
        if b in (4.0, 8.0):
            continue
        off = (9, 5) if b != 3.0 else (12, -14)
        ax.annotate(f"$\\beta={b:g}$", (k, d), textcoords="offset points",
                    xytext=off, fontsize=10, color=MUTED)
    ax.scatter([SWEEP[8.0]["kl"]], [SWEEP[8.0]["bce"]], s=120, color=RED, zorder=4)
    ax.annotate("$\\beta=4$ и $\\beta=8$: коллапс,\nкод не несёт ничего",
                (SWEEP[8.0]["kl"], SWEEP[8.0]["bce"]),
                textcoords="offset points", xytext=(30, -22), fontsize=10, color=RED,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.0))
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("rate: $D_{KL}$ на объект, нат (шкала symlog)")
    ax.set_ylabel("distortion: BCE на объект, нат")
    ax.set_title("Кривая «скорость — искажение»", fontsize=13)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)

    ax2 = axes[1]
    for b, c in [(0.25, GREEN), (1.0, BLUE), (2.0, GOLD), (8.0, RED)]:
        ax2.plot(range(1, 9), np.maximum(SWEEP[b]["kl_per_dim"], 1e-6), marker="o",
                 ms=5, color=c, lw=1.7,
                 label=f"$\\beta={b:g}$, активных {SWEEP[b]['active']}")
    ax2.axhline(0.01, color=MUTED, lw=0.9, ls=(0, (3, 3)))
    ax2.set_yscale("log")
    ax2.set_xlabel("координата латента (по убыванию KL)")
    ax2.set_ylabel("вклад координаты в KL, нат")
    ax2.set_title("Кто из восьми координат работает", fontsize=13)
    ax2.legend(frameon=False, fontsize=9.5)
    ax2.grid(True, color=GRID, lw=0.4, alpha=0.4); ax2.set_axisbelow(True)
    save(fig, OUT / "rate_distortion.png")


fig_rate_distortion()


# ---------------------------------------- fig 85.4: decoded prior grid
def fig_prior_samples():
    q = norm.ppf(np.linspace(0.02, 0.98, 11))
    G1, G2 = np.meshgrid(q, q)
    coords = np.stack([G1.ravel(), G2.ravel()], 1)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.7))
    div = {}
    near = {}
    panels = [(0.0, "$\\beta=0$: сетка prior бьёт мимо кодов"),
              (1.0, "$\\beta=1$: сетка prior лежит на данных")]
    for ax, (b, title) in zip(axes, panels):
        flat = decode_grid(MODELS2[b], coords)
        div[b] = float(flat.std(0).mean())
        dist = np.sqrt(((flat[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)).min(1)
        near[b] = float(dist.mean())
        imgs = flat.reshape(11, 11, 8, 8)
        tiles = [[np.pad(imgs[i, j], 1, constant_values=0.0) for j in range(11)]
                 for i in range(11)]
        canvas = np.block(tiles)
        ax.imshow(canvas, cmap="gray_r", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=12.5)
    F["grid_div_ae"] = round(div[0.0], 4)
    F["grid_div_vae"] = round(div[1.0], 4)
    F["grid_div_ratio"] = round(div[1.0] / div[0.0], 2)
    F["grid_near_ae"] = round(near[0.0], 2)
    F["grid_near_vae"] = round(near[1.0], 2)
    F["grid_near_ratio"] = round(near[0.0] / near[1.0], 2)
    print("  grid nearest-train distance AE", F["grid_near_ae"], "VAE", F["grid_near_vae"])
    assert near[0.0] > near[1.0], (near[0.0], near[1.0])
    assert div[1.0] > 1.4 * div[0.0], (div[0.0], div[1.0])
    fig.suptitle("Декодированная сетка квантилей $\\mathcal{N}(0,I)$: 11×11 точек", y=0.97)
    save(fig, OUT / "prior_samples.png")


fig_prior_samples()
print("  grid diversity AE", F["grid_div_ae"], "VAE", F["grid_div_vae"])


# ---------------------------------------- fig 85.5: aggregate posterior vs prior
def fig_aggregate():
    rng = np.random.default_rng(8540)
    ev = EV2TR[1.0]
    sig = np.exp(0.5 * ev["logvar"])
    zq = ev["mu"] + sig * rng.standard_normal(ev["mu"].shape)
    zp = rng.standard_normal((len(zq), 2))
    rq = np.linalg.norm(zq, axis=1); rp = np.linalg.norm(zp, axis=1)
    F["rad_q"] = round(float(rq.mean()), 2)
    F["rad_p"] = round(float(rp.mean()), 2)
    F["sigma_med"] = round(float(np.median(sig)), 3)

    grid = np.stack(np.meshgrid(np.linspace(-1.9, 1.9, 39), np.linspace(-1.9, 1.9, 39)), -1).reshape(-1, 2)
    d = np.sqrt(((grid[:, None, :] - ev["mu"][None, :, :]) ** 2).sum(-1)).min(1)
    hole = grid[int(np.argmax(d))]
    dense = grid[int(np.argmin(d))]
    F["hole_dist"] = round(float(d.max()), 2)
    F["dense_dist"] = round(float(d.min()), 3)
    F["hole_x"] = round(float(hole[0]), 2); F["hole_y"] = round(float(hole[1]), 2)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    ax.scatter(zp[:, 0], zp[:, 1], s=6, color=FAINT, alpha=0.5, lw=0, label="$z\\sim p(z)$")
    ax.scatter(zq[:, 0], zq[:, 1], s=6, color=BLUE, alpha=0.55, lw=0,
               label="$z\\sim q(z)$ (агрегат)")
    ax.scatter([hole[0]], [hole[1]], s=170, marker="X", color=RED, zorder=5, label="дыра")
    ax.scatter([dense[0]], [dense[1]], s=170, marker="P", color=GREEN, zorder=5,
               label="плотная область")
    ax.set_xlim(-3.4, 3.4); ax.set_ylim(-3.4, 3.4); ax.set_aspect("equal")
    ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.set_title(f"Агрегат $q(z)$ и prior: средний радиус {F['rad_q']:.2f} против {F['rad_p']:.2f}",
                 fontsize=12.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)

    ax2 = axes[1]
    ax2.hist(rp, bins=40, color=FAINT, alpha=0.55, density=True, label="$\\|z\\|$, prior")
    ax2.hist(rq, bins=40, color=BLUE, alpha=0.55, density=True, label="$\\|z\\|$, агрегат")
    ax2.set_xlabel("радиус $\\|z\\|$"); ax2.set_ylabel("плотность")
    ax2.legend(frameon=False, fontsize=10)
    ax2.set_title("Радиальные профили не совпадают", fontsize=12.5)
    ax2.grid(True, color=GRID, lw=0.4, alpha=0.4); ax2.set_axisbelow(True)
    save(fig, OUT / "aggregate.png")

    imgs = decode_grid(MODELS2[1.0], np.stack([hole, dense]))
    F["hole_ambiguity"] = round(float(np.mean(np.minimum(imgs[0], 1 - imgs[0]))), 3)
    F["dense_ambiguity"] = round(float(np.mean(np.minimum(imgs[1], 1 - imgs[1]))), 3)
    assert F["hole_ambiguity"] > F["dense_ambiguity"]
    fig2, ax3 = plt.subplots(1, 2, figsize=(5.6, 3.1))
    titles = [f"дыра ({F['hole_x']:g}; {F['hole_y']:g})", "плотная область"]
    for a, im, t, c in zip(ax3, imgs, titles, [RED, GREEN]):
        a.imshow(im.reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
        a.set_xticks([]); a.set_yticks([]); a.set_title(t, fontsize=11, color=c)
    fig2.suptitle("Две точки prior, два декодирования", fontsize=12)
    save(fig2, SIDE / "hole.png")


fig_aggregate()
print("  radius q/p", F["rad_q"], F["rad_p"], "| hole", F["hole_x"], F["hole_y"], F["hole_dist"])


# ---------------------------------------- fig 85.6: interpolation and Jensen gap
def fig_interpolation():
    rng = np.random.default_rng(8550)
    p = MODELS2[1.0]
    ev = evaluate(p, Xte)
    sig = np.exp(0.5 * ev["logvar"])
    i = int(np.argmax(sig.sum(1)))
    Z = ev["mu"][i] + sig[i] * rng.standard_normal((200, 2))
    dec = decode_grid(p, Z)
    mean_of_dec = dec.mean(0)
    dec_of_mean = decode_grid(p, ev["mu"][i][None, :])[0]
    F["jensen_gap"] = round(float(np.abs(mean_of_dec - dec_of_mean).mean()), 4)
    F["jensen_gap_max"] = round(float(np.abs(mean_of_dec - dec_of_mean).max()), 3)
    F["pix_var_max"] = round(float(dec.var(0).max()), 4)
    F["sigma_max_obj"] = round(float(sig[i].max()), 3)
    assert F["jensen_gap"] > 0

    a = int(np.where(yte == 0)[0][0]); b = int(np.where(yte == 8)[0][0])
    za, zb = ev["mu"][a], ev["mu"][b]
    ts = np.linspace(0, 1, 8)
    lin = np.stack([(1 - t) * za + t * zb for t in ts])
    F["lin_mid_norm"] = round(float(np.linalg.norm(0.5 * (za + zb))), 2)
    F["end_norm"] = round(float((np.linalg.norm(za) + np.linalg.norm(zb)) / 2), 2)
    imgs_lin = decode_grid(p, lin)

    fig = plt.figure(figsize=(11.4, 5.8))
    gs = fig.add_gridspec(2, 8, height_ratios=[1.05, 1.0], hspace=0.5, wspace=0.2)
    for k in range(8):
        a1 = fig.add_subplot(gs[0, k])
        a1.imshow(imgs_lin[k].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
        a1.set_xticks([]); a1.set_yticks([])
        a1.set_title(f"$t={ts[k]:.2f}$", fontsize=9.5, color=MUTED)
    fig.text(0.5, 0.96, "Прямая в латенте между кодами двух тестовых цифр",
             ha="center", fontsize=13)

    panels = [(dec_of_mean, "$g(\\mathbb{E}z)$", INK, "gray_r"),
              (mean_of_dec, "$\\mathbb{E}\\,g(z)$", BLUE, "gray_r"),
              (np.abs(mean_of_dec - dec_of_mean),
               f"модуль разности,\nмакс {F['jensen_gap_max']:.2f}", RED, "magma_r"),
              (dec.var(0), f"дисперсия пикселя,\nмакс {F['pix_var_max']:.3f}", GOLD, "magma_r")]
    for k, (im, t, c, cm) in enumerate(panels):
        a2 = fig.add_subplot(gs[1, 2 * k:2 * k + 2])
        a2.imshow(im.reshape(8, 8), cmap=cm)
        a2.set_xticks([]); a2.set_yticks([]); a2.set_title(t, fontsize=10.5, color=c)
    fig.text(0.5, 0.47, "Один объект, двести жребиев: средний код и средняя картинка — разное",
             ha="center", fontsize=13)
    save(fig, OUT / "interpolation.png")


fig_interpolation()
print("  jensen gap", F["jensen_gap"], "max", F["jensen_gap_max"])


# ---------------------------------------- sidenote: shape of the KL penalty
def side_kl():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
    mu = np.linspace(-3, 3, 200)
    axes[0].plot(mu, 0.5 * mu ** 2, color=BLUE, lw=2)
    axes[0].set_xlabel("$\\mu$ при $\\sigma=1$"); axes[0].set_ylabel("KL, нат")
    s = np.linspace(0.15, 3, 300)
    axes[1].plot(s, 0.5 * (s ** 2 - np.log(s ** 2) - 1), color=RED, lw=2)
    axes[1].axvline(1, color=MUTED, lw=0.9, ls=(0, (3, 3)))
    axes[1].set_xlabel("$\\sigma$ при $\\mu=0$")
    F["kl_sigma_half"] = round(float(0.5 * (0.25 - np.log(0.25) - 1)), 3)
    F["kl_sigma_two"] = round(float(0.5 * (4 - np.log(4) - 1)), 3)
    F["kl_mu_one"] = 0.5
    assert F["kl_sigma_two"] > F["kl_sigma_half"] > F["kl_mu_one"] * 0.5
    for ax in axes:
        ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    fig.suptitle("Штраф KL: за сдвиг и за ширину", fontsize=12)
    save(fig, SIDE / "kl_shape.png")


side_kl()


# ---------------------------------------- sidenote: Pinsker inequality
def side_pinsker():
    rng = np.random.default_rng(8560)
    kls, tvs = [], []
    for _ in range(600):
        k = int(rng.integers(2, 6))
        p = rng.dirichlet(np.ones(k) * rng.uniform(0.4, 3))
        q = rng.dirichlet(np.ones(k) * rng.uniform(0.4, 3))
        kl = float(np.sum(p * np.log(p / q)))
        tv = float(0.5 * np.abs(p - q).sum())
        if kl < 2.5:
            kls.append(kl); tvs.append(tv)
    kls = np.array(kls); tvs = np.array(tvs)
    assert np.all(tvs <= np.sqrt(kls / 2) + 1e-9)
    F["pinsker_pairs"] = int(len(kls))
    F["pinsker_max_ratio"] = round(float(np.max(tvs / np.sqrt(np.maximum(kls, 1e-12) / 2))), 3)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.scatter(kls, tvs, s=12, color=BLUE, alpha=0.55, lw=0, label="случайные пары $p,q$")
    xs = np.linspace(0, float(kls.max()), 200)
    ax.plot(xs, np.minimum(np.sqrt(xs / 2), 1), color=RED, lw=2,
            label="граница $\\sqrt{D_{KL}/2}$")
    ax.set_xlabel("$D_{KL}(p\\|q)$, нат"); ax.set_ylabel("расстояние по вариации")
    ax.legend(frameon=False, fontsize=9.5)
    ax.set_title("Неравенство Пинскера держит крышу", fontsize=12)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "pinsker.png")


side_pinsker()
print("  pinsker pairs", F["pinsker_pairs"], "max ratio", F["pinsker_max_ratio"])


# ---------------------------------------- sidenote: typical set shell
def side_typical():
    rng = np.random.default_rng(8570)
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    for d, c in [(2, GREEN), (8, BLUE), (64, VIOLET)]:
        r = np.linalg.norm(rng.standard_normal((20000, d)), axis=1)
        ax.hist(r, bins=70, density=True, color=c, alpha=0.5, label=f"$d={d}$")
        F[f"shell_mean_d{d}"] = round(float(r.mean()), 2)
        F[f"shell_p05_d{d}"] = round(float(np.mean(r < 0.5) * 100), 2)
    assert F["shell_p05_d64"] == 0.0
    ax.set_xlabel("$\\|z\\|$"); ax.set_ylabel("плотность")
    ax.legend(frameon=False, fontsize=10)
    ax.set_title("Нормальный шар пуст в центре", fontsize=12)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "typical_set.png")


side_typical()


# ---------------------------------------- sidenote: blur is an honest average
def side_blur():
    d = load_digits()
    a = d.data[np.where(d.target == 3)[0][0]] / 16.0
    b = d.data[np.where(d.target == 5)[0][0]] / 16.0
    mid = 0.5 * (a + b)
    mse_mid = float(0.5 * (np.mean((mid - a) ** 2) + np.mean((mid - b) ** 2)))
    mse_pick = float(0.5 * (np.mean((a - a) ** 2) + np.mean((a - b) ** 2)))
    F["mse_mid"] = round(mse_mid, 4)
    F["mse_pick"] = round(mse_pick, 4)
    F["mse_ratio"] = round(mse_pick / mse_mid, 2)
    assert mse_mid < mse_pick
    fig, axes = plt.subplots(1, 3, figsize=(6.2, 2.6))
    titles = ["вариант A", f"среднее\nMSE {mse_mid:.3f}", "вариант B"]
    for ax, im, t in zip(axes, [a, mid, b], titles):
        ax.imshow(im.reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, fontsize=10)
    fig.suptitle("Размытие — оптимум MSE, а не лень", fontsize=11.5)
    save(fig, SIDE / "blur.png")


side_blur()
print("  blur mse mid/pick", F["mse_mid"], F["mse_pick"], F["mse_ratio"])

# ------------------------------- closed-form toy behind the widget vae-latent-lab
def toy_model(beta, spread=1.6, n=5):
    """Decoder g(z)=z, targets t_i; optimum mu=2t/(2+beta), sigma^2=beta/(beta+2)."""
    shrink = 2 / (2 + beta)
    sigma = np.sqrt(beta / (beta + 2))
    t = spread * (np.arange(n) - (n - 1) / 2)
    mu = shrink * t
    dist = float(np.mean((mu - t) ** 2 + sigma ** 2))
    rate = float(np.mean(0.5 * (mu ** 2 + sigma ** 2 - np.log(sigma ** 2) - 1)))
    edges = 0.5 * (t[:-1] + t[1:])
    lo = np.concatenate([[-np.inf], edges]); hi = np.concatenate([edges, [np.inf]])
    acc = float(np.mean(norm.cdf((hi - mu) / sigma) - norm.cdf((lo - mu) / sigma)))
    return shrink, sigma, dist, rate, acc


def toy_objective(mu, sigma, t, beta):
    return (mu - t) ** 2 + sigma ** 2 + beta * 0.5 * (mu ** 2 + sigma ** 2 - np.log(sigma ** 2) - 1)


for b in (0.5, 1.0, 2.0, 8.0):
    sh, sg, dd, rr, aa = toy_model(b)
    tag = str(b).replace(".", "").rstrip("0") or "0"
    F[f"toy_shrink_b{b:g}"] = round(sh, 3)
    F[f"toy_sigma_b{b:g}"] = round(sg, 3)
    F[f"toy_dist_b{b:g}"] = round(dd, 3)
    F[f"toy_rate_b{b:g}"] = round(rr, 3)
    F[f"toy_acc_b{b:g}"] = round(aa * 100, 1)
    # verify the closed form really is the minimum for a single target t=1.6
    t0, mu0, s0 = 1.6, 2 * 1.6 / (2 + b), np.sqrt(b / (b + 2))
    base = toy_objective(mu0, s0, t0, b)
    grid_mu = mu0 + np.linspace(-0.4, 0.4, 81)
    grid_s = s0 * np.exp(np.linspace(-0.5, 0.5, 81))
    MM, SS = np.meshgrid(grid_mu, grid_s)
    assert base <= toy_objective(MM, SS, t0, b).min() + 1e-9
F["toy_acc_random5"] = 20.0
print("  toy beta=1:", F["toy_shrink_b1"], F["toy_sigma_b1"], F["toy_rate_b1"],
      F["toy_dist_b1"], F["toy_acc_b1"])
assert F["toy_acc_b1"] > F["toy_acc_b8"] > 20.0
assert abs(F["toy_shrink_b1"] - 2 / 3) < 1e-3 and abs(F["toy_sigma_b1"] - 3 ** -0.5) < 1e-3

FACTS.parent.mkdir(parents=True, exist_ok=True)
FACTS.write_text(json.dumps(F, ensure_ascii=False, indent=1), encoding="utf-8")
print("\nfacts:")
for k, v in F.items():
    print(f"  {k} = {v}")
print("\nOK")
