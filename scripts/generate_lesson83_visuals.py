"""Deterministic figures for lesson 83: latent scales, style mixing, inversion, adversarial shift.

REAL data: sklearn `load_digits` (1797 handwritten 8x8 digits, UCI optical recognition set).
A linear PCA generator G(w) = m + V w is used as an honest, fully computable stand-in for
StyleGAN's synthesis network: it shares the two properties the lesson needs (a latent code that
enters at several scales, and a learned manifold outside which inversion fails), while every
number stays reproducible on a laptop. All quoted numbers are asserted below.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "83"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "83"
FACTS = ROOT / "scripts" / "data" / "lesson83_facts.json"

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

facts: dict[str, float] = {}


def save(fig, path, *, dpi=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def show(ax, img, title=None, cmap="gray_r", fs=8.5, color=INK):
    ax.imshow(img.reshape(8, 8), cmap=cmap, vmin=0, vmax=16, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(LINE)
    if title:
        ax.set_title(title, fontsize=fs, color=color, pad=3)


# ---------------------------------------------------------------- data & linear generator
digits = load_digits()
X = digits.data.astype(float)            # 1797 x 64, values 0..16
y = digits.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=83, stratify=y)

mean = Xtr.mean(axis=0)
U, S, Vt = np.linalg.svd(Xtr - mean, full_matrices=False)
lam = S ** 2 / (len(Xtr) - 1)
var_share = lam / lam.sum()
K = 20                                    # latent dimension of the toy generator
V = Vt[:K]                                # K x 64, orthonormal rows


def code(x):
    return (x - mean) @ V.T


def gen(w):
    return mean + w @ V


facts["n_digits"] = len(X)
facts["n_train"] = len(Xtr)
facts["n_test"] = len(Xte)
facts["var_first4"] = float(var_share[:4].sum() * 100)
facts["var_17_20"] = float(var_share[16:20].sum() * 100)
facts["var_k20"] = float(var_share[:K].sum() * 100)
assert facts["n_digits"] == 1797 and facts["n_train"] == 1257 and facts["n_test"] == 540
assert 45 < facts["var_first4"] < 52, facts["var_first4"]
assert 3.0 < facts["var_17_20"] < 6.0, facts["var_17_20"]
assert 88 < facts["var_k20"] < 92, facts["var_k20"]

# how uneven is the image step for an equal step in the *whitened* latent
sq = np.sqrt(lam[:K])
facts["step_ratio"] = float(sq[0] / sq[K - 1])
assert 3.9 < facts["step_ratio"] < 4.1, facts["step_ratio"]


# ---------------------------------------------------------------- measurable features
gx = np.tile(np.arange(8, dtype=float), (8, 1))
chk = (-1.0) ** (np.indices((8, 8)).sum(axis=0))


def f_center(img):
    """Горизонтальный момент: крупный масштаб, линейный функционал."""
    im = img.reshape(8, 8)
    return float(((gx - 3.5) * im).mean())


def f_ink(img):
    """Общее количество чернил: тоже крупный масштаб."""
    return float(img.mean())


def f_high(img):
    """Шахматная (найквистова) мода: самый мелкий масштаб, линейный функционал."""
    return float((chk * img.reshape(8, 8)).mean())


FEATS = [("горизонтальный момент (крупный масштаб)", f_center, BLUE),
         ("общее количество чернил", f_ink, GREEN),
         ("шахматная мода (самый мелкий масштаб)", f_high, RED)]

# ---------------------------------------------------------------- style mixing experiment
rng = np.random.default_rng(83)
idx = rng.permutation(len(Xte))
pairs = [(int(idx[2 * i]), int(idx[2 * i + 1])) for i in range(200)]
cuts = np.arange(0, K + 1)

shares = np.zeros((len(FEATS), len(cuts)))
counts = np.zeros(len(FEATS))
for fi, (_, fn, _) in enumerate(FEATS):
    rows = []
    for a, b in pairs:
        wa, wb = code(Xte[a]), code(Xte[b])
        fa, fb = fn(gen(wa)), fn(gen(wb))
        if abs(fa - fb) < 1e-6:
            continue
        rowv = []
        for c in cuts:
            w = np.concatenate([wa[:c], wb[c:]])
            rowv.append((fn(gen(w)) - fb) / (fa - fb))
        rows.append(rowv)
    shares[fi] = np.median(np.array(rows), axis=0)
    counts[fi] = len(rows)
assert (counts == 200).all()

c50 = []
for fi in range(len(FEATS)):
    reached = np.where(shares[fi] >= 0.5)[0]
    c50.append(int(cuts[reached[0]]))
facts["c50_center"], facts["c50_ink"], facts["c50_high"] = c50
facts["share_center_at4"] = float(shares[0][4] * 100)
facts["share_high_at4"] = float(shares[2][4] * 100)
assert c50[0] == 4 and c50[1] == 12 and c50[2] == 14, c50
assert 75 < facts["share_center_at4"] < 82 and facts["share_high_at4"] < 6, (facts["share_center_at4"], facts["share_high_at4"])

# ---------------------------------------------------------------- inversion: on and off manifold
def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


ks = np.array([2, 4, 6, 8, 10, 14, 20, 30, 40, 64])


def invert_err(x, k):
    Vk = Vt[:k]
    return rmse(mean + ((x - mean) @ Vk.T) @ Vk, x)


off_rng = np.random.default_rng(8317)
noise_img = np.clip(off_rng.normal(X.mean(), X.std(), 64), 0, 16)
check = (np.indices((8, 8)).sum(axis=0) % 2 * 16.0).reshape(64)
frame = np.zeros((8, 8)); frame[0, :] = frame[-1, :] = frame[:, 0] = frame[:, -1] = 16.0
frame = frame.reshape(64)

curve_digit = np.array([np.mean([invert_err(Xte[i], k) for i in range(120)]) for k in ks])
curve_check = np.array([invert_err(check, k) for k in ks])
curve_frame = np.array([invert_err(frame, k) for k in ks])

facts["inv_digit_k20"] = float(curve_digit[ks.tolist().index(20)])
facts["inv_check_k20"] = float(curve_check[ks.tolist().index(20)])
facts["inv_frame_k20"] = float(curve_frame[ks.tolist().index(20)])
facts["inv_ratio_check"] = facts["inv_check_k20"] / facts["inv_digit_k20"]
assert 1.3 < facts["inv_digit_k20"] < 1.45, facts["inv_digit_k20"]
assert 6.5 < facts["inv_ratio_check"] < 6.8, facts["inv_ratio_check"]

# ---------------------------------------------------------------- editing direction: ink axis
Wtr = code(Xtr)
ink_tr = np.array([f_ink(x) for x in Xtr])
A = np.hstack([Wtr, np.ones((len(Wtr), 1))])
coef, *_ = np.linalg.lstsq(A, ink_tr, rcond=None)
d = coef[:K] / np.linalg.norm(coef[:K])
facts["ink_r2"] = float(1 - np.sum((A @ coef - ink_tr) ** 2) / np.sum((ink_tr - ink_tr.mean()) ** 2))
assert 0.95 < facts["ink_r2"] < 0.97, facts["ink_r2"]
sigma_d = float(np.std(Wtr @ d))
facts["sigma_d"] = sigma_d


# ---------------------------------------------------------------- softmax classifier (numpy)
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_softmax(Xa, ya, *, epochs=60, lr=0.5, seed=0, attack=None, eps=0.0):
    r = np.random.default_rng(seed)
    Wm = np.zeros((Xa.shape[1], 10)); bm = np.zeros(10)
    Y = np.eye(10)[ya]
    n = len(Xa)
    for ep in range(epochs):
        order = r.permutation(n)
        for s in range(0, n, 64):
            j = order[s:s + 64]
            xb, yb = Xa[j], Y[j]
            if attack is not None and eps > 0:
                xb = attack(xb, ya[j], Wm, bm, eps)
            p = softmax(xb @ Wm + bm)
            g = (p - yb) / len(j)
            Wm -= lr * (xb.T @ g + 1e-4 * Wm)
            bm -= lr * g.sum(axis=0)
    return Wm, bm


def acc(Xa, ya, Wm, bm):
    return float(np.mean((Xa @ Wm + bm).argmax(axis=1) == ya))


def grad_x(xb, yb, Wm, bm):
    p = softmax(xb @ Wm + bm)
    p[np.arange(len(yb)), yb] -= 1.0
    return p @ Wm.T


def fgsm(xb, yb, Wm, bm, eps):
    return np.clip(xb + eps * np.sign(grad_x(xb, yb, Wm, bm)), 0, 1)


def pgd(xb, yb, Wm, bm, eps, steps=10):
    a = eps / 4
    x0 = xb
    xa = np.clip(xb + np.random.default_rng(7).uniform(-eps, eps, xb.shape), 0, 1)
    for _ in range(steps):
        xa = xa + a * np.sign(grad_x(xa, yb, Wm, bm))
        xa = np.clip(np.clip(xa, x0 - eps, x0 + eps), 0, 1)
    return xa


def rand_noise(xb, yb, Wm, bm, eps):
    r = np.random.default_rng(11)
    return np.clip(xb + eps * r.choice([-1.0, 1.0], xb.shape), 0, 1)


Xtr1, Xte1 = Xtr / 16.0, Xte / 16.0
Wm, bm = train_softmax(Xtr1, ytr, seed=83)
facts["clean_acc"] = acc(Xte1, yte, Wm, bm) * 100
assert 97.0 < facts["clean_acc"] < 97.8, facts["clean_acc"]

eps_grid = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20])
acc_fgsm, acc_pgd, acc_rnd = [], [], []
for e in eps_grid:
    if e == 0:
        acc_fgsm.append(facts["clean_acc"]); acc_pgd.append(facts["clean_acc"]); acc_rnd.append(facts["clean_acc"]); continue
    acc_fgsm.append(acc(fgsm(Xte1, yte, Wm, bm, e), yte, Wm, bm) * 100)
    acc_pgd.append(acc(pgd(Xte1, yte, Wm, bm, e), yte, Wm, bm) * 100)
    acc_rnd.append(acc(rand_noise(Xte1, yte, Wm, bm, e), yte, Wm, bm) * 100)
acc_fgsm = np.array(acc_fgsm); acc_pgd = np.array(acc_pgd); acc_rnd = np.array(acc_rnd)

i8 = eps_grid.tolist().index(0.15)
facts["eps_show"] = 0.15
facts["acc_fgsm_15"] = float(acc_fgsm[i8])
facts["acc_pgd_15"] = float(acc_pgd[i8])
facts["acc_rnd_15"] = float(acc_rnd[i8])
assert 34 < facts["acc_fgsm_15"] < 37 and 93 < facts["acc_rnd_15"] < 95.5, (facts["acc_fgsm_15"], facts["acc_rnd_15"])
assert facts["acc_pgd_15"] <= facts["acc_fgsm_15"] + 1e-9

# norms of one perturbation
xadv = fgsm(Xte1, yte, Wm, bm, 0.15)
delta = xadv - Xte1
facts["linf"] = float(np.abs(delta).max())
facts["l2_mean"] = float(np.linalg.norm(delta, axis=1).mean())
facts["l2_bound"] = float(0.15 * np.sqrt(64))
assert abs(facts["linf"] - 0.15) < 1e-9
assert facts["l2_mean"] < facts["l2_bound"] + 1e-9

# ---------------------------------------------------------------- adversarial training
Wr, br = train_softmax(Xtr1, ytr, seed=83, attack=fgsm, eps=0.15)
facts["rob_clean_acc"] = acc(Xte1, yte, Wr, br) * 100
facts["rob_pgd_acc"] = acc(pgd(Xte1, yte, Wr, br, 0.15), yte, Wr, br) * 100
facts["std_pgd_acc"] = facts["acc_pgd_15"]
facts["rob_gain"] = facts["rob_pgd_acc"] - facts["std_pgd_acc"]
facts["clean_cost"] = facts["clean_acc"] - facts["rob_clean_acc"]
assert 23 < facts["rob_gain"] < 26, facts["rob_gain"]
assert facts["clean_cost"] > 0, facts["clean_cost"]

# transfer attack: examples built on the standard model, fed to the robust one
xtr_adv = pgd(Xte1, yte, Wm, bm, 0.15)
facts["transfer_acc"] = acc(xtr_adv, yte, Wr, br) * 100
assert facts["transfer_acc"] > facts["rob_pgd_acc"], (facts["transfer_acc"], facts["rob_pgd_acc"])

# ---------------------------------------------------------------- identity cost of editing
alphas = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
sample = [int(i) for i in rng.choice(len(Xte), 200, replace=False)]
keep, inkchg = [], []
for al in alphas:
    ok, dk = 0, []
    for i in sample:
        w = code(Xte[i])
        base = np.clip(gen(w), 0, 16)
        edt = np.clip(gen(w + al * sigma_d * d), 0, 16)
        p0 = (base / 16.0) @ Wm + bm
        p1 = (edt / 16.0) @ Wm + bm
        ok += int(p0.argmax() == p1.argmax())
        dk.append(f_ink(edt) - f_ink(base))
    keep.append(ok / len(sample) * 100); inkchg.append(float(np.mean(dk)))
keep = np.array(keep); inkchg = np.array(inkchg)
facts["keep_a1"] = float(keep[alphas.tolist().index(1.0)])
facts["keep_a3"] = float(keep[alphas.tolist().index(3.0)])
facts["keep_a8"] = float(keep[alphas.tolist().index(8.0)])
assert facts["keep_a8"] < facts["keep_a1"] and facts["keep_a8"] > 88, facts["keep_a8"]
facts["ink_a3"] = float(inkchg[alphas.tolist().index(3.0)])
facts["ink_a8"] = float(inkchg[alphas.tolist().index(8.0)])
assert 3.0 < facts["ink_a8"] < 3.3, facts["ink_a8"]
assert facts["keep_a1"] > 90 and facts["keep_a3"] < facts["keep_a1"], (facts["keep_a1"], facts["keep_a3"])
assert facts["ink_a3"] > 0

# cycle test: +alpha then -alpha with clipping to the valid pixel range
cyc = {}
for k in (10, 40):
    Vk = Vt[:k]
    errs_by_a = []
    for al in alphas:
        dk = np.zeros(k); dk[:min(K, k)] = d[:min(K, k)]
        if np.linalg.norm(dk) > 0:
            dk = dk / np.linalg.norm(dk)
        e = []
        for i in sample[:120]:
            w = (Xte[i] - mean) @ Vk.T
            up = np.clip(mean + (w + al * sigma_d * dk) @ Vk, 0, 16)
            wu = (up - mean) @ Vk.T
            back = np.clip(mean + (wu - al * sigma_d * dk) @ Vk, 0, 16)
            e.append(rmse(back, np.clip(mean + w @ Vk, 0, 16)))
        errs_by_a.append(float(np.mean(e)))
    cyc[k] = np.array(errs_by_a)
facts["cycle10_a3"] = float(cyc[10][alphas.tolist().index(3.0)])
facts["cycle40_a3"] = float(cyc[40][alphas.tolist().index(3.0)])
facts["cycle_recon10"] = float(np.mean([invert_err(Xte[i], 10) for i in sample[:120]]))
facts["cycle_recon40"] = float(np.mean([invert_err(Xte[i], 40) for i in sample[:120]]))
assert facts["cycle_recon40"] < facts["cycle_recon10"]
assert facts["cycle40_a3"] > facts["cycle10_a3"], (facts["cycle10_a3"], facts["cycle40_a3"])


# ================================================================ figures
def fig_scales():
    fig, axes = plt.subplots(2, 5, figsize=(9.6, 5.0))
    show(axes[0, 0], mean, "средний образ\n(константа)", color=MUTED)
    for j in range(4):
        show(axes[0, j + 1], mean + 3.2 * np.sqrt(lam[j]) * Vt[j],
             f"+ ось {j + 1}\n{var_share[j] * 100:.1f}% дисперсии", color=BLUE)
    show(axes[1, 0], mean, "тот же старт", color=MUTED)
    for j in range(4):
        c = 16 + j
        show(axes[1, j + 1], mean + 3.2 * np.sqrt(lam[c]) * Vt[c],
             f"+ ось {c + 1}\n{var_share[c] * 100:.1f}% дисперсии", color=RED)
    fig.suptitle("Ранние координаты двигают геометрию, поздние правят детали", y=1.02, fontsize=13.5)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.42)
    save(fig, OUT / "latent_scales.png")


def fig_mixing_grid():
    show_cuts = [0, 2, 4, 6, 10, 20]
    rows = pairs[:3]
    fig, axes = plt.subplots(len(rows), len(show_cuts) + 2, figsize=(10.4, 5.0))
    for r, (a, b) in enumerate(rows):
        wa, wb = code(Xte[a]), code(Xte[b])
        show(axes[r, 0], np.clip(gen(wb), 0, 16), "источник B" if r == 0 else None, color=GREEN)
        for j, c in enumerate(show_cuts):
            w = np.concatenate([wa[:c], wb[c:]])
            show(axes[r, j + 1], np.clip(gen(w), 0, 16), f"c = {c}" if r == 0 else None, color=INK)
        show(axes[r, -1], np.clip(gen(wa), 0, 16), "источник A" if r == 0 else None, color=BLUE)
    fig.suptitle("Смешение кодов: слева всё от B, справа всё от A", y=1.03, fontsize=13.5)
    fig.text(0.5, -0.02, "c — число ранних координат, взятых у A", ha="center", fontsize=10, color=MUTED)
    fig.tight_layout()
    save(fig, OUT / "mixing_grid.png")


def fig_mixing_curves():
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for fi, (name, _, col) in enumerate(FEATS):
        ax.plot(cuts, shares[fi] * 100, color=col, lw=2.2, marker="o", ms=3.4, label=name)
        ax.axvline(c50[fi], color=col, lw=0.9, ls=(0, (3, 3)), alpha=0.7)
    ax.axhline(50, color=MUTED, lw=0.9, ls=(0, (2, 2)))
    ax.set_xlabel("граница смешения c (сколько ранних координат взято у A)")
    ax.set_ylabel("доля признака, унаследованная от A, %")
    ax.set_title("Крупный признак переходит к A рано, мелкий — поздно")
    ax.set_xticks(range(0, K + 1, 2))
    ax.annotate(f"c = {c50[0]}", (c50[0], 8), color=BLUE, fontsize=10, ha="left")
    ax.annotate(f"c = {c50[2]}", (c50[2], 8), color=RED, fontsize=10, ha="left")
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "mixing_curves.png")


def fig_inversion():
    fig = plt.figure(figsize=(10.6, 4.4))
    gs = fig.add_gridspec(3, 5, width_ratios=[0.55, 0.55, 0.12, 1.9, 0.02])
    samples = [(Xte[pairs[0][0]], "цифра", BLUE), (check, "шахматка", RED), (frame, "рамка", GOLD)]
    for r, (img, name, col) in enumerate(samples):
        a0 = fig.add_subplot(gs[r, 0]); show(a0, img, "оригинал" if r == 0 else None, color=col)
        a1 = fig.add_subplot(gs[r, 1])
        Vk = Vt[:K]
        show(a1, np.clip(mean + ((img - mean) @ Vk.T) @ Vk, 0, 16), "инверсия" if r == 0 else None, color=col)
        a0.set_ylabel(name, fontsize=9, color=col)
    ax = fig.add_subplot(gs[:, 3])
    ax.plot(ks, curve_digit, color=BLUE, lw=2.2, marker="o", ms=4, label="реальные цифры (среднее по 120)")
    ax.plot(ks, curve_check, color=RED, lw=2.2, marker="s", ms=4, label="шахматка (вне многообразия)")
    ax.plot(ks, curve_frame, color=GOLD, lw=2.2, marker="^", ms=4, label="рамка (вне многообразия)")
    ax.axvline(K, color=MUTED, lw=0.9, ls=(0, (2, 2)))
    ax.set_xlabel("размерность латента k"); ax.set_ylabel("ошибка восстановления, RMSE пикселя")
    ax.set_title("Что генератор умеет повторить, а что — нет", fontsize=13)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "inversion.png")


def fig_adversarial():
    i = int(np.argmax((Xte1 @ Wm + bm).argmax(axis=1) == yte))
    x0 = Xte1[i:i + 1]; y0 = yte[i:i + 1]
    xa = fgsm(x0, y0, Wm, bm, 0.15)
    dl = xa - x0
    p0 = softmax(x0 @ Wm + bm)[0]; p1 = softmax(xa @ Wm + bm)[0]
    facts["ex_true"] = int(y0[0]); facts["ex_adv"] = int(p1.argmax())
    facts["ex_p0"] = float(p0.max()); facts["ex_p1"] = float(p1.max())
    assert facts["ex_true"] == 7 and facts["ex_adv"] == 2, (facts["ex_true"], facts["ex_adv"])
    assert facts["ex_p0"] > 0.98 and 0.45 < facts["ex_p1"] < 0.6, (facts["ex_p0"], facts["ex_p1"])
    fig = plt.figure(figsize=(10.6, 4.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.7, 0.7, 0.7, 2.3])
    for j, (img, ttl, col) in enumerate([
            (x0[0] * 16, f"исход: {int(p0.argmax())}\np = {p0.max():.2f}", BLUE),
            ((dl[0] * 10 + 0.5) * 16, "возмущение ×10", MUTED),
            (xa[0] * 16, f"после атаки: {int(p1.argmax())}\np = {p1.max():.2f}", RED)]):
        a = fig.add_subplot(gs[0, j]); show(a, img, ttl, fs=9.5, color=col)
    ax = fig.add_subplot(gs[0, 3])
    ax.plot(eps_grid, acc_fgsm, color=RED, lw=2.2, marker="o", ms=4, label="FGSM (один шаг по знаку градиента)")
    ax.plot(eps_grid, acc_pgd, color=VIOLET, lw=2.2, marker="s", ms=4, label="PGD, 10 шагов")
    ax.plot(eps_grid, acc_rnd, color=GREEN, lw=2.2, marker="^", ms=4, label="случайный шум той же нормы")
    ax.axvline(0.15, color=MUTED, lw=0.9, ls=(0, (2, 2)))
    ax.set_xlabel(r"бюджет $\varepsilon$ по норме $\ell_\infty$ (доля шкалы яркости)")
    ax.set_ylabel("точность на тесте, %")
    ax.set_title("Одинаковая норма, разный результат", fontsize=13)
    ax.legend(loc="lower left", frameon=False, fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "adversarial.png")


def fig_defense():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.3))
    names = ["обычная\nмодель", "обученная\nна атаках"]
    clean = [facts["clean_acc"], facts["rob_clean_acc"]]
    rob = [facts["std_pgd_acc"], facts["rob_pgd_acc"]]
    xpos = np.arange(2)
    ax.bar(xpos - 0.19, clean, width=0.36, color=BLUE, label="чистая точность")
    ax.bar(xpos + 0.19, rob, width=0.36, color=RED, label=r"под PGD, $\varepsilon=0{,}15$")
    for k in range(2):
        ax.text(xpos[k] - 0.19, clean[k] + 1.6, f"{clean[k]:.1f}", ha="center", fontsize=10, color=BLUE)
        ax.text(xpos[k] + 0.19, rob[k] + 1.6, f"{rob[k]:.1f}", ha="center", fontsize=10, color=RED)
    ax.set_xticks(xpos); ax.set_xticklabels(names); ax.set_ylim(0, 122)
    ax.set_ylabel("точность, %"); ax.set_title("Устойчивость покупается точностью", fontsize=13)
    ax.legend(loc="upper center", frameon=False, fontsize=9.5, ncol=2)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)

    ax2.plot(alphas, keep, color=VIOLET, lw=2.2, marker="o", ms=4, label="сохранена исходная цифра, %")
    ax2b = ax2.twinx()
    ax2b.plot(alphas, inkchg, color=GOLD, lw=2.2, marker="s", ms=4, label="прирост чернил")
    ax2b.set_ylabel("средний прирост чернил", color=GOLD)
    ax2b.tick_params(axis="y", colors=GOLD)
    ax2.set_xlabel(r"шаг вдоль направления $\alpha$")
    ax2.set_ylabel("идентичность сохранена, %", color=VIOLET)
    ax2.tick_params(axis="y", colors=VIOLET)
    ax2.set_title("У редактирования есть рабочий диапазон", fontsize=13)
    ax2.grid(True, color=GRID, lw=0.4, alpha=0.4); ax2.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "editing_defense.png")


# ---------------------------------------------------------------- margins
def side_step():
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    ax.bar(np.arange(1, K + 1), sq[:K], color=[BLUE if i < 4 else (VIOLET if i < 12 else RED) for i in range(K)])
    ax.set_yscale("log")
    ax.set_xlabel("номер координаты", fontsize=9)
    ax.set_ylabel("длина шага в пикселях (лог)", fontsize=8.5)
    ax.set_title(f"одинаковый шаг в латенте —\nразный сдвиг картинки (×{facts['step_ratio']:.1f})", fontsize=9)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "step.png")


def side_cycle():
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    ax.plot(alphas, cyc[10], color=BLUE, lw=1.9, marker="o", ms=3.4, label="k = 10")
    ax.plot(alphas, cyc[40], color=RED, lw=1.9, marker="s", ms=3.4, label="k = 40")
    ax.set_xlabel(r"$\alpha$ туда и обратно", fontsize=9)
    ax.set_ylabel("ошибка возврата", fontsize=9)
    ax.set_title("богатый латент лучше\nвосстанавливает, хуже возвращается", fontsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "cycle.png")


def side_balls():
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False, ec=RED, lw=2.0))
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color=BLUE, lw=2.0)
    ax.plot([0, 1], [0, 1], color=GOLD, lw=1.6)
    ax.plot([0, 1], [0, 0], color=MUTED, lw=1.2, ls=(0, (3, 3)))
    ax.annotate(r"$\ell_\infty \leq \varepsilon$", (1.02, 1.06), color=RED, fontsize=10)
    ax.annotate(r"$\ell_2 \leq \varepsilon$", (0.62, -0.86), color=BLUE, fontsize=10)
    ax.annotate(r"угол: $\sqrt{d}\,\varepsilon$", (0.36, 0.52), color=GOLD, fontsize=9.5, rotation=45)
    ax.set_xlim(-1.5, 1.6); ax.set_ylim(-1.35, 1.4)
    ax.set_title(r"куб шире шара в $\sqrt{d}$ раз", fontsize=9.5)
    save(fig, SIDE / "balls.png")


fig_scales()
fig_mixing_grid()
fig_mixing_curves()
fig_inversion()
fig_adversarial()
fig_defense()
side_step()
side_cycle()
side_balls()

FACTS.write_text(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in facts.items()},
                            ensure_ascii=False, indent=1))
for k, v in facts.items():
    print(f"{k:>18} = {v}")
print("lesson 83 figures written")
