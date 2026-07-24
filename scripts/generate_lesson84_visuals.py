"""Deterministic figures and asserted numbers for lesson 84: contrastive alignment
of two modalities in one shared space (CLIP-like InfoNCE).

Real data: sklearn `load_digits` (1797 handwritten digits, 8x8). Each image is split
into two "modalities": the EVEN scan lines are the "snapshot", the ODD scan lines are the
"caption" — two different sensors describing one and the same object. Two small encoders
are trained with a symmetric InfoNCE loss to put the two views of one object next to each
other in a common 32-dimensional space. This is an honest model example of a cross-modal
pair, not a real text corpus, and the lesson says so explicitly.

Everything quoted in the prose is computed here and asserted.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "84"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "84"
FACTS = ROOT / "scripts" / "data" / "lesson84_facts.json"

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

F: dict[str, float] = {}


def save(fig, path, *, dpi=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------- data
def load_pairs():
    dig = load_digits()
    images = dig.images                      # (1797, 8, 8), values 0..16
    labels = dig.target
    left = images[:, ::2, :].reshape(len(images), -1)   # even scan lines: the "snapshot"
    right = images[:, 1::2, :].reshape(len(images), -1)  # odd scan lines: the "caption"
    rng = np.random.default_rng(84)
    perm = rng.permutation(len(images))
    ntr = 1200
    tr, te = perm[:ntr], perm[ntr:]

    def std(a, mu, sd):
        return (a - mu) / sd

    # a floor on the scale: some border pixels are almost always zero and dividing
    # by their tiny std would turn any shift of the sensor into an astronomic number
    mu_l, sd_l = left[tr].mean(0), np.maximum(left[tr].std(0), 1.0)
    mu_r, sd_r = right[tr].mean(0), np.maximum(right[tr].std(0), 1.0)
    data = dict(
        Xl_tr=std(left[tr], mu_l, sd_l), Xr_tr=std(right[tr], mu_r, sd_r),
        Xl_te=std(left[te], mu_l, sd_l), Xr_te=std(right[te], mu_r, sd_r),
        y_tr=labels[tr], y_te=labels[te],
        images_te=images[te], images_tr=images[tr],
        norm=(mu_l, sd_l, mu_r, sd_r),
    )
    return data


def unit(a):
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)


def infonce_grad(A, Bm, tau, pos_mask=None):
    """Symmetric InfoNCE. Returns loss and gradients wrt A, Bm (pre-normalisation)."""
    n = len(A)
    na = np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    nb = np.linalg.norm(Bm, axis=1, keepdims=True) + 1e-12
    a, b = A / na, Bm / nb
    S = a @ b.T / tau
    if pos_mask is None:
        pos_mask = np.eye(n)
    pos = pos_mask / pos_mask.sum(1, keepdims=True)          # rows
    posc = pos_mask / pos_mask.sum(0, keepdims=True)         # cols

    P = np.exp(S - S.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    Q = np.exp(S - S.max(0, keepdims=True))
    Q /= Q.sum(0, keepdims=True)

    l_row = -np.mean(np.log((P * pos_mask).sum(1) + 1e-12))
    l_col = -np.mean(np.log((Q * pos_mask).sum(0) + 1e-12))
    loss = 0.5 * (l_row + l_col)

    G = 0.5 * ((P - pos) + (Q - posc)) / n
    dA_hat = G @ b / tau
    dB_hat = G.T @ a / tau
    dA = (dA_hat - (dA_hat * a).sum(1, keepdims=True) * a) / na
    dB = (dB_hat - (dB_hat * b).sum(1, keepdims=True) * b) / nb
    return loss, dA, dB


HIDDEN, DIM = 64, 32


def init_encoder(rng, d_in):
    return [rng.normal(0, np.sqrt(2 / d_in), (d_in, HIDDEN)),
            rng.normal(0, np.sqrt(1 / HIDDEN), (HIDDEN, DIM))]


def enc_forward(W, x):
    z = x @ W[0]
    h = np.maximum(z, 0)
    return h @ W[1], (z, h)


def enc_backward(W, x, cache, dA):
    z, h = cache
    g1 = h.T @ dA
    dh = dA @ W[1].T
    dz = dh * (z > 0)
    g0 = x.T @ dz
    return [g0, g1]


def train(data, *, tau=0.07, batch=64, steps=6000, seed=84, class_positive=False):
    rng = np.random.default_rng(seed)
    Xl, Xr, y = data["Xl_tr"], data["Xr_tr"], data["y_tr"]
    Wv = init_encoder(rng, Xl.shape[1])
    Wu = init_encoder(rng, Xr.shape[1])
    state = [[np.zeros_like(p) for p in W] for W in (Wv, Wu)]
    state2 = [[np.zeros_like(p) for p in W] for W in (Wv, Wu)]
    lr, b1, b2, eps = 0.01, 0.9, 0.999, 1e-8
    n = len(Xl)
    losses = []
    for t in range(1, steps + 1):
        idx = rng.choice(n, batch, replace=False)
        xl, xr = Xl[idx], Xr[idx]
        A, ca = enc_forward(Wv, xl)
        Bm, cb = enc_forward(Wu, xr)
        mask = None
        if class_positive:
            mask = (y[idx][:, None] == y[idx][None, :]).astype(float)
        loss, dA, dB = infonce_grad(A, Bm, tau, mask)
        grads = (enc_backward(Wv, xl, ca, dA), enc_backward(Wu, xr, cb, dB))
        for wi, W in enumerate((Wv, Wu)):
            for k in range(2):
                g = grads[wi][k]
                m, v = state[wi][k], state2[wi][k]
                m *= b1; m += (1 - b1) * g
                v *= b2; v += (1 - b2) * g * g
                W[k] -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        losses.append(loss)
    return Wv, Wu, np.array(losses)


def embed(W, x):
    return enc_forward(W, x)[0]


def evaluate(data, Wv, Wu):
    v = unit(embed(Wv, data["Xl_te"]))
    u = unit(embed(Wu, data["Xr_te"]))
    S = v @ u.T
    n = len(S)
    order = np.argsort(-S, axis=1)
    ranks = np.array([np.where(order[i] == i)[0][0] for i in range(n)]) + 1
    top1 = order[:, 0]
    y = data["y_te"]
    res = dict(
        n=n,
        r1=float(np.mean(ranks == 1)), r5=float(np.mean(ranks <= 5)),
        r10=float(np.mean(ranks <= 10)),
        med_rank=float(np.median(ranks)),
        class_top1=float(np.mean(y[top1] == y)),
    )
    wrong = ranks > 1
    res["same_class_among_errors"] = float(np.mean(y[top1[wrong]] == y[wrong]))
    res["S"] = S
    res["v"] = v; res["u"] = u; res["ranks"] = ranks
    return res


def prototypes(data, Wv, Wu):
    """Zero-shot: class prototype in the 'text' space = mean of caption-embeddings."""
    u_tr = unit(embed(Wu, data["Xr_tr"]))
    v_te = unit(embed(Wv, data["Xl_te"]))
    y_tr, y_te = data["y_tr"], data["y_te"]
    P = np.stack([unit(u_tr[y_tr == c].mean(0)[None, :])[0] for c in range(10)])
    acc_ens = float(np.mean(np.argmax(v_te @ P.T, axis=1) == y_te))
    rng = np.random.default_rng(3)
    accs = []
    for _ in range(40):
        Q = np.stack([u_tr[rng.choice(np.where(y_tr == c)[0])] for c in range(10)])
        accs.append(np.mean(np.argmax(v_te @ Q.T, axis=1) == y_te))
    return acc_ens, float(np.mean(accs)), float(np.std(accs)), P, v_te


# ================================================================ main run
data = load_pairs()
F["n_total"] = int(len(data["y_tr"]) + len(data["y_te"]))
F["n_train"] = int(len(data["y_tr"]))
F["n_test"] = int(len(data["y_te"]))
assert F["n_total"] == 1797 and F["n_train"] == 1200 and F["n_test"] == 597

TAU, BATCH, STEPS = 0.1, 64, 4000
Wv, Wu, losses = train(data, tau=TAU, batch=BATCH, steps=STEPS)
base = evaluate(data, Wv, Wu)
F["tau"] = TAU; F["batch"] = BATCH; F["dim"] = DIM; F["steps"] = STEPS
F["hidden"] = HIDDEN
F["epochs"] = round(STEPS * BATCH / F["n_train"])
F["r1"] = round(base["r1"] * 100, 1)
F["r5"] = round(base["r5"] * 100, 1)
F["r10"] = round(base["r10"] * 100, 1)
F["med_rank"] = base["med_rank"]
F["class_top1"] = round(base["class_top1"] * 100, 1)
F["same_class_among_errors"] = round(base["same_class_among_errors"] * 100, 1)
F["random_r1"] = round(100 / F["n_test"], 2)
print("base:", {k: F[k] for k in ("r1", "r5", "r10", "med_rank", "class_top1", "same_class_among_errors")})
assert F["r1"] > 5 * F["random_r1"], "instance retrieval must beat chance a lot"
assert F["class_top1"] > 70, "class-level match should be high"
assert F["same_class_among_errors"] > 30, "many top-1 errors are same-class"

# ---- positive vs negative similarity distributions
S = base["S"]
pos_sim = np.diag(S)
off = ~np.eye(len(S), dtype=bool)
neg_sim = S[off]
F["pos_mean"] = round(float(pos_sim.mean()), 3)
F["neg_mean"] = round(float(neg_sim.mean()), 3)
F["neg_max_mean"] = round(float(S[off].reshape(len(S), -1).max(1).mean()), 3)
assert F["pos_mean"] > F["neg_mean"] + 0.3

# ---- margin / selective retrieval
srt = np.sort(S, axis=1)
margin = srt[:, -1] - srt[:, -2]
correct = base["ranks"] == 1
o = np.argsort(-margin)
cov = np.arange(1, len(o) + 1) / len(o)
acc_cov = np.cumsum(correct[o]) / np.arange(1, len(o) + 1)
i20 = int(0.2 * len(o)) - 1
F["acc_at_cov20"] = round(float(acc_cov[i20] * 100), 1)
F["margin_median"] = round(float(np.median(margin)), 3)
assert F["acc_at_cov20"] > F["r1"], "selective retrieval must be better than blanket top-1"

# ---- temperature sweep
taus = [0.02, 0.05, 0.07, 0.1, 0.2, 0.5, 1.0]
tau_r1, tau_cls = [], []
for t in taus:
    w1, w2, _ = train(data, tau=t, batch=BATCH, steps=STEPS)
    ev = evaluate(data, w1, w2)
    tau_r1.append(ev["r1"] * 100); tau_cls.append(ev["class_top1"] * 100)
F["tau_grid"] = taus
F["tau_r1"] = [round(x, 1) for x in tau_r1]
F["tau_cls"] = [round(x, 1) for x in tau_cls]
F["tau_best"] = taus[int(np.argmax(tau_r1))]
F["tau_r1_best"] = round(max(tau_r1), 1)
F["tau_r1_at_1"] = round(tau_r1[-1], 1)
F["tau_cls_best"] = taus[int(np.argmax(tau_cls))]
F["tau_cls_best_val"] = round(max(tau_cls), 1)
print("tau sweep:", list(zip(taus, F["tau_r1"], F["tau_cls"])))
assert F["tau_r1_best"] > F["tau_r1_at_1"] + 5, "large tau must hurt"

# ---- batch-size sweep at an EQUAL number of passes over the data
batches = [8, 16, 32, 64, 128, 256]
bat_r1 = []
for b in batches:
    st = int(round(F["epochs"] * F["n_train"] / b))
    w1, w2, _ = train(data, tau=TAU, batch=b, steps=st)
    bat_r1.append(evaluate(data, w1, w2)["r1"] * 100)
F["batch_grid"] = batches
F["batch_r1"] = [round(x, 1) for x in bat_r1]
F["batch_best"] = batches[int(np.argmax(bat_r1))]
F["batch_r1_best"] = round(max(bat_r1), 1)
F["batch_r1_8"] = round(bat_r1[0], 1)
F["batch_r1_256"] = round(bat_r1[-1], 1)
F["batch_spread"] = round(max(bat_r1) - min(bat_r1), 1)
print("batch sweep:", list(zip(batches, F["batch_r1"])))
assert F["batch_spread"] < 15, "on a small collection batch size is not decisive"
assert F["batch_r1_256"] < F["batch_r1_best"], "the biggest batch is not the best here"

# ---- how many candidates: the same model, growing pool
pool_sizes = [8, 16, 32, 64, 128, 256, 597]
rngp = np.random.default_rng(21)
pool_r1 = []
for m in pool_sizes:
    hits = []
    for _ in range(300):
        idx = rngp.choice(len(base["S"]), m, replace=False)
        sub = base["S"][np.ix_(idx, idx)]
        hits.append(np.mean(np.argmax(sub, axis=1) == np.arange(m)))
    pool_r1.append(float(np.mean(hits)) * 100)
F["pool_sizes"] = pool_sizes
F["pool_r1"] = [round(x, 1) for x in pool_r1]
F["pool_r1_8"] = round(pool_r1[0], 1)
F["pool_r1_full"] = round(pool_r1[-1], 1)
print("pool sweep:", list(zip(pool_sizes, F["pool_r1"])))
assert F["pool_r1_8"] > F["pool_r1_full"] + 15, "a small pool flatters retrieval"

# ---- false negatives inside a batch of 64
rng = np.random.default_rng(7)
cnt = []
for _ in range(2000):
    idx = rng.choice(len(data["y_tr"]), BATCH, replace=False)
    yy = data["y_tr"][idx]
    same = (yy[:, None] == yy[None, :]).sum(1) - 1
    cnt.append(same.mean())
F["false_neg_per_row"] = round(float(np.mean(cnt)), 2)
F["false_neg_share"] = round(float(np.mean(cnt)) / (BATCH - 1) * 100, 1)
assert 4.0 < F["false_neg_per_row"] < 8.5

# ---- gradient concentration: share of the hardest negative
def hard_share(tau):
    Z = S / tau
    P = np.exp(Z - Z.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    shares = []
    for i in range(len(P)):
        row = np.delete(P[i], i)
        shares.append(row.max() / row.sum())
    return float(np.mean(shares))


F["hard_share_002"] = round(hard_share(0.02) * 100, 1)
F["hard_share_05"] = round(hard_share(0.5) * 100, 1)
print("hard share:", F["hard_share_002"], F["hard_share_05"])
assert F["hard_share_002"] > 2 * F["hard_share_05"]

# ---- multi-positive (class-positive) training
Wv_mp, Wu_mp, _ = train(data, tau=TAU, batch=BATCH, steps=STEPS, class_positive=True)
mp = evaluate(data, Wv_mp, Wu_mp)
F["mp_r1"] = round(mp["r1"] * 100, 1)
F["mp_class_top1"] = round(mp["class_top1"] * 100, 1)
print("multi-positive:", F["mp_r1"], F["mp_class_top1"])
assert F["mp_class_top1"] > F["class_top1"], "class-positives improve class match"
assert F["mp_r1"] < F["r1"], "class-positives blur instance identity"

# ---- zero-shot prototypes: single exemplar vs ensemble
acc_ens, acc_one, acc_one_sd, P, v_te = prototypes(data, Wv, Wu)
F["proto_ens"] = round(acc_ens * 100, 1)
F["proto_one"] = round(acc_one * 100, 1)
F["proto_one_sd"] = round(acc_one_sd * 100, 1)
print("prototype:", F["proto_ens"], F["proto_one"], "+-", F["proto_one_sd"])
assert F["proto_ens"] > F["proto_one"] + 3, "ensemble of prompts must help"

# ---- domain shift: the same digits, shifted by one pixel
def shifted(data, Wv, Wu, roll=1):
    img = np.roll(data["images_te"], roll, axis=2)
    left = img[:, ::2, :].reshape(len(img), -1)
    right = img[:, 1::2, :].reshape(len(img), -1)
    mu_l, sd_l, mu_r, sd_r = data["norm"]
    xl = (left - mu_l) / sd_l
    xr = (right - mu_r) / sd_r
    d2 = dict(data)
    d2["Xl_te"], d2["Xr_te"] = xl, xr
    ev = evaluate(d2, Wv, Wu)
    raw_norm = np.linalg.norm(embed(Wv, xl), axis=1).mean()
    return ev, raw_norm


ev_sh, norm_sh = shifted(data, Wv, Wu)
norm_ok = np.linalg.norm(embed(Wv, data["Xl_te"]), axis=1).mean()
F["shift_r1"] = round(ev_sh["r1"] * 100, 1)
F["shift_class_top1"] = round(ev_sh["class_top1"] * 100, 1)
F["norm_ratio"] = round(float(norm_sh / norm_ok), 2)
proto_shift = float(np.mean(np.argmax(ev_sh["v"] @ P.T, axis=1) == data["y_te"]))
F["proto_shift"] = round(proto_shift * 100, 1)
print("shift:", F["shift_r1"], F["shift_class_top1"], F["norm_ratio"], F["proto_shift"])
assert F["shift_r1"] < F["r1"] - 5 and F["proto_shift"] < F["proto_ens"] - 5

# ---- worked example numbers used in the prose / exercises
s_tau1 = np.array([1.0, 0.0, -1.0])
for name, t in (("t1", 1.0), ("t05", 0.5)):
    z = s_tau1 / t
    p = np.exp(z - z.max()); p /= p.sum()
    F[f"toy_p_{name}"] = [round(float(x), 3) for x in p]
    F[f"toy_loss_{name}"] = round(float(-np.log(p[0])), 3)
assert F["toy_loss_t05"] < F["toy_loss_t1"]

# batch of 4, diagonal (3,2,4,1), off-diagonal 0
diag = np.array([3.0, 2.0, 4.0, 1.0])
Sm = np.zeros((4, 4)); Sm[np.arange(4), np.arange(4)] = diag
row_loss = [-np.log(np.exp(Sm[i, i]) / np.exp(Sm[i]).sum()) for i in range(4)]
F["toy4_loss"] = round(float(np.mean(row_loss)), 3)
Sm2 = Sm.copy(); Sm2[0, 1] = 2.8
row_loss2 = [-np.log(np.exp(Sm2[i, i]) / np.exp(Sm2[i]).sum()) for i in range(4)]
F["toy4_loss_hard"] = round(float(np.mean(row_loss2)), 3)
F["toy4_delta"] = round(F["toy4_loss_hard"] - F["toy4_loss"], 3)
assert F["toy4_delta"] > 0.1

print("FACTS:", json.dumps(F, ensure_ascii=False, indent=1, default=str))


# ================================================================ figures
def fig_matrix():
    """84.1 — a batch becomes a matrix of comparisons; false negatives are visible."""
    rng2 = np.random.default_rng(11)
    idx = np.sort(rng2.choice(len(data["y_te"]), 12, replace=False))
    Sb = base["S"][np.ix_(idx, idx)]
    yy = data["y_te"][idx]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.6),
                                  gridspec_kw={"width_ratios": [1.35, 1]})
    im = ax.imshow(Sb, cmap="BrBG", vmin=-1, vmax=1)
    for i in range(12):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=GREEN, lw=2.2))
        for j in range(12):
            if i != j and yy[i] == yy[j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=RED, lw=1.8, ls=(0, (2, 1))))
    ax.set_xticks(range(12)); ax.set_yticks(range(12))
    ax.set_xticklabels(yy, fontsize=8); ax.set_yticklabels(yy, fontsize=8)
    ax.set_xlabel("«подписи» (нечётные строки), цифра"); ax.set_ylabel("«снимки» (чётные строки), цифра")
    ax.set_title("матрица косинусов $B\\times B$", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    row = Sb[0]
    for t, c, lbl in ((0.02, RED, "$\\tau=0{,}02$"), (0.5, BLUE, "$\\tau=0{,}5$")):
        z = row / t
        p = np.exp(z - z.max()); p /= p.sum()
        ax2.plot(range(12), p, "o-", color=c, lw=1.8, ms=5, label=lbl)
    ax2.axvline(0, color=GREEN, lw=1.2, ls=(0, (3, 2)))
    ax2.set_xlabel("столбец строки 1"); ax2.set_ylabel("softmax-вероятность")
    ax2.set_title("одна строка при двух температурах", fontsize=12)
    ax2.legend(frameon=False, fontsize=10)
    ax2.grid(True, color=GRID, lw=0.4, alpha=0.5); ax2.set_axisbelow(True)
    fig.suptitle("Зелёное — размеченные пары, красный пунктир — «отрицательные» той же цифры", y=1.02, fontsize=12.5)
    fig.tight_layout()
    save(fig, OUT / "pairs_matrix.png")


def fig_temperature():
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    a0.plot(taus, tau_r1, "o-", color=BLUE, lw=2.2, ms=6, label="Recall@1: та же пара")
    a0.plot(taus, tau_cls, "s--", color=GOLD, lw=2.0, ms=6, label="top-1: та же цифра")
    a0.set_xscale("log")
    a0.set_xlabel(r"температура $\tau$ (лог. шкала)"); a0.set_ylabel("точность на тесте, %")
    a0.set_title("холодный softmax точнее в паре, тёплый — в классе", fontsize=12)
    a0.legend(frameon=False, fontsize=9.5, loc="lower left")
    a0.annotate(f"лучшее $\\tau={F['tau_best']}$: {F['tau_r1_best']}%",
                (F["tau_best"], F["tau_r1_best"]), textcoords="offset points",
                xytext=(6, -24), fontsize=10, color=GREEN)
    tg = np.array([0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    hs = [hard_share(t) * 100 for t in tg]
    a1.plot(tg, hs, "o-", color=RED, lw=2.2, ms=6)
    a1.set_xscale("log")
    a1.set_xlabel(r"температура $\tau$ (лог. шкала)")
    a1.set_ylabel("доля градиента у самого трудного negative, %")
    a1.set_title("холодный softmax отдаёт весь штраф одному сопернику", fontsize=12)
    for ax in (a0, a1):
        ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "temperature.png")


def fig_batch():
    fig, (ax, a1) = plt.subplots(1, 2, figsize=(11.4, 4.4))
    ax.plot(batches, bat_r1, "o-", color=GREEN, lw=2.4, ms=7, label="Recall@1, %")
    ax.set_xscale("log", base=2)
    ax.set_xticks(batches); ax.set_xticklabels(batches)
    ax.set_xlabel("размер батча $B$ при равном числе проходов по данным")
    ax.set_ylabel("Recall@1 на тесте, %")
    ax.set_title("батч решает не всё: разброс всего "
                 f"{F['batch_spread']} п.п.", fontsize=12)
    ax2 = ax.twinx()
    exp_fn = [(b - 1) * F["false_neg_share"] / 100 for b in batches]
    ax2.plot(batches, exp_fn, "s--", color=RED, lw=1.8, ms=6, label="ложных negatives в строке")
    ax2.set_ylabel("ожидаемое число ложных negatives", color=RED)
    ax2.tick_params(axis="y", colors=RED)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", frameon=False, fontsize=9.5)
    a1.plot(pool_sizes, pool_r1, "o-", color=VIOLET, lw=2.4, ms=7, label="Recall@1 модели")
    a1.plot(pool_sizes, [100 / m for m in pool_sizes], "s--", color=RED, lw=1.6, ms=5,
            label="случайный выбор")
    a1.set_xscale("log", base=2)
    a1.set_xticks(pool_sizes); a1.set_xticklabels(pool_sizes)
    a1.set_xlabel("число кандидатов в пуле поиска")
    a1.set_ylabel("Recall@1, %")
    a1.set_title("та же модель, разный пул: метрика падает", fontsize=12)
    a1.legend(frameon=False, fontsize=9.5)
    for a in (ax, a1):
        a.grid(True, color=GRID, lw=0.4, alpha=0.5); a.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "batch_negatives.png")


def fig_hist():
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.hist(neg_sim, bins=80, density=True, color=LINE, label="несогласованные пары")
    ax.hist(pos_sim, bins=40, density=True, color=BLUE, alpha=0.85, label="размеченные пары")
    ax.axvline(F["pos_mean"], color=BLUE, lw=1.6)
    ax.axvline(F["neg_mean"], color=MUTED, lw=1.6)
    ax.set_xlabel("косинусная близость в общем пространстве")
    ax.set_ylabel("плотность")
    ax.set_title(f"Средние: пара {F['pos_mean']}, не-пара {F['neg_mean']} — но хвосты пересекаются")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "similarity_hist.png")


def fig_margin():
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    a0.plot(cov * 100, acc_cov * 100, color=VIOLET, lw=2.4)
    a0.axhline(F["r1"], color=MUTED, lw=1.4, ls=(0, (4, 3)))
    a0.annotate(f"сплошной ответ: {F['r1']}%", (60, F["r1"] + 1.5), fontsize=10, color=MUTED)
    a0.plot([20], [F["acc_at_cov20"]], "o", color=RED, ms=8)
    a0.annotate(f"20% самых уверенных: {F['acc_at_cov20']}%", (22, F["acc_at_cov20"]),
                fontsize=10, color=RED)
    a0.set_xlabel("покрытие: доля запросов, на которые отвечаем, %")
    a0.set_ylabel("точность top-1 на отвеченных, %")
    a0.set_title("отказ по малому margin поднимает точность", fontsize=12)
    ks = np.arange(1, 21)
    rk = [np.mean(base["ranks"] <= k) * 100 for k in ks]
    a1.plot(ks, rk, "o-", color=BLUE, lw=2.2, ms=5)
    a1.axhline(F["random_r1"], color=RED, lw=1.4, ls=(0, (4, 3)))
    a1.annotate(f"случайный выбор: {F['random_r1']}%", (5, F["random_r1"] + 2), fontsize=10, color=RED)
    a1.set_xlabel("$K$"); a1.set_ylabel("Recall@K, %")
    a1.set_title(f"поиск среди {F['n_test']} кандидатов", fontsize=12)
    for ax in (a0, a1):
        ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "retrieval_margin.png")


def fig_space():
    """84.6 — what retrieval actually returns: query and its three nearest captions."""
    rng3 = np.random.default_rng(4)
    ranks = base["ranks"]; S = base["S"]; y = data["y_te"]
    hits = np.where(ranks == 1)[0]; miss = np.where(ranks > 1)[0]
    rows = list(rng3.choice(hits, 2, replace=False)) + list(rng3.choice(miss, 2, replace=False))
    img = data["images_te"]
    fig, axes = plt.subplots(4, 4, figsize=(7.4, 7.6))
    for r, i in enumerate(rows):
        ax = axes[r][0]
        ax.imshow(img[i][::2, :], cmap="bone_r", aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(INK); sp.set_linewidth(1.6)
        ax.set_ylabel(f"запрос: цифра {y[i]}", fontsize=9, color=INK)
        top = np.argsort(-S[i])[:3]
        for k, j in enumerate(top):
            a = axes[r][k + 1]
            a.imshow(img[j][1::2, :], cmap="bone_r", aspect="auto")
            a.set_xticks([]); a.set_yticks([])
            col = GREEN if j == i else (GOLD if y[j] == y[i] else RED)
            for sp in a.spines.values():
                sp.set_color(col); sp.set_linewidth(2.4)
            a.set_title(f"{k+1}: cos {S[i, j]:.2f}", fontsize=9, color=col)
    for k, t in enumerate(["«снимок» (чётные строки)", "первый сосед", "второй", "третий"]):
        axes[0][k].set_xlabel("")
        axes[3][k].set_xlabel(t, fontsize=8.5, color=MUTED)
    fig.suptitle("Зелёная рамка — та самая подпись, жёлтая — чужая, но той же цифры,\n"
                 "красная — содержательно чужая", y=0.98, fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, OUT / "retrieval_examples.png")


def fig_shift():
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.0, 4.2),
                                 gridspec_kw={"width_ratios": [1, 1.1]})
    names = ["Recall@1\n(пара)", "top-1 класс", "zero-shot\nпо прототипам"]
    before = [F["r1"], F["class_top1"], F["proto_ens"]]
    after = [F["shift_r1"], F["shift_class_top1"], F["proto_shift"]]
    xs = np.arange(3)
    a0.bar(xs - 0.19, before, width=0.36, color=BLUE, label="исходный домен")
    a0.bar(xs + 0.19, after, width=0.36, color=RED, label="сдвиг на 1 пиксель")
    for x, b, a in zip(xs, before, after):
        a0.text(x - 0.19, b + 1.2, f"{b:.0f}", ha="center", fontsize=9.5, color=BLUE)
        a0.text(x + 0.19, a + 1.2, f"{a:.0f}", ha="center", fontsize=9.5, color=RED)
    a0.set_xticks(xs); a0.set_xticklabels(names, fontsize=10)
    a0.set_ylabel("%"); a0.set_ylim(0, 105)
    a0.set_title("те же цифры, чуть другой сенсор", fontsize=12)
    a0.legend(frameon=False, fontsize=10)
    a1.hist(np.linalg.norm(embed(Wv, data["Xl_te"]), axis=1), bins=40, color=BLUE, alpha=0.8,
            label="исходные снимки")
    img = np.roll(data["images_te"], 1, axis=2)
    mu_l, sd_l, _, _ = data["norm"]
    xl = (img[:, ::2, :].reshape(len(img), -1) - mu_l) / sd_l
    a1.hist(np.linalg.norm(embed(Wv, xl), axis=1), bins=40, color=RED, alpha=0.7, label="сдвинутые")
    a1.set_xlabel("длина вектора ДО нормировки")
    a1.set_ylabel("число объектов")
    a1.set_title(f"нормировка прячет сдвиг: длины упали в {F['norm_ratio']} раза", fontsize=12)
    a1.legend(frameon=False, fontsize=10)
    for ax in (a0, a1):
        ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "domain_shift.png")


# ---------------------------------------------------------------- margins
def side_halves():
    idx = [0, 1, 2]
    fig, axes = plt.subplots(1, 3, figsize=(4.2, 1.9))
    for ax, i in zip(axes, idx):
        ax.imshow(data["images_tr"][i], cmap="bone_r")
        for r in (0.5, 2.5, 4.5, 6.5):
            ax.axhline(r, color=RED, lw=1.2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(str(data["y_tr"][i]), fontsize=9, color=MUTED)
    fig.suptitle("чётные строки — «снимок», нечётные — «подпись»", y=1.10, fontsize=8.5)
    fig.tight_layout()
    save(fig, SIDE / "halves.png")


def side_fris():
    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    ax.set_aspect("equal"); ax.axis("off")
    z = (0.0, 0.0); a = (-1.5, 0.6); b = (1.9, 0.9)
    ax.plot(*z, "o", color=INK, ms=9)
    ax.annotate("объект $z$", (z[0], z[1] - 0.45), ha="center", fontsize=9, color=INK)
    for p, c, lbl, r in ((a, GREEN, "своя пара $a$", "$r_1$"), (b, RED, "соперник $b$", "$r_2$")):
        ax.plot(*p, "o", color=c, ms=9)
        ax.plot([z[0], p[0]], [z[1], p[1]], color=c, lw=1.6)
        ax.annotate(lbl, (p[0], p[1] + 0.35), ha="center", fontsize=9, color=c)
        ax.annotate(r, ((z[0] + p[0]) / 2, (z[1] + p[1]) / 2 - 0.3), fontsize=10, color=c)
    ax.set_xlim(-2.4, 2.8); ax.set_ylim(-1.1, 1.7)
    ax.set_title("$F=(r_2-r_1)/(r_2+r_1)$", fontsize=10)
    save(fig, SIDE / "fris.png")


def side_temp_bars():
    row = base["S"][0].copy()
    top = np.argsort(-row)[:6]
    fig, ax = plt.subplots(figsize=(4.0, 2.3))
    w = 0.38
    for k, (t, c) in enumerate(((0.05, RED), (0.5, BLUE))):
        z = row[top] / t
        p = np.exp(z - z.max()); p /= np.exp(row / t - (row / t).max()).sum()
        ax.bar(np.arange(6) + (k - 0.5) * w, p, width=w, color=c, label=f"$\\tau={t}$")
    ax.set_xticks(range(6)); ax.set_xticklabels(["1-й", "2-й", "3-й", "4-й", "5-й", "6-й"], fontsize=8)
    ax.set_ylabel("вероятность", fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("резкий и мягкий softmax", fontsize=9)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "softmax_temp.png")


def side_circle():
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.set_aspect("equal"); ax.axis("off")
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color=LINE, lw=1.2)
    rng4 = np.random.default_rng(2)
    for _ in range(7):
        ang = rng4.uniform(0.1, 1.5)
        r = rng4.uniform(0.35, 1.8)
        p = np.array([np.cos(ang), np.sin(ang)])
        ax.plot([0, r * p[0]], [0, r * p[1]], color=LINE, lw=0.8)
        ax.plot(r * p[0], r * p[1], "o", color=MUTED, ms=4)
        ax.plot(p[0], p[1], "o", color=BLUE, ms=6)
    ax.set_xlim(-0.2, 2.0); ax.set_ylim(-0.2, 2.0)
    ax.set_title("нормировка: остаётся только угол", fontsize=9)
    save(fig, SIDE / "circle.png")


fig_matrix()
fig_temperature()
fig_batch()
fig_hist()
fig_margin()
fig_space()
fig_shift()
side_halves()
side_fris()
side_temp_bars()
side_circle()

FACTS.parent.mkdir(parents=True, exist_ok=True)
FACTS.write_text(json.dumps(F, ensure_ascii=False, indent=1, default=str), encoding="utf8")
print("lesson 84 figures and facts written")
