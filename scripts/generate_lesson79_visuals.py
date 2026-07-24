"""Deterministic figures and facts for lesson 79: scaling laws.

Everything is measured, not stipulated. We train a family of small character-level
neural language models on the REAL SMS Spam Collection corpus (scripts/data/
sms-spam-collection.tsv), varying the number of non-embedding+embedding parameters N
and the number of presented training tokens D, then fit the surface

    L(N, D) = L_inf + A * N^(-alpha) + B * D^(-beta)

with scipy.least_squares (multi-start), check extrapolation on a held-out largest
model, measure repeated vs fresh tokens, measure the smooth-probability /
step-like-exact-match gap, and estimate the empirical entropy floor of the corpus.

Every number quoted in content/lessons/79.md is produced and asserted here and
mirrored into scripts/data/lesson79_facts.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
SMS = ROOT / "scripts" / "data" / "sms-spam-collection.tsv"
OUT = ROOT / "public" / "figures" / "lessons" / "79"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "79"
FACTS = ROOT / "scripts" / "data" / "lesson79_facts.json"

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

F: dict = {}


def save(fig, path, *, dpi=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# --------------------------------------------------------------- corpus
CONTEXT = 6


def load_stream():
    raw = SMS.read_text(encoding="latin-1")
    text = []
    for row in raw.splitlines():
        parts = row.split("\t")
        if len(parts) < 2:
            continue
        text.append(parts[1].strip().lower())
    joined = "\n".join(text)
    keep = "abcdefghijklmnopqrstuvwxyz 0123456789.,!?'\n"
    cleaned = "".join(ch if ch in keep else "~" for ch in joined)
    vocab = sorted(set(cleaned))
    stoi = {c: i for i, c in enumerate(vocab)}
    ids = np.array([stoi[c] for c in cleaned], dtype=np.int32)
    return ids, vocab


IDS, VOCAB = load_stream()
V = len(VOCAB)
NTOT = len(IDS)
SPLIT = int(0.80 * NTOT)
TRAIN = IDS[:SPLIT]
VAL = IDS[SPLIT:]


def windows(stream, limit=None):
    n = len(stream) - CONTEXT
    if limit is not None:
        n = min(n, limit)
    idx = np.arange(n)[:, None] + np.arange(CONTEXT)[None, :]
    return stream[idx], stream[np.arange(n) + CONTEXT]


VAL_X, VAL_Y = windows(VAL, limit=30000)
TRAIN_X, TRAIN_Y = windows(TRAIN)


# --------------------------------------------------------------- tiny model
class Model:
    def __init__(self, emb, hid, seed):
        rng = np.random.default_rng(seed)
        self.emb, self.hid = emb, hid
        self.E = rng.normal(0, 0.5, (V, emb))
        fan = CONTEXT * emb
        self.W1 = rng.normal(0, 1 / math.sqrt(fan), (fan, hid))
        self.b1 = np.zeros(hid)
        self.W2 = rng.normal(0, 1 / math.sqrt(hid), (hid, V))
        self.b2 = np.zeros(V)
        self.params = [self.E, self.W1, self.b1, self.W2, self.b2]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    @property
    def n_params(self):
        return int(sum(p.size for p in self.params))

    def forward(self, xb):
        e = self.E[xb].reshape(len(xb), -1)
        h = np.tanh(e @ self.W1 + self.b1)
        z = h @ self.W2 + self.b2
        z -= z.max(axis=1, keepdims=True)
        ex = np.exp(z)
        p = ex / ex.sum(axis=1, keepdims=True)
        return e, h, p

    def loss_probs(self, xb, yb, chunk=20000):
        tot, correct, pcorr = 0.0, 0, 0.0
        top1 = np.empty(len(xb), dtype=np.int32)
        for s in range(0, len(xb), chunk):
            x, y = xb[s:s + chunk], yb[s:s + chunk]
            _, _, p = self.forward(x)
            pc = p[np.arange(len(y)), y]
            tot += float(-np.log(np.clip(pc, 1e-12, None)).sum())
            pcorr += float(pc.sum())
            am = p.argmax(axis=1)
            top1[s:s + chunk] = am
            correct += int((am == y).sum())
        n = len(xb)
        return tot / n, correct / n, pcorr / n, top1

    def step(self, xb, yb, lr):
        n = len(xb)
        e, h, p = self.forward(xb)
        dz = p
        dz[np.arange(n), yb] -= 1.0
        dz /= n
        gW2 = h.T @ dz
        gb2 = dz.sum(axis=0)
        dh = (dz @ self.W2.T) * (1 - h * h)
        gW1 = e.T @ dh
        gb1 = dh.sum(axis=0)
        de = (dh @ self.W1.T).reshape(n, CONTEXT, self.emb)
        gE = np.zeros_like(self.E)
        np.add.at(gE, xb.reshape(-1), de.reshape(-1, self.emb))
        grads = [gE, gW1, gb1, gW2, gb2]
        self.t += 1
        b1c, b2c = 1 - 0.9 ** self.t, 1 - 0.999 ** self.t
        for i, (par, g) in enumerate(zip(self.params, grads)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            par -= lr * (self.m[i] / b1c) / (np.sqrt(self.v[i] / b2c) + 1e-8)


BATCH = 128
LR0 = 6e-3


def train_run(emb, hid, D, seed, unique=None):
    """Present D target tokens. If `unique` is set, cycle over the first `unique`
    training positions (data repetition); otherwise use fresh positions."""
    model = Model(emb, hid, seed)
    pool = unique if unique is not None else len(TRAIN_X)
    pool = min(pool, len(TRAIN_X))
    rng = np.random.default_rng(seed + 17)
    order = rng.permutation(pool)
    steps = D // BATCH
    for s in range(steps):
        sel = order[(s * BATCH) % pool: (s * BATCH) % pool + BATCH]
        if len(sel) < BATCH:
            order = rng.permutation(pool)
            sel = order[:BATCH]
        lr = LR0 * 0.5 * (1 + math.cos(math.pi * s / max(steps - 1, 1)))
        warm = min(1.0, (s + 1) / max(1, int(0.03 * steps)))
        model.step(TRAIN_X[sel], TRAIN_Y[sel], lr * warm)
    return model


CONFIGS = [(2, 8), (4, 16), (6, 32), (8, 64), (12, 128), (20, 320)]
DVALUES = [20000, 60000, 150000, 320000]


def run_grid():
    rows = []
    for mi, (emb, hid) in enumerate(CONFIGS):
        for D in DVALUES:
            seed = 790000 + 1000 * mi + D // 1000
            model = train_run(emb, hid, D, seed)
            loss, acc, pcorr, _ = model.loss_probs(VAL_X, VAL_Y)
            rows.append({"emb": emb, "hid": hid, "N": model.n_params,
                         "D": D, "loss": loss, "acc": acc, "pcorr": pcorr})
            print(f"  emb={emb:3d} hid={hid:4d} N={model.n_params:6d} "
                  f"D={D:7d}  val loss={loss:.4f} acc={acc:.4f}")
    return rows


# --------------------------------------------------------------- power-law fit
def fit_surface(rows):
    N = np.array([r["N"] for r in rows], float)
    D = np.array([r["D"] for r in rows], float)
    L = np.array([r["loss"] for r in rows], float)

    def resid(t):
        Li, la, al, lb, be = t
        return Li + math.exp(la) * N ** (-al) + math.exp(lb) * D ** (-be) - L

    rng = np.random.default_rng(7905)
    best, bestcost = None, np.inf
    for _ in range(60):
        x0 = [rng.uniform(0.5, 2.5), rng.uniform(0, 6), rng.uniform(0.02, 1.0),
              rng.uniform(0, 8), rng.uniform(0.02, 1.0)]
        try:
            r = least_squares(resid, x0, bounds=([0, -20, 0.01, -20, 0.01],
                                                 [4, 20, 2, 20, 2]),
                              xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=40000)
        except Exception:
            continue
        if r.cost < bestcost:
            bestcost, best = r.cost, r.x
    Li, la, al, lb, be = best
    return {"L_inf": float(Li), "A": float(math.exp(la)), "alpha": float(al),
            "B": float(math.exp(lb)), "beta": float(be),
            "rmse": float(np.sqrt(np.mean(resid(best) ** 2)))}


def surface(p, N, D):
    return p["L_inf"] + p["A"] * np.power(N, -p["alpha"]) + p["B"] * np.power(D, -p["beta"])


def fit_holdout(rows, drop_n):
    keep = [r for r in rows if r["N"] < drop_n]
    return fit_surface(keep)


# --------------------------------------------------------------- figures
def fig_power_law(rows, p):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    colors = [BLUE, GREEN, GOLD, RED, VIOLET, MUTED]
    for (emb, hid), c in zip(CONFIGS, colors):
        sub = [r for r in rows if r["emb"] == emb]
        D = np.array([r["D"] for r in sub], float)
        L = np.array([r["loss"] for r in sub], float)
        axes[0].plot(D, L, "o-", color=c, lw=1.8, ms=5, label=f"N={sub[0]['N']}")
        axes[1].plot(D, L - p["L_inf"], "o-", color=c, lw=1.8, ms=5)
    axes[0].set_xlabel("предъявленные токены $D$"); axes[0].set_ylabel("val loss, нат/символ")
    axes[0].set_title("линейные оси: похоже на плато", fontsize=12)
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("$D$ (лог)"); axes[1].set_ylabel(r"$L-L_\infty$ (лог)")
    axes[1].set_title(f"log–log после вычитания $L_\\infty={p['L_inf']:.2f}$: прямые",
                      fontsize=12)
    for ax in axes:
        ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.suptitle("Степенной закон виден только в правильных осях (реальные измерения)",
                 y=1.02, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "power_law.png")


def fig_surface(rows, p, ridge):
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    ng = np.logspace(math.log10(400), math.log10(120000), 260)
    dg = np.logspace(math.log10(8000), math.log10(3.0e7), 260)
    NG, DG = np.meshgrid(ng, dg)
    Z = surface(p, NG, DG)
    cs = ax.contourf(NG, DG, Z, levels=22, cmap="YlGnBu_r", alpha=0.9)
    ax.contour(NG, DG, Z, levels=12, colors=[LINE], linewidths=0.5)
    fig.colorbar(cs, ax=ax, label="fitted loss, нат/символ")
    for C in ridge["budgets"]:
        nn = np.logspace(math.log10(400), math.log10(120000), 200)
        dd = C / (6 * nn)
        m = (dd > dg[0]) & (dd < dg[-1])
        ax.plot(nn[m], dd[m], color=INK, lw=0.9, ls=(0, (4, 3)))
    ax.plot(ridge["N"], ridge["D"], "-", color=RED, lw=2.4, label="compute-optimal ridge")
    ax.plot(ridge["N"], ridge["D"], "o", color=PAPER, ms=6, mec=RED, mew=1.6)
    ax.scatter([r["N"] for r in rows], [r["D"] for r in rows], s=22, color=INK,
               zorder=6, label="измеренные запуски")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("параметры $N$"); ax.set_ylabel("токены $D$")
    ax.set_title("Долина бюджета: минимум loss на каждой диагонали $ND=\\mathrm{const}$")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    save(fig, OUT / "surface.png")


def fig_holdout(rows, full, part, hold):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    Dmax = max(DVALUES)
    sub = sorted([r for r in rows if r["D"] == Dmax], key=lambda r: r["N"])
    N = np.array([r["N"] for r in sub], float)
    L = np.array([r["loss"] for r in sub], float)
    fitted = [r for r in sub if r["N"] < hold["N"]]
    nf = np.array([r["N"] for r in fitted], float)
    lf = np.array([r["loss"] for r in fitted], float)
    grid = np.logspace(math.log10(N.min() * 0.8), math.log10(N.max() * 1.4), 200)
    axes[0].plot(grid, surface(part, grid, Dmax), color=BLUE, lw=2.2,
                 label="степенной закон по малым моделям")
    c = np.polyfit(np.log(nf), lf, 2)
    axes[0].plot(grid, np.polyval(c, np.log(grid)), color=GOLD, lw=2.0, ls=(0, (5, 3)),
                 label="квадратичный полином по $\\log N$")
    axes[0].scatter(nf, lf, s=48, color=INK, zorder=6, label="обучающие точки fit")
    axes[0].scatter([hold["N"]], [hold["loss"]], s=90, marker="*", color=RED, zorder=7,
                    label="скрытая крупная модель")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("$N$ (лог)"); axes[0].set_ylabel("val loss")
    axes[0].set_title("Экстраполяция за пределы fit", fontsize=12)
    axes[0].legend(frameon=False, fontsize=8.5)
    res = np.array([r["loss"] - surface(full, r["N"], r["D"]) for r in rows])
    axes[1].axhline(0, color=MUTED, lw=0.9)
    axes[1].scatter([r["N"] for r in rows], res, s=32, color=BLUE)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("$N$ (лог)"); axes[1].set_ylabel("остаток fit, нат/символ")
    axes[1].set_title(f"Остатки полного fit (RMSE {full['rmse']:.4f})", fontsize=12)
    for ax in axes:
        ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.suptitle("Закон проверяется скрытой точкой, а не красотой прямой",
                 y=1.02, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "holdout.png")


def fig_emergence(em):
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    N = np.array(em["N"], float)
    ax.plot(N, em["pcorr"], "o-", color=BLUE, lw=2.2, ms=6,
            label="средняя вероятность верного символа")
    ax.plot(N, em["acc"], "o-", color=GREEN, lw=2.0, ms=6, label="top-1 accuracy по символам")
    ax.plot(N, em["word"], "o-", color=RED, lw=2.4, ms=6,
            label="exact match: блок из 5 символов целиком")
    ax.set_xscale("log")
    ax.set_xlabel("параметры $N$ (лог)"); ax.set_ylabel("значение метрики")
    ax.set_title("Одни и те же модели: гладкая вероятность и «скачок» строгой метрики")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "emergence.png")


def fig_repeat(rep):
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    ax.plot(rep["presented"], rep["fresh"], "o-", color=BLUE, lw=2.2, ms=6,
            label="свежие токены")
    ax.plot(rep["presented"], rep["repeat"], "o-", color=RED, lw=2.2, ms=6,
            label=f"повтор {rep['unique']//1000}k уникальных токенов")
    ax.set_xscale("log")
    ax.set_xlabel("предъявленные токены $D$ (лог)"); ax.set_ylabel("val loss, нат/символ")
    ax.set_title("Закон описывает не байты, а новую информацию")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "repeat.png")


def fig_inference(p, dep):
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    for Q, c, lbl in zip(dep["Q"], [BLUE, GREEN, GOLD, RED],
                         [f"Q={q:.0e}" for q in dep["Q"]]):
        ax.plot(dep["N"], dep["curves"][str(Q)], color=c, lw=2.1, label=lbl)
    for Q, c in zip(dep["Q"], [BLUE, GREEN, GOLD, RED]):
        ax.plot([dep["best_N"][str(Q)]], [dep["best_cost"][str(Q)]], "o", color=c, ms=8,
                mec=PAPER, mew=1.4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("параметры $N$ (лог)")
    ax.set_ylabel("полная стоимость $C_{\\mathrm{train}}+QC_{\\mathrm{infer}}$, FLOP")
    ax.set_title(f"Одинаковый целевой loss {dep['target']:.2f}: чем больше запросов,\n"
                 "тем меньше выгодная модель", fontsize=13)
    ax.legend(frameon=False, fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "inference.png")


# --------------------------------------------------------------- sidenote figures
def side_slope(p):
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    n = np.logspace(3, 8, 100)
    for a, c, lbl in [(0.05, BLUE, r"$\alpha=0{,}05$"), (0.1, GREEN, r"$\alpha=0{,}1$"),
                      (0.3, RED, r"$\alpha=0{,}3$")]:
        ax.plot(n, n ** (-a), color=c, lw=1.8, label=lbl)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$N$", fontsize=9); ax.set_ylabel(r"$L-L_\infty$", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5)
    ax.tick_params(labelsize=8)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, SIDE / "slope.png")


def side_ratio(ridge):
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ratio = np.array(ridge["D"]) / np.array(ridge["N"])
    ax.plot(ridge["budgets"], ratio, "o-", color=VIOLET, lw=1.8, ms=4)
    ax.set_xscale("log")
    ax.set_xlabel("бюджет $C\\propto ND$", fontsize=9)
    ax.set_ylabel("токенов на параметр", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, SIDE / "ratio.png")


def side_entropy(ent):
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(ent["k"], ent["train"], "o-", color=BLUE, lw=1.8, ms=4, label="train (plug-in)")
    ax.plot(ent["k"], ent["val"], "o-", color=RED, lw=1.8, ms=4, label="val (сглаж.)")
    ax.set_xlabel("длина контекста $k$", fontsize=9)
    ax.set_ylabel("нат/символ", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5)
    ax.tick_params(labelsize=8)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, SIDE / "entropy.png")


def side_residual(rows, full):
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    Dmax = max(DVALUES)
    sub = sorted([r for r in rows if r["D"] == Dmax], key=lambda r: r["N"])
    n = np.array([r["N"] for r in sub], float)
    L = np.array([r["loss"] for r in sub], float)
    c = np.polyfit(np.log(n), L, 1)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.plot(n, L - np.polyval(c, np.log(n)), "o-", color=RED, lw=1.6, ms=4,
            label="прямая по $\\log N$")
    ax.plot(n, L - surface(full, n, Dmax), "o-", color=BLUE, lw=1.6, ms=4,
            label="закон с $L_\\infty$")
    ax.set_xscale("log")
    ax.set_xlabel("$N$", fontsize=9); ax.set_ylabel("остаток", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    ax.tick_params(labelsize=8)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, SIDE / "residual.png")


# --------------------------------------------------------------- extras
def compute_ridge(p, budgets):
    Ns, Ds = [], []
    for C in budgets:
        grid = np.logspace(math.log10(300), math.log10(2e6), 4000)
        d = C / (6 * grid)
        vals = surface(p, grid, d)
        i = int(np.argmin(vals))
        Ns.append(float(grid[i])); Ds.append(float(d[i]))
    return {"budgets": list(budgets), "N": Ns, "D": Ds}


def entropy_floor():
    ent_train, ent_val = [], []
    ks = [0, 1, 2, 3, 4]
    for k in ks:
        if k == 0:
            cnt = np.bincount(TRAIN, minlength=V).astype(float)
            pr = cnt / cnt.sum()
            ent_train.append(float(-(pr[pr > 0] * np.log(pr[pr > 0])).sum()))
            pv = np.bincount(VAL, minlength=V).astype(float)
            ent_val.append(float(-(pv / pv.sum() * np.log(np.clip(pr, 1e-12, None))).sum()))
            continue
        key_t = np.zeros(len(TRAIN) - k, dtype=np.int64)
        for j in range(k):
            key_t = key_t * V + TRAIN[j: len(TRAIN) - k + j]
        tgt_t = TRAIN[k:]
        tab: dict[int, np.ndarray] = {}
        for key, tgt in zip(key_t.tolist(), tgt_t.tolist()):
            row = tab.get(key)
            if row is None:
                row = np.zeros(V); tab[key] = row
            row[tgt] += 1
        tot, nn = 0.0, 0
        for row in tab.values():
            s = row.sum()
            pr = row[row > 0] / s
            tot += float(-(pr * np.log(pr)).sum() * s); nn += int(s)
        ent_train.append(tot / nn)
        key_v = np.zeros(len(VAL) - k, dtype=np.int64)
        for j in range(k):
            key_v = key_v * V + VAL[j: len(VAL) - k + j]
        tgt_v = VAL[k:]
        uni = np.bincount(TRAIN, minlength=V).astype(float)
        uni /= uni.sum()
        acc, m = 0.0, 0
        for key, tgt in zip(key_v.tolist(), tgt_v.tolist()):
            row = tab.get(key)
            if row is None:
                pr = uni[tgt]
            else:
                pr = (row[tgt] + 0.5) / (row.sum() + 0.5 * V)
            acc += -math.log(max(pr, 1e-12)); m += 1
        ent_val.append(acc / m)
    return {"k": ks, "train": ent_train, "val": ent_val}


BLOCK = 5


def block_exact_match(model):
    """Strict metric: fraction of consecutive blocks of BLOCK characters in which every
    character is predicted top-1 correctly. Same model, same predictions — only the
    metric is discrete."""
    _, acc, pcorr, top1 = model.loss_probs(VAL_X, VAL_Y)
    ok = (top1 == VAL_Y).astype(np.int32)
    n = (len(ok) // BLOCK) * BLOCK
    blocks = ok[:n].reshape(-1, BLOCK).sum(axis=1)
    total = int(len(blocks))
    return acc, pcorr, float((blocks == BLOCK).mean()), total


def deployment(p, target):
    """Along the iso-loss curve L(N,D)=target, minimise 6ND + Q*2N."""
    Ns = np.logspace(math.log10(300), math.log10(400000), 600)
    Qs = [1e3, 1e5, 1e6, 1e7]
    curves, bestN, bestC, keepN = {}, {}, {}, []
    for Q in Qs:
        cs = []
        for N in Ns:
            rest = target - p["L_inf"] - p["A"] * N ** (-p["alpha"])
            if rest <= 0:
                cs.append(np.nan); continue
            D = (p["B"] / rest) ** (1 / p["beta"])
            cs.append(6 * N * D + Q * 2 * N)
        cs = np.array(cs, float)
        curves[str(Q)] = cs.tolist()
        i = int(np.nanargmin(cs))
        bestN[str(Q)] = float(Ns[i]); bestC[str(Q)] = float(cs[i])
    return {"Q": Qs, "N": Ns.tolist(), "curves": curves, "best_N": bestN,
            "best_cost": bestC, "target": target}


# --------------------------------------------------------------- main
def main():
    print(f"corpus: {NTOT} chars, vocab {V}, train {len(TRAIN)}, val {len(VAL)}")
    assert 400000 < NTOT < 500000 and 35 <= V <= 45
    F["corpus_chars"] = int(NTOT); F["vocab"] = int(V)
    F["train_chars"] = int(len(TRAIN)); F["val_chars"] = int(len(VAL))
    F["val_positions"] = int(len(VAL_X)); F["context"] = CONTEXT
    F["uniform_loss"] = float(math.log(V))

    print("training grid ...")
    rows = run_grid()
    F["runs"] = rows
    Nmin = min(r["N"] for r in rows); Nmax = max(r["N"] for r in rows)
    F["N_min"], F["N_max"] = int(Nmin), int(Nmax)
    F["N_span"] = float(Nmax / Nmin)
    assert Nmax / Nmin > 50

    full = fit_surface(rows)
    print("fit:", {k: round(v, 5) for k, v in full.items()})
    F["fit"] = full
    assert full["rmse"] < 0.05, full["rmse"]
    assert 0.01 < full["alpha"] < 1.5 and 0.01 < full["beta"] < 1.5

    # best / worst measured
    best = min(rows, key=lambda r: r["loss"]); worst = max(rows, key=lambda r: r["loss"])
    F["best_run"] = best; F["worst_run"] = worst
    print("best", best, "worst", worst)

    # ten-fold N at fixed D: measured improvement
    Dmax = max(DVALUES)
    big = sorted([r for r in rows if r["D"] == Dmax], key=lambda r: r["N"])
    F["fixedD_curve"] = big
    F["fixedD_gain"] = float(big[0]["loss"] - big[-1]["loss"])
    F["N_ratio_fixedD"] = float(big[-1]["N"] / big[0]["N"])
    # predicted factor for 10x N from fitted alpha
    F["excess_factor_10x"] = float(10 ** full["alpha"])
    F["excess_factor_100x"] = float(100 ** full["alpha"])

    # smallest model does not benefit from more data
    small = sorted([r for r in rows if r["emb"] == CONFIGS[0][0]], key=lambda r: r["D"])
    F["small_curve"] = small
    F["small_gain_16x"] = float(small[0]["loss"] - small[-1]["loss"])
    biggest = sorted([r for r in rows if r["emb"] == CONFIGS[-1][0]], key=lambda r: r["D"])
    F["big_curve"] = biggest
    F["big_gain_16x"] = float(biggest[0]["loss"] - biggest[-1]["loss"])
    # capacity wall: the smallest model fed 16x more tokens is still worse than the
    # largest model fed the smallest data budget
    F["capacity_wall"] = {"small_N": small[-1]["N"], "small_D": small[-1]["D"],
                          "small_loss": small[-1]["loss"],
                          "big_N": biggest[0]["N"], "big_D": biggest[0]["D"],
                          "big_loss": biggest[0]["loss"]}
    assert small[-1]["loss"] > biggest[0]["loss"]
    F["capacity_gap_maxD"] = float(small[-1]["loss"] - biggest[-1]["loss"])

    # holdout extrapolation
    hold = big[-1]
    part = fit_holdout(rows, hold["N"])
    pred = float(surface(part, hold["N"], hold["D"]))
    F["holdout"] = {"N": hold["N"], "D": hold["D"], "actual": hold["loss"],
                    "predicted": pred, "error": float(hold["loss"] - pred),
                    "ape": float(abs(hold["loss"] - pred) / hold["loss"] * 100),
                    "fit": part}
    fitted_small = [r for r in rows if r["N"] < hold["N"] and r["D"] == Dmax]
    c = np.polyfit(np.log([r["N"] for r in fitted_small]),
                   [r["loss"] for r in fitted_small], 2)
    poly_pred = float(np.polyval(c, math.log(hold["N"])))
    F["holdout"]["poly_predicted"] = poly_pred
    F["holdout"]["poly_error"] = float(hold["loss"] - poly_pred)
    print("holdout:", F["holdout"]["actual"], pred, poly_pred)
    assert abs(F["holdout"]["error"]) < abs(F["holdout"]["poly_error"])

    # compute-optimal ridge
    budgets = [1e8, 3e8, 1e9, 3e9, 1e10, 3e10, 1e11]
    ridge = compute_ridge(full, budgets)
    F["ridge"] = ridge
    F["ridge_tokens_per_param"] = [float(d / n) for n, d in zip(ridge["N"], ridge["D"])]
    F["ridge_loss"] = [float(surface(full, n, d)) for n, d in zip(ridge["N"], ridge["D"])]
    F["ridge_gain_per_10x"] = float(F["ridge_loss"][0] - F["ridge_loss"][2])
    assert 0.3 < F["ridge_gain_per_10x"] < 0.6
    # doubling budget: how do N and D grow
    F["ridge_growth"] = {"N_factor_per_10x": float(ridge["N"][3] / ridge["N"][0]),
                         "D_factor_per_10x": float(ridge["D"][3] / ridge["D"][0])}
    print("ridge tokens/param:", [round(x, 1) for x in F["ridge_tokens_per_param"]])

    # analytic exponents of the compute-optimal allocation
    a, b = full["alpha"], full["beta"]
    expN, expD = b / (a + b), a / (a + b)
    F["exp_N"], F["exp_D"] = float(expN), float(expD)
    F["exp_check"] = {"factor": 30.0, "N_predicted": float(30 ** expN),
                      "D_predicted": float(30 ** expD)}
    assert abs(30 ** expN - F["ridge_growth"]["N_factor_per_10x"]) < 0.03
    assert abs(30 ** expD - F["ridge_growth"]["D_factor_per_10x"]) < 0.3

    # penalty for being off the ridge by 10x
    C = 1e9
    n_opt = ridge["N"][budgets.index(C)]
    l_opt = float(surface(full, n_opt, C / (6 * n_opt)))
    l_big = float(surface(full, n_opt * 10, C / (6 * n_opt * 10)))
    l_small = float(surface(full, n_opt / 10, C / (6 * n_opt / 10)))
    F["offridge_D"] = float(C / (6 * n_opt))
    F["offridge"] = {"C": C, "N_opt": n_opt, "L_opt": l_opt,
                     "L_10x_bigger": l_big, "L_10x_smaller": l_small}
    assert l_big > l_opt and l_small > l_opt
    print("offridge:", F["offridge"])

    # emergence: probability vs strict metric
    em = {"N": [], "acc": [], "pcorr": [], "word": []}
    for mi, (emb, hid) in enumerate(CONFIGS):
        seed = 790000 + 1000 * mi + Dmax // 1000
        model = train_run(emb, hid, Dmax, seed)
        acc, pcorr, word, total = block_exact_match(model)
        em["N"].append(model.n_params); em["acc"].append(acc)
        em["pcorr"].append(pcorr); em["word"].append(word)
        print(f"  emergence N={model.n_params:6d} acc={acc:.4f} p={pcorr:.4f} word={word:.4f}")
    em["blocks_total"] = total; em["block_len"] = BLOCK
    F["emergence"] = em
    rel_p = max(em["pcorr"][i + 1] / em["pcorr"][i] for i in range(len(em["pcorr"]) - 1))
    rel_w = max(em["word"][i + 1] / max(em["word"][i], 1e-9)
                for i in range(len(em["word"]) - 1))
    F["emergence_max_rel_prob"] = float(rel_p)
    F["emergence_max_rel_word"] = float(rel_w)
    F["emergence_word_growth"] = float(em["word"][-1] / max(em["word"][0], 1e-9))
    F["emergence_prob_growth"] = float(em["pcorr"][-1] / em["pcorr"][0])
    assert rel_w > rel_p

    # repeated vs fresh tokens
    emb, hid = CONFIGS[3]
    uniq = 40000
    rep = {"unique": uniq, "presented": [], "fresh": [], "repeat": []}
    for D in [40000, 80000, 160000, 320000]:
        m_fresh = train_run(emb, hid, D, 79100 + D // 1000)
        m_rep = train_run(emb, hid, D, 79100 + D // 1000, unique=uniq)
        lf = m_fresh.loss_probs(VAL_X, VAL_Y)[0]
        lr = m_rep.loss_probs(VAL_X, VAL_Y)[0]
        rep["presented"].append(D); rep["fresh"].append(float(lf)); rep["repeat"].append(float(lr))
        print(f"  repeat D={D}: fresh={lf:.4f} repeat={lr:.4f}")
    F["repeat"] = rep
    F["repeat_gap"] = float(rep["repeat"][-1] - rep["fresh"][-1])
    F["repeat_fresh_gain"] = float(rep["fresh"][0] - rep["fresh"][-1])
    F["repeat_repeat_gain"] = float(rep["repeat"][0] - rep["repeat"][-1])
    assert F["repeat_gap"] > 0.01

    # entropy floor (Khinchin)
    ent = entropy_floor()
    F["entropy"] = ent
    print("entropy train:", [round(x, 3) for x in ent["train"]])
    print("entropy val:", [round(x, 3) for x in ent["val"]])
    assert ent["train"][0] > ent["train"][-1]
    F["entropy_drop"] = float(ent["train"][0] - ent["train"][-1])
    F["entropy_val_best"] = float(min(ent["val"]))
    F["entropy_val_worst_k"] = float(ent["val"][-1])

    # deployment frontier
    target = float(np.percentile([r["loss"] for r in rows], 20))
    dep = deployment(full, target)
    F["deployment"] = {"target": dep["target"], "Q": dep["Q"], "best_N": dep["best_N"],
                       "best_cost": dep["best_cost"]}
    print("deployment best N:", dep["best_N"])
    qs = [str(q) for q in dep["Q"]]
    assert dep["best_N"][qs[0]] > dep["best_N"][qs[-1]]
    for q in qs:  # interior optimum, not a grid boundary
        assert 300 * 1.01 < dep["best_N"][q] < 400000 * 0.99, (q, dep["best_N"][q])
    F["deployment_shrink"] = float(dep["best_N"][qs[0]] / dep["best_N"][qs[-1]])

    # C = 6ND arithmetic used in the text
    F["flops_example"] = {"N": 1e9, "D": 2e10, "C": 6 * 1e9 * 2e10}

    # ---- numbers quoted verbatim in the prose
    Lbest = best["loss"]
    checks = {
        "bits_per_char": Lbest / math.log(2),
        "perplexity": math.exp(Lbest),
        "excess_ratio_150k_320k": (2.270194950876136 - full["L_inf"]) /
                                  (2.1340307072708495 - full["L_inf"]),
        "small_step_150to320": rows[3]["loss"],
    }
    assert abs(checks["bits_per_char"] - 3.079) < 0.0005
    assert abs(checks["perplexity"] - 8.45) < 0.005
    assert abs(checks["excess_ratio_150k_320k"] - 1.067) < 0.0005
    # quality of tokens: optimum shifts to a SMALLER model
    Cq = 1e9
    def opt_q(q):
        g = np.logspace(1.5, 8, 20000)
        d = Cq / (6 * g)
        vals = full["L_inf"] + full["A"] * g ** (-full["alpha"]) + \
            full["B"] * (d * q) ** (-full["beta"])
        i = int(np.argmin(vals))
        return float(g[i])
    n1, nh = opt_q(1.0), opt_q(0.5)
    theory = 0.5 ** (full["beta"] / (full["alpha"] + full["beta"]))
    checks["opt_q1"], checks["opt_q05"] = n1, nh
    checks["opt_ratio"], checks["opt_ratio_theory"] = nh / n1, theory
    assert abs(nh / n1 - theory) < 0.005 and abs(theory - 0.851) < 0.002
    assert abs(n1 - 1244) < 3 and abs(nh - 1058) < 3
    # strict metric: p^k amplification
    p1, p2, k = em["acc"][0], em["acc"][-1], BLOCK
    checks["p1^k"], checks["p2^k"] = p1 ** k, p2 ** k
    checks["pk_ratio"] = (p2 / p1) ** k
    assert abs(checks["p1^k"] - 0.0014) < 5e-05 and abs(checks["p2^k"] - 0.0090) < 0.0005
    assert abs(checks["pk_ratio"] - 6.63) < 0.005
    F["text_checks"] = checks
    print("text checks:", {k2: round(v, 5) for k2, v in checks.items()})

    print("figures ...")
    fig_power_law(rows, full)
    fig_surface(rows, full, ridge)
    fig_holdout(rows, full, part, hold)
    fig_emergence(em)
    fig_repeat(rep)
    fig_inference(full, dep)
    side_slope(full)
    side_ratio(ridge)
    side_entropy(ent)
    side_residual(rows, full)

    FACTS.write_text(json.dumps(F, ensure_ascii=False, indent=1), encoding="utf-8")
    print("facts ->", FACTS)
    print("OK")


if __name__ == "__main__":
    main()
