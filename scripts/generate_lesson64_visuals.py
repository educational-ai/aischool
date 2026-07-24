"""Deterministic figures for lesson 64: federated learning on the REAL SMS spam corpus.

The corpus is split among simulated clients (IID and label-skewed non-IID), a logistic
regression is trained by FedAvg and compared with centralised SGD, then we measure client
drift, the two objectives (per-record vs per-client weights), communication cost under
quantisation/sparsification and the price of user-level differential privacy.

Every number quoted in the lesson text is computed here and asserted.
Run:  python3 scripts/generate_lesson64_visuals.py
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SMS = ROOT / "scripts" / "data" / "sms-spam-collection.tsv"
OUT = ROOT / "public" / "figures" / "lessons" / "64"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "64"
FACTS = ROOT / "scripts" / "data" / "lesson64_facts.json"

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

FACT: dict[str, float] = {}


def save(fig, path, *, dpi=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ----------------------------------------------------------------- data
CLIP = 2.5       # порог обрезки нормы обновления в DP-эксперименте
D = 256          # hashing-trick dimension (+ bias column)
TOKEN = re.compile(r"[a-zа-я0-9£$]+")


def load_sms():
    labels, texts = [], []
    with open(SMS, encoding="utf-8", errors="replace") as f:
        for raw in f:
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue
            part = raw.split("\t", 1)
            if len(part) != 2:
                continue
            labels.append(1 if part[0].strip() == "spam" else 0)
            texts.append(part[1])
    y = np.array(labels, dtype=float)
    X = np.zeros((len(texts), D + 1), dtype=np.float32)
    X[:, D] = 1.0                                     # bias
    for i, t in enumerate(texts):
        for tok in set(TOKEN.findall(t.lower())):
            X[i, (hash(tok) if False else _h(tok)) % D] = 1.0
    return X, y


def _h(s: str) -> int:
    """Stable (non-salted) string hash so runs are reproducible."""
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def logistic_grad(w, Xb, yb):
    p = sigmoid(Xb @ w)
    return Xb.T @ (p - yb) / len(yb)


def metrics(w, X, y):
    p = sigmoid(X @ w)
    pred = (p >= 0.5).astype(float)
    acc = float((pred == y).mean())
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return acc, f1


# ---------------------------------------------------------- partitioning
def dirichlet_partition(idx, y, K, alpha, rng):
    """Label-skewed split: for each class the shares across clients are Dirichlet."""
    parts = [[] for _ in range(K)]
    for cls in (0.0, 1.0):
        ids = idx[y[idx] == cls]
        rng.shuffle(ids)
        prop = rng.dirichlet(np.repeat(alpha, K))
        cuts = (np.cumsum(prop) * len(ids)).astype(int)[:-1]
        for k, chunk in enumerate(np.split(ids, cuts)):
            parts[k].extend(chunk.tolist())
    out = []
    for p in parts:
        p = np.array(p, dtype=int)
        rng.shuffle(p)
        out.append(p)
    return out


def iid_partition(idx, K, rng):
    ids = idx.copy()
    rng.shuffle(ids)
    return [np.array(c, dtype=int) for c in np.array_split(ids, K)]


# -------------------------------------------------------------- FedAvg
def local_train(w0, X, y, epochs, lr, batch, rng):
    w = w0.copy()
    n = len(y)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            b = order[s:s + batch]
            if len(b) == 0:
                continue
            w -= lr * logistic_grad(w, X[b], y[b])
    return w


def fedavg(parts, X, y, Xte, yte, *, rounds, epochs, lr=0.6, batch=16,
           frac=0.5, seed=0, weight="records", clip=None, sigma=0.0,
           quant=None, topk=None, track_drift=False):
    rng = np.random.default_rng(seed)
    w = np.zeros(D + 1)
    hist, drift = [], []
    K = len(parts)
    m = max(1, int(round(frac * K)))
    for t in range(rounds):
        sel = rng.choice(K, size=m, replace=False)
        sel = [k for k in sel if len(parts[k]) > 0]
        deltas, sizes = [], []
        for k in sel:
            idx = parts[k]
            wk = local_train(w, X[idx], y[idx], epochs, lr, batch, rng)
            d = wk - w
            if clip is not None:
                nrm = np.linalg.norm(d)
                d = d * min(1.0, clip / (nrm + 1e-12))
            if quant is not None:                       # uniform b-bit quantisation
                lo, hi = d.min(), d.max()
                levels = 2 ** quant - 1
                step = (hi - lo) / levels if hi > lo else 1.0
                d = lo + np.round((d - lo) / step) * step
            if topk is not None:                        # keep the largest coordinates
                keep = max(1, int(round(topk * len(d))))
                thr = np.partition(np.abs(d), -keep)[-keep]
                d = np.where(np.abs(d) >= thr, d, 0.0)
            deltas.append(d)
            sizes.append(len(idx))
        Dm = np.array(deltas)
        if weight == "records":
            wgt = np.array(sizes, dtype=float)
        else:
            wgt = np.ones(len(sizes))
        wgt = wgt / wgt.sum()
        agg = wgt @ Dm
        if sigma > 0 and clip is not None:
            agg = agg + rng.normal(0, sigma * clip / len(sizes), size=agg.shape)
        if track_drift:
            centre = Dm.mean(axis=0)
            drift.append(float(np.mean([np.linalg.norm(d - centre) for d in Dm])))
        w = w + agg
        hist.append(metrics(w, Xte, yte)[1])
    return w, np.array(hist), np.array(drift)


def central(X, y, Xte, yte, *, epochs, lr=0.6, batch=16, seed=0):
    rng = np.random.default_rng(seed)
    w = local_train(np.zeros(D + 1), X, y, epochs, lr, batch, rng)
    return w, metrics(w, Xte, yte)


# =====================================================================
X, y = load_sms()
N = len(y)
SPAM = float(y.sum())
FACT["messages"] = N
FACT["spam"] = SPAM
FACT["spam_share"] = SPAM / N
assert N == 5574, N
assert SPAM == 747, SPAM

rng0 = np.random.default_rng(64)
perm = rng0.permutation(N)
ntr = int(0.8 * N)
tr, te = perm[:ntr], perm[ntr:]
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
FACT["train"] = len(tr)
FACT["test"] = len(te)

K = 20
ROUNDS = 40

parts_iid = iid_partition(np.arange(len(tr)), K, np.random.default_rng(1))
parts_nid = dirichlet_partition(np.arange(len(tr)), ytr, K, 0.5, np.random.default_rng(49))
assert min(len(p) for p in parts_nid) >= 30, [len(p) for p in parts_nid]
sizes_nid = np.array([len(p) for p in parts_nid])
spam_share = np.array([float(ytr[p].mean()) if len(p) else 0.0 for p in parts_nid])
FACT["client_min"] = int(sizes_nid.min())
FACT["client_max"] = int(sizes_nid.max())
FACT["size_ratio"] = float(sizes_nid.max() / max(1, sizes_nid.min()))
FACT["spam_share_min"] = float(spam_share.min())
FACT["spam_share_max"] = float(spam_share.max())
FACT["clients_no_spam"] = int((spam_share == 0).sum())

# --- central baseline with a comparable budget (40 rounds x 1 epoch over half the data)
w_c, (acc_c, f1_c) = central(Xtr, ytr, Xte, yte, epochs=20, seed=7)
FACT["central_acc"] = acc_c
FACT["central_f1"] = f1_c
print(f"central: acc={acc_c:.4f} f1={f1_c:.4f}")

LONG = 100
TARGET = 0.80
runs = {}
for tag, parts, E in [("iid_e1", parts_iid, 1), ("iid_e5", parts_iid, 5),
                      ("iid_e20", parts_iid, 20), ("nid_e1", parts_nid, 1),
                      ("nid_e5", parts_nid, 5), ("nid_e20", parts_nid, 20)]:
    _, h, dr = fedavg(parts, Xtr, ytr, Xte, yte, rounds=LONG, epochs=E,
                      seed=11, track_drift=True)
    runs[tag] = (h, dr)
    FACT[f"f1_{tag}"] = float(h[-1])
    FACT[f"drift_{tag}"] = float(dr.mean())
    hit = next((i + 1 for i, v in enumerate(h) if v >= TARGET), None)
    FACT[f"rounds80_{tag}"] = hit
    print(f"{tag}: final F1={h[-1]:.4f}  rounds to {TARGET}={hit}  drift={dr.mean():.4f}")

FACT["rounds_price_of_skew_e1"] = FACT["rounds80_nid_e1"] / FACT["rounds80_iid_e1"]
FACT["drift_ratio_skew_e1"] = FACT["drift_nid_e1"] / FACT["drift_iid_e1"]
FACT["drift_ratio_e20_e1"] = FACT["drift_nid_e20"] / FACT["drift_nid_e1"]
FACT["gap_iid_nid_final_e1"] = FACT["f1_iid_e1"] - FACT["f1_nid_e1"]


# ---------------------------------------- fig 64.1: FedAvg vs central, IID vs non-IID
def fig_curves():
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8),
                             gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    r = np.arange(1, LONG + 1)
    ax.axhline(f1_c, color=MUTED, lw=1.4, ls=(0, (5, 3)),
               label=f"центральное обучение, F1 = {f1_c:.2f}")
    ax.axhline(TARGET, color=LINE, lw=1.0)
    for tag, c, lab in [("iid_e1", BLUE, "IID, $E=1$"),
                        ("iid_e5", GREEN, "IID, $E=5$"),
                        ("nid_e1", GOLD, "non-IID, $E=1$"),
                        ("nid_e5", RED, "non-IID, $E=5$")]:
        ax.plot(r, runs[tag][0], color=c, lw=2.0, label=f"{lab}: итог {runs[tag][0][-1]:.2f}")
    ax.set_xlim(1, 60)
    ax.set_xlabel("раунд коммуникации"); ax.set_ylabel("F1 на общем тесте (класс «спам»)")
    ax.set_title("Одни письма, разное расселение", fontsize=13)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)

    ax = axes[1]
    tags = ["iid_e1", "iid_e5", "iid_e20", "nid_e1", "nid_e5", "nid_e20"]
    vals = [FACT[f"rounds80_{t}"] for t in tags]
    cols = [BLUE, BLUE, BLUE, RED, RED, RED]
    ax.bar(np.arange(6), vals, color=cols, alpha=0.85)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.4, str(v), ha="center", fontsize=10, color=cols[i])
    ax.set_xticks(np.arange(6))
    ax.set_xticklabels(["1", "5", "20", "1", "5", "20"], fontsize=10)
    ax.set_xlabel("локальных эпох $E$:   слева IID, справа non-IID", fontsize=10)
    ax.set_ylabel(f"раундов до F1 = {TARGET}")
    ax.set_title("Цена неоднородности — в раундах", fontsize=13)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "fedavg_curves.png")


# ---------------------------------------- fig 64.2: client drift (toy quadratics)
def fig_drift():
    # модельный пример: две квадратичные локальные потери
    a1, a2, m1, m2 = 1.0, 2.0, np.array([0.0, 0.0]), np.array([4.0, 1.5])
    eta = 0.1
    w0 = np.array([1.0, 3.0])

    def step(w, a, m):
        return w - eta * 2 * a * (w - m)

    tr1, tr2 = [w0.copy()], [w0.copy()]
    for _ in range(5):
        tr1.append(step(tr1[-1], a1, m1))
        tr2.append(step(tr2[-1], a2, m2))
    tr1, tr2 = np.array(tr1), np.array(tr2)
    avg = (tr1 + tr2) / 2

    def gstep(w):
        return w - eta * (2 * a1 * (w - m1) + 2 * a2 * (w - m2)) / 2

    cen = [w0.copy()]
    for _ in range(5):
        cen.append(gstep(cen[-1]))
    cen = np.array(cen)
    d1 = float(np.linalg.norm(avg[1] - cen[1]))
    d5 = float(np.linalg.norm(avg[5] - cen[5]))
    FACT["toy_drift_1"] = d1
    FACT["toy_drift_5"] = d5
    assert d1 < 1e-12, d1
    assert d5 > 0.1, d5

    gx, gy = np.meshgrid(np.linspace(-1.5, 5.5, 300), np.linspace(-1.5, 4.0, 300))
    F1 = a1 * ((gx - m1[0]) ** 2 + (gy - m1[1]) ** 2)
    F2 = a2 * ((gx - m2[0]) ** 2 + (gy - m2[1]) ** 2)
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.contour(gx, gy, F1, levels=8, colors=BLUE, linewidths=0.6, alpha=0.55)
    ax.contour(gx, gy, F2, levels=8, colors=RED, linewidths=0.6, alpha=0.55)
    ax.plot(tr1[:, 0], tr1[:, 1], "o-", color=BLUE, ms=4.5, lw=1.8, label="клиент 1 (пять локальных шагов)")
    ax.plot(tr2[:, 0], tr2[:, 1], "o-", color=RED, ms=4.5, lw=1.8, label="клиент 2 (пять локальных шагов)")
    ax.plot(avg[:, 0], avg[:, 1], "s--", color=VIOLET, ms=5, lw=1.8, label="среднее локальных весов")
    ax.plot(cen[:, 0], cen[:, 1], "^-", color=GREEN, ms=6, lw=2.2, label="центральные шаги по $F$")
    ax.annotate("", xy=avg[5], xytext=cen[5],
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))
    ax.text((avg[5, 0] + cen[5, 0]) / 2 + 0.1, (avg[5, 1] + cen[5, 1]) / 2 + 0.15,
            f"client drift = {d5:.2f}", fontsize=10.5, color=INK)
    ax.scatter(*m1, marker="*", s=170, color=BLUE, zorder=6)
    ax.scatter(*m2, marker="*", s=170, color=RED, zorder=6)
    ax.set_xlabel("$w_1$"); ax.set_ylabel("$w_2$")
    ax.set_title("После одного шага среднее совпадает с центральным, после пяти — нет")
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "client_drift.png")


# ---------------------------------------- fig 64.3: two objectives, worst client
def per_client_f1(w, parts, Xs, ys):
    out = []
    for p in parts:
        if len(p) < 20:
            out.append(np.nan); continue
        out.append(metrics(w, Xs[p], ys[p])[0])
    return np.array(out)


def fig_objectives():
    parts_te = dirichlet_partition(np.arange(len(te)), yte, K, 0.5, np.random.default_rng(49))
    w_rec, h_rec, _ = fedavg(parts_nid, Xtr, ytr, Xte, yte, rounds=ROUNDS, epochs=5,
                             seed=11, weight="records")
    w_usr, h_usr, _ = fedavg(parts_nid, Xtr, ytr, Xte, yte, rounds=ROUNDS, epochs=5,
                             seed=11, weight="clients")
    acc_rec = per_client_f1(w_rec, parts_te, Xte, yte)
    acc_usr = per_client_f1(w_usr, parts_te, Xte, yte)

    # --- персонализация: одна локальная эпоха дообучения поверх глобальной модели
    rngp = np.random.default_rng(3)
    acc_per = []
    for ktr, kte in zip(parts_nid, parts_te):
        if len(kte) < 20:
            acc_per.append(np.nan); continue
        wp = local_train(w_rec, Xtr[ktr], ytr[ktr], 1, 0.1, 16, rngp)
        acc_per.append(metrics(wp, Xte[kte], yte[kte])[0])
    acc_per = np.array(acc_per)
    FACT["mean_client_personal"] = float(np.nanmean(acc_per))
    FACT["worst_personal"] = float(np.nanmin(acc_per))
    FACT["personal_gain_worst"] = FACT["worst_personal"] - float(np.nanmin(acc_rec))
    FACT["personal_better_share"] = float(np.nanmean(acc_per >= acc_rec))
    print("personal:", FACT["mean_client_personal"], FACT["worst_personal"])
    FACT["f1_records"] = float(h_rec[-1])
    FACT["f1_clients"] = float(h_usr[-1])
    FACT["worst_records"] = float(np.nanmin(acc_rec))
    FACT["worst_clients"] = float(np.nanmin(acc_usr))
    FACT["mean_client_records"] = float(np.nanmean(acc_rec))
    FACT["mean_client_clients"] = float(np.nanmean(acc_usr))
    print("objectives:", FACT["f1_records"], FACT["f1_clients"],
          FACT["worst_records"], FACT["worst_clients"])

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    order = np.argsort(sizes_nid)
    xs = np.arange(K)
    ok = ~np.isnan(acc_rec[order])
    ax.plot(xs[ok], acc_rec[order][ok], "o", color=BLUE, ms=7,
            label="веса по строкам")
    ax.plot(xs[ok], acc_usr[order][ok], "s", color=RED, ms=6,
            label="веса по клиентам")
    ax.plot(xs[ok], acc_per[order][ok], "^", color=GREEN, ms=6,
            label="+ локальное дообучение")
    ax.set_ylim(0.88, 1.02)
    ax.set_xlabel("клиенты, отсортированные по размеру"); ax.set_ylabel("точность на локальном тесте")
    ax.set_title("Точность у каждого клиента (где тест ≥ 20 строк)", fontsize=12)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)

    ax = axes[1]
    labels = ["глобальная F1", "средний клиент", "худший клиент"]
    v1 = [FACT["f1_records"], FACT["mean_client_records"], FACT["worst_records"]]
    v2 = [FACT["f1_clients"], FACT["mean_client_clients"], FACT["worst_clients"]]
    xp = np.arange(3)
    ax.bar(xp - 0.19, v1, width=0.36, color=BLUE, label="по строкам")
    ax.bar(xp + 0.19, v2, width=0.36, color=RED, label="по клиентам")
    for i in range(3):
        ax.text(xp[i] - 0.19, v1[i] + 0.012, f"{v1[i]:.2f}", ha="center", fontsize=9.5, color=BLUE)
        ax.text(xp[i] + 0.19, v2[i] + 0.012, f"{v2[i]:.2f}", ha="center", fontsize=9.5, color=RED)
    ax.set_xticks(xp); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.28)
    ax.set_title("Что именно улучшилось", fontsize=13)
    ax.legend(frameon=False, fontsize=10, loc="upper center", ncol=2)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.suptitle("Среднее по строкам, среднее по клиентам и худший клиент — три разных числа", y=1.02, fontsize=14)
    fig.tight_layout()
    save(fig, OUT / "objectives.png")


# ---------------------------------------- fig 64.4: communication budget
def fig_comm():
    P = 10_000_000
    per_client = P * 4
    FACT["bytes_client_mb"] = per_client / 1e6
    FACT["bytes_round_gb"] = per_client * 500 / 1e9
    FACT["bytes_q8_gb"] = FACT["bytes_round_gb"] / 4
    FACT["bytes_sp_gb"] = FACT["bytes_round_gb"] * 0.01 * 1.5
    assert abs(FACT["bytes_round_gb"] - 20.0) < 1e-9

    variants = [("float32", None, None, BLUE),
                ("8 бит", 8, None, GOLD),
                ("4 бита", 4, None, VIOLET),
                ("top-10 %", None, 0.10, RED),
                ("top-1 %", None, 0.01, GREEN)]
    res = []
    for name, q, tk, c in variants:
        _, h, _ = fedavg(parts_nid, Xtr, ytr, Xte, yte, rounds=ROUNDS, epochs=5,
                         seed=11, quant=q, topk=tk)
        bits = 32 if q is None else q
        share = 1.0 if tk is None else tk
        cost = (D + 1) * bits * share / 8 / 1024      # kB на клиента за раунд
        res.append((name, cost, float(h[-1]), c))
        FACT[f"comm_f1_{name}"] = float(h[-1])
        FACT[f"comm_kb_{name}"] = cost
        print(f"comm {name}: {cost:.2f} kB, F1={h[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    offs = [(14, -6), (8, 12), (-104, 8), (-104, -26), (12, -6)]
    for i, (name, cost, f1, c) in enumerate(res):
        ax.scatter(cost, f1, s=150, color=c, zorder=5)
        ax.annotate(f"{name}\n{cost:.3f} кБ, F1 {f1:.2f}", (cost, f1),
                    textcoords="offset points", xytext=offs[i], fontsize=10, color=c)
    ax.set_xscale("log")
    ax.set_xlabel("байтов на клиента за раунд (лог. шкала, кБ)")
    ax.set_ylabel("F1 после 40 раундов")
    ax.set_xlim(0.005, 6)
    lo = min(r[2] for r in res); hi = max(r[2] for r in res)
    ax.set_ylim(lo - 0.12, hi + 0.06)
    ax.set_title("Сжатие обновлений: сколько качества стоит один килобайт")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "communication.png")


# ---------------------------------------- fig 64.5: privacy price
def fig_privacy():
    _, h_base, _ = fedavg(parts_nid, Xtr, ytr, Xte, yte, rounds=ROUNDS, epochs=5, seed=11)
    curves = [("без DP", h_base, MUTED)]
    for sig, c in [(0.2, GREEN), (0.5, GOLD), (1.0, RED)]:
        _, h, _ = fedavg(parts_nid, Xtr, ytr, Xte, yte, rounds=ROUNDS, epochs=5,
                         seed=11, clip=CLIP, sigma=sig)
        curves.append((f"$C={CLIP}$, $\\sigma={sig}$", h, c))
        FACT[f"f1_dp_{sig}"] = float(h[-1])
        print(f"dp sigma={sig}: F1={h[-1]:.4f}")
    _, h_clip, _ = fedavg(parts_nid, Xtr, ytr, Xte, yte, rounds=ROUNDS, epochs=5,
                          seed=11, clip=CLIP, sigma=0.0)
    FACT["f1_clip_only"] = float(h_clip[-1])

    # реальные нормы обновлений первого раунда
    rng = np.random.default_rng(11)
    w0 = np.zeros(D + 1)
    norms = []
    for p in parts_nid:
        if len(p) == 0:
            continue
        wk = local_train(w0, Xtr[p], ytr[p], 5, 0.6, 16, rng)
        norms.append(float(np.linalg.norm(wk - w0)))
    norms = np.array(norms)
    FACT["norm_min"] = float(norms.min())
    FACT["norm_max"] = float(norms.max())
    FACT["norm_median"] = float(np.median(norms))
    FACT["clipped_at_C"] = float((norms > CLIP).mean())
    print("norms:", norms.min(), np.median(norms), norms.max(), FACT["clipped_at_C"])

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    r = np.arange(1, ROUNDS + 1)
    for lab, h, c in curves:
        axes[0].plot(r, h, color=c, lw=2.0, label=f"{lab}: итог {h[-1]:.2f}")
    axes[0].set_xlabel("раунд"); axes[0].set_ylabel("F1 на тесте")
    axes[0].set_title("Шум приватности стоит качества", fontsize=13)
    axes[0].legend(frameon=False, fontsize=9.5, loc="lower right")
    axes[0].grid(True, color=GRID, lw=0.4, alpha=0.5); axes[0].set_axisbelow(True)

    axes[1].hist(norms, bins=12, color=BLUE, alpha=0.75, edgecolor=PAPER)
    axes[1].axvline(CLIP, color=RED, lw=2.0, ls=(0, (5, 3)))
    axes[1].text(CLIP * 1.04, axes[1].get_ylim()[1] * 0.85, f"порог $C={CLIP}$\nобрезано {FACT['clipped_at_C']*100:.0f} %",
                 color=RED, fontsize=10)
    axes[1].set_xlabel(r"норма обновления $\|\Delta_k\|$"); axes[1].set_ylabel("число клиентов")
    axes[1].set_title("Кого именно обрезает порог", fontsize=13)
    axes[1].grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); axes[1].set_axisbelow(True)
    fig.suptitle("Обрезка и шум: формальная гарантия покупается точностью", y=1.02, fontsize=14.5)
    fig.tight_layout()
    save(fig, OUT / "privacy.png")


# ---------------------------------------- sidenote: client sizes and skew
def side_clients():
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    order = np.argsort(-sizes_nid)
    ax.bar(np.arange(K), sizes_nid[order], color=BLUE, alpha=0.85)
    ax2 = ax.twinx()
    ax2.plot(np.arange(K), spam_share[order] * 100, "o-", color=RED, ms=3.5, lw=1.4)
    ax2.set_ylabel("доля спама, %", color=RED, fontsize=9)
    ax2.tick_params(labelsize=8, colors=RED)
    ax.set_xlabel("клиенты", fontsize=9); ax.set_ylabel("строк", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title("Размер и состав клиента", fontsize=10.5)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, SIDE / "clients.png", dpi=170)


# ---------------------------------------- sidenote: drift vs local steps
def side_drift():
    Es = [1, 2, 5, 10, 20]
    vals = []
    for E in Es:
        key = {1: "nid_e1", 5: "nid_e5", 20: "nid_e20"}.get(E)
        if key:
            vals.append(FACT[f"drift_{key}"])
        else:
            _, _, dr = fedavg(parts_nid, Xtr, ytr, Xte, yte, rounds=20, epochs=E,
                              seed=11, track_drift=True)
            vals.append(float(dr.mean()))
    FACT["drift_curve"] = [round(v, 4) for v in vals]
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    ax.plot(Es, vals, "o-", color=VIOLET, lw=2.0, ms=5)
    ax.set_xlabel("локальных эпох $E$", fontsize=9)
    ax.set_ylabel("разброс обновлений", fontsize=9)
    ax.set_title("Чем дольше локально — тем дальше врозь", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, SIDE / "drift.png", dpi=170)


# ---------------------------------------- sidenote: secure aggregation masks
def side_masks():
    rng = np.random.default_rng(64)
    upd = np.array([3.0, -1.0, 2.0])
    masks = np.array([7.0, -4.0])
    masks = np.append(masks, -masks.sum())
    masked = upd + masks
    assert abs(masked.sum() - upd.sum()) < 1e-9
    FACT["mask_sum"] = float(upd.sum())
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    xp = np.arange(3)
    ax.bar(xp - 0.19, upd, width=0.36, color=BLUE, label="истинное $\\Delta_k$")
    ax.bar(xp + 0.19, masked, width=0.36, color=GOLD, label="то, что видит сервер")
    ax.axhline(0, color=LINE, lw=1)
    ax.set_xticks(xp); ax.set_xticklabels(["A", "B", "C"], fontsize=9)
    ax.set_ylabel("значение", fontsize=9); ax.tick_params(labelsize=8)
    ax.set_title(f"Маски гасятся: сумма = {upd.sum():.0f}", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, SIDE / "masks.png", dpi=170)


# ---------------------------------------- toy weighting example
def toy_weights():
    nA, nB, wA, wB = 1000, 10, 0.0, 10.0
    rec = (nA * wA + nB * wB) / (nA + nB)
    usr = (wA + wB) / 2
    FACT["toy_records"] = rec
    FACT["toy_clients"] = usr
    assert abs(rec - 0.0990099) < 1e-6
    assert usr == 5.0
    # homework check: n=10,30,60; w=1,2,4
    ns = np.array([10, 30, 60]); ws = np.array([1, 2, 4])
    FACT["hw_records"] = float((ns * ws).sum() / ns.sum())
    FACT["hw_clients"] = float(ws.mean())
    assert abs(FACT["hw_records"] - 3.1) < 1e-9
    # clipping example
    FACT["clip_factor"] = 3 / 12
    norms = np.array([1, 2, 3, 6, 12])
    FACT["clip_share_example"] = float((norms > 3).mean())


if __name__ == "__main__":
    toy_weights()
    fig_curves()
    fig_drift()
    fig_objectives()
    fig_comm()
    fig_privacy()
    side_clients()
    side_drift()
    side_masks()

    # ---- редакционные проверки: всё, что цитируется в тексте
    assert FACT["central_f1"] > 0.85, FACT["central_f1"]
    assert FACT["rounds80_nid_e1"] > FACT["rounds80_iid_e1"], FACT
    assert FACT["drift_nid_e1"] > FACT["drift_iid_e1"], FACT
    assert FACT["drift_nid_e20"] > FACT["drift_nid_e1"], FACT
    assert FACT["worst_clients"] >= FACT["worst_records"] - 1e-9
    assert FACT["f1_dp_1.0"] < FACT["f1_clip_only"], FACT
    assert 0.15 < FACT["clipped_at_C"] < 0.85, FACT["clipped_at_C"]

    # ---- точные пины: каждое число, названное в прозе урока 64
    def pin(key, value, nd):
        got = round(float(FACT[key]), nd)
        assert got == value, f"{key}: в тексте {value}, посчитано {got} ({FACT[key]})"

    # корпус и централизованный потолок
    pin("messages", 5574, 0); pin("spam", 747, 0)
    pin("spam_share", 0.134, 3); pin("train", 4459, 0); pin("test", 1115, 0)
    pin("central_acc", 0.970, 3); pin("central_f1", 0.88, 2)
    pin("client_min", 40, 0); pin("client_max", 1002, 0)
    pin("size_ratio", 25.05, 2); pin("clients_no_spam", 2, 0)
    pin("spam_share_min", 0.0, 3); pin("spam_share_max", 1.0, 3)
    # раунды до F1 = 0,80
    pin("rounds80_iid_e1", 6, 0); pin("rounds80_nid_e1", 19, 0)
    pin("rounds80_iid_e5", 2, 0); pin("rounds80_nid_e5", 8, 0)
    pin("rounds80_nid_e20", 12, 0); pin("rounds_price_of_skew_e1", 3.2, 1)
    pin("f1_nid_e1", 0.88, 2)
    # client drift
    pin("toy_drift_1", 0.0, 6); pin("toy_drift_5", 0.34, 2)
    pin("toy_records", 0.099, 3); pin("toy_clients", 5.0, 3)
    pin("drift_nid_e1", 0.65, 2); pin("drift_iid_e1", 0.33, 2)
    pin("drift_nid_e5", 1.32, 2); pin("drift_nid_e20", 2.19, 2)
    pin("drift_ratio_e20_e1", 3.4, 1); pin("drift_ratio_skew_e1", 1.97, 2)
    # цели агрегации и персонализация
    pin("f1_records", 0.85, 2); pin("f1_clients", 0.87, 2)
    pin("mean_client_records", 0.970, 3); pin("worst_records", 0.921, 3)
    pin("mean_client_clients", 0.974, 3); pin("worst_clients", 0.936, 3)
    pin("mean_client_personal", 0.967, 3); pin("worst_personal", 0.917, 3)
    pin("personal_better_share", 0.5, 3)
    # коммуникация
    pin("comm_kb_float32", 1.00, 2); pin("comm_kb_8 бит", 0.25, 2)
    pin("comm_kb_4 бита", 0.13, 2); pin("comm_kb_top-10 %", 0.10, 2)
    pin("comm_kb_top-1 %", 0.01, 2)
    pin("comm_f1_float32", 0.85, 2); pin("comm_f1_8 бит", 0.85, 2)
    pin("comm_f1_4 бита", 0.85, 2)
    pin("comm_f1_top-10 %", 0.79, 2); pin("comm_f1_top-1 %", 0.49, 2)
    pin("bytes_client_mb", 40.0, 1); pin("bytes_round_gb", 20.0, 1)
    pin("bytes_sp_gb", 0.3, 1)
    # приватность
    pin("norm_min", 1.71, 2); pin("norm_max", 5.09, 2); pin("norm_median", 2.73, 2)
    pin("clipped_at_C", 0.55, 2); pin("f1_clip_only", 0.85, 2)
    pin("f1_dp_0.2", 0.73, 2); pin("f1_dp_0.5", 0.65, 2); pin("f1_dp_1.0", 0.53, 2)
    pin("clip_factor", 0.25, 2); pin("clip_share_example", 0.4, 2)
    # ответы к задачам и полям
    pin("hw_records", 3.1, 2); pin("hw_clients", 2.33, 2); pin("mask_sum", 4.0, 3)
    print("pins: OK")

    FACTS.write_text(json.dumps(FACT, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    print("\n--- facts ---")
    for k in sorted(FACT):
        print(f"{k}: {FACT[k]}")
    print("\nOK")
