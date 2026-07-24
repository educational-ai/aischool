"""Deterministic figures for lesson 76: attention as query-key-value addressing.

Every number quoted in the lesson is computed here and asserted.

Real data: scripts/data/sms-spam-collection.tsv (5574 SMS). We build PPMI + SVD word
embeddings from the corpus itself and use them as keys/values of one attention head;
we also TRAIN a one-head attention-pooling classifier on the corpus and compare it with
mean pooling. Synthetic parts (variance of the dot product, timing) are explicitly
seeded and marked as model experiments in the text.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SMS = ROOT / "scripts" / "data" / "sms-spam-collection.tsv"
OUT = ROOT / "public" / "figures" / "lessons" / "76"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "76"
FACTS = ROOT / "scripts" / "data" / "lesson76_facts.json"

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

facts: dict[str, float | int | str | list] = {}


def save(fig, path, *, dpi=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def softmax(s):
    s = np.asarray(s, dtype=float)
    e = np.exp(s - s.max())
    return e / e.sum()


def entropy(a):
    a = np.asarray(a, dtype=float)
    m = a > 0
    return float(-(a[m] * np.log(a[m])).sum())


# ============================================================ real corpus + embeddings
TOKEN = re.compile(r"[a-z']+")


def load_sms():
    labels, docs = [], []
    with open(SMS, encoding="utf8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            lab, _, text = line.partition("\t")
            toks = [t for t in TOKEN.findall(text.lower()) if len(t) > 1]
            if not toks:
                continue
            labels.append(1 if lab == "spam" else 0)
            docs.append(toks)
    return np.array(labels), docs


LABELS, DOCS = load_sms()
facts["n_sms"] = int(len(DOCS))
facts["n_spam"] = int(LABELS.sum())
facts["spam_share_pct"] = round(100 * float(LABELS.mean()), 1)
assert len(DOCS) > 5000 and 0.10 < LABELS.mean() < 0.16

counts: dict[str, int] = {}
for d in DOCS:
    for t in d:
        counts[t] = counts.get(t, 0) + 1
VOCAB = sorted([w for w, c in counts.items() if c >= 10])
IDX = {w: i for i, w in enumerate(VOCAB)}
V = len(VOCAB)
facts["vocab"] = V
assert 900 < V < 2500


def embeddings(dim=32, window=4):
    """PPMI co-occurrence + truncated SVD, deterministic."""
    Cm = np.zeros((V, V))
    for d in DOCS:
        ids = [IDX[t] for t in d if t in IDX]
        for i, a in enumerate(ids):
            for b in ids[max(0, i - window):i]:
                Cm[a, b] += 1.0
                Cm[b, a] += 1.0
    total = Cm.sum()
    row = Cm.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = Cm * total / (row * row.T)
        M = np.where(Cm > 0, np.log(np.maximum(P, 1e-12)), 0.0)
    M = np.maximum(M, 0.0)
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    E = U[:, :dim] * S[:dim]
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    return E


DIM = 32
E = embeddings(DIM)
facts["emb_dim"] = DIM


def sim_top(word, k=5):
    v = E[IDX[word]]
    s = E @ v
    s[IDX[word]] = -9
    o = np.argsort(-s)[:k]
    return [(VOCAB[i], round(float(s[i]), 3)) for i in o]


facts["near_free"] = sim_top("free", 4)
facts["near_call"] = sim_top("call", 4)
assert facts["near_free"][0] == ["nokia", 0.912] or facts["near_free"][0] == ("nokia", 0.912)
assert facts["near_call"][0][0] == "land" and abs(facts["near_call"][0][1] - 0.847) < 5e-4
assert abs(facts["near_free"][1][1] - 0.850) < 5e-4 and facts["near_free"][1][0] == "mobile"
assert abs(facts["near_call"][3][1] - 0.799) < 5e-4 and facts["near_call"][3][0] == "claim"
assert facts["n_sms"] == 5569 and facts["n_spam"] == 747 and facts["vocab"] == 1045
print("near free:", facts["near_free"])
print("near call:", facts["near_call"])


# ============================================================ fig 76.1 mechanism, exact arithmetic
def fig_mechanism():
    s = np.array([2.0, 1.0, 0.0])
    v = np.array([1.0, 4.0, -2.0])
    a = softmax(s)
    out = float(a @ v)
    a_mask = softmax(s[:2])
    out_mask = float(a_mask @ v[:2])
    facts["toy_scores"] = [2, 1, 0]
    facts["toy_values"] = [1, 4, -2]
    facts["toy_weights"] = [round(float(x), 4) for x in a]
    facts["toy_out"] = round(out, 4)
    facts["toy_weights_mask"] = [round(float(x), 4) for x in a_mask]
    facts["toy_out_mask"] = round(out_mask, 4)
    facts["toy_entropy"] = round(entropy(a), 3)
    assert abs(a[0] - 0.6652) < 1e-3 and abs(out - 1.4640) < 1e-3
    assert abs(a_mask[0] - 0.7311) < 1e-3 and abs(out_mask - 1.8068) < 1e-3

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9))
    names = ["позиция 1", "позиция 2", "позиция 3"]
    ax = axes[0]
    ax.bar(names, s, color=[BLUE, GREEN, GOLD], width=0.6)
    ax.set_title("scores $s_j=q^\\top k_j/\\sqrt{d_k}$", fontsize=11.5)
    ax.set_ylim(-0.3, 2.6)
    for i, val in enumerate(s):
        ax.text(i, val + 0.08, f"{val:.0f}", ha="center", fontsize=10, color=INK)
    ax = axes[1]
    ax.bar(names, a, color=[BLUE, GREEN, GOLD], width=0.6)
    ax.set_title("веса $a_j=\\mathrm{softmax}(s)_j$", fontsize=11.5)
    ax.set_ylim(0, 0.85)
    for i, val in enumerate(a):
        ax.text(i, val + 0.02, f"{val:.3f}", ha="center", fontsize=10, color=INK)
    ax.text(1, 0.78, "сумма = 1", ha="center", fontsize=10, color=MUTED)
    ax = axes[2]
    ax.hlines(0, -2.6, 4.6, color=LINE, lw=1.2)
    for val, c, nm in zip(v, [BLUE, GREEN, GOLD], ["$v_1$", "$v_2$", "$v_3$"]):
        ax.plot(val, 0, "o", color=c, markersize=11)
        ax.annotate(nm, (val, 0.22), ha="center", fontsize=11, color=c)
    ax.plot([v.min(), v.max()], [-0.28, -0.28], color=MUTED, lw=2.0)
    ax.text(1.0, -0.52, "выпуклая оболочка value", ha="center", fontsize=9.5, color=MUTED)
    ax.plot(out, 0, "*", color=RED, markersize=20)
    ax.annotate(f"выход $z={out:.3f}$", (out, 0.5), ha="center", fontsize=11, color=RED)
    ax.plot(out_mask, 0, "*", color=VIOLET, markersize=15)
    ax.annotate(f"с маской $j\\leq 2$: {out_mask:.3f}", (out_mask, -0.95), ha="center", fontsize=10, color=VIOLET)
    ax.set_ylim(-1.25, 0.95); ax.set_xlim(-2.8, 4.8); ax.set_yticks([])
    ax.set_title("выход — смесь value, а не key", fontsize=11.5)
    for ax in axes[:2]:
        ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
        ax.tick_params(labelsize=9.5)
    fig.suptitle("Три шага одной головы: совместимость, конкуренция, перенос содержимого", y=1.04, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "mechanism.png")
    print("mechanism:", facts["toy_weights"], facts["toy_out"])


# ============================================================ fig 76.2 real attention map
def pick_message():
    best = None
    for lab, d in zip(LABELS, DOCS):
        toks = [t for t in d if t in IDX]
        # keep the order, drop repeats
        seen, uniq = set(), []
        for t in toks:
            if t not in seen:
                seen.add(t); uniq.append(t)
        if lab == 1 and 8 <= len(uniq) <= 9 and "call" in uniq and "free" in uniq:
            best = uniq
            break
    assert best is not None
    return best


def fig_attention_map():
    toks = pick_message()
    facts["demo_tokens"] = toks
    n = len(toks)
    Xe = E[[IDX[t] for t in toks]]
    S = (Xe @ Xe.T) * np.sqrt(DIM) / np.sqrt(DIM)  # identity projections, scaled by sqrt(d)
    S = (Xe @ Xe.T) * (DIM / np.sqrt(DIM))         # raw cosines are tiny: use q=k=sqrt(d)*e
    A = np.array([softmax(r) for r in S])
    mask = np.tril(np.ones((n, n)))
    Sm = np.where(mask > 0, S, -np.inf)
    Am = np.array([softmax(r[np.isfinite(r)]).tolist() + [0.0] * (n - int(np.isfinite(r).sum())) for r in Sm])

    qi = toks.index("call")
    j_free = toks.index("free")
    facts["demo_query"] = "call"
    facts["demo_self_weight"] = round(float(A[qi, qi]), 3)
    facts["demo_free_weight"] = round(float(A[qi, j_free]), 3)
    facts["demo_entropy_full"] = round(entropy(A[qi]), 3)
    facts["demo_entropy_uniform"] = round(float(np.log(n)), 3)
    facts["demo_entropy_last_causal"] = round(entropy(Am[-1][Am[-1] > 0]), 3)
    facts["demo_first_row_selfweight"] = round(float(Am[0, 0]), 3)
    facts["demo_n"] = n
    facts["demo_self_weight_causal"] = round(float(Am[qi, qi]), 3)
    facts["demo_call_orange_causal"] = round(float(Am[toks.index("orange"), toks.index("on")]), 3)
    assert abs(Am[0, 0] - 1.0) < 1e-9
    assert facts["demo_self_weight"] > facts["demo_free_weight"]
    assert facts["demo_entropy_full"] < facts["demo_entropy_uniform"]
    assert abs(facts["demo_self_weight"] - 0.616) < 5e-4
    assert abs(facts["demo_free_weight"] - 0.036) < 5e-4
    assert abs(facts["demo_entropy_full"] - 1.413) < 5e-4
    assert abs(facts["demo_self_weight_causal"] - 0.793) < 5e-4

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.1))
    for ax, M, ttl in ((axes[0], A, "полная карта: каждый видит всех"),
                       (axes[1], Am, "причинная маска: только прошлое и себя")):
        im = ax.imshow(M, cmap="YlGnBu", vmin=0, vmax=max(A.max(), Am.max()))
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(toks, rotation=45, ha="right", fontsize=9.5)
        ax.set_yticklabels(toks, fontsize=9.5)
        ax.set_title(ttl, fontsize=11.5)
        ax.set_xlabel("ключ $j$", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("запрос $i$", fontsize=10)
        for i in range(n):
            for j in range(n):
                if M[i, j] >= 0.16:
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7.5,
                            color=PAPER if M[i, j] > 0.45 else INK)
    fig.colorbar(im, ax=axes, shrink=0.72, label="вес $a_{ij}$")
    fig.suptitle("Одна голова на реальном SMS: embeddings корпуса как key и value", y=0.99, fontsize=13.5)
    save(fig, OUT / "attention_map.png")
    print("map:", toks, facts["demo_self_weight"], facts["demo_free_weight"])


# ============================================================ fig 76.3 why divide by sqrt(d)
def fig_scaling():
    rng = np.random.default_rng(760)
    dims = [4, 8, 16, 32, 64, 128, 256, 512]
    reps = 4000
    var_raw, maxw_raw, maxw_sc, ent_raw, ent_sc, grad_raw, grad_sc = [], [], [], [], [], [], []
    n_keys = 8
    for d in dims:
        q = rng.standard_normal((reps, d))
        k = rng.standard_normal((reps, n_keys, d))
        s = np.einsum("rd,rkd->rk", q, k)
        var_raw.append(float(s.var()))
        for arr_s, mw, en, gr in ((s, maxw_raw, ent_raw, grad_raw),
                                  (s / np.sqrt(d), maxw_sc, ent_sc, grad_sc)):
            A = np.exp(arr_s - arr_s.max(1, keepdims=True))
            A /= A.sum(1, keepdims=True)
            mw.append(float(A.max(1).mean()))
            en.append(float(np.mean([entropy(r) for r in A[:400]])))
            gr.append(float(np.mean((A * (1 - A)).sum(1))))
    facts["var_ratio_d"] = [round(v / d, 3) for v, d in zip(var_raw, dims)]
    facts["maxw_raw_d512"] = round(maxw_raw[-1], 3)
    facts["maxw_sc_d512"] = round(maxw_sc[-1], 3)
    facts["ent_raw_d512"] = round(ent_raw[-1], 3)
    facts["ent_sc_d512"] = round(ent_sc[-1], 3)
    facts["ent_uniform8"] = round(float(np.log(8)), 3)
    facts["grad_raw_d512"] = round(grad_raw[-1], 4)
    facts["grad_sc_d512"] = round(grad_sc[-1], 4)
    facts["grad_ratio_d512"] = round(grad_sc[-1] / grad_raw[-1], 1)
    facts["maxw_sc_d4"] = round(maxw_sc[0], 3)
    facts["var_d64"] = round(var_raw[dims.index(64)], 1)
    assert all(0.94 < r < 1.06 for r in facts["var_ratio_d"])
    assert facts["maxw_raw_d512"] > 0.94 > facts["maxw_sc_d512"]
    assert facts["grad_ratio_d512"] > 10
    assert abs(facts["var_d64"] - 64.2) < 0.05
    assert abs(facts["maxw_raw_d512"] - 0.955) < 5e-4
    assert abs(facts["maxw_sc_d512"] - 0.360) < 5e-4
    assert abs(facts["ent_raw_d512"] - 0.107) < 5e-4
    assert abs(facts["ent_sc_d512"] - 1.729) < 5e-4
    assert abs(facts["grad_raw_d512"] - 0.0648) < 5e-5
    assert abs(facts["grad_sc_d512"] - 0.7688) < 5e-5
    assert min(facts["var_ratio_d"]) == 0.977 and max(facts["var_ratio_d"]) == 1.003

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9))
    ax = axes[0]
    ax.plot(dims, var_raw, "o-", color=RED, lw=2.0, label="измеренная $\\mathrm{Var}(q^\\top k)$")
    ax.plot(dims, dims, ls=(0, (5, 3)), color=MUTED, lw=1.4, label="$d_k$")
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlabel("$d_k$"); ax.set_title("дисперсия растёт как $d_k$", fontsize=11.5)
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    ax.plot(dims, maxw_raw, "o-", color=RED, lw=2.0, label="без деления")
    ax.plot(dims, maxw_sc, "o-", color=BLUE, lw=2.0, label="делим на $\\sqrt{d_k}$")
    ax.axhline(1 / n_keys, color=MUTED, lw=1.0, ls=(0, (2, 2)))
    ax.text(6, 1 / n_keys + 0.03, "равномерно 1/8", fontsize=9, color=MUTED)
    ax.set_xscale("log", base=2); ax.set_ylim(0, 1.05)
    ax.set_xlabel("$d_k$"); ax.set_title("максимальный вес softmax", fontsize=11.5)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    ax = axes[2]
    ax.plot(dims, grad_raw, "o-", color=RED, lw=2.0, label="без деления")
    ax.plot(dims, grad_sc, "o-", color=BLUE, lw=2.0, label="делим на $\\sqrt{d_k}$")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("$d_k$"); ax.set_title("чувствительность $\\sum_j a_j(1-a_j)$", fontsize=11.5)
    ax.legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True); ax.tick_params(labelsize=9)
    fig.suptitle("Модельный эксперимент (seed 760): почему logits делят на корень размерности", y=1.04, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "scaling.png")
    print("scaling:", facts["maxw_raw_d512"], facts["maxw_sc_d512"], facts["grad_ratio_d512"])


# ============================================================ fig 76.4 quadratic cost, measured
def fig_cost():
    rng = np.random.default_rng(7604)
    ns = [128, 256, 512, 1024, 2048]
    d = 64
    times, mems = [], []
    for n in ns:
        Q = rng.standard_normal((n, d)).astype(np.float32)
        K = rng.standard_normal((n, d)).astype(np.float32)
        Q @ K.T  # warm up
        best = []
        for _ in range(5):
            t0 = time.perf_counter()
            S = Q @ K.T
            S = np.exp(S - S.max(1, keepdims=True))
            S /= S.sum(1, keepdims=True)
            best.append(time.perf_counter() - t0)
        times.append(float(np.median(best)))
        mems.append(n * n * 4 / 2 ** 20)
    logn, logt = np.log(ns), np.log(times)
    b = float(np.polyfit(logn, logt, 1)[0])
    facts["cost_ns"] = ns
    facts["cost_ms"] = [round(1000 * t, 2) for t in times]
    facts["cost_slope"] = round(b, 2)
    facts["mem_2048_mib"] = round(mems[-1], 1)
    facts["mem_8192_mib"] = round(8192 * 8192 * 4 / 2 ** 20, 1)
    facts["cost_ratio_1024_512"] = round(times[3] / times[2], 2)
    assert 1.6 < b < 2.4, b
    assert 3.2 < facts["cost_ratio_1024_512"] < 4.8
    assert abs(facts["mem_2048_mib"] - 16.0) < 0.1
    assert abs(facts["mem_8192_mib"] - 256.0) < 0.1

    win = 128
    ns_w = np.array([256, 512, 1024, 2048, 4096, 8192])
    full_cells = ns_w.astype(float) ** 2
    local_cells = ns_w.astype(float) * win
    facts["local_saving_8192"] = int(round(full_cells[-1] / local_cells[-1]))
    assert facts["local_saving_8192"] == 64

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
    ax = axes[0]
    ax.plot(ns, [1000 * t for t in times], "o-", color=RED, lw=2.2, label=f"измерено, наклон $b={b:.2f}$")
    ref = 1000 * times[0] * (np.array(ns) / ns[0]) ** 2
    ax.plot(ns, ref, ls=(0, (5, 3)), color=MUTED, lw=1.5, label="эталон $n^2$")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("длина контекста $n$"); ax.set_ylabel("время, мс")
    ax.set_title("реальный замер $QK^\\top$ и softmax, $d_k=64$, float32", fontsize=11.5)
    ax.legend(frameon=False, fontsize=9.5)
    ax = axes[1]
    ax.plot(ns_w, full_cells, "o-", color=RED, lw=2.2, label="полная карта: $n^2$ ячеек")
    ax.plot(ns_w, local_cells, "o-", color=GREEN, lw=2.2, label="окно 128: $128n$ ячеек")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("длина контекста $n$"); ax.set_ylabel("число ячеек карты")
    ax.set_title(f"при $n=8192$ окно дешевле в {facts['local_saving_8192']} раз", fontsize=11.5)
    ax.legend(frameon=False, fontsize=9.5)
    for ax in axes:
        ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True); ax.tick_params(labelsize=9)
    fig.suptitle("Квадратичная цена внимания: измеренная и подсчитанная", y=1.02, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "cost.png")
    print("cost:", facts["cost_ms"], "slope", b)


# ============================================================ fig 76.5 attention pooling on real SMS
def train_pooling():
    rng = np.random.default_rng(7605)
    ids = [[IDX[t] for t in d if t in IDX] for d in DOCS]
    keep = [i for i, x in enumerate(ids) if len(x) >= 2]
    keep = np.array(keep)
    perm = rng.permutation(len(keep))
    keep = keep[perm]
    ntr = int(0.75 * len(keep))
    tr, te = keep[:ntr], keep[ntr:]
    y = LABELS

    def fit(mode, epochs=300, lr=3.0):
        u = np.zeros(DIM); v = np.zeros(DIM); b = 0.0
        mu = np.zeros(DIM); mv = np.zeros(DIM); mb = 0.0
        for ep in range(epochs):
            gu = np.zeros(DIM); gv = np.zeros(DIM); gb = 0.0
            for i in tr:
                Em = E[ids[i]]
                if mode == "attn":
                    s = (Em @ u) / np.sqrt(DIM)
                    a = softmax(s)
                else:
                    a = np.full(len(Em), 1.0 / len(Em))
                c = a @ Em
                z = float(v @ c) + b
                p = 1 / (1 + np.exp(-z))
                g = p - y[i]
                gv += g * c; gb += g
                if mode == "attn":
                    da = g * (Em @ v)
                    ds = a * (da - float(a @ da))
                    gu += (ds @ Em) / np.sqrt(DIM)
            m = len(tr)
            gu /= m; gv /= m; gb /= m
            mu = 0.9 * mu + gu; mv = 0.9 * mv + gv; mb = 0.9 * mb + gb
            u -= lr * mu; v -= lr * mv; b -= lr * mb
        return u, v, b

    def acc(params, mode, idxs):
        u, v, b = params
        ok = 0
        for i in idxs:
            Em = E[ids[i]]
            a = softmax((Em @ u) / np.sqrt(DIM)) if mode == "attn" else np.full(len(Em), 1.0 / len(Em))
            z = float(v @ (a @ Em)) + b
            ok += int((z > 0) == bool(y[i]))
        return ok / len(idxs)

    p_mean = fit("mean")
    p_attn = fit("attn")
    a_mean = acc(p_mean, "mean", te)
    a_attn = acc(p_attn, "attn", te)
    base = 1 - y[te].mean()
    L = np.array([len(ids[i]) for i in te])
    mid = te[(L >= 10) & (L < 25)]
    long_acc = (acc(p_mean, "mean", mid), acc(p_attn, "attn", mid), int(len(mid)))
    return p_attn, ids, tr, te, a_mean, a_attn, float(base), long_acc


def fig_pooling():
    (u, v, b), ids, tr, te, a_mean, a_attn, base, long_acc = train_pooling()
    facts["pool_mid_n"] = long_acc[2]
    facts["pool_mid_mean"] = round(100 * long_acc[0], 1)
    facts["pool_mid_attn"] = round(100 * long_acc[1], 1)
    facts["pool_err_mean"] = int(round((1 - a_mean) * len(te)))
    facts["pool_err_attn"] = int(round((1 - a_attn) * len(te)))
    facts["pool_test_n"] = int(len(te))
    facts["pool_acc_mean"] = round(100 * a_mean, 1)
    facts["pool_acc_attn"] = round(100 * a_attn, 1)
    facts["pool_acc_base"] = round(100 * base, 1)
    facts["pool_gain_pp"] = round(100 * (a_attn - a_mean), 1)
    assert a_attn > a_mean > 0.5
    assert a_attn > base
    assert facts["pool_acc_mean"] == 96.3 and facts["pool_acc_attn"] == 96.6
    assert facts["pool_acc_base"] == 87.0 and facts["pool_test_n"] == 1368
    assert facts["pool_err_mean"] == 51 and facts["pool_err_attn"] == 47
    assert facts["pool_mid_n"] == 586
    assert facts["pool_mid_mean"] == 94.5 and facts["pool_mid_attn"] == 95.2

    # which words the trained head raises: score of each vocabulary word
    sc = (E @ u) / np.sqrt(DIM)
    freq = np.array([counts[w] for w in VOCAB])
    common = freq >= 40
    order_hi = np.argsort(-np.where(common, sc, -9))[:10]
    order_lo = np.argsort(np.where(common, sc, 9))[:6]
    top_words = [(VOCAB[i], round(float(sc[i]), 2)) for i in order_hi]
    low_words = [(VOCAB[i], round(float(sc[i]), 2)) for i in order_lo]
    facts["attn_top_words"] = top_words
    facts["attn_low_words"] = low_words

    # weights inside the demo message under the trained head
    toks = facts["demo_tokens"]
    Em = E[[IDX[t] for t in toks]]
    a_demo = softmax((Em @ u) / np.sqrt(DIM))
    facts["demo_trained_weights"] = [round(float(x), 3) for x in a_demo]
    facts["demo_trained_max_token"] = toks[int(np.argmax(a_demo))]
    facts["demo_trained_max_weight"] = round(float(a_demo.max()), 3)
    facts["demo_uniform_weight"] = round(1 / len(toks), 3)
    assert a_demo.max() > 1.4 * (1 / len(toks))
    assert facts["demo_trained_max_token"] == "optout"
    assert abs(facts["demo_trained_max_weight"] - 0.187) < 5e-4
    assert 1.6 < facts["demo_trained_max_weight"] / facts["demo_uniform_weight"] < 1.75
    assert [w for w, _ in top_words[:5]] == ["tone", "uk", "co", "www", "pobox"]
    assert [round(x, 2) for _, x in top_words[:5]] == [0.36, 0.33, 0.32, 0.22, 0.19]
    assert [w for w, _ in low_words[:3]] == ["that", "and", "it"]
    assert [round(x, 2) for _, x in low_words[:3]] == [-1.49, -1.39, -1.37]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4),
                             gridspec_kw={"width_ratios": [1.0, 1.25]})
    ax = axes[0]
    names = ["всегда «не спам»", "среднее по токенам", "внимание (1 голова)"]
    vals = [100 * base, 100 * a_mean, 100 * a_attn]
    bars = ax.barh(names, vals, color=[FAINT, BLUE, RED], height=0.55)
    ax.set_xlim(0, 104); ax.set_xlabel("точность на отложенных SMS, %")
    for r, val in zip(bars, vals):
        ax.text(val + 1.2, r.get_y() + r.get_height() / 2, f"{val:.1f}", va="center", fontsize=10.5, color=INK)
    ax.set_title("одна обученная голова против среднего", fontsize=11.5)
    ax.tick_params(labelsize=9.5)
    ax = axes[1]
    ws = [w for w, _ in top_words][::-1]
    vs = [s for _, s in top_words][::-1]
    ax.barh(ws, vs, color=RED, height=0.6)
    ax.set_xlabel("score $u^\\top e_w/\\sqrt{d}$, обученный запрос")
    ax.set_title("какие слова голова поднимает выше всех", fontsize=11.5)
    ax.tick_params(labelsize=9.5)
    for ax in axes:
        ax.grid(True, axis="x", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.suptitle("Реальный SMS-корпус: внимание как обучаемое взвешивание слов", y=1.02, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "pooling.png")
    print("pooling:", facts["pool_acc_mean"], facts["pool_acc_attn"], top_words[:5])


# ============================================================ fig 76.6 dilution + heads
def fig_dilution():
    s_anchor, s_other = 4.0, 1.0
    ks = np.arange(0, 41)
    w = np.exp(s_anchor) / (np.exp(s_anchor) + (1 + ks) * np.exp(s_other))
    facts["dilute_w0"] = round(float(w[0]), 3)
    facts["dilute_w10"] = round(float(w[10]), 3)
    facts["dilute_w40"] = round(float(w[40]), 3)
    facts["dilute_half_k"] = int(np.argmax(w <= facts["dilute_w0"] / 2))
    assert abs(facts["dilute_w0"] - 0.953) < 5e-4
    assert abs(facts["dilute_w10"] - 0.646) < 5e-4
    assert abs(facts["dilute_w40"] - 0.329) < 5e-4
    assert facts["dilute_half_k"] == 22
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    ax.plot(ks, w, color=RED, lw=2.0)
    ax.plot([0, 10, 40], [w[0], w[10], w[40]], "o", color=INK, markersize=5)
    ax.set_xlabel("добавлено нерелевантных токенов", fontsize=9)
    ax.set_ylabel("вес опоры", fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.set_title("score не менялся — вес упал", fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)
    save(fig, SIDE / "dilution.png")
    print("dilution:", facts["dilute_w0"], facts["dilute_w10"], facts["dilute_w40"])


def side_permutation():
    toks = facts["demo_tokens"][:5]
    Xe = E[[IDX[t] for t in toks]]
    S = (Xe @ Xe.T) * (DIM / np.sqrt(DIM))
    A = np.array([softmax(r) for r in S])
    p = [2, 0, 4, 1, 3]
    Ap = A[np.ix_(p, p)]
    Sp = (Xe[p] @ Xe[p].T) * (DIM / np.sqrt(DIM))
    Ap2 = np.array([softmax(r) for r in Sp])
    facts["perm_maxdiff"] = float(np.abs(Ap - Ap2).max())
    assert facts["perm_maxdiff"] < 1e-12
    fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.4))
    for ax, M, t in ((axes[0], A, "исходный порядок"), (axes[1], Ap2, "переставленный")):
        ax.imshow(M, cmap="YlGnBu", vmin=0, vmax=A.max())
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, fontsize=8.5)
    fig.suptitle("та же карта, только переставленная", y=1.02, fontsize=9)
    fig.tight_layout()
    save(fig, SIDE / "permutation.png")


def side_posenc():
    d, n = 64, 64
    pos = np.arange(n)[:, None]
    i = np.arange(d // 2)[None, :]
    ang = pos / np.power(10000.0, 2 * i / d)
    PE = np.zeros((n, d))
    PE[:, 0::2] = np.sin(ang); PE[:, 1::2] = np.cos(ang)
    dots = PE[32] @ PE.T
    facts["pe_self_dot"] = round(float(dots[32]), 2)
    facts["pe_dot_lag1"] = round(float(dots[33]), 2)
    facts["pe_dot_lag8"] = round(float(dots[40]), 2)
    facts["pe_dot_lag32"] = round(float(dots[63] if False else dots[0]), 2)
    assert abs(facts["pe_self_dot"] - 32.0) < 0.01
    assert abs(facts["pe_dot_lag1"] - 30.92) < 0.01
    assert abs(facts["pe_dot_lag8"] - 22.41) < 0.01
    fig, axes = plt.subplots(2, 1, figsize=(4.2, 3.6),
                             gridspec_kw={"height_ratios": [1.4, 1.0]})
    axes[0].imshow(PE.T, cmap="RdBu", aspect="auto", vmin=-1, vmax=1)
    axes[0].set_xlabel("позиция", fontsize=8); axes[0].set_ylabel("координата", fontsize=8)
    axes[0].tick_params(labelsize=7)
    axes[0].set_title("синусоидальный код позиции", fontsize=9)
    axes[1].plot(np.arange(n) - 32, dots, color=BLUE, lw=1.6)
    axes[1].set_xlabel("сдвиг позиции", fontsize=8); axes[1].set_ylabel("$PE\\cdot PE$", fontsize=8)
    axes[1].tick_params(labelsize=7)
    axes[1].grid(True, color=GRID, lw=0.4, alpha=0.5)
    fig.tight_layout()
    save(fig, SIDE / "posenc.png")
    print("pe:", facts["pe_self_dot"], facts["pe_dot_lag1"], facts["pe_dot_lag8"])


def side_kvcache():
    n = np.arange(1, 513)
    without = n ** 2          # пересчёт всей карты на каждом шаге
    with_cache = n            # только новая строка
    facts["kv_ratio_512"] = int(512)
    facts["kv_total_ratio"] = round(float(without.sum() / with_cache.sum()), 1)
    assert facts["kv_total_ratio"] > 100
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.plot(n, without, color=RED, lw=1.8, label="без кэша: $O(n^2)$")
    ax.plot(n, with_cache, color=GREEN, lw=1.8, label="KV-cache: $O(n)$")
    ax.set_yscale("log")
    ax.set_xlabel("номер сгенерированного токена", fontsize=9)
    ax.set_ylabel("операций на шаг", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("что даёт кэш ключей и значений", fontsize=9)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)
    save(fig, SIDE / "kvcache.png")


# ============================================================ fig 76.7 two heads see different things
def fig_heads():
    """Two hand-built heads on the demo message: one local (previous token),
    one content-based (nearest embedding). Shows specialisation without mysticism."""
    toks = facts["demo_tokens"]
    n = len(toks)
    Xe = E[[IDX[t] for t in toks]]
    # head A: positional — prefers the previous token
    pos = np.arange(n)
    Sa = -3.0 * np.abs(pos[:, None] - (pos[:, None] - 1))
    Sa = -3.0 * (np.abs(pos[None, :] - (pos[:, None] - 1)))
    Aa = np.array([softmax(r) for r in Sa])
    # head B: content — cosine similarity of embeddings
    Sb = (Xe @ Xe.T) * (DIM / np.sqrt(DIM))
    np.fill_diagonal(Sb, -np.inf)
    Ab = np.array([softmax(r[np.isfinite(r)]) for r in Sb])
    Ab_full = np.zeros((n, n))
    for i in range(n):
        j = 0
        for k in range(n):
            if k == i:
                continue
            Ab_full[i, k] = Ab[i][j]; j += 1
    facts["headA_prev_weight"] = round(float(np.mean([Aa[i, i - 1] for i in range(1, n)])), 3)
    facts["headB_maxpair"] = [toks[int(np.argmax(Ab_full) // n)], toks[int(np.argmax(Ab_full) % n)]]
    facts["headB_maxweight"] = round(float(Ab_full.max()), 3)
    assert abs(facts["headA_prev_weight"] - 0.911) < 5e-4
    assert facts["headB_maxpair"] == ["mobileupd", "motorola"]
    assert abs(facts["headB_maxweight"] - 0.354) < 5e-4
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7))
    for ax, M, t in ((axes[0], Aa, "голова A: «смотри на предыдущий токен» (позиция)"),
                     (axes[1], Ab_full, "голова B: «ищи похожего по смыслу» (содержание)")):
        ax.imshow(M, cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(toks, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(toks, fontsize=9)
        ax.set_title(t, fontsize=11)
        ax.set_xlabel("ключ $j$", fontsize=9.5); ax.set_ylabel("запрос $i$", fontsize=9.5)
    fig.suptitle("Разные головы — разные подпространства и разные вопросы к контексту", y=1.0, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "heads.png")
    print("heads:", facts["headA_prev_weight"], facts["headB_maxpair"])


fig_mechanism()
fig_attention_map()
fig_scaling()
fig_cost()
fig_pooling()
fig_heads()
fig_dilution()
side_permutation()
side_posenc()
side_kvcache()

FACTS.write_text(json.dumps(facts, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf8")
print(json.dumps(facts, ensure_ascii=False, indent=1, sort_keys=True))
print("lesson 76 figures written")
