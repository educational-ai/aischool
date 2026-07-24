"""Deterministic figures for lesson 77: the transformer block as a system of paths.

Everything quoted in the prose is computed here and asserted:
  * exact parameter budget of a pre-norm block (d=512, 8 heads, expansion 4);
  * measured forward time of MHA vs MLP in numpy -> empirical crossover in n;
  * signal through depth: norm and cosine with the initial embedding, with and
    without the residual path, pre-norm vs post-norm;
  * routing: sensitivity of the last position to the FIRST token embedding,
    with attention on and off (exactly zero without attention);
  * a linear-probe ablation on REAL SMS messages (scripts/data/sms-spam-collection.tsv):
    what a frozen random block delivers to the head when parts are removed;
  * layer normalization on REAL digits (sklearn load_digits): shift/scale invariance.

Run: python3 scripts/generate_lesson77_visuals.py
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression, Ridge

ROOT = Path(__file__).resolve().parents[1]
SMS = ROOT / "scripts" / "data" / "sms-spam-collection.tsv"
OUT = ROOT / "public" / "figures" / "lessons" / "77"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "77"

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


# ============================================================ block machinery
def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def layer_norm(x, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def make_block(rng, d, heads=4, expansion=4, gain=1.0, res_scale=1.0):
    """res_scale повторяет приём GPT-2: выходные проекции подслоёв делят на sqrt(2L)."""
    s = gain / np.sqrt(d)
    return {
        "Wq": rng.normal(0, s, (d, d)), "Wk": rng.normal(0, s, (d, d)),
        "Wv": rng.normal(0, s, (d, d)), "Wo": rng.normal(0, s, (d, d)) * res_scale,
        "W1": rng.normal(0, s, (d, expansion * d)),
        "W2": rng.normal(0, np.sqrt(1.0 / (expansion * d)) * gain, (expansion * d, d)) * res_scale,
        "heads": heads,
    }


def mha(x, p, causal=False):
    d = x.shape[-1]
    h = p["heads"]
    dh = d // h
    q = (x @ p["Wq"]).reshape(-1, h, dh).transpose(1, 0, 2)
    k = (x @ p["Wk"]).reshape(-1, h, dh).transpose(1, 0, 2)
    v = (x @ p["Wv"]).reshape(-1, h, dh).transpose(1, 0, 2)
    sc = q @ k.transpose(0, 2, 1) / np.sqrt(dh)
    if causal:
        n = x.shape[0]
        sc = sc + np.triu(np.full((n, n), -1e9), 1)
    sc = sc - sc.max(axis=-1, keepdims=True)
    a = np.exp(sc)
    a /= a.sum(axis=-1, keepdims=True)
    o = (a @ v).transpose(1, 0, 2).reshape(-1, d)
    return o @ p["Wo"]


def mlp(x, p):
    return gelu(x @ p["W1"]) @ p["W2"]


def run_stack(x, blocks, *, residual=True, use_attn=True, use_mlp=True,
              norm="pre", causal=False):
    """One forward pass through a stack of frozen random blocks."""
    for p in blocks:
        if use_attn:
            if norm == "pre":
                f = mha(layer_norm(x), p, causal)
                x = x + f if residual else f
            else:
                f = mha(x, p, causal)
                x = layer_norm(x + f) if residual else layer_norm(f)
        if use_mlp:
            if norm == "pre":
                f = mlp(layer_norm(x), p)
                x = x + f if residual else f
            else:
                f = mlp(x, p)
                x = layer_norm(x + f) if residual else layer_norm(f)
    return x


# ============================================================ fig 77.1 anatomy
def fig_anatomy():
    d = 512
    mha_p = 4 * d * d + 4 * d
    mlp_p = d * (4 * d) + 4 * d + (4 * d) * d + d
    ln_p = 2 * 2 * d
    total = mha_p + mlp_p + ln_p
    assert (mha_p, mlp_p, ln_p, total) == (1050624, 2099712, 2048, 3152384)
    FACTS.update(mha_params=mha_p, mlp_params=mlp_p, ln_params=ln_p, block_params=total,
                 mlp_share=mlp_p / total, mha_share=mha_p / total)

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    def box(x, y, w, h, text, color, fc=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                    linewidth=1.6, edgecolor=color,
                                    facecolor=fc or PAPER, zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=11.5, color=INK, zorder=4)

    def arrow(x1, y1, x2, y2, color=MUTED, style="-|>", lw=1.4, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=13, color=color, lw=lw,
                                     linestyle=ls, zorder=2))

    ax.text(1.05, 9.5, "вход  $X\\in\\mathbb{R}^{n\\times d}$", fontsize=12, color=MUTED)
    arrow(1.2, 9.2, 1.2, 0.7, color=GOLD, lw=2.4, style="-")
    ax.text(0.35, 5.0, "residual stream", rotation=90, ha="center", va="center",
            fontsize=11, color=GOLD)

    box(3.2, 7.3, 2.2, 0.9, "LN", VIOLET, WASH)
    box(6.0, 7.3, 3.2, 0.9, f"MHA  ·  {mha_p:,} пар.".replace(",", " "), BLUE)
    box(3.2, 2.6, 2.2, 0.9, "LN", VIOLET, WASH)
    box(6.0, 2.6, 3.2, 0.9, f"MLP $d\\!\\to\\!4d\\!\\to\\!d$ · {mlp_p:,} пар.".replace(",", " "), RED)

    for y, col in ((7.75, BLUE), (3.05, RED)):
        arrow(1.2, y, 3.2, y, color=col)
        arrow(5.4, y, 6.0, y, color=col)
        arrow(9.2, y, 9.7, y, color=col)
        arrow(9.7, y, 9.7, y - 1.5, color=col, style="-")
        arrow(9.7, y - 1.5, 1.2, y - 1.5, color=col)
    for y in (6.25, 1.55):
        ax.scatter([1.2], [y], s=230, facecolor=PAPER, edgecolor=GOLD, zorder=5, lw=1.6)
        ax.text(1.2, y, "+", ha="center", va="center", fontsize=15, color=GOLD, zorder=6)

    ax.text(6.05, 8.55, "смешивает ПОЗИЦИИ  ($n\\times n$ внимание)", fontsize=10.5, color=BLUE)
    ax.text(6.05, 3.85, "смешивает КАНАЛЫ (каждая строка отдельно)", fontsize=10.5, color=RED)
    ax.text(3.05, 0.35, f"всего в блоке {total:,} параметров;  MLP — "
                        f"{100 * mlp_p / total:.1f}%,  MHA — {100 * mha_p / total:.1f}%,  "
                        f"LN — {ln_p} шт.".replace(",", " "),
            fontsize=10.5, color=MUTED)
    ax.set_title("Грамматика pre-norm блока: две ветви и один сквозной путь", pad=14)
    save(fig, OUT / "block_anatomy.png")
    print(f"anatomy: MHA={mha_p} MLP={mlp_p} LN={ln_p} total={total} "
          f"mlp_share={mlp_p / total:.4f}")


# ============================================================ fig 77.2 params
def fig_params():
    ds = [128, 256, 512, 1024]
    mha_p = [4 * d * d + 4 * d for d in ds]
    mlp_p = [8 * d * d + 5 * d for d in ds]
    ln_p = [4 * d for d in ds]
    ratios = [m / a for m, a in zip(mlp_p, mha_p)]
    assert all(abs(r - 2.0) < 0.02 for r in ratios)
    exps = np.array([1, 2, 4, 8, 16])
    share = (2 * exps * 512 ** 2 + (exps + 1) * 512) / (
        2 * exps * 512 ** 2 + (exps + 1) * 512 + 4 * 512 ** 2 + 4 * 512 + 4 * 512)
    FACTS.update(mlp_share_exp2=share[1], mlp_share_exp8=share[3], ratio512=ratios[2])

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    ax = axes[0]
    xs = np.arange(len(ds))
    ax.bar(xs - 0.2, np.array(mha_p) / 1e6, width=0.36, color=BLUE, label="MHA")
    ax.bar(xs + 0.2, np.array(mlp_p) / 1e6, width=0.36, color=RED, label="MLP (4d)")
    for i, d in enumerate(ds):
        ax.text(xs[i], mlp_p[i] / 1e6 + 0.25, f"×{ratios[i]:.2f}", ha="center",
                fontsize=10, color=MUTED)
    ax.set_xticks(xs); ax.set_xticklabels([f"d={d}" for d in ds])
    ax.set_ylabel("параметров в блоке, млн")
    ax.set_title("MLP вдвое тяжелее внимания", fontsize=13)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)

    ax = axes[1]
    ax.plot(exps, 100 * share, "o-", color=RED, lw=2.2)
    ax.axhline(100 * share[2], color=MUTED, lw=0.8, ls=(0, (2, 2)))
    ax.text(4.15, 100 * share[2] + 3, f"expansion 4: {100 * share[2]:.1f}%",
            fontsize=10.5, color=MUTED)
    ax.set_xscale("log", base=2); ax.set_xticks(exps)
    ax.set_xticklabels([str(int(e)) for e in exps])
    ax.set_xlabel("expansion MLP"); ax.set_ylabel("доля параметров блока в MLP, %")
    ax.set_ylim(20, 95)
    ax.set_title("Ширина MLP решает бюджет блока ($d=512$)", fontsize=13)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "param_split.png")
    print(f"params: ratio d=512 -> {ratios[2]:.3f}; share exp2={share[1]:.3f} "
          f"exp4={share[2]:.3f} exp8={share[3]:.3f}")


# ============================================================ fig 77.3 depth
def depth_curves(mode, L=24, d=64, n=16, seed=77):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0, 1, (n, d))
    blocks = [make_block(rng, d, res_scale=1.0 / np.sqrt(2 * L)) for _ in range(L)]
    x = x0.copy()
    norms, coss = [], []
    for p in blocks:
        x = run_stack(x, [p],
                      residual=(mode != "no-res"),
                      norm=("post" if mode == "post" else "pre"))
        norms.append(np.linalg.norm(x) / np.linalg.norm(x0))
        c = np.sum(x * x0, axis=1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(x0, axis=1))
        coss.append(float(c.mean()))
    return np.array(norms), np.array(coss)


def fig_depth():
    res_n, res_c = depth_curves("pre")
    nor_n, nor_c = depth_curves("no-res")
    pos_n, pos_c = depth_curves("post")
    FACTS.update(cos_res24=res_c[-1], cos_nores24=nor_c[-1], cos_post24=pos_c[-1],
                 norm_res24=res_n[-1], norm_nores24=nor_n[-1])
    assert res_c[-1] > 0.8 > nor_c[-1], (res_c[-1], nor_c[-1])
    assert abs(nor_c[-1]) < 0.1
    assert 0.9 < res_n[-1] < 1.4 and nor_n[-1] < 0.3, (res_n[-1], nor_n[-1])

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    ell = np.arange(1, len(res_n) + 1)
    ax = axes[0]
    ax.plot(ell, res_n, "o-", ms=3.4, color=BLUE, label="residual, pre-norm")
    ax.plot(ell, nor_n, "s-", ms=3.4, color=RED, label="без residual")
    ax.plot(ell, pos_n, "^-", ms=3.4, color=GREEN, label="residual, post-norm")
    ax.set_xlabel("номер блока $\\ell$")
    ax.set_ylabel("$\\|X^{(\\ell)}\\|_F\\,/\\,\\|X^{(0)}\\|_F$")
    ax.set_title("Норма представления по глубине", fontsize=13)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)

    ax = axes[1]
    ax.plot(ell, res_c, "o-", ms=3.4, color=BLUE, label="residual, pre-norm")
    ax.plot(ell, nor_c, "s-", ms=3.4, color=RED, label="без residual")
    ax.plot(ell, pos_c, "^-", ms=3.4, color=GREEN, label="residual, post-norm")
    ax.axhline(0, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    ax.annotate(f"{res_c[-1]:.2f}", (24, res_c[-1]), xytext=(-34, 10),
                textcoords="offset points", color=BLUE, fontsize=10.5)
    ax.annotate(f"{nor_c[-1]:.2f}", (24, nor_c[-1]), xytext=(-34, -16),
                textcoords="offset points", color=RED, fontsize=10.5)
    ax.set_ylim(-0.35, 1.05)
    ax.set_xlabel("номер блока $\\ell$")
    ax.set_ylabel("средний $\\cos(x^{(\\ell)}_i,\\;x^{(0)}_i)$")
    ax.set_title("Помнит ли позиция саму себя", fontsize=13)
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "residual_depth.png")
    print(f"depth: cos res={res_c[-1]:.3f} nores={nor_c[-1]:.3f} post={pos_c[-1]:.3f}; "
          f"norm res={res_n[-1]:.2f} nores={nor_n[-1]:.2f}")


# ============================================================ fig 77.4 routing
def sensitivity(use_attn, L=6, d=64, n=16, seed=177, eps=1e-4):
    """||d h_last / d x_first|| estimated by central differences on random directions."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0, 1, (n, d))
    blocks = [make_block(rng, d) for _ in range(L)]
    base = run_stack(x0, blocks, use_attn=use_attn, causal=True)[-1]
    vals = []
    for _ in range(8):
        u = rng.normal(0, 1, d)
        u /= np.linalg.norm(u)
        xp = x0.copy(); xp[0] += eps * u
        xm = x0.copy(); xm[0] -= eps * u
        hp = run_stack(xp, blocks, use_attn=use_attn, causal=True)[-1]
        hm = run_stack(xm, blocks, use_attn=use_attn, causal=True)[-1]
        vals.append(np.linalg.norm(hp - hm) / (2 * eps))
    return float(np.mean(vals)), float(np.linalg.norm(base))


# ---- real SMS probe ablation ------------------------------------------------
def sms_sequences(seq_len=16, vocab_size=400, d=128, seed=77):
    """Реальные SMS: первые seq_len слов сообщений длиной не меньше seq_len."""
    rows, labels = [], []
    with open(SMS, encoding="utf8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            toks = re.findall(r"[a-z0-9']+", parts[1].lower())
            if len(toks) >= seq_len:
                rows.append(toks[:seq_len])
                labels.append(1 if parts[0] == "spam" else 0)
    freq: dict[str, int] = {}
    for t in rows:
        for w in t:
            freq[w] = freq.get(w, 0) + 1
    top = [w for w, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:vocab_size]]
    idx = {w: i + 1 for i, w in enumerate(top)}
    ids = np.array([[idx.get(w, 0) for w in t] for t in rows])
    rng = np.random.default_rng(seed)
    emb = rng.normal(0, 1, (vocab_size + 1, d)) / np.sqrt(d)
    pos = rng.normal(0, 1, (seq_len, d)) / np.sqrt(d)
    return ids, np.array(labels), emb, pos


def probe_ablation():
    seq_len, d = 16, 128
    ids, spam, emb, pos = sms_sequences(seq_len=seq_len, d=d)
    n_msg = len(ids)
    rare = (ids == 0).mean(axis=1)          # доля редких (внесловарных) слов
    rng = np.random.default_rng(7700)
    blocks = [make_block(rng, d, heads=4) for _ in range(2)]
    perm = rng.permutation(n_msg)
    ntr = int(0.7 * n_msg)
    tr, te = perm[:ntr], perm[ntr:]

    variants = {
        "полный блок": (True, True, True),
        "без attention": (False, True, True),
        "без MLP": (True, False, True),
        "без positional": (True, True, False),
    }
    out = {}
    for name, (use_attn, use_mlp, with_pos) in variants.items():
        feats = np.zeros((n_msg, d))
        for i in range(n_msg):
            x = emb[ids[i]] + (pos if with_pos else 0.0)
            h = run_stack(x, blocks, use_attn=use_attn, use_mlp=use_mlp, causal=True)
            feats[i] = h[-1]
        mu, sd = feats[tr].mean(0), feats[tr].std(0) + 1e-9
        F = (feats - mu) / sd
        pred = Ridge(alpha=1.0).fit(F[tr], rare[tr]).predict(F[te])
        r2 = float(1 - np.sum((rare[te] - pred) ** 2) / np.sum((rare[te] - rare[te].mean()) ** 2))
        acc = float(LogisticRegression(max_iter=3000).fit(F[tr], spam[tr]).score(F[te], spam[te]))
        out[name] = (r2, acc)
        print(f"  probe {name}: R2(rare)={r2:.3f} spam acc={acc:.3f}")
    base_rate = float(max(spam.mean(), 1 - spam.mean()))
    return out, n_msg, len(tr), len(te), base_rate, float(rare.mean())


def fig_routing():
    depths = [1, 2, 4, 6, 8, 12]
    with_a = [sensitivity(True, L=L)[0] for L in depths]
    no_a = [sensitivity(False, L=L)[0] for L in depths]
    assert max(no_a) < 1e-9, no_a
    assert min(with_a) > 1e-3, with_a
    probe, n_msg, n_tr, n_te, base_rate, mean_frac = probe_ablation()
    full_r2 = probe["полный блок"][0]
    noattn_r2 = probe["без attention"][0]
    assert full_r2 > 0.75 and noattn_r2 < 0.2, (full_r2, noattn_r2)
    assert probe["полный блок"][1] > base_rate + 0.05
    assert abs(probe["без attention"][1] - base_rate) < 0.06
    FACTS.update(sens_L6=with_a[depths.index(6)], sens_noattn=max(no_a),
                 probe_full_r2=full_r2, probe_noattn_r2=noattn_r2,
                 probe_nomlp_r2=probe["без MLP"][0], probe_nopos_r2=probe["без positional"][0],
                 probe_full_acc=probe["полный блок"][1],
                 probe_nomlp_acc=probe["без MLP"][1],
                 probe_noattn_acc=probe["без attention"][1],
                 probe_nopos_acc=probe["без positional"][1],
                 sms_msgs=n_msg, sms_train=n_tr, sms_test=n_te,
                 sms_base_rate=base_rate, sms_mean_rare=mean_frac)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    ax = axes[0]
    ax.semilogy(depths, np.maximum(with_a, 1e-18), "o-", color=BLUE, lw=2.0,
                label="блок с attention")
    ax.semilogy(depths, np.maximum(no_a, 1e-18), "s-", color=RED, lw=2.0,
                label="только MLP (attention убран)")
    ax.set_ylim(1e-19, 10)
    ax.set_xlabel("число блоков $L$")
    ax.set_ylabel("$\\|\\partial h_{n}/\\partial x_{1}\\|$")
    ax.set_title("Кто вообще может увидеть первый токен", fontsize=13)
    ax.text(1.2, 1e-16, "машинный нуль: зависимости нет", fontsize=10, color=RED)
    ax.legend(frameon=False, fontsize=10, loc="center right")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)

    ax = axes[1]
    names = list(probe.keys())
    r2s = [probe[k][0] for k in names]
    accs = [probe[k][1] for k in names]
    ys = np.arange(len(names))[::-1]
    ax.barh(ys + 0.18, r2s, height=0.34, color=BLUE, label="$R^2$: доля редких слов")
    ax.barh(ys - 0.18, accs, height=0.34, color=GOLD, label="accuracy: спам или нет")
    ax.axvline(base_rate, color=MUTED, lw=1.0, ls=(0, (3, 2)))
    ax.text(base_rate - 0.02, ys[-1] - 0.62, f"частота\nбольшинства {base_rate:.2f}",
            fontsize=9, color=MUTED, ha="right")
    for y, v in zip(ys + 0.18, r2s):
        ax.text(max(v, 0) + 0.015, y - 0.06, f"{v:.2f}", fontsize=9.5, color=BLUE)
    for y, v in zip(ys - 0.18, accs):
        ax.text(v + 0.015, y - 0.06, f"{v:.2f}", fontsize=9.5, color=GOLD)
    ax.set_yticks(ys); ax.set_yticklabels(names)
    ax.set_xlim(-0.05, 1.42)
    ax.set_ylim(ys[-1] - 1.0, ys[0] + 0.7)
    ax.set_xlabel("качество линейного probe на тесте")
    ax.set_title("Что блок доносит до головы (реальные SMS)", fontsize=13)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.grid(True, axis="x", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "routing_ablation.png")
    print(f"routing: sens L=6 attn={with_a[depths.index(6)]:.4f}, no-attn max={max(no_a):.2e}; "
          f"msgs={n_msg} base={base_rate:.3f} mean rare={mean_frac:.4f}")


# ============================================================ fig 77.5 layernorm
def fig_layernorm():
    dig = load_digits()
    X = dig.data
    d = X.shape[1]
    a, b = 3.0, 7.0
    Xs = a * X + b
    L1, L2 = layer_norm(X), layer_norm(Xs)
    diff = float(np.abs(L1 - L2).max())
    assert diff < 1e-6, diff   # остаточный след eps=1e-5
    norms_before = np.linalg.norm(X, axis=1)
    norms_after = np.linalg.norm(L1, axis=1)
    assert abs(norms_after.mean() - np.sqrt(d)) < 0.02, norms_after.mean()
    spread = float(norms_before.max() / norms_before.min())
    FACTS.update(ln_diff=diff, ln_norm=float(norms_after.mean()), ln_sqrt_d=np.sqrt(d),
                 ln_spread=spread, ln_norm_min=float(norms_before.min()),
                 ln_norm_max=float(norms_before.max()),
                 ln_std_after=float(L1.std(axis=1).mean()))

    idx = 17
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.2))
    rows = [(X[idx], Xs[idx], "до LN"), (L1[idx], L2[idx], "после LN")]
    for r, (v1, v2, tag) in enumerate(rows):
        lo = min(v1.min(), v2.min()); hi = max(v1.max(), v2.max())
        for c, (v, ttl) in enumerate([(v1, "исходный $x$"), (v2, "$3x+7$")]):
            ax = axes[r][c]
            ax.imshow(v.reshape(8, 8), cmap="Greys", vmin=lo, vmax=hi)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{ttl}, {tag}", fontsize=11.5)
        ax = axes[r][2]
        ax.plot(v1, color=BLUE, lw=1.8, label="$x$")
        ax.plot(v2, color=RED, lw=1.4, ls=(0, (4, 3)), label="$3x+7$")
        ax.set_xlabel("канал $j$")
        ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9.5)
        ax.set_title("совпадают" if r else "разные кривые", fontsize=11.5)
    fig.suptitle("Layer normalization стирает сдвиг и масштаб позиции "
                 f"(реальные цифры, max |разность| = {diff:.1e})", fontsize=13.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, OUT / "layernorm_invariance.png")
    print(f"layernorm: max diff={diff:.2e}, mean norm after={norms_after.mean():.3f} "
          f"(sqrt d={np.sqrt(d):.3f}), spread before={spread:.1f}")


# ============================================================ fig 77.6 compute
def fig_compute():
    d = 512
    heads = 8
    ns = [64, 128, 256, 512, 1024, 2048, 4096]
    rng = np.random.default_rng(3)
    Wq, Wk, Wv, Wo = (rng.normal(0, 0.05, (d, d)).astype(np.float32) for _ in range(4))
    W1 = rng.normal(0, 0.05, (d, 4 * d)).astype(np.float32)
    W2 = rng.normal(0, 0.05, (4 * d, d)).astype(np.float32)

    def t_mha(x):
        p = {"Wq": Wq, "Wk": Wk, "Wv": Wv, "Wo": Wo, "heads": heads}
        return mha(x, p)

    def t_mlp(x):
        return gelu(x @ W1) @ W2

    tm, tf = [], []
    for n in ns:
        x = rng.normal(0, 1, (n, d)).astype(np.float32)
        for fn, store in ((t_mha, tm), (t_mlp, tf)):
            fn(x); fn(x)                                  # прогрев
            reps = 3 if n >= 2048 else 10
            t0 = time.perf_counter()
            for _ in range(reps):
                fn(x)
            store.append((time.perf_counter() - t0) / reps * 1000)
    tm, tf = np.array(tm), np.array(tf)
    ratio = tm / tf
    assert ratio[0] < 1.0, ratio
    assert ratio[-1] > 1.0, ratio
    k = int(np.argmax(ratio > 1.0))
    lo, hi = ns[k - 1], ns[k]
    lr = np.log(ratio)
    cross = float(np.exp(np.log(lo) + (np.log(hi) - np.log(lo)) * (-lr[k - 1]) / (lr[k] - lr[k - 1])))
    FACTS.update(cross_lo=lo, cross_hi=hi, cross_n=cross,
                 t_mha_64=tm[0], t_mlp_64=tf[0], t_mha_4096=tm[-1], t_mlp_4096=tf[-1],
                 ratio_64=ratio[0], ratio_4096=ratio[-1], theory_cross=4 * d)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    ax = axes[0]
    ax.loglog(ns, tm, "o-", color=BLUE, lw=2.0, label="MHA (замер)")
    ax.loglog(ns, tf, "s-", color=RED, lw=2.0, label="MLP $4d$ (замер)")
    ref = tf[0] * np.array(ns) / ns[0]
    ax.loglog(ns, ref, color=MUTED, lw=1.0, ls=(0, (3, 2)), label="наклон $\\propto n$")
    ax.loglog(ns, tm[0] * (np.array(ns) / ns[0]) ** 2, color=FAINT, lw=1.0,
              ls=(0, (1, 2)), label="наклон $\\propto n^{2}$")
    ax.set_xlabel("длина последовательности $n$"); ax.set_ylabel("время forward, мс")
    ax.set_title(f"Реальный замер, $d={d}$, float32", fontsize=13)
    ax.legend(frameon=False, fontsize=9.5)
    ax.grid(True, which="both", color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)

    ax = axes[1]
    ax.semilogx(ns, ratio, "o-", color=VIOLET, lw=2.2)
    ax.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (3, 2)))
    ax.axvline(cross, color=GOLD, lw=1.2)
    ax.text(cross * 1.08, 0.35, f"перелом ≈ {cross:.0f}", fontsize=10.5, color=GOLD)
    ax.set_xlabel("длина последовательности $n$")
    ax.set_ylabel("время MHA / время MLP")
    ax.set_title("Кто дороже — зависит от режима", fontsize=13)
    ax.grid(True, which="both", color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "compute_crossover.png")
    print(f"compute: ratio n=64 -> {ratio[0]:.2f}, n=4096 -> {ratio[-1]:.2f}; "
          f"crossover between {lo} and {hi}, interp {cross:.0f}")


# ============================================================ sidenote images
def side_paths():
    L = 24
    ks = np.arange(L + 1)
    from math import comb
    counts = np.array([comb(L, int(k)) for k in ks], dtype=float)
    total = counts.sum()
    assert total == 2.0 ** L
    mean_len = float((ks * counts).sum() / total)
    assert abs(mean_len - L / 2) < 1e-9
    frac_short = float(counts[ks <= 8].sum() / total)
    FACTS.update(paths_total=total, paths_mean=mean_len, paths_short=frac_short)

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.bar(ks, counts / total, color=BLUE, width=0.82)
    ax.axvline(mean_len, color=RED, lw=1.4)
    ax.text(mean_len + 0.6, 0.13, f"средняя\nглубина {mean_len:.0f}", fontsize=9, color=RED)
    ax.set_xlabel("сколько подслоёв прошёл путь")
    ax.set_ylabel("доля путей")
    ax.set_title(f"$2^{{{L}}}$ = {total / 1e6:.1f} млн путей", fontsize=11.5)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "paths.png")
    print(f"paths: total={total:.0f} mean={mean_len:.1f} share(<=8)={frac_short:.4f}")


def side_cache():
    n = 512
    d = 512
    t = np.arange(1, n + 1)
    no_cache = np.cumsum(4 * t ** 2 * d + 8 * t * d * d)     # пересчёт всего префикса
    cache = np.cumsum(4 * t * d + 8 * d * d)                 # только новый токен
    ratio = float(no_cache[-1] / cache[-1])
    FACTS.update(cache_ratio=ratio, cache_n=n)
    assert ratio > 100

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.loglog(t, no_cache / 1e9, color=RED, lw=2.0, label="без cache")
    ax.loglog(t, cache / 1e9, color=BLUE, lw=2.0, label="с KV-cache")
    ax.set_xlabel("сгенерировано токенов")
    ax.set_ylabel("накопленные GFLOP")
    ax.set_title(f"на {n} токенах разрыв ×{ratio:.0f}", fontsize=11.5)
    ax.legend(frameon=False, fontsize=9.5)
    ax.grid(True, which="both", color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "kv_cache.png")
    print(f"cache: ratio at n={n} -> {ratio:.1f}")


def side_gelu():
    x = np.linspace(-4, 4, 400)
    g = gelu(x)
    r = np.maximum(x, 0)
    gmin = float(g.min())
    xmin = float(x[np.argmin(g)])
    FACTS.update(gelu_min=gmin, gelu_argmin=xmin, gelu_at0=float(gelu(np.array([0.0]))[0]))
    assert -0.2 < gmin < -0.1

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(x, r, color=MUTED, lw=1.4, ls=(0, (4, 3)), label="ReLU")
    ax.plot(x, g, color=GREEN, lw=2.2, label="GELU")
    ax.scatter([xmin], [gmin], s=30, color=RED, zorder=5)
    ax.text(xmin + 0.2, gmin - 0.08, f"мин {gmin:.3f}\nпри x={xmin:.2f}", fontsize=9, color=RED)
    ax.axhline(0, color=LINE, lw=0.8); ax.axvline(0, color=LINE, lw=0.8)
    ax.set_title("Нелинейность внутри MLP", fontsize=11.5)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "gelu.png")
    print(f"gelu: min={gmin:.4f} at x={xmin:.3f}")


def side_shift():
    """Норма представления в pre-norm стеке: LN снимает масштаб, residual его копит."""
    d, n, L = 64, 16, 24
    rng = np.random.default_rng(9)
    x0 = rng.normal(0, 1, (n, d))
    blocks = [make_block(rng, d) for _ in range(L)]
    xs = [1.0]
    x = x0.copy()
    for p in blocks:
        x = run_stack(x, [p])
        xs.append(float(np.linalg.norm(x) / np.linalg.norm(x0)))
    growth = xs[-1]
    sqrt_law = np.sqrt(np.arange(len(xs)) * (growth ** 2 - 1) / L + 1)
    FACTS.update(growth24=growth)
    assert growth > 1.5

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(range(len(xs)), xs, "o-", ms=3.2, color=BLUE, label="замер")
    ax.plot(range(len(xs)), sqrt_law, color=MUTED, lw=1.2, ls=(0, (3, 2)),
            label="$\\sqrt{\\;\\cdot\\;}$-закон")
    ax.set_xlabel("блок"); ax.set_ylabel("норма / начальная")
    ax.set_title(f"поток растёт до ×{growth:.2f}", fontsize=11.5)
    ax.legend(frameon=False, fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "stream_growth.png")
    print(f"stream growth after 24 blocks: {growth:.3f}")


def main():
    fig_anatomy()
    fig_params()
    fig_depth()
    fig_routing()
    fig_layernorm()
    fig_compute()
    side_paths()
    side_cache()
    side_gelu()
    side_shift()
    print("\n--- FACTS ---")
    for k, v in FACTS.items():
        print(f"{k} = {v}")


if __name__ == "__main__":
    main()
