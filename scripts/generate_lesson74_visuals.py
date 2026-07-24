"""Deterministic figures for lesson 74: gated memory (LSTM and GRU).

Everything quoted in the lesson is computed here and asserted:
  * the arithmetic of one LSTM coordinate and the half-life of a constant forget gate;
  * REAL training (numpy, hand-written BPTT) of a tanh-RNN, a GRU and an LSTM with matched
    parameter counts on the delayed-copy task, accuracy versus delay;
  * measured median |dL/dh_t| by lag for RNN and LSTM (the gradient-reach figure);
  * the gate traces of the trained LSTM (real gates, not a drawing);
  * REAL bike-sharing hourly data: naive, seasonal-naive, ridge and LSTM next-hour MAE.

Run: python3 scripts/generate_lesson74_visuals.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BIKE = ROOT / "scripts" / "data" / "bike-sharing-hour.csv"
OUT = ROOT / "public" / "figures" / "lessons" / "74"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "74"
FACTS = ROOT / "scripts" / "data" / "lesson74_facts.json"

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

FACT = {}


def save(fig, path, *, dpi=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


# ============================================================ recurrent cells (numpy, BPTT)
def init_params(kind, d_in, n, d_out, rng, forget_bias=3.0, update_bias=-2.0):
    def xav(a, b):
        return rng.normal(0, np.sqrt(2.0 / (a + b)), (a, b))

    p = {}
    if kind == "rnn":
        p["Wx"] = xav(d_in, n); p["Wh"] = xav(n, n); p["b"] = np.zeros(n)
    elif kind == "lstm":
        p["Wx"] = xav(d_in, 4 * n); p["Wh"] = xav(n, 4 * n); p["b"] = np.zeros(4 * n)
        p["b"][n:2 * n] = forget_bias          # forget-gate bias prior
    elif kind == "gru":
        p["Wx"] = xav(d_in, 3 * n); p["Wh"] = xav(n, 3 * n); p["b"] = np.zeros(3 * n)
        p["b"][:n] = update_bias               # update-gate bias: keep the state by default
    p["Wy"] = xav(n, d_out); p["by"] = np.zeros(d_out)
    return p


def n_params(kind, d_in, n, d_out):
    g = {"rnn": 1, "gru": 3, "lstm": 4}[kind]
    return g * (d_in * n + n * n + n) + n * d_out + d_out


def run_cell(kind, p, X, n):
    """X: (T, B, d_in). Returns cache with all per-step quantities."""
    T, B, _ = X.shape
    h = np.zeros((B, n)); c = np.zeros((B, n))
    hs = [h]; cs = [c]; gates = []
    for t in range(T):
        x = X[t]
        if kind == "rnn":
            z = x @ p["Wx"] + h @ p["Wh"] + p["b"]
            h = np.tanh(z)
            gates.append((h,))
        elif kind == "lstm":
            z = x @ p["Wx"] + h @ p["Wh"] + p["b"]
            i = sig(z[:, :n]); f = sig(z[:, n:2 * n]); o = sig(z[:, 2 * n:3 * n])
            g = np.tanh(z[:, 3 * n:])
            c = f * c + i * g
            tc = np.tanh(c)
            h = o * tc
            gates.append((i, f, o, g, tc))
        elif kind == "gru":
            a = x @ p["Wx"][:, :2 * n] + h @ p["Wh"][:, :2 * n] + p["b"][:2 * n]
            zg = sig(a[:, :n]); r = sig(a[:, n:2 * n])
            rh = r * h
            nb = x @ p["Wx"][:, 2 * n:] + rh @ p["Wh"][:, 2 * n:] + p["b"][2 * n:]
            nn = np.tanh(nb)
            h_prev = h
            h = (1 - zg) * h_prev + zg * nn
            gates.append((zg, r, nn, rh))
        hs.append(h); cs.append(c)
    return {"hs": hs, "cs": cs, "gates": gates, "X": X}


def backward(kind, p, cache, dh_last, n):
    X = cache["X"]; T = X.shape[0]
    hs, cs, gates = cache["hs"], cache["cs"], cache["gates"]
    grads = {k: np.zeros_like(v) for k, v in p.items()}
    dh = dh_last.copy()
    dc = np.zeros_like(hs[0])
    reach = np.zeros(T)                     # mean ||dL/dh_t|| per step
    reach_c = np.zeros(T)                   # mean ||dL/dc_t|| per step (LSTM memory path)
    for t in range(T - 1, -1, -1):
        reach[t] = np.sqrt((dh ** 2).sum(axis=1)).mean()
        x = X[t]; h_prev = hs[t]
        if kind == "rnn":
            (h,) = gates[t]
            dz = dh * (1 - h ** 2)
            grads["Wx"] += x.T @ dz; grads["Wh"] += h_prev.T @ dz; grads["b"] += dz.sum(0)
            dh = dz @ p["Wh"].T
        elif kind == "lstm":
            i, f, o, g, tc = gates[t]
            c_prev = cs[t]; c = cs[t + 1]
            dc = dc + dh * o * (1 - tc ** 2)
            reach_c[t] = np.sqrt((dc ** 2).sum(axis=1)).mean()
            do = dh * tc; di = dc * g; df = dc * c_prev; dg = dc * i
            dz = np.concatenate([di * i * (1 - i), df * f * (1 - f),
                                 do * o * (1 - o), dg * (1 - g ** 2)], axis=1)
            grads["Wx"] += x.T @ dz; grads["Wh"] += h_prev.T @ dz; grads["b"] += dz.sum(0)
            dh = dz @ p["Wh"].T
            dc = dc * f
        elif kind == "gru":
            zg, r, nn, rh = gates[t]
            dz_ = dh * (nn - h_prev)
            dn = dh * zg
            dh_prev = dh * (1 - zg)
            dnb = dn * (1 - nn ** 2)
            grads["Wx"][:, 2 * n:] += x.T @ dnb
            grads["Wh"][:, 2 * n:] += rh.T @ dnb
            grads["b"][2 * n:] += dnb.sum(0)
            drh = dnb @ p["Wh"][:, 2 * n:].T
            dr = drh * h_prev
            dh_prev = dh_prev + drh * r
            da = np.concatenate([dz_ * zg * (1 - zg), dr * r * (1 - r)], axis=1)
            grads["Wx"][:, :2 * n] += x.T @ da
            grads["Wh"][:, :2 * n] += h_prev.T @ da
            grads["b"][:2 * n] += da.sum(0)
            dh = dh_prev + da @ p["Wh"][:, :2 * n].T
    if reach_c.max() == 0:
        reach_c = reach
    return grads, reach, reach_c


def adam_step(p, g, m, v, t, lr=3e-3, clip=1.0):
    total = np.sqrt(sum((gg ** 2).sum() for gg in g.values()))
    scale = min(1.0, clip / (total + 1e-12))
    for k in p:
        gk = g[k] * scale
        m[k] = 0.9 * m[k] + 0.1 * gk
        v[k] = 0.999 * v[k] + 0.001 * gk ** 2
        mh = m[k] / (1 - 0.9 ** t); vh = v[k] / (1 - 0.999 ** t)
        p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)


# ============================================================ delayed-copy task
VOCAB = 6   # 0,1 -> the bit to remember; 2,3,4 -> irrelevant chatter; 5 -> query


def copy_batch(B, D, rng):
    """Delayed copy with distractors: bit, then D random irrelevant symbols, then query."""
    T = D + 2
    bits = rng.integers(0, 2, B)
    X = np.zeros((T, B, VOCAB))
    X[0, np.arange(B), bits] = 1.0
    noise = rng.integers(2, 5, (D, B))
    for t in range(D):
        X[1 + t, np.arange(B), noise[t]] = 1.0
    X[T - 1, :, VOCAB - 1] = 1.0
    return X, bits


def train_copy(kind, n, D, steps=1000, B=96, seed=0, lr=2e-3, forget_bias=3.0):
    rng = np.random.default_rng(seed)
    p = init_params(kind, VOCAB, n, 2, rng, forget_bias=forget_bias)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(x) for k, x in p.items()}
    reach_acc = None
    for s in range(1, steps + 1):
        X, y = copy_batch(B, D, rng)
        cache = run_cell(kind, p, X, n)
        hT = cache["hs"][-1]
        logits = hT @ p["Wy"] + p["by"]
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits); prob = e / e.sum(axis=1, keepdims=True)
        dlog = prob.copy(); dlog[np.arange(B), y] -= 1.0; dlog /= B
        gW = hT.T @ dlog; gb = dlog.sum(0)
        dh_last = dlog @ p["Wy"].T
        grads, reach, _ = backward(kind, p, cache, dh_last, n)
        grads["Wy"] += gW; grads["by"] += gb
        if s == 1:
            reach_acc = reach.copy()
        adam_step(p, grads, m, v, s, lr=lr)
    # test accuracy on fresh data
    rng_t = np.random.default_rng(seed + 9000)
    Xt, yt = copy_batch(2000, D, rng_t)
    ct = run_cell(kind, p, Xt, n)
    pred = (ct["hs"][-1] @ p["Wy"] + p["by"]).argmax(axis=1)
    acc = float((pred == yt).mean())
    return acc, p, reach_acc, ct


# ============================================================ fig 74.1 — one cell, arithmetic
def fig_cell():
    c_prev, f, i, g, o = 2.0, 0.8, 0.25, -0.4, 0.6
    c = f * c_prev + i * g
    h = o * np.tanh(c)
    assert abs(c - 1.5) < 0.05
    assert abs(h - 0.5430890) < 5e-08, h
    c2 = f * c_prev
    h2 = o * np.tanh(c2)
    FACT["cell_c"] = round(c, 4); FACT["cell_h"] = round(h, 4)
    FACT["cell_c_noinput"] = round(c2, 4); FACT["cell_h_noinput"] = round(h2, 4)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    # memory highway
    ax.annotate("", xy=(9.4, 4.6), xytext=(0.6, 4.6),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=3))
    ax.text(0.6, 5.1, "$c_{t-1}$", color=BLUE, fontsize=14)
    ax.text(9.0, 5.1, "$c_t$", color=BLUE, fontsize=14)
    ax.text(5.0, 5.62, "магистраль памяти: только × и +", color=BLUE, ha="center", fontsize=11)
    for x0, lab, col in [(2.6, "×  $f_t$", RED), (5.2, "+  $i_t\\widetilde c_t$", GREEN)]:
        ax.add_patch(plt.Circle((x0, 4.6), 0.42, facecolor=PAPER, edgecolor=col, lw=2, zorder=5))
        ax.text(x0, 5.12, lab, color=col, ha="center", fontsize=12)
    # gates row
    ax.add_patch(plt.Rectangle((0.6, 1.1), 8.8, 1.7, facecolor=WASH, edgecolor=LINE))
    ax.text(1.0, 2.42, "$[x_t,h_{t-1}]$ и четыре линейных слоя", color=MUTED, fontsize=11)
    for x0, lab, col in [(1.6, "$f_t=\\sigma(\\cdot)$", RED), (3.9, "$i_t=\\sigma(\\cdot)$", GREEN),
                         (6.1, "$\\widetilde c_t=\\tanh(\\cdot)$", GOLD), (8.3, "$o_t=\\sigma(\\cdot)$", VIOLET)]:
        ax.text(x0, 1.55, lab, color=col, ha="center", fontsize=12)
    ax.annotate("", xy=(2.6, 4.1), xytext=(1.6, 2.85), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4))
    ax.annotate("", xy=(5.2, 4.1), xytext=(4.6, 2.85), arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.4))
    ax.annotate("", xy=(8.2, 4.28), xytext=(8.7, 3.35), arrowprops=dict(arrowstyle="-|>", color=VIOLET, lw=1.4))
    ax.text(8.6, 3.02, "$h_t=o_t\\tanh(c_t)$", color=VIOLET, fontsize=12, ha="right")
    ax.set_title("Аддитивный путь состояния и четыре ворот", loc="left")

    # numeric panel
    ax2.axis("off")
    ax2.set_title("Одна координата, числа", loc="left")
    rows = [
        ("$c_{t-1}$", "2,0000", INK),
        ("$f_t\\,c_{t-1}$", f"{f * c_prev:.4f}".replace(".", ","), RED),
        ("$i_t\\,\\widetilde c_t$", f"{i * g:.4f}".replace(".", ","), GREEN),
        ("$c_t$", f"{c:.4f}".replace(".", ","), BLUE),
        ("$\\tanh c_t$", f"{np.tanh(c):.4f}".replace(".", ","), MUTED),
        ("$h_t=o_t\\tanh c_t$", f"{h:.4f}".replace(".", ","), VIOLET),
        ("$h_t$ при $i_t=0$", f"{h2:.4f}".replace(".", ","), MUTED),
    ]
    for k, (lab, val, col) in enumerate(rows):
        y = 0.92 - 0.13 * k
        ax2.text(0.02, y, lab, color=col, fontsize=13, transform=ax2.transAxes)
        ax2.text(0.92, y, val, color=col, fontsize=13, ha="right", transform=ax2.transAxes)
        ax2.plot([0.02, 0.92], [y - 0.035, y - 0.035], color=GRID, lw=0.7, transform=ax2.transAxes)
    ax2.text(0.02, -0.04, "$f_t=0{,}8$, $i_t=0{,}25$, $\\widetilde c_t=-0{,}4$, $o_t=0{,}6$",
             color=MUTED, fontsize=11, transform=ax2.transAxes)
    save(fig, OUT / "cell.png")
    print(f"cell: c_t={c:.4f} h_t={h:.4f} h_noinput={h2:.4f}")


# ============================================================ fig 74.2 — regimes of the gates
def fig_regimes():
    T = 80
    modes = [
        ("хранить: $f=1$, $i=0$", 1.0, 0.0, BLUE),
        ("накапливать: $f=1$, $i=0{,}2$", 1.0, 0.2, GREEN),
        ("забывать: $f=0{,}9$, $i=0$", 0.9, 0.0, GOLD),
        ("заменить: $f=0$, $i=1$", 0.0, 1.0, RED),
    ]
    cand = 0.2
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    finals = {}
    for lab, f, i, col in modes:
        c = 1.0; traj = [c]
        for t in range(T):
            c = f * c + i * cand
            traj.append(c)
        finals[lab.split(":")[0]] = traj[-1]
        ax.plot(range(T + 1), traj, color=col, lw=2.3, label=lab)
    acc_final = round(finals["накапливать"], 4)
    assert abs(acc_final - (1.0 + 80 * 0.2 * cand)) < 1e-9, acc_final
    assert abs(finals["забывать"] - 0.9 ** 80) < 1e-9
    FACT["regime_accumulate_c80"] = acc_final
    FACT["regime_forget_c80"] = round(0.9 ** 80, 6)
    hbound = float(np.tanh(acc_final))
    FACT["regime_accumulate_tanh"] = round(hbound, 6)
    assert hbound > 0.999
    ax.axhline(1.0, color=GRID, lw=1)
    ax.set_xlabel("шаг $t$"); ax.set_ylabel("состояние памяти $c_t$")
    ax.set_title("Четыре режима ворот из одной формулы $c_t=f c_{t-1}+i\\widetilde c_t$")
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax.text(41, 3.2, f"через 80 шагов $c=$ {str(acc_final).replace('.', ',')}", color=GREEN, fontsize=11)
    ax.text(41, 0.35, f"$0{{,}}9^{{80}}=${np.round(0.9 ** 80, 5)}".replace(".", ","), color=GOLD, fontsize=11)
    save(fig, OUT / "regimes.png")
    print(f"regimes: accumulate c80={acc_final}, forget c80={0.9**80:.6f}, tanh={hbound:.6f}")


# ============================================================ fig 74.3 — copy task, real training
DELAYS = [25, 50, 100, 200]
SEEDS = [1, 2, 3]
CFG = [("rnn", 64, "простая RNN, $n=64$", RED),
       ("gru", 36, "GRU, $n=36$", GREEN),
       ("lstm", 31, "LSTM, $n=31$", BLUE)]


def fig_copy():
    params = {k: n_params(k, VOCAB, n, 2) for k, n, _, _ in CFG}
    spread = (max(params.values()) - min(params.values())) / min(params.values())
    assert spread < 0.05, (params, spread)
    FACT["params_matched"] = params
    FACT["params_spread_pct"] = round(100 * spread, 2)

    res = {}
    for kind, n, lab, col in CFG:
        rows = []
        for D in DELAYS:
            accs = [train_copy(kind, n, D, seed=sd)[0] for sd in SEEDS]
            rows.append(accs)
            print(f"copy {kind} n={n} D={D}: {[round(a, 3) for a in accs]} "
                  f"mean={np.mean(accs):.3f}", flush=True)
        res[kind] = np.array(rows)
    FACT["copy_delays"] = DELAYS
    FACT["copy_seeds"] = SEEDS
    FACT["copy_acc_mean"] = {k: [round(float(x), 3) for x in v.mean(axis=1)] for k, v in res.items()}
    FACT["copy_acc_all"] = {k: [[round(float(a), 3) for a in row] for row in v] for k, v in res.items()}
    m = FACT["copy_acc_mean"]
    assert m["lstm"][0] > 0.99 and m["gru"][0] > 0.99 and m["rnn"][0] > 0.99
    assert m["rnn"][1] < 0.95 and m["lstm"][1] > 0.99
    assert m["rnn"][2] < 0.95 and m["lstm"][2] > 0.99 and m["gru"][2] > 0.99
    assert m["rnn"][3] < 0.6 and m["gru"][3] < 0.6 and m["lstm"][3] < 0.9

    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    for kind, n, lab, col in CFG:
        ax.plot(DELAYS, res[kind].mean(axis=1), "o-", color=col, lw=2.4, ms=7,
                label=f"{lab}, {params[kind]} параметров")
        for j, D in enumerate(DELAYS):
            ax.plot([D] * len(SEEDS), res[kind][j], "o", color=col, ms=3.5, alpha=0.45)
    ax.axhline(0.5, color=MUTED, lw=1.2, ls=(0, (5, 3)))
    ax.text(DELAYS[0], 0.515, "уровень угадывания", color=MUTED, fontsize=11)
    ax.set_ylim(0.42, 1.04)
    ax.set_xlabel("длина паузы $D$: столько посторонних символов между битом и запросом")
    ax.set_ylabel("доля верных ответов на тесте")
    ax.set_title("«Запомни первый бит»: одинаковый бюджет параметров, разная память")
    ax.legend(frameon=False, fontsize=10.5, loc="lower left")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "copy.png")


# ============================================================ forget-gate bias check (real)
def check_forget_bias():
    """Same LSTM, same lr, only the forget-gate bias differs. D = 100."""
    res = {}
    for b in (1.0, 3.0):
        accs = [train_copy("lstm", 31, 100, seed=sd, forget_bias=b)[0] for sd in SEEDS]
        res[b] = [round(float(a), 3) for a in accs]
        print(f"forget bias {b}: {res[b]} mean={np.mean(accs):.3f}", flush=True)
    FACT["forget_bias_acc"] = {str(k): v for k, v in res.items()}
    FACT["forget_bias_solved"] = {str(k): int(sum(a > 0.95 for a in v)) for k, v in res.items()}
    assert FACT["forget_bias_solved"]["1.0"] == 0
    assert FACT["forget_bias_solved"]["3.0"] == len(SEEDS)


# ============================================================ fig 74.4 — gradient reach (real)
def _reach(kind, p, n, D, B=256, seed=4242):
    rng = np.random.default_rng(seed)
    X, y = copy_batch(B, D, rng)
    cache = run_cell(kind, p, X, n)
    hT = cache["hs"][-1]
    logits = hT @ p["Wy"] + p["by"]
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits); prob = e / e.sum(axis=1, keepdims=True)
    dlog = prob.copy(); dlog[np.arange(B), y] -= 1.0; dlog /= B
    acc = float((logits.argmax(axis=1) == y).mean())
    _, rh, rc = backward(kind, p, cache, dlog @ p["Wy"].T, n)
    return rh, rc, acc


def fig_gradient():
    D = 100
    rng = np.random.default_rng(7401)
    p_rnn = init_params("rnn", VOCAB, 64, 2, rng)
    p_lstm = init_params("lstm", VOCAB, 31, 2, rng)
    rh_r, _, _ = _reach("rnn", p_rnn, 64, D)
    _, rc_l, _ = _reach("lstm", p_lstm, 31, D)
    keep_r = float(rh_r[0] / rh_r[-1]); keep_l = float(rc_l[0] / rc_l[-1])
    FACT["grad_D"] = D
    FACT["grad_init_keep_rnn"] = float(f"{keep_r:.3g}")
    FACT["grad_init_keep_lstm"] = float(f"{keep_l:.3g}")
    FACT["grad_init_drop_rnn"] = float(f"{1 / keep_r:.3g}")
    FACT["grad_init_drop_lstm"] = float(f"{1 / keep_l:.3g}")
    FACT["grad_init_advantage"] = float(f"{keep_l / keep_r:.3g}")
    print(f"gradient@init: rnn keep={keep_r:.3e}, lstm keep={keep_l:.3e}, "
          f"advantage={keep_l / keep_r:.3e}", flush=True)
    assert keep_l > keep_r * 1e3, (keep_r, keep_l)

    # after training: the RNN seed that did solve D=100 has learned a near-identity path
    acc_tr, p_tr, _, _ = train_copy("rnn", 64, D, seed=1)
    rh_t, _, _ = _reach("rnn", p_tr, 64, D)
    keep_tr = float(rh_t[0] / rh_t[-1])
    FACT["grad_trained_rnn_acc"] = round(acc_tr, 3)
    FACT["grad_trained_rnn_keep"] = float(f"{keep_tr:.3g}")
    print(f"gradient trained rnn: acc={acc_tr:.3f}, keep={keep_tr:.3f}", flush=True)
    assert keep_tr > 100 * keep_r

    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    lags = np.arange(D + 2)[::-1]
    ax.semilogy(lags, np.maximum(rh_r, 1e-30), color=RED, lw=2.4,
                label=r"RNN до обучения: $\|\partial\mathcal{L}/\partial h_t\|$")
    ax.semilogy(lags, np.maximum(rc_l, 1e-30), color=BLUE, lw=2.4,
                label=r"LSTM до обучения: $\|\partial\mathcal{L}/\partial c_t\|$")
    ax.semilogy(lags, np.maximum(rh_t, 1e-30), color=GOLD, lw=2.0, ls=(0, (5, 3)),
                label="RNN после обучения (нашла почти тождественный путь)")
    ax.set_xlabel("лаг: сколько шагов назад от запроса")
    ax.set_ylabel("норма градиента по состоянию")
    ax.set_title("Куда доходит сигнал обучения при $D=100$")
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=10.5, loc="lower left")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5, which="both"); ax.set_axisbelow(True)
    save(fig, OUT / "gradient.png")


# ============================================================ fig 74.5 — trained gates in time
def fig_gates():
    D = 30
    acc, p, _, _ = train_copy("lstm", 31, D, seed=5)
    print(f"gates: trained LSTM acc={acc:.3f}")
    assert acc > 0.9, acc
    FACT["gates_acc"] = round(acc, 3)
    rng = np.random.default_rng(11)
    X, y = copy_batch(512, D, rng)
    n = 31
    cache = run_cell("lstm", p, X, n)
    I = np.array([g[0].mean() for g in cache["gates"]])
    F = np.array([g[1].mean() for g in cache["gates"]])
    O = np.array([g[2].mean() for g in cache["gates"]])
    # coordinate that best separates the two bits at the query step
    hT = cache["hs"][-1]
    sep = np.abs(hT[y == 1].mean(0) - hT[y == 0].mean(0))
    j = int(sep.argmax())
    cj = np.array([cs[:, j] for cs in cache["cs"][1:]])
    c1 = cj[:, y == 1].mean(1); c0 = cj[:, y == 0].mean(1)
    gap = np.abs(c1 - c0)
    f_pause = float(F[1:D].mean())
    FACT["gates_D"] = D
    FACT["gate_coord"] = j
    FACT["gate_f_pause"] = round(f_pause, 3)
    FACT["gate_f_halflife"] = round(float(np.log(0.5) / np.log(f_pause)), 1)
    FACT["gate_i_first"] = round(float(I[0]), 3)
    FACT["gate_i_pause"] = round(float(I[1:D].mean()), 3)
    FACT["gate_o_pause"] = round(float(O[1:D].mean()), 3)
    FACT["gate_o_query"] = round(float(O[-1]), 3)
    FACT["cell_gap_t1"] = round(float(gap[1]), 3)
    FACT["cell_gap_end"] = round(float(gap[D]), 3)
    FACT["cell_gap_growth"] = round(float(gap[D] / gap[1]), 1)
    print("gates:", {k: FACT[k] for k in
                     ["gates_acc", "gate_f_pause", "gate_f_halflife", "gate_i_first",
                      "gate_i_pause", "gate_o_pause", "gate_o_query",
                      "cell_gap_t1", "cell_gap_end", "cell_gap_growth"]})
    assert f_pause > 0.8
    assert FACT["cell_gap_end"] > FACT["cell_gap_t1"]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.8, 6.6), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1]})
    t = np.arange(len(F))
    a1.plot(t, F, color=RED, lw=2.2, label="$\\overline{f_t}$ — забывание")
    a1.plot(t, I, color=GREEN, lw=2.2, label="$\\overline{i_t}$ — запись")
    a1.plot(t, O, color=VIOLET, lw=2.2, label="$\\overline{o_t}$ — выдача")
    a1.set_ylim(0, 1.05); a1.set_ylabel("среднее по координатам")
    a1.set_title(f"Обученная LSTM, задача $D={D}$: ворота размечают события (точность {acc:.2f})".replace(".", ","))
    a1.legend(frameon=False, fontsize=10.5, ncol=3, loc="lower center")
    a2.plot(t, c1, color=BLUE, lw=2.2, label="бит = 1")
    a2.plot(t, c0, color=GOLD, lw=2.2, label="бит = 0")
    a2.fill_between(t, c0, c1, color=BLUE, alpha=0.08)
    a2.set_ylabel(f"память $c_t$, координата {j}")
    a2.set_xlabel("шаг $t$: 0 — символ, 1..%d — пауза, %d — запрос" % (D, D + 1))
    a2.legend(frameon=False, fontsize=10.5, loc="upper left")
    for ax in (a1, a2):
        ax.axvline(0, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.axvline(D + 1, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "gates.png")


# ============================================================ real data: bike sharing
def load_bike():
    cnt, hr = [], []
    with open(BIKE) as f:
        for row in csv.DictReader(f):
            cnt.append(float(row["cnt"])); hr.append(int(row["hr"]))
    return np.array(cnt), np.array(hr)


def fig_bike():
    cnt, hr = load_bike()
    FACT["bike_rows"] = int(len(cnt))
    assert len(cnt) == 17379, len(cnt)
    L = 24
    T_all = len(cnt)
    idx = np.arange(L, T_all)
    split = int(0.7 * len(idx))
    tr, te = idx[:split], idx[split:]
    mu, sd = cnt[:tr[-1] + 1].mean(), cnt[:tr[-1] + 1].std()
    FACT["bike_train_mean"] = round(float(mu), 1)
    FACT["bike_train_std"] = round(float(sd), 1)

    def window(ids):
        X = np.stack([(cnt[i - L:i] - mu) / sd for i in ids])          # (N, L)
        hsin = np.sin(2 * np.pi * hr[ids] / 24); hcos = np.cos(2 * np.pi * hr[ids] / 24)
        y = cnt[ids]
        return X, hsin, hcos, y

    Xtr, str_, ctr, ytr = window(tr)
    Xte, ste, cte, yte = window(te)
    naive = cnt[te - 1]
    seasonal = cnt[te - 24]
    mae_naive = float(np.abs(naive - yte).mean())
    mae_seasonal = float(np.abs(seasonal - yte).mean())

    # ridge on the same window + calendar pair
    Ftr = np.column_stack([Xtr, str_, ctr, np.ones(len(ytr))])
    Fte = np.column_stack([Xte, ste, cte, np.ones(len(yte))])
    A = Ftr.T @ Ftr + 1.0 * np.eye(Ftr.shape[1])
    w = np.linalg.solve(A, Ftr.T @ ((ytr - mu) / sd))
    mae_ridge = float(np.abs((Fte @ w) * sd + mu - yte).mean())

    # LSTM over the 24-hour window, input = (value, sin h, cos h)
    n = 24
    rng = np.random.default_rng(74)
    p = init_params("lstm", 3, n, 1, rng)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(x) for k, x in p.items()}
    hs_tr = np.sin(2 * np.pi * ((hr[tr][:, None] - np.arange(L, 0, -1)[None, :]) % 24) / 24)
    hc_tr = np.cos(2 * np.pi * ((hr[tr][:, None] - np.arange(L, 0, -1)[None, :]) % 24) / 24)
    hs_te = np.sin(2 * np.pi * ((hr[te][:, None] - np.arange(L, 0, -1)[None, :]) % 24) / 24)
    hc_te = np.cos(2 * np.pi * ((hr[te][:, None] - np.arange(L, 0, -1)[None, :]) % 24) / 24)

    def seq(X, hs, hc, ids):
        return np.stack([X, hs, hc], axis=2).transpose(1, 0, 2)        # (L, N, 3)

    ytr_s = (ytr - mu) / sd
    B = 256
    for s in range(1, 601):
        b = rng.integers(0, len(ytr), B)
        Xb = seq(Xtr[b], hs_tr[b], hc_tr[b], None)
        cache = run_cell("lstm", p, Xb, n)
        hT = cache["hs"][-1]
        pred = (hT @ p["Wy"] + p["by"])[:, 0]
        diff = pred - ytr_s[b]
        loss = float((diff ** 2).mean())
        d = (2 * diff / B)[:, None]
        grads, _, _ = backward("lstm", p, cache, d @ p["Wy"].T, n)
        grads["Wy"] += hT.T @ d; grads["by"] += d.sum(0)
        adam_step(p, grads, m, v, s, lr=5e-3)
        if s % 200 == 0:
            print(f"bike lstm step {s}: train mse={loss:.4f}")
    Xb = seq(Xte, hs_te, hc_te, None)
    cache = run_cell("lstm", p, Xb, n)
    pred = ((cache["hs"][-1] @ p["Wy"] + p["by"])[:, 0]) * sd + mu
    mae_lstm = float(np.abs(pred - yte).mean())

    print(f"bike MAE: naive={mae_naive:.1f} seasonal24={mae_seasonal:.1f} "
          f"ridge={mae_ridge:.1f} lstm={mae_lstm:.1f}")
    FACT["bike_mae_naive"] = round(mae_naive, 1)
    FACT["bike_mae_seasonal"] = round(mae_seasonal, 1)
    FACT["bike_mae_ridge"] = round(mae_ridge, 1)
    FACT["bike_mae_lstm"] = round(mae_lstm, 1)
    FACT["bike_test_n"] = int(len(yte))
    FACT["bike_gain_pct"] = round(100 * (mae_seasonal - mae_lstm) / mae_seasonal, 1)
    assert mae_lstm < mae_naive and mae_lstm < mae_seasonal
    assert mae_lstm < mae_ridge * 1.05

    # peak hours (top decile of train target)
    thr = np.quantile(ytr, 0.9)
    FACT["bike_peak_threshold"] = round(float(thr), 1)
    peak = yte > thr
    FACT["bike_peak_share_pct"] = round(100 * float(peak.mean()), 1)
    FACT["bike_mae_lstm_peak"] = round(float(np.abs(pred - yte)[peak].mean()), 1)
    FACT["bike_mae_seasonal_peak"] = round(float(np.abs(seasonal - yte)[peak].mean()), 1)
    assert FACT["bike_mae_lstm_peak"] > FACT["bike_mae_lstm"] * 1.5

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.0, 6.8),
                                 gridspec_kw={"height_ratios": [1.5, 1]})
    k0, k1 = 300, 300 + 24 * 7
    tt = np.arange(k1 - k0)
    a1.plot(tt, yte[k0:k1], color=INK, lw=2.0, label="реальный прокат")
    a1.plot(tt, pred[k0:k1], color=BLUE, lw=2.0, label=f"LSTM (MAE {mae_lstm:.1f})".replace(".", ","))
    a1.plot(tt, seasonal[k0:k1], color=GOLD, lw=1.6, ls=(0, (5, 3)),
            label=f"«как сутки назад» (MAE {mae_seasonal:.1f})".replace(".", ","))
    a1.set_ylabel("поездок в час"); a1.set_xlabel("часы одной недели тестового периода")
    a1.set_title("Реальный велопрокат: прогноз на час вперёд")
    a1.legend(frameon=False, fontsize=10.5, ncol=3, loc="upper left")
    a1.grid(True, color=GRID, lw=0.4, alpha=0.5); a1.set_axisbelow(True)

    names = ["как час назад", "как сутки назад", "ridge по 24 часам", "LSTM"]
    vals = [mae_naive, mae_seasonal, mae_ridge, mae_lstm]
    cols = [MUTED, GOLD, GREEN, BLUE]
    a2.barh(names, vals, color=cols, height=0.55)
    for k, vv in enumerate(vals):
        a2.text(vv + 1.2, k, f"{vv:.1f}".replace(".", ","), va="center", color=cols[k], fontsize=11)
    a2.set_xlabel(f"MAE на тесте, поездок в час ({len(yte)} часов)")
    a2.set_xlim(0, max(vals) * 1.18)
    a2.grid(True, axis="x", color=GRID, lw=0.4, alpha=0.5); a2.set_axisbelow(True)
    save(fig, OUT / "bike.png")


# ============================================================ fig 74.6 — masking / padding
def fig_mask():
    """A padded batch: naive zero-padding drags the state, masking freezes it."""
    n = 1
    T = 20; real_len = 8
    f, i = 0.85, 0.6
    c_mask = []; c_nomask = []
    c1 = 1.0; c2 = 1.0
    rng = np.random.default_rng(3)
    cand = np.concatenate([rng.normal(0.4, 0.2, real_len), np.zeros(T - real_len)])
    for t in range(T):
        real = t < real_len
        c2 = f * c2 + i * cand[t]                 # padding treated as a real zero input
        c1 = (f * c1 + i * cand[t]) if real else c1
        c_mask.append(c1); c_nomask.append(c2)
    drift = abs(c_nomask[-1] - c_mask[-1])
    FACT["mask_c_true"] = round(float(c_mask[-1]), 3)
    FACT["mask_c_padded"] = round(float(c_nomask[-1]), 3)
    FACT["mask_drift_pct"] = round(100 * float(drift / c_mask[-1]), 1)
    assert FACT["mask_drift_pct"] > 50
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.plot(range(T), c_mask, color=BLUE, lw=2.4, label="с маской: на padding состояние заморожено")
    ax.plot(range(T), c_nomask, color=RED, lw=2.4, ls=(0, (5, 3)),
            label="без маски: нули дожёвывают память")
    ax.axvspan(real_len - 0.5, T - 1, color=WASH)
    ax.text(real_len + 0.2, max(c_mask) * 0.95, "padding", color=MUTED, fontsize=11)
    ax.set_xlabel("шаг $t$"); ax.set_ylabel("память $c_t$")
    ax.set_title("Дополнение нулями — это не «ничего не произошло»")
    ax.legend(frameon=False, fontsize=10.5, loc="lower left")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "mask.png")
    print(f"mask: c_true={c_mask[-1]:.3f}, c_padded={c_nomask[-1]:.3f}, drift={FACT['mask_drift_pct']}%")


# ============================================================ sidenote images
def side_halflife():
    fs = np.linspace(0.5, 0.999, 400)
    th = np.log(0.5) / np.log(fs)
    marks = [0.9, 0.95, 0.99, 0.999]
    vals = [float(np.log(0.5) / np.log(f)) for f in marks]
    FACT["halflife"] = {str(f): round(v, 1) for f, v in zip(marks, vals)}
    assert abs(vals[0] - 6.5788) < 5e-05 and abs(vals[2] - 68.9676) < 1e-3, vals
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.semilogy(fs, th, color=BLUE, lw=2.2)
    for f, v in zip(marks, vals):
        ax.plot([f], [v], "o", color=RED, ms=5)
        ax.annotate(f"{v:.0f}".replace(".", ","), (f, v), textcoords="offset points",
                    xytext=(-16, 4), color=RED, fontsize=10)
    ax.set_xlabel("значение forget gate $f$"); ax.set_ylabel("$t_{1/2}$, шагов")
    ax.set_title("Полураспад памяти", fontsize=12)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5, which="both"); ax.set_axisbelow(True)
    save(fig, SIDE / "halflife.png")
    print("halflife:", FACT["halflife"])


def side_sigmoid():
    x = np.linspace(-8, 8, 400)
    s = sig(x)
    d = s * (1 - s)
    FACT["sigmoid_at_4"] = round(float(sig(4)), 4)
    FACT["sigmoid_deriv_at_4"] = round(float(sig(4) * (1 - sig(4))), 4)
    FACT["sigmoid_deriv_max"] = 0.25
    assert abs(FACT["sigmoid_at_4"] - 0.9820) < 5e-05
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(x, s, color=BLUE, lw=2.2, label="$\\sigma(z)$")
    ax.plot(x, d, color=RED, lw=2.0, label="$\\sigma'(z)$")
    ax.axhline(0.25, color=GRID, lw=1)
    ax.set_xlabel("$z$"); ax.set_title("Ворота и их чувствительность", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "sigmoid.png")


def side_params():
    n = 64; d_in, d_out = VOCAB, 2
    counts = {"RNN": n_params("rnn", d_in, n, d_out),
              "GRU": n_params("gru", d_in, n, d_out),
              "LSTM": n_params("lstm", d_in, n, d_out)}
    FACT["params_equal_width"] = counts
    assert counts["LSTM"] == 4 * (d_in * n + n * n + n) + n * d_out + d_out
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.barh(list(counts), list(counts.values()), color=[RED, GREEN, BLUE], height=0.5)
    for k, (name, v) in enumerate(counts.items()):
        ax.text(v * 1.02, k, str(v), va="center", color=MUTED, fontsize=10)
    ax.set_xlim(0, max(counts.values()) * 1.28)
    ax.set_xlabel(f"параметров при $n=64$, вход {VOCAB}")
    ax.set_title("Цена ворот", fontsize=12)
    ax.grid(True, axis="x", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "params.png")
    print("params equal width:", counts)


def side_gru():
    """Interpolation view: h = (1 - z) h_prev + z h_new for three values of z."""
    T = 40
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    target = 1.0
    for z, col in [(0.05, BLUE), (0.2, GREEN), (0.6, RED)]:
        h = 0.0; traj = []
        for t in range(T):
            h = (1 - z) * h + z * target
            traj.append(h)
        ax.plot(traj, color=col, lw=2.0, label=f"$z={z}$".replace(".", ","))
    n_steps = int(np.ceil(np.log(0.5) / np.log(1 - 0.2)))
    FACT["gru_z02_halfway"] = n_steps
    assert n_steps == 4, n_steps
    ax.set_xlabel("шаг"); ax.set_ylabel("$h_t$")
    ax.set_title("Update gate = скорость", fontsize=12)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "gru.png")


# ============================================================
def main():
    fig_cell()
    fig_regimes()
    side_halflife()
    side_sigmoid()
    side_params()
    side_gru()
    fig_mask()
    fig_copy()
    check_forget_bias()
    fig_gradient()
    fig_gates()
    fig_bike()
    FACTS.write_text(json.dumps(FACT, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nFACTS:", json.dumps(FACT, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
