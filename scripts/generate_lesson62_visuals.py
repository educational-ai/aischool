"""Deterministic figures for lesson 62: optimizers (momentum, AdaGrad, RMSProp, Adam, AdamW).

Every number quoted in the prose is computed here and asserted.

Real data:
  * sklearn load_breast_cancer (569x30, raw feature scales differ by ~5 orders) —
    logistic regression trained by GD / Momentum / AdaGrad / RMSProp / Adam, each with
    its OWN best learning rate from a grid;
  * scripts/data/sms-spam-collection.tsv — sparse bag-of-words, rare vs frequent word
    effective step under AdaGrad.
Synthetic (explicitly a model example, fixed seed): the anisotropic quadratic valley,
the memory-of-a-spike experiment and the bias-correction curves.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer

ROOT = Path(__file__).resolve().parents[1]
SMS = ROOT / "scripts" / "data" / "sms-spam-collection.tsv"
OUT = ROOT / "public" / "figures" / "lessons" / "62"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "62"

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


def fmt(x: float, n: int = 2) -> str:
    return f"{x:.{n}f}".replace(".", ",")


# ============================================================ optimizers (generic)
def run_opt(kind, grad, theta0, lr, steps, *, beta=0.9, b1=0.9, b2=0.999, eps=1e-8,
            loss=None, record=False):
    """Return (theta, history-of-loss, trajectory)."""
    th = theta0.astype(float).copy()
    v = np.zeros_like(th); m = np.zeros_like(th); G = np.zeros_like(th)
    hist, traj = [], [th.copy()]
    for t in range(1, steps + 1):
        g = grad(th)
        if not np.all(np.isfinite(g)):
            hist.append(np.inf); break
        if kind == "gd":
            th = th - lr * g
        elif kind == "momentum":
            v = beta * v + g
            th = th - lr * v
        elif kind == "adagrad":
            G = G + g * g
            th = th - lr * g / (np.sqrt(G) + eps)
        elif kind == "rmsprop":
            v = 0.9 * v + 0.1 * g * g
            th = th - lr * g / (np.sqrt(v) + eps)
        elif kind == "adam":
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
            th = th - lr * mh / (np.sqrt(vh) + eps)
        else:
            raise ValueError(kind)
        if not np.all(np.isfinite(th)):
            hist.append(np.inf); break
        if loss is not None:
            hist.append(float(loss(th)))
        if record:
            traj.append(th.copy())
    return th, np.array(hist, dtype=float), np.array(traj)


NAMES = {"gd": "GD", "momentum": "Momentum", "adagrad": "AdaGrad",
         "rmsprop": "RMSProp", "adam": "Adam"}
COLS = {"gd": RED, "momentum": BLUE, "adagrad": GOLD, "rmsprop": GREEN, "adam": VIOLET}


# ============================================================ fig 62.1 — the valley
def fig_valley() -> None:
    """f(x,y) = x^2 + 100 y^2, model example."""
    def grad(z):
        return np.array([2 * z[0], 200 * z[1]])

    def loss(z):
        return z[0] ** 2 + 100 * z[1] ** 2

    z0 = np.array([9.0, 1.0])
    steps = 60
    runs = {}
    for kind in ("gd", "momentum", "adam"):
        best = None
        for lr in np.logspace(-4, 0.6, 120):
            _, h, traj = run_opt(kind, grad, z0, lr, steps, loss=loss, record=True)
            fin = h[-1] if len(h) == steps and np.isfinite(h[-1]) else np.inf
            if best is None or fin < best[1][-1]:
                best = (lr, h, traj)
        runs[kind] = best
        print(f"  valley {NAMES[kind]:9s} lr*={best[0]:.4g} f60={best[1][-1]:.4g}")

    f0 = loss(z0)
    fin = {k: runs[k][1][-1] for k in runs}
    # zig-zag count: how many times the y-coordinate changes sign
    def crossings(traj):
        s = np.sign(traj[:, 1])
        return int(np.sum(s[1:] * s[:-1] < 0))
    cr_gd = crossings(runs["gd"][2]); cr_mom = crossings(runs["momentum"][2])
    print(f"valley: f0={f0:.1f} gd={fin['gd']:.3f} mom={fin['momentum']:.3f} adam={fin['adam']:.4g}")
    print(f"valley crossings: gd={cr_gd} momentum={cr_mom}")
    assert abs(f0 - 181.0) < 0.05
    assert fin["gd"] > 5 * fin["momentum"] and fin["adam"] < 1.0
    assert cr_gd >= 25 and cr_mom <= cr_gd / 2
    FACTS.update(valley_f0=f0, valley_gd=fin["gd"], valley_mom=fin["momentum"],
                 valley_adam=fin["adam"], valley_cr_gd=cr_gd, valley_cr_mom=cr_mom)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    xs = np.linspace(-10.5, 10.5, 400); ys = np.linspace(-1.35, 1.35, 400)
    XX, YY = np.meshgrid(xs, ys); ZZ = XX ** 2 + 100 * YY ** 2
    ax = axes[0]
    ax.contour(XX, YY, ZZ, levels=[2, 8, 25, 60, 120, 200, 320], colors=[GRID], linewidths=0.8)
    for kind in ("gd", "momentum", "adam"):
        lr, h, traj = runs[kind]
        ax.plot(traj[:, 0], traj[:, 1], color=COLS[kind], lw=1.7, marker="o", ms=2.6,
                label=f"{NAMES[kind]} ($\\eta={lr:g}$)")
    ax.plot([0], [0], marker="*", ms=15, color=INK, zorder=6)
    ax.set_xlabel("$x$ (пологое направление)"); ax.set_ylabel("$y$ (крутое)")
    ax.set_title("60 шагов в овраге $x^2+100y^2$", fontsize=13)
    ax.legend(loc="lower left", frameon=False, fontsize=9.5)
    ax = axes[1]
    for kind in ("gd", "momentum", "adam"):
        lr, h, traj = runs[kind]
        ax.semilogy(np.arange(1, len(h) + 1), np.maximum(h, 1e-12), color=COLS[kind],
                    lw=2.0, label=f"{NAMES[kind]}: $f_{{60}}={h[-1]:.3g}$")
    ax.set_xlabel("число вычисленных градиентов"); ax.set_ylabel("$f$ (лог. шкала)")
    ax.set_title("Тот же бюджет — разное падение", fontsize=13)
    ax.legend(frameon=False, fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "valley.png")


# ============================================================ fig 62.2 — memory of a spike
def fig_memory() -> None:
    T = 120
    g = np.full(T, 0.1)
    g[20] = 3.0
    b2 = 0.9
    Gada = np.cumsum(g ** 2)
    vrms = np.zeros(T); acc = 0.0
    for t in range(T):
        acc = b2 * acc + (1 - b2) * g[t] ** 2
        vrms[t] = acc
    step_ada = 1 / (np.sqrt(Gada) + 1e-8)
    step_rms = 1 / (np.sqrt(vrms) + 1e-8)
    before = step_ada[19]; after = step_ada[-1]
    rms_before = step_rms[19]; rms_after = step_rms[-1]
    drop_ada = before / after
    ratio_rms = rms_after / rms_before
    print(f"memory: adagrad step {before:.3f}->{after:.3f} (x{drop_ada:.1f}); "
          f"rmsprop {rms_before:.3f}->{rms_after:.3f} (ratio {ratio_rms:.3f})")
    assert drop_ada > 5 and 0.9 < ratio_rms < 1.1
    # time for RMSProp to come back within 5% of pre-spike step
    back = next(t for t in range(21, T) if step_rms[t] >= 0.5 * rms_before)
    back_after = back - 20
    back10 = next(t for t in range(21, T) if abs(step_rms[t] - rms_before) / rms_before < 0.10) - 20
    print(f"memory: rmsprop back to half the old step after {back_after} steps, "
          f"within 10% after {back10} (window 1/(1-b2)={1/(1-b2):.0f})")
    assert back_after <= 45 and back10 <= 80
    FACTS.update(mem_rms_back10=back10)
    FACTS.update(mem_ada_before=before, mem_ada_after=after, mem_drop=drop_ada,
                 mem_rms_back=back_after)

    fig, axes = plt.subplots(2, 1, figsize=(9.4, 6.0), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.25]})
    ax = axes[0]
    ax.plot(np.arange(T), g, color=INK, lw=1.6)
    ax.annotate("выброс $g=3$ на шаге 20", xy=(20, 3.0), xytext=(34, 2.4), fontsize=10,
                color=MUTED, arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax.set_ylabel("$|g_t|$"); ax.set_title("Один большой градиент и память двух накопителей", fontsize=13)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax = axes[1]
    ax.plot(np.arange(T), step_ada, color=GOLD, lw=2.2, label="AdaGrad: $1/\\sqrt{G_t}$")
    ax.plot(np.arange(T), step_rms, color=GREEN, lw=2.2, label="RMSProp: $1/\\sqrt{v_t}$, $\\beta_2=0{,}9$")
    ax.axhline(before, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.axvline(20 + back_after, color=GREEN, lw=0.9, ls=(0, (2, 3)))
    ax.text(20 + back_after + 2, before * 0.5, f"половина прежнего шага\nвосстановлена через {back_after} шагов",
            fontsize=9.5, color=GREEN)
    ax.set_yscale("log"); ax.set_xlabel("шаг $t$"); ax.set_ylabel("множитель шага")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "memory.png")


# ============================================================ fig 62.3 — bias correction
def fig_bias() -> None:
    T = 60
    b1, b2 = 0.9, 0.999
    m = np.zeros(T + 1); v = np.zeros(T + 1)
    for t in range(1, T + 1):
        m[t] = b1 * m[t - 1] + (1 - b1) * 1.0
        v[t] = b2 * v[t - 1] + (1 - b2) * 1.0
    t = np.arange(1, T + 1)
    mh = m[1:] / (1 - b1 ** t)
    vh = v[1:] / (1 - b2 ** t)
    raw_ratio = m[1:] / (np.sqrt(v[1:]) + 1e-8)
    print(f"bias: m1={m[1]:.3f} m10={m[10]:.4f} v1={v[1]:.4f} sqrt(v1)={np.sqrt(v[1]):.4f}")
    print(f"bias: uncorrected first step ratio m1/sqrt(v1)={raw_ratio[0]:.3f}; corrected={mh[0]/(np.sqrt(vh[0])+1e-8):.4f}")
    assert abs(m[1] - 0.1) < 0.05 and abs(v[1] - 0.001) < 1e-12
    assert abs(raw_ratio[0] - 3.1622767) < 5e-08
    assert np.allclose(mh, 1.0) and np.allclose(vh, 1.0)
    # how many steps until uncorrected m is within 1% of 1
    k99 = int(np.argmax(m[1:] > 0.99)) + 1
    print(f"bias: uncorrected m reaches 0.99 at t={k99}")
    assert k99 == 44
    FACTS.update(bias_m1=m[1], bias_v1=v[1], bias_raw_ratio=raw_ratio[0], bias_k99=k99)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
    ax = axes[0]
    ax.plot(t, m[1:], color=BLUE, lw=2.2, label="$m_t$ без коррекции")
    ax.plot(t, mh, color=RED, lw=2.2, label="$\\widehat m_t=m_t/(1-\\beta_1^t)$")
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.axvline(k99, color=BLUE, lw=0.8, ls=(0, (2, 3)))
    ax.text(k99 + 1, 0.55, f"0,99 достигнуто\nтолько на шаге {k99}", fontsize=9.5, color=BLUE)
    ax.set_xlabel("шаг $t$"); ax.set_ylabel("оценка среднего градиента")
    ax.set_title("Постоянный $g=1$: старт из нуля тянет вниз", fontsize=13)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax = axes[1]
    ax.plot(t, raw_ratio, color=GOLD, lw=2.2, label="$m_t/\\sqrt{v_t}$ без коррекции")
    ax.plot(t, mh / np.sqrt(vh), color=GREEN, lw=2.2, label="с коррекцией — ровно 1")
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.set_xlabel("шаг $t$"); ax.set_ylabel("множитель шага")
    ax.set_title("Первый шаг был бы в $\\sqrt{10}\\approx3{,}16$ раза длиннее", fontsize=13)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "bias.png")


# ============================================================ real data: breast cancer
def logistic(X, y):
    n = X.shape[0]

    def loss(w):
        z = X @ w
        return float(np.mean(np.logaddexp(0, z) - y * z))

    def grad(w):
        z = X @ w
        p = 0.5 * (1 + np.tanh(0.5 * np.clip(z, -60, 60)))
        return X.T @ (p - y) / n

    return loss, grad


def bc_data():
    d = load_breast_cancer()
    X = d.data.astype(float); y = d.target.astype(float)
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    mu = X.mean(0); sd = X.std(0)
    Xs = np.hstack([(X - mu) / sd, np.ones((X.shape[0], 1))])
    return X, Xb, Xs, y, sd


GRID_LR = np.logspace(-7, 1, 33)


def best_run(kind, X, y, steps):
    loss, grad = logistic(X, y)
    w0 = np.zeros(X.shape[1])
    best = None
    for lr in GRID_LR:
        _, h, _ = run_opt(kind, grad, w0, lr, steps, loss=loss)
        fin = h[-1] if len(h) and np.isfinite(h[-1]) else np.inf
        if best is None or fin < best[0]:
            best = (fin, lr, h)
    return best


def fig_real_scales() -> None:
    X, Xb, Xs, y, sd = bc_data()
    steps = 300
    raw, std = {}, {}
    for kind in ("gd", "momentum", "adagrad", "rmsprop", "adam"):
        raw[kind] = best_run(kind, Xb, y, steps)
        std[kind] = best_run(kind, Xs, y, steps)
        print(f"  raw {NAMES[kind]:9s} lr*={raw[kind][1]:.3g} loss={raw[kind][0]:.4f} | "
              f"std lr*={std[kind][1]:.3g} loss={std[kind][0]:.4f}")
    base = float(np.mean(np.logaddexp(0, 0) - y * 0))
    print(f"real: n={Xb.shape[0]}, d={Xb.shape[1]-1}, loss at w=0 = {base:.4f}")
    assert abs(base - np.log(2)) < 1e-12
    lr_span = max(r[1] for r in raw.values()) / min(r[1] for r in raw.values())
    print(f"real: best-lr spread across methods (raw) = {lr_span:.3g}x")
    assert raw["gd"][0] > raw["adam"][0] and raw["adam"][0] < 0.2
    assert lr_span >= 1e3
    FACTS.update(bc_n=Xb.shape[0], bc_d=Xb.shape[1] - 1,
                 bc_gd_raw=raw["gd"][0], bc_adam_raw=raw["adam"][0],
                 bc_adagrad_raw=raw["adagrad"][0], bc_rms_raw=raw["rmsprop"][0],
                 bc_mom_raw=raw["momentum"][0], bc_gd_std=std["gd"][0],
                 bc_adam_std=std["adam"][0], bc_lr_gd=raw["gd"][1], bc_lr_adam=raw["adam"][1],
                 bc_lr_span=lr_span,
                 bc_mom_std=std["momentum"][0], bc_rms_std=std["rmsprop"][0],
                 bc_adagrad_std=std["adagrad"][0])
    # every number quoted in the caption of fig. 62.5, to three decimals
    for key, val in (("gd", 0.040), ("momentum", 0.037), ("rmsprop", 0.035),
                     ("adagrad", 0.038), ("adam", 0.027)):
        assert abs(round(std[key][0], 3) - val) < 1e-9, (key, std[key][0])
    for key, val in (("gd", 0.250), ("momentum", 0.199), ("rmsprop", 0.229),
                     ("adagrad", 0.176), ("adam", 0.098)):
        assert abs(round(raw[key][0], 3) - val) < 1e-9, (key, raw[key][0])

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), sharey=True)
    for ax, book, title in ((axes[0], raw, "сырые признаки (масштабы врозь)"),
                            (axes[1], std, "после стандартизации")):
        for kind in ("gd", "momentum", "adagrad", "rmsprop", "adam"):
            fin, lr, h = book[kind]
            ax.plot(np.arange(1, len(h) + 1), h, color=COLS[kind], lw=2.0,
                    label=f"{NAMES[kind]} ($\\eta^*={lr:.0e}$): {h[-1]:.3f}")
        ax.axhline(np.log(2), color=MUTED, lw=0.8, ls=(0, (4, 3)))
        ax.set_xlabel("итерация (полный градиент)"); ax.set_title(title, fontsize=13)
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
        ax.set_ylim(0, 0.78)
    axes[0].set_ylabel("логистическая потеря на обучении")
    fig.suptitle("Реальные данные (breast cancer, 569×30): у каждого метода свой лучший $\\eta$",
                 fontsize=14, y=1.02)
    save(fig, OUT / "real_scales.png")

    # ---- sidenote: feature scale spread + condition number
    Xc = X - X.mean(0)
    cond_raw = np.linalg.cond(Xc.T @ Xc / X.shape[0])
    Xz = (X - X.mean(0)) / X.std(0)
    cond_std = np.linalg.cond(Xz.T @ Xz / X.shape[0])
    ratio = sd.max() / sd.min()
    print(f"sidenote scales: sd ratio={ratio:.3g}, cond raw={cond_raw:.3g}, cond std={cond_std:.3g}")
    assert ratio > 1e4 and cond_raw > 1e7 and cond_std < 1e5
    FACTS.update(sd_ratio=ratio, cond_raw=cond_raw, cond_std=cond_std)
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    order = np.argsort(sd)
    ax.semilogy(np.arange(1, 31), sd[order], marker="o", ms=4, color=BLUE, lw=1.4)
    ax.set_xlabel("признаки, по возрастанию разброса"); ax.set_ylabel("стандартное отклонение")
    ax.set_title("Разброс шкал: $\\times{}$%s" % f"{ratio:.0f}".replace(",", " "), fontsize=12)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "scales.png")


# ============================================================ fig 62.4 — learning-rate sensitivity
def fig_lr_curve() -> None:
    X, Xb, Xs, y, sd = bc_data()
    loss, grad = logistic(Xb, y)
    w0 = np.zeros(Xb.shape[1])
    steps = 200
    curves = {}
    for kind in ("gd", "momentum", "adagrad", "rmsprop", "adam"):
        fin = []
        for lr in GRID_LR:
            _, h, _ = run_opt(kind, grad, w0, lr, steps, loss=loss)
            v = h[-1] if len(h) and np.isfinite(h[-1]) else np.nan
            fin.append(v if np.isfinite(v) else np.nan)
        curves[kind] = np.array(fin)
        i = int(np.nanargmin(curves[kind]))
        print(f"  lrcurve {NAMES[kind]:9s} best lr={GRID_LR[i]:.3g} loss={curves[kind][i]:.4f}")
    best_lr = {k: GRID_LR[int(np.nanargmin(v))] for k, v in curves.items()}
    span = max(best_lr.values()) / min(best_lr.values())
    # what happens if everyone gets Adam's best lr
    lr_common = best_lr["adam"]
    j = int(np.argmin(np.abs(GRID_LR - lr_common)))
    at_common = {k: curves[k][j] for k in curves}
    print(f"lrcurve: span of best lr = {span:.3g}x; at Adam's lr={lr_common:.3g}: "
          + ", ".join(f"{NAMES[k]}={at_common[k]:.3f}" for k in at_common))
    assert span >= 900
    gd_at = at_common["gd"]; adam_at = at_common["adam"]
    assert (not np.isfinite(gd_at)) or gd_at > 3 * adam_at
    n_broken = int(sum(1 for k in at_common if not np.isfinite(at_common[k])))
    print(f"lrcurve: methods that diverge at Adam's lr: {n_broken} of 5")
    FACTS.update(lr_broken=n_broken)
    best_loss = {k: float(np.nanmin(v)) for k, v in curves.items()}
    FACTS.update(lr_span_std=span, lr_common=lr_common,
                 lr_common_gd=gd_at, lr_common_adam=adam_at,
                 lr_best_gd=best_lr["gd"], lr_best_adam=best_lr["adam"],
                 lr_best_loss_gd=best_loss["gd"], lr_best_loss_adam=best_loss["adam"],
                 lr_common_mom=float(at_common["momentum"]))
    # numbers quoted in the caption of fig. 62.6
    assert abs(round(best_loss["gd"], 3) - 0.288) < 0.0005, best_loss["gd"]
    assert abs(round(best_loss["adam"], 3) - 0.134) < 0.0005, best_loss["adam"]
    assert round(gd_at) == 188 and round(float(at_common["momentum"])) == 101
    assert abs(best_lr["gd"] - 3.16e-5) < 1e-7 and abs(best_lr["adam"] - 3.16e-2) < 1e-4

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for kind in ("gd", "momentum", "adagrad", "rmsprop", "adam"):
        ax.semilogx(GRID_LR, curves[kind], color=COLS[kind], lw=2.0, marker="o", ms=3,
                    label=f"{NAMES[kind]} (лучший $\\eta$ = {best_lr[kind]:.3g})")
        i = int(np.nanargmin(curves[kind]))
        ax.plot([GRID_LR[i]], [curves[kind][i]], marker="v", ms=9, color=COLS[kind])
    ax.axvline(lr_common, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.text(lr_common * 0.6, 600, "общий $\\eta$ для всех —\nнесправедливое сравнение",
            fontsize=10, color=MUTED, ha="right")
    ax.set_xlabel("learning rate $\\eta$ (лог. шкала)")
    ax.set_ylabel("потеря после 200 итераций (лог. шкала)")
    ax.set_yscale("log"); ax.set_ylim(0.08, 3e3)
    ax.set_title("У каждого оптимизатора своя рабочая зона шага (сырой breast cancer)", fontsize=13.5)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "lr_curve.png")


# ============================================================ real sparse text: SMS
def sms_data():
    labels, texts = [], []
    with open(SMS, encoding="utf-8", errors="replace") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 2:
                continue
            labels.append(1.0 if row[0].strip() == "spam" else 0.0)
            texts.append(row[1].lower())
    y = np.array(labels)
    toks = [re.findall(r"[a-z']+", t) for t in texts]
    rng = np.random.default_rng(62)
    idx = rng.permutation(len(y))
    ntr = int(0.8 * len(y))
    tr, te = idx[:ntr], idx[ntr:]
    from collections import Counter
    cnt = Counter()
    for i in tr:
        cnt.update(set(toks[i]))
    vocab = {w: j for j, (w, c) in enumerate(sorted(cnt.items()) ) if c >= 5}
    vocab = {w: j for j, w in enumerate(sorted(vocab))}
    d = len(vocab)

    def mat(ids):
        M = np.zeros((len(ids), d + 1))
        for r, i in enumerate(ids):
            for w in set(toks[i]):
                j = vocab.get(w)
                if j is not None:
                    M[r, j] = 1.0
            M[r, d] = 1.0
        return M

    return mat(tr), y[tr], mat(te), y[te], vocab, cnt


def fig_sparse() -> None:
    Xtr, ytr, Xte, yte, vocab, cnt = sms_data()
    d = len(vocab)
    print(f"sms: train={Xtr.shape[0]}, test={Xte.shape[0]}, vocab={d}, "
          f"density={Xtr[:, :d].mean()*100:.2f}%")
    density = Xtr[:, :d].mean() * 100
    assert 4000 < Xtr.shape[0] < 4600 and 1000 < Xte.shape[0] < 1200
    assert density < 1.0

    loss_tr, grad_tr = logistic(Xtr, ytr)

    def epoch(kind, lr, batches=1, seed=62, b2=0.999):
        rng = np.random.default_rng(seed)
        w = np.zeros(Xtr.shape[1])
        G = np.zeros_like(w); v = np.zeros_like(w); mm = np.zeros_like(w)
        t = 0
        for _ in range(batches):
            order = rng.permutation(Xtr.shape[0])
            for s in range(0, len(order), 32):
                ids = order[s:s + 32]
                Xb = Xtr[ids]; yb = ytr[ids]
                z = Xb @ w
                p = 0.5 * (1 + np.tanh(0.5 * np.clip(z, -60, 60)))
                g = Xb.T @ (p - yb) / len(ids)
                t += 1
                if kind == "gd":
                    w = w - lr * g
                elif kind == "adagrad":
                    G += g * g
                    w = w - lr * g / (np.sqrt(G) + 1e-8)
                elif kind == "adam":
                    mm = 0.9 * mm + 0.1 * g
                    v = b2 * v + (1 - b2) * g * g
                    w = w - lr * (mm / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + 1e-8)
        return w, G

    def acc(w, X, yv):
        return float(np.mean(((X @ w) > 0).astype(float) == yv))

    res = {}
    for kind in ("gd", "adagrad", "adam"):
        best = None
        for lr in np.logspace(-3, 1, 17):
            w, G = epoch(kind, lr, batches=3)
            a = acc(w, Xte, yte)
            if best is None or a > best[0]:
                best = (a, lr, w, G)
        res[kind] = best
        print(f"  sms {NAMES[kind]:8s} lr*={best[1]:.3g} test acc={best[0]*100:.2f}% "
              f"train loss={loss_tr(best[2]):.4f}")
    acc_gd = res["gd"][0]; acc_ada = res["adagrad"][0]
    assert acc_ada > acc_gd
    FACTS.update(sms_train=Xtr.shape[0], sms_test=Xte.shape[0], sms_vocab=d,
                 sms_density=density, sms_acc_gd=acc_gd * 100, sms_acc_ada=acc_ada * 100,
                 sms_acc_adam=res["adam"][0] * 100)

    # rare vs frequent word: AdaGrad accumulator and effective multiplier
    G = res["adagrad"][3]
    inv = sorted(vocab.items(), key=lambda kv: kv[1])
    freqs = np.array([cnt[w] for w, j in inv])
    mult = 1.0 / (np.sqrt(G[:d]) + 1e-8)
    ok = G[:d] > 0
    common = freqs >= 300
    rare = (freqs >= 5) & (freqs <= 8)
    m_common = float(np.median(mult[ok & common]))
    m_rare = float(np.median(mult[ok & rare]))
    print(f"sms: median AdaGrad multiplier — frequent words {m_common:.1f}, rare {m_rare:.1f}, "
          f"ratio {m_rare/m_common:.1f}")
    assert m_rare > 3 * m_common
    n_common = int(np.sum(common)); n_rare = int(np.sum(rare & ok))
    FACTS.update(sms_mult_common=m_common, sms_mult_rare=m_rare,
                 sms_mult_ratio=m_rare / m_common, sms_n_common=n_common, sms_n_rare=n_rare)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    ax = axes[0]
    ax.loglog(freqs[ok], mult[ok], ".", ms=3.5, color=BLUE, alpha=0.5)
    ax.axhline(m_common, color=RED, lw=1.6, ls=(0, (5, 3)),
               label=f"частые (≥300 писем): медиана {m_common:.1f}")
    ax.axhline(m_rare, color=GOLD, lw=1.6, ls=(0, (5, 3)),
               label=f"редкие (5–8 писем): медиана {m_rare:.1f}")
    ax.set_xlabel("в скольких обучающих SMS встретилось слово")
    ax.set_ylabel("множитель шага AdaGrad $1/\\sqrt{G}$")
    ax.set_title("Редкое слово сохраняет крупный шаг", fontsize=13)
    ax.legend(frameon=False, fontsize=9.5, loc="lower left")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4, which="both"); ax.set_axisbelow(True)
    ax = axes[1]
    names = ["SGD", "AdaGrad", "Adam"]
    vals = [acc_gd * 100, acc_ada * 100, res["adam"][0] * 100]
    cols = [RED, GOLD, VIOLET]
    ax.bar(names, vals, color=cols, width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.12, f"{v:.2f}%", ha="center", fontsize=11, color=INK)
    ax.set_ylim(min(vals) - 1.6, max(vals) + 0.9)
    ax.set_ylabel("точность на отложенных SMS")
    ax.set_title("3 эпохи, батч 32, у каждого свой лучший $\\eta$", fontsize=13)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.suptitle(f"Реальный SMS-спам: {Xtr.shape[0]} обучающих писем, словарь {d}, "
                 f"заполненность {density:.2f}%", fontsize=13.5, y=1.03)
    save(fig, OUT / "sparse_real.png")


# ============================================================ fig 62.5 — Adam + L2 vs AdamW
def fig_adamw() -> None:
    X, Xb, Xs, y, sd = bc_data()
    loss, grad = logistic(Xs, y)
    d = Xs.shape[1]
    steps = 400
    lr = 0.05
    lam = 0.05

    def run(mode):
        w = np.zeros(d); m = np.zeros(d); v = np.zeros(d)
        for t in range(1, steps + 1):
            g = grad(w)
            if mode == "l2":
                g = g + lam * w
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            mh = m / (1 - 0.9 ** t); vh = v / (1 - 0.999 ** t)
            upd = lr * mh / (np.sqrt(vh) + 1e-8)
            if mode == "w":
                w = w - upd - lr * lam * w
            else:
                w = w - upd
        return w

    w_l2 = run("l2"); w_w = run("w"); w_plain = run("none")
    n_l2 = float(np.linalg.norm(w_l2[:-1])); n_w = float(np.linalg.norm(w_w[:-1]))
    n_p = float(np.linalg.norm(w_plain[:-1]))
    # spread of shrinkage across coordinates: |w| relative to the un-penalised run
    sh_l2 = np.abs(w_l2[:-1]) / (np.abs(w_plain[:-1]) + 1e-12)
    sh_w = np.abs(w_w[:-1]) / (np.abs(w_plain[:-1]) + 1e-12)
    print(f"adamw: ||w|| plain={n_p:.2f}, Adam+L2={n_l2:.2f}, AdamW={n_w:.2f}")
    print(f"adamw: shrink factor spread — L2 {sh_l2.min():.2f}..{sh_l2.max():.2f}, "
          f"AdamW {sh_w.min():.2f}..{sh_w.max():.2f}")
    print(f"adamw: loss plain={loss(w_plain):.4f}, L2={loss(w_l2):.4f}, W={loss(w_w):.4f}")
    spread_l2 = float(sh_l2.max() / sh_l2.min()); spread_w = float(sh_w.max() / sh_w.min())
    assert n_l2 < n_p and n_w < n_p
    assert spread_l2 > 5 * spread_w
    FACTS.update(aw_norm_plain=n_p, aw_norm_l2=n_l2, aw_norm_w=n_w,
                 aw_spread_l2=spread_l2, aw_spread_w=spread_w,
                 aw_loss_plain=loss(w_plain), aw_loss_l2=loss(w_l2), aw_loss_w=loss(w_w),
                 aw_lam=lam, aw_lr=lr)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))
    ax = axes[0]
    o = np.argsort(-np.abs(w_plain[:-1]))
    xx = np.arange(1, d)
    ax.plot(xx, np.abs(w_plain[:-1])[o], color=MUTED, lw=1.6, label=f"Adam без decay, $\\|w\\|={n_p:.2f}$")
    ax.plot(xx, np.abs(w_l2[:-1])[o], color=RED, lw=1.9, label=f"Adam + $L_2$ в градиенте, $\\|w\\|={n_l2:.2f}$")
    ax.plot(xx, np.abs(w_w[:-1])[o], color=BLUE, lw=1.9, label=f"AdamW (отдельный decay), $\\|w\\|={n_w:.2f}$")
    ax.set_xlabel("координаты, по убыванию $|w_j|$"); ax.set_ylabel("$|w_j|$")
    ax.set_title(f"$\\lambda={lam}$, $\\eta={lr}$, 400 шагов", fontsize=13)
    ax.legend(frameon=False, fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax = axes[1]
    ax.hist(sh_l2, bins=14, color=RED, alpha=0.55, label=f"$L_2$ в градиенте: разброс ×{spread_l2:.1f}")
    ax.hist(sh_w, bins=14, color=BLUE, alpha=0.55, label=f"AdamW: разброс ×{spread_w:.1f}")
    ax.set_xlabel("во сколько раз сжат вес относительно прогона без decay")
    ax.set_ylabel("число координат")
    ax.set_title("Одинаково ли сжались веса?", fontsize=13)
    ax.legend(frameon=False, fontsize=9.5)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "adamw.png")


# ============================================================ sidenote: clipping
def side_clip() -> None:
    X, Xb, Xs, y, sd = bc_data()
    loss, grad = logistic(Xs, y)
    rng = np.random.default_rng(62)
    w = np.zeros(Xs.shape[1])
    norms = []
    for t in range(1, 301):
        ids = rng.integers(0, Xs.shape[0], 16)
        Xbb = Xs[ids]; yb = y[ids]
        z = Xbb @ w
        p = 0.5 * (1 + np.tanh(0.5 * np.clip(z, -60, 60)))
        g = Xbb.T @ (p - yb) / len(ids)
        nn = float(np.linalg.norm(g)); norms.append(nn)
        w = w - 0.1 * g
    norms = np.array(norms)
    c = float(np.quantile(norms, 0.9))
    frac = float(np.mean(norms > c)) * 100
    print(f"clip: median norm={np.median(norms):.3f}, max={norms.max():.3f}, "
          f"c=q90={c:.3f}, clipped {frac:.1f}%, max/median={norms.max()/np.median(norms):.1f}")
    assert 8 < frac < 12
    FACTS.update(clip_med=float(np.median(norms)), clip_max=float(norms.max()),
                 clip_c=c, clip_frac=frac, clip_ratio=float(norms.max() / np.median(norms)))
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(np.arange(1, 301), norms, color=BLUE, lw=1.0)
    ax.axhline(c, color=RED, lw=1.5, ls=(0, (5, 3)))
    ax.text(8, c * 1.06, f"порог $c={c:.2f}$ — обрезано {frac:.0f}% шагов", fontsize=9, color=RED)
    ax.set_xlabel("шаг мини-батча"); ax.set_ylabel("$\\|g_t\\|$")
    ax.set_title("Норма стохастического градиента", fontsize=12)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "clip.png")


# ============================================================ sidenote: epsilon
def side_eps() -> None:
    v = np.logspace(-14, 0, 400)
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for eps, col, lab in ((1e-8, BLUE, "$\\varepsilon=10^{-8}$"),
                          (1e-4, GOLD, "$\\varepsilon=10^{-4}$"),
                          (1e-2, RED, "$\\varepsilon=10^{-2}$")):
        ax.loglog(np.sqrt(v), 1 / (np.sqrt(v) + eps), color=col, lw=2.0, label=lab)
    ax.loglog(np.sqrt(v), 1 / np.sqrt(v), color=MUTED, lw=0.9, ls=(0, (3, 3)), label="$1/\\sqrt{v}$")
    ax.set_xlabel("$\\sqrt{v}$"); ax.set_ylabel("множитель шага")
    ax.set_title("$\\varepsilon$ — потолок адаптивности", fontsize=12)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4, which="both"); ax.set_axisbelow(True)
    save(fig, SIDE / "epsilon.png")
    # asserted claim: at sqrt(v)=1e-3 the multiplier with eps=1e-2 is ~100 while pure is 1000
    m_eps = 1 / (1e-3 + 1e-2); m_pure = 1 / 1e-3
    print(f"epsilon: at sqrt(v)=1e-3 multiplier {m_eps:.1f} vs pure {m_pure:.0f} "
          f"(damped {m_pure/m_eps:.1f}x)")
    assert abs(m_eps - 90.909) < 0.0005
    FACTS.update(eps_mult=m_eps, eps_pure=m_pure, eps_damp=m_pure / m_eps)


# ============================================================ sidenote: rescaling makes it round
def side_rescale() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.9))
    th = np.linspace(0, 2 * np.pi, 300)
    ax = axes[0]
    ax.plot(np.cos(th) * 3, np.sin(th) * 0.3, color=RED, lw=2.0)
    ax.plot(np.cos(th) * 1.5, np.sin(th) * 0.15, color=RED, lw=1.0, alpha=0.6)
    ax.set_title("$x^2+100y^2$", fontsize=11); ax.set_xlim(-3.6, 3.6); ax.set_ylim(-1.8, 1.8)
    ax = axes[1]
    ax.plot(np.cos(th) * 1.5, np.sin(th) * 1.5, color=GREEN, lw=2.0)
    ax.plot(np.cos(th) * 0.75, np.sin(th) * 0.75, color=GREEN, lw=1.0, alpha=0.6)
    ax.set_title("$x^2+u^2$", fontsize=11); ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        for s in ax.spines.values():
            s.set_color(LINE)
    fig.suptitle("Замена $u=10y$ делает овраг круглым", fontsize=11, y=1.02)
    save(fig, SIDE / "rescale.png")



# ============================================================ numbers quoted in prose
def check_prose() -> None:
    import math
    rho = 99 / 101
    T = math.log(1000) / math.log(1 / rho)
    print(f"prose: rho={rho:.4f}, T(1000x)={T:.1f}, rho^60={rho**60:.3f}")
    assert abs(rho - 0.9802) < 5e-05 and 345 < T < 347 and abs(rho ** 60 - 0.30) < 0.01
    r10 = 9 / 11
    print(f"prose: kappa=10 rho={r10:.3f}, rho^60={r10**60:.2e}")
    assert abs(r10 - 0.818) < 0.0005 and 5.5e-6 < r10 ** 60 < 6.5e-6 and 1.6e5 < 1 / r10 ** 60 < 1.8e5
    v = [0.0]
    for _ in range(4):
        v.append(0.8 * v[-1] + 2)
    print(f"prose: momentum v1..v4 = {v[1:]}, limit {2/0.2}")
    assert all(abs(a - b) < 1e-9 for a, b in zip(v[1:], [2.0, 3.6, 4.88, 5.904]))
    n90 = math.log(0.1) / math.log(0.8)
    assert 10 < n90 < 10.4
    ratios = [FACTS["valley_f0"] / FACTS[k] for k in ("valley_gd", "valley_mom", "valley_adam")]
    print(f"prose: f0/f60 = {ratios[0]:.1f}, {ratios[1]:.0f}, {ratios[2]:.0f}")
    assert 21 < ratios[0] < 22 and 8500 < ratios[1] < 9200 and 24000 < ratios[2] < 25000
    assert abs(math.sqrt(400 / 6) - 8.16) < 0.005
    assert 1.9e5 < 1.5 ** 30 < 2.2e5
    zeros = 100 - FACTS["sms_density"]
    print(f"prose: SMS zeros {zeros:.2f}%")
    assert 99.1 < zeros < 99.3
    raw_span = FACTS["bc_gd_raw"] / FACTS["bc_adam_raw"]
    std_span = FACTS["bc_gd_std"] / FACTS["bc_adam_std"]
    print(f"prose: worst/best raw {raw_span:.2f}, std {std_span:.2f}")
    assert 2.4 < raw_span < 2.6 and 1.4 < std_span < 1.55
    assert abs(math.log(2) - 0.6931) < 5e-05


def fig_flows() -> None:
    """Непрерывный поток, его дискретизация и то, где живёт ускорение.

    Плохо обусловленный квадратичный овраг (kappa = 80). Поток dx/dt = -grad f идёт
    прямо ко дну: у него нет ограничения на шаг. Спуск - явная схема Эйлера для
    этого потока, и его шаг упёрт в 2/L, отсюда зигзаг. Ускорение (тяжёлый шарик)
    имеет смысл именно в дискретном мире.
    """
    from scipy.integrate import solve_ivp

    theta = np.deg2rad(32.0)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    H = rot @ np.diag([1.0, 80.0]) @ rot.T
    x0 = np.array([3.2, 2.1])
    mu, L = 1.0, 80.0
    kappa = L / mu

    f = lambda x: 0.5 * float(x @ H @ x)
    f0 = f(x0)

    t = np.linspace(0.0, 9.0, 900)
    flow = solve_ivp(lambda _t, x: -(H @ x), (t[0], t[-1]), x0,
                     t_eval=t, rtol=1e-10, atol=1e-12).y.T

    a_gd = 2 / (mu + L)                                   # оптимальный шаг спуска
    x = x0.copy(); gd = [x.copy()]
    for _ in range(300):
        x = x - a_gd * (H @ x); gd.append(x.copy())
    gd = np.asarray(gd)

    beta = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** 2
    a_hb = 4 / ((np.sqrt(L) + np.sqrt(mu)) ** 2)
    x = x0.copy(); xp = x0.copy(); hb = [x.copy()]
    for _ in range(300):
        xn = x - a_hb * (H @ x) + beta * (x - xp)
        xp, x = x, xn
        hb.append(x.copy())
    hb = np.asarray(hb)

    f_gd = np.array([f(p) for p in gd]) / f0
    f_hb = np.array([f(p) for p in hb]) / f0

    def steps_to(v, thr):
        i = int(np.argmax(v < thr))
        assert v[i] < thr
        return i

    n_gd3, n_hb3 = steps_to(f_gd, 1e-3), steps_to(f_hb, 1e-3)
    n_gd6, n_hb6 = steps_to(f_gd, 1e-6), steps_to(f_hb, 1e-6)
    gain3, gain6 = n_gd3 / n_hb3, n_gd6 / n_hb6

    assert abs(a_gd - 0.024691) < 5e-7, a_gd
    assert abs(a_hb - 0.040450) < 5e-7, a_hb
    assert abs(beta - 0.638208) < 5e-7, beta
    assert abs(np.sqrt(kappa) - 8.9443) < 5e-5
    assert (n_gd3, n_hb3) == (139, 27), (n_gd3, n_hb3)
    assert (n_gd6, n_hb6) == (277, 44), (n_gd6, n_hb6)
    assert abs(gain3 - 5.15) < 5e-3 and abs(gain6 - 6.30) < 5e-3

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.3))

    xs = np.linspace(-0.9, 3.7, 320); ys = np.linspace(-0.9, 2.5, 320)
    xx, yy = np.meshgrid(xs, ys)
    zz = 0.5 * (H[0, 0] * xx ** 2 + 2 * H[0, 1] * xx * yy + H[1, 1] * yy ** 2)
    a1.contour(xx, yy, zz, levels=np.geomspace(0.05, 260, 14), colors=[GRID], linewidths=0.7)
    a1.plot(flow[:, 0], flow[:, 1], color=BLUE, lw=3.4, alpha=0.95,
            label=r"поток $\dot x=-\nabla f$", zorder=3)
    a1.plot(gd[:60, 0], gd[:60, 1], color=RED, lw=1.0, marker="o", ms=1.9, alpha=0.9,
            label=f"спуск, шаг {fmt(a_gd, 3)}", zorder=4)
    a1.plot(hb[:60, 0], hb[:60, 1], color=GREEN, lw=1.4, marker="o", ms=2.4, alpha=0.95,
            label="тяжёлый шарик", zorder=5)
    a1.plot(0, 0, "*", color=INK, ms=15, zorder=6)
    a1.set_xlim(-0.9, 3.7); a1.set_ylim(-0.9, 2.5)
    a1.set_xticks([]); a1.set_yticks([])
    a1.set_title(f"Овраг $\\kappa={kappa:.0f}$: поток и две его дискретизации", fontsize=12.5)
    a1.legend(frameon=False, fontsize=9.5, loc="upper left")

    it = np.arange(len(f_gd))
    a2.semilogy(it, np.maximum(f_gd, 1e-16), color=RED, lw=2.2, label="спуск")
    a2.semilogy(it, np.maximum(f_hb, 1e-16), color=GREEN, lw=2.2, label="тяжёлый шарик")
    a2.axhline(1e-6, color=LINE, lw=1.0, ls=(0, (4, 3)))
    a2.set_xlim(0, 300); a2.set_ylim(1e-12, 2)
    a2.set_xlabel("число шагов"); a2.set_ylabel("$f/f_0$")
    a2.set_title("Ускорение живёт в дискретном мире", fontsize=12.5)
    a2.grid(True, color=GRID, lw=0.6, alpha=0.6); a2.set_axisbelow(True)
    a2.legend(frameon=False, fontsize=10, loc="upper right")
    a2.annotate(f"до $10^{{-6}}$: {n_hb6} против {n_gd6} шагов",
                xy=(n_hb6, 1e-6), xytext=(70, 2e-10), fontsize=10.5, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=LINE, lw=1.0))

    fig.suptitle("Спуск — это шаг по потоку; момент нужен там, где шаг ограничен", y=1.02, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "flows.png")

    FACTS.update(flow_kappa=kappa, flow_sqrt_kappa=float(np.sqrt(kappa)),
                 flow_a_gd=a_gd, flow_a_hb=a_hb, flow_beta=beta,
                 flow_gd3=n_gd3, flow_hb3=n_hb3, flow_gd6=n_gd6, flow_hb6=n_hb6,
                 flow_gain3=gain3, flow_gain6=gain6)
    print(f"flows: kappa={kappa:.0f} sqrt={np.sqrt(kappa):.4f} gd3={n_gd3} hb3={n_hb3} "
          f"gd6={n_gd6} hb6={n_hb6} gain3={gain3:.2f} gain6={gain6:.2f}")


def main() -> None:
    fig_valley()
    fig_flows()
    fig_memory()
    fig_bias()
    fig_real_scales()
    fig_lr_curve()
    fig_sparse()
    fig_adamw()
    side_clip()
    side_eps()
    side_rescale()
    check_prose()
    (ROOT / "scripts" / "data").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "scripts" / "data" / "lesson62_facts.json", "w") as f:
        json.dump({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                   for k, v in FACTS.items()}, f, ensure_ascii=False, indent=1)
    print("\n=== FACTS ===")
    for k, v in FACTS.items():
        print(f"{k:20s} {v}")


if __name__ == "__main__":
    main()
