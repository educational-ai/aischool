"""Deterministic figures for lesson 61: mini-batch SGD, learning rate and early stopping.

Everything is measured on the REAL bike-sharing-hour.csv (17379 hours) and on the REAL
SMS Spam Collection (class balance only). A linear model with cyclic + weather features is
trained by full-batch GD and by SGD with different batch sizes; the noise of the mini-batch
gradient, the step-size thresholds, the schedules and the early-stopping curves are all
computed, printed and asserted. No number in the lesson text is invented.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BIKE = ROOT / "scripts" / "data" / "bike-sharing-hour.csv"
SMS = ROOT / "scripts" / "data" / "sms-spam-collection.tsv"
OUT = ROOT / "public" / "figures" / "lessons" / "61"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "61"

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


# ---------------------------------------------------------------- real data
def load_bike():
    hr, temp, hum, wind, work, cnt = [], [], [], [], [], []
    with open(BIKE) as f:
        for row in csv.DictReader(f):
            hr.append(int(row["hr"])); temp.append(float(row["temp"]))
            hum.append(float(row["hum"])); wind.append(float(row["windspeed"]))
            work.append(float(row["workingday"])); cnt.append(float(row["cnt"]))
    hr = np.array(hr); cnt = np.array(cnt)
    ang = 2 * np.pi * hr / 24.0
    cols = [np.ones_like(cnt)]
    for k in (1, 2, 3):
        cols += [np.sin(k * ang), np.cos(k * ang)]
    for v in (temp, hum, wind, work):
        v = np.array(v)
        cols.append((v - v.mean()) / (v.std() + 1e-12))
    X = np.column_stack(cols)
    y = (cnt - cnt.mean()) / cnt.std()
    return X, y, cnt


X, y, RAW = load_bike()
N, D = X.shape
FACTS["n_rows"] = N
FACTS["n_features"] = D - 1
H = X.T @ X / N
EIG = np.linalg.eigvalsh(H)
L = float(EIG[-1])
MU = float(EIG[0])
FACTS["L"] = L
FACTS["mu"] = MU
FACTS["kappa"] = L / MU
THETA_STAR = np.linalg.solve(H, X.T @ y / N)
MSE_STAR = float(np.mean((X @ THETA_STAR - y) ** 2))
FACTS["mse_star"] = MSE_STAR
FACTS["r2_star"] = 1 - MSE_STAR


def grad(theta, idx=None):
    if idx is None:
        r = X @ theta - y
        return X.T @ r / N
    Xb = X[idx]; rb = Xb @ theta - y[idx]
    return Xb.T @ rb / len(idx)


def mse(theta):
    return float(np.mean((X @ theta - y) ** 2))


# ------------------------------------------------- fig 1: cloud of batch gradients
def fig_gradient_cloud():
    rng = np.random.default_rng(61)
    theta0 = np.zeros(D)
    g = grad(theta0)
    # two most informative coordinates: sin(1) and cos(1) of the hour
    i, j = 1, 2
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9), sharex=True, sharey=True)
    stats = {}
    for ax, b in zip(axes, (4, 32, 256)):
        pts = np.array([grad(theta0, rng.integers(0, N, b))[[i, j]] for _ in range(400)])
        sd = float(np.mean(np.std(pts, axis=0)))
        stats[b] = sd
        ax.scatter(pts[:, 0], pts[:, 1], s=12, color=BLUE, alpha=0.35, linewidths=0)
        ax.annotate("", xy=(g[i], g[j]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.0))
        ax.plot([g[i]], [g[j]], "o", color=RED, ms=7, zorder=6)
        ax.axhline(0, color=LINE, lw=0.8); ax.axvline(0, color=LINE, lw=0.8)
        ax.set_title(f"батч $b={b}$: разброс {sd:.3f}", fontsize=11)
        ax.set_xlabel(r"$\partial\widehat R/\partial w_{\sin}$", fontsize=10)
        ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    axes[0].set_ylabel(r"$\partial\widehat R/\partial w_{\cos}$", fontsize=10)
    fig.suptitle("Облако мини-батчей центрировано на полном градиенте (реальный велопрокат)",
                 y=1.04, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "gradient_cloud.png")
    FACTS["sd_b4"] = stats[4]; FACTS["sd_b32"] = stats[32]; FACTS["sd_b256"] = stats[256]
    FACTS["ratio_4_256"] = stats[4] / stats[256]
    print("cloud sd:", {k: round(v, 4) for k, v in stats.items()},
          "ratio 4/256 =", round(stats[4] / stats[256], 2))
    assert stats[4] > stats[32] > stats[256]
    assert 6.0 < stats[4] / stats[256] < 10.0     # expected 8 = sqrt(64)


# ------------------------------------------------- fig 2: the 1/sqrt(b) law
def fig_noise_law():
    rng = np.random.default_rng(612)
    theta0 = np.zeros(D)
    g = grad(theta0)
    bs = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    sds = []
    for b in bs:
        est = np.array([grad(theta0, rng.integers(0, N, int(b))) for _ in range(300)])
        sds.append(float(np.sqrt(np.mean(np.sum((est - g) ** 2, axis=1)))))
    sds = np.array(sds)
    slope, inter = np.polyfit(np.log(bs), np.log(sds), 1)
    FACTS["slope"] = float(slope)
    FACTS["sd_b1"] = float(sds[0])
    FACTS["sd_b100_pred"] = float(np.exp(inter) * 100.0 ** slope)
    print(f"noise law: slope={slope:.3f}, sd(b=1)={sds[0]:.3f}, sd(b=1024)={sds[-1]:.4f}")
    assert -0.53 < slope < -0.47
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    a0.loglog(bs, sds, "o-", color=BLUE, lw=2.0, ms=6, label="измеренный разброс")
    a0.loglog(bs, np.exp(inter) * bs ** slope, color=RED, lw=1.6, ls=(0, (5, 3)),
              label=f"наклон {slope:.2f}" + r" $\approx-1/2$")
    a0.set_xlabel("размер батча $b$"); a0.set_ylabel(r"$\sqrt{\mathbb{E}\|\widehat g-g\|^2}$")
    a0.set_title("Шум убывает как $1/\\sqrt{b}$", fontsize=12.5)
    a0.legend(frameon=False, fontsize=10)
    a0.grid(True, which="both", color=GRID, lw=0.4, alpha=0.4); a0.set_axisbelow(True)
    # cost per unit of noise reduction
    a1.plot(bs, bs, color=GOLD, lw=2.2, label="цена шага $\\propto b$")
    a1.plot(bs, sds[0] / sds, color=BLUE, lw=2.2, label="во сколько раз тише шум")
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel("размер батча $b$"); a1.set_ylabel("во сколько раз")
    a1.set_title("Плата растёт вчетверо быстрее выигрыша", fontsize=12.5)
    a1.legend(frameon=False, fontsize=10)
    a1.grid(True, which="both", color=GRID, lw=0.4, alpha=0.4); a1.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "noise_law.png")


# ------------------------------------------------- fig 3: equal budget
def run_sgd(b, eta, epochs_examples, seed, decay=None, record_every=None):
    rng = np.random.default_rng(seed)
    theta = np.zeros(D)
    steps = int(epochs_examples // b)
    rec_x, rec_y = [], []
    if record_every is None:
        record_every = max(1, steps // 120)
    for t in range(steps):
        lr = eta if decay is None else decay(eta, t)
        theta = theta - lr * grad(theta, rng.integers(0, N, b))
        if t % record_every == 0 or t == steps - 1:
            rec_x.append((t + 1) * b); rec_y.append(mse(theta))
    return theta, np.array(rec_x), np.array(rec_y)


def fig_budget():
    budget = 400_000
    eta = 0.35 / L
    curves = {}
    for b in (8, 64, 512):
        _, ex, ls = run_sgd(b, eta, budget, seed=7 + b,
                            record_every=max(1, int(budget // b) // 400))
        curves[b] = (ex, ls, ex / b)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.2, 4.3), sharey=True)
    for (b, (ex, ls, st)), c in zip(curves.items(), (RED, BLUE, GREEN)):
        a0.plot(st, ls, color=c, lw=1.8, label=f"$b={b}$")
        a1.plot(ex, ls, color=c, lw=1.8, label=f"$b={b}$")
    for ax in (a0, a1):
        ax.axhline(MSE_STAR, color=MUTED, lw=1.0, ls=(0, (2, 2)))
        ax.set_yscale("log"); ax.set_xscale("log")
        ax.legend(frameon=False, fontsize=10)
        ax.grid(True, which="both", color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    a0.set_xlabel("число обновлений"); a1.set_xlabel("просмотрено примеров")
    a0.set_ylabel("MSE на всей выборке")
    a0.set_title("По шагам выигрывает малый батч", fontsize=12.5)
    a1.set_title("По прочитанным данным разрыв почти исчезает", fontsize=12.5)
    fig.suptitle("Одна и та же тренировка на двух осях времени (реальный велопрокат)",
                 y=1.03, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "budget.png")
    fin = {b: float(curves[b][1][-1]) for b in curves}
    steps = {b: int(budget // b) for b in curves}
    early = {}
    for b in curves:
        ex, ls, _ = curves[b]
        k = int(np.argmin(np.abs(ex - 1024)))
        early[b] = float(ls[k])
    FACTS.update({f"final_b{b}": fin[b] for b in fin})
    FACTS.update({f"steps_b{b}": steps[b] for b in steps})
    FACTS.update({f"early_b{b}": early[b] for b in early})
    print("budget finals:", {k: round(v, 4) for k, v in fin.items()}, "steps:", steps,
          "at 1024 examples:", {k: round(v, 4) for k, v in early.items()})
    assert steps[8] == 50000 and steps[512] == 781
    assert early[8] < early[512]      # early on, cheap steps win
    assert fin[8] > fin[512]          # late, the quieter estimate finishes closer


# ------------------------------------------------- fig 4: learning-rate regimes
def full_gd(eta, steps):
    theta = np.zeros(D)
    out = [mse(theta)]
    for _ in range(steps):
        theta = theta - eta * grad(theta)
        v = mse(theta)
        out.append(v if np.isfinite(v) and v < 1e12 else 1e12)
    return np.array(out)


def fig_lr_regimes():
    steps = 120
    etas = [(0.5 / L, GREEN, "0{,}5/L — плавно"),
            (1.9 / L, GOLD, "1{,}9/L — качает"),
            (2.05 / L, RED, "2{,}05/L — расходится")]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.2, 4.3))
    for e, c, lab in etas:
        cur = full_gd(e, steps)
        a0.plot(cur, color=c, lw=2.0, label=lab)
    a0.axhline(MSE_STAR, color=MUTED, lw=1.0, ls=(0, (2, 2)))
    a0.set_yscale("log"); a0.set_ylim(0.2, 1e4)
    a0.set_xlabel("шаг полного градиента"); a0.set_ylabel("MSE")
    a0.set_title(f"Порог устойчивости $2/L$ при $L={L:.2f}$", fontsize=12.5)
    a0.legend(frameon=False, fontsize=10)
    # noise floor for constant-step SGD
    floors = []
    grid = [0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6]
    for f in grid:
        th, _, ls = run_sgd(32, f / L, 600_000, seed=99)
        floors.append(float(np.mean(ls[-15:])))
    a1.plot([f / L for f in grid], np.array(floors) - MSE_STAR, "o-", color=VIOLET, lw=2.0, ms=6)
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.axhline(0, color=MUTED, lw=0.8)
    a1.set_xlabel(r"постоянный шаг $\eta$"); a1.set_ylabel(r"избыток MSE над минимумом")
    a1.set_title("Постоянный шаг оставляет пол из шума", fontsize=12.5)
    for ax in (a0, a1):
        ax.grid(True, which="both", color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "lr_regimes.png")
    FACTS["eta_max"] = 2.0 / L
    FACTS["floor_small"] = floors[0]; FACTS["floor_big"] = floors[-1]
    FACTS["floor_ratio"] = (floors[-1] - MSE_STAR) / (floors[0] - MSE_STAR)
    print(f"L={L:.3f} mu={MU:.4f} kappa={L/MU:.1f} 2/L={2/L:.3f} "
          f"floors={[round(f,4) for f in floors]}")
    assert floors[-1] > floors[0] > MSE_STAR
    assert full_gd(2.05 / L, steps)[-1] > 1e3
    assert full_gd(0.5 / L, steps)[-1] < 2 * MSE_STAR


# ------------------------------------------------- fig 5: schedules
def fig_schedules():
    budget = 300_000
    b = 32
    steps = budget // b
    base = 1.2 / L
    scheds = [
        ("постоянный", None, BLUE),
        ("$1/t$", lambda e, t: e / (1 + t / 50.0), RED),
        ("косинус", lambda e, t: e * 0.5 * (1 + np.cos(np.pi * t / steps)), GREEN),
        ("прогрев + косинус",
         lambda e, t: e * min(1.0, (t + 1) / 400.0) * 0.5 * (1 + np.cos(np.pi * t / steps)), GOLD),
    ]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.2, 4.3))
    finals = {}
    for name, dec, c in scheds:
        _, ex, ls = run_sgd(b, base, budget, seed=2024, decay=dec)
        a0.plot(ex, ls, color=c, lw=1.9, label=name)
        finals[name] = float(np.mean(ls[-5:]))
        tt = np.arange(steps)
        a1.plot(tt * b, [base if dec is None else dec(base, int(t)) for t in tt], color=c, lw=1.9)
    a0.axhline(MSE_STAR, color=MUTED, lw=1.0, ls=(0, (2, 2)))
    a0.set_yscale("log"); a0.set_xlabel("просмотрено примеров"); a0.set_ylabel("MSE")
    a0.set_title("Угасание шага снимает дрожание", fontsize=12.5)
    a0.legend(frameon=False, fontsize=10)
    a1.set_xlabel("просмотрено примеров"); a1.set_ylabel(r"шаг $\eta_t$")
    a1.set_title("Сами расписания", fontsize=12.5)
    for ax in (a0, a1):
        ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "schedules.png")
    FACTS["sched_const"] = finals["постоянный"]
    FACTS["sched_cos"] = finals["косинус"]
    FACTS["sched_gain"] = (finals["постоянный"] - MSE_STAR) / (finals["косинус"] - MSE_STAR)
    print("schedules:", {k: round(v, 5) for k, v in finals.items()},
          "gain =", round(FACTS["sched_gain"], 1))
    assert finals["косинус"] < finals["постоянный"]


# ------------------------------------------------- fig 6: early stopping (real, noisy labels)
def early_stopping_run():
    """Small REAL subsample + rich basis + extra label noise: GD overfits, validation turns."""
    rng = np.random.default_rng(4242)
    nsub, ntr, K, steps = 200, 50, 16, 30000
    sub = rng.permutation(N)[:nsub]
    ang = np.arctan2(X[sub, 1], X[sub, 2])          # recover the hour angle
    cols = [np.ones(nsub)]
    for k in range(1, K + 1):
        cols += [np.sin(k * ang), np.cos(k * ang)]
    for c in range(7, D):
        cols.append(X[sub, c])
        for k in (1, 2):
            cols += [X[sub, c] * np.sin(k * ang), X[sub, c] * np.cos(k * ang)]
    F = np.column_stack(cols)
    F = F / (np.linalg.norm(F, axis=0, keepdims=True) / np.sqrt(nsub))
    t = y[sub] + rng.normal(0, 0.9, nsub)           # extra label noise, fixed seed
    Ftr, ttr, Fva, tva = F[:ntr], t[:ntr], F[ntr:], t[ntr:]
    Lf = float(np.linalg.eigvalsh(Ftr.T @ Ftr / ntr)[-1])
    eta = 0.02 / Lf
    theta = np.zeros(F.shape[1])
    tr, va = [], []
    for _ in range(steps):
        tr.append(float(np.mean((Ftr @ theta - ttr) ** 2)))
        va.append(float(np.mean((Fva @ theta - tva) ** 2)))
        theta = theta - eta * (Ftr.T @ (Ftr @ theta - ttr) / ntr)
    # ridge path on the same split, for the "stopping ~ shrinkage" comparison
    G = Ftr.T @ Ftr / ntr
    rhs = Ftr.T @ ttr / ntr
    alphas = np.logspace(-4, 1, 120)
    rv = []
    for a in alphas:
        w = np.linalg.solve(G + a * np.eye(F.shape[1]), rhs)
        rv.append(float(np.mean((Fva @ w - tva) ** 2)))
    rv = np.array(rv)
    ja = int(np.argmin(rv))
    return (np.array(tr), np.array(va), eta,
            {"dim": F.shape[1], "ntr": ntr, "nva": nsub - ntr,
             "ridge_best_val": rv[ja], "ridge_alpha": float(alphas[ja])})


def fig_early_stopping():
    tr, va, eta, info = early_stopping_run()
    best = int(np.argmin(va))
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    it = np.arange(1, len(tr) + 1)
    ax.plot(it, tr, color=BLUE, lw=2.0, label="train MSE")
    ax.plot(it, va, color=RED, lw=2.0, label="validation MSE")
    ax.axvspan(best + 1, len(tr), color=WASH, alpha=0.85, zorder=0)
    ax.axvline(best + 1, color=GOLD, lw=1.8, ls=(0, (4, 3)))
    ax.axhline(info["ridge_best_val"], color=GREEN, lw=1.4, ls=(0, (2, 2)))
    ax.text(len(tr), info["ridge_best_val"] * 0.93, "лучший ridge на том же разбиении",
            ha="right", fontsize=9, color=GREEN)
    ax.annotate(f"минимум validation:\nшаг {best + 1}, MSE {va[best]:.2f}",
                xy=(best + 1, va[best]), xytext=(best * 8.0, va[best] * 0.45),
                fontsize=10, color=GOLD,
                arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=1.2))
    ax.set_xscale("log"); ax.set_xlim(1, len(tr)); ax.set_ylim(0, 3.6)
    ax.set_xlabel("шаг градиентного спуска (лог)"); ax.set_ylabel("MSE")
    ax.set_title("Ранняя остановка на реальных данных с зашумлёнными метками", fontsize=13.5)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.grid(True, which="both", color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "early_stopping.png")
    FACTS["es_dim"] = info["dim"]; FACTS["es_ntr"] = info["ntr"]; FACTS["es_nva"] = info["nva"]
    FACTS["es_best_step"] = best + 1
    FACTS["es_best_val"] = float(va[best])
    FACTS["es_final_val"] = float(va[-1])
    FACTS["es_best_train"] = float(tr[best])
    FACTS["es_final_train"] = float(tr[-1])
    FACTS["es_worse"] = float(va[-1] / va[best])
    FACTS["es_train0"] = float(tr[0])
    FACTS["es_ridge_val"] = info["ridge_best_val"]
    FACTS["es_ridge_alpha"] = info["ridge_alpha"]
    print(f"early stopping: dim={info['dim']} ntr={info['ntr']} best step {best + 1}, "
          f"val {va[best]:.3f} -> final val {va[-1]:.3f} ({va[-1]/va[best]:.2f}x), "
          f"train {tr[best]:.3f} -> {tr[-1]:.3f}; best ridge val {info['ridge_best_val']:.3f} "
          f"at alpha {info['ridge_alpha']:.4f}")
    assert va[-1] > va[best] * 1.5 and tr[-1] < tr[best] / 5
    assert 20 < best + 1 < 500
    assert abs(info["ridge_best_val"] - va[best]) < 0.3 * va[best]
    assert tr[0] > tr[best] > tr[-1]          # train falls monotonically over the run


# ------------------------------------------------- fig 7: filter factors, stopping vs ridge
def fig_filter():
    lam = np.logspace(-3, 0.25, 300)
    eta = 0.5
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for t, c in [(10, BLUE), (50, GREEN), (200, RED)]:
        f_gd = 1 - (1 - eta * lam) ** t
        alpha = 1.0 / (eta * t)
        f_r = lam / (lam + alpha)
        ax.plot(lam, f_gd, color=c, lw=2.1, label=f"спуск, {t} шагов")
        ax.plot(lam, f_r, color=c, lw=1.4, ls=(0, (4, 3)),
                label=f"ridge, $\\alpha=1/(\\eta t)={alpha:.3f}$")
    ax.set_xscale("log"); ax.set_ylim(-0.05, 1.1)
    ax.axhline(1, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    ax.set_xlabel(r"собственное число $\lambda$ направления")
    ax.set_ylabel("доля выученного сигнала")
    ax.set_title("Ранняя остановка гасит те же направления, что и ridge", fontsize=13.5)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax.grid(True, which="both", color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "filter_factors.png")
    lam0, t = 0.02, 50
    FACTS["filt_gd"] = float(1 - (1 - eta * lam0) ** t)
    FACTS["filt_ridge"] = float(lam0 / (lam0 + 1 / (eta * t)))
    dev = float(np.max(np.abs((1 - (1 - eta * lam) ** t) - lam / (lam + 1 / (eta * t)))))
    FACTS["filt_maxdev"] = dev
    lam_fast, lam_slow = 1.5, 0.001
    FACTS["filt_fast"] = float(1 - (1 - eta * lam_fast) ** t)
    FACTS["filt_slow"] = float(1 - (1 - eta * lam_slow) ** t)
    print(f"filter at lambda=0.02, t=50: gd={FACTS['filt_gd']:.3f} ridge={FACTS['filt_ridge']:.3f}, "
          f"max deviation over the grid {dev:.3f}; lambda=1.5 -> {FACTS['filt_fast']:.4f}, "
          f"lambda=0.001 -> {FACTS['filt_slow']:.4f}")
    assert dev < 0.25 and FACTS["filt_fast"] > 0.999 and FACTS["filt_slow"] < 0.03


# ------------------------------------------------- margins
def side_sqrt():
    b = np.logspace(0, 3, 200)
    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    ax.plot(b, 1 / np.sqrt(b), color=BLUE, lw=2.0, label="шум $1/\\sqrt{b}$")
    ax.plot(b, 1 / b, color=RED, lw=1.6, ls=(0, (4, 3)), label="если бы $1/b$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("батч $b$", fontsize=9); ax.set_ylabel("шум", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("вчетверо дороже — вдвое тише", fontsize=9)
    ax.grid(True, which="both", color=GRID, lw=0.4, alpha=0.4)
    save(fig, SIDE / "sqrt_cost.png")


def side_shuffle():
    rng = np.random.default_rng(3)
    lab = np.array([0] * 60 + [1] * 30 + [2] * 30)
    sh = rng.permutation(lab)
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(4.2, 2.8), sharex=True)
    for ax, arr, ttl in ((a0, lab, "отсортировано"), (a1, sh, "перемешано")):
        for i, v in enumerate(arr):
            ax.bar(i, 1, width=1.0, color=[BLUE, GOLD, RED][v], linewidth=0)
        ax.set_yticks([]); ax.set_title(ttl, fontsize=8.5)
        for k in range(0, len(arr) + 1, 20):
            ax.axvline(k - 0.5, color=INK, lw=0.7)
    a1.set_xlabel("объекты, батчи по 20", fontsize=8)
    fig.suptitle("состав батча", y=1.02, fontsize=9)
    fig.tight_layout()
    save(fig, SIDE / "shuffle.png")


def side_rare():
    ham = spam = 0
    with open(SMS, encoding="utf8", errors="replace") as f:
        for row in f:
            if row.startswith("spam\t"):
                spam += 1
            elif row.startswith("ham\t"):
                ham += 1
    p = spam / (spam + ham)
    FACTS["sms_total"] = spam + ham
    FACTS["sms_spam"] = spam
    FACTS["sms_rate"] = p
    b = np.arange(1, 65)
    miss = (1 - p) ** b
    FACTS["miss_b8"] = float((1 - p) ** 8)
    FACTS["miss_b32"] = float((1 - p) ** 32)
    print(f"sms: {spam}/{spam+ham} = {p:.3f}; miss(8)={FACTS['miss_b8']:.3f}, "
          f"miss(32)={FACTS['miss_b32']:.4f}")
    assert 0.12 < p < 0.15
    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    ax.plot(b, miss, color=RED, lw=2.0)
    ax.plot([8], [FACTS["miss_b8"]], "o", color=INK, ms=6)
    ax.annotate(f"b=8: {FACTS['miss_b8']:.2f}", (8, FACTS["miss_b8"]),
                xytext=(14, 0.6), fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9))
    ax.set_xlabel("размер батча", fontsize=9)
    ax.set_ylabel("P(нет ни одного спама)", fontsize=8.5)
    ax.set_title(f"доля спама {p:.3f} (реальные SMS)", fontsize=9)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4)
    save(fig, SIDE / "rare_miss.png")


def side_accum():
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    ax.axis("off")
    for k in range(4):
        ax.add_patch(plt.Rectangle((k * 1.1, 1.2), 0.9, 0.6, color=BLUE, alpha=0.75))
        ax.text(k * 1.1 + 0.45, 1.5, f"µ{k+1}", ha="center", va="center",
                color=PAPER, fontsize=9)
        ax.annotate("", xy=(2.2, 0.75), xytext=(k * 1.1 + 0.45, 1.15),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0))
    ax.add_patch(plt.Rectangle((1.0, 0.15), 2.4, 0.55, color=GOLD, alpha=0.85))
    ax.text(2.2, 0.42, "одно обновление", ha="center", va="center", color=PAPER, fontsize=9)
    ax.set_xlim(-0.2, 4.6); ax.set_ylim(0, 2.1)
    ax.set_title("4 микробатча = эффективный батч", fontsize=9)
    save(fig, SIDE / "accum.png")


fig_gradient_cloud()
fig_noise_law()
fig_budget()
fig_lr_regimes()
fig_schedules()
fig_early_stopping()
fig_filter()
side_sqrt()
side_shuffle()
side_rare()
side_accum()

print("\n--- FACTS quoted in lesson 61 ---")
for k, v in FACTS.items():
    print(f"{k:16s} {v}")
print("lesson 61 figures written")
