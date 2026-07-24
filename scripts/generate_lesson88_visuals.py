"""Deterministic figures for lesson 88: video, 3D and world models.

Every number quoted in the lesson text is computed here and asserted.

Real data: MTA daily ridership (scripts/data/mta-daily-ridership.csv) -- an honest
autoregressive rollout showing that a tiny one-step error becomes a large multi-step one.
Everything else is an explicitly declared synthetic model with a fixed seed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MTA = ROOT / "scripts" / "data" / "mta-daily-ridership.csv"
OUT = ROOT / "public" / "figures" / "lessons" / "88"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "88"

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


def grid(ax):
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5)
    ax.set_axisbelow(True)


# ============================================================ fig 88.1 (real data)
def fig_rollout_real():
    dates, sub = [], []
    with open(MTA) as f:
        for row in csv.DictReader(f):
            dates.append(row["date"][:10])
            sub.append(float(row["subways_total_estimated_ridership"]))
    y = np.array(sub) / 1e6                      # millions of rides
    n = len(y)
    n_train = 1200
    L = 7                                        # lags 1..7 -- one week

    def design(series, idx):
        X = np.stack([series[idx - k] for k in range(1, L + 1)] + [np.ones(len(idx))], axis=1)
        return X

    tr = np.arange(L, n_train)
    Xtr, ytr = design(y, tr), y[tr]
    w, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)

    te = np.arange(n_train, n)
    pred1 = design(y, te) @ w
    rmse1 = float(np.sqrt(np.mean((pred1 - y[te]) ** 2)))

    H = 30
    starts = np.arange(n_train, n - H)
    errs = np.zeros((len(starts), H))
    for i, s in enumerate(starts):
        hist = list(y[s - L:s])
        for h in range(H):
            x = np.array(hist[-1:-L - 1:-1] + [1.0])
            nxt = float(x @ w)
            hist.append(nxt)
            errs[i, h] = nxt - y[s + h]
    rmse_h = np.sqrt(np.mean(errs ** 2, axis=0))
    bias_h = errs.mean(axis=0)

    FACTS["mta_days"] = n
    FACTS["mta_mean"] = float(y.mean())
    FACTS["rmse1"] = rmse1
    FACTS["rmse7"] = float(rmse_h[6])
    FACTS["rmse30"] = float(rmse_h[29])
    FACTS["ratio30"] = float(rmse_h[29] / rmse1)
    FACTS["bias30"] = float(bias_h[29])
    FACTS["bias1"] = float(bias_h[0])
    FACTS["rel1"] = float(rmse1 / y[te].mean() * 100)
    FACTS["rel30"] = float(rmse_h[29] / y[te].mean() * 100)
    print("MTA n=%d mean=%.3f  rmse1=%.4f rmse7=%.4f rmse30=%.4f ratio=%.2f bias30=%+.3f"
          % (n, y.mean(), rmse1, rmse_h[6], rmse_h[29], FACTS["ratio30"], bias_h[29]))
    assert n == 1776
    assert rmse1 < rmse_h[6] < rmse_h[29]
    assert FACTS["ratio30"] > 1.4

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
    s0 = int(starts[len(starts) // 2])
    hist = list(y[s0 - L:s0])
    roll = []
    for h in range(H):
        x = np.array(hist[-1:-L - 1:-1] + [1.0])
        hist.append(float(x @ w)); roll.append(hist[-1])
    tf = design(y, np.arange(s0, s0 + H)) @ w
    t = np.arange(H)
    ax = axes[0]
    ax.plot(t, y[s0:s0 + H], color=INK, lw=2.0, label="реальные поездки")
    ax.plot(t, tf, color=BLUE, lw=1.8, ls=(0, (4, 3)), label="прогноз на 1 шаг (teacher forcing)")
    ax.plot(t, roll, color=RED, lw=2.2, label="свободный rollout на 30 шагов")
    ax.set_xlabel("шаг горизонта, сутки"); ax.set_ylabel("поездки, млн")
    ax.set_title("Один прогноз против цепочки", fontsize=12.5)
    ax.legend(frameon=False, fontsize=9.5, loc="lower left"); grid(ax)

    ax = axes[1]
    ax.plot(np.arange(1, H + 1), rmse_h, color=RED, lw=2.4, label="RMSE свободного rollout")
    ax.axhline(rmse1, color=BLUE, lw=1.8, ls=(0, (5, 3)), label="RMSE одного шага")
    ax.scatter([1, 7, 30], [rmse_h[0], rmse_h[6], rmse_h[29]], s=45, color=INK, zorder=5)
    for h, v in [(1, rmse_h[0]), (7, rmse_h[6]), (30, rmse_h[29])]:
        ax.annotate(f"{v:.2f}", (h, v), textcoords="offset points", xytext=(6, 7),
                    fontsize=10, color=INK)
    ax.set_xlabel("горизонт $h$, шагов"); ax.set_ylabel("RMSE, млн поездок")
    ax.set_title("Ошибка растёт с горизонтом", fontsize=12.5)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left"); grid(ax)
    fig.suptitle("Метрика одного шага не измеряет качество долгого прогноза (реальные данные MTA)",
                 y=1.02, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "rollout_real.png")


# ============================================================ fig 88.2 (fork / blur)
def fig_fork_blur():
    rng = np.random.default_rng(2026)
    N = 20000
    side = rng.integers(0, 2, N) * 2 - 1            # -1 left, +1 right
    ang = np.deg2rad(28) * side + rng.normal(0, np.deg2rad(2.5), N)
    R = 1.0 + rng.normal(0, 0.02, N)
    xs, ys = R * np.sin(ang), R * np.cos(ang)
    mx, my = float(xs.mean()), float(ys.mean())
    mode_x = float(xs[side == 1].mean())
    d_mean = np.hypot(xs - mx, ys - my)
    d_mode = np.hypot(xs - mode_x, ys - float(ys[side == 1].mean()))
    mse_mean = float(np.mean(d_mean ** 2))
    mse_mode = float(np.mean(d_mode ** 2))
    frac = float(np.mean(d_mean < 0.10))
    FACTS["fork_mean_x"] = mx
    FACTS["fork_mode_x"] = mode_x
    FACTS["fork_mse_mean"] = mse_mean
    FACTS["fork_mse_mode"] = mse_mode
    FACTS["fork_frac"] = frac * 100
    FACTS["fork_ratio"] = mse_mode / mse_mean
    print("fork: mean_x=%.4f mode_x=%.3f mse_mean=%.4f mse_mode=%.4f frac=%.3f%%"
          % (mx, mode_x, mse_mean, mse_mode, frac * 100))
    assert abs(mx) < 0.02 and mse_mean < mse_mode and frac < 0.02

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    ax = axes[0]
    ax.plot([0, 0], [-0.55, 0], color=MUTED, lw=2.0)
    for s, c in [(-1, BLUE), (1, GREEN)]:
        a = np.deg2rad(28) * s
        ax.plot([0, np.sin(a)], [0, np.cos(a)], color=c, lw=2.2)
    ax.scatter([mx], [my], s=120, color=RED, marker="X", zorder=6)
    ax.annotate("прогноз MSE", (mx, my), textcoords="offset points", xytext=(10, -4),
                fontsize=10, color=RED)
    ax.scatter(xs[:300], ys[:300], s=6, color=INK, alpha=0.35)
    ax.set_title("Две одинаково правдоподобные ветви", fontsize=12)
    ax.set_xlim(-0.85, 0.85); ax.set_ylim(-0.6, 1.15)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); grid(ax)

    ax = axes[1]
    ax.hist(xs, bins=90, color=BLUE, alpha=0.75)
    ax.axvline(mx, color=RED, lw=2.2)
    ax.annotate(f"среднее $x={mx:.3f}$\nздесь почти нет исходов",
                (mx, ax.get_ylim()[1] * 0.72), textcoords="offset points", xytext=(8, 0),
                fontsize=10, color=RED)
    ax.set_xlabel("$x$ через $T$ кадров"); ax.set_ylabel("число траекторий")
    ax.set_title("Плотность будущего двугорбая", fontsize=12); grid(ax)

    ax = axes[2]
    gx = np.linspace(-0.8, 0.8, 240); gy = np.linspace(0.4, 1.2, 120)
    GX, GY = np.meshgrid(gx, gy)
    blob = lambda cx, cy: np.exp(-((GX - cx) ** 2 + (GY - cy) ** 2) / (2 * 0.045 ** 2))
    img = 0.5 * blob(np.sin(np.deg2rad(28)), np.cos(np.deg2rad(28))) \
        + 0.5 * blob(-np.sin(np.deg2rad(28)), np.cos(np.deg2rad(28)))
    ax.imshow(img, extent=(gx[0], gx[-1], gy[0], gy[-1]), origin="lower",
              cmap="bone_r", vmin=0, vmax=1, aspect="auto")
    ax.set_title("Тот же прогноз в пикселях: два призрака", fontsize=12)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    fig.suptitle("Усреднение по будущему минимизирует MSE и при этом невозможно физически",
                 y=1.02, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "fork_blur.png")


# ============================================================ fig 88.3 (pinhole + epipolar)
def fig_geometry():
    fx = fy = 800.0; cx, cy = 320.0, 240.0
    pts = np.array([[1, 0, 5], [0, 1, 10], [-2, -1, 4], [1, 0, 10]], float)
    proj = np.stack([fx * pts[:, 0] / pts[:, 2] + cx, fy * pts[:, 1] / pts[:, 2] + cy], axis=1)
    FACTS["proj_a"] = tuple(proj[0]); FACTS["proj_b"] = tuple(proj[1])
    FACTS["proj_c"] = tuple(proj[2]); FACTS["proj_d"] = tuple(proj[3])
    print("projections:", proj.tolist())
    assert np.allclose(proj[0], [480, 240]) and np.allclose(proj[1], [320, 320])
    assert np.allclose(proj[2], [-80, 40]) and np.allclose(proj[3], [400, 240])

    # two cameras, baseline along x; epipolar residual of a wrong match
    B = 0.6
    X = np.array([0.35, 0.10, 4.0])
    u1 = np.array([fx * X[0] / X[2] + cx, fy * X[1] / X[2] + cy])
    u2 = np.array([fx * (X[0] - B) / X[2] + cx, fy * X[1] / X[2] + cy])
    disp = u1[0] - u2[0]
    Zrec = fx * B / disp
    wrong = u2 + np.array([0.0, 26.0])
    resid = abs(wrong[1] - u2[1])
    FACTS["disp"] = float(disp); FACTS["Zrec"] = float(Zrec)
    FACTS["epi_resid"] = float(resid)
    FACTS["baseline"] = B
    print("disparity=%.1f px, Z=%.3f m, epi residual=%.0f px" % (disp, Zrec, resid))
    assert abs(Zrec - 4.0) < 1e-6

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    ax = axes[0]
    ax.axhline(0, color=GRID, lw=1)
    ax.scatter([0, B], [0, 0], s=70, color=INK, zorder=5)
    ax.text(0, -0.28, "камера 1", ha="center", fontsize=10, color=MUTED)
    ax.text(B, -0.28, "камера 2", ha="center", fontsize=10, color=MUTED)
    ax.scatter([X[0]], [X[2]], s=90, color=RED, zorder=6)
    ax.text(X[0] + 0.06, X[2], "$X$", fontsize=12, color=RED)
    for c, col in [(0.0, BLUE), (B, GREEN)]:
        ax.plot([c, X[0] + (X[0] - c) * 0.28], [0, X[2] * 1.28], color=col, lw=1.8)
    ax.plot([0, B], [0, 0], color=GOLD, lw=2.4)
    ax.text(B / 2, 0.16, f"база {B} м", ha="center", fontsize=10, color=GOLD)
    ax.set_xlim(-0.6, 1.1); ax.set_ylim(-0.5, 5.4)
    ax.set_xlabel("$X$, м"); ax.set_ylabel("глубина $Z$, м")
    ax.set_title("Один пиксель задаёт луч, два — точку", fontsize=12.5); grid(ax)

    ax = axes[1]
    ax.add_patch(plt.Rectangle((0, 0), 640, 480, fc=WASH, ec=LINE))
    ax.axhline(u2[1], color=GREEN, lw=2.0)
    ax.text(20, u2[1] + 10, "эпиполярная линия", fontsize=10, color=GREEN)
    ax.scatter([u2[0]], [u2[1]], s=80, color=BLUE, zorder=5)
    ax.text(u2[0] + 12, u2[1] - 18, "верное соответствие", fontsize=10, color=BLUE)
    ax.scatter([wrong[0]], [wrong[1]], s=80, color=RED, marker="X", zorder=5)
    ax.plot([wrong[0], u2[0]], [wrong[1], u2[1]], color=RED, lw=1.4, ls=(0, (3, 3)))
    ax.text(wrong[0] + 12, wrong[1] + 6, f"невязка {resid:.0f} px", fontsize=10, color=RED)
    ax.set_xlim(0, 640); ax.set_ylim(480, 0)
    ax.set_xlabel("$u$, пиксели"); ax.set_ylabel("$v$, пиксели")
    ax.set_title("Проверка второго кадра: линия, а не вся картинка", fontsize=12.5)
    fig.suptitle("Геометрия даёт измеримый инвариант новому ракурсу", y=1.02, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "geometry.png")


# ============================================================ fig 88.4 (volume rendering)
def fig_volume():
    dt = 0.05
    t = np.arange(0, 3.0, dt) + dt / 2
    sig = np.where((t > 1.5) & (t < 1.7), 60.0, 0.02)      # surface at 1.5..1.7 m
    col = np.clip(0.25 + 0.25 * t / 3, 0, 1)
    col[(t > 1.5) & (t < 1.7)] = 0.88                      # bright surface
    alpha = 1 - np.exp(-sig * dt)
    T = np.concatenate([[1.0], np.cumprod(1 - alpha)[:-1]])
    w = T * alpha
    C = float((w * col).sum())
    Tend = float(np.prod(1 - alpha))
    kmax = int(np.argmax(w))
    FACTS["vr_alpha_surface"] = float(alpha[kmax])
    FACTS["vr_wmax"] = float(w.max())
    FACTS["vr_tmax"] = float(t[kmax])
    FACTS["vr_wsum"] = float(w.sum())
    FACTS["vr_C"] = C
    FACTS["vr_Tend"] = Tend

    sig4 = sig.copy(); sig4[(t > 1.5) & (t < 1.7)] = 240.0
    a4 = 1 - np.exp(-sig4 * dt)
    T4 = np.concatenate([[1.0], np.cumprod(1 - a4)[:-1]])
    C4 = float(((T4 * a4) * col).sum())
    FACTS["vr_C4"] = C4
    FACTS["vr_dC"] = abs(C4 - C)
    print("volume: alpha=%.3f wmax=%.3f at %.2f, sum w=%.4f, C=%.4f, Tend=%.2e, dC=%.4f"
          % (alpha[kmax], w.max(), t[kmax], w.sum(), C, Tend, abs(C4 - C)))
    assert w.sum() + Tend > 0.999 and abs(C4 - C) < 0.02

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))
    for ax, val, ttl, col_ in [
            (axes[0], sig, r"плотность $\sigma(t)$", BLUE),
            (axes[1], T, r"пропускание $T(t)$", GOLD),
            (axes[2], w, r"вес $w_i=T_i\alpha_i$", RED)]:
        ax.plot(t, val, color=col_, lw=2.2)
        ax.fill_between(t, 0, val, color=col_, alpha=0.12)
        ax.set_xlabel("$t$ вдоль луча, м"); ax.set_title(ttl, fontsize=12.5); grid(ax)
    axes[2].annotate(f"пик {w.max():.2f} при $t={t[kmax]:.2f}$",
                     (t[kmax], w.max()), textcoords="offset points", xytext=(14, -6),
                     fontsize=10, color=RED)
    axes[1].annotate(f"после поверхности $T={Tend:.0e}$", (2.4, 0.35),
                     fontsize=10, color=MUTED)
    fig.suptitle(f"Луч возвращает $C={C:.3f}$: почти весь вес забирает первая непрозрачность",
                 y=1.03, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "volume.png")


# ============================================================ fig 88.5 (intervention + exploitation)
def fig_world_model():
    rng = np.random.default_rng(88)
    # (a) same prefix, three interventions
    dt = 0.1; H = 26
    def roll(a, noise):
        p = np.array([0.0, 0.0]); th = np.pi / 2; v = 1.0
        pts = [p.copy()]
        for k in range(H):
            th += a * dt + noise * rng.normal(0, 0.05)
            p = p + v * dt * np.array([np.cos(th), np.sin(th)])
            pts.append(p.copy())
        return np.array(pts)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    ax = axes[0]
    spread = {}
    for a, c, name in [(-1.2, BLUE, "руль вправо"), (0.0, MUTED, "прямо"), (1.2, GREEN, "руль влево")]:
        ens = np.array([roll(a, 1.0) for _ in range(24)])
        for e in ens:
            ax.plot(e[:, 0], e[:, 1], color=c, lw=0.7, alpha=0.25)
        m = ens.mean(axis=0)
        ax.plot(m[:, 0], m[:, 1], color=c, lw=2.4, label=name)
        spread[a] = float(np.mean(np.linalg.norm(ens[:, -1] - m[-1], axis=1)))
    ax.plot([0, 0], [-0.9, 0], color=INK, lw=3.0)
    ax.text(0.02, -0.8, "общий префикс наблюдений", fontsize=10, color=INK)
    ax.set_xlabel("$x$, м"); ax.set_ylabel("$y$, м")
    ax.set_title("Один префикс, три вмешательства", fontsize=12.5)
    ax.legend(frameon=False, fontsize=10, loc="upper left"); grid(ax)
    sep = float(np.linalg.norm(roll(1.2, 0)[-1] - roll(-1.2, 0)[-1]))
    FACTS["branch_sep"] = sep
    FACTS["branch_spread"] = spread[0.0]

    # (b) model exploitation outside the support of behaviour data
    def true_r(a):
        return 1.0 - 2.2 * (a - 0.15) ** 2
    a_obs = rng.uniform(-0.3, 0.3, 220)
    r_obs = true_r(a_obs) + rng.normal(0, 0.02, len(a_obs))
    grid_a = np.linspace(-1, 1, 401)
    preds = []
    for _ in range(12):
        idx = rng.integers(0, len(a_obs), len(a_obs))
        c = np.polyfit(a_obs[idx], r_obs[idx], 5)
        preds.append(np.polyval(c, grid_a))
    P = np.array(preds); mu = P.mean(axis=0); sd = P.std(axis=0)
    k = int(np.argmax(mu))
    a_plan, r_pred, r_true = float(grid_a[k]), float(mu[k]), float(true_r(grid_a[k]))
    lam = 1.0
    kp = int(np.argmax(mu - lam * sd))
    a_pess, r_pess_true = float(grid_a[kp]), float(true_r(grid_a[kp]))
    FACTS.update(a_plan=a_plan, r_pred=r_pred, r_true=r_true,
                 a_pess=a_pess, r_pess_true=r_pess_true,
                 gap=r_pred - r_true, sd_edge=float(sd[k]),
                 a_star=0.15, r_star=float(true_r(0.15)))
    print("exploit: a_plan=%.2f pred=%.2f true=%.2f gap=%.2f | pess a=%.2f true=%.2f sd_edge=%.2f"
          % (a_plan, r_pred, r_true, r_pred - r_true, a_pess, r_pess_true, sd[k]))
    assert abs(a_plan) > 0.9 and r_pred - r_true > 1.0 and abs(a_pess) <= 0.45

    ax = axes[1]
    ax.axvspan(-0.3, 0.3, color=WASH, label="диапазон данных")
    ax.plot(grid_a, true_r(grid_a), color=INK, lw=2.2, label="истинная награда")
    ax.plot(grid_a, mu, color=RED, lw=2.2, label="модель (среднее ансамбля)")
    ax.fill_between(grid_a, mu - sd, mu + sd, color=RED, alpha=0.15)
    ax.plot(grid_a, mu - lam * sd, color=GREEN, lw=1.8, ls=(0, (5, 3)),
            label=r"пессимизм $\mu-\sigma$")
    ax.scatter([a_plan], [r_pred], s=80, color=RED, zorder=6)
    ax.scatter([a_plan], [r_true], s=80, color=INK, marker="X", zorder=6)
    ax.annotate(f"обещано {r_pred:.2f}\nна деле {r_true:.2f}", (a_plan, r_pred),
                textcoords="offset points", xytext=(-118, -6), fontsize=10, color=RED)
    ax.scatter([a_pess], [true_r(a_pess)], s=80, color=GREEN, zorder=6)
    ax.set_ylim(-3.0, 6.0)
    ax.set_xlabel("действие $a$ (поворот руля)"); ax.set_ylabel("награда")
    ax.set_title("Планировщик уходит туда, где модель не проверяли", fontsize=12.5)
    ax.legend(frameon=False, fontsize=9.5, loc="lower center"); grid(ax)
    fig.suptitle("Модель мира отвечает на действие — и ошибается вне опоры данных",
                 y=1.02, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "world_model.png")


# ============================================================ fig 88.6 (physics residual)
def fig_physics():
    m1, m2 = 1.0, 2.0
    u1, u2 = 3.0, -1.0
    p0 = m1 * u1 + m2 * u2
    E0 = 0.5 * m1 * u1 ** 2 + 0.5 * m2 * u2 ** 2

    def after(e):
        v1 = (m1 * u1 + m2 * u2 - m2 * e * (u1 - u2)) / (m1 + m2)
        v2 = (m1 * u1 + m2 * u2 + m1 * e * (u1 - u2)) / (m1 + m2)
        return v1, v2

    rows = []
    for e in (1.0, 0.6, 0.0):
        v1, v2 = after(e)
        p1 = m1 * v1 + m2 * v2
        E1 = 0.5 * m1 * v1 ** 2 + 0.5 * m2 * v2 ** 2
        rows.append((e, v1, v2, p1 - p0, E1, E0 - E1))
    FACTS["p0"] = p0; FACTS["E0"] = E0
    FACTS["v_el"] = rows[0][1:3]; FACTS["E_e06"] = rows[1][4]
    FACTS["dE_e06"] = rows[1][5]; FACTS["dE_e06_pct"] = rows[1][5] / E0 * 100
    FACTS["v_e06"] = rows[1][1:3]
    print("collision:", [(f"{r[0]}", round(r[1], 3), round(r[2], 3), round(r[3], 12),
                          round(r[4], 3), round(r[5], 3)) for r in rows])
    assert abs(rows[1][3]) < 1e-12 and rows[1][5] > 0

    # estimate restitution from noisy frames of a bouncing ball
    rng = np.random.default_rng(2026)
    dt = 1 / 30; g = 9.81; e_true = 0.78
    def bounce_series(sigma, trials=400):
        """Three frames before and after the bounce; gravity removed, slope = velocity."""
        tt = np.arange(1, 4) * dt
        vb, va = 4.0, e_true * 4.0
        est = []
        for _ in range(trials):
            y_pre = -vb * (-tt) - 0.5 * g * tt ** 2 + rng.normal(0, sigma, 3)
            y_post = va * tt - 0.5 * g * tt ** 2 + rng.normal(0, sigma, 3)
            z_pre = y_pre + 0.5 * g * tt ** 2
            z_post = y_post + 0.5 * g * tt ** 2
            sp = np.polyfit(-tt, z_pre, 1)[0]
            sa = np.polyfit(tt, z_post, 1)[0]
            est.append(abs(sa / sp))
        return np.array(est)
    e_clean = bounce_series(0.0)
    e_noisy = bounce_series(0.004)
    FACTS["e_true"] = e_true
    FACTS["e_clean"] = float(e_clean.mean())
    FACTS["e_noisy_mean"] = float(e_noisy.mean())
    FACTS["e_noisy_sd"] = float(e_noisy.std())
    print("restitution: clean=%.3f noisy=%.3f +- %.3f" % (e_clean.mean(), e_noisy.mean(), e_noisy.std()))
    assert abs(e_clean.mean() - e_true) < 1e-9 and e_noisy.std() > 0.02

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
    ax = axes[0]
    labels = ["упругий\n$e=1$", "частично упругий\n$e=0{,}6$", "слипание\n$e=0$"]
    xpos = np.arange(3)
    ax.bar(xpos - 0.18, [0, 0, 0], width=0.34, color=BLUE, label="невязка импульса")
    ax.bar(xpos + 0.18, [r[5] for r in rows], width=0.34, color=RED, label="потеря энергии, Дж")
    for i, r in enumerate(rows):
        ax.text(i + 0.18, r[5] + 0.12, f"{r[5]:.2f}", ha="center", fontsize=10, color=RED)
        ax.text(i - 0.18, 0.12, "0", ha="center", fontsize=10, color=BLUE)
    ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Дж (и кг·м/с)")
    ax.set_title("Импульс сохраняется всегда, энергия — нет", fontsize=12.5)
    ax.legend(frameon=False, fontsize=10); grid(ax)

    ax = axes[1]
    ax.hist(e_noisy, bins=40, color=GOLD, alpha=0.8)
    ax.axvline(e_true, color=INK, lw=2.2)
    ax.annotate(f"истинное $e={e_true}$", (e_true, ax.get_ylim()[1] * 0.85),
                textcoords="offset points", xytext=(8, 0), fontsize=10, color=INK)
    ax.set_xlabel("оценка $e$ по трём кадрам с шумом 4 мм")
    ax.set_ylabel("число испытаний")
    ax.set_title(f"Разброс оценки: $\\pm${e_noisy.std():.2f}", fontsize=12.5); grid(ax)
    fig.suptitle("Физику проверяют законами сохранения, а не ощущением реализма",
                 y=1.02, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "physics.png")


# ============================================================ sidenote figures
def side_teleport():
    t = np.linspace(0, 1.2, 25)
    y = 3.0 * t - 4.9 * t ** 2 / 2 + 1.0
    y2 = y.copy(); y2[13:] += 0.42
    r1 = y - np.polyval(np.polyfit(t, y, 2), t)
    r2 = y2 - np.polyval(np.polyfit(t, y2, 2), t)
    FACTS["res_ok"] = float(np.abs(r1).max())
    FACTS["res_jump"] = float(np.abs(r2).max())
    assert FACTS["res_jump"] > 20 * max(FACTS["res_ok"], 1e-9)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(t, y, color=BLUE, lw=2.0, marker="o", ms=3, label="согласованная")
    ax.plot(t, y2, color=RED, lw=2.0, marker="o", ms=3, label="со скачком")
    ax.set_xlabel("время, с"); ax.set_ylabel("высота центра, м")
    ax.legend(frameon=False, fontsize=9); grid(ax)
    ax.set_title("Каждый кадр резок, движение — нет", fontsize=11)
    fig.tight_layout(); save(fig, SIDE / "teleport.png")


def side_occlusion():
    k = np.arange(0, 31)
    sd = 0.06 * np.sqrt(k) + 0.012 * k
    FACTS["occl_sd10"] = float(sd[10]); FACTS["occl_sd30"] = float(sd[30])
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.fill_between(k, -2 * sd, 2 * sd, color=VIOLET, alpha=0.18)
    ax.plot(k, 2 * sd, color=VIOLET, lw=2.0)
    ax.plot(k, -2 * sd, color=VIOLET, lw=2.0)
    ax.axhline(0, color=INK, lw=1.2, ls=(0, (4, 3)))
    ax.set_xlabel("кадров под окклюзией"); ax.set_ylabel("положение, м")
    ax.set_title("Неопределённость обязана расти", fontsize=11); grid(ax)
    fig.tight_layout(); save(fig, SIDE / "occlusion.png")


def side_loop():
    ang = np.linspace(0, 2 * np.pi, 200)
    drift = 0.045 * ang / (2 * np.pi)
    FACTS["loop_drift"] = float(drift[-1] * 100)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    r = 1 + drift
    ax.plot(np.cos(ang), np.sin(ang), color=GRID, lw=1.6)
    ax.plot(r * np.cos(ang), r * np.sin(ang), color=GOLD, lw=2.2)
    ax.scatter([1], [0], s=50, color=BLUE, zorder=5)
    ax.scatter([r[-1]], [0], s=50, color=RED, zorder=5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Круг камер: возврат с невязкой", fontsize=11)
    fig.tight_layout(); save(fig, SIDE / "loop.png")


def side_depth():
    Z = np.linspace(2, 20, 200)
    h = 800 * 1.8 / Z
    FACTS["h5"] = float(800 * 1.8 / 5); FACTS["h10"] = float(800 * 1.8 / 10)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(Z, h, color=GREEN, lw=2.2)
    ax.scatter([5, 10], [FACTS["h5"], FACTS["h10"]], s=45, color=INK, zorder=5)
    ax.annotate(f"{FACTS['h5']:.0f} px", (5, FACTS["h5"]), textcoords="offset points",
                xytext=(8, 4), fontsize=9.5)
    ax.annotate(f"{FACTS['h10']:.0f} px", (10, FACTS["h10"]), textcoords="offset points",
                xytext=(8, 6), fontsize=9.5)
    ax.set_xlabel("глубина $Z$, м"); ax.set_ylabel("высота в кадре, px")
    ax.set_title("Размер падает как $1/Z$", fontsize=11); grid(ax)
    fig.tight_layout(); save(fig, SIDE / "depth.png")



# ============================================================ widget consistency check
def widget_check():
    """Числа, которые урок цитирует по лаборатории world-rollout-lab."""
    import math
    DT, G, E = 1 / 30, 9.81, 0.78

    def step(s, g):
        vy = s[3] - g * DT
        x, y, vx = s[0] + s[2] * DT, s[1] + vy * DT, s[2]
        if y < 0:
            y, vy = -y * E, -vy * E
        if x > 4.6:
            x, vx = 9.2 - x, -vx
        if x < 0:
            x, vx = -x, -vx
        return [x, y, vx, vy]

    truth = [0.2, 2.2, 1.6, 0.6]
    model = list(truth)
    hit = None
    tf = []
    for i in range(1, 181):
        one = step(truth, G * 1.04)
        nxt = step(truth, G)
        tf.append(math.hypot(one[0] - nxt[0], one[1] - nxt[1]))
        truth = nxt
        model = step(model, G * 1.04)
        err = math.hypot(model[0] - truth[0], model[1] - truth[1])
        if hit is None and err >= 0.2:
            hit = i
    one_step = sum(tf[:90]) / 90
    FACTS["widget_one_step"] = one_step
    FACTS["widget_hit02"] = hit
    FACTS["widget_hit02_sec"] = hit * DT
    print("widget: one-step=%.5f m, 0.2 m reached at step %d (%.2f s)" % (one_step, hit, hit * DT))
    assert hit == 94 and round(one_step, 4) == 0.0004


def main():
    fig_rollout_real()
    fig_fork_blur()
    fig_geometry()
    fig_volume()
    fig_world_model()
    fig_physics()
    side_teleport(); side_occlusion(); side_loop(); side_depth()
    widget_check()
    out = {k: (list(v) if isinstance(v, tuple) else v) for k, v in FACTS.items()}
    (ROOT / "scripts" / "data" / "lesson88_facts.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    print("\n--- facts ---")
    for k, v in out.items():
        print(k, v)
    print("OK")


if __name__ == "__main__":
    main()
