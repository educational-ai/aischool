"""Deterministic figures for lesson 39: PCA, embeddings and recommendations.

A biased low-rank matrix factorization fitted on the REAL MovieLens 100K ratings:
one prediction decomposed into global mean + biases + dot product, the 2D taste map
of real movies, and test RMSE against the number of latent factors with a head/tail
split. Numbers reproduced and asserted.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "scripts" / "data" / "ml-100k"
OUT = ROOT / "public" / "figures" / "lessons" / "39"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "39"

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


def save(fig, path, *, dpi=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def load():
    rows = np.loadtxt(DATA / "u.data", dtype=int)  # user, item, rating, timestamp
    titles = {}
    with open(DATA / "u.item", encoding="latin-1") as f:
        for line in f:
            p = line.split("|")
            titles[int(p[0])] = p[1]
    return rows, titles


def split(rows, seed=17, frac=0.9):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(rows))
    cut = int(frac * len(rows))
    return rows[perm[:cut]], rows[perm[cut:]]


def fit(train, nu, ni, d, lam=15.0, lam_b=10.0, sweeps=14, seed=0):
    rng = np.random.default_rng(seed)
    mu = train[:, 2].mean()
    u = train[:, 0] - 1; it = train[:, 1] - 1; r = train[:, 2].astype(float)
    a = np.zeros(nu); b = np.zeros(ni)
    P = rng.normal(0, 0.1, (nu, d)); Q = rng.normal(0, 0.1, (ni, d))
    u_rows = [np.where(u == uu)[0] for uu in range(nu)]
    i_rows = [np.where(it == ii)[0] for ii in range(ni)]
    for _ in range(sweeps):
        dot = np.sum(P[u] * Q[it], axis=1)
        a = np.bincount(u, r - mu - b[it] - dot, nu) / (np.bincount(u, None, nu) + lam_b)
        b = np.bincount(it, r - mu - a[u] - dot, ni) / (np.bincount(it, None, ni) + lam_b)
        tgt = r - mu - a[u] - b[it]
        for uu in range(nu):
            idx = u_rows[uu]
            if len(idx) == 0:
                continue
            Qi = Q[it[idx]]
            P[uu] = np.linalg.solve(Qi.T @ Qi + lam * np.eye(d), Qi.T @ tgt[idx])
        for ii in range(ni):
            idx = i_rows[ii]
            if len(idx) == 0:
                continue
            Pu = P[u[idx]]
            Q[ii] = np.linalg.solve(Pu.T @ Pu + lam * np.eye(d), Pu.T @ tgt[idx])
    return dict(mu=mu, a=a, b=b, P=P, Q=Q)


def predict(m, u, it):
    return m["mu"] + m["a"][u - 1] + m["b"][it - 1] + np.sum(m["P"][u - 1] * m["Q"][it - 1], axis=1)


def rmse(m, test):
    p = predict(m, test[:, 0], test[:, 1])
    return float(np.sqrt(np.mean((p - test[:, 2]) ** 2)))


# ---------------------------------------- fig 39.1: one prediction decomposed
def fig_decompose() -> None:
    rows, titles = load()
    nu, ni = rows[:, 0].max(), rows[:, 1].max()
    tr, te = split(rows)
    m = fit(tr, nu, ni, d=8, seed=1)
    mu = m["mu"]
    # pick a real observed test rating where every term contributes and the prediction is close
    best = None
    for row in te[:4000]:
        uu, ii, rr = int(row[0]), int(row[1]), int(row[2])
        aa = m["a"][uu - 1]; bb = m["b"][ii - 1]; dd = float(m["P"][uu - 1] @ m["Q"][ii - 1])
        pr = mu + aa + bb + dd
        if dd > 0.45 and bb > 0.2 and abs(aa) > 0.1 and abs(pr - rr) < 0.4:
            best = (uu, ii, rr, aa, bb, dd, pr); break
    u_, i_, r_, a_, b_, dot, pred = best
    print(f"decompose: mu={mu:.2f} a={a_:.2f} b={b_:.2f} dot={dot:.2f} pred={pred:.2f} true={r_}")
    assert abs(pred - (mu + a_ + b_ + dot)) < 1e-9
    parts = [("общая\nсредняя μ", mu, MUTED), ("+ вкус\nпользователя a", a_, BLUE),
             ("+ репутация\nфильма b", b_, GREEN), ("+ совпадение\nвкусов p·q", dot, GOLD)]
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    x = 0; base = 0
    for name, val, col in parts:
        ax.bar(x, val, bottom=base if val >= 0 else base + val, width=0.62, color=col, alpha=0.85)
        ax.text(x, base + val + (0.1 if val >= 0 else -0.2), f"{val:+.2f}" if x > 0 else f"{val:.2f}",
                ha="center", va="bottom" if val >= 0 else "top", fontsize=11, color=INK)
        ax.text(x, -0.55, name, ha="center", va="top", fontsize=9.5, color=col)
        base += val; x += 1
    ax.bar(x, pred, width=0.62, color=INK, alpha=0.85)
    ax.text(x, pred + 0.1, f"{pred:.2f}", ha="center", fontsize=12, color=INK)
    ax.text(x, -0.55, f"прогноз\n«{titles[i_][:18]}»", ha="center", va="top", fontsize=9.5, color=INK)
    ax.axhline(0, color=LINE, lw=1)
    ax.set_ylabel("вклад в оценку (баллы)")
    ax.set_title(f"Из чего складывается прогноз оценки (истинная оценка: {r_})")
    ax.set_xticks([]); ax.set_ylim(-1.2, max(pred, mu) + 1)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, OUT / "decompose.png")


# ---------------------------------------- fig 39.2: 2D taste map of real movies
def fig_tastemap() -> None:
    rows, titles = load()
    nu, ni = rows[:, 0].max(), rows[:, 1].max()
    tr, te = split(rows)
    m = fit(tr, nu, ni, d=2, seed=2)
    counts = np.bincount(rows[:, 1], minlength=ni + 1)[1:]
    Q = m["Q"]
    # a curated set of iconic films, one representative per franchise
    wanted = ["Star Wars", "Toy Story", "Fargo", "Pulp Fiction", "Titanic",
              "Terminator 2", "Silence of the Lambs", "Godfather, The", "Jurassic Park",
              "Contact", "Scream", "Lion King", "Fugitive", "English Patient"]
    fig, ax = plt.subplots(figsize=(9.4, 6.6))
    big = np.argsort(counts)[::-1][:220]
    ax.scatter(Q[big, 0], Q[big, 1], s=np.sqrt(counts[big]) * 2.2, color=BLUE, alpha=0.22, edgecolors="none")
    placed = []  # greedy declutter: skip labels closer than a threshold
    for w in wanted:
        cand = [ii for ii in range(ni) if counts[ii] >= 120 and w.lower() in titles[ii + 1].lower()]
        if not cand:
            continue
        ii = max(cand, key=lambda k: counts[k])
        pos = Q[ii]
        if any(np.hypot(pos[0] - px, pos[1] - py) < 0.16 for px, py in placed):
            continue
        placed.append((pos[0], pos[1]))
        ax.scatter(pos[0], pos[1], s=44, color=RED, zorder=5, edgecolors=PAPER, linewidths=0.8)
        ax.annotate(titles[ii + 1].split(" (")[0][:20], (pos[0], pos[1]), fontsize=9, color=INK,
                    xytext=(5, 4), textcoords="offset points", zorder=6)
    ax.set_xlabel("первый скрытый фактор"); ax.set_ylabel("второй скрытый фактор")
    ax.set_title("Карта вкусов: близкие фильмы нравятся похожей публике")
    ax.text(0.02, 0.02, "оси не имеют закреплённого смысла: карту можно повернуть,\nпрогнозы не изменятся",
            transform=ax.transAxes, fontsize=9.5, color=MUTED, va="bottom")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    print(f"tastemap: {len(big)} movies plotted, d=2")
    save(fig, OUT / "tastemap.png")


# ---------------------------------------- fig 39.3: RMSE vs number of factors + head/tail
def fig_rmse() -> None:
    rows, titles = load()
    nu, ni = rows[:, 0].max(), rows[:, 1].max()
    tr, te = split(rows)
    counts = np.bincount(tr[:, 1], minlength=ni + 1)[1:]
    thr = np.median(counts[counts > 0])
    head_mask = counts[te[:, 1] - 1] >= thr
    ds = [0, 2, 8, 16, 32]
    all_r, head_r, tail_r = [], [], []
    for d in ds:
        m = fit(tr, nu, ni, d=max(d, 1), seed=3) if d > 0 else fit_bias(tr, nu, ni)
        p = predict(m, te[:, 0], te[:, 1])
        all_r.append(float(np.sqrt(np.mean((p - te[:, 2]) ** 2))))
        head_r.append(float(np.sqrt(np.mean((p[head_mask] - te[head_mask, 2]) ** 2))))
        tail_r.append(float(np.sqrt(np.mean((p[~head_mask] - te[~head_mask, 2]) ** 2))))
    print(f"rmse all={np.round(all_r,3)} head={np.round(head_r,3)} tail={np.round(tail_r,3)}")
    assert all_r[0] > all_r[2] and all_r[2] < 0.96 and tail_r[2] > head_r[2]
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    xs = ["только\nсмещения", "d=2", "d=8", "d=16", "d=32"]
    ax.plot(xs, all_r, "o-", color=INK, lw=2.2, markersize=6, label="все оценки")
    ax.plot(xs, head_r, "s--", color=GREEN, lw=1.8, markersize=5, label="популярные (head)")
    ax.plot(xs, tail_r, "^--", color=RED, lw=1.8, markersize=5, label="редкие (tail)")
    ax.set_ylabel("RMSE на тесте (баллы)")
    ax.set_title("Скрытые факторы улучшают прогноз; редкие фильмы предсказать труднее")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.annotate("одни смещения — уже\nсильный базовый уровень", xy=(0, all_r[0]),
                xytext=(0.6, all_r[0] + 0.02), fontsize=9.5, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=LINE, lw=1.0))
    save(fig, OUT / "rmse.png")


def fit_bias(train, nu, ni, lam_b=5.0, sweeps=14):
    mu = train[:, 2].mean(); u = train[:, 0] - 1; it = train[:, 1] - 1; r = train[:, 2].astype(float)
    a = np.zeros(nu); b = np.zeros(ni)
    for _ in range(sweeps):
        a = np.bincount(u, r - mu - b[it], nu) / (np.bincount(u, None, nu) + lam_b)
        b = np.bincount(it, r - mu - a[u], ni) / (np.bincount(it, None, ni) + lam_b)
    return dict(mu=mu, a=a, b=b, P=np.zeros((nu, 1)), Q=np.zeros((ni, 1)))


# ---------------------------------------- margins
def side_rotation() -> None:
    rng = np.random.default_rng(9)
    pts = rng.normal(0, 1, (8, 2))
    th = 0.9; R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(4.4, 2.3))
    for ax, M, tit in [(a0, np.eye(2), "исходная карта"), (a1, R, "после поворота")]:
        p = pts @ M.T
        ax.scatter(p[:, 0], p[:, 1], s=20, color=BLUE)
        for i in range(0, 8, 2):
            ax.plot(p[i:i + 2, 0], p[i:i + 2, 1], color=LINE, lw=0.8)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(tit, fontsize=8.5)
    fig.suptitle("поворот не меняет расстояния и прогнозы", y=1.05, fontsize=9)
    fig.tight_layout()
    save(fig, SIDE / "rotation.png")


def side_coldstart() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.3))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.text(5, 5.4, "новый пользователь: оценок нет", ha="center", fontsize=9, color=INK)
    from matplotlib.patches import Rectangle
    for k, (lab, col) in enumerate([("популярное", GREEN), ("по жанру", BLUE), ("пара вопросов", GOLD)]):
        ax.add_patch(Rectangle((0.6 + k * 3.1, 2.2), 2.6, 1.4, fc=WASH, ec=col, lw=1.5))
        ax.text(0.6 + k * 3.1 + 1.3, 2.9, lab, ha="center", va="center", fontsize=9, color=col)
    ax.annotate("", xy=(5, 2.0), xytext=(5, 1.0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4))
    ax.text(5, 0.7, "первый вектор вкуса", ha="center", fontsize=8.5, color=MUTED)
    ax.set_title("холодный старт", fontsize=10)
    save(fig, SIDE / "coldstart.png")


def side_bias() -> None:
    rows, _ = load()
    ni = rows[:, 1].max()
    b = np.bincount(rows[:, 1], rows[:, 2] - rows[:, 2].mean(), ni + 1)[1:] / (np.bincount(rows[:, 1], None, ni + 1)[1:] + 5)
    fig, ax = plt.subplots(figsize=(4.0, 2.3))
    ax.hist(b, bins=40, color=VIOLET, alpha=0.75)
    ax.axvline(0, color=INK, lw=1.0, ls=(0, (3, 3)))
    ax.set_xlabel("смещение фильма b (баллы)", fontsize=9); ax.set_yticks([])
    ax.set_title("одни фильмы нравятся всем сильнее", fontsize=9)
    save(fig, SIDE / "bias.png")


fig_decompose()
fig_tastemap()
fig_rmse()
side_rotation()
side_coldstart()
side_bias()
print("lesson 39 figures written")
