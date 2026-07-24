"""Deterministic figures for lesson 80: pairwise preferences and the Bradley-Terry model.

REAL data: MovieLens 100K (scripts/data/ml-100k). Every user who rated two of the
selected films with different ratings casts one pairwise vote for the higher-rated film;
equal ratings become ties. That turns a real rating table into a real paired-comparison
tournament with a sparse comparison graph, ties, cycles and two rater groups.

Every number quoted in the lesson text is computed here and asserted.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ML = ROOT / "scripts" / "data" / "ml-100k"
OUT = ROOT / "public" / "figures" / "lessons" / "80"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "80"
FACTS = ROOT / "scripts" / "data" / "lesson80_facts.json"

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

FACTSD: dict[str, float | int | str] = {}


def fact(key, value):
    FACTSD[key] = value
    return value


def save(fig, path, *, dpi=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------- data
N_ITEMS = 60


def load():
    raw = np.loadtxt(ML / "u.data", dtype=np.int64)          # user item rating ts
    users, items, ratings = raw[:, 0], raw[:, 1], raw[:, 2]
    counts = np.bincount(items, minlength=items.max() + 1)
    top = np.argsort(-counts)[:N_ITEMS]
    top = top[np.argsort(-counts[top])]
    pos = -np.ones(counts.size, dtype=np.int64)
    pos[top] = np.arange(N_ITEMS)
    keep = pos[items] >= 0
    users, cols, ratings = users[keep], pos[items[keep]], ratings[keep]

    titles = {}
    years = {}
    genres = {}
    with open(ML / "u.item", encoding="latin-1") as f:
        for row in f:
            p = row.rstrip("\n").split("|")
            mid = int(p[0])
            titles[mid] = p[1]
            years[mid] = int(p[2][-4:]) if p[2] else 0
            genres[mid] = np.array([int(v) for v in p[5:24]])
    gender = {}
    age = {}
    with open(ML / "u.user", encoding="latin-1") as f:
        for row in f:
            p = row.rstrip("\n").split("|")
            gender[int(p[0])] = p[2]
            age[int(p[0])] = int(p[1])

    names = [titles[int(m)] for m in top]
    short = [n.split(" (")[0] for n in names]
    yrs = np.array([years[int(m)] for m in top])
    gen = np.array([genres[int(m)] for m in top])
    pop = counts[top].astype(float)
    return dict(users=users, cols=cols, ratings=ratings, top=top, names=names,
                short=short, years=yrs, genres=gen, pop=pop,
                gender=gender, age=age)


def per_user_matrices(D, user_filter=None):
    """W[u] wins, T[u] ties for each user; shape (nu, N, N)."""
    uids = np.unique(D["users"])
    if user_filter is not None:
        uids = np.array([u for u in uids if user_filter(u)])
    W = np.zeros((uids.size, N_ITEMS, N_ITEMS), dtype=np.int16)
    T = np.zeros((uids.size, N_ITEMS, N_ITEMS), dtype=np.int16)
    order = np.argsort(D["users"], kind="stable")
    us, cs, rs = D["users"][order], D["cols"][order], D["ratings"][order]
    idx = {u: k for k, u in enumerate(uids)}
    start = 0
    for i in range(1, us.size + 1):
        if i == us.size or us[i] != us[start]:
            u = int(us[start])
            if u in idx:
                c = cs[start:i]
                r = rs[start:i].astype(np.int16)
                gt = (r[:, None] > r[None, :]).astype(np.int16)
                eq = (r[:, None] == r[None, :]).astype(np.int16)
                np.fill_diagonal(eq, 0)
                k = idx[u]
                W[k][np.ix_(c, c)] = gt
                T[k][np.ix_(c, c)] = eq
            start = i
    return uids, W, T


def fit_bt(Wm, iters=400, tol=1e-12):
    """Zermelo/MM iterations for Bradley-Terry; returns centred log-scores."""
    N = Wm.shape[0]
    Nij = Wm + Wm.T
    wins = Wm.sum(axis=1).astype(float)
    p = np.ones(N)
    live = wins > 0
    for _ in range(iters):
        denom = np.zeros(N)
        S = p[:, None] + p[None, :]
        np.fill_diagonal(S, 1.0)
        denom = (Nij / S).sum(axis=1) - Nij.diagonal() / 1.0
        new = np.where(denom > 0, wins / np.maximum(denom, 1e-12), p)
        new = np.where(live, new, p)
        new = new / np.exp(np.mean(np.log(np.maximum(new, 1e-12))))
        if np.max(np.abs(np.log(np.maximum(new, 1e-12)) - np.log(np.maximum(p, 1e-12)))) < tol:
            p = new
            break
        p = new
    r = np.log(np.maximum(p, 1e-12))
    return r - r.mean()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def se_scores(Wm, r):
    """Fisher-information standard errors for the scores (fixed sum-zero gauge)."""
    Nij = Wm + Wm.T
    P = sigmoid(r[:, None] - r[None, :])
    I = Nij * P * (1 - P)
    np.fill_diagonal(I, 0.0)
    L = np.diag(I.sum(axis=1)) - I
    C = np.linalg.pinv(L)                          # sum-zero gauge
    return np.sqrt(np.clip(np.diag(C), 0, None))


# ---------------------------------------------------------------- figures
def fig1_curve(D, Wm, r):
    """Sigmoid + empirical win rates from the real tournament."""
    Nij = Wm + Wm.T
    iu = np.triu_indices(N_ITEMS, 1)
    n = Nij[iu].astype(float)
    w = Wm[iu].astype(float)
    d = (r[:, None] - r[None, :])[iu]
    mask = n >= 30
    d, w, n = d[mask], w[mask], n[mask]
    edges = np.array([-2.4, -1.4, -0.9, -0.55, -0.28, -0.09, 0.09, 0.28, 0.55, 0.9, 1.4, 2.4])
    xs, ys, ws = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (d >= a) & (d < b)
        if m.sum() >= 5:
            xs.append(float(np.average(d[m], weights=n[m])))
            ys.append(float(w[m].sum() / n[m].sum()))
            ws.append(float(n[m].sum()))
    xs, ys, ws = np.array(xs), np.array(ys), np.array(ws)
    err = float(np.max(np.abs(ys - sigmoid(xs))))
    fact("n_pairs_calib", int(mask.sum()))
    fact("calib_max_err", round(err, 3))
    assert err < 0.09, err

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    g = np.linspace(-3, 3, 400)
    ax.plot(g, sigmoid(g), color=BLUE, lw=2.4, label=r"$\sigma(r_i-r_j)$ — модель")
    ax.scatter(xs, ys, s=np.sqrt(ws) * 1.4, color=RED, zorder=6,
               label="реальная доля побед (MovieLens)")
    for lvl, lab in [(0.5, "0,50"), (0.75, "0,75"), (0.9, "0,90")]:
        z = float(np.log(lvl / (1 - lvl)))
        ax.plot([z, z], [0, lvl], color=MUTED, lw=0.8, ls=(0, (2, 2)))
        ax.plot([-3, z], [lvl, lvl], color=MUTED, lw=0.8, ls=(0, (2, 2)))
        ax.text(-2.95, lvl + 0.015, lab, fontsize=9.5, color=MUTED)
        ax.text(z + 0.05, 0.03, ("Δr = %.2f" % z).replace(".", ","), fontsize=9.5, color=MUTED)
    ax.set_xlim(-3, 3); ax.set_ylim(0, 1)
    ax.set_xlabel(r"разность скрытых баллов $r_i-r_j$")
    ax.set_ylabel(r"$\Pr(i\succ j)$")
    ax.set_title("Разность баллов становится вероятностью — и это проверяется")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "bt_curve.png")


def fig2_scores_vs_mean(D, Wm, r, se):
    mean_r = np.zeros(N_ITEMS)
    for c in range(N_ITEMS):
        mean_r[c] = D["ratings"][D["cols"] == c].mean()
    rho = float(np.corrcoef(r, mean_r)[0, 1])
    order_bt = np.argsort(-r)
    order_mn = np.argsort(-mean_r)
    # Kendall tau between the two rankings
    a = np.argsort(np.argsort(-r)); b = np.argsort(np.argsort(-mean_r))
    conc = 0; disc = 0
    for i, j in itertools.combinations(range(N_ITEMS), 2):
        s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
        conc += s > 0; disc += s < 0
    tau = (conc - disc) / (conc + disc)
    fact("bt_mean_corr", round(rho, 3))
    fact("bt_mean_tau", round(float(tau), 3))
    fact("top_bt", D["short"][int(order_bt[0])])
    fact("top_bt_score", round(float(r[order_bt[0]]), 2))
    fact("top_mean", D["short"][int(order_mn[0])])
    fact("top_mean_value", round(float(mean_r[order_mn[0]]), 2))
    rank_bt = np.argsort(np.argsort(-r))
    rank_mn = np.argsort(np.argsort(-mean_r))
    gaps = rank_mn - rank_bt
    kmax = int(np.argmax(np.abs(gaps)))
    fact("rank_gap_movie", D["short"][kmax])
    fact("rank_gap", int(abs(gaps[kmax])))
    fact("rank_gap_bt", int(rank_bt[kmax]) + 1)
    fact("rank_gap_mean", int(rank_mn[kmax]) + 1)
    assert rho > 0.9 and tau > 0.75

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.errorbar(mean_r, r, yerr=1.96 * se, fmt="none", ecolor=LINE, lw=1.0, zorder=2)
    ax.scatter(mean_r, r, s=np.sqrt(D["pop"]) * 3.0, color=BLUE, alpha=0.85, zorder=4)
    marks = [(int(order_bt[0]), (8, -4)), (int(order_bt[-1]), (10, 0)), (kmax, (10, -12))]
    for k, off in marks:
        ax.annotate(D["short"][k], (mean_r[k], r[k]), fontsize=9.5,
                    xytext=off, textcoords="offset points", color=INK)
    ax.scatter([mean_r[kmax]], [r[kmax]], s=130, color=RED, zorder=6,
               edgecolors=PAPER, linewidths=1.2)
    ax.set_xlabel("средняя оценка фильма (1–5)")
    ax.set_ylabel(r"скрытый балл Брэдли–Терри $r_i$")
    ax.set_title("Балл сравнений и средняя оценка: корреляция 0,98 — и всё же разные шкалы")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    ax.text(0.02, 0.95, "размер точки — число оценок,\nусы — 95% интервал балла,\n"
            f"красным — наибольшее расхождение рангов ({abs(int(gaps[kmax]))} позиций)",
            transform=ax.transAxes, fontsize=9.5, color=MUTED, va="top")
    save(fig, OUT / "scores_vs_mean.png")
    return mean_r


def fig3_graph(D, uids, W):
    """Bridge experiment: two clusters, one bridge edge, then three cross edges."""
    Wall = W.sum(axis=0).astype(float)
    sel = np.argsort(-D["pop"])[:12]
    left, right = sel[:6], sel[6:]
    def sub(mask_pairs):
        M = np.zeros((12, 12))
        for a in range(12):
            for b in range(12):
                if mask_pairs[a, b]:
                    M[a, b] = Wall[sel[a], sel[b]]
        return M
    inside = np.zeros((12, 12), dtype=bool)
    inside[:6, :6] = True; inside[6:, 6:] = True
    np.fill_diagonal(inside, False)
    bridge = inside.copy(); bridge[0, 6] = bridge[6, 0] = True
    plus = bridge.copy()
    for a, b in [(1, 7), (2, 8), (3, 9)]:
        plus[a, b] = plus[b, a] = True

    res = {}
    for name, mask in [("bridge", bridge), ("plus", plus)]:
        M = sub(mask)
        rr = fit_bt(M)
        ss = se_scores(M, rr)
        d = rr[:6].mean() - rr[6:].mean()
        # se of the between-cluster contrast
        Nij = M + M.T
        P = sigmoid(rr[:, None] - rr[None, :])
        I = Nij * P * (1 - P); np.fill_diagonal(I, 0.0)
        C = np.linalg.pinv(np.diag(I.sum(axis=1)) - I)
        c = np.concatenate([np.full(6, 1 / 6), np.full(6, -1 / 6)])
        se_c = float(np.sqrt(c @ C @ c))
        res[name] = dict(r=rr, se=ss, d=float(d), se_c=se_c, M=M, mask=mask)
    ratio = res["bridge"]["se_c"] / res["plus"]["se_c"]
    fact("bridge_se", round(res["bridge"]["se_c"], 3))
    fact("plus_se", round(res["plus"]["se_c"], 3))
    fact("bridge_ratio", round(float(ratio), 1))
    fact("bridge_votes", int(res["bridge"]["M"].sum()))
    fact("plus_votes", int(res["plus"]["M"].sum()))
    fact("bridge_extra_votes_pct", round(100 * (res["plus"]["M"].sum() / res["bridge"]["M"].sum() - 1), 1))
    assert ratio > 1.6, ratio

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
    ang = np.linspace(0, 2 * np.pi, 7)[:6]
    posL = np.stack([-1.35 + 0.62 * np.cos(ang), 0.62 * np.sin(ang)], axis=1)
    posR = np.stack([1.35 + 0.62 * np.cos(ang), 0.62 * np.sin(ang)], axis=1)
    P = np.vstack([posL, posR])
    for ax, key, ttl in [(axes[0], "bridge", "один мост"), (axes[1], "plus", "мост + три ребра")]:
        R = res[key]
        for a in range(12):
            for b in range(a + 1, 12):
                if R["mask"][a, b]:
                    cross = (a < 6) != (b < 6)
                    ax.plot([P[a, 0], P[b, 0]], [P[a, 1], P[b, 1]],
                            color=RED if cross else LINE, lw=2.2 if cross else 0.9,
                            alpha=0.95 if cross else 0.7, zorder=1)
        ax.scatter(P[:, 0], P[:, 1], s=190, c=[BLUE] * 6 + [GREEN] * 6, zorder=5,
                   edgecolors=PAPER, linewidths=1.4)
        ax.set_title(f"{ttl}: разрыв групп ±{1.96 * R['se_c']:.2f}".replace(".", ","),
                     fontsize=12.5, pad=8)
        ax.set_xlim(-2.3, 2.3); ax.set_ylim(-1.15, 1.15); ax.axis("off")
    fig.suptitle("Разница между двумя сообществами держится на переходных рёбрах",
                 fontsize=14.5, y=1.02)
    save(fig, OUT / "graph_bridge.png")


def fig4_ties(D, Wm, Tm, r):
    Nij = Wm + Wm.T
    iu = np.triu_indices(N_ITEMS, 1)
    d = np.abs(r[:, None] - r[None, :])[iu]
    ties = (Tm + Tm.T)[iu].astype(float) / 2
    dec = Nij[iu].astype(float)
    total_ties = float(ties.sum()); total_dec = float(dec.sum())
    share = total_ties / (total_ties + total_dec)
    fact("votes_decisive", int(total_dec))
    fact("votes_ties", int(total_ties))
    fact("tie_share_pct", round(100 * share, 1))
    bins = np.array([0, 0.15, 0.3, 0.5, 0.75, 1.1, 2.2])
    xs, ys = [], []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (d >= a) & (d < b)
        xs.append((a + b) / 2)
        ys.append(float(ties[m].sum() / (ties[m].sum() + dec[m].sum())))
    ys = np.array(ys)
    fact("tie_share_close_pct", round(100 * ys[0], 1))
    fact("tie_share_far_pct", round(100 * ys[-1], 1))
    assert ys[0] > ys[-1] + 0.05, ys

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(range(len(ys)), 100 * ys, color=GOLD, width=0.62, alpha=0.9)
    ax.set_xticks(range(len(ys)))
    ax.set_xticklabels([f"{a:.2f}–{b:.2f}".replace(".", ",") for a, b in zip(bins[:-1], bins[1:])],
                       fontsize=9.5)
    for i, v in enumerate(ys):
        ax.text(i, 100 * v + 0.7, f"{100 * v:.0f}%", ha="center", fontsize=10, color=INK)
    ax.set_xlabel(r"разность скрытых баллов $|r_i-r_j|$")
    ax.set_ylabel("доля ничьих, %")
    ax.set_title("Ничьи скапливаются там, где модель сомневается")
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "ties.png")


def fig5_groups(D, W_all, uids_all):
    male = {u for u, g in D["gender"].items() if g == "M"}
    uidsM, WM, _ = per_user_matrices(D, lambda u: u in male)
    uidsF, WF, _ = per_user_matrices(D, lambda u: u not in male)
    Wm_ = WM.sum(axis=0).astype(float); Wf_ = WF.sum(axis=0).astype(float)
    rm = fit_bt(Wm_); rf = fit_bt(Wf_)
    sem = se_scores(Wm_, rm); sef = se_scores(Wf_, rf)
    rho = float(np.corrcoef(rm, rf)[0, 1])
    flips = []
    for i, j in itertools.combinations(range(N_ITEMS), 2):
        dm = rm[i] - rm[j]; df = rf[i] - rf[j]
        if dm * df < 0:
            sig = abs(dm) > 1.96 * np.hypot(sem[i], sem[j]) and abs(df) > 1.96 * np.hypot(sef[i], sef[j])
            flips.append((abs(dm) + abs(df), i, j, dm, df, sig))
    flips.sort(reverse=True)
    n_sig = sum(1 for f in flips if f[5])
    fact("n_men", int(uidsM.size)); fact("n_women", int(uidsF.size))
    fact("group_corr", round(rho, 3))
    fact("n_flips", len(flips))
    fact("n_flips_sig", int(n_sig))
    top = flips[0]
    fact("flip_a", D["short"][top[1]]); fact("flip_b", D["short"][top[2]])
    fact("flip_dm", round(float(top[3]), 2)); fact("flip_df", round(float(top[4]), 2))
    assert rho > 0.8 and n_sig >= 1

    ra = fit_bt(W_all.sum(axis=0).astype(float))
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot([-1.6, 1.9], [-1.6, 1.9], color=LINE, lw=1.0, ls=(0, (4, 3)))
    ax.scatter(rm, rf, s=42, color=BLUE, alpha=0.8, zorder=4)
    i, j = top[1], top[2]
    for k, c in [(i, RED), (j, GOLD)]:
        ax.scatter([rm[k]], [rf[k]], s=120, color=c, zorder=6, edgecolors=PAPER, linewidths=1.3)
        ax.annotate(D["short"][k], (rm[k], rf[k]), fontsize=10, xytext=(7, -3),
                    textcoords="offset points", color=INK)
    ax.plot([rm[i], rm[j]], [rf[i], rf[j]], color=RED, lw=1.6, ls=(0, (2, 2)), zorder=5)
    ax.set_xlabel("балл по голосам мужчин")
    ax.set_ylabel("балл по голосам женщин")
    ax.set_title(("Один скаляр усредняет два вкуса: корреляция %.2f, %d пар меняют сторону"
                  % (rho, len(flips))).replace("0.", "0,"))
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "groups.png")
    return rm, rf


def surface_proxy(D, Wm, r):
    """Reward model on surface features only (popularity, year): honest accuracy."""
    X = np.column_stack([np.log(D["pop"]), D["years"].astype(float)])
    X = (X - X.mean(0)) / X.std(0)
    iu = np.triu_indices(N_ITEMS, 1)
    n_ij = (Wm + Wm.T)[iu].astype(float); w_ij = Wm[iu].astype(float)
    dX = X[iu[0]] - X[iu[1]]
    w = np.zeros(X.shape[1])
    for _ in range(6000):
        p = sigmoid(dX @ w)
        w -= 3.0 * dX.T @ (n_ij * p - w_ij) / n_ij.sum()
    acc = float(np.where(sigmoid(dX @ w) > 0.5, w_ij, n_ij - w_ij).sum() / n_ij.sum())
    proxy = X @ w
    fact("surface_acc_pct", round(100 * acc, 1))
    fact("surface_corr", round(float(np.corrcoef(proxy, r)[0, 1]), 2))
    acc_bt = float(np.where(sigmoid((r[iu[0]] - r[iu[1]])) > 0.5, w_ij, n_ij - w_ij).sum() / n_ij.sum())
    fact("bt_acc_pct", round(100 * acc_bt, 1))
    assert acc > 0.55 and acc_bt > acc


def fig6_goodhart(D, Wm, r):
    """Best-of-n against reward models of two qualities: real overoptimization."""
    Nij = Wm + Wm.T
    P = np.where(Nij > 0, Wm / np.maximum(Nij, 1), 0.5)
    pairs = list(itertools.combinations(range(N_ITEMS), 2))

    def scarce_model(frac, k, seed=80):
        rng = np.random.default_rng(seed)
        M = np.zeros((N_ITEMS, N_ITEMS))
        for (i, j) in pairs:
            if rng.random() > frac:
                continue
            w = rng.binomial(k, P[i, j])
            M[i, j] += w; M[j, i] += k - w
        assert (M + M.T).sum(axis=1).min() > 0, "граф распался"
        return fit_bt(M), M

    r_poor, M_poor = scarce_model(0.12, 1)
    r_rich = fit_bt(Wm * 1.0)
    fact("poor_votes", int(M_poor.sum()))
    fact("poor_pairs", int((M_poor + M_poor.T > 0)[np.triu_indices(N_ITEMS, 1)].sum()))
    fact("poor_corr", round(float(np.corrcoef(r_poor, r)[0, 1]), 2))

    ns = np.arange(1, 41)
    B = 8000

    def bon(score):
        rng2 = np.random.default_rng(7)
        t = np.zeros(ns.size); p = np.zeros(ns.size)
        for k, n in enumerate(ns):
            s = rng2.integers(0, N_ITEMS, size=(B, n))
            pick = s[np.arange(B), np.argmax(score[s], axis=1)]
            t[k] = r[pick].mean(); p[k] = score[pick].mean()
        return t, p

    t_poor, p_poor = bon(r_poor)
    t_rich, _ = bon(r_rich)
    best = int(np.argmax(t_poor))
    fact("bon_best_n", int(ns[best]))
    fact("bon_peak_true", round(float(t_poor[best]), 2))
    fact("bon_end_true", round(float(t_poor[-1]), 2))
    fact("bon_drop_pct", round(100 * float(1 - t_poor[-1] / t_poor[best]), 0))
    fact("bon_rich_end", round(float(t_rich[-1]), 2))
    fact("bt_max", round(float(r.max()), 2))
    assert t_poor[best] - t_poor[-1] > 0.1, (t_poor[best], t_poor[-1])
    assert p_poor[-1] > p_poor[best]
    assert t_rich[-1] > t_poor[-1]

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.axhline(float(r.max()), color=LINE, lw=1.2, ls=(0, (4, 3)))
    ax.text(1.2, float(r.max()) + 0.03, "лучший фильм по людям", fontsize=9.5, color=MUTED)
    scale = float(r.std() / r_poor.std())
    ax.plot(ns, p_poor * scale, color=BLUE, lw=2.2, ls=(0, (5, 3)),
            label="балл в глазах слабой RM (в её же масштабе)")
    ax.plot(ns, t_poor, color=RED, lw=2.6, label="настоящий балл выбранного (слабая RM)")
    ax.plot(ns, t_rich, color=GREEN, lw=2.2, label="настоящий балл выбранного (богатая RM)")
    ax.axvline(ns[best], color=MUTED, lw=0.9, ls=(0, (2, 2)))
    ax.annotate(f"пик при n={ns[best]}", (ns[best], t_poor[best]), fontsize=10,
                xytext=(10, 16), textcoords="offset points", color=MUTED)
    ax.set_xlabel("n — из скольких случайных фильмов берём лучший по reward model")
    ax.set_ylabel("средний балл выбранного фильма")
    ax.set_title("Переоптимизация: сильнее давим на proxy — теряем то, что мерили")
    ax.set_ylim(-0.05, 1.75)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "goodhart.png")


def fig7_cycles(D, Wm, r):
    """Real tournament is transitive; the cycle is an explicitly labelled model example."""
    Nij = Wm + Wm.T
    P = np.where(Nij > 0, Wm / np.maximum(Nij, 1), 0.5)
    A = P > 0.5
    triples = 0; cycles = 0
    for i, j, k in itertools.combinations(range(N_ITEMS), 3):
        triples += 1
        if A[i, j] + A[j, k] + A[k, i] == 3 or A[j, i] + A[k, j] + A[i, k] == 3:
            cycles += 1
    fact("n_triples", triples)
    fact("n_cycles", cycles)
    fact("min_votes_pair", int(Nij[np.triu_indices(N_ITEMS, 1)].min()))
    fact("median_votes_pair", int(np.median(Nij[np.triu_indices(N_ITEMS, 1)])))
    assert cycles == 0, cycles

    iu = np.triu_indices(N_ITEMS, 1)
    resid = P - sigmoid(r[:, None] - r[None, :])
    ru = resid[iu]
    fact("resid_max", round(float(np.max(np.abs(ru))), 3))
    fact("resid_rms", round(float(np.sqrt(np.mean(ru ** 2))), 3))
    kbest = int(np.argmax(np.abs(ru)))
    ai, bi = int(iu[0][kbest]), int(iu[1][kbest])
    fact("resid_a", D["short"][ai]); fact("resid_b", D["short"][bi])
    fact("resid_pemp", round(float(P[ai, bi]), 2))
    fact("resid_pfit", round(float(sigmoid(r[ai] - r[bi])), 2))
    fact("resid_votes", int(Nij[ai, bi]))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.9),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    ordr = np.argsort(-r)[:24]
    Z = resid[np.ix_(ordr, ordr)]
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-0.2, vmax=0.2)
    ax.set_title("Реальные остатки: факт минус модель", fontsize=12.5, pad=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("24 фильма с наибольшим баллом, по убыванию")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    ax = axes[1]
    ax.axis("off")
    pos = np.array([[0.0, 0.9], [-0.85, -0.5], [0.85, -0.5]])
    labs = ["A: коротко", "B: подробно", "C: дружелюбно"]
    for t in range(3):
        u, v = pos[t] * 0.72, pos[(t + 1) % 3] * 0.72
        ax.annotate("", xy=v, xytext=u, arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.0))
        mid = (u + v) / 2 * 1.08
        ax.text(mid[0], mid[1], "2/3", fontsize=11, color=RED, ha="center", va="center",
                bbox=dict(fc=PAPER, ec="none", pad=1.5))
    for t in range(3):
        ax.text(pos[t, 0], pos[t, 1] * 1.12, labs[t], fontsize=10.5, ha="center", color=INK)
    ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.2, 1.35)
    ax.set_title("Модельный пример: три вкуса дают цикл\n(в реальных данных 0 циклов из "
                 f"{triples})", fontsize=12.5, pad=10)
    fig.suptitle("Одномерная шкала: где она держится и где ломается", fontsize=14.5, y=1.04)
    save(fig, OUT / "cycles.png")


# ---------------------------------------------------------------- sidenote images
def side_odds():
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    d = np.linspace(0, 3, 200)
    ax.plot(d, sigmoid(d), color=BLUE, lw=2.2)
    for z, lab in [(np.log(3), "3:1"), (np.log(9), "9:1")]:
        ax.plot([z, z], [0.5, sigmoid(z)], color=RED, lw=1.0, ls=(0, (2, 2)))
        ax.scatter([z], [sigmoid(z)], s=28, color=RED, zorder=5)
        ax.text(z + 0.06, sigmoid(z) - 0.06, lab, fontsize=9.5, color=RED)
    ax.set_xlabel(r"$\Delta r$", fontsize=10); ax.set_ylabel("шанс победы", fontsize=10)
    ax.set_ylim(0.45, 1.0)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)
    save(fig, SIDE / "odds.png")


def side_precision(D, W):
    """CI width of a score against the number of votes: the 1/sqrt(n) law, on real data."""
    Wall = W.sum(axis=0).astype(float)
    fracs = [0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
    xs, ys = [], []
    for f in fracs:
        M = Wall * f
        rr = fit_bt(M)
        ss = se_scores(M, rr)
        xs.append(M.sum()); ys.append(float(np.median(ss)))
    xs, ys = np.array(xs), np.array(ys)
    slope = float(np.polyfit(np.log(xs), np.log(ys), 1)[0])
    fact("se_slope", round(slope, 2))
    fact("se_full_median", round(float(ys[-1]), 4))
    fact("votes_total", int(xs[-1]))
    assert -0.55 < slope < -0.45, slope
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    ax.loglog(xs, ys, "o-", color=BLUE, lw=2.0, ms=5)
    ax.loglog(xs, ys[-1] * (xs / xs[-1]) ** -0.5, color=RED, lw=1.2, ls=(0, (3, 2)),
              label=r"$n^{-1/2}$")
    ax.set_xlabel("число голосов", fontsize=10)
    ax.set_ylabel("медианная s.e. балла", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which="both", color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)
    save(fig, SIDE / "precision.png")


def side_active(D, Wm, r):
    """Greedy active pair choice vs random on the real tournament (8 seeds, fixed)."""
    N = N_ITEMS
    Nij = (Wm + Wm.T).astype(float)
    P = np.where(Nij > 0, Wm / np.maximum(Nij, 1), 0.5)
    pairs = list(itertools.combinations(range(N), 2))
    truth = np.argsort(np.argsort(-r))

    def tau(rr):
        a = np.argsort(np.argsort(-rr)); c = d = 0
        for i, j in itertools.combinations(range(N), 2):
            s = np.sign(a[i] - a[j]) * np.sign(truth[i] - truth[j])
            c += s > 0; d += s < 0
        return (c - d) / (c + d)

    def run(active, seed, rounds=9, per=25, votes=4, lam=0.1):
        rng = np.random.default_rng(seed)
        M = np.zeros((N, N)); rr = np.zeros(N); ts = []; xs = []
        for t in range(rounds):
            if active and t > 0:
                S = sigmoid(rr[:, None] - rr[None, :])
                I = M + M.T
                I = I * S * (1 - S) + lam
                np.fill_diagonal(I, 0.0)
                C = np.linalg.pinv(np.diag(I.sum(axis=1)) - I)
                sc = [((C[i, i] + C[j, j] - 2 * C[i, j]) * S[i, j] * (1 - S[i, j]), k)
                      for k, (i, j) in enumerate(pairs)]
                sc.sort(reverse=True)
                chosen = [pairs[k] for _, k in sc[:per]]
            else:
                chosen = [pairs[k] for k in rng.choice(len(pairs), per, replace=False)]
            for (i, j) in chosen:
                w = rng.binomial(votes, P[i, j])
                M[i, j] += w; M[j, i] += votes - w
            rr = fit_bt(M + 1e-9)
            ts.append(tau(rr)); xs.append(M.sum())
        return np.array(xs), np.array(ts)

    seeds = range(1, 9)
    xs, R = None, []
    A = []
    for s in seeds:
        xs, t = run(False, s); R.append(t)
        _, t = run(True, s); A.append(t)
    R = np.mean(R, axis=0); A = np.mean(A, axis=0)
    fact("active_tau_early", round(float(A[1]), 2))
    fact("random_tau_early", round(float(R[1]), 2))
    fact("active_tau_end", round(float(A[-1]), 2))
    fact("random_tau_end", round(float(R[-1]), 2))
    fact("active_budget", int(xs[-1]))
    fact("active_budget_early", int(xs[1]))
    assert A[1] > R[1] and A[-1] < R[-1]

    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    ax.plot(xs, R, color=MUTED, lw=2.0, marker="o", ms=3, label="случайные пары")
    ax.plot(xs, A, color=GREEN, lw=2.2, marker="o", ms=3, label="жадный активный выбор")
    ax.set_xlabel("число голосов", fontsize=10)
    ax.set_ylabel(r"ранговая $\tau$ к истине", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)
    save(fig, SIDE / "active.png")


def side_position(D, Wm, r):
    """Position bias, honest synthetic with fixed seed: 5% shift breaks the ranking."""
    rng = np.random.default_rng(1952)
    n = 40
    true = np.linspace(-1.2, 1.2, n)
    votes = 40
    for bias, name in [(0.0, "no"), (0.35, "yes")]:
        pass
    res = {}
    for bias in (0.0, 0.35):
        M = np.zeros((n, n))
        for i, j in itertools.combinations(range(n), 2):
            p = sigmoid(true[i] - true[j] + bias)   # i is always shown first
            w = rng.binomial(votes, p)
            M[i, j] += w; M[j, i] += votes - w
        rr = fit_bt(M)
        res[bias] = rr
    err = float(np.max(np.abs(res[0.35] - res[0.0])))
    fact("posbias_shift", round(err, 2))
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    ax.scatter(true, res[0.0], s=18, color=BLUE, label="без bias")
    ax.scatter(true, res[0.35], s=18, color=RED, label="сдвиг 0,35")
    ax.plot([-1.3, 1.3], [-1.3, 1.3], color=LINE, lw=1.0, ls=(0, (3, 2)))
    ax.set_xlabel("истинный балл", fontsize=10)
    ax.set_ylabel("оценка", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)
    save(fig, SIDE / "position.png")


def main():
    D = load()
    uids, W, T = per_user_matrices(D)
    Wm = W.sum(axis=0).astype(float)
    Tm = T.sum(axis=0).astype(float)
    r = fit_bt(Wm)
    se = se_scores(Wm, r)
    fact("n_items", N_ITEMS)
    fact("n_users", int(uids.size))
    fact("min_pop", int(D["pop"].min()))
    fact("se_median", round(float(np.median(se)), 3))
    fact("se_max", round(float(se.max()), 3))
    fact("score_range", round(float(r.max() - r.min()), 2))
    dense = float((Wm + Wm.T > 0)[np.triu_indices(N_ITEMS, 1)].mean())
    fact("graph_density_pct", round(100 * dense, 1))

    fig1_curve(D, Wm, r)
    mean_r = fig2_scores_vs_mean(D, Wm, r, se)
    fig3_graph(D, uids, W)
    fig4_ties(D, Wm, Tm, r)
    fig5_groups(D, W, uids)
    surface_proxy(D, Wm, r)
    fig6_goodhart(D, Wm, r)
    fig7_cycles(D, Wm, r)
    side_odds()
    side_precision(D, W)
    side_active(D, Wm, r)
    side_position(D, Wm, r)

    FACTS.write_text(json.dumps(FACTSD, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in FACTSD.items():
        print(f"{k:24s} {v}")


if __name__ == "__main__":
    main()
