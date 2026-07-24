"""Deterministic figures for lesson 65: PageRank as a stationary flow.

Real data: MovieLens 100K (scripts/data/ml-100k). We build an HONEST directed graph of
"recommendation links": movie i points to the k movies most often liked by the same users
(conditional probability P(j liked | i liked)). The graph is directed and asymmetric —
exactly the structure PageRank was invented for. Every number quoted in the lesson text is
computed here and asserted.

Also: exact toy examples (3-node cycle, 6-node trap, dangling node), convergence of the
power method for three alphas, personalization and its linearity, top-10 stability.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ML = ROOT / "scripts" / "data" / "ml-100k"
OUT = ROOT / "public" / "figures" / "lessons" / "65"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "65"
FACTS = ROOT / "scripts" / "data" / "lesson65_facts.json"

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

FACT: dict = {}


def save(fig, path, *, dpi=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ============================================================ generic PageRank machinery
def pagerank(P, alpha, v, tol=1e-12, maxit=2000):
    """Power method for p_{t+1} = alpha p_t P + (1-alpha) v (row vectors)."""
    n = P.shape[0]
    p = v.copy()
    hist = []
    for t in range(1, maxit + 1):
        q = alpha * (p @ P) + (1 - alpha) * v
        q = q / q.sum()
        d = np.abs(q - p).sum()
        hist.append(d)
        p = q
        if d < tol:
            break
    return p, t, np.array(hist)


def row_stochastic(A):
    """A: adjacency (0/1 or weights). Dangling rows left as zeros."""
    out = A.sum(axis=1, keepdims=True)
    P = np.zeros_like(A, dtype=float)
    nz = out[:, 0] > 0
    P[nz] = A[nz] / out[nz]
    return P


# ============================================================ toy graphs (exact numbers)
def toy_cycle():
    """A -> B,C ; B -> C ; C -> A."""
    A = np.array([[0, 1, 1],
                  [0, 0, 1],
                  [1, 0, 0]], dtype=float)
    P = row_stochastic(A)
    v = np.ones(3) / 3
    pi_exact = np.array([0.4, 0.2, 0.4])
    # check stationarity of the exact answer for alpha = 1
    assert np.allclose(pi_exact @ P, pi_exact), pi_exact @ P
    # three steps from p0 = (1,0,0)
    p = np.array([1.0, 0.0, 0.0])
    steps = [p.copy()]
    for _ in range(3):
        p = p @ P
        steps.append(p.copy())
    assert np.allclose(steps[1], [0, 0.5, 0.5])
    assert np.allclose(steps[2], [0.5, 0, 0.5])
    assert np.allclose(steps[3], [0.5, 0.25, 0.25])
    pg, it, _ = pagerank(P, 0.85, v)
    FACT["toy_cycle_pi"] = [round(x, 4) for x in pi_exact]
    FACT["toy_cycle_steps"] = [[round(x, 4) for x in s] for s in steps]
    FACT["toy_cycle_pagerank_085"] = [round(x, 4) for x in pg]
    FACT["toy_cycle_iters_085"] = it
    return A, P, steps, pg


def toy_trap():
    """A->B; B->C; C->A,D; D->E; E->F; F->D. Nodes D,E,F form an absorbing trap."""
    lab = ["A", "B", "C", "D", "E", "F"]
    A = np.zeros((6, 6))
    edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)]
    for i, j in edges:
        A[i, j] = 1
    P = row_stochastic(A)
    v = np.ones(6) / 6
    # no teleport: mass drains into the trap
    p = v.copy()
    for _ in range(400):
        p = p @ P
    trap_mass = p[3:].sum()
    assert trap_mass > 0.999, trap_mass
    pg, it, _ = pagerank(P, 0.85, v)
    trap_mass_tp = pg[3:].sum()
    assert 0.6 < trap_mass_tp < 0.95, trap_mass_tp
    FACT["trap_mass_noteleport"] = round(float(trap_mass), 4)
    FACT["trap_mass_teleport085"] = round(float(trap_mass_tp), 4)
    FACT["trap_pagerank_085"] = {lab[i]: round(float(pg[i]), 4) for i in range(6)}
    FACT["trap_iters_085"] = it
    return lab, A, P, p, pg, edges


def toy_dangling():
    """A->B; B->C; C has no outgoing links: probability mass evaporates."""
    A = np.zeros((3, 3))
    A[0, 1] = 1
    A[1, 2] = 1
    P = row_stochastic(A)
    p = np.ones(3) / 3
    mass = [1.0]
    for _ in range(6):
        p = p @ P
        mass.append(float(p.sum()))
    assert abs(mass[1] - 2 / 3) < 1e-12, mass
    assert mass[-1] < 1e-9, mass
    FACT["dangling_mass"] = [round(m, 6) for m in mass]
    return mass


# ============================================================ real graph from MovieLens
def build_movielens_graph(min_likes=45, thr=0.6):
    likes = defaultdict(set)          # movie -> set of users who rated >= 4
    with open(ML / "u.data") as f:
        for line in f:
            u, m, r, _ = line.split("\t")
            if int(r) >= 4:
                likes[int(m)].add(int(u))
    titles, genres = {}, {}
    with open(ML / "u.item", encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            titles[int(parts[0])] = parts[1]
            genres[int(parts[0])] = np.array([int(x) for x in parts[5:24]])
    ids = sorted(m for m in likes if len(likes[m]) >= min_likes)
    n = len(ids)
    idx = {m: i for i, m in enumerate(ids)}
    sets = [likes[m] for m in ids]
    lk = np.array([len(s) for s in sets], dtype=float)
    # co-like counts
    co = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            c = len(sets[a] & sets[b])
            co[a, b] = c
            co[b, a] = c
    cond = co / lk[:, None]           # cond[i, j] = P(j liked | i liked)
    A = (cond >= thr).astype(float)   # i -> j : "most of those who liked i also liked j"
    np.fill_diagonal(A, 0.0)
    return ids, titles, genres, A, lk, cond


def real_analysis():
    ids, titles, genres, A, lk, cond = build_movielens_graph()
    n = A.shape[0]
    m = int(A.sum())
    v = np.ones(n) / n
    outdeg = A.sum(axis=1)
    n_dangling = int((outdeg == 0).sum())
    assert n_dangling > 0, "the real graph must contain dangling nodes"
    P = row_stochastic(A)
    P[outdeg == 0] = v                # standard repair of dangling rows
    assert np.allclose(P.sum(axis=1), 1.0)
    indeg = A.sum(axis=0)

    pr, it85, hist85 = pagerank(P, 0.85, v)
    _, it70, hist70 = pagerank(P, 0.70, v)
    _, it95, hist95 = pagerank(P, 0.95, v)

    order = np.argsort(-pr)
    top5 = [(titles[ids[i]], round(float(pr[i]), 4), int(indeg[i])) for i in order[:5]]
    ordd = np.argsort(-indeg)
    top5_deg = [(titles[ids[i]], int(indeg[i]), round(float(pr[i]), 4)) for i in ordd[:5]]

    # Spearman correlation between in-degree and PageRank (rank correlation, no scipy needed)
    def rank(x):
        """Mid-ranks: ties share the average rank, as Spearman's rho requires.
        In-degrees have many ties, so naive argsort ranking would fake precision."""
        order = np.argsort(x, kind="stable")
        r = np.empty(len(x), dtype=float)
        sx = x[order]
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and sx[j + 1] == sx[i]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2 + 1
            i = j + 1
        return r
    r1, r2 = rank(indeg), rank(pr)
    rho = float(np.corrcoef(r1, r2)[0, 1])
    assert 0.9 < rho < 1.0, rho

    # the sharpest disagreement: same in-degree, different PageRank
    ratio = pr / (indeg + 1)
    hi = int(np.argmax(np.where(indeg >= 50, ratio, -1)))
    lo = int(np.argmin(np.where(indeg >= 50, ratio, 1e9)))
    # who wins on rank but loses on degree
    top_pr, top_deg = order[0], ordd[0]
    assert top_pr != top_deg, "the headline example requires disagreement at the very top"
    swap = dict(
        winner=titles[ids[top_pr]], winner_indeg=int(indeg[top_pr]),
        winner_pr=round(float(pr[top_pr]), 4),
        winner_deg_place=int(np.where(ordd == top_pr)[0][0]) + 1,
        loser=titles[ids[top_deg]], loser_indeg=int(indeg[top_deg]),
        loser_pr=round(float(pr[top_deg]), 4),
        loser_pr_place=int(np.where(order == top_deg)[0][0]) + 1,
        winner_outdeg=int(outdeg[top_pr]), loser_outdeg=int(outdeg[top_deg]),
        share_from_loser=round(1 / int(outdeg[top_deg]), 4),
        share_from_winner=round(1 / int(outdeg[top_pr]), 4),
    )
    assert A[top_deg, top_pr] == 1 and A[top_pr, top_deg] == 1, "trilogy nodes link both ways"
    # how much the top-10 lists disagree
    deg10, pr10 = set(ordd[:10]), set(order[:10])
    top10_common = len(deg10 & pr10)

    # second eigenvalue of G = alpha P + (1-alpha) 1 v^T
    G = 0.85 * P + 0.15 * np.outer(np.ones(n), v)
    ev = np.sort(np.abs(np.linalg.eigvals(G)))[::-1]
    lam2 = float(ev[1])
    assert abs(ev[0] - 1) < 1e-9, ev[0]
    assert lam2 <= 0.85 + 1e-9, lam2

    # personalization: teleport onto Sci-Fi movies only (genre column 15)
    sci = np.array([genres[mid][15] for mid in ids], dtype=float)
    assert sci.sum() >= 10, sci.sum()
    v_sci = sci / sci.sum()
    pr_sci, _, _ = pagerank(P, 0.85, v_sci)
    order_s = np.argsort(-pr_sci)
    top5_sci = [(titles[ids[i]], round(float(pr_sci[i]), 4)) for i in order_s[:5]]
    top10, top10s = set(order[:10]), set(order_s[:10])
    jac = len(top10 & top10s) / len(top10 | top10s)
    newcomers = len(top10s - top10)
    new_names = [titles[ids[i]] for i in order_s[:10] if i not in top10]
    # a movie that is NOT sci-fi yet gains from a sci-fi teleport (flow travels along edges)
    gain = (pr_sci - pr) / pr
    non_sci = np.where(sci == 0)[0]
    g = int(non_sci[np.argmax(gain[non_sci] * (pr[non_sci] > 1 / n))])
    FACT["sci_spillover"] = [titles[ids[g]], round(float(gain[g]) * 100, 1),
                             round(float(pr[g]), 4), round(float(pr_sci[g]), 4)]
    assert gain[g] > 0, gain[g]
    FACT["sci_new_names"] = new_names

    # linearity of PageRank in the teleport vector
    lam = 0.5
    mixed, _, _ = pagerank(P, 0.85, lam * v + (1 - lam) * v_sci, tol=1e-14)
    lin_err = float(np.abs(mixed - (lam * pr + (1 - lam) * pr_sci)).max())
    assert lin_err < 1e-9, lin_err

    # top-10 stability under alpha
    alphas = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]
    jacs, iters = [], []
    base = set(order[:10])
    for a in alphas:
        pa, ita, _ = pagerank(P, a, v)
        oa = set(np.argsort(-pa)[:10])
        jacs.append(len(base & oa) / len(base | oa))
        iters.append(ita)
    jac_min = min(jacs)
    assert jac_min < 1.0, jacs

    FACT.update({
        "n_nodes": n, "n_edges": m, "min_likes": 45, "threshold": 0.6,
        "n_dangling": n_dangling,
        "outdeg_max": int(outdeg.max()), "outdeg_mean": round(float(outdeg.mean()), 2),
        "swap": swap, "top10_common_deg_pr": top10_common,
        "iters_070": it70, "iters_085": it85, "iters_095": it95,
        "top5_pagerank": top5, "top5_indegree": top5_deg,
        "spearman_deg_pr": round(rho, 3),
        "indeg_max": int(indeg.max()), "indeg_mean": round(float(indeg.mean()), 3),
        "hi_ratio": [titles[ids[hi]], int(indeg[hi]), round(float(pr[hi]), 4)],
        "lo_ratio": [titles[ids[lo]], int(indeg[lo]), round(float(pr[lo]), 4)],
        "lambda2_085": round(lam2, 4),
        "sci_count": int(sci.sum()),
        "top5_sci": top5_sci,
        "jaccard_sci": round(jac, 3), "newcomers_sci": newcomers,
        "linearity_err": lin_err,
        "alphas": alphas, "jaccard_alpha": [round(j, 3) for j in jacs],
        "iters_alpha": iters,
        "pr_min": round(float(pr.min()), 5), "pr_max": round(float(pr.max()), 5),
        "pr_ratio_maxmin": round(float(pr.max() / pr.min()), 1),
        "uniform": round(1 / n, 5),
        "dense_cells": n * n, "sparse_cells": m,
    })

    # ---- numbers quoted in the prose that are derived from the ones above
    ratio_sparse = n * n / m
    FACT["sparsity_ratio"] = round(float(ratio_sparse), 1)
    assert abs(round(ratio_sparse) - 41) == 0, ratio_sparse

    # all the mass of the in-degree champion goes to its single out-neighbour
    flow = 0.85 * pr[top_deg] / outdeg[top_deg]
    FACT["flow_from_indegree_champion"] = round(float(flow), 4)
    assert abs(flow - 0.1844) < 5e-05, flow

    # teleport floor (1 - alpha)/n and the measured minimum rank
    floor = 0.15 / n
    FACT["teleport_floor_085"] = round(float(floor), 5)
    assert pr.min() > floor, (pr.min(), floor)
    assert abs(round(floor, 5) - 0.00039) < 5e-06, floor

    # worst-case iteration bound 2 alpha^t <= 1e-6 for alpha = 0.85
    t_bound = np.log(2e6) / np.log(1 / 0.85)
    FACT["iter_bound_1e6_085"] = [round(float(t_bound), 1), int(np.ceil(t_bound))]
    assert abs(t_bound - 89.3) < 0.05, t_bound
    assert int(np.ceil(t_bound)) == 90, t_bound

    # mean session length 1/(1-alpha)
    FACT["session_length"] = {a: round(1 / (1 - a), 2) for a in (0.70, 0.85, 0.95)}
    assert abs(1 / (1 - 0.85) - 6.67) < 0.005
    assert abs(1 / (1 - 0.70) - 3.33) < 0.005
    assert abs(1 / (1 - 0.95) - 20.0) < 0.05
    return dict(ids=ids, titles=titles, A=A, P=P, v=v, pr=pr, indeg=indeg,
                hist=(hist70, hist85, hist95), pr_sci=pr_sci, order=order,
                order_s=order_s, alphas=alphas, jacs=jacs, iters=iters,
                hi=hi, lo=lo, lam2=lam2, top_pr=int(top_pr), top_deg=int(top_deg))


# ============================================================ figure 1: toy graph 3 views
def draw_graph(ax, pos, edges, labels, sizes=None, colors=None, title=None,
               dashed=(), curve=0.16):
    ax.set_aspect("equal"); ax.axis("off")
    for a, b in edges:
        x0, y0 = pos[a]; x1, y1 = pos[b]
        st = "dashed" if (a, b) in dashed else "solid"
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED if st == "solid" else FAINT,
                                    lw=1.5 if st == "solid" else 1.0, linestyle=st,
                                    shrinkA=17, shrinkB=19,
                                    connectionstyle=f"arc3,rad={curve}"))
    for i, lb in enumerate(labels):
        s = 700 if sizes is None else sizes[i]
        c = BLUE if colors is None else colors[i]
        ax.scatter(*pos[i], s=s, color=c, zorder=5, edgecolors=PAPER, linewidths=1.5)
        ax.text(pos[i][0], pos[i][1], lb, ha="center", va="center", color=PAPER,
                fontsize=11, zorder=6, fontweight="bold")
    if title:
        ax.set_title(title, fontsize=11.5)


def fig_toy(A, P, steps, pg):
    lab = ["A", "B", "C"]
    pos = {0: (0, 1), 1: (0.95, -0.6), 2: (-0.95, -0.6)}
    edges = [(0, 1), (0, 2), (1, 2), (2, 0)]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9))
    draw_graph(axes[0], pos, edges, lab, title="граф ссылок")
    axes[0].set_xlim(-1.7, 1.7); axes[0].set_ylim(-1.4, 1.7)

    ax = axes[1]
    ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            txt = "0" if P[i, j] == 0 else ("1" if P[i, j] == 1 else "1/2")
            ax.text(j, i, txt, ha="center", va="center",
                    color=PAPER if P[i, j] > 0.6 else INK, fontsize=12)
    ax.set_xticks(range(3), lab); ax.set_yticks(range(3), lab)
    ax.set_title("матрица переходов $P$", fontsize=11.5)
    ax.set_xlabel("куда"); ax.set_ylabel("откуда")

    ax = axes[2]
    x = np.arange(3)
    w = 0.26
    for t, (c, s) in enumerate(zip([FAINT, GOLD, BLUE], steps[:3])):
        ax.bar(x + (t - 1) * w, s, width=w, color=c, label=f"$p_{t}$")
    ax.plot(x, [0.4, 0.2, 0.4], "o", color=RED, ms=9, zorder=5, label=r"$\pi$")
    ax.set_xticks(x, lab); ax.set_ylim(0, 1.1)
    ax.set_ylabel("вероятность")
    ax.set_title("три шага и предел", fontsize=11.5)
    ax.legend(frameon=False, fontsize=9.5, ncol=2)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.suptitle("Один граф — три языка: рёбра, матрица, поток вероятности", y=1.03, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "toy_graph.png")


# ============================================================ figure 2: trap + teleport
def fig_trap(lab, A, p_noteleport, pg, edges):
    pos = {0: (-1.6, 1.0), 1: (-2.3, -0.3), 2: (-0.9, -0.5),
           3: (1.0, 1.0), 4: (1.9, -0.3), 5: (0.3, -0.5)}
    fig = plt.figure(figsize=(11.0, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[0, 0])
    sizes = [420 + 5200 * x for x in pg]
    cols = [BLUE, BLUE, BLUE, RED, RED, RED]
    draw_graph(ax, pos, edges, lab, sizes=sizes, colors=cols,
               title="ловушка $D,E,F$: войти можно, выйти нельзя")
    ax.set_xlim(-3.1, 2.8); ax.set_ylim(-1.4, 1.8)
    ax.text(1.05, 1.6, "ловушка", color=RED, fontsize=11, ha="center")

    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(6)
    ax.bar(x - 0.2, p_noteleport, width=0.4, color=FAINT, label=r"без телепортации ($\alpha=1$)")
    ax.bar(x + 0.2, pg, width=0.4, color=GREEN, label=r"PageRank, $\alpha=0{,}85$")
    ax.set_xticks(x, lab); ax.set_ylabel("вероятность")
    ax.set_title("телепортация возвращает массу в $A,B,C$", fontsize=11.5)
    ax.legend(frameon=False, fontsize=9.5)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "trap.png")


# ============================================================ figure 3: convergence
def fig_convergence(hist, iters):
    h70, h85, h95 = hist
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for h, a, c in [(h70, 0.70, BLUE), (h85, 0.85, GREEN), (h95, 0.95, RED)]:
        ax.semilogy(np.arange(1, len(h) + 1), h, color=c, lw=2.0,
                    label=f"$\\alpha={a:.2f}$ — {len(h)} итераций")
        t = np.arange(1, len(h) + 1)
        ax.semilogy(t, 2 * a ** t, color=c, lw=1.0, ls=(0, (3, 3)), alpha=0.7)
    ax.axhline(1e-12, color=MUTED, lw=0.9, ls=(0, (2, 2)))
    ax.text(2, 1.6e-12, "порог $10^{-12}$", fontsize=9.5, color=MUTED)
    ax.set_xlabel("номер итерации $t$")
    ax.set_ylabel(r"$\|p_{t+1}-p_t\|_1$")
    ax.set_title("Скорость сходимости задаёт $\\alpha$: ошибка падает как $\\alpha^t$")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "convergence.png")


# ============================================================ figure 4: degree vs rank
def fig_degree(R):
    pr, indeg, titles, ids = R["pr"], R["indeg"], R["titles"], R["ids"]
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    jitter = np.linspace(-0.12, 0.12, len(pr))
    ax.scatter(np.log10(1 + indeg) + jitter * 0.0, np.log10(pr), s=26,
               color=BLUE, alpha=0.45, edgecolors="none")
    for i, c, off, ha in [(R["top_pr"], RED, (-14, 10), "right"),
                          (R["top_deg"], GREEN, (-14, -30), "right"),
                          (R["lo"], GOLD, (12, -30), "left")]:
        ax.scatter(np.log10(1 + indeg[i]), np.log10(pr[i]), s=90, color=c, zorder=6)
        name = titles[ids[i]]
        name = name if len(name) < 30 else name[:28] + "…"
        ax.annotate(f"{name}\nвход {int(indeg[i])}, ранг {pr[i]:.4f}",
                    (np.log10(1 + indeg[i]), np.log10(pr[i])),
                    textcoords="offset points", xytext=off, ha=ha,
                    fontsize=9, color=c)
    ax.set_ylim(np.log10(pr).min() - 0.15, np.log10(pr).max() + 0.55)
    ax.set_xlim(-0.12, np.log10(1 + indeg).max() + 0.22)
    ax.axhline(np.log10(1 / len(pr)), color=MUTED, lw=1.0, ls=(0, (4, 3)))
    ax.text(0.02, np.log10(1 / len(pr)) + 0.05, "равномерный уровень $1/n$",
            fontsize=9.5, color=MUTED)
    ax.set_xlabel(r"$\log_{10}(1+d_i^{\mathrm{вх}})$")
    ax.set_ylabel(r"$\log_{10}\pi_i$")
    ax.set_title("Ранг растёт со степенью, но не определяется ею")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "degree_vs_rank.png")


# ============================================================ figure 5: personalization
def fig_personalization(R):
    pr, pr_s, titles, ids = R["pr"], R["pr_sci"], R["titles"], R["ids"]
    idxs = list(dict.fromkeys(list(R["order"][:8]) + list(R["order_s"][:8])))
    idxs = sorted(idxs, key=lambda i: -pr_s[i])[:12]
    FACT["personalization_points"] = len(idxs)   # quoted in the alt text
    assert len(idxs) == 9, len(idxs)
    names = []
    for i in idxs:
        t = titles[ids[i]]
        names.append(t if len(t) <= 28 else t[:26] + "…")
    y = np.arange(len(idxs))[::-1]
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    for k, i in enumerate(idxs):
        ax.plot([pr[i], pr_s[i]], [y[k], y[k]], color=LINE, lw=1.4, zorder=1)
    ax.scatter([pr[i] for i in idxs], y, s=60, color=BLUE, zorder=3, label="равномерная телепортация")
    ax.scatter([pr_s[i] for i in idxs], y, s=60, color=VIOLET, zorder=3, label="телепортация в фантастику")
    ax.set_yticks(y, names, fontsize=9.5)
    ax.set_xlabel(r"$\pi_i$")
    ax.set_title("Персонализация двигает не только числа, но и порядок")
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax.grid(True, axis="x", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "personalization.png")


# ============================================================ figure 6: alpha stability
def fig_alpha(R):
    a, j, it = R["alphas"], R["jacs"], R["iters"]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax0.plot(a, j, "o-", color=GREEN, lw=2.0, ms=7)
    ax0.set_ylim(0, 1.05)
    ax0.set_xlabel(r"$\alpha$"); ax0.set_ylabel("Жаккар с первой десяткой при $0{,}85$")
    ax0.set_title("верхушка устойчива внутри диапазона, ломается на краях", fontsize=11.5)
    ax0.grid(True, color=GRID, lw=0.4, alpha=0.5); ax0.set_axisbelow(True)
    ax1.plot(a, it, "o-", color=RED, lw=2.0, ms=7)
    ax1.set_xlabel(r"$\alpha$"); ax1.set_ylabel("итераций до $10^{-12}$")
    ax1.set_title("цена уважения к ссылкам", fontsize=11.5)
    ax1.grid(True, color=GRID, lw=0.4, alpha=0.5); ax1.set_axisbelow(True)
    fig.suptitle("Один параметр управляет и смыслом, и стоимостью", y=1.02, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "alpha_tradeoff.png")


# ============================================================ sidenote images
def side_decay():
    t = np.arange(0, 121)
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    for a, c in [(0.70, BLUE), (0.85, GREEN), (0.95, RED)]:
        ax.semilogy(t, a ** t, color=c, lw=1.7, label=f"$\\alpha={a:.2f}$")
    ax.axhline(1e-6, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    ax.set_xlabel("итерация", fontsize=9); ax.set_ylabel(r"$\alpha^t$", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("забывание старта", fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "decay.png")


def side_sparsity(A):
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    ii, jj = np.nonzero(A)
    ax.scatter(jj, ii, s=1.2, color=BLUE, alpha=0.7, edgecolors="none")
    ax.set_xlim(0, A.shape[0]); ax.set_ylim(A.shape[0], 0)
    ax.set_aspect("equal")
    ax.set_xlabel("куда", fontsize=9); ax.set_ylabel("откуда", fontsize=9)
    ax.set_title(f"{int(A.sum())} рёбер\nиз {A.shape[0]**2} клеток", fontsize=9.5)
    save(fig, SIDE / "sparsity.png")


def side_cost():
    n = np.logspace(3, 9, 60)
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    ax.loglog(n, n ** 2, color=RED, lw=1.8, label=r"плотная: $n^2$")
    ax.loglog(n, 10 * n, color=GREEN, lw=1.8, label=r"разреженная: $|E|\approx10n$")
    ax.set_xlabel("число страниц $n$", fontsize=9)
    ax.set_ylabel("операций на шаг", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("разреженность решает всё", fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4, which="both"); ax.set_axisbelow(True)
    save(fig, SIDE / "cost.png")


def side_dangling(mass):
    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    t = np.arange(len(mass))
    ax.plot(t, mass, "o-", color=RED, lw=1.8, ms=5)
    ax.axhline(1.0, color=GREEN, lw=1.2, ls=(0, (3, 3)))
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("шаг", fontsize=9); ax.set_ylabel(r"$\sum_i p_{t,i}$", fontsize=9)
    ax.set_title("тупик крадёт вероятность", fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "dangling.png")


# ============================================================ run
A3, P3, steps3, pg3 = toy_cycle()
labT, AT, PT, pT, pgT, edgesT = toy_trap()
massD = toy_dangling()
R = real_analysis()

fig_toy(A3, P3, steps3, pg3)
fig_trap(labT, AT, pT, pgT, edgesT)
fig_convergence(R["hist"], (FACT["iters_070"], FACT["iters_085"], FACT["iters_095"]))
fig_degree(R)
fig_personalization(R)
fig_alpha(R)
side_decay()
side_sparsity(R["A"])
side_cost()
side_dangling(massD)

FACTS.write_text(json.dumps(FACT, ensure_ascii=False, indent=2), encoding="utf8")
for k, val in FACT.items():
    print(f"{k}: {val}")
print("lesson 65 figures written")
