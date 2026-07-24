"""Deterministic figures for lesson 72: self-play, MCTS and the league.

Everything is computed, nothing is drawn by hand:

* exact win-rate matrices of mixed strategies in Rock-Paper-Scissors and in the
  five-action Rock-Paper-Scissors-Lizard-Spock game (closed form p^T A q);
* best-response self-play dynamics: the latest iterate stays maximally
  exploitable while the running average (the "league mixture") converges;
* PUCT selection at a two-action root, simulated visit by visit;
* real MCTS against a perfect minimax player in tic-tac-toe (seeded PCG64);
* Elo fitted by maximum likelihood to an eight-agent tournament, with residuals.

Every number quoted in the lesson text is printed and asserted here.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "72"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "72"
FACTS = ROOT / "scripts" / "data" / "lesson72_facts.json"

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

facts: dict[str, float] = {}


def save(fig, path, *, dpi=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def note(key, value):
    facts[key] = value if isinstance(value, (list, str)) else round(float(value), 6)
    return value


# ============================================================ games


def rps_matrix() -> np.ndarray:
    """rock(0) paper(1) scissors(2); A[i,j] = +1 if i beats j."""
    A = np.zeros((3, 3))
    for i, j in [(0, 2), (1, 0), (2, 1)]:
        A[i, j] = 1.0
        A[j, i] = -1.0
    return A


def rpsls_matrix() -> np.ndarray:
    """rock paper scissors lizard spock; each action beats exactly two others."""
    beats = {
        0: (2, 3),      # rock crushes scissors, crushes lizard
        1: (0, 4),      # paper covers rock, disproves spock
        2: (1, 3),      # scissors cut paper, decapitate lizard
        3: (4, 1),      # lizard poisons spock, eats paper
        4: (2, 0),      # spock smashes scissors, vaporizes rock
    }
    A = np.zeros((5, 5))
    for i, targets in beats.items():
        for j in targets:
            A[i, j] = 1.0
            A[j, i] = -1.0
    return A


def winrate(A: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    """Draws count as half a point, so win rate = 1/2 + 1/2 * p^T A q."""
    return 0.5 + 0.5 * float(p @ A @ q)


# ============================================================ fig 72.1


def fig_cycle():
    A = rps_matrix()
    names = ["равномерный", "камень-70%", "бумага (л.о.)", "ножницы (л.о.)"]
    short = ["U", "K", "P", "S"]
    strat = np.array([
        [1 / 3, 1 / 3, 1 / 3],
        [0.70, 0.15, 0.15],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    n = len(strat)
    M = np.array([[winrate(A, strat[i], strat[j]) for j in range(n)] for i in range(n)])

    note("m_PK", M[2, 1]); note("m_SP", M[3, 2]); note("m_KS", M[1, 3])
    note("m_UK", M[0, 1])
    mean_rates = np.array([(M[i].sum() - 0.5) / (n - 1) for i in range(n)])
    worst = np.array([min(M[i, j] for j in range(n) if j != i) for i in range(n)])
    for i, s in enumerate(short):
        note(f"mean_{s}", mean_rates[i]); note(f"worst_{s}", worst[i])
    assert abs(M[2, 1] - 0.775) < 1e-9, M[2, 1]
    assert abs(M[3, 2] - 1.0) < 1e-9
    assert abs(M[1, 3] - 0.775) < 1e-9
    assert abs(M[0, 1] - 0.5) < 1e-9
    # cycle P -> S -> K -> P among the three non-uniform players
    assert M[3, 2] > 0.5 and M[1, 3] > 0.5 and M[2, 1] > 0.5

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    im = ax.imshow(M, cmap="RdYlBu_r", vmin=0.0, vmax=1.0)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                    color=INK if 0.25 < M[i, j] < 0.78 else PAPER, fontsize=11)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=22, ha="right", fontsize=10)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_title("Win rate строки против столбца")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)

    ax = axes[1]
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.25, 1.45); ax.axis("off")
    pos = {2: (0.0, 1.0), 3: (0.95, -0.55), 1: (-0.95, -0.55)}
    for i, (x, y) in pos.items():
        ax.add_patch(plt.Circle((x, y), 0.28, color=WASH, ec=LINE, lw=1.2, zorder=3))
        ax.text(x, y, short[i], ha="center", va="center", fontsize=15,
                color=INK, zorder=4)
    edges = [(3, 2), (1, 3), (2, 1)]
    for a, b in edges:
        xa, ya = pos[a]; xb, yb = pos[b]
        dx, dy = xb - xa, yb - ya
        L = math.hypot(dx, dy)
        ux, uy = dx / L, dy / L
        ax.annotate("", xy=(xb - 0.31 * ux, yb - 0.31 * uy),
                    xytext=(xa + 0.31 * ux, ya + 0.31 * uy),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.2,
                                    shrinkA=0, shrinkB=0))
        mx, my = (xa + xb) / 2 + 0.16 * uy, (ya + yb) / 2 - 0.16 * ux
        ax.text(mx, my, f"{M[a, b]:.2f}", ha="center", va="center",
                fontsize=10, color=RED)
    ax.text(0.0, -1.15, "стрелка = «побеждает чаще половины»",
            ha="center", fontsize=10, color=MUTED)
    ax.set_title("Граф доминирования замкнут в цикл")
    fig.suptitle("Победа над предшественником не выстраивает лестницу", fontsize=15, y=1.02)
    save(fig, OUT / "cycle_matrix.png")
    print("fig1 cycle:", np.round(M, 3).tolist())
    return M, mean_rates, worst


# ============================================================ fig 72.2


def fig_selfplay():
    A = rpsls_matrix()
    K = A.shape[0]
    T = 2000

    # (a) best response to the LATEST opponent strategy: a pure cycle
    cur = np.zeros(K); cur[0] = 1.0
    latest_seq = [0]
    for _ in range(T):
        a = int(np.argmax(A @ cur))
        cur = np.zeros(K); cur[a] = 1.0
        latest_seq.append(a)
    period = None
    for p in range(1, 30):
        tail = latest_seq[-60:]
        if all(tail[i] == tail[i + p] for i in range(len(tail) - p)):
            period = p
            break
    note("latest_period", period)
    # ties in argmax are broken by the smaller index, so the orbit closes on the
    # three classic actions inside the five-action game
    assert period == 3, period
    note("latest_orbit", [int(a) for a in latest_seq[:7]])

    # (b) fictitious play: best response to the AVERAGE of the archive
    hist = np.zeros(K); hist[0] = 1.0
    expl_last, expl_avg, steps = [], [], []
    for t in range(1, T + 1):
        avg = hist / hist.sum()
        a = int(np.argmax(A @ avg))
        last = np.zeros(K); last[a] = 1.0
        hist[a] += 1.0
        avg2 = hist / hist.sum()
        expl_last.append(float(np.max(A @ last)))
        expl_avg.append(float(np.max(A @ avg2)))
        steps.append(t)
    e_last = float(np.mean(expl_last))
    e_avg_10 = expl_avg[9]; e_avg_200 = expl_avg[199]; e_avg_2000 = expl_avg[1999]
    note("expl_last_mean", e_last)
    note("expl_avg_10", e_avg_10)
    note("expl_avg_200", e_avg_200)
    note("expl_avg_2000", e_avg_2000)
    assert abs(e_last - 1.0) < 1e-12, e_last
    assert e_avg_2000 < 0.02, e_avg_2000
    assert e_avg_10 > e_avg_200 > e_avg_2000
    # отношения, которые цитируются в прозе словами
    ratio_10_2000 = e_avg_10 / e_avg_2000
    ratio_last_2000 = e_last / e_avg_2000
    note("expl_ratio_10_over_2000", ratio_10_2000)
    note("expl_ratio_last_over_avg2000", ratio_last_2000)
    assert 14.0 < ratio_10_2000 < 15.0, ratio_10_2000     # «почти в пятнадцать раз»
    assert 50.0 < ratio_last_2000 < 60.0, ratio_last_2000  # «более чем в пятьдесят раз»

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    ax = axes[0]
    labels = ["камень", "бумага", "ножницы", "ящерица", "Спок"]
    ax.step(range(41), latest_seq[:41], where="post", color=RED, lw=2.0)
    ax.scatter(range(41), latest_seq[:41], s=18, color=RED, zorder=4)
    ax.set_yticks(range(5)); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("итерация self-play")
    ax.set_title(f"Лучший ответ на последнюю версию:\nчистый цикл периода {period}", fontsize=13)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)

    ax = axes[1]
    ax.plot(steps, expl_last, color=RED, lw=2.2, label="последняя версия")
    ax.plot(steps, expl_avg, color=BLUE, lw=2.2, label="средняя по архиву (лига)")
    ax.plot(steps, [1.0 / t for t in steps], color=MUTED, lw=1.2, ls=(0, (4, 3)),
            label=r"$1/t$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("число итераций $t$"); ax.set_ylabel("эксплуатируемость")
    ax.set_title("Чемпион уязвим, смесь архива — нет", fontsize=13)
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.suptitle("Self-play против зеркала вращается, self-play против архива сходится",
                 fontsize=15, y=1.03)
    save(fig, OUT / "selfplay_exploitability.png")
    print(f"fig2: period={period} expl_last={e_last:.3f} "
          f"expl_avg(10,200,2000)=({e_avg_10:.4f},{e_avg_200:.4f},{e_avg_2000:.4f})")


# ============================================================ fig 72.3


def fig_puct():
    Q = np.array([0.20, 0.35])
    P = np.array([0.70, 0.30])
    c = 1.0
    N = np.array([20.0, 5.0])
    Ntot = N.sum()

    def puct(N, Ntot):
        return Q + c * P * math.sqrt(Ntot) / (1.0 + N)

    u0 = puct(N, Ntot)
    note("puct_a1_small", u0[0]); note("puct_a2_small", u0[1])
    assert abs(u0[0] - (0.20 + 0.7 * 5.0 / 21.0)) < 1e-12
    assert abs(u0[1] - (0.35 + 0.3 * 5.0 / 6.0)) < 1e-12
    note("puct_choice_small", 1 if u0[0] >= u0[1] else 2)

    N2 = np.array([320.0, 80.0])
    u1 = puct(N2, 400.0)
    note("puct_a1_big", u1[0]); note("puct_a2_big", u1[1])
    note("puct_choice_big", 1 if u1[0] >= u1[1] else 2)
    assert u0[1] > u0[0] and u1[1] > u1[0]

    # honest simulation of the selection loop from an empty root
    n = np.zeros(2)
    share, prior_frac = [], []
    total = 600
    for t in range(1, total + 1):
        idx = int(np.argmax(Q + c * P * math.sqrt(max(t - 1, 1)) / (1.0 + n)))
        n[idx] += 1
        share.append(n[0] / n.sum())
        bonus = c * P * math.sqrt(max(t - 1, 1)) / (1.0 + n)
        prior_frac.append(float(np.mean(bonus / (np.abs(Q) + bonus))))
    final_share = share[-1]
    note("puct_share_a1_600", final_share)
    note("puct_share_a1_30", share[29])
    assert share[29] > final_share, (share[29], final_share)
    assert 0.15 < final_share < 0.55, final_share

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    ax = axes[0]
    ns = np.arange(0, 60)
    for k, (col, lab) in enumerate([(BLUE, "действие 1: $Q=0{,}20$, $P=0{,}70$"),
                                    (RED, "действие 2: $Q=0{,}35$, $P=0{,}30$")]):
        ax.plot(ns, Q[k] + c * P[k] * math.sqrt(100.0) / (1.0 + ns), color=col, lw=2.2,
                label=lab)
        ax.axhline(Q[k], color=col, lw=1.0, ls=(0, (4, 3)), alpha=0.7)
    ax.set_xlabel("$N(s,a)$ — сколько раз ветвь уже посещали")
    ax.set_ylabel("индекс PUCT")
    ax.set_title(r"Приор ведёт вначале, $Q$ — потом ($\sum_b N=100$)", fontsize=13)
    ax.legend(frameon=False, fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax.text(34, Q[0] + 0.035, "пунктир — чистое $Q$", fontsize=9.5, color=MUTED)

    ax = axes[1]
    ax.plot(range(1, total + 1), share, color=VIOLET, lw=2.2)
    ax.axhline(P[0], color=GOLD, lw=1.4, ls=(0, (5, 3)))
    ax.text(total * 0.42, P[0] + 0.02, "приор действия 1 = 0,70", color=GOLD, fontsize=10)
    ax.axhline(final_share, color=MUTED, lw=1.0, ls=":")
    ax.text(12, final_share - 0.07, f"доля к {total}-й симуляции = {final_share:.2f}",
            color=MUTED, fontsize=10)
    ax.set_xlabel("номер симуляции"); ax.set_ylabel("доля посещений действия 1")
    ax.set_ylim(0, 1)
    ax.set_title("Счётчики уходят от приора к фактам", fontsize=13)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.suptitle("PUCT: бандит внутри каждого узла дерева", fontsize=15, y=1.03)
    save(fig, OUT / "puct_prior.png")
    print(f"fig3: puct small ({u0[0]:.4f},{u0[1]:.4f}) big ({u1[0]:.4f},{u1[1]:.4f}) "
          f"share30={share[29]:.3f} share600={final_share:.3f}")


# ============================================================ tic-tac-toe engine

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7),
         (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def winner(board):
    for a, b, c in LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return 0


def moves(board):
    return [i for i in range(9) if board[i] == 0]


@lru_cache(maxsize=None)
def minimax(board, player):
    """Value for `player` to move, in {-1,0,1}."""
    w = winner(board)
    if w:
        return 1 if w == player else -1
    if not moves(board):
        return 0
    best = -2
    for m in moves(board):
        nb = list(board); nb[m] = player
        v = -minimax(tuple(nb), -player)
        if v > best:
            best = v
    return best


@lru_cache(maxsize=None)
def minimax_move(board, player):
    best, bm = -2, None
    for m in moves(board):
        nb = list(board); nb[m] = player
        v = -minimax(tuple(nb), -player)
        if v > best:
            best, bm = v, m
    return bm


def mcts_move(board, player, budget, rng, c=1.4):
    """Plain UCT with random rollouts; returns the most visited root child."""
    root_moves = moves(board)
    N = np.zeros(len(root_moves))
    Wsum = np.zeros(len(root_moves))
    for sim in range(budget):
        # selection at the root only (one-ply tree with rollouts below)
        if 0 in N:
            k = int(np.argmin(N > 0)) if np.any(N == 0) else 0
            k = int(np.where(N == 0)[0][0])
        else:
            tot = N.sum()
            k = int(np.argmax(Wsum / N + c * np.sqrt(np.log(tot) / N)))
        nb = list(board); nb[root_moves[k]] = player
        # rollout
        cur, turn = nb, -player
        w = winner(tuple(cur))
        while not w and moves(cur):
            avail = moves(cur)
            cur = list(cur)
            cur[avail[rng.integers(len(avail))]] = turn
            turn = -turn
            w = winner(tuple(cur))
        z = 0.0 if w == 0 else (1.0 if w == player else -1.0)
        N[k] += 1
        Wsum[k] += z
    return root_moves[int(np.argmax(N))]


def play_game(budget, mcts_is_x, rng):
    board = [0] * 9
    player = 1
    while True:
        w = winner(tuple(board))
        if w:
            return w
        if not moves(board):
            return 0
        mcts_turn = (player == 1) == mcts_is_x
        if mcts_turn:
            m = mcts_move(tuple(board), player, budget, rng)
        else:
            m = minimax_move(tuple(board), player)
        board[m] = player
        player = -player


def fig_mcts_budget():
    budgets = [10, 50, 200, 800]
    games = 200
    losses, draws = [], []
    for B in budgets:
        rng = np.random.default_rng(72000 + B)
        loss = draw = 0
        for g in range(games):
            mcts_is_x = (g % 2 == 0)
            w = play_game(B, mcts_is_x, rng)
            mcts_side = 1 if mcts_is_x else -1
            if w == 0:
                draw += 1
            elif w == mcts_side:
                raise AssertionError("perfect minimax cannot lose")
            else:
                loss += 1
        losses.append(loss / games); draws.append(draw / games)
        print(f"  budget {B:4d}: loss {loss/games:.3f} draw {draw/games:.3f}")
    for B, l in zip(budgets, losses):
        note(f"mcts_loss_{B}", l)
    note("mcts_games_per_budget", games)
    assert losses[0] > losses[-1], losses
    assert losses[-1] < 0.35, losses
    assert losses[0] > 0.45, losses
    for B, d in zip(budgets, draws):
        note(f"mcts_draw_{B}", d)
    loss_ratio = losses[0] / losses[-1]
    budget_ratio = budgets[-1] / budgets[0]
    note("mcts_loss_ratio_10_over_800", loss_ratio)
    note("mcts_budget_ratio", float(budget_ratio))
    assert abs(budget_ratio - 80.0) < 1e-9, budget_ratio   # «в 80 раз»
    assert 38.0 < loss_ratio < 39.0, loss_ratio            # «почти в 39 раз»
    for B, l, d in zip(budgets, losses, draws):
        assert abs(l + d - 1.0) < 1e-12, (B, l, d)         # побед нет вовсе

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    x = np.arange(len(budgets))
    ax.bar(x - 0.19, losses, width=0.38, color=RED, label="поражения MCTS")
    ax.bar(x + 0.19, draws, width=0.38, color=GREEN, label="ничьи (оптимум)")
    for i, (l, d) in enumerate(zip(losses, draws)):
        ax.text(i - 0.19, l + 0.015, f"{l:.2f}", ha="center", fontsize=11, color=RED)
        ax.text(i + 0.19, d + 0.015, f"{d:.2f}", ha="center", fontsize=11, color=GREEN)
    ax.set_xticks(x); ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("бюджет симуляций на ход")
    ax.set_ylabel(f"доля из {games} партий")
    ax.set_ylim(0, 1.05)
    ax.set_title("MCTS против идеального minimax: бюджет поиска = сила игры")
    ax.legend(frameon=False, fontsize=10, loc="upper center", ncol=2)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax.text(0.02, 0.02, "побед у MCTS нет ни разу: идеальная игра непобедима",
            transform=ax.transAxes, fontsize=10, color=MUTED)
    save(fig, OUT / "mcts_budget.png")
    return budgets, losses, draws


# ============================================================ fig 72.5


def league_strategies():
    names = ["равномерный", "камень-60", "бумага-60", "ножницы-60",
             "ящерица-60", "Спок-60", "камень+Спок", "ящерица+ножницы"]
    S = []
    S.append(np.full(5, 1 / 5))
    for k in range(5):
        p = np.full(5, 0.10); p[k] = 0.60
        S.append(p)
    p = np.zeros(5); p[0] = 0.5; p[4] = 0.5; S.append(p)
    p = np.zeros(5); p[3] = 0.5; p[2] = 0.5; S.append(p)
    S = np.array(S)
    assert np.allclose(S.sum(axis=1), 1.0)
    return names, S


def fit_elo(M):
    """Maximum likelihood Elo (Bradley-Terry) on the win-rate matrix."""
    n = M.shape[0]
    r = np.zeros(n)
    for _ in range(20000):
        g = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                p = 1.0 / (1.0 + math.exp(-(r[i] - r[j])))
                g[i] += M[i, j] - p
        r += 0.05 * g
        r -= r.mean()
    return r


def fig_elo_residual():
    A = rpsls_matrix()
    names, S = league_strategies()
    n = len(S)
    M = np.array([[winrate(A, S[i], S[j]) for j in range(n)] for i in range(n)])
    assert np.allclose(M + M.T, 1.0)
    r = fit_elo(M)
    Pred = 1.0 / (1.0 + np.exp(-(r[:, None] - r[None, :])))
    np.fill_diagonal(Pred, 0.5)
    R = M - Pred
    np.fill_diagonal(R, 0.0)
    off = ~np.eye(n, dtype=bool)
    spread = float(400.0 / math.log(10.0) * (r.max() - r.min()))
    rms = float(np.sqrt(np.mean(R[off] ** 2)))
    mx = float(np.max(np.abs(R)))
    note("elo_spread_points", spread)
    note("elo_resid_rms", rms)
    note("elo_resid_max", mx)
    note("league_max_offdiag", float(M[off].max()))
    note("league_min_offdiag", float(M[off].min()))
    assert spread < 80.0, spread
    assert rms > 0.08, rms
    # цикл трёх стратегий с p=0,7: разность рейтингов и невязка по кругу
    d70 = 400.0 * math.log10(0.7 / 0.3)
    note("elo_gap_at_70", d70)
    note("elo_cycle_sum_at_70", 3.0 * d70)
    assert abs(d70 - 147.2) < 0.05, d70
    assert abs(3.0 * d70 - 441.6) < 0.15, 3.0 * d70
    print(f"fig5: elo spread {spread:.2f} pts, resid rms {rms:.3f}, max {mx:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.8),
                             gridspec_kw={"width_ratios": [1.15, 0.55, 1.15],
                                          "wspace": 0.85})
    ax = axes[0]
    im = ax.imshow(M, cmap="RdYlBu_r", vmin=0.2, vmax=0.8)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=55, ha="right", fontsize=8.5)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_title("Матрица попарных win rate", fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)

    ax = axes[1]
    order = np.argsort(-r)
    ax.barh(range(n), (400.0 / math.log(10.0)) * r[order], color=BLUE)
    ax.set_yticks(range(n)); ax.set_yticklabels([names[i] for i in order], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Elo, очки")
    ax.set_title(f"Elo: весь разброс\n{spread:.1f} очка", fontsize=13)
    ax.grid(True, axis="x", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)

    ax = axes[2]
    im = ax.imshow(R, cmap="PuOr", vmin=-0.32, vmax=0.32)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=55, ha="right", fontsize=8.5)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_title(f"Остатки: наблюдение − Elo\n(RMS {rms:.3f})", fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    fig.suptitle("Одно число не вмещает турнир: рейтинг плоский, остатки — нет",
                 fontsize=15, y=1.04)
    save(fig, OUT / "elo_residual.png")
    return names, M, r


# ============================================================ fig 72.6


def maxmin_mixture(M, iters=400000):
    """Fictitious play on the meta-game: returns a maxmin mixture over agents."""
    n = M.shape[0]
    payoff = M - 0.5                      # zero-sum, symmetric
    cntA = np.zeros(n); cntB = np.zeros(n)
    cntA[0] = 1.0; cntB[0] = 1.0
    for _ in range(iters):
        a = int(np.argmax(payoff @ (cntB / cntB.sum())))
        b = int(np.argmin((cntA / cntA.sum()) @ payoff))
        cntA[a] += 1; cntB[b] += 1
    return cntA / cntA.sum()


def fig_league_choice(names, M):
    n = M.shape[0]
    off = ~np.eye(n, dtype=bool)
    mean = np.array([M[i, off[i]].mean() for i in range(n)])
    worst = np.array([M[i, off[i]].min() for i in range(n)])
    # равновесная (равномерная) стратегия исключена из лиги: интересен вопрос,
    # соберут ли односторонние стили устойчивую смесь без неё
    sub = M[1:, 1:]
    mix_sub = maxmin_mixture(sub)
    mix = np.zeros(n); mix[1:] = mix_sub
    mix_scores = mix @ M                       # смесь против каждого чистого
    mix_worst = float(mix_scores.min())
    note("league_mean_best", float(mean.max()))
    note("league_mean_best_name", names[int(np.argmax(mean))])
    note("league_worst_best", float(worst.max()))
    note("league_worst_best_name", names[int(np.argmax(worst))])
    note("league_worst_of_mean_best", float(worst[int(np.argmax(mean))]))
    note("mix_worst", mix_worst)
    note("mix_support", int((mix > 0.01).sum()))
    note("mix_best_single_worst", float(worst[1:].max()))
    note("mix_worst_vs_best_single", mix_worst - float(worst[1:].max()))
    assert mix_worst > worst[1:].max() + 0.1, (mix_worst, worst[1:].max())
    assert 4 <= int((mix > 0.01).sum()) <= 7
    print(f"fig6: best mean {mean.max():.3f} ({names[int(np.argmax(mean))]}), "
          f"best worst-case {worst.max():.3f}, mixture worst {mix_worst:.4f}, "
          f"support {(mix>0.01).sum()}")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0),
                             gridspec_kw={"width_ratios": [1.3, 1.0]})
    ax = axes[0]
    x = np.arange(n)
    ax.bar(x - 0.2, mean, width=0.4, color=BLUE, label="средний win rate")
    ax.bar(x + 0.2, worst, width=0.4, color=RED, label="худший случай")
    ax.axhline(0.5, color=MUTED, lw=1.0, ls=(0, (4, 3)))
    ax.axhline(mix_worst, color=GREEN, lw=2.0)
    ax.text(3.4, mix_worst + 0.145, f"худший случай смеси = {mix_worst:.3f}",
            ha="center", color=GREEN, fontsize=10)
    ax.annotate("", xy=(3.4, mix_worst + 0.01), xytext=(3.4, mix_worst + 0.135),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.3))
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=55, ha="right", fontsize=9)
    ax.set_ylabel("win rate"); ax.set_ylim(0, 0.72)
    ax.set_title("Средняя сила и худший случай — разные шкалы", fontsize=13)
    ax.legend(frameon=False, fontsize=10, ncol=2, loc="upper left")
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)

    ax = axes[1]
    ax.bar(np.arange(n), mix, color=VIOLET)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(names, rotation=55, ha="right", fontsize=9)
    ax.set_ylabel("вес в максиминной смеси")
    ax.set_title("Чемпион — не агент, а смесь лиги", fontsize=13)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.suptitle("Три правила выбора чемпиона дают три разных ответа", fontsize=15, y=1.04)
    save(fig, OUT / "league_choice.png")
    return mean, worst, mix


# ============================================================ sidenote images


def side_interval():
    def wilson(k, n, z=1.96):
        ph = k / n
        d = 1 + z * z / n
        c = (ph + z * z / (2 * n)) / d
        h = z / d * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
        return c - h, c + h

    rows = [(11, 20), (55, 100), (220, 400)]
    res = []
    for k, n in rows:
        lo, hi = wilson(k, n)
        res.append((n, k / n, lo, hi, hi - lo))
    note("ci20_lo", res[0][2]); note("ci20_hi", res[0][3]); note("ci20_w", res[0][4])
    note("ci400_lo", res[2][2]); note("ci400_hi", res[2][3]); note("ci400_w", res[2][4])
    assert res[0][2] < 0.5 < res[0][3], res[0]
    assert res[2][2] > 0.5, res[2]
    print(f"side interval: n=20 [{res[0][2]:.3f},{res[0][3]:.3f}], "
          f"n=400 [{res[2][2]:.3f},{res[2][3]:.3f}]")

    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    for i, (n, ph, lo, hi, w) in enumerate(res):
        col = RED if lo < 0.5 else GREEN
        ax.plot([lo, hi], [i, i], color=col, lw=3.2, solid_capstyle="round")
        ax.plot([ph], [i], "o", color=INK, ms=6)
        ax.text(hi + 0.012, i, f"ширина {w:.2f}", va="center", fontsize=9, color=MUTED)
    ax.axvline(0.5, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax.set_yticks(range(3)); ax.set_yticklabels(["20 партий", "100 партий", "400 партий"],
                                                fontsize=10)
    ax.set_xlim(0.2, 0.85); ax.set_xlabel("win rate, интервал Уилсона 95%")
    ax.set_title("55% побед: сколько партий нужно?", fontsize=12)
    ax.grid(True, axis="x", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "interval.png")


def side_visits():
    rng = np.random.default_rng(7211)
    board = tuple([0] * 9)
    budget = 4000
    root_moves = moves(list(board))
    N = np.zeros(9)
    Wsum = np.zeros(9)
    cnt = np.zeros(9)
    for sim in range(budget):
        if np.any(cnt[root_moves] == 0):
            k = root_moves[int(np.where(cnt[root_moves] == 0)[0][0])]
        else:
            tot = cnt[root_moves].sum()
            vals = [Wsum[m] / cnt[m] + 1.4 * math.sqrt(math.log(tot) / cnt[m])
                    for m in root_moves]
            k = root_moves[int(np.argmax(vals))]
        nb = list(board); nb[k] = 1
        cur, turn = nb, -1
        w = winner(tuple(cur))
        while not w and moves(cur):
            avail = moves(cur)
            cur = list(cur)
            cur[avail[rng.integers(len(avail))]] = turn
            turn = -turn
            w = winner(tuple(cur))
        z = 0.0 if w == 0 else (1.0 if w == 1 else -1.0)
        cnt[k] += 1; Wsum[k] += z
    pi = cnt / cnt.sum()
    note("visit_center", float(pi[4]))
    note("visit_corner", float(pi[0]))
    note("visit_edge", float(pi[1]))
    assert pi[4] == pi.max(), pi
    assert pi[4] > pi[1]
    print("side visits:", np.round(pi, 3).reshape(3, 3).tolist())

    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    im = ax.imshow(pi.reshape(3, 3), cmap="YlGnBu", vmin=0)
    for i in range(3):
        for j in range(3):
            v = pi[3 * i + j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=PAPER if v > pi.max() * 0.6 else INK, fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"$\\pi$ из счётчиков, {budget} симуляций", fontsize=12)
    save(fig, SIDE / "visits.png")


def side_firstmove():
    rng = np.random.default_rng(7212)
    games = 200000
    xw = ow = dr = 0
    for _ in range(games):
        board = [0] * 9
        turn = 1
        w = 0
        while not w and moves(board):
            avail = moves(board)
            board[avail[rng.integers(len(avail))]] = turn
            turn = -turn
            w = winner(tuple(board))
        if w == 1:
            xw += 1
        elif w == -1:
            ow += 1
        else:
            dr += 1
    px, po, pd = xw / games, ow / games, dr / games
    note("rand_x", px); note("rand_o", po); note("rand_draw", pd)
    note("rand_gap", px - po)
    assert px > po + 0.25, (px, po)
    print(f"side firstmove: X {px:.3f} O {po:.3f} draw {pd:.3f}")

    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    ax.barh([2, 1, 0], [px, pd, po], color=[BLUE, MUTED, RED], height=0.6)
    for y, v in zip([2, 1, 0], [px, pd, po]):
        ax.text(v + 0.008, y, f"{v*100:.1f}%", va="center", fontsize=11, color=INK)
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels(["победа X", "ничья", "победа O"],
                                                 fontsize=10)
    ax.set_xlim(0, 0.72); ax.set_xlabel(f"доля из {games//1000} тыс. случайных партий")
    ax.set_title("Право первого хода само по себе\nдаёт перевес", fontsize=12)
    ax.grid(True, axis="x", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "firstmove.png")


# ============================================================ main


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SIDE.mkdir(parents=True, exist_ok=True)
    print("== fig 1"); fig_cycle()
    print("== fig 2"); fig_selfplay()
    print("== fig 3"); fig_puct()
    print("== fig 4"); fig_mcts_budget()
    print("== fig 5"); names, M, r = fig_elo_residual()
    print("== fig 6"); fig_league_choice(names, M)
    print("== sidenotes"); side_interval(); side_visits(); side_firstmove()
    FACTS.write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nfacts ->", FACTS)
    for k, v in facts.items():
        print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
