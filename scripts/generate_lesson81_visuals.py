"""Deterministic figures for lesson 81: preference optimization (PPO, DPO, GRPO).

Real data: MovieLens 100K (scripts/data/ml-100k). Half of the users play the role of
annotators: from their ratings we build a small set of pairwise comparisons and fit a
Bradley-Terry reward model. The other half of the users are the independent judges: their
mean rating of a film is the "true" value the reward model is only a proxy for.

The policy over films is pi_beta(y) ~ pi_ref(y) exp(r(y)/beta). Sweeping beta traces the
Goodhart curve: proxy reward grows monotonically with KL while the independent value peaks
and falls. Every number quoted in the lesson is computed here and asserted.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ML = ROOT / "scripts" / "data" / "ml-100k"
OUT = ROOT / "public" / "figures" / "lessons" / "81"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "81"
FACTS = ROOT / "scripts" / "data" / "lesson81_facts.json"

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

F: dict[str, float | int | str] = {}


def save(fig, path, *, dpi=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def kl(p, q):
    m = p > 1e-15
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


# ------------------------------------------------------------------ real data
# Annotators = real users aged <= 25, independent judges = real users aged >= 35.
def load():
    raw = np.loadtxt(ML / "u.data", dtype=int)          # user item rating ts
    age = {}
    with open(ML / "u.user", encoding="latin-1") as f:
        for row in f:
            p = row.split("|")
            age[int(p[0])] = int(p[1])
    ages = np.array([age[int(u)] for u in raw[:, 0]])
    ann, judge = raw[ages <= 25], raw[ages >= 35]
    titles = {}
    with open(ML / "u.item", encoding="latin-1") as f:
        for row in f:
            p = row.split("|")
            titles[int(p[0])] = p[1]
    return ann, judge, titles, len(np.unique(ann[:, 0])), len(np.unique(judge[:, 0]))


ANN, JUDGE, TITLES, N_ANN, N_JUDGE = load()
F["n_ratings"] = int(ANN.shape[0] + JUDGE.shape[0])
F["n_annotators"] = N_ANN
F["n_judges"] = N_JUDGE

# candidate "responses": K most rated films (by annotators)
K = 40
items, counts = np.unique(ANN[:, 1], return_counts=True)
order = np.argsort(-counts)[:K]
CAND = items[order]
CNT = counts[order].astype(float)
IDX = {int(it): j for j, it in enumerate(CAND)}
F["n_candidates"] = K
F["min_cand_count_ann"] = int(CNT.min())

# true value = mean rating of independent judges
val = np.zeros(K)
jc = np.zeros(K)
for u, it, r, _ in JUDGE:
    j = IDX.get(int(it))
    if j is not None:
        val[j] += r
        jc[j] += 1
assert jc.min() >= 40, jc.min()
val = val / jc
F["min_cand_count_judge"] = int(jc.min())
F["true_best_title"] = TITLES[int(CAND[int(np.argmax(val))])]
F["true_best_value"] = round(float(val.max()), 3)
F["true_worst_value"] = round(float(val.min()), 3)

# ------------------------------------------------------- small preference set
def sample_pairs(data, n_pairs, seed):
    """n_pairs comparisons: one annotator, two candidate films with different ratings."""
    by_user: dict[int, list[tuple[int, int]]] = {}
    for u, it, r, _ in data:
        j = IDX.get(int(it))
        if j is not None:
            by_user.setdefault(int(u), []).append((j, int(r)))
    us = sorted(u for u, v in by_user.items() if len(v) >= 2)
    rng = np.random.default_rng(seed)
    pairs = []
    while len(pairs) < n_pairs:
        u = us[rng.integers(len(us))]
        lst = by_user[u]
        a, b = rng.choice(len(lst), size=2, replace=False)
        (ja, ra), (jb, rb) = lst[a], lst[b]
        if ra == rb:
            continue
        pairs.append((ja, jb) if ra > rb else (jb, ja))
    return np.array(pairs), len(us)


N_PAIRS = 2000
PAIRS, N_ANN_USED = sample_pairs(ANN, N_PAIRS, seed=810)
ANN_TEST, _ = sample_pairs(ANN, 4000, seed=814)      # same taste, unseen comparisons
TEST_PAIRS, _ = sample_pairs(JUDGE, 4000, seed=811)  # другая группа людей
F["n_pairs"] = N_PAIRS
F["n_test_pairs"] = int(TEST_PAIRS.shape[0])


def fit_bt(pairs, k, l2=1e-2, steps=4000, lr=0.5):
    s = np.zeros(k)
    for _ in range(steps):
        d = s[pairs[:, 0]] - s[pairs[:, 1]]
        p = 1.0 / (1.0 + np.exp(-d))
        g = np.zeros(k)
        np.add.at(g, pairs[:, 0], (1 - p))
        np.add.at(g, pairs[:, 1], -(1 - p))
        g = g / len(pairs) - l2 * s
        s = s + lr * g
    return s - s.mean()


R = fit_bt(PAIRS, K)
F["reward_max"] = round(float(R.max()), 3)
F["reward_min"] = round(float(R.min()), 3)
F["proxy_best_title"] = TITLES[int(CAND[int(np.argmax(R))])]
F["proxy_best_true_value"] = round(float(val[int(np.argmax(R))]), 3)

acc_tr = float(np.mean(R[PAIRS[:, 0]] > R[PAIRS[:, 1]]))
acc_own = float(np.mean(R[ANN_TEST[:, 0]] > R[ANN_TEST[:, 1]]))
acc_te = float(np.mean(R[TEST_PAIRS[:, 0]] > R[TEST_PAIRS[:, 1]]))
F["pair_acc_train"] = round(acc_tr, 3)
F["pair_acc_own_group"] = round(acc_own, 3)
F["pair_acc_test"] = round(acc_te, 3)
assert 0.55 < acc_te < 0.80, acc_te
assert acc_own > acc_te                                # своя группа предсказывается лучше

PI_REF = CNT / CNT.sum()
F["ref_top_share"] = round(float(PI_REF.max()), 3)
F["ref_value"] = round(float(PI_REF @ val), 3)
F["ref_proxy"] = round(float(PI_REF @ R), 3)


def policy(beta):
    return softmax(np.log(PI_REF) + R / beta)


# ---------------------------------------- fig 81.1: Goodhart curve on real data
BETAS = np.exp(np.linspace(np.log(6.0), np.log(0.05), 90))
KLS = np.array([kl(policy(b), PI_REF) for b in BETAS])
PROXY = np.array([policy(b) @ R for b in BETAS])
TRUE = np.array([policy(b) @ val for b in BETAS])

# small label budget: 200 comparisons, averaged over 12 independent annotation samples
SMALL = 200
kl_s, true_s = [], []
for sd in range(820, 832):
    Rs = fit_bt(sample_pairs(ANN, SMALL, seed=sd)[0], K)
    pol = [softmax(np.log(PI_REF) + Rs / b) for b in BETAS]
    kl_s.append([kl(p, PI_REF) for p in pol])
    true_s.append([p @ val for p in pol])
KL_S = np.mean(kl_s, axis=0)
TRUE_S = np.array(true_s)
LO, HI = TRUE_S.min(axis=0), TRUE_S.max(axis=0)
drops = [float(row.max() - row[-1]) for row in TRUE_S]
F["n_pairs_small"] = SMALL
F["small_seeds"] = len(drops)
F["small_mean_drop"] = round(float(np.mean(drops)), 3)
F["small_max_drop"] = round(float(np.max(drops)), 3)
F["small_end_spread"] = round(float(TRUE_S[:, -1].max() - TRUE_S[:, -1].min()), 3)
F["small_mid_spread"] = round(float(TRUE_S[:, 45].max() - TRUE_S[:, 45].min()), 3)
F["small_mid_kl"] = round(float(KL_S[45]), 2)
print("drops", F["small_mean_drop"], F["small_max_drop"],
      "spread end/mid", F["small_end_spread"], F["small_mid_spread"])
assert F["small_end_spread"] > 0.2
assert F["small_end_spread"] > 2 * F["small_mid_spread"]

i_star = int(np.argmax(TRUE))
F["beta_star"] = round(float(BETAS[i_star]), 3)
F["kl_star"] = round(float(KLS[i_star]), 3)
F["true_star"] = round(float(TRUE[i_star]), 3)
F["true_end"] = round(float(TRUE[-1]), 3)
F["kl_end"] = round(float(KLS[-1]), 3)
F["proxy_star"] = round(float(PROXY[i_star]), 3)
F["proxy_end"] = round(float(PROXY[-1]), 3)
F["gain_star"] = round(float(TRUE[i_star] - PI_REF @ val), 3)
F["loss_end"] = round(float(TRUE[i_star] - TRUE[-1]), 3)
assert 0 < i_star < len(BETAS) - 5, i_star            # peak is interior
assert np.all(np.diff(PROXY) > -1e-9)                  # proxy grows monotonically
assert TRUE[-1] < TRUE[i_star]


def fig_goodhart() -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.plot(KLS, PROXY, color=RED, lw=2.4, label="proxy: обученный reward")
    ax.set_xlabel(r"$D_{\mathrm{KL}}(\pi_\beta\,\|\,\pi_{\mathrm{ref}})$, нат")
    ax.set_ylabel("средний proxy-reward", color=RED)
    ax.tick_params(axis="y", colors=RED)
    ax2 = ax.twinx()
    ax2.plot(KLS, TRUE, color=BLUE, lw=2.6,
             label=f"истина: судьи 35+ ({N_PAIRS} сравнений)")
    ax2.fill_between(KL_S, LO, HI, color=GREEN, alpha=0.16)
    ax2.plot(KL_S, TRUE_S.mean(axis=0), color=GREEN, lw=1.6, ls=(0, (5, 3)),
             label=f"то же при {SMALL} сравнениях: {len(drops)} разметок")
    ax2.set_ylabel("средняя оценка независимых зрителей", color=BLUE)
    ax2.tick_params(axis="y", colors=BLUE)
    ax2.axvline(KLS[i_star], color=GOLD, lw=1.6, ls=(0, (5, 3)))
    ax2.annotate(f"максимум истины\nKL={KLS[i_star]:.2f}, "
                 fr"$\beta={BETAS[i_star]:.2f}$",
                 xy=(KLS[i_star], TRUE[i_star]),
                 xytext=(KLS[i_star] + 0.35, TRUE[i_star] - 0.10),
                 fontsize=10, color=GOLD)
    ax2.plot([KLS[i_star]], [TRUE[i_star]], "o", color=GOLD, ms=8, zorder=6)
    ax2.axhline(PI_REF @ val, color=MUTED, lw=1.0, ls=(0, (2, 2)))
    ax2.text(0.05, PI_REF @ val + 0.012, "уровень reference", ha="left",
             fontsize=9.5, color=MUTED)
    ax.set_title("Proxy растёт всю дорогу, истина разворачивается (MovieLens 100K)")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", frameon=False, fontsize=10)
    save(fig, OUT / "goodhart.png")
    print(f"goodhart: beta*={BETAS[i_star]:.3f} KL*={KLS[i_star]:.3f} "
          f"true*={TRUE[i_star]:.3f} true_end={TRUE[-1]:.3f}")


# ---------------------------------------- fig 81.2: exponential tilt of reference
def fig_tilt() -> None:
    top = np.argsort(-policy(0.2))[:12]
    names = [TITLES[int(CAND[j])].split(" (")[0][:18] for j in top]
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    w = 0.2
    xs = np.arange(len(top))
    ax.bar(xs - 1.5 * w, PI_REF[top], w, color=MUTED, label=r"$\pi_{\mathrm{ref}}$")
    for k, (b, c) in enumerate([(2.0, GREEN), (0.6, BLUE), (0.2, RED)]):
        p = policy(b)
        ax.bar(xs + (k - 0.5) * w, p[top], w, color=c,
               label=fr"$\beta={b}$ (KL={kl(p, PI_REF):.2f})")
        F[f"share_top_beta{str(b).replace('.', '')}"] = round(float(p.max()), 3)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("вероятность ответа")
    ax.set_title(r"Малое $\beta$ выжимает распределение в один ответ")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "tilt.png")
    print("tilt drawn")


# ---------------------------------------- fig 81.3: PPO / DPO / GRPO to one target
BETA_OPT = 0.6
PI_STAR = policy(BETA_OPT)
EPS = 0.2


def run_dpo(steps=400, lr=0.5):
    z = np.log(PI_REF).copy()
    hist = []
    p_lab = 1.0 / (1.0 + np.exp(-(R[:, None] - R[None, :])))     # BT labels
    W = np.ones((K, K)) - np.eye(K)
    for _ in range(steps):
        pi = softmax(z)
        h = np.log(pi) - np.log(PI_REF)
        d = BETA_OPT * (h[:, None] - h[None, :])
        s = 1.0 / (1.0 + np.exp(-d))
        # dL/dh_i summed over pairs (i,j): -(p_lab - sigma) * beta
        gh = -BETA_OPT * np.sum(W * (p_lab - s), axis=1) * 2 / (K * (K - 1))
        gz = pi * (gh - pi @ gh)                                  # chain through softmax
        z = z - lr * 1000 * gz
        hist.append(kl(softmax(z), PI_STAR))
    return np.array(hist)


def run_ppo(steps=400, inner=10, lr=0.08, group=False, G=64, seed=812):
    """Clipped surrogate + KL penalty, exact expectations over the K responses.

    group=True imitates GRPO: the advantage is standardised inside a sampled group.
    """
    rng = np.random.default_rng(seed)
    z = np.log(PI_REF).copy()
    hist = []
    for _ in range(steps):
        pi_old = softmax(z)
        if group:
            g = rng.choice(K, size=G, p=pi_old)                   # группа ответов
            adv = (R - R[g].mean()) / max(float(R[g].std()), 0.05)
        else:
            adv = R - pi_old @ R
        for _ in range(inner):
            pi = softmax(z)
            rho = pi / pi_old
            take = rho * adv <= np.clip(rho, 1 - EPS, 1 + EPS) * adv
            u = np.where(take, adv, 0.0)                           # d/dpi суррогата
            u = u - BETA_OPT * (np.log(np.maximum(pi, 1e-12)) - np.log(PI_REF) + 1.0)
            z = z + lr * pi * (u - pi @ u)
            z = z - z.max()
        hist.append(kl(softmax(z), PI_STAR))
    return np.array(hist), softmax(z)


H_DPO = run_dpo()
H_PPO, PI_PPO = run_ppo()
H_GRPO, PI_GRPO = run_ppo(group=True, G=64)
H_G16, PI_G16 = run_ppo(group=True, G=16)
F["grpo_group"] = 64
F["grpo_group_small"] = 16
F["dpo_final_kl"] = float(f"{H_DPO[-1]:.2e}")
F["ppo_final_kl"] = float(f"{H_PPO[-1]:.2e}")
F["grpo_final_kl"] = round(float(H_GRPO[-1]), 3)
F["grpo_kl_ref"] = round(kl(PI_GRPO, PI_REF), 2)
F["star_kl_ref"] = round(kl(PI_STAR, PI_REF), 2)
F["grpo16_top_share"] = round(float(PI_G16.max()), 3)
F["grpo16_kl_ref"] = round(kl(PI_G16, PI_REF), 2)
F["star_top_share"] = round(float(PI_STAR.max()), 3)
assert H_DPO[-1] < 5e-3, H_DPO[-1]
assert H_PPO[-1] < 5e-3, H_PPO[-1]
assert H_GRPO[-1] > 0.05, H_GRPO[-1]
assert kl(PI_GRPO, PI_REF) > kl(PI_STAR, PI_REF)
assert PI_G16.max() > 0.9                        # вырожденная группа = коллапс
# какое beta даёт ту же policy, что и GRPO?
grid = np.linspace(0.02, 0.8, 157)
BETA_EFF = float(grid[int(np.argmin([kl(PI_GRPO, policy(b)) for b in grid]))])
F["grpo_beta_eff"] = round(BETA_EFF, 3)
F["grpo_kl_to_eff"] = round(kl(PI_GRPO, policy(BETA_EFF)), 3)
F["grpo_scale"] = round(BETA_EFF / BETA_OPT, 3)
F["beta_opt"] = BETA_OPT
assert BETA_EFF < BETA_OPT and F["grpo_kl_to_eff"] < 0.05


def fig_methods() -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.6),
                                  gridspec_kw={"width_ratios": [1.15, 1]})
    ax.semilogy(H_DPO, color=GREEN, lw=2.2, label="DPO (пары, offline)")
    ax.semilogy(H_PPO, color=BLUE, lw=2.2, label="PPO (clip + KL)")
    ax.semilogy(H_GRPO, color=RED, lw=2.2, label="GRPO, группа 64")
    ax.semilogy(H_G16, color=GOLD, lw=1.8, ls=(0, (4, 3)), label="GRPO, группа 16")
    ax.set_xlabel("шаг оптимизации")
    ax.set_ylabel(r"$D_{\mathrm{KL}}(\pi_t\,\|\,\pi^*)$")
    ax.set_title("Разная механика — одна цель", fontsize=13)
    ax.legend(frameon=False, fontsize=9.0)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    top = np.argsort(-PI_STAR)[:8]
    xs = np.arange(len(top)); w = 0.21
    ax2.bar(xs - 1.5 * w, PI_REF[top], w, color=MUTED, label=r"$\pi_{\mathrm{ref}}$")
    ax2.bar(xs - 0.5 * w, PI_STAR[top], w, color=BLUE, label=r"$\pi^*$ (PPO $=$ DPO)")
    ax2.bar(xs + 0.5 * w, PI_GRPO[top], w, color=RED,
            label=fr"GRPO-64 $\approx\pi$ при $\beta={BETA_EFF:.2f}$")
    ax2.bar(xs + 1.5 * w, PI_G16[top], w, color=GOLD, label="GRPO-16: коллапс")
    ax2.set_xticks(xs); ax2.set_xticklabels([str(i + 1) for i in xs], fontsize=9)
    ax2.set_xlabel("ответ (по убыванию $\\pi^*$)")
    ax2.set_ylabel("вероятность")
    ax2.set_title("Нормировка меняет эффективное $\\beta$", fontsize=13)
    ax2.legend(frameon=False, fontsize=8.5)
    ax2.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax2.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "methods.png")
    print(f"methods: dpo={H_DPO[-1]:.2e} ppo={H_PPO[-1]:.2e} grpo={H_GRPO[-1]:.3f}")


# ---------------------------------------- fig 81.4: clipped objective
def fig_clip() -> None:
    rho = np.linspace(0.4, 1.8, 400)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.2, 3.9), sharey=True)
    for ax, A, col, ttl in [(a0, 2.0, GREEN, "$\\widehat A=+2$: выгодное действие"),
                            (a1, -2.0, RED, "$\\widehat A=-2$: вредное действие")]:
        unc = rho * A
        clipped = np.clip(rho, 1 - EPS, 1 + EPS) * A
        obj = np.minimum(unc, clipped)
        ax.plot(rho, unc, color=MUTED, lw=1.2, ls=(0, (4, 3)), label="без clipping")
        ax.plot(rho, obj, color=col, lw=2.6, label="clipped objective")
        ax.axvspan(1 - EPS, 1 + EPS, color=WASH, alpha=0.9, zorder=0)
        ax.axvline(1.0, color=LINE, lw=1.0)
        ax.set_xlabel(r"$\rho=\pi_\theta/\pi_{\theta_{\rm old}}$")
        ax.set_title(ttl, fontsize=12)
        ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    a0.set_ylabel("вклад в целевую функцию")
    ppo_plateau = float(min((1 + EPS) * 2.0, (1 + EPS) * 2.0))
    F["clip_plateau_pos"] = round(ppo_plateau, 2)
    F["clip_value_neg_at_08"] = round(float(min(0.8 * -2.0, np.clip(0.8, 0.8, 1.2) * -2.0)), 2)
    fig.suptitle("Clipping снимает награду за слишком большой шаг, но не за откат",
                 y=1.02, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "clip.png")
    print("clip drawn")


# ---------------------------------------- fig 81.5: group normalisation (real users)
def fig_groups() -> None:
    by_user: dict[int, list[int]] = {}
    for u, it, r, _ in ANN:
        by_user.setdefault(int(u), []).append(int(r))
    cand = [(u, np.array(v)) for u, v in by_user.items() if len(v) >= 60]
    means = np.array([v.mean() for _, v in cand])
    stds = np.array([v.std() for _, v in cand])
    gen = int(np.argmax(means)); harsh = int(np.argmin(means))
    rng = np.random.default_rng(813)
    g1 = rng.choice(cand[gen][1], 8, replace=False).astype(float)
    g2 = rng.choice(cand[harsh][1], 8, replace=False).astype(float)
    F["user_gen_mean"] = round(float(means[gen]), 2)
    F["user_harsh_mean"] = round(float(means[harsh]), 2)
    F["user_gen_std"] = round(float(stds[gen]), 2)
    F["user_harsh_std"] = round(float(stds[harsh]), 2)
    assert means[gen] - means[harsh] > 1.0
    n1 = (g1 - g1.mean()) / (g1.std() + 1e-8)
    n2 = (g2 - g2.mean()) / (g2.std() + 1e-8)
    F["raw_gap"] = round(float(g1.mean() - g2.mean()), 2)
    F["norm_gap"] = round(float(abs(n1.mean() - n2.mean())), 3)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.2, 4.0), sharex=True)
    xs = np.arange(8)
    for ax, y1, y2, ttl in [(a0, g1, g2, "сырой reward"),
                            (a1, n1, n2, "advantage, нормированный внутри группы")]:
        ax.bar(xs - 0.2, y1, 0.4, color=GOLD, label=f"щедрый prompt (среднее {means[gen]:.2f})")
        ax.bar(xs + 0.2, y2, 0.4, color=VIOLET, label=f"строгий prompt (среднее {means[harsh]:.2f})")
        ax.axhline(0, color=LINE, lw=1.0)
        ax.set_title(ttl, fontsize=12.5)
        ax.set_xlabel("ответ в группе")
        ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    a0.set_ylim(0, 6.4)
    a0.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=2)
    fig.suptitle("Группа сравнивает ответы с их же соседями, а не с чужим prompt",
                 y=1.02, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "groups.png")
    print(f"groups: raw gap {F['raw_gap']}, normalised gap {F['norm_gap']}")


# ---------------------------------------- margins
def side_beta() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    bb = np.exp(np.linspace(np.log(4.0), np.log(0.05), 120))
    share = np.array([policy(b).max() for b in bb])
    ax.plot(bb, share, color=BLUE, lw=2.0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta$ (лог)", fontsize=9)
    ax.set_ylabel("доля лидера", fontsize=9)
    ax.axhline(PI_REF.max(), color=MUTED, lw=1.0, ls=(0, (2, 2)))
    ax.text(4.0, PI_REF.max() + 0.02, "reference", fontsize=8, color=MUTED)
    ax.set_title("чем меньше $\\beta$, тем уже policy", fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "beta.png")


def side_sigmoid() -> None:
    d = np.linspace(-6, 6, 300)
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    for b, c in [(1.0, BLUE), (0.3, GOLD)]:
        loss = -np.log(1.0 / (1.0 + np.exp(-b * d)))
        ax.plot(d, loss, color=c, lw=2.0, label=fr"$\beta={b}$")
    ax.set_xlabel("разрыв log-отношений $\\Delta$", fontsize=9)
    ax.set_ylabel(r"$-\log\sigma(\beta\Delta)$", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("DPO: логистическая цена пары", fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    F["dpo_loss_at_zero"] = round(float(-np.log(0.5)), 3)
    save(fig, SIDE / "sigmoid.png")


def side_groups() -> None:
    """Вкус разметчиков против вкуса судей: тот же фильм, две оценки."""
    yv = np.zeros(K); yc = np.zeros(K)
    for u, it, r, _ in ANN:
        j = IDX.get(int(it))
        if j is not None:
            yv[j] += r; yc[j] += 1
    yv = yv / yc
    c = float(np.corrcoef(yv, val)[0, 1])
    d = yv - val
    jmax = int(np.argmax(d))
    F["corr_group_tastes"] = round(c, 3)
    F["gap_title"] = TITLES[int(CAND[jmax])]
    F["gap_ann"] = round(float(yv[jmax]), 2)
    F["gap_judge"] = round(float(val[jmax]), 2)
    F["gap_size"] = round(float(d[jmax]), 2)
    assert 0.5 < c < 0.9 and d[jmax] > 0.3
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.scatter(yv, val, s=26, color=BLUE, alpha=0.8)
    lim = [min(yv.min(), val.min()) - 0.1, max(yv.max(), val.max()) + 0.1]
    ax.plot(lim, lim, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax.plot([yv[jmax]], [val[jmax]], "o", color=RED, ms=8)
    ax.annotate(TITLES[int(CAND[jmax])].split(" (")[0], (yv[jmax], val[jmax]),
                xytext=(-6, -14), textcoords="offset points", fontsize=8, color=RED,
                ha="right")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("средняя оценка разметчиков (до 25)", fontsize=8.5)
    ax.set_ylabel("судей (35+)", fontsize=8.5)
    ax.set_title(f"один фильм, два вкуса: $r={c:.2f}$", fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "groups.png")
    print(f"side groups: corr={c:.3f}, max gap {TITLES[int(CAND[jmax])]} {d[jmax]:.2f}")


fig_goodhart()
fig_tilt()
fig_methods()
fig_clip()
fig_groups()
side_beta()
side_sigmoid()
side_groups()

# a few numbers quoted directly in the prose
F["tilt_ratio_beta1"] = round(float(np.exp(2 / 1.0)), 2)
F["tilt_ratio_beta05"] = round(float(np.exp(2 / 0.5)), 2)
F["dpo_example_margin"] = round(float(0.5 * ((-4 - -5) - (-3.5 - -4))), 3)
F["dpo_example_loss"] = round(float(-np.log(1 / (1 + np.exp(-0.5 * ((-4 + 5) - (-3.5 + 4)))))), 3)
F["ppo_ratio_example"] = round(0.27 / 0.20, 3)

FACTS.write_text(json.dumps(F, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf8")
print(json.dumps(F, ensure_ascii=False, indent=1, sort_keys=True))
print("lesson 81 figures written")
