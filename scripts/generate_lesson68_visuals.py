"""Deterministic figures for lesson 68: breaking a substitution cipher with MCMC.

Real Russian corpus: the text of this very textbook (content/lessons/01..40 as train,
41..50 as held-out test fragments), normalised to 33 letters plus space. Every number
quoted in the lesson is computed here and asserted.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LESSONS = ROOT / "content" / "lessons"
OUT = ROOT / "public" / "figures" / "lessons" / "68"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "68"

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


# ------------------------------------------------------------------ corpus
ALPHA = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
SPACE = len(ALPHA)                # index 33
M = len(ALPHA)                    # 33 letters permuted by the key
A = M + 1                         # 34 symbols in the alphabet
IDX = {ch: i for i, ch in enumerate(ALPHA)}
IDX[" "] = SPACE
LAM = 0.1                         # add-lambda smoothing


def normalise(raw: str) -> str:
    s = unicodedata.normalize("NFC", raw).lower()
    s = re.sub(r"[^а-яё]+", " ", s)
    return s.strip()


def read_range(lo: int, hi: int) -> str:
    parts = []
    for n in range(lo, hi + 1):
        parts.append(normalise((LESSONS / f"{n:02d}.md").read_text(encoding="utf8")))
    return " ".join(parts)


TRAIN = read_range(1, 40)
TEST = read_range(41, 50)
train_ids = np.array([IDX[c] for c in TRAIN], dtype=np.int64)
test_ids = np.array([IDX[c] for c in TEST], dtype=np.int64)
print(f"corpus: train {len(TRAIN)} chars, test {len(TEST)} chars")
assert len(TRAIN) > 800_000 and len(TEST) > 120_000

# ------------------------------------------------------------------ n-gram models
uni_c = np.bincount(train_ids, minlength=A).astype(float)
bi_c = np.zeros((A, A))
np.add.at(bi_c, (train_ids[:-1], train_ids[1:]), 1.0)
tri_c = np.zeros((A, A, A))
np.add.at(tri_c, (train_ids[:-2], train_ids[1:-1], train_ids[2:]), 1.0)

LOG_UNI = np.log((uni_c + LAM) / (uni_c.sum() + LAM * A))
LOG_BI = np.log((bi_c + LAM) / (bi_c.sum(axis=1, keepdims=True) + LAM * A))
LOG_TRI = np.log((tri_c + LAM) / (tri_c.sum(axis=2, keepdims=True) + LAM * A))

letter_freq = uni_c[:M] / uni_c[:M].sum()
top_order = np.argsort(-letter_freq)
TOP_LETTER = ALPHA[top_order[0]]
TOP_FREQ = letter_freq[top_order[0]]
SECOND_LETTER = ALPHA[top_order[1]]
SECOND_FREQ = letter_freq[top_order[1]]
RARE_LETTER = ALPHA[top_order[-1]]
RARE_FREQ = letter_freq[top_order[-1]]
print(f"letters: top {TOP_LETTER}={TOP_FREQ:.4f}, second {SECOND_LETTER}={SECOND_FREQ:.4f}, "
      f"rarest {RARE_LETTER}={RARE_FREQ:.6f}")
assert TOP_LETTER == "о" and 0.09 < TOP_FREQ < 0.13

space_share = uni_c[SPACE] / uni_c.sum()
print(f"space share of all symbols: {space_share:.4f}")


def cross_entropy_bits(order: int, ids: np.ndarray) -> float:
    """Bits per symbol of held-out text under the trained model."""
    if order == 1:
        lg = LOG_UNI[ids]
    elif order == 2:
        lg = LOG_BI[ids[:-1], ids[1:]]
    else:
        lg = LOG_TRI[ids[:-2], ids[1:-1], ids[2:]]
    return float(-lg.mean() / math.log(2.0))


H0 = math.log2(A)
H1 = cross_entropy_bits(1, test_ids)
H2 = cross_entropy_bits(2, test_ids)
H3 = cross_entropy_bits(3, test_ids)
KEY_BITS = float(sum(math.log2(i) for i in range(1, M + 1)))
LOG10_KEYS = float(sum(math.log10(i) for i in range(1, M + 1)))
KEYS_MANTISSA = 10 ** (LOG10_KEYS - math.floor(LOG10_KEYS))
REDUND = H0 - H3
UNICITY = KEY_BITS / REDUND
print(f"entropy bits/char: H0={H0:.3f} H1={H1:.3f} H2={H2:.3f} H3={H3:.3f}")
print(f"key: log2(33!)={KEY_BITS:.1f} bits, 33! = {KEYS_MANTISSA:.2f}e{int(math.floor(LOG10_KEYS))}")
print(f"redundancy D={REDUND:.2f} bits/char, unicity distance U={UNICITY:.1f} chars")
assert 5.0 < H0 < 5.1 and 4.2 < H1 < 4.6 and 3.3 < H2 < 3.9 and 2.4 < H3 < 3.2
assert 122 < KEY_BITS < 123.5 and 8.5 < KEYS_MANTISSA < 8.8
assert 30 < UNICITY < 90 and 2.0 < REDUND < 2.5

# ------------------------------------------------------------------ cipher machinery


def make_key(rng: np.random.Generator) -> np.ndarray:
    """key[plain] = cipher symbol; space maps to itself."""
    k = np.arange(A)
    k[:M] = rng.permutation(M)
    return k


def encipher(ids: np.ndarray, key: np.ndarray) -> np.ndarray:
    return key[ids]


def sparse_trigrams(y: np.ndarray):
    codes = y[:-2] * A * A + y[1:-1] * A + y[2:]
    uniq, cnt = np.unique(codes, return_counts=True)
    i = uniq // (A * A); j = (uniq // A) % A; k = uniq % A
    return i, j, k, cnt.astype(float)


def sparse_bigrams(y: np.ndarray):
    codes = y[:-1] * A + y[1:]
    uniq, cnt = np.unique(codes, return_counts=True)
    return uniq // A, uniq % A, cnt.astype(float)


def sparse_unigrams(y: np.ndarray):
    uniq, cnt = np.unique(y, return_counts=True)
    return uniq, cnt.astype(float)


def make_scorer(y: np.ndarray, order: int):
    """Returns score(inv) where inv[cipher] = guessed plain symbol."""
    if order == 3:
        i, j, k, c = sparse_trigrams(y)

        def score(inv):
            return float(np.dot(LOG_TRI[inv[i], inv[j], inv[k]], c))
    elif order == 2:
        i, j, c = sparse_bigrams(y)

        def score(inv):
            return float(np.dot(LOG_BI[inv[i], inv[j]], c))
    else:
        i, c = sparse_unigrams(y)

        def score(inv):
            return float(np.dot(LOG_UNI[inv[i]], c))
    return score


T_SCALE = 0.03      # start temperature per character of ciphertext
T_DROP = 0.03       # geometric cooling factor over the whole run


def temperature(t: int, iters: int, n: int) -> float:
    return T_SCALE * n * (T_DROP ** (t / iters))


def run_chain(y, order, rng, iters, mode="anneal", track=False, x_true=None):
    """Metropolis / greedy search over inverse keys. Returns dict with best state."""
    score = make_scorer(y, order)
    inv = np.arange(A)
    inv[:M] = rng.permutation(M)
    cur = score(inv)
    best, best_inv = cur, inv.copy()
    hist = []
    acc = 0
    n = len(y)
    for t in range(iters):
        a, b = rng.choice(M, size=2, replace=False)
        inv[a], inv[b] = inv[b], inv[a]
        new = score(inv)
        d = new - cur
        if mode == "greedy":
            ok = d > 0
        else:
            T = temperature(t, iters, n) if mode == "anneal" else T_SCALE * n
            ok = d >= 0 or rng.random() < math.exp(d / T)
        if ok:
            cur = new; acc += 1
            if cur > best:
                best, best_inv = cur, inv.copy()
        else:
            inv[a], inv[b] = inv[b], inv[a]
        if track and (t % 50 == 0 or t == iters - 1):
            hist.append((t, cur, best))
    out = {"best": best, "inv": best_inv, "hist": hist, "acc_rate": acc / iters}
    if x_true is not None:
        out["acc"] = float(np.mean(best_inv[y] == x_true))
    return out


def fragment(start: int, length: int) -> np.ndarray:
    return test_ids[start:start + length]


# ------------------------------------------------------------------ experiment 1: length curve
LENGTHS = [50, 100, 200, 500, 1000, 2000]
ORDERS = [1, 2, 3]
CACHE = ROOT / "scripts" / "data" / "lesson68_facts.json"
if CACHE.exists():
    facts = json.loads(CACHE.read_text())
else:
    facts = {}

if "length_curve" not in facts:
    curve = {}
    for order in ORDERS:
        for L in LENGTHS:
            accs = []
            for j in range(12):
                rng = np.random.default_rng(6800 + 100 * L + j)
                x = fragment(3000 + 4001 * j, L)
                key = make_key(rng)
                y = encipher(x, key)
                best = None
                for r in range(6):
                    cr = np.random.default_rng(68000 + 1000 * order + 37 * L + 7 * j + r)
                    res = run_chain(y, order, cr, 20000, mode="anneal", x_true=x)
                    if best is None or res["best"] > best["best"]:
                        best = res
                accs.append(best["acc"])
            curve[f"{order}|{L}"] = accs
            print(f"  order {order}, L={L}: median acc {np.median(accs):.3f}")
    facts["length_curve"] = curve
    CACHE.write_text(json.dumps(facts, ensure_ascii=False))

curve = facts["length_curve"]


def med(order, L):
    return float(np.median(curve[f"{order}|{L}"]))


ACC3_1000 = med(3, 1000); ACC3_100 = med(3, 100); ACC3_200 = med(3, 200)
ACC2_1000 = med(2, 1000); ACC1_1000 = med(1, 1000); ACC3_2000 = med(3, 2000)
ACC3_50 = med(3, 50)
print(f"acc trigram: 50->{ACC3_50:.2f} 100->{ACC3_100:.2f} 200->{ACC3_200:.2f} "
      f"1000->{ACC3_1000:.2f} 2000->{ACC3_2000:.2f}; bigram 1000->{ACC2_1000:.2f}; "
      f"unigram 1000->{ACC1_1000:.2f}")
assert ACC3_1000 > 0.9 and ACC3_50 < 0.5 and ACC1_1000 < 0.6
assert ACC3_200 > 0.9 and ACC3_1000 >= ACC1_1000

# ------------------------------------------------------------------ experiment 2: greedy vs MCMC
DUEL_LEN = 400
if "duel" not in facts:
    duel = {"greedy": [], "mcmc": [], "greedy_acc": [], "mcmc_acc": [], "true": [],
            "curves": {}}
    for j in range(20):
        rng = np.random.default_rng(6820 + j)
        x = fragment(50_000 + 3301 * j, DUEL_LEN)
        key = make_key(rng)
        y = encipher(x, key)
        inv_true = np.zeros(A, dtype=int); inv_true[key] = np.arange(A)
        duel["true"].append(make_scorer(y, 3)(inv_true))
        gr = run_chain(y, 3, np.random.default_rng(68200 + j), 20000, mode="greedy", x_true=x)
        mc = run_chain(y, 3, np.random.default_rng(68300 + j), 20000, mode="anneal", x_true=x)
        duel["greedy"].append(gr["best"]); duel["mcmc"].append(mc["best"])
        duel["greedy_acc"].append(gr["acc"]); duel["mcmc_acc"].append(mc["acc"])
    # six starts on ONE fixed ciphertext, for the trajectory panel
    rng = np.random.default_rng(6820)
    x0 = fragment(50_000, DUEL_LEN)
    key0 = make_key(rng)
    y0 = encipher(x0, key0)
    inv0 = np.zeros(A, dtype=int); inv0[key0] = np.arange(A)
    duel["true_score"] = make_scorer(y0, 3)(inv0)
    for r in range(6):
        gr = run_chain(y0, 3, np.random.default_rng(68400 + r), 20000, mode="greedy",
                       track=True, x_true=x0)
        mc = run_chain(y0, 3, np.random.default_rng(68500 + r), 20000, mode="anneal",
                       track=True, x_true=x0)
        duel["curves"][f"g{r}"] = gr["hist"]; duel["curves"][f"m{r}"] = mc["hist"]
        if r == 0:
            duel["acc_rate_mcmc"] = mc["acc_rate"]; duel["acc_rate_greedy"] = gr["acc_rate"]
    facts["duel"] = duel
    CACHE.write_text(json.dumps(facts, ensure_ascii=False))

duel = facts["duel"]
G_ACC = float(np.median(duel["greedy_acc"])); MC_ACC = float(np.median(duel["mcmc_acc"]))
G_WINS = int(np.sum(np.array(duel["mcmc"]) > np.array(duel["greedy"])))
TRUE_SCORE = duel["true_score"]
ACC_RATE = duel["acc_rate_mcmc"]
OVERSHOOT = int(np.sum(np.array(duel["mcmc"]) > np.array(duel["true"]) + 1e-6))
GAP_TRUE = float(np.median(np.array(duel["true"]) - np.array(duel["greedy"])))
ACC_RATE_G = duel["acc_rate_greedy"]
print(f"duel: greedy median acc {G_ACC:.2f}, mcmc median acc {MC_ACC:.2f}, "
      f"mcmc better score in {G_WINS}/20; acceptance rate {ACC_RATE:.3f} vs greedy "
      f"{ACC_RATE_G:.4f}; overshoot true score in {OVERSHOOT}/20; median greedy gap "
      f"{GAP_TRUE:.0f} nats")
assert MC_ACC > G_ACC and G_WINS >= 15 and GAP_TRUE > 0

# ------------------------------------------------------------------ experiment 3: unfolding text
SNAP_ITERS = [0, 2000, 3800, 4400, 20000]
if "unfold" not in facts:
    rng = np.random.default_rng(6899)
    x = fragment(120_000, 600)
    key = make_key(rng)
    y = encipher(x, key)
    score = make_scorer(y, 3)
    cr = np.random.default_rng(68990)
    inv = np.arange(A); inv[:M] = cr.permutation(M)
    cur = score(inv); best, best_inv = cur, inv.copy()
    snaps = []; series = []
    show = slice(0, 62)
    for t in range(SNAP_ITERS[-1] + 1):
        if t in SNAP_ITERS:
            dec = "".join((ALPHA + " ")[c] for c in best_inv[y][show])
            snaps.append((t, dec, float(np.mean(best_inv[y] == x)), best))
        a, b = cr.choice(M, size=2, replace=False)
        inv[a], inv[b] = inv[b], inv[a]
        new = score(inv); d = new - cur
        T = temperature(t, SNAP_ITERS[-1], len(y))
        if d >= 0 or cr.random() < math.exp(d / T):
            cur = new
            if cur > best:
                best, best_inv = cur, inv.copy()
        else:
            inv[a], inv[b] = inv[b], inv[a]
        if t % 100 == 0:
            series.append((t, float(np.mean(best_inv[y] == x)), best))
    facts["unfold"] = {"snaps": snaps, "series": series,
                       "truth": "".join((ALPHA + " ")[c] for c in x[show])}
    CACHE.write_text(json.dumps(facts, ensure_ascii=False))

unfold = facts["unfold"]
UNFOLD_FINAL = unfold["snaps"][-1][2]
UNFOLD_AT100 = unfold["snaps"][1][2]
print(f"unfold: acc after {unfold['snaps'][1][0]} iters {UNFOLD_AT100:.2f}, final {UNFOLD_FINAL:.2f}")
assert UNFOLD_FINAL > 0.9

# ------------------------------------------------------------------ experiment 4: per-letter stability
STAB_LEN = 200
if "stability" not in facts:
    rng = np.random.default_rng(6877)
    x = fragment(100_000, STAB_LEN)
    key = make_key(rng)
    y = encipher(x, key)
    inv_true = np.zeros(A, dtype=int); inv_true[key] = np.arange(A)
    score = make_scorer(y, 3)
    base = score(inv_true)
    # how many distinct letters does a fragment of each length even contain?
    coverage = {}
    for L in [50, 100, 200, 500, 1000, 2000]:
        vals = [int(np.unique(fragment(7000 + 811 * t, L)[fragment(7000 + 811 * t, L) < M]).size)
                for t in range(60)]
        coverage[str(L)] = float(np.median(vals))
    # agreement of 20 independent searches (each = best of three restarts)
    hits = np.zeros(M)
    R = 20
    run_scores, run_accs, keys_seen = [], [], set()
    for r in range(R):
        best = None
        for k in range(3):
            res = run_chain(y, 3, np.random.default_rng(68770 + 10 * r + k), 20000,
                            mode="anneal", x_true=x)
            if best is None or res["best"] > best["best"]:
                best = res
        hits += (best["inv"][key[:M]] == np.arange(M)).astype(float)
        run_scores.append(best["best"]); run_accs.append(best["acc"])
        keys_seen.add(tuple(int(v) for v in best["inv"][:M]))
    counts = np.bincount(y, minlength=A)[:M].astype(float)
    # counts indexed by cipher symbol -> re-index to plain letters
    plain_counts = np.bincount(x, minlength=A)[:M].astype(float)
    facts["stability"] = {"hits": (hits / R).tolist(), "counts": plain_counts.tolist(),
                          "coverage": coverage, "base": base,
                          "run_scores": run_scores, "run_accs": run_accs,
                          "distinct_keys": len(keys_seen)}
    CACHE.write_text(json.dumps(facts, ensure_ascii=False))

stab_hits = np.array(facts["stability"]["hits"])
stab_counts = np.array(facts["stability"]["counts"])
coverage = {int(k): v for k, v in facts["stability"]["coverage"].items()}
present = stab_counts > 0
FREQ_MASK = stab_counts >= 6
RARE_MASK = present & (stab_counts <= 2)
STAB_FREQ = float(stab_hits[FREQ_MASK].mean())
STAB_RARE = float(stab_hits[RARE_MASK].mean()) if RARE_MASK.any() else 0.0
COV_50, COV_200, COV_1000 = coverage[50], coverage[200], coverage[1000]
present_letters = int((stab_counts > 0).sum())
N_SURE = int(((stab_hits >= 0.9) & present).sum())
N_DISPUTED = int(((stab_hits <= 0.5) & present).sum())
RUN_SCORES = np.array(facts["stability"]["run_scores"])
RUN_ACCS = np.array(facts["stability"]["run_accs"])
DISTINCT_KEYS = facts["stability"]["distinct_keys"]
BASE_SCORE = facts["stability"]["base"]
BEST_GAP = float(np.max(RUN_SCORES) - BASE_SCORE)
MED_ACC_200 = float(np.median(RUN_ACCS))
N_FREQ = int(FREQ_MASK.sum()); N_RARE = int(RARE_MASK.sum())
print(f"stability on {STAB_LEN} chars: {N_FREQ} frequent letters agree {STAB_FREQ:.2f}, "
      f"{N_RARE} rare letters agree {STAB_RARE:.2f}")
print(f"  {present_letters} letters occur; {N_SURE} agreed in >=90% of searches, "
      f"{N_DISPUTED} in <=50%; {DISTINCT_KEYS} distinct keys among 20 searches; "
      f"best score - true score = {BEST_GAP:.1f} nats; median positional accuracy "
      f"{MED_ACC_200:.2f}")
print(f"coverage (distinct letters, median): 50->{COV_50:.0f}, 200->{COV_200:.0f}, "
      f"1000->{COV_1000:.0f} of {M}")
assert STAB_FREQ > STAB_RARE + 0.2 and COV_50 < COV_200 < COV_1000 <= M

# ------------------------------------------------------------------ figure 1: pattern survives
def fig_pattern() -> None:
    rng = np.random.default_rng(1968)
    phrase = "модель языка помнит привычки букв"
    x = np.array([IDX[c] for c in phrase])
    key = make_key(rng)
    y = key[x]
    ctext = "".join((ALPHA + " ")[c] for c in y)
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 5.6),
                             gridspec_kw={"height_ratios": [1.0, 1.35]})
    ax = axes[0]; ax.axis("off")
    n = len(phrase)
    for i, (p, c) in enumerate(zip(phrase, ctext)):
        ax.text(i, 1.0, p, ha="center", va="center", fontsize=12.5, color=INK)
        ax.text(i, 0.0, c, ha="center", va="center", fontsize=12.5, color=BLUE)
    marks = {"о": RED, "м": GREEN, "и": GOLD}
    for i, p in enumerate(phrase):
        if p in marks:
            ax.plot([i, i], [0.85, 0.15], color=marks[p], lw=1.0, alpha=0.75)
            ax.plot(i, 1.0, "o", color=marks[p], ms=13, alpha=0.18)
            ax.plot(i, 0.0, "o", color=marks[p], ms=13, alpha=0.18)
    ax.text(-1.6, 1.0, "текст", ha="right", va="center", fontsize=10, color=MUTED)
    ax.text(-1.6, 0.0, "шифр", ha="right", va="center", fontsize=10, color=MUTED)
    ax.set_xlim(-6, n + 1); ax.set_ylim(-0.6, 1.6)
    ax.set_title("Ключ меняет имена букв, но не рисунок повторов", fontsize=13.5)

    ax = axes[1]
    order = np.argsort(-letter_freq)[:12]
    xs = np.arange(len(order))
    ax.bar(xs - 0.2, letter_freq[order] * 100, width=0.4, color=INK, label="открытый текст")
    perm_freq = np.zeros(M)
    perm_freq[key[:M]] = letter_freq
    cipher_order = np.argsort(-perm_freq)[:12]
    ax.bar(xs + 0.2, perm_freq[cipher_order] * 100, width=0.4, color=BLUE,
           label="шифротекст (тот же ключ)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{ALPHA[a]} / {ALPHA[b]}" for a, b in zip(order, cipher_order)],
                       fontsize=10)
    ax.set_ylabel("доля среди букв, %")
    ax.set_title(f"Гистограмма частот не изменилась — сменились подписи "
                 f"(«{TOP_LETTER}» {TOP_FREQ * 100:.1f} %)", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "cipher_pattern.png")
    print("fig 1 drawn")


# ------------------------------------------------------------------ figure 2: greedy vs mcmc
def fig_duel() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4),
                             gridspec_kw={"width_ratios": [1.5, 1.0]})
    ax = axes[0]
    for j in range(6):
        g = np.array(duel["curves"][f"g{j}"], dtype=float)
        m = np.array(duel["curves"][f"m{j}"], dtype=float)
        ax.plot(g[:, 0], g[:, 2], color=RED, lw=1.3, alpha=0.75,
                label="жадный подъём" if j == 0 else None)
        ax.plot(m[:, 0], m[:, 2], color=BLUE, lw=1.3, alpha=0.75,
                label="цепь Метрополиса" if j == 0 else None)
    ax.axhline(TRUE_SCORE, color=GREEN, lw=1.6, ls=(0, (5, 3)),
               label="балл истинного ключа")
    ax.set_xlabel("итерация"); ax.set_ylabel("лучший балл $L$")
    ax.set_title(f"Шесть стартов на одном шифротексте ({DUEL_LEN} знаков)", fontsize=12.5)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.45); ax.set_axisbelow(True)

    ax = axes[1]
    ga = np.array(duel["greedy_acc"]); ma = np.array(duel["mcmc_acc"])
    ax.scatter(ga, ma, s=44, color=VIOLET, zorder=5, alpha=0.85)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.0, ls=(0, (3, 3)))
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("точность жадного подъёма"); ax.set_ylabel("точность цепи")
    ax.set_title(f"20 шифротекстов: медианы {G_ACC:.2f} против {MC_ACC:.2f}", fontsize=12)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.45); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "greedy_vs_mcmc.png")
    print("fig 2 drawn")


# ------------------------------------------------------------------ figure 3: length curve
def fig_length() -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for order, col, name in [(1, GOLD, "униграммы"), (2, GREEN, "биграммы"),
                             (3, BLUE, "триграммы")]:
        meds, lo, hi = [], [], []
        for L in LENGTHS:
            a = np.array(curve[f"{order}|{L}"])
            meds.append(np.median(a)); lo.append(np.quantile(a, 0.25)); hi.append(np.quantile(a, 0.75))
        ax.plot(LENGTHS, meds, color=col, lw=2.2, marker="o", ms=5, label=name)
        ax.fill_between(LENGTHS, lo, hi, color=col, alpha=0.15, lw=0)
    ax.axvline(UNICITY, color=RED, lw=1.4, ls=(0, (4, 3)))
    ax.text(UNICITY * 1.06, 0.93, f"расстояние единственности\nШеннона ≈ {UNICITY:.0f} знаков",
            fontsize=9.5, color=RED, va="top")
    ax.set_xscale("log")
    ax.set_xticks(LENGTHS); ax.set_xticklabels([str(v) for v in LENGTHS])
    ax.set_xlabel("длина шифротекста, знаков")
    ax.set_ylabel("доля верно расшифрованных позиций")
    ax.set_title("Информация приходит с длиной текста (медиана и межквартильный размах, 12 опытов)",
                 fontsize=12.5)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.45); ax.set_axisbelow(True)
    save(fig, OUT / "length_curve.png")
    print("fig 3 drawn")


# ------------------------------------------------------------------ figure 4: text unfolding
def fig_unfold() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 5.8),
                             gridspec_kw={"height_ratios": [1.25, 1.0]})
    ax = axes[0]; ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    rows = unfold["snaps"]
    for i, (t, dec, acc, sc) in enumerate(rows):
        yy = 0.92 - i * 0.19
        ax.text(0.0, yy, f"шаг {t}", fontsize=10, color=MUTED, va="center")
        ax.text(0.14, yy, dec, fontsize=11.5, family="monospace",
                color=INK if acc > 0.8 else MUTED, va="center")
        ax.text(0.99, yy, f"{acc * 100:.0f} %", fontsize=10, color=BLUE,
                ha="right", va="center")
    ax.text(0.0, 0.92 - len(rows) * 0.19, "истина", fontsize=10, color=GREEN, va="center")
    ax.text(0.14, 0.92 - len(rows) * 0.19, unfold["truth"], fontsize=11.5,
            family="monospace", color=GREEN, va="center")
    ax.set_title("Один запуск: текст проявляется из шума", fontsize=13.5)

    ax = axes[1]
    s = np.array(unfold["series"], dtype=float)
    ax.plot(s[:, 0], s[:, 1], color=BLUE, lw=2.0)
    for t, dec, acc, sc in rows:
        ax.plot(t, acc, "o", color=RED, ms=6, zorder=5)
    ax.set_xlabel("итерация"); ax.set_ylabel("доля верных позиций")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Прогресс идёт скачками: обмен пары частых букв разом чинит много позиций",
                 fontsize=11.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.45); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "unfolding.png")
    print("fig 4 drawn")


# ------------------------------------------------------------------ figure 5: per-letter stability
def fig_stability() -> None:
    order = np.argsort(-stab_counts)
    keep = [i for i in order if stab_counts[i] > 0]
    xs = np.arange(len(keep))
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 6.2), sharex=True)

    ax = axes[0]
    ax.bar(xs, [stab_counts[i] for i in keep], color=VIOLET, width=0.72)
    ax.set_ylabel("вхождений в фрагменте")
    ax.set_title(f"Фрагмент в {STAB_LEN} знаков содержит лишь {int(sum(stab_counts > 0))} "
                 f"букв из {M}: остальные строки ключа текст не проверяет", fontsize=12.5)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.45); ax.set_axisbelow(True)

    ax = axes[1]
    cols = [BLUE if stab_hits[i] >= 0.75 else (GOLD if stab_hits[i] >= 0.4 else RED)
            for i in keep]
    ax.bar(xs, [max(stab_hits[i], 0.02) for i in keep], color=cols, width=0.72)
    ax.set_xticks(xs)
    ax.set_xticklabels([ALPHA[i] for i in keep], fontsize=10)
    for tick, i in zip(ax.get_xticklabels(), keep):
        if stab_hits[i] <= 0.5:
            tick.set_color(RED); tick.set_fontweight("bold")
    for j, i in enumerate(keep):
        if stab_hits[i] <= 0.5:
            ax.annotate("спорная", (j, 0.06), rotation=90, fontsize=8.5, color=RED,
                        ha="center", va="bottom")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("доля из 20 независимых\nпоисков, где буква угадана")
    ax.set_xlabel("буквы открытого текста, упорядоченные по числу вхождений")
    ax.set_title(f"{N_SURE} буквы совпали почти во всех поисках, {N_DISPUTED} кочуют "
                 f"между почти равными по баллу расшифровками", fontsize=12)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.45); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "letter_confidence.png")
    print("fig 5 drawn")


# ------------------------------------------------------------------ sidenote images
def side_freq() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    order = np.argsort(-letter_freq)[:10]
    ax.bar(range(10), letter_freq[order] * 100, color=BLUE)
    ax.set_xticks(range(10)); ax.set_xticklabels([ALPHA[i] for i in order], fontsize=9)
    ax.set_ylabel("%", fontsize=9)
    ax.set_title("десять самых частых букв корпуса", fontsize=9)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "freq.png")


def side_accept() -> None:
    d = np.linspace(-8, 2, 200)
    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    for beta, c, lab in [(0.25, RED, "0,25"), (1.0, GOLD, "1"), (4.0, BLUE, "4")]:
        ax.plot(d, np.minimum(1, np.exp(beta * d)), color=c, lw=1.8, label=rf"$\beta={lab}$")
    ax.set_xlabel(r"$\Delta L$", fontsize=9); ax.set_ylabel("вероятность принять", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("холодная цепь почти не идёт вниз", fontsize=9)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "accept.png")


def side_unicity() -> None:
    n = np.linspace(0, 120, 200)
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    ax.plot(n, REDUND * n, color=BLUE, lw=2.0, label="накопленная избыточность")
    ax.axhline(KEY_BITS, color=RED, lw=1.6, ls=(0, (4, 3)), label="энтропия ключа")
    ax.plot([UNICITY], [KEY_BITS], "o", color=INK, ms=6)
    ax.annotate(f"{UNICITY:.0f}", (UNICITY, KEY_BITS), textcoords="offset points",
                xytext=(6, -14), fontsize=9, color=INK)
    ax.set_xlabel("длина шифротекста", fontsize=9); ax.set_ylabel("бит", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("расстояние единственности", fontsize=9)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "unicity.png")


def side_entropy() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    vals = [H0, H1, H2, H3]
    names = ["равномерно", "униграммы", "биграммы", "триграммы"]
    ax.bar(range(4), vals, color=[LINE, GOLD, GREEN, BLUE])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.08, f"{v:.2f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=8, rotation=18)
    ax.set_ylabel("бит на знак", fontsize=9); ax.set_ylim(0, 6)
    ax.set_title("чем длиннее контекст, тем меньше сюрприз", fontsize=9)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "entropy.png")


fig_pattern()
fig_duel()
fig_length()
fig_unfold()
fig_stability()
side_freq()
side_accept()
side_unicity()
side_entropy()

print("---- numbers quoted in lesson 68 ----")
print(json.dumps({
    "train_chars": len(TRAIN), "test_chars": len(TEST),
    "top_letter": TOP_LETTER, "top_freq_pct": round(TOP_FREQ * 100, 1),
    "second_letter": SECOND_LETTER, "second_freq_pct": round(SECOND_FREQ * 100, 1),
    "space_share_pct": round(space_share * 100, 1),
    "H0": round(H0, 2), "H1": round(H1, 2), "H2": round(H2, 2), "H3": round(H3, 2),
    "key_bits": round(KEY_BITS, 1), "keys": f"{KEYS_MANTISSA:.2f}e{int(math.floor(LOG10_KEYS))}",
    "redundancy": round(REDUND, 2), "unicity": round(UNICITY, 1),
    "acc3": {L: round(med(3, L), 2) for L in LENGTHS},
    "acc2_1000": round(ACC2_1000, 2), "acc1_1000": round(ACC1_1000, 2),
    "greedy_acc": round(G_ACC, 2), "mcmc_acc": round(MC_ACC, 2), "mcmc_wins": G_WINS,
    "overshoot": OVERSHOOT, "greedy_gap_nats": round(GAP_TRUE, 0), "acc_rate_greedy": round(ACC_RATE_G, 4),
    "acc_rate": round(ACC_RATE, 3),
    "unfold_100": round(UNFOLD_AT100, 2), "unfold_final": round(UNFOLD_FINAL, 2),
    "stab_freq": round(STAB_FREQ, 2), "stab_rare": round(STAB_RARE, 2),
    "coverage": coverage, "stab_len": STAB_LEN, "letters_in_fragment": present_letters,
    "n_sure": N_SURE, "n_disputed": N_DISPUTED, "distinct_keys": DISTINCT_KEYS,
    "best_minus_true_nats": round(BEST_GAP, 1), "med_acc_200": round(MED_ACC_200, 2),
    "duel_len": DUEL_LEN,
    "n_freq": N_FREQ, "n_rare": N_RARE,
}, ensure_ascii=False, indent=1))
# ------------------------------------------------------------------ every number quoted in prose
assert abs(TOP_FREQ * 100 - 10.4) < 0.05
assert abs(SECOND_FREQ * 100 - 8.7) < 0.05 and SECOND_LETTER == "е"
assert abs(space_share * 100 - 14.6) < 0.05
assert abs(H0 - 5.09) < 0.005 and abs(H1 - 4.42) < 0.01
assert abs(H2 - 3.59) < 0.005 and abs(H3 - 2.88) < 0.01
assert abs(KEY_BITS - 122.7) < 0.05 and abs(REDUND - 2.20) < 0.01
assert abs(UNICITY - 56) < 1.0
assert len(TRAIN) == 921_345 and len(TEST) == 128_344
assert abs(med(3, 50) - 0.22) < 0.005 and abs(med(3, 100) - 0.71) < 0.01
assert med(3, 200) == 1.0 and med(3, 500) == 1.0 and med(3, 2000) == 1.0
assert abs(med(2, 50) - 0.34) < 0.005 and abs(med(2, 100) - 0.68) < 0.01
assert abs(med(2, 200) - 0.92) < 0.005 and med(2, 500) == 1.0
assert abs(med(1, 1000) - 0.30) < 0.005 and abs(med(1, 2000) - 0.39) < 0.005
assert abs(G_ACC - 0.15) < 0.005 and MC_ACC == 1.0 and G_WINS == 18
assert abs(GAP_TRUE - 880) < 5 and OVERSHOOT == 2
assert abs(ACC_RATE - 0.1195) < 5e-05 and abs(ACC_RATE_G - 0.00265) < 5e-06
assert abs(UNFOLD_AT100 - 0.18) < 0.005 and UNFOLD_FINAL == 1.0
assert present_letters == 25 and N_SURE == 23 and N_DISPUTED == 2
assert DISTINCT_KEYS == 20 and 0 < BEST_GAP < 1.0 and abs(MED_ACC_200 - 0.99) < 0.005
assert abs(STAB_FREQ - 0.91) < 0.005 and abs(STAB_RARE - 0.60) < 0.01
assert COV_50 == 19 and COV_200 == 27 and COV_1000 == 32
SNAP_ACC = [r[2] for r in unfold["snaps"]]
assert abs(SNAP_ACC[0] - 0.173) < 0.0005 and abs(SNAP_ACC[1] - 0.18) < 0.005
assert abs(SNAP_ACC[2] - 0.58) < 0.005 and abs(SNAP_ACC[3] - 0.745) < 0.005
assert SNAP_ACC[4] == 1.0
assert abs(letter_freq[IDX["а"]] * 100 - 8.0) < 0.05
assert abs(letter_freq[IDX["и"]] * 100 - 7.6) < 0.05
assert M * (M - 1) // 2 == 528
assert abs(KEYS_MANTISSA - 8.68) < 0.005          # "33! ≈ 8,68·10^36"
assert int(math.floor(LOG10_KEYS)) == 36
assert abs(BEST_GAP - 0.6) < 0.05                 # "превышение составило 0,6 нат"
# figure 68.2 caption: how many of six chains reach the true-key level, and where greedy stalls
MCMC_REACH = sum(1 for r in range(6)
                 if float(np.array(duel["curves"][f"m{r}"], dtype=float)[-1, 1]) > TRUE_SCORE - 5)
GREEDY_ENDS = np.array([float(np.array(duel["curves"][f"g{r}"], dtype=float)[-1, 1])
                        for r in range(6)])
print(f"trajectory panel: {MCMC_REACH}/6 chains reach the true key; greedy plateau "
      f"{GREEDY_ENDS.max():.0f}..{GREEDY_ENDS.min():.0f}, true score {TRUE_SCORE:.0f}")
assert MCMC_REACH == 3                            # "три из шести доходят до уровня истины"
assert GREEDY_ENDS.max() < TRUE_SCORE - 500 and -1780 < GREEDY_ENDS.min()
assert abs(TRUE_SCORE + 788) < 3                  # "уровень истинного ключа около минус 790"
assert N_FREQ == 14 and N_RARE == 6               # "с шестью и более вхождениями" / "один-два"
print("all quoted numbers asserted")
print("lesson 68 figures written")
