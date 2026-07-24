"""Deterministic figures for lesson 78: the pretraining corpus as a hidden program.

Everything is measured on the REAL SMS Spam Collection (scripts/data/sms-spam-collection.tsv,
5574 messages, two natural "domains": ham and spam). A word-level bigram language model with
add-k smoothing plays the role of the pretrained model: it is cheap, deterministic and shows
all four effects of the lesson honestly —

  * self-supervision arithmetic: one document yields n-1 next-token targets;
  * exact and near duplicates: how many tokens are copies, and what dedup removes;
  * training mixture: per-domain validation loss moves in opposite directions;
  * repetition -> memorization: seen loss falls while held-out loss rises;
  * benchmark contamination: exact and near-duplicate leaks inflate the score
    while the clean part of the benchmark does not move.

Every number quoted in the lesson text is computed here and asserted.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SMS = ROOT / "scripts" / "data" / "sms-spam-collection.tsv"
OUT = ROOT / "public" / "figures" / "lessons" / "78"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "78"

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

FACTS: dict[str, float | int | str] = {}


def save(fig, path, *, dpi=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------- corpus
TOKEN_RE = re.compile(r"[a-z0-9£$']+|[.,!?;:]")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_corpus() -> list[tuple[str, list[str]]]:
    out = []
    with open(SMS, encoding="utf8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            label, _, text = line.partition("\t")
            out.append((label.strip(), tokenize(text)))
    return out


CORPUS = load_corpus()


# ---------------------------------------------------------------- fig 78.1
def fig_next_token() -> None:
    n_docs = len(CORPUS)
    n_ham = sum(1 for l, _ in CORPUS if l == "ham")
    n_spam = n_docs - n_ham
    n_tok = sum(len(t) for _, t in CORPUS)
    ham_tok = sum(len(t) for l, t in CORPUS if l == "ham")
    spam_tok = n_tok - ham_tok
    pairs = n_tok - n_docs
    assert (n_docs, n_ham, n_spam) == (5574, 4827, 747)
    assert (n_tok, ham_tok, spam_tok) == (106143, 84365, 21778)
    assert pairs == 100569
    FACTS.update(n_docs=n_docs, n_ham=n_ham, n_spam=n_spam, n_tok=n_tok,
                 ham_tok=ham_tok, spam_tok=spam_tok, pairs=pairs,
                 spam_tok_share=round(100 * spam_tok / n_tok, 1),
                 spam_doc_share=round(100 * n_spam / n_docs, 1),
                 tok_per_doc=round(n_tok / n_docs, 1))
    assert FACTS["spam_tok_share"] == 20.5 and FACTS["spam_doc_share"] == 13.4
    assert FACTS["tok_per_doc"] == 19.0

    demo = ["ok", "lar", "joking", "wif", "u", "oni"]
    m = len(demo)

    fig = plt.figure(figsize=(11.0, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.24)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(-0.6, m + 0.2); ax.set_ylim(-0.2, 3.2); ax.axis("off")
    for i, w in enumerate(demo):
        ax.add_patch(plt.Rectangle((i, 2.1), 0.92, 0.7, fc=WASH, ec=LINE, lw=1.0))
        ax.text(i + 0.46, 2.45, w, ha="center", va="center", fontsize=11.5, color=INK)
    for i in range(m - 1):
        ax.add_patch(plt.Rectangle((i, 0.9), 0.92, 0.7, fc="#eef1f6", ec=BLUE, lw=1.0))
        ax.text(i + 0.46, 1.25, demo[i + 1], ha="center", va="center", fontsize=11.5, color=BLUE)
        ax.annotate("", xy=(i + 0.46, 1.68), xytext=(i + 0.46, 2.05),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))
    ax.add_patch(plt.Rectangle((m - 1, 0.9), 0.92, 0.7, fc=PAPER, ec=LINE, lw=1.0, ls=(0, (3, 3))))
    ax.text(m - 0.54, 1.25, "—", ha="center", va="center", fontsize=11.5, color=FAINT)
    ax.text(-0.5, 2.45, "вход", ha="right", va="center", fontsize=11, color=MUTED)
    ax.text(-0.5, 1.25, "target", ha="right", va="center", fontsize=11, color=BLUE)
    ax.text(m / 2, 0.35, f"один документ из {m} токенов даёт {m - 1} обучающих пар",
            ha="center", fontsize=11.5, color=INK)
    ax.set_title("Сдвиг на одну позицию делает разметку", fontsize=13.5)

    ax2 = fig.add_subplot(gs[0, 1])
    mask = np.tril(np.ones((m, m)))
    ax2.imshow(mask, cmap=mpl.colors.ListedColormap([PAPER, "#dbe3ee"]), vmin=0, vmax=1)
    for i in range(m):
        for j in range(m):
            ax2.text(j, i, "1" if j <= i else "0", ha="center", va="center",
                     fontsize=10, color=BLUE if j <= i else FAINT)
    ax2.set_xticks(range(m)); ax2.set_yticks(range(m))
    ax2.set_xticklabels(demo, rotation=45, fontsize=9, ha="right")
    ax2.set_yticklabels(demo, fontsize=9)
    ax2.set_xlabel("на что смотрим"); ax2.set_ylabel("из какой позиции")
    ax2.set_title("Causal mask: будущее закрыто", fontsize=13.5)
    for s in ax2.spines.values():
        s.set_color(LINE)
    fig.suptitle(
        f"Реальный корпус SMS: {n_docs} документов, {n_tok} токенов, {pairs} обучающих пар",
        y=1.04, fontsize=13.5)
    save(fig, OUT / "next-token.png")
    print("fig1", FACTS)


# ---------------------------------------------------------------- duplicates
def norm_key(toks: list[str]) -> str:
    return " ".join(toks)


def shingles(toks: list[str], k: int = 3) -> set:
    if len(toks) < k:
        return {tuple(toks)}
    return {tuple(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def duplicate_stats():
    keys = [norm_key(t) for _, t in CORPUS]
    counts = Counter(keys)
    n_unique = len(counts)
    extra = sum(v - 1 for v in counts.values() if v > 1)
    redundant_tok = sum(len(k.split()) * (v - 1) for k, v in counts.items() if v > 1)
    top = counts.most_common(6)
    return counts, n_unique, extra, redundant_tok, top


def near_dup_pairs():
    S = [shingles(t) for _, t in CORPUS]
    inv = defaultdict(list)
    for i, s in enumerate(S):
        for g in s:
            inv[g].append(i)
    cand = set()
    for g, lst in inv.items():
        if len(lst) <= 40:
            for a in range(len(lst)):
                for b in range(a + 1, len(lst)):
                    cand.add((lst[a], lst[b]))
    js = np.array([len(S[a] & S[b]) / len(S[a] | S[b]) for a, b in cand])
    return js


def fig_duplicates() -> None:
    counts, n_unique, extra, redundant_tok, top = duplicate_stats()
    js = near_dup_pairs()
    n_tok = FACTS["n_tok"]
    assert (n_unique, extra) == (5154, 420)
    top_text, top_count = top[0]
    assert top_text == "sorry , i'll call later" and top_count == 30
    thresholds = [0.5, 0.7, 0.8, 0.9, 1.0]
    pair_counts = [int((js >= t).sum()) for t in thresholds]
    assert pair_counts == [1485, 1137, 1078, 1011, 959], pair_counts
    FACTS.update(n_unique=n_unique, extra_copies=extra, redundant_tok=redundant_tok,
                 redundant_share=round(100 * redundant_tok / n_tok, 1),
                 dup_doc_share=round(100 * extra / len(CORPUS), 1),
                 top_count=top_count, pairs_j50=pair_counts[0], pairs_j80=pair_counts[2],
                 pairs_j100=pair_counts[4], weight_top=top_count)
    assert FACTS["redundant_share"] == 9.0, FACTS["redundant_share"]
    assert FACTS["dup_doc_share"] == 7.5

    labels = []
    for text, c in top:
        short = text if len(text) <= 42 else text[:39] + "…"
        labels.append(short)
    vals = [c for _, c in top]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4),
                             gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.35})
    ax = axes[0]
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, color=RED, alpha=0.82, height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v + 0.4, yi, str(v), va="center", fontsize=11, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("сколько раз документ встречается дословно")
    ax.set_xlim(0, max(vals) * 1.22)
    ax.set_title("Шесть самых частых копий", fontsize=13)
    ax.grid(True, axis="x", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)

    ax2 = axes[1]
    ax2.plot(thresholds, pair_counts, marker="o", color=BLUE, lw=2.2)
    for t, c in zip(thresholds, pair_counts):
        ax2.annotate(str(c), (t, c), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=10, color=BLUE)
    ax2.set_xlabel("порог Jaccard по 3-граммам")
    ax2.set_ylabel("число пар-кандидатов")
    ax2.set_ylim(880, 1620)
    ax2.set_title("Пара «почти копий» зависит от порога", fontsize=13)
    ax2.grid(True, color=GRID, lw=0.4, alpha=0.5); ax2.set_axisbelow(True)
    fig.suptitle(
        f"Точный dedup убирает {extra} документов и {redundant_tok} токенов "
        f"({str(FACTS['redundant_share']).replace('.', ',')}% корпуса)", y=1.05, fontsize=13.5)
    save(fig, OUT / "duplicates.png")

    # sidenote: длины документов
    lens = np.array([len(t) for _, t in CORPUS])
    FACTS.update(len_median=float(np.median(lens)), len_mean=round(float(lens.mean()), 1),
                 len_max=int(lens.max()), len_p99=float(np.percentile(lens, 99)))
    assert FACTS["len_median"] == 15.0 and FACTS["len_max"] == 210
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    ax.hist(lens, bins=np.arange(0, 121, 4), color=BLUE, alpha=0.85)
    ax.axvline(float(np.median(lens)), color=RED, lw=1.8)
    ax.text(float(np.median(lens)) + 3, ax.get_ylim()[1] * 0.82, "медиана 15",
            fontsize=9.5, color=RED)
    ax.axvline(lens.mean(), color=GOLD, lw=1.6, ls=(0, (4, 3)))
    ax.text(lens.mean() + 3, ax.get_ylim()[1] * 0.62, "среднее 19,0", fontsize=9.5, color=GOLD)
    ax.set_xlabel("длина документа, токенов"); ax.set_ylabel("документов")
    ax.set_title("Длины скошены: хвост тянет вес", fontsize=11)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "lengths.png")

    # sidenote: Zipf
    freq = Counter(t for _, toks in CORPUS for t in toks)
    ranks = np.arange(1, 2001)
    vals_f = np.array([c for _, c in freq.most_common(2000)])
    FACTS.update(vocab_all=len(freq), top_token=freq.most_common(1)[0][0],
                 top_token_count=freq.most_common(1)[0][1],
                 hapax=sum(1 for c in freq.values() if c == 1))
    FACTS["hapax_share"] = round(100 * FACTS["hapax"] / FACTS["vocab_all"], 1)
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    ax.loglog(ranks, vals_f[:2000], color=VIOLET, lw=1.8)
    ax.set_xlabel("ранг токена"); ax.set_ylabel("частота")
    ax.set_title("Закон Ципфа на реальном корпусе", fontsize=11)
    ax.grid(True, which="both", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "zipf.png")

    # sidenote: сколько пар выживает при разных порогах — уже в фигуре; вместо этого
    # доля токенов, оставшихся после каждого правила dedup
    rules = ["без\ndedup", "точный\ndedup", "J ≥ 0,9", "J ≥ 0,7"]
    kept = [n_tok, n_tok - redundant_tok]
    S = [shingles(t) for _, t in CORPUS]
    for thr in (0.9, 0.7):
        removed_docs = set()
        inv = defaultdict(list)
        for i, s in enumerate(S):
            for g in s:
                inv[g].append(i)
        cand = set()
        for g, lst in inv.items():
            if len(lst) <= 40:
                for a in range(len(lst)):
                    for b in range(a + 1, len(lst)):
                        cand.add((lst[a], lst[b]))
        for a, b in sorted(cand):
            if a in removed_docs or b in removed_docs:
                continue
            if len(S[a] & S[b]) / len(S[a] | S[b]) >= thr:
                removed_docs.add(b)
        kept.append(n_tok - sum(len(CORPUS[i][1]) for i in removed_docs))
    FACTS["kept_j90"] = kept[2]; FACTS["kept_j70"] = kept[3]
    FACTS["kept_j70_share"] = round(100 * kept[3] / n_tok, 1)
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    ax.bar(rules, [100 * k / n_tok for k in kept],
           color=[MUTED, BLUE, GREEN, GOLD], alpha=0.85)
    for i, k in enumerate(kept):
        ax.text(i, 100 * k / n_tok + 1.4, f"{100 * k / n_tok:.1f}%", ha="center",
                fontsize=9.5, color=INK)
    ax.set_ylim(0, 112); ax.set_ylabel("осталось токенов, %")
    ax.set_title("Строгость правила = цена в токенах", fontsize=11)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "dedup-cost.png")
    print("fig2", {k: FACTS[k] for k in
                  ("n_unique", "extra_copies", "redundant_tok", "redundant_share",
                   "pairs_j80", "kept_j70_share", "hapax_share", "top_token",
                   "top_token_count", "vocab_all")})


# ------------------------------------------------------- bigram language model
class Bigram:
    def __init__(self, docs, vocab_size, k=0.3):
        self.uni = Counter(); self.bi = Counter(); self.V = vocab_size; self.k = k
        for d in docs:
            for a, b in zip(d, d[1:]):
                self.uni[a] += 1
                self.bi[(a, b)] += 1
        best = defaultdict(lambda: (None, -1))
        for (a, b), c in self.bi.items():
            if c > best[a][1]:
                best[a] = (b, c)
        self.best = {a: b for a, (b, c) in best.items()}
        self.fallback = self.uni.most_common(1)[0][0]

    def nll(self, docs):
        tot = 0.0; n = 0
        for d in docs:
            for a, b in zip(d, d[1:]):
                p = (self.bi[(a, b)] + self.k) / (self.uni[a] + self.k * self.V)
                tot -= math.log(p); n += 1
        return tot / n

    def acc(self, docs):
        ok = 0; n = 0
        for d in docs:
            for a, b in zip(d, d[1:]):
                ok += self.best.get(a, self.fallback) == b
                n += 1
        return ok / n


def wrap(toks):
    return ["<s>"] + toks + ["</s>"]


def take_tokens(pool, budget, rng):
    out = []; got = 0; i = 0
    idx = list(range(len(pool))); rng.shuffle(idx)
    while got < budget:
        d = pool[idx[i % len(idx)]]
        out.append(d); got += len(d) - 1; i += 1
    return out


# ---------------------------------------------------------------- fig 78.3 mixture
def fig_mixture() -> None:
    docs = [(l, wrap(t)) for l, t in CORPUS]
    rng = random.Random(78); rng.shuffle(docs)
    ham = [d for l, d in docs if l == "ham"]
    spam = [d for l, d in docs if l == "spam"]
    ham_val, ham_tr = ham[:400], ham[400:]
    spam_val, spam_tr = spam[:120], spam[120:]
    counts = Counter(t for d in ham_tr + spam_tr for t in d)
    V = {w for w, c in counts.items() if c >= 2}
    Vs = len(V) + 1
    mp = lambda d: [t if t in V else "<unk>" for t in d]
    ham_tr = [mp(d) for d in ham_tr]; spam_tr = [mp(d) for d in spam_tr]
    ham_val = [mp(d) for d in ham_val]; spam_val = [mp(d) for d in spam_val]
    ham_pool_tok = sum(len(d) - 1 for d in ham_tr)
    spam_pool_tok = sum(len(d) - 1 for d in spam_tr)
    assert (ham_pool_tok, spam_pool_tok) == (82008, 18814)
    FACTS.update(vocab_kept=len(V), ham_pool_tok=ham_pool_tok, spam_pool_tok=spam_pool_tok)
    assert FACTS["vocab_kept"] == 4086

    BUD = 60000
    ws = [round(0.05 * i, 2) for i in range(21)]
    hv, sv = [], []
    for w in ws:
        r = random.Random(1000 + int(round(w * 100)))
        s = take_tokens(spam_tr, int(BUD * w), r) if w > 0 else []
        h = take_tokens(ham_tr, int(BUD * (1 - w)), r) if w < 1 else []
        m = Bigram(s + h, Vs)
        hv.append(m.nll(ham_val)); sv.append(m.nll(spam_val))
    hv = np.array(hv); sv = np.array(sv)
    eq = 0.5 * (hv + sv)
    nat = 0.8 * hv + 0.2 * sv
    w_eq = ws[int(np.argmin(eq))]
    w_nat = ws[int(np.argmin(nat))]
    i0 = ws.index(0.0); i20 = ws.index(0.2); i100 = ws.index(1.0)
    FACTS.update(
        mix_ham0=round(float(hv[i0]), 2), mix_spam0=round(float(sv[i0]), 2),
        mix_ham20=round(float(hv[i20]), 2), mix_spam20=round(float(sv[i20]), 2),
        mix_ham100=round(float(hv[i100]), 2), mix_spam100=round(float(sv[i100]), 2),
        w_eq=w_eq, w_nat=w_nat,
        eq_min=round(float(eq.min()), 2), eq_at0=round(float(eq[i0]), 2),
        spam_epochs_at_1=round(BUD / spam_pool_tok, 2),
        spam_epochs_at_20=round(BUD * 0.2 / spam_pool_tok, 2),
        budget=BUD)
    assert FACTS["mix_ham0"] == 5.96 and FACTS["mix_spam0"] == 7.37, FACTS
    assert FACTS["mix_ham100"] == 7.33 and FACTS["mix_spam100"] == 5.25, FACTS
    assert FACTS["mix_ham20"] == 6.07 and FACTS["mix_spam20"] == 6.17, FACTS
    assert w_eq == 0.6 and w_nat == 0.25, (w_eq, w_nat)
    assert FACTS["spam_epochs_at_1"] == 3.19 and FACTS["spam_epochs_at_20"] == 0.64
    assert hv[0] < hv[-1] and sv[0] > sv[-1]

    fig, ax = plt.subplots(figsize=(9.2, 5.1))
    ax.plot(ws, hv, color=BLUE, lw=2.4, label="разговорные SMS (ham), отложенный NLL")
    ax.plot(ws, sv, color=RED, lw=2.4, label="рекламные SMS (spam), отложенный NLL")
    ax.plot(ws, eq, color=GREEN, lw=2.0, ls=(0, (5, 3)), label="среднее с весами 50/50")
    ax.plot(ws, nat, color=GOLD, lw=2.0, ls=(0, (2, 2)), label="среднее с весами 80/20")
    ax.scatter([w_eq], [eq.min()], s=70, color=GREEN, zorder=6)
    ax.scatter([w_nat], [nat.min()], s=70, color=GOLD, zorder=6)
    ax.annotate(f"оптимум 50/50: w={w_eq}".replace(".", ","), (w_eq, eq.min()),
                textcoords="offset points", xytext=(6, 16), fontsize=10, color=GREEN)
    ax.annotate(f"оптимум 80/20: w={w_nat}".replace(".", ","), (w_nat, nat.min()),
                textcoords="offset points", xytext=(6, -22), fontsize=10, color=GOLD)
    ax.set_xlabel("доля рекламных токенов в обучающем потоке, $w$")
    ax.set_ylabel("средний NLL на токен, натуральные логарифмы")
    ax.set_title("Смесь — это обмен: один домен выигрывает ровно там, где второй теряет",
                 fontsize=13.5)
    ax.legend(frameon=False, fontsize=10, loc="upper center")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "mixture.png")

    # sidenote: эффективные эпохи
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    wgrid = np.linspace(0.01, 1.0, 120)
    ax.plot(wgrid, BUD * wgrid / spam_pool_tok, color=RED, lw=2.0, label="малый источник")
    ax.plot(wgrid, BUD * wgrid / ham_pool_tok, color=BLUE, lw=2.0, label="большой источник")
    ax.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (3, 3)))
    ax.text(0.03, 1.12, "одна эпоха", fontsize=9, color=MUTED)
    ax.set_xlabel("вес источника $w_j$"); ax.set_ylabel("эффективных эпох $E_j$")
    ax.set_title("Один вес — разное число проходов", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "epochs.png")
    print("fig3", {k: FACTS[k] for k in
                   ("mix_ham0", "mix_spam0", "mix_ham20", "mix_spam20", "mix_ham100",
                    "mix_spam100", "w_eq", "w_nat", "spam_epochs_at_1", "vocab_kept")})


# ------------------------------------------------- fig 78.4 / 78.5: repeats and leakage
def split_for_leak():
    docs = [wrap(t) for _, t in CORPUS]
    rng = random.Random(78); rng.shuffle(docs)
    val, bench, pool = docs[:600], docs[600:1000], docs[1000:]
    counts = Counter(t for d in pool for t in d)
    V = {w for w, c in counts.items() if c >= 2}
    mp = lambda d: [t if t in V else "<unk>" for t in d]
    return [mp(d) for d in val], [mp(d) for d in bench], [mp(d) for d in pool], len(V) + 1


def fig_repeats() -> None:
    val, bench, pool, Vs = split_for_leak()
    BUD = 60000
    prem, rest = pool[:150], pool[150:]
    ptok = sum(len(d) - 1 for d in prem)
    assert ptok == 3386, ptok
    ks = [1, 2, 4, 8, 16]
    seen, held = [], []
    for k in ks:
        r = random.Random(500 + k)
        ds = prem * k + take_tokens(rest, BUD - ptok * k, r)
        m = Bigram(ds, Vs)
        seen.append(m.nll(prem)); held.append(m.nll(val))
    FACTS.update(rep_seen1=round(seen[0], 2), rep_held1=round(held[0], 2),
                 rep_seen16=round(seen[-1], 2), rep_held16=round(held[-1], 2),
                 rep_prem_tok=ptok, rep_prem_docs=len(prem),
                 rep_share16=round(100 * ptok * 16 / BUD, 1))
    assert FACTS["rep_seen1"] == 5.59 and FACTS["rep_held1"] == 5.99, FACTS
    assert FACTS["rep_seen16"] == 4.14 and FACTS["rep_held16"] == 6.64, FACTS
    assert FACTS["rep_share16"] == 90.3
    assert all(seen[i] > seen[i + 1] for i in range(len(ks) - 1))
    assert all(held[i] < held[i + 1] for i in range(len(ks) - 1))

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.plot(ks, seen, marker="o", color=RED, lw=2.4,
            label="loss на повторённых документах (то, что видит обучение)")
    ax.plot(ks, held, marker="s", color=BLUE, lw=2.4,
            label="loss на отложенных документах (то, что нужно на деле)")
    for k, s, h in zip(ks, seen, held):
        ax.annotate(f"{s:.2f}".replace(".", ","), (k, s), textcoords="offset points",
                    xytext=(0, -18), ha="center", fontsize=9.5, color=RED)
        ax.annotate(f"{h:.2f}".replace(".", ","), (k, h), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9.5, color=BLUE)
    ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("сколько раз повторён «премиальный» набор из 150 документов")
    ax.set_ylabel("средний NLL на токен")
    ax.set_title("Повтор снижает loss там, где смотрит оптимизатор, и повышает там, где важно",
                 fontsize=13.2)
    ax.legend(frameon=False, fontsize=10, loc="center left")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "repeats.png")
    print("fig4", {k: FACTS[k] for k in ("rep_seen1", "rep_held1", "rep_seen16", "rep_held16")})


def fig_contamination() -> None:
    val, bench, pool, Vs = split_for_leak()
    BUD = 40000
    half = len(bench) // 2
    leaked_pool, clean = bench[:half], bench[half:]
    COP = 3

    def near(d, r):
        return ["<s>"] + [t for t in d[1:-1] if r.random() > 0.3] + ["</s>"]

    ps = [0.0, 0.25, 0.5, 0.75, 1.0]
    res = {}
    for mode in ("exact", "near"):
        rows = []
        for p in ps:
            r = random.Random(900)
            n = int(half * p)
            leak = []
            for i in range(n):
                for _ in range(COP):
                    leak.append(leaked_pool[i] if mode == "exact" else near(leaked_pool[i], r))
            ltok = sum(len(d) - 1 for d in leak)
            m = Bigram(leak + take_tokens(pool, BUD - ltok, r), Vs)
            rows.append((m.acc(leaked_pool), m.acc(clean), m.acc(bench)))
        res[mode] = rows

    ex = res["exact"]; nr = res["near"]
    pct = lambda x: round(100 * x, 1)
    FACTS.update(
        leak_base=pct(ex[0][0]), leak_exact100=pct(ex[-1][0]), leak_near100=pct(nr[-1][0]),
        leak_clean_base=pct(ex[0][1]), leak_clean100=pct(ex[-1][1]),
        leak_bench_base=pct(ex[0][2]), leak_bench100=pct(ex[-1][2]),
        leak_exact50=pct(ex[2][0]), leak_near50=pct(nr[2][0]), leak_copies=COP)
    assert FACTS["leak_base"] == 19.4 and FACTS["leak_exact100"] == 36.2, FACTS
    assert FACTS["leak_near100"] == 30.1 and FACTS["leak_clean100"] == 18.8, FACTS
    assert FACTS["leak_bench_base"] == 19.3 and FACTS["leak_bench100"] == 27.9, FACTS
    assert abs(FACTS["leak_clean_base"] - FACTS["leak_clean100"]) < 1.0

    xs = [100 * p for p in ps]
    fig, ax = plt.subplots(figsize=(9.2, 5.1))
    ax.plot(xs, [100 * r[0] for r in ex], marker="o", color=RED, lw=2.4,
            label="точные копии в train: accuracy на «утёкших» вопросах")
    ax.plot(xs, [100 * r[0] for r in nr], marker="^", color=GOLD, lw=2.2,
            label="почти копии (выброшено 30% токенов)")
    ax.plot(xs, [100 * r[2] for r in ex], marker="d", color=VIOLET, lw=2.0, ls=(0, (5, 3)),
            label="средняя accuracy по всему benchmark")
    ax.plot(xs, [100 * r[1] for r in ex], marker="s", color=BLUE, lw=2.4,
            label="чистая половина benchmark — настоящий навык")
    ax.set_xlabel("доля тестовых документов, попавших в обучение, %")
    ax.set_ylabel("доля верно угаданных следующих токенов, %")
    ax.set_ylim(15, 40)
    ax.set_title("Утечка поднимает метрику, не меняя навыка", fontsize=13.5)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "contamination.png")

    # sidenote: разрыв «оригинал минус перефразировка» как индикатор
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    gap = [100 * (a[0] - b[1]) for a, b in zip(ex, ex)]
    ax.bar([str(int(100 * p)) + "%" for p in ps], gap, color=RED, alpha=0.85)
    for i, g in enumerate(gap):
        ax.text(i, g + 0.4, f"{g:.1f}", ha="center", fontsize=9.5, color=INK)
    ax.set_xlabel("доля утечки"); ax.set_ylabel("разрыв, п. п.")
    ax.set_ylim(0, max(gap) * 1.25)
    ax.set_title("Разрыв «утёкшие минус чистые»", fontsize=11)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "leak-gap.png")
    FACTS["leak_gap100"] = round(gap[-1], 1)
    assert FACTS["leak_gap100"] == 17.4, FACTS["leak_gap100"]
    print("fig5", {k: FACTS[k] for k in
                   ("leak_base", "leak_exact100", "leak_near100", "leak_clean100",
                    "leak_bench_base", "leak_bench100", "leak_gap100")})


# ---------------------------------------------------------------- fig 78.2 pipeline
def fig_pipeline() -> None:
    """Каскад фильтров: сколько токенов доживает до обучения (реальные доли SMS-корпуса)."""
    n_tok = FACTS["n_tok"]
    # ступень 1: точный dedup (оставляем первое вхождение)
    seen = set(); stage1 = []
    for _, t in CORPUS:
        k = " ".join(t)
        if k in seen:
            continue
        seen.add(k); stage1.append(t)
    # ступень 2: near-dup по 3-граммам, порог 0,7 (жадно, поверх stage1)
    S = [shingles(t) for t in stage1]
    inv = defaultdict(list)
    for i, s in enumerate(S):
        for g in s:
            inv[g].append(i)
    cand = set()
    for g, lst in inv.items():
        if len(lst) <= 40:
            for a in range(len(lst)):
                for b in range(a + 1, len(lst)):
                    cand.add((lst[a], lst[b]))
    dead = set()
    for a, b in sorted(cand):
        if a in dead or b in dead:
            continue
        if len(S[a] & S[b]) / len(S[a] | S[b]) >= 0.7:
            dead.add(b)
    stage2 = [t for i, t in enumerate(stage1) if i not in dead]
    # ступень 3: фильтр длины; ступень 4: доля букв
    stage3 = [t for t in stage2 if len(t) >= 5]

    def letter_share(t):
        chars = "".join(t)
        return sum(c.isalpha() for c in chars) / len(chars) if chars else 0.0

    stage4 = [t for t in stage3 if letter_share(t) >= 0.5]
    vals = [n_tok] + [sum(len(t) for t in st) for st in (stage1, stage2, stage3, stage4)]
    labels = ["сырой корпус", "точный dedup", "near-dup $J\\geq0{,}7$",
              "длина $\\geq5$ токенов", "доля букв $\\geq0{,}5$"]
    assert all(vals[i] > vals[i + 1] for i in range(4)), vals
    FACTS.update(pipe_dedup=vals[1], pipe_near=vals[2], pipe_len=vals[3],
                 pipe_letters=vals[4],
                 pipe_final_share=round(100 * vals[4] / n_tok, 1),
                 pipe_near_share=round(100 * vals[2] / n_tok, 1),
                 pipe_len_lost=vals[2] - vals[3],
                 pipe_letters_lost=vals[3] - vals[4])
    assert FACTS["pipe_final_share"] == 88.0, FACTS["pipe_final_share"]
    assert FACTS["pipe_near_share"] == 88.3, FACTS["pipe_near_share"]
    dedup = stage2

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    ys = np.arange(len(vals))[::-1]
    colors = [MUTED, BLUE, GREEN, GOLD, VIOLET]
    ax.barh(ys, [100 * v / n_tok for v in vals], color=colors, alpha=0.85, height=0.6)
    for y, v in zip(ys, vals):
        ax.text(100 * v / n_tok + 0.8, y, f"{v} токенов ({100 * v / n_tok:.1f}%)".replace(".", ","),
                va="center", fontsize=10.5, color=INK)
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(0, 118); ax.set_xlabel("доля токенов исходного корпуса, %")
    ax.set_title("Каждый фильтр — это решение, чей текст не попадёт в обучение",
                 fontsize=13.5)
    ax.grid(True, axis="x", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "pipeline.png")

    # sidenote: что именно выкидывает фильтр длины — распределение по классам
    dropped = [t for t in dedup if len(t) < 5]
    lab_by_key = {}
    for lab, t in CORPUS:
        lab_by_key.setdefault(" ".join(t), lab)
    d_spam = sum(1 for t in dropped if lab_by_key[" ".join(t)] == "spam")
    d_ham = len(dropped) - d_spam
    FACTS.update(drop_short=len(dropped), drop_short_ham=d_ham, drop_short_spam=d_spam)
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    ax.bar(["разговорные", "рекламные"], [d_ham, d_spam], color=[BLUE, RED], alpha=0.85)
    for i, v in enumerate([d_ham, d_spam]):
        ax.text(i, v + max(d_ham, d_spam) * 0.03, str(v), ha="center", fontsize=10, color=INK)
    ax.set_ylabel("документов удалено")
    ax.set_title("Фильтр длины бьёт по одному жанру", fontsize=11)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "short-filter.png")
    print("fig2b", {k: FACTS[k] for k in
                    ("pipe_len", "pipe_letters", "pipe_final_share", "drop_short",
                     "drop_short_ham", "drop_short_spam")})


def prose_checks() -> None:
    """Числа, которые встречаются в тексте урока как производные от измеренных."""
    n = FACTS["n_docs"]
    pairs_total = n * (n - 1) // 2
    assert pairs_total == 15531951, pairs_total
    avg_copy = FACTS["redundant_tok"] / FACTS["extra_copies"]
    assert round(avg_copy, 1) == 22.8, avg_copy
    ppl1 = math.exp(FACTS["mix_ham0"]); ppl2 = math.exp(FACTS["mix_spam0"])
    assert round(ppl1) == 388 and round(ppl2) == 1588, (ppl1, ppl2)
    assert round(ppl2 / ppl1, 1) == 4.1
    ratio = (FACTS["leak_near100"] - FACTS["leak_base"]) / (FACTS["leak_exact100"] - FACTS["leak_base"])
    assert round(ratio, 2) == 0.64, ratio
    mid = 0.5 * FACTS["leak_exact100"] + 0.5 * FACTS["leak_clean100"]
    assert round(mid, 1) == 27.5, mid
    assert FACTS["drop_short_ham"] / FACTS["drop_short_spam"] == 45.0
    assert round(FACTS["pipe_dedup"] / FACTS["n_tok"] * 100 - FACTS["pipe_near_share"], 1) == 2.7
    assert round(100 - 100 * FACTS["kept_j70"] / FACTS["n_tok"], 1) == 11.6
    assert round(100 * FACTS["kept_j90"] / FACTS["n_tok"], 1) == 90.1
    assert round(FACTS["spam_pool_tok"] / FACTS["budget"], 3) == 0.314
    assert round(1 / (2 * math.sqrt(128)), 3) == 0.044
    FACTS.update(pairs_total=pairs_total, avg_copy=round(avg_copy, 1),
                 ppl_ham0=round(ppl1), ppl_spam0=round(ppl2), near_ratio=round(ratio, 2))
    print("prose_checks ok:", {k: FACTS[k] for k in
                               ("pairs_total", "avg_copy", "ppl_ham0", "ppl_spam0", "near_ratio")})


def main() -> None:
    fig_next_token()
    fig_duplicates()
    fig_pipeline()
    fig_mixture()
    fig_repeats()
    fig_contamination()
    prose_checks()
    print("\n--- ВСЕ ЧИСЛА УРОКА 78 ---")
    for k in sorted(FACTS):
        print(f"{k} = {FACTS[k]}")


if __name__ == "__main__":
    main()
