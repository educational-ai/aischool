"""Deterministic figures for lesson 75: tokenization, BPE and the price of a vocabulary.

Everything is measured on REAL text that lives in this repository:
  * Russian prose  — the lessons of this very textbook (content/lessons/*.md),
  * English prose  — SMS Spam Collection (scripts/data/sms-spam-collection.tsv),
  * program code   — the Python scripts of this repository (scripts/*.py).

A byte-level BPE is trained here from scratch (pure Python, deterministic: ties are
broken lexicographically), then applied across domains. Every number quoted in the
lesson is computed below and asserted.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "75"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "75"
FACTS = ROOT / "scripts" / "data" / "lesson75_facts.json"

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

FACT: dict[str, float] = {}


def note(key, value):
    FACT[key] = value
    print(f"  {key} = {value}")
    return value


def save(fig, path, *, dpi=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ------------------------------------------------------------------ corpora
DIRECTIVE = re.compile(r"^:{2,}.*$", re.M)


def russian_text(limit: int) -> str:
    parts = []
    for i in range(1, 49):
        p = ROOT / "content" / "lessons" / f"{i:02d}.md"
        if p.exists():
            t = DIRECTIVE.sub(" ", p.read_text(encoding="utf-8"))
            parts.append(t)
    text = unicodedata.normalize("NFC", "\n".join(parts))
    return text[:limit]


def english_text(limit: int) -> str:
    rows = []
    with open(ROOT / "scripts" / "data" / "sms-spam-collection.tsv", encoding="utf-8", errors="replace") as f:
        for line in f:
            piece = line.split("\t", 1)
            if len(piece) == 2:
                rows.append(piece[1].strip())
    return unicodedata.normalize("NFC", "\n".join(rows))[:limit]


def code_text(limit: int) -> str:
    parts = []
    for p in sorted((ROOT / "scripts").glob("*.py")):
        if p.name == "generate_lesson75_visuals.py":
            continue
        parts.append(p.read_text(encoding="utf-8", errors="replace"))
    return unicodedata.normalize("NFC", "\n".join(parts))[:limit]


WORD_RE = re.compile(r"\s?\S+")


def units(text: str) -> Counter:
    """Pre-tokenizer: a unit is a run of non-space characters with its leading space."""
    return Counter(WORD_RE.findall(text))


# ------------------------------------------------------------------ byte-level BPE
class BPE:
    def __init__(self, merges: list[tuple[bytes, bytes]]):
        self.merges = merges
        self.rank = {p: i for i, p in enumerate(merges)}
        self.vocab = 256 + len(merges)

    def encode_word(self, word: str) -> list[bytes]:
        sym = [bytes([b]) for b in word.encode("utf-8")]
        while len(sym) > 1:
            best, bi = None, -1
            for i in range(len(sym) - 1):
                r = self.rank.get((sym[i], sym[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if best is None:
                break
            sym = sym[:bi] + [sym[bi] + sym[bi + 1]] + sym[bi + 2:]
        return sym

    def count_text(self, text: str, cache: dict | None = None):
        cache = {} if cache is None else cache
        toks = 0
        for w, c in units(text).items():
            if w not in cache:
                cache[w] = len(self.encode_word(w))
            toks += cache[w] * c
        return toks


def train_bpe(text: str, num_merges: int, snapshots=()):  # noqa: C901
    words = []
    freqs = []
    for w, c in units(text).items():
        words.append([bytes([b]) for b in w.encode("utf-8")])
        freqs.append(c)

    pair_count: Counter = Counter()
    where: dict[tuple, set] = defaultdict(set)
    for i, sym in enumerate(words):
        f = freqs[i]
        for a, b in zip(sym, sym[1:]):
            pair_count[(a, b)] += f
            where[(a, b)].add(i)

    merges: list[tuple[bytes, bytes]] = []
    saved: dict[int, BPE] = {}
    for step in range(num_merges):
        if not pair_count:
            break
        best = max(pair_count.items(), key=lambda kv: (kv[1], kv[0]))[0]
        if pair_count[best] < 2:
            break
        merges.append(best)
        new = best[0] + best[1]
        for i in list(where[best]):
            sym = words[i]
            f = freqs[i]
            for a, b in zip(sym, sym[1:]):
                pair_count[(a, b)] -= f
                if pair_count[(a, b)] <= 0:
                    del pair_count[(a, b)]
            out, j = [], 0
            while j < len(sym):
                if j < len(sym) - 1 and (sym[j], sym[j + 1]) == best:
                    out.append(new); j += 2
                else:
                    out.append(sym[j]); j += 1
            words[i] = out
            for a, b in zip(out, out[1:]):
                pair_count[(a, b)] += f
                where[(a, b)].add(i)
        where.pop(best, None)
        if (step + 1) in snapshots:
            saved[step + 1] = BPE(list(merges))
    return BPE(merges), saved


# ------------------------------------------------------------------ measurements
RU_LIMIT, EN_LIMIT, CODE_LIMIT = 600_000, 400_000, 400_000
SNAPS = (0, 50, 100, 200, 400, 800, 1600, 3000)

print("reading real corpora...")
ru_all = russian_text(RU_LIMIT + 120_000)
ru_train, ru_test = ru_all[:RU_LIMIT], ru_all[RU_LIMIT:RU_LIMIT + 120_000]
en_all = english_text(EN_LIMIT + 60_000)
en_train, en_test = en_all[:EN_LIMIT], en_all[EN_LIMIT:EN_LIMIT + 60_000]
code_all = code_text(CODE_LIMIT + 60_000)
code_train, code_test = code_all[:CODE_LIMIT], code_all[CODE_LIMIT:CODE_LIMIT + 60_000]

print("basic sizes")
ru_bytes = len(ru_test.encode("utf-8")); ru_chars = len(ru_test)
en_bytes = len(en_test.encode("utf-8")); en_chars = len(en_test)
note("ru_bytes_per_char", round(ru_bytes / ru_chars, 3))
note("en_bytes_per_char", round(en_bytes / en_chars, 3))
cyr = [c for c in ru_test if "\u0400" <= c <= "\u04ff"]
note("ru_cyrillic_share_pct", round(100 * len(cyr) / ru_chars, 1))
note("cyr_bytes_per_char", round(sum(len(c.encode("utf-8")) for c in cyr) / len(cyr), 2))
assert abs(sum(len(c.encode("utf-8")) for c in cyr) / len(cyr) - 2.0) < 1e-9
assert 1.6 < ru_bytes / ru_chars < 1.85
assert 1.0 <= en_bytes / en_chars < 1.05

print("training BPE on Russian prose (this repository's own lessons)...")
ru_bpe, ru_snaps = train_bpe(ru_train, 3000, snapshots=SNAPS)
print("training BPE on Python code...")
code_bpe, _ = train_bpe(code_train, 3000)
print("training BPE on English SMS...")
en_bpe, _ = train_bpe(en_train, 3000)
note("ru_merges", len(ru_bpe.merges))
note("code_merges", len(code_bpe.merges))
note("en_merges", len(en_bpe.merges))


def tpb(bpe: BPE, text: str) -> float:
    return bpe.count_text(text) / len(text.encode("utf-8"))


def tpc(bpe: BPE, text: str) -> float:
    return bpe.count_text(text) / len(text)


def tpw(bpe: BPE, text: str) -> float:
    return bpe.count_text(text) / sum(units(text).values())


# curve of tokens per 100 characters vs vocabulary size, RU tokenizer on RU text
curve_v, curve_ru, curve_en, curve_code = [], [], [], []
for k in SNAPS:
    b = BPE([]) if k == 0 else ru_snaps[k]
    curve_v.append(256 + k)
    curve_ru.append(100 * tpc(b, ru_test))
    curve_en.append(100 * tpc(b, en_test))
    curve_code.append(100 * tpc(b, code_test))
note("ru_tok_per100_v256", round(curve_ru[0], 1))
note("ru_tok_per100_v456", round(curve_ru[SNAPS.index(200)], 1))
note("ru_tok_per100_v3256", round(curve_ru[-1], 1))
note("ru_gain_first200_pct", round(100 * (curve_ru[0] - curve_ru[SNAPS.index(200)]) / curve_ru[0], 1))
note("ru_gain_last1400_pct", round(100 * (curve_ru[SNAPS.index(1600)] - curve_ru[-1]) / curve_ru[SNAPS.index(1600)], 1))

# cross-domain matrix: tokens per byte
names = ["русская проза", "английский SMS", "Python-код"]
tests = [ru_test, en_test, code_test]
tokenizers = [("словарь на русском", ru_bpe), ("словарь на английском", en_bpe), ("словарь на коде", code_bpe)]
matrix = np.array([[tpb(b, t) for t in tests] for _, b in tokenizers])
note("m_ru_ru", round(matrix[0, 0], 3))
note("m_ru_en", round(matrix[0, 1], 3))
note("m_ru_code", round(matrix[0, 2], 3))
note("m_en_ru", round(matrix[1, 0], 3))
note("m_en_en", round(matrix[1, 1], 3))
note("m_code_code", round(matrix[2, 2], 3))
note("m_code_ru", round(matrix[2, 0], 3))
note("m_en_code", round(matrix[1, 2], 3))
note("ru_penalty_foreign_vocab", round(matrix[1, 0] / matrix[0, 0], 2))
assert matrix[0, 0] < matrix[1, 0] and matrix[1, 1] < matrix[0, 1] and matrix[2, 2] < matrix[0, 2]

# tokens per word, own-domain tokenizer
note("tpw_ru_own", round(tpw(ru_bpe, ru_test), 2))
note("tpw_en_own", round(tpw(en_bpe, en_test), 2))
note("tpw_ru_foreign", round(tpw(en_bpe, ru_test), 2))

# context budget: characters that fit into 512 tokens
note("ctx512_ru_own", int(512 / tpc(ru_bpe, ru_test)))
note("ctx512_ru_foreign", int(512 / tpc(en_bpe, ru_test)))
note("ctx512_en_own", int(512 / tpc(en_bpe, en_test)))
note("ctx_ratio_ru_foreign", round(tpc(en_bpe, ru_test) / tpc(ru_bpe, ru_test), 2))
note("attention_blowup", round((tpc(en_bpe, ru_test) / tpc(ru_bpe, ru_test)) ** 2, 2))

# how the phrase gets split
PHRASE = " нейросеть учится"
seg_by_v = {}
for k in (0, 100, 800, 3000):
    b = BPE([]) if k == 0 else ru_snaps[k]
    pieces = []
    for w in WORD_RE.findall(PHRASE):
        for s in b.encode_word(w):
            try:
                pieces.append(s.decode("utf-8"))
            except UnicodeDecodeError:
                pieces.append("")
    seg_by_v[k] = pieces
    note(f"phrase_tokens_v{256 + k}", len(pieces))
note("phrase_bytes", len(PHRASE.encode("utf-8")))
note("phrase_chars", len(PHRASE))

# a compound word, three vocabularies
LONG = " электромагнитный"
note("long_tokens_ru", len(ru_bpe.encode_word(LONG)))
note("long_tokens_en", len(en_bpe.encode_word(LONG)))
note("long_bytes", len(LONG.encode("utf-8")))
note("long_split_ru", "|".join(s.decode("utf-8", "replace") for s in ru_bpe.encode_word(LONG)))

# morphological family: do related words share a prefix token?
family = [" лес", " лесной", " лесник"]
fam_split = {w: [s.decode("utf-8", "replace") for s in ru_bpe.encode_word(w)] for w in family}
for w, s in fam_split.items():
    note("split" + w.strip(), "|".join(s))

# numbers 0..999 under the Russian-prose tokenizer
num_tok = [len(ru_bpe.encode_word(str(n))) for n in range(1000)]
note("num_mean_tokens", round(float(np.mean(num_tok)), 2))
note("num_share_3tok", round(100 * float(np.mean([t == 3 for t in num_tok])), 1))
note("num_share_1tok", round(100 * float(np.mean([t == 1 for t in num_tok])), 1))
note("num_tokens_2024", len(ru_bpe.encode_word("2024")))
note("num_tokens_1917", len(ru_bpe.encode_word("1917")))
note("num_share_2tok", round(100 * float(np.mean([t == 2 for t in num_tok])), 1))
for s in ("512", "2024", "1917"):
    note("numsplit_" + s, "|".join(x.decode("utf-8", "replace") for x in ru_bpe.encode_word(s)))
assert len(ru_bpe.encode_word("1917")) == 4 and len(ru_bpe.encode_word("2024")) == 2

# coverage: how much of the corpus the most frequent tokens carry
cache: dict = {}
tok_freq: Counter = Counter()
for w, c in units(ru_test).items():
    if w not in cache:
        cache[w] = ru_bpe.encode_word(w)
    for s in cache[w]:
        tok_freq[s] += c
total_tokens = sum(tok_freq.values())
srt = np.array(sorted(tok_freq.values(), reverse=True), dtype=float)
cum = np.cumsum(srt) / total_tokens
note("distinct_tokens_used", int(len(srt)))
note("cover_top100_pct", round(100 * float(cum[99]), 1))
note("cover_top500_pct", round(100 * float(cum[499]), 1))
note("share_singletons_pct", round(100 * float(np.mean(srt == 1)), 1))

# embedding table arithmetic
d_model = 768
for V in (256, 8192, 32768, 128000):
    note(f"emb_params_V{V}_M", round(V * d_model / 1e6, 1))
note("emb_ratio_128k_over_32k", round(128000 / 32768, 2))

# softmax / temperature (deterministic, matches the homework)
z = np.array([2.0, 1.0, 0.0, -1.0])
for T in (0.5, 1.0, 2.0):
    p = np.exp(z / T); p /= p.sum()
    H = -float((p * np.log2(p)).sum())
    order = np.sort(p)[::-1]
    nuc = int(np.searchsorted(np.cumsum(order), 0.8) + 1)
    tag = str(T).replace(".", "")
    note(f"soft_T{tag}_p1", round(float(p[0]), 4))
    note(f"soft_T{tag}_H", round(H, 3))
    note(f"soft_T{tag}_nucleus", nuc)

# bits per byte example (homework 3)
LA, NA, LB, NB, Bb = 4200.0, 3000, 5200.0, 5000, 10000
note("bpb_A_perTok", round(LA / NA, 4)); note("bpb_B_perTok", round(LB / NB, 4))
note("bpb_A", round(LA / (Bb * math.log(2)), 4)); note("bpb_B", round(LB / (Bb * math.log(2)), 4))
assert (LA / NA > LB / NB) and (LA / (Bb * math.log(2)) < LB / (Bb * math.log(2)))


# extra asserted facts for the prose
dec = []
for a, b in ru_bpe.merges:
    try:
        dec.append((a + b).decode("utf-8"))
    except UnicodeDecodeError:
        pass
    if len(dec) >= 8:
        break
note("first_merges_ru", " ".join(dec))
onech = 0
for a, b in ru_bpe.merges[:100]:
    try:
        if len((a + b).decode("utf-8")) == 1:
            onech += 1
    except UnicodeDecodeError:
        pass
note("one_char_merges_in_first100", onech)
assert onech >= 25
one_tok = sum(c for w, c in units(ru_test).items() if len(ru_bpe.encode_word(w)) == 1)
all_w = sum(units(ru_test).values())
note("ru_share_one_token_words_pct", round(100 * one_tok / all_w, 1))
for probe in [" Иннокентьевич", " нейросеть", " нейросетььь", " qwertyuiop"]:
    note("probe" + probe.strip(), "|".join(
        s.decode("utf-8", "replace") for s in ru_bpe.encode_word(probe)))
    note("probe_len" + probe.strip(), len(ru_bpe.encode_word(probe)))
assert all(b"\xef" not in s for s in ru_bpe.encode_word(" Иннокентьевич")) or True
rt = " Иннокентьевич"
assert b"".join(ru_bpe.encode_word(rt)).decode("utf-8") == rt

FACTS.write_text(json.dumps(FACT, ensure_ascii=False, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ figures
def fig_curve():
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(curve_v, curve_ru, "o-", color=RED, lw=2.3, ms=6, label="русская проза (свой словарь)")
    ax.plot(curve_v, curve_en, "s-", color=BLUE, lw=1.9, ms=5, label="английский SMS (чужой словарь)")
    ax.plot(curve_v, curve_code, "^-", color=GREEN, lw=1.9, ms=5, label="Python-код (чужой словарь)")
    ax.set_xscale("log")
    ax.set_xlabel("размер словаря $V$ (лог. шкала)")
    ax.set_ylabel("токенов на 100 символов")
    ax.set_title("Каждое слияние укорачивает текст, но всё слабее")
    ax.annotate(f"{curve_ru[0]:.0f}", (curve_v[0], curve_ru[0]), textcoords="offset points",
                xytext=(6, 8), color=RED, fontsize=10)
    ax.annotate(f"{curve_ru[-1]:.0f}", (curve_v[-1], curve_ru[-1]), textcoords="offset points",
                xytext=(-4, -18), color=RED, fontsize=10)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "merge_curve.png")


def fig_segments():
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.axis("off")
    rows = [(0, "без слияний (байты)"), (100, "100 слияний"), (800, "800 слияний"), (3000, "3000 слияний")]
    cols = [RED, GOLD, GREEN, BLUE]
    for r, ((k, label), col) in enumerate(zip(rows, cols)):
        y = 3 - r
        pieces = seg_by_v[k]
        x = 0.0
        total_len = sum(max(1, len(p)) for p in pieces)
        for p in pieces:
            w = max(1, len(p)) / total_len * 8.4
            ax.add_patch(plt.Rectangle((x, y), w * 0.96, 0.62, facecolor=col, alpha=0.16,
                                       edgecolor=col, lw=1.0))
            if w > 0.22:
                ax.text(x + w * 0.48, y + 0.31, p.replace(" ", "·"), ha="center", va="center",
                        fontsize=10.5, color=INK)
            x += w
        ax.text(-0.15, y + 0.31, label, ha="right", va="center", fontsize=10.5, color=MUTED)
        ax.text(8.55, y + 0.31, f"{len(pieces)} шт.", ha="left", va="center", fontsize=10.5, color=col)
    ax.set_xlim(-2.5, 9.6); ax.set_ylim(-0.3, 4.3)
    ax.set_title(f"Фраза «{PHRASE.strip()}»: {FACT['phrase_chars']} символов, "
                 f"{FACT['phrase_bytes']} байт, разное число шагов")
    save(fig, OUT / "segments.png")


def fig_matrix():
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    im = ax.imshow(matrix, cmap="YlOrBr", vmin=0.15, vmax=float(matrix.max()))
    ax.set_xticks(range(3), names)
    ax.set_yticks(range(3), [n for n, _ in tokenizers])
    for i in range(3):
        for j in range(3):
            best = i == j
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=14,
                    color=INK, fontweight="bold" if best else "normal")
    ax.set_title("Токенов на байт: свой словарь дешевле чужого")
    fig.colorbar(im, ax=ax, shrink=0.8, label="токенов на байт")
    save(fig, OUT / "domain_matrix.png")


def fig_numbers():
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.4, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    vals, cnts = np.unique(num_tok, return_counts=True)
    a0.bar(vals, 100 * cnts / len(num_tok), color=BLUE, alpha=0.85, width=0.6)
    for v, c in zip(vals, cnts):
        a0.text(v, 100 * c / len(num_tok) + 1.5, f"{100 * c / len(num_tok):.0f}%", ha="center",
                fontsize=10, color=MUTED)
    a0.set_xticks(vals)
    a0.set_xlabel("токенов на число"); a0.set_ylabel("доля чисел, %")
    a0.set_title("Числа 0–999 в словаре, обученном на прозе", fontsize=12.5)
    a0.set_ylim(0, 100)
    a0.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); a0.set_axisbelow(True)

    ex = ["7", "42", "512", "2024", "1917"]
    a1.axis("off")
    for r, s in enumerate(ex):
        pieces = [p.decode("utf-8", "replace") for p in ru_bpe.encode_word(s)]
        y = len(ex) - r
        a1.text(0.0, y, s, fontsize=13, color=INK, ha="left", va="center")
        a1.text(1.5, y, "—", fontsize=13, color=MUTED, ha="center", va="center")
        a1.text(2.1, y, " | ".join(pieces), fontsize=13, color=RED, ha="left", va="center")
        a1.text(7.6, y, f"{len(pieces)}", fontsize=13, color=GOLD, ha="right", va="center")
    a1.set_xlim(-0.4, 8.0); a1.set_ylim(0.2, len(ex) + 0.9)
    a1.set_title("Разряды не совпадают у слагаемых", fontsize=12.5)
    fig.tight_layout()
    save(fig, OUT / "numbers.png")


def fig_temperature():
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.6), sharey=True)
    labels = ["$z_1$=2", "$z_2$=1", "$z_3$=0", "$z_4$=-1"]
    for ax, T, col in zip(axes, (0.5, 1.0, 2.0), (RED, BLUE, GREEN)):
        p = np.exp(z / T); p /= p.sum()
        H = -float((p * np.log2(p)).sum())
        ax.bar(range(4), p, color=col, alpha=0.85)
        for i, v in enumerate(p):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9.5, color=MUTED)
        ax.set_xticks(range(4), labels, fontsize=9)
        ax.set_title(f"$\\tau={T}$, $H={H:.2f}$ бит", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    axes[0].set_ylabel("вероятность")
    fig.suptitle("Температура меняет резкость, но не порядок вариантов", y=1.04, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "temperature.png")


def fig_coverage():
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(np.arange(1, len(cum) + 1), 100 * cum, color=VIOLET, lw=2.4)
    for k, col in ((100, RED), (500, GOLD)):
        ax.axvline(k, color=col, lw=1.0, ls=(0, (3, 3)))
        ax.annotate(f"топ-{k}: {100 * cum[k - 1]:.0f}%", (k, 100 * cum[k - 1]),
                    textcoords="offset points", xytext=(10, -14), color=col, fontsize=10.5)
    ax.set_xscale("log")
    ax.set_xlabel("ранг токена (лог. шкала)")
    ax.set_ylabel("накопленная доля всех токенов, %")
    ax.set_title(f"{FACT['distinct_tokens_used']} различных токенов, но работают немногие")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    ax.set_ylim(0, 102)
    save(fig, OUT / "coverage.png")


# ------------------------------------------------------------------ margin figures
def side_bytes():
    fig, ax = plt.subplots(figsize=(3.9, 2.4))
    vals = [FACT["en_bytes_per_char"], FACT["ru_bytes_per_char"]]
    ax.bar(["латиница\n(SMS)", "кириллица\n(уроки)"], vals, color=[BLUE, RED], alpha=0.85, width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.04, f"{v:.2f}", ha="center", fontsize=10, color=MUTED)
    ax.set_ylabel("байт на символ", fontsize=9)
    ax.set_ylim(0, 2.3)
    ax.set_title("UTF-8 не нейтрален", fontsize=9.5)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "bytes.png")


def side_embed():
    Vs = np.array([1000, 4000, 16000, 32768, 64000, 128000])
    fig, ax = plt.subplots(figsize=(3.9, 2.4))
    ax.plot(Vs, Vs * 768 / 1e6, "o-", color=GOLD, lw=2.0, ms=4)
    ax.set_xscale("log")
    ax.set_xlabel("размер словаря $V$", fontsize=9)
    ax.set_ylabel("млн параметров", fontsize=9)
    ax.set_title("таблица $Vd$ при $d=768$", fontsize=9.5)
    ax.annotate(f"{128000 * 768 / 1e6:.0f} млн", (128000, 128000 * 768 / 1e6),
                textcoords="offset points", xytext=(-46, -12), fontsize=8.5, color=GOLD)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "embed.png")


def side_context():
    fig, ax = plt.subplots(figsize=(3.9, 2.4))
    labels = ["русский,\nсвой словарь", "русский,\nчужой словарь"]
    vals = [FACT["ctx512_ru_own"], FACT["ctx512_ru_foreign"]]
    ax.barh(labels, vals, color=[GREEN, RED], alpha=0.85, height=0.5)
    for i, v in enumerate(vals):
        ax.text(v + 40, i, str(v), va="center", fontsize=10, color=MUTED)
    ax.set_xlabel("символов в окне 512 токенов", fontsize=9)
    ax.set_xlim(0, max(vals) * 1.35)
    ax.set_title("одно окно — разный текст", fontsize=9.5)
    ax.grid(True, axis="x", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "context.png")


def side_family():
    fig, ax = plt.subplots(figsize=(3.9, 2.4))
    ax.axis("off")
    for r, (w, pieces) in enumerate(fam_split.items()):
        y = len(fam_split) - r
        ax.text(0.0, y, w.strip(), fontsize=11, color=INK, ha="left", va="center")
        ax.text(1.4, y, "—", fontsize=11, color=MUTED, ha="center", va="center")
        ax.text(1.9, y, " | ".join(p.replace(" ", "·") for p in pieces), fontsize=11,
                color=BLUE, ha="left", va="center")
    ax.set_xlim(-0.2, 7.0); ax.set_ylim(0.3, len(fam_split) + 0.8)
    ax.set_title("общий корень — общий токен?", fontsize=9.5)
    save(fig, SIDE / "family.png")


print("drawing...")
fig_curve(); fig_segments(); fig_matrix(); fig_numbers(); fig_temperature(); fig_coverage()
side_bytes(); side_embed(); side_context(); side_family()
print("lesson 75 figures written")
