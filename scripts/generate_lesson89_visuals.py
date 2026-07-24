"""Deterministic figures for lesson 89: agent scaffold, verifier and error budget.

Every number quoted in the lesson is computed here from the REAL SMS Spam Collection
(scripts/data/sms-spam-collection.tsv, 5574 messages) and asserted.

The lesson's agent performs a multi-step task: classify a batch of messages.  One
"step" is one message.  Step accuracy p is measured; end-to-end task success is
p^n both in theory and empirically.  Three verifiers are compared on the SAME
errors: a restatement (no independent evidence), a confidence threshold (same
evidence, softer) and a structurally independent classifier (different features).
Retry/budget curves and a reliability diagram close the picture.

Outputs:
  public/figures/lessons/89/{chain,verifier,checkpoints,budget,calibration}.png
  public/figures/sidenotes/89/{trust,compound,confusion}.png
  scripts/data/lesson89_facts.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

ROOT = Path(__file__).resolve().parents[1]
SMS = ROOT / "scripts" / "data" / "sms-spam-collection.tsv"
OUT = ROOT / "public" / "figures" / "lessons" / "89"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "89"
FACTS = ROOT / "scripts" / "data" / "lesson89_facts.json"

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


def save(fig, path, *, dpi=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------- real data
def load_sms():
    texts, labels = [], []
    with open(SMS, encoding="utf8", errors="replace") as f:
        for raw in f:
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue
            lab, _, txt = raw.partition("\t")
            if lab not in ("ham", "spam"):
                continue
            texts.append(txt)
            labels.append(1 if lab == "spam" else 0)
    return np.array(texts, dtype=object), np.array(labels)


URL = re.compile(r"(http|www\.|\.com|\.co\.uk)", re.I)
LONGNUM = re.compile(r"\d{5,}")


def structural(texts):
    """Evidence deliberately disjoint from the word counts the agent uses."""
    rows = []
    for t in texts:
        n = max(len(t), 1)
        letters = sum(ch.isalpha() for ch in t)
        rows.append([
            len(t) / 100.0,
            sum(ch.isdigit() for ch in t) / n,
            (sum(ch.isupper() for ch in t) / max(letters, 1)),
            t.count("!") / 3.0,
            1.0 if URL.search(t) else 0.0,
            1.0 if LONGNUM.search(t) else 0.0,
            1.0 if ("£" in t or "$" in t) else 0.0,
            sum(not ch.isalnum() and not ch.isspace() for ch in t) / n,
        ])
    return np.array(rows)


def build():
    texts, y = load_sms()
    rng = np.random.default_rng(89)
    order = rng.permutation(len(y))
    texts, y = texts[order], y[order]
    cut = 500          # разметка дорога: у агента только 500 размеченных примеров
    tr, te = slice(0, cut), slice(cut, len(y))

    vec = CountVectorizer(lowercase=True, token_pattern=r"[^\W_]+")
    Xtr = vec.fit_transform(texts[tr])
    Xte = vec.transform(texts[te])
    nb = MultinomialNB(alpha=0.2).fit(Xtr, y[tr])
    pred = nb.predict(Xte)
    prob = nb.predict_proba(Xte)
    conf = prob.max(axis=1)
    ok = (pred == y[te])

    Str, Ste = structural(texts[tr]), structural(texts[te])
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(Str, y[tr])
    ind = lr.predict(Ste)

    return {
        "n_all": int(len(y)), "n_train": int(cut), "n_test": int(len(y) - cut),
        "spam_share": float(y.mean()),
        "ok": ok, "conf": conf, "pred": pred, "true": y[te], "ind": ind,
        "vocab": int(len(vec.vocabulary_)),
    }


D = build()
ok = D["ok"]; conf = D["conf"]; ind = D["ind"]; pred = D["pred"]
p = float(ok.mean())
n_err = int((~ok).sum())
FACT.update(n_all=D["n_all"], n_train=D["n_train"], n_test=D["n_test"],
            vocab=D["vocab"], spam_share=round(D["spam_share"], 4),
            p_step=round(p, 4), n_err=n_err)
print(f"data: {D['n_all']} messages, test {D['n_test']}, vocab {D['vocab']}")
print(f"step accuracy p = {p:.4f}, errors = {n_err}")
assert D["n_all"] == 5574
assert 0.96 < p < 0.99 and n_err > 60


# ------------------------------------------------- fig 89.1: p^n and reality
def fig_chain():
    ns = [1, 2, 5, 10, 20, 40]
    rng = np.random.default_rng(890)
    emp = []
    for n in ns:
        hits = []
        for _ in range(400):
            v = rng.permutation(ok)
            k = len(v) // n
            blocks = v[: k * n].reshape(k, n)
            hits.append(blocks.all(axis=1).mean())
        emp.append(float(np.mean(hits)))
    theo = [p ** n for n in ns]
    for n, e, t in zip(ns, emp, theo):
        print(f"  n={n:>2}: empirical {e:.3f}, p^n {t:.3f}")
        assert abs(e - t) < 0.035, (n, e, t)
    FACT["chain_emp"] = {str(n): round(e, 4) for n, e in zip(ns, emp)}
    FACT["chain_theo"] = {str(n): round(t, 4) for n, t in zip(ns, theo)}

    grid = np.arange(1, 61)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for pp, c, ls in [(0.999, GREEN, "-"), (p, BLUE, "-"), (0.95, GOLD, "-"), (0.90, RED, "-")]:
        ax.plot(grid, pp ** grid, color=c, lw=2.2 if abs(pp - p) < 1e-9 else 1.7,
                ls=ls, label=f"p = {pp:.3f}".replace(".", ",")
                + (" — измерено на SMS" if abs(pp - p) < 1e-9 else ""))
    ax.scatter(ns, emp, s=55, color=INK, zorder=6, label="доля безошибочных партий (реальные данные)")
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    ax.text(59, 0.52, "половина задач проваливается", ha="right", fontsize=9.5, color=MUTED)
    ax.set_xlabel("число обязательных шагов $n$")
    ax.set_ylabel("вероятность безошибочной задачи $p^n$")
    ax.set_title("Хорошая точность шага — плохая надёжность цепочки")
    ax.set_ylim(0, 1.02); ax.set_xlim(1, 60)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "chain.png")


fig_chain()

# ------------------------------------ verifiers measured on the SAME errors
f_ind = float((ind[ok] != pred[ok]).mean())          # false flags on correct answers
d_ind = float((ind[~ok] != pred[~ok]).mean())        # detection on errors
# confidence verifier tuned to the SAME false-flag budget
tau = float(np.quantile(conf[ok], f_ind))
flag_conf = conf < tau
f_conf = float(flag_conf[ok].mean())
d_conf = float(flag_conf[~ok].mean())
print(f"independent verifier: detect {d_ind:.3f}, false flags {f_ind:.3f}")
print(f"confidence verifier (tau={tau:.6f}): detect {d_conf:.3f}, false flags {f_conf:.3f}")
assert d_ind > d_conf + 0.15 and d_ind > 0.5
assert abs(f_conf - f_ind) < 0.01

# самые опасные ошибки — уверенные: их порог уверенности не ловит вовсе
sure = conf > 0.99
sure_err = (~ok) & sure
d_ind_sure = float((ind[sure_err] != pred[sure_err]).mean())
d_conf_sure = float((conf[sure_err] < tau).mean())
print(f"confident errors: {int(sure_err.sum())} of {n_err}; "
      f"independent catches {d_ind_sure:.3f}, confidence catches {d_conf_sure:.3f}")
assert d_conf_sure == 0.0 and d_ind_sure > 0.25 and sure_err.sum() > 20
FACT.update(n_sure_err=int(sure_err.sum()), d_ind_sure=round(d_ind_sure, 4),
            d_conf_sure=round(d_conf_sure, 4),
            sure_err_share=round(float(sure_err.sum()) / n_err, 4))
FACT.update(d_ind=round(d_ind, 4), f_ind=round(f_ind, 4),
            d_conf=round(d_conf, 4), f_conf=round(f_conf, 4), tau=round(tau, 6))
FACT["n_err_caught_ind"] = int((ind[~ok] != pred[~ok]).sum())
FACT["n_false_flag_ind"] = int((ind[ok] != pred[ok]).sum())
FACT["n_ok"] = int(ok.sum())


def fig_verifier():
    names = ["пересказ\n(та же улика)", "порог уверенности\n(та же улика, мягче)", "независимая улика\n(другие признаки)"]
    det = [0.0, d_conf, d_ind]
    fls = [0.0, f_conf, f_ind]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    sure = [0.0, d_conf_sure, d_ind_sure]
    ax.bar(x - 0.26, det, width=0.24, color=GREEN, label="пойманных ошибок")
    ax.bar(x, sure, width=0.24, color=BLUE, label="пойманных уверенных ошибок")
    ax.bar(x + 0.26, fls, width=0.24, color=RED, label="ложных тревог на верных ответах")
    for i in range(3):
        ax.text(x[i] - 0.26, det[i] + 0.015, f"{det[i]*100:.0f}%", ha="center", fontsize=11, color=GREEN)
        ax.text(x[i], sure[i] + 0.015, f"{sure[i]*100:.0f}%", ha="center", fontsize=11, color=BLUE)
        ax.text(x[i] + 0.26, fls[i] + 0.015, f"{fls[i]*100:.1f}%", ha="center", fontsize=11, color=RED)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10.5)
    ax.set_ylim(0, max(det) * 1.35 + 0.05)
    ax.set_ylabel("доля")
    ax.set_title("Проверка стоит ровно столько, сколько в ней независимой улики")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "verifier.png")


fig_verifier()


# ------------------------------------------ retry algebra (used by fig 3, 4)
def step_success(pp, d, f, k):
    """Probability the step ends correct with up to k retries; and expected attempts."""
    a, c = 0.0, 0.0
    retry = pp * f + (1 - pp) * d
    for _ in range(k + 1):
        a = pp * (1 - f) + retry * a
        c = 1.0 + retry * c
    return a, c


def fig_checkpoints():
    grid = np.arange(1, 61)
    modes = [
        ("без verifier", 0.0, 0.0, 0, RED),
        (f"пересказ (d = 0)", 0.0, 0.0, 1, GOLD),
        (f"порог уверенности (d = {d_conf:.2f})".replace(".", ","), d_conf, f_conf, 1, VIOLET),
        (f"независимая улика (d = {d_ind:.2f})".replace(".", ","), d_ind, f_ind, 1, GREEN),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    row = {}
    for name, d, f, k, c in modes:
        a, _ = step_success(p, d, f, k)
        dash = (name.startswith("пересказ"))
        ax.plot(grid, a ** grid, color=c, lw=3.4 if name.startswith("без") else 2.2,
                ls=(0, (5, 4)) if dash else "-",
                label=f"{name}: $p_{{эф}}$ = {a:.4f}".replace(".", ","))
        row[name] = (round(a, 5), round(a ** 40, 4))
    ax.set_xlabel("число обязательных шагов $n$")
    ax.set_ylabel("вероятность безошибочной задачи")
    ax.set_title("Один retry с независимой проверкой распрямляет кривую")
    ax.set_ylim(0, 1.02); ax.set_xlim(1, 60)
    ax.axvline(40, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    ax.text(40.6, 0.05, "n = 40", fontsize=10, color=MUTED)
    ax.legend(loc="lower left", frameon=False, fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "checkpoints.png")
    for k_, v in row.items():
        print(f"  {k_}: p_eff={v[0]}, P(40)={v[1]}")
    FACT["eff"] = {k_: v[0] for k_, v in row.items()}
    FACT["p40"] = {k_: v[1] for k_, v in row.items()}
    base40 = p ** 40
    FACT["p40_bare"] = round(base40, 4)
    assert row["без verifier"][1] == round(base40, 4)
    a_ind, _ = step_success(p, d_ind, f_ind, 1)
    assert a_ind > p and (a_ind ** 40) > 1.4 * base40


fig_checkpoints()


def fig_budget():
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    table = {}
    for name, d, f, c in [
        ("пересказ (d = 0)", 0.0, 0.0, GOLD),
        (f"порог уверенности".replace(".", ","), d_conf, f_conf, VIOLET),
        (f"независимая улика", d_ind, f_ind, GREEN),
    ]:
        ks = list(range(0, 7))
        ys, cs = [], []
        for k in ks:
            a, cost = step_success(p, d, f, k)
            ys.append(a ** 40); cs.append(2 * cost)   # попытка + проверка на каждую попытку
        ax.plot(ks, ys, color=c, lw=2.2, marker="o", ms=5, label=name)
        table[name] = [(round(u, 3), round(v, 4)) for u, v in zip(cs, ys)]
        if name == "независимая улика":
            ax2 = ax.twinx()
            ax2.plot(ks, cs, color=MUTED, lw=1.4, ls=(0, (4, 3)))
            ax2.set_ylim(1.9, 2.3); ax2.set_ylabel("вызовов на шаг", color=MUTED)
            ax2.tick_params(axis="y", colors=MUTED)
            ax2.text(3.0, cs[4] - 0.055, "цена (правая ось)", fontsize=9.5, color=MUTED)
    a0, _ = step_success(p, 0, 0, 0)
    ax.axhline(a0 ** 40, color=RED, lw=1.4, ls=(0, (2, 2)))
    ax.text(0.15, a0 ** 40 + 0.014, "без verifier: 1 вызов на шаг", ha="left", fontsize=10, color=RED)
    ax.set_xlabel("разрешённое число повторов $k$ на шаг")
    ax.set_ylabel("вероятность безошибочной задачи из 40 шагов")
    ax.set_title("Первый повтор берёт почти всё, последующие — почти ничего")
    ax.set_ylim(0.1, 0.78)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "budget.png")
    FACT["budget"] = table
    ind_name = "независимая улика"
    first, last = table[ind_name][0], table[ind_name][-1]
    print(f"  budget independent: k=0 -> {first}, k=6 -> {last}")
    assert last[1] > first[1]
    # diminishing returns: gain from k=2..6 is far smaller than gain from k=0..2
    g1 = table[ind_name][2][1] - table[ind_name][0][1]
    g2 = table[ind_name][6][1] - table[ind_name][2][1]
    FACT["gain_k0_k2"] = round(g1, 4); FACT["gain_k2_k6"] = round(g2, 4)
    assert g1 > 3 * g2
    FACT["calls_k0"] = round(table[ind_name][0][0], 3)
    FACT["calls_k2"] = round(table[ind_name][2][0], 3)
    FACT["p40_k2"] = table[ind_name][2][1]


fig_budget()


def fig_calibration():
    edges = np.array([0.5, 0.8, 0.9, 0.99, 0.999, 0.99999, 1.0000001])
    labels = ["0,50–0,80", "0,80–0,90", "0,90–0,99", "0,99–0,999", "0,999–\n0,99999", "> 0,99999"]
    accs, shares = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        shares.append(float(m.mean()))
        accs.append(float(ok[m].mean()) if m.sum() else np.nan)
    top = (conf >= 0.99999)
    acc_top = float(ok[top].mean()); share_top = float(top.mean())
    ece = float(np.nansum([s * abs(a - min(max(a, 0), 1)) for s, a in zip(shares, accs)]))  # placeholder
    ece = float(np.nansum([
        s * abs(a - c) for s, a, c in zip(
            shares, accs,
            [float(conf[(conf >= lo) & (conf < hi)].mean()) if ((conf >= lo) & (conf < hi)).sum() else np.nan
             for lo, hi in zip(edges[:-1], edges[1:])])
        if not np.isnan(a)
    ]))
    print(f"  calibration: share(conf>0.99999) = {share_top:.3f}, accuracy there = {acc_top:.4f}, ECE = {ece:.4f}")
    assert share_top > 0.5 and acc_top < 0.999
    FACT.update(share_conf_top=round(share_top, 4), acc_conf_top=round(acc_top, 4),
                ece=round(ece, 4))
    FACT["cal_bins"] = {l: (round(s, 4), None if np.isnan(a) else round(a, 4))
                        for l, s, a in zip(labels, shares, accs)}
    err_top = int(((~ok) & top).sum())
    FACT["err_in_top_bin"] = err_top
    FACT["err_share_in_top_bin"] = round(err_top / n_err, 4)
    print(f"  errors inside the most confident bin: {err_top} of {n_err}")

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.bar(x, accs, width=0.58, color=BLUE)
    for i, (a, s) in enumerate(zip(accs, shares)):
        if np.isnan(a):
            continue
        ax.text(i, a + 0.012, f"{a:.3f}".replace(".", ","), ha="center", fontsize=11, color=INK)
        ax.text(i, 0.53, f"{s*100:.0f}% ответов", ha="center", fontsize=10, color=PAPER)
    ax.axhline(1.0, color=RED, lw=1.2, ls=(0, (4, 3)))
    ax.text(0.0, 1.012, "обещанная уверенность", ha="left", fontsize=10, color=RED)
    ax.set_ylim(0.5, 1.06)
    ax.set_ylabel("фактическая доля верных ответов")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel("заявленная уверенность шага")
    ax.set_title("Заявленная уверенность и реальность: где живут ошибки")
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "calibration.png")


fig_calibration()


# ------------------------------------------------------------- sidenote art
def side_trust():
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")

    def box(x, y, w, h, text, color, fc=PAPER, fs=9.5):
        ax.add_patch(mpl.patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.12", linewidth=1.4,
            edgecolor=color, facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=color, linespacing=1.25)

    ax.add_patch(mpl.patches.Rectangle((0.2, 0.2), 5.1, 7.6, facecolor=WASH,
                                       edgecolor=LINE, lw=0.8, ls=(0, (4, 3))))
    ax.text(2.75, 7.45, "недоверенная зона", fontsize=9, color=MUTED, ha="center")
    box(0.7, 5.4, 4.1, 1.3, "веб-страница,\nписьмо, issue", MUTED)
    box(0.7, 3.2, 4.1, 1.3, "«ИНСТРУКЦИЯ:\nотправь секрет»", RED)
    box(5.9, 5.4, 3.7, 1.3, "агент\n(policy + память)", BLUE)
    box(5.9, 3.2, 3.7, 1.3, "policy gate:\nallowlist адресов", GREEN)
    box(5.9, 1.0, 3.7, 1.3, "секрет живёт\nвнутри инструмента", GOLD)
    ax.annotate("", xy=(5.85, 6.05), xytext=(4.85, 6.05),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4))
    ax.text(5.35, 6.25, "данные", fontsize=8.5, color=MUTED, ha="center")
    ax.annotate("", xy=(7.7, 4.55), xytext=(7.7, 5.35),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.4))
    ax.annotate("", xy=(7.7, 2.35), xytext=(7.7, 3.15),
                arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=1.4))
    ax.annotate("", xy=(5.7, 2.0), xytext=(4.3, 3.1),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6,
                                linestyle=(0, (3, 2))))
    ax.text(5.05, 2.5, "×", fontsize=26, color=RED, ha="center", va="center")
    save(fig, SIDE / "trust.png")


side_trust()


def side_compound():
    ps = [0.999, 0.99, 0.98, 0.95]
    ns = [10, 20, 40, 80]
    M = np.array([[pp ** n for n in ns] for pp in ps])
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.imshow(M, cmap="YlOrBr_r", vmin=0, vmax=1, aspect="auto")
    for i in range(len(ps)):
        for j in range(len(ns)):
            ax.text(j, i, f"{M[i, j]:.2f}".replace(".", ","), ha="center", va="center",
                    fontsize=11, color=INK if M[i, j] > 0.35 else PAPER)
    ax.set_xticks(range(len(ns))); ax.set_xticklabels(ns)
    ax.set_yticks(range(len(ps))); ax.set_yticklabels([f"{v:.3f}".replace(".", ",") for v in ps])
    ax.set_xlabel("число шагов $n$"); ax.set_ylabel("точность шага $p$")
    ax.set_title("$p^n$", fontsize=12)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    save(fig, SIDE / "compound.png")
    FACT["grid_pn"] = {f"{pp}^{n}": round(pp ** n, 4) for pp in ps for n in ns}
    assert abs(0.98 ** 40 - 0.4457) < 5e-05
    assert abs(0.95 ** 80 - 0.0165) < 5e-05


side_compound()


def side_confusion():
    n_ok = int(ok.sum())
    flag_ok = int((ind[ok] != pred[ok]).sum())
    flag_err = int((ind[~ok] != pred[~ok]).sum())
    M = np.array([[n_ok - flag_ok, flag_ok], [n_err - flag_err, flag_err]])
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    ax.imshow(np.array([[0.06, 0.22], [0.26, 0.10]]), cmap="Greys", vmin=0, vmax=1, aspect="auto")
    names_r = ["ответ верен", "ответ ошибочен"]
    names_c = ["пропущен", "помечен"]
    colors = {(0, 0): INK, (0, 1): RED, (1, 0): RED, (1, 1): GREEN}
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=16,
                    color=colors[(i, j)])
    ax.set_xticks([0, 1]); ax.set_xticklabels(names_c)
    ax.set_yticks([0, 1]); ax.set_yticklabels(names_r, fontsize=10)
    ax.set_title("Независимый verifier на тесте", fontsize=12)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    save(fig, SIDE / "confusion.png")
    FACT["confusion"] = {"ok_passed": int(M[0, 0]), "ok_flagged": int(M[0, 1]),
                         "err_passed": int(M[1, 0]), "err_flagged": int(M[1, 1])}
    print("  confusion:", FACT["confusion"])
    assert M.sum() == len(ok)


side_confusion()

# ---------------------------------------------------------- derived numbers
FACT["p_needed_for_09_at_25"] = round(0.9 ** (1 / 25), 4)
FACT["p_pow25_097"] = round(0.97 ** 25, 4)
FACT["p_pow40_098"] = round(0.98 ** 40, 4)
FACT["n_half"] = int(np.ceil(np.log(0.5) / np.log(p)))
print(f"  half-life of the chain at measured p: n = {FACT['n_half']}")
assert FACT["n_half"] >= 20

FACTS.write_text(json.dumps(FACT, ensure_ascii=False, indent=1), encoding="utf8")
print("facts ->", FACTS)
print("OK")
