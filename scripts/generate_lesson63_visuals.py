"""Deterministic figures for lesson 63: oracle inequalities and the price of model selection.

Every number quoted in the prose of content/lessons/63.md is computed here and asserted.

Real data:
  * scripts/data/sms-spam-collection.tsv  -- real hyperparameter search over 120 candidates
  * scripts/data/bike-sharing-hour.csv    -- structural risk minimisation over polynomial degree

Synthetic parts (declared as such in the text) always use a fixed seed:
  * coin-flip candidates evaluated on REAL sms labels (winner's curse)
  * pure-noise predictors for Freedman's paradox
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

ROOT = Path(__file__).resolve().parents[1]
SMS = ROOT / "scripts" / "data" / "sms-spam-collection.tsv"
BIKE = ROOT / "scripts" / "data" / "bike-sharing-hour.csv"
OUT = ROOT / "public" / "figures" / "lessons" / "63"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "63"

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


def eps(M: float, n: float, delta: float = 0.05) -> float:
    """Hoeffding + union bound simultaneous deviation for M candidates."""
    return math.sqrt(math.log(2 * M / delta) / (2 * n))


def load_sms():
    labels, texts = [], []
    with open(SMS, encoding="utf8", errors="replace") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 2:
                continue
            labels.append(1 if row[0].strip() == "spam" else 0)
            texts.append(row[1])
    return np.array(labels), texts


def bike_hourly():
    hour, cnt = [], []
    with open(BIKE) as f:
        for row in csv.DictReader(f):
            hour.append(int(row["hr"])); cnt.append(int(row["cnt"]))
    hour, cnt = np.array(hour), np.array(cnt)
    return hour, cnt


# ---------------------------------------------------------------- fig 63.1
def fig_winners_curse(labels) -> None:
    """M coin-flip candidates (true accuracy 1/2) scored on REAL sms labels."""
    rng = np.random.default_rng(63)
    n = 200
    idx = rng.choice(len(labels), size=n, replace=False)
    y = labels[idx]
    FACTS["wc_n"] = n
    FACTS["wc_spam_share"] = float(y.mean())

    # one draw of M = 1000 candidates
    M = 1000
    preds = rng.integers(0, 2, size=(M, n))
    acc = (preds == y).mean(axis=1)
    FACTS["wc_M"] = M
    FACTS["wc_mean_acc"] = float(acc.mean())
    FACTS["wc_max_acc"] = float(acc.max())
    FACTS["wc_min_acc"] = float(acc.min())

    # expected maximum vs M, 2000 repeats
    Ms = [1, 10, 100, 1000]
    reps = 2000
    means = []
    for m in Ms:
        p = rng.integers(0, 2, size=(reps, m, n))
        a = (p == y).mean(axis=2)
        means.append(float(a.max(axis=1).mean()))
    for m, v in zip(Ms, means):
        FACTS[f"wc_expmax_{m}"] = v
        FACTS[f"wc_bound_{m}"] = 0.5 + math.sqrt(math.log(max(m, 1.0001)) / (2 * n))

    assert 0.49 < FACTS["wc_mean_acc"] < 0.51, FACTS["wc_mean_acc"]
    assert means[0] < means[1] < means[2] < means[3]
    assert FACTS["wc_expmax_1000"] > 0.58

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.2, 4.4))
    ax0.hist(acc, bins=22, color=BLUE, alpha=0.75, edgecolor=PAPER)
    ax0.axvline(0.5, color=INK, lw=1.8, ls=(0, (5, 3)))
    ax0.axvline(acc.max(), color=RED, lw=2.2)
    ax0.text(0.5, ax0.get_ylim()[1] * 0.95, " истинная точность 0,50",
             color=INK, fontsize=10, ha="left", va="top")
    ax0.text(acc.max(), ax0.get_ylim()[1] * 0.62,
             f" победитель {acc.max():.3f} ", color=RED, fontsize=10, ha="right", va="top")
    ax0.set_xlabel("точность кандидата на validation ($n=200$)")
    ax0.set_ylabel("число кандидатов")
    ax0.set_title("Тысяча монеток на реальных метках", fontsize=12.5)

    ax1.plot(Ms, means, "o-", color=RED, lw=2.2, label="средний максимум (2000 повторов)")
    ax1.plot(Ms, [FACTS[f"wc_bound_{m}"] for m in Ms], "s--", color=GOLD, lw=1.8,
             label=r"$0{,}5+\sqrt{\ln M/(2n)}$")
    ax1.axhline(0.5, color=INK, lw=1.6, ls=(0, (5, 3)), label="истинная точность каждого")
    ax1.set_xscale("log")
    ax1.set_xlabel("число просмотренных кандидатов $M$")
    ax1.set_ylabel("точность лучшего")
    ax1.set_title("Максимум растёт, качество — нет", fontsize=12.5)
    ax1.legend(loc="upper left", frameon=False, fontsize=9.5)
    for ax in (ax0, ax1):
        ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    fig.suptitle("Проклятие победителя: минимум шумных оценок смещён", y=1.02, fontsize=14)
    fig.tight_layout()
    save(fig, OUT / "winners_curse.png")
    print("fig1 winners_curse:", {k: round(v, 4) for k, v in FACTS.items() if k.startswith("wc_")})


# ---------------------------------------------------------------- fig 63.2
def fig_penalty() -> None:
    Ms = np.logspace(0, 6, 400)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for n, c in [(500, RED), (2000, GOLD), (10000, GREEN)]:
        ax.plot(Ms, [eps(m, n) for m in Ms], color=c, lw=2.2, label=f"$n={n}$")
    ax.set_xscale("log")
    ax.set_xlabel("число заранее перечисленных кандидатов $M$")
    ax.set_ylabel(r"$\varepsilon=\sqrt{\ln(2M/\delta)/(2n)}$")
    ax.set_title(r"Цена каталога растёт как $\sqrt{\ln M}$, а не как $M$")
    ax.legend(loc="upper left", frameon=False, fontsize=10.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)

    for m in (1, 100, 10000):
        v = eps(m, 2000)
        ax.plot([m], [v], "o", color=INK, ms=6, zorder=6)
        ax.annotate(f"{v:.3f}".replace(".", ","), (m, v), textcoords="offset points",
                    xytext=(6, 8), fontsize=9.5, color=INK)
    save(fig, OUT / "penalty_curve.png")

    for tag, (m, n) in {
        "e_1_1000": (1, 1000), "e_100_1000": (100, 1000), "e_10000_1000": (10000, 1000),
        "e_1_2000": (1, 2000), "e_100_2000": (100, 2000), "e_10000_2000": (10000, 2000),
        "e_100_8000": (100, 8000), "e_1e6_2000": (10 ** 6, 2000),
        "e_10_2000": (10, 2000), "e_1000_2000": (1000, 2000),
        "e_3_800": (3, 800), "e_1000_800": (1000, 800),
    }.items():
        FACTS[tag] = eps(m, n)
    FACTS["e_ratio_1e6"] = FACTS["e_1e6_2000"] / FACTS["e_1_2000"]
    # heuristic optimism of the maximum, and how many looks a hold-out survives
    FACTS["opt_100_2000"] = math.sqrt(math.log(100) / (2 * 2000))
    FACTS["looks_2000_003"] = math.exp(2 * 2000 * 0.03 ** 2)
    assert abs(FACTS["e_1_1000"] - 0.043) < 0.0005
    assert abs(FACTS["e_100_1000"] - 0.064) < 0.0005
    assert abs(FACTS["e_10000_1000"] - 0.080) < 0.0005
    print("fig2 penalty:", {k: round(v, 4) for k, v in FACTS.items() if k.startswith("e_")})


# ---------------------------------------------------------------- fig 63.3
def fig_sms_selection(labels, texts) -> None:
    """Real grid search on real SMS spam: winner's validation score is optimistic."""
    rng = np.random.default_rng(6300)
    n_all = len(labels)
    perm = rng.permutation(n_all)
    n_tr, n_val = 3000, 500
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_val], perm[n_tr + n_val:]
    ytr, yva, yte = labels[tr], labels[va], labels[te]
    Ttr = [texts[i] for i in tr]; Tva = [texts[i] for i in va]; Tte = [texts[i] for i in te]

    grid = []
    for ngram in [(1, 1), (1, 2)]:
        for min_df in [1, 2, 5]:
            for binary in [False, True]:
                for alpha in [0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
                    grid.append((ngram, min_df, binary, alpha))
    val_err, test_err = [], []
    for ngram, min_df, binary, alpha in grid:
        vec = CountVectorizer(ngram_range=ngram, min_df=min_df, binary=binary)
        Xtr = vec.fit_transform(Ttr)
        clf = MultinomialNB(alpha=alpha).fit(Xtr, ytr)
        val_err.append(1 - clf.score(vec.transform(Tva), yva))
        test_err.append(1 - clf.score(vec.transform(Tte), yte))
    val_err = np.array(val_err); test_err = np.array(test_err)
    M = len(grid)
    win = int(np.argmin(val_err))
    best_test = int(np.argmin(test_err))

    FACTS["sms_M"] = M
    FACTS["sms_ntr"] = n_tr
    FACTS["sms_nval"] = n_val
    FACTS["sms_ntest"] = len(te)
    FACTS["sms_win_val"] = float(val_err[win])
    FACTS["sms_win_test"] = float(test_err[win])
    FACTS["sms_gap_win"] = float(test_err[win] - val_err[win])
    FACTS["sms_gap_mean"] = float(np.mean(test_err - val_err))
    gaps_others = np.delete(test_err - val_err, win)
    FACTS["sms_gap_mean_others"] = float(np.mean(gaps_others))
    assert len(gaps_others) == M - 1
    FACTS["sms_oracle_test"] = float(test_err[best_test])
    FACTS["sms_regret"] = float(test_err[win] - test_err[best_test])
    FACTS["sms_eps"] = eps(M, n_val)
    FACTS["sms_val_spread"] = float(val_err.max() - val_err.min())
    FACTS["sms_val_se"] = float(math.sqrt(val_err[win] * (1 - val_err[win]) / n_val))
    FACTS["sms_se_test"] = float(math.sqrt(test_err[win] * (1 - test_err[win]) / len(te)))
    FACTS["sms_1se_hi"] = float(val_err.min() + FACTS["sms_val_se"])
    FACTS["sms_1se_share"] = float(np.mean(val_err <= FACTS["sms_1se_hi"]))

    assert M == 120, M
    assert FACTS["sms_win_val"] < FACTS["sms_win_test"], "winner should look better on val"
    assert FACTS["sms_regret"] >= 0
    assert FACTS["sms_regret"] < FACTS["sms_eps"], "regret must respect the oracle bound"

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.scatter(val_err * 100, test_err * 100, s=34, color=BLUE, alpha=0.65,
               edgecolor=PAPER, linewidth=0.5, label=f"кандидаты ($M={M}$)")
    lo = min(val_err.min(), test_err.min()) * 100 - 0.3
    hi = max(val_err.max(), test_err.max()) * 100 + 0.3
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.2, ls=(0, (4, 3)), label="val = test")
    ax.scatter([val_err[win] * 100], [test_err[win] * 100], s=150, color=RED, zorder=6,
               label="победитель по validation")
    ax.scatter([val_err[best_test] * 100], [test_err[best_test] * 100], s=150, color=GREEN,
               marker="D", zorder=6, label="оракул (лучший по test)")
    ax.annotate(
        f"val {val_err[win]*100:.2f}%, test {test_err[win]*100:.2f}%".replace(".", ","),
        (val_err[win] * 100, test_err[win] * 100), textcoords="offset points",
        xytext=(12, -14), fontsize=10, color=RED)
    ax.set_xlabel("ошибка на validation, % ($n=500$)")
    ax.set_ylabel("ошибка на честном test, %")
    ax.set_title("Реальный перебор 120 конфигураций на SMS-спаме")
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "sms_selection.png")
    print("fig3 sms:", {k: round(v, 4) for k, v in FACTS.items() if k.startswith("sms_")})


# ---------------------------------------------------------------- fig 63.4
def fig_srm(hour, cnt) -> None:
    """Structural risk minimisation over polynomial degree on real bike data.

    Train: 30 real hourly observations. Target of the test error: the TRUE hourly
    mean curve, computed from all 17379 rows of the dataset.
    """
    rng = np.random.default_rng(631)
    tr = rng.permutation(len(cnt))[:30]
    n = len(tr)
    x, y = (hour[tr] - 11.5) / 11.5, cnt[tr].astype(float)
    mu = np.array([cnt[hour == h].mean() for h in range(24)])
    xh = (np.arange(24) - 11.5) / 11.5

    degs = list(range(1, 13))
    tr_err, te_err, bic = [], [], []
    for d in degs:
        co = np.polyfit(x, y, d)
        rss = float(np.sum((y - np.polyval(co, x)) ** 2))
        tr_err.append(math.sqrt(rss / n) / y.std())
        te_err.append(float(np.sqrt(np.mean((mu - np.polyval(co, xh)) ** 2)) / mu.std()))
        bic.append(n * math.log(rss / n) + (d + 2) * math.log(n))
    tr_err = np.array(tr_err); te_err = np.array(te_err); bic = np.array(bic)
    d_srm = degs[int(np.argmin(bic))]
    d_test = degs[int(np.argmin(te_err))]

    FACTS["srm_ntrain"] = n
    FACTS["srm_d_srm"] = d_srm
    FACTS["srm_d_test"] = d_test
    FACTS["srm_tr_1"] = float(tr_err[0])
    FACTS["srm_tr_12"] = float(tr_err[-1])
    FACTS["srm_te_1"] = float(te_err[0])
    FACTS["srm_te_best"] = float(te_err.min())
    FACTS["srm_te_11"] = float(te_err[10])
    FACTS["srm_te_at_srm"] = float(te_err[d_srm - 1])
    assert tr_err[-1] < tr_err[0], "train error must fall with degree"
    assert te_err[10] > te_err.min() * 1.8, "test error must blow up at high degree"
    assert d_srm == 3 and d_test == 8, (d_srm, d_test)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.2, 4.6))
    ax0.plot(degs, tr_err, "o-", color=BLUE, lw=2.0, label="ошибка на 30 обучающих часах")
    ax0.plot(degs, te_err, "s-", color=RED, lw=2.2, label="отклонение от истинной кривой спроса")
    ax0.axvline(d_test, color=RED, lw=1.0, ls=(0, (2, 2)))
    ax0.axvline(d_srm, color=GREEN, lw=1.0, ls=(0, (2, 2)))
    ax0.set_xlabel("степень полинома $k$ — номер вложенного класса")
    ax0.set_ylabel("нормированная ошибка")
    ax0.set_title("Обучение падает, истина — нет", fontsize=12.5)
    ax0.set_xticks(degs); ax0.legend(loc="upper center", frameon=False, fontsize=9)
    ax1.plot(degs, bic, "d-", color=GREEN, lw=2.2)
    ax1.plot([d_srm], [bic.min()], "o", color=GREEN, ms=11, zorder=6)
    ax1.axvline(d_srm, color=GREEN, lw=1.0, ls=(0, (2, 2)))
    ax1.set_xlabel("степень полинома $k$")
    ax1.set_ylabel("$n\\ln(\\widehat R)+(k+2)\\ln n$")
    ax1.set_title(f"Штраф за сложность выбирает $k={d_srm}$", fontsize=12.5)
    ax1.set_xticks(degs)
    for ax in (ax0, ax1):
        ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    fig.suptitle("Структурная минимизация риска: эмпирический риск плюс плата за богатство класса",
                 y=1.02, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "srm_bike.png")
    print("fig4 srm:", {k: round(float(v), 4) for k, v in FACTS.items() if k.startswith("srm_")})


# ---------------------------------------------------------------- fig 63.5
def fig_freedman(hour, cnt) -> None:
    """Freedman's paradox: selecting among pure-noise predictors on a real target."""
    rng = np.random.default_rng(1983)
    idx = rng.choice(len(cnt), size=200, replace=False)
    y = np.log1p(cnt[idx].astype(float))
    y = (y - y.mean()) / y.std()
    ysel, ynew = y[:100], y[100:]
    P = 500
    X = rng.normal(size=(200, P))          # pure noise, fixed seed, declared in text
    Xsel, Xnew = X[:100], X[100:]

    corr = np.array([abs(np.corrcoef(Xsel[:, j], ysel)[0, 1]) for j in range(P)])
    order = np.argsort(-corr)
    ks = [1, 2, 5, 10, 20, 30, 40]
    r2_sel, r2_new = [], []
    for k in ks:
        cols = order[:k]
        A = np.column_stack([np.ones(100), Xsel[:, cols]])
        w, *_ = np.linalg.lstsq(A, ysel, rcond=None)
        r2_sel.append(1 - np.sum((ysel - A @ w) ** 2) / np.sum((ysel - ysel.mean()) ** 2))
        B = np.column_stack([np.ones(100), Xnew[:, cols]])
        r2_new.append(1 - np.sum((ynew - B @ w) ** 2) / np.sum((ynew - ynew.mean()) ** 2))
    r2_sel = np.array(r2_sel); r2_new = np.array(r2_new)

    FACTS["fp_P"] = P
    FACTS["fp_maxcorr_theory"] = math.sqrt(2 * math.log(P)) / math.sqrt(100)
    FACTS["fp_n"] = 100
    FACTS["fp_maxcorr"] = float(corr.max())
    FACTS["fp_k20_sel"] = float(r2_sel[ks.index(20)])
    FACTS["fp_k20_new"] = float(r2_new[ks.index(20)])
    FACTS["fp_k40_sel"] = float(r2_sel[ks.index(40)])
    FACTS["fp_k40_new"] = float(r2_new[ks.index(40)])
    assert FACTS["fp_k20_sel"] > 0.3
    assert FACTS["fp_k20_new"] < 0.0
    assert FACTS["fp_maxcorr"] > 0.25

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(ks, r2_sel, "o-", color=RED, lw=2.2, label="$R^2$ на той же выборке, где отбирали")
    ax.plot(ks, r2_new, "s-", color=BLUE, lw=2.2, label="$R^2$ на свежих 100 наблюдениях")
    ax.axhline(0, color=INK, lw=1.2, ls=(0, (5, 3)))
    ax.fill_between(ks, r2_new, 0, where=(r2_new < 0), color=RED, alpha=0.07)
    ax.set_xlabel("сколько шумовых признаков отобрано из 500")
    ax.set_ylabel("$R^2$")
    ax.set_title("Парадокс Фридмана: отбор создаёт качество из ничего")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "freedman.png")
    print("fig5 freedman:", {k: round(float(v), 4) for k, v in FACTS.items() if k.startswith("fp_")})


# ---------------------------------------------------------------- fig 63.6
def fig_nested_cv() -> None:
    K_out, K_in = 5, 4
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, K_out + 1.6); ax.axis("off")
    for i in range(K_out):
        y = K_out - i
        for j in range(K_out):
            x0 = 0.4 + j * 1.05
            is_test = (j == i)
            ax.add_patch(plt.Rectangle((x0, y - 0.34), 1.0, 0.68,
                                       facecolor=RED if is_test else WASH,
                                       edgecolor=LINE, lw=0.8, alpha=0.9 if is_test else 1))
        ax.annotate("", xy=(6.15, y), xytext=(5.75, y),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
        for j in range(K_in):
            x0 = 6.3 + j * 0.78
            ax.add_patch(plt.Rectangle((x0, y - 0.28), 0.72, 0.56,
                                       facecolor=GOLD if j == (i % K_in) else PAPER,
                                       edgecolor=BLUE, lw=0.9, alpha=0.85 if j == (i % K_in) else 1))
        ax.text(9.65, y, f"внешний\nfold {i+1}", fontsize=8.5, color=MUTED, va="center")
    ax.text(2.9, K_out + 0.95, "внешнее разбиение: красный блок оценивает ПРОЦЕДУРУ",
            ha="center", fontsize=11, color=INK)
    ax.text(7.7, K_out + 0.95, "внутренний CV:\nвыбор гиперпараметра",
            ha="center", fontsize=10.5, color=BLUE)
    ax.text(2.9, 0.25, "модель обучается здесь", ha="center", fontsize=9.5, color=MUTED)
    ax.text(7.7, 0.25, f"{K_out}×{K_in} = {K_out*K_in} внутренних обучений на один гиперпараметр",
            ha="center", fontsize=9.5, color=MUTED)
    fig.suptitle("Вложенная кросс-валидация: внешний fold не участвовал в выборе", y=0.98, fontsize=13.5)
    save(fig, OUT / "nested_cv.png")
    FACTS["ncv_inner_per_outer"] = K_in
    FACTS["ncv_outer"] = K_out
    print("fig6 nested_cv drawn")


# ---------------------------------------------------------------- sidenote images
def side_max_vs_mean() -> None:
    rng = np.random.default_rng(7)
    Ms = np.array([1, 3, 10, 30, 100, 300, 1000])
    reps = 4000
    mx, mn = [], []
    for m in Ms:
        z = rng.normal(size=(reps, m))
        mx.append(z.max(axis=1).mean()); mn.append(z.mean(axis=1).mean())
    FACTS["side_max_1000"] = float(mx[-1])
    assert 3.0 < FACTS["side_max_1000"] < 3.4
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    ax.plot(Ms, mx, "o-", color=RED, lw=1.8, label="максимум")
    ax.plot(Ms, mn, "s-", color=BLUE, lw=1.6, label="среднее")
    ax.set_xscale("log"); ax.set_xlabel("$M$ независимых оценок", fontsize=9)
    ax.set_ylabel("в единицах $\\sigma$", fontsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.set_title("среднее стоит, максимум ползёт", fontsize=9)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "max_vs_mean.png")


def side_sqrt_log() -> None:
    Ms = [1, 10, 1000, 10 ** 6]
    vals = [eps(m, 2000) for m in Ms]
    FACTS["side_eps_1"] = vals[0]; FACTS["side_eps_1e6"] = vals[-1]
    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    ax.bar([str(m) for m in Ms], vals, color=[BLUE, BLUE, GOLD, RED], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.002, f"{v:.3f}".replace(".", ","), ha="center", fontsize=8, color=INK)
    ax.set_ylim(0, max(vals) * 1.22)
    ax.set_xlabel("$M$ кандидатов", fontsize=9)
    ax.set_ylabel(r"$\varepsilon$ при $n=2000$", fontsize=9)
    ax.set_title("миллион дороже одного лишь вдвое", fontsize=9)
    save(fig, SIDE / "sqrt_log.png")


def side_forking() -> None:
    choices = [("очистка", 3), ("признаки", 4), ("модель", 5), ("метрика", 3)]
    total = 1
    for _, k in choices:
        total *= k
    FACTS["fork_total"] = total
    assert total == 180
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ys = [8.6, 6.6, 4.6, 2.6]
    for i, ((name, k), y) in enumerate(zip(choices, ys)):
        ax.text(0.2, y + 0.75, f"{name}: {k}", fontsize=9, color=MUTED)
        for j in range(k):
            x = 1.2 + j * (7.6 / max(k - 1, 1))
            ax.plot([x], [y], "o", color=[BLUE, GREEN, GOLD, RED][i], ms=7)
            if i < 3:
                for j2 in range(choices[i + 1][1]):
                    x2 = 1.2 + j2 * (7.6 / max(choices[i + 1][1] - 1, 1))
                    ax.plot([x, x2], [y, ys[i + 1]], color=LINE, lw=0.35, zorder=0)
    ax.text(5, 0.8, f"{total} путей анализа", ha="center", fontsize=10.5, color=INK)
    ax.set_title("сад расходящихся тропок", fontsize=9.5)
    save(fig, SIDE / "forking.png")


# ---------------------------------------------------------------- run
def main() -> None:
    labels, texts = load_sms()
    hour, cnt = bike_hourly()
    FACTS["sms_rows"] = len(labels)
    FACTS["bike_rows"] = len(cnt)
    assert len(labels) == 5572, len(labels)

    fig_winners_curse(labels)
    fig_penalty()
    fig_sms_selection(labels, texts)
    fig_srm(hour, cnt)
    fig_freedman(hour, cnt)
    fig_nested_cv()
    side_max_vs_mean()
    side_sqrt_log()
    side_forking()

    # every number quoted verbatim in content/lessons/63.md -- final guard
    checks = {
        # simultaneous Hoeffding bound
        "e_1_1000": (0.043, 3), "e_100_1000": (0.064, 3), "e_10000_1000": (0.080, 3),
        "e_1_2000": (0.030, 3), "e_100_2000": (0.046, 3), "e_10000_2000": (0.057, 3),
        "e_100_8000": (0.023, 3), "e_1e6_2000": (0.066, 3), "e_ratio_1e6": (2.18, 2),
        # winner's curse on real sms labels
        "wc_spam_share": (0.12, 2), "wc_mean_acc": (0.500, 3), "wc_max_acc": (0.610, 3),
        "wc_min_acc": (0.400, 3), "wc_expmax_10": (0.554, 3), "wc_expmax_100": (0.588, 3),
        "wc_expmax_1000": (0.614, 3), "wc_bound_10": (0.576, 3), "wc_bound_100": (0.607, 3),
        "wc_bound_1000": (0.631, 3),
        # real grid search on sms spam
        "sms_win_val": (0.0160, 4), "sms_win_test": (0.0183, 4), "sms_gap_win": (0.0023, 4),
        "sms_gap_mean": (-0.0023, 4), "sms_gap_mean_others": (-0.0023, 4), "sms_oracle_test": (0.0140, 4), "sms_regret": (0.0043, 4),
        "sms_eps": (0.092, 3), "sms_val_spread": (0.0200, 4), "sms_val_se": (0.0056, 4),
        # structural risk minimisation on bike sharing
        "srm_tr_1": (0.89, 2), "srm_tr_12": (0.60, 2), "srm_te_1": (0.83, 2),
        "srm_te_best": (0.47, 2), "srm_te_11": (1.10, 2), "srm_te_at_srm": (0.58, 2),
        # freedman's paradox
        "fp_maxcorr": (0.31, 2), "fp_k20_sel": (0.57, 2), "fp_k20_new": (-0.51, 2),
        "fp_k40_sel": (0.77, 2), "fp_k40_new": (-0.71, 2),
        # margins
        "side_max_1000": (3.24, 2),
        "e_10_2000": (0.039, 3), "e_1000_2000": (0.051, 3),
        "e_3_800": (0.055, 3), "e_1000_800": (0.081, 3),
        "opt_100_2000": (0.034, 3), "fp_maxcorr_theory": (0.35, 2),
        "sms_se_test": (0.0029, 4), "sms_1se_share": (0.525, 3), "sms_1se_hi": (0.0216, 4),
    }
    for k, (v, nd) in checks.items():
        assert round(float(FACTS[k]), nd) == v, (k, FACTS[k], v)
    # BIC penalty per parameter at the SRM sample size (margin note)
    FACTS["bic_pen_per_param_30"] = math.log(FACTS["srm_ntrain"])
    assert round(FACTS["bic_pen_per_param_30"], 1) == 3.4, FACTS["bic_pen_per_param_30"]
    assert FACTS["fork_total"] == 180
    assert FACTS["srm_d_srm"] == 3 and FACTS["srm_d_test"] == 8
    assert FACTS["sms_M"] == 120 and FACTS["sms_ntest"] == 2072
    assert FACTS["srm_ntrain"] == 30 and FACTS["fp_P"] == 500
    assert FACTS["ncv_outer"] * FACTS["ncv_inner_per_outer"] == 20
    assert FACTS["bike_rows"] == 17379
    assert 36 < FACTS["looks_2000_003"] < 37, FACTS["looks_2000_003"]

    out = ROOT / "scripts" / "data" / "lesson63_facts.json"
    out.write_text(json.dumps({k: (float(v)) for k, v in FACTS.items()},
                              ensure_ascii=False, indent=1), encoding="utf8")
    print("\n--- FACTS ---")
    for k, v in FACTS.items():
        print(f"{k:22s} {v}")
    print("lesson 63 figures written")


main()
