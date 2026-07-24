"""Deterministic figures for lesson 54: generative classifiers (LDA, QDA, naive Bayes).

Real data: iris (scripts/data/iris.data), SMS Spam Collection (scripts/data/sms-spam-collection.tsv),
sklearn load_breast_cancer. One deliberately synthetic 1D example (fixed formulas, no rng) shows
how a rare prior moves the boundary. Every number quoted in the lesson is computed and asserted.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IRIS = ROOT / "scripts" / "data" / "iris.data"
SMS = ROOT / "scripts" / "data" / "sms-spam-collection.tsv"
OUT = ROOT / "public" / "figures" / "lessons" / "54"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "54"

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


# ---------------------------------------------------------------- data
def load_iris_petals():
    """Return (X, y) for versicolor(0) / virginica(1): petal length, petal width."""
    xs, ys = [], []
    with open(IRIS) as f:
        for row in csv.reader(f):
            if len(row) < 5:
                continue
            name = row[4]
            if name == "Iris-versicolor":
                lab = 0
            elif name == "Iris-virginica":
                lab = 1
            else:
                continue
            xs.append([float(row[2]), float(row[3])])
            ys.append(lab)
    return np.array(xs), np.array(ys)


def gauss_params(X, y):
    mu = [X[y == k].mean(axis=0) for k in (0, 1)]
    cov = [np.cov(X[y == k].T, bias=False) for k in (0, 1)]
    n = [np.sum(y == k) for k in (0, 1)]
    pooled = ((n[0] - 1) * cov[0] + (n[1] - 1) * cov[1]) / (n[0] + n[1] - 2)
    return mu, cov, pooled, n


def log_gauss(X, mu, S):
    Si = np.linalg.inv(S)
    d = X - mu
    q = np.einsum("...i,ij,...j->...", d, Si, d)
    return -0.5 * q - 0.5 * math.log(np.linalg.det(S)) - math.log(2 * math.pi)


def ellipse(mu, S, p=0.9, n=200):
    vals, vecs = np.linalg.eigh(S)
    r = math.sqrt(-2 * math.log(1 - p))
    t = np.linspace(0, 2 * math.pi, n)
    circ = np.stack([np.cos(t), np.sin(t)])
    pts = vecs @ (np.diag(np.sqrt(vals) * r) @ circ)
    return mu[0] + pts[0], mu[1] + pts[1]


def boundary(ax, fn, xlim, ylim, color, lw=2.4, ls="-", label=None):
    gx = np.linspace(*xlim, 400)
    gy = np.linspace(*ylim, 400)
    GX, GY = np.meshgrid(gx, gy)
    Z = fn(np.stack([GX, GY], axis=-1))
    ax.contour(GX, GY, Z, levels=[0.0], colors=[color], linewidths=[lw], linestyles=[ls])
    if label:
        ax.plot([], [], color=color, lw=lw, ls=ls, label=label)


# ---------------------------------------------- fig 54.1: class as a density
def fig_clouds():
    X, y = load_iris_petals()
    mu, cov, pooled, n = gauss_params(X, y)
    prior = np.array([0.5, 0.5])

    def disc(P):
        return (log_gauss(P, mu[1], pooled) + math.log(prior[1])
                - log_gauss(P, mu[0], pooled) - math.log(prior[0]))

    pred = (disc(X) > 0).astype(int)
    acc = float((pred == y).mean())
    FACTS["iris_n_per_class"] = int(n[0])
    FACTS["mu_versicolor"] = [round(float(v), 2) for v in mu[0]]
    FACTS["mu_virginica"] = [round(float(v), 2) for v in mu[1]]
    FACTS["lda_full_acc"] = round(acc, 3)
    FACTS["lda_errors"] = int((pred != y).sum())
    print("fig1: mu0", mu[0], "mu1", mu[1], "acc", acc, "errors", (pred != y).sum())
    assert n[0] == n[1] == 50
    assert 0.92 <= acc <= 0.98

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    cols = [BLUE, RED]
    names = ["versicolor", "virginica"]
    for k in (0, 1):
        ax.scatter(X[y == k, 0], X[y == k, 1], s=34, color=cols[k], alpha=0.75,
                   edgecolors=PAPER, linewidths=0.6, label=f"{names[k]} (n={n[k]})", zorder=4)
        for p in (0.5, 0.9):
            ex, ey = ellipse(mu[k], pooled, p)
            ax.plot(ex, ey, color=cols[k], lw=1.2, alpha=0.75 if p == 0.5 else 0.45, zorder=3)
        ax.plot(*mu[k], marker="P", ms=13, color=cols[k], mec=PAPER, mew=1.4, zorder=6)
    boundary(ax, disc, (2.6, 7.4), (0.8, 2.8), INK, lw=2.2, label="граница равных апостериорных")
    ax.set_xlabel("длина лепестка, см"); ax.set_ylabel("ширина лепестка, см")
    ax.set_title(f"Класс — это плотность, а не только цвет точек (ирис, {acc*100:.0f}% верно)")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "clouds.png")


# ---------------------------------------------- fig 54.2: LDA vs QDA
def fig_lda_qda():
    X, y = load_iris_petals()
    mu, cov, pooled, n = gauss_params(X, y)
    lp = math.log(0.5)

    def disc_lda(P):
        return log_gauss(P, mu[1], pooled) - log_gauss(P, mu[0], pooled)

    def disc_qda(P):
        return log_gauss(P, mu[1], cov[1]) - log_gauss(P, mu[0], cov[0])

    acc_l = float(((disc_lda(X) > 0).astype(int) == y).mean())
    acc_q = float(((disc_qda(X) > 0).astype(int) == y).mean())
    # honest split
    rng = np.random.default_rng(54)
    idx = rng.permutation(len(y))
    tr, te = idx[:60], idx[60:]
    mu_t, cov_t, pooled_t, _ = gauss_params(X[tr], y[tr])

    def dl(P):
        return log_gauss(P, mu_t[1], pooled_t) - log_gauss(P, mu_t[0], pooled_t)

    def dq(P):
        return log_gauss(P, mu_t[1], cov_t[1]) - log_gauss(P, mu_t[0], cov_t[0])

    te_l = float(((dl(X[te]) > 0).astype(int) == y[te]).mean())
    te_q = float(((dq(X[te]) > 0).astype(int) == y[te]).mean())
    FACTS["lda_resub"] = round(acc_l, 3)
    FACTS["qda_resub"] = round(acc_q, 3)
    FACTS["lda_test60"] = round(te_l, 3)
    FACTS["qda_test60"] = round(te_q, 3)
    FACTS["cov_det_ratio"] = round(float(np.linalg.det(cov[1]) / np.linalg.det(cov[0])), 2)
    print("fig2: resub LDA", acc_l, "QDA", acc_q, "| test LDA", te_l, "QDA", te_q,
          "| det ratio", FACTS["cov_det_ratio"])
    assert acc_q >= acc_l
    assert FACTS["cov_det_ratio"] > 1.5

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.0), sharex=True, sharey=True)
    cols = [BLUE, RED]
    for ax, kind, dfn, acc in ((axes[0], "LDA: одна общая $\\Sigma$", disc_lda, acc_l),
                               (axes[1], "QDA: своя $\\Sigma_k$ у каждого класса", disc_qda, acc_q)):
        for k in (0, 1):
            ax.scatter(X[y == k, 0], X[y == k, 1], s=26, color=cols[k], alpha=0.7,
                       edgecolors=PAPER, linewidths=0.5, zorder=4)
            S = pooled if dfn is disc_lda else cov[k]
            for p in (0.5, 0.9):
                ex, ey = ellipse(mu[k], S, p)
                ax.plot(ex, ey, color=cols[k], lw=1.2, alpha=0.8 if p == 0.5 else 0.45, zorder=3)
            ax.plot(*mu[k], marker="P", ms=11, color=cols[k], mec=PAPER, mew=1.2, zorder=6)
        boundary(ax, dfn, (2.6, 7.4), (0.8, 2.8), INK, lw=2.2)
        ax.set_title(f"{kind}\nверно {acc*100:.0f}%", fontsize=13)
        ax.set_xlabel("длина лепестка, см")
        ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    axes[0].set_ylabel("ширина лепестка, см")
    fig.suptitle("Форма облаков решает, будет ли граница прямой", fontsize=15, y=1.02)
    save(fig, OUT / "lda_qda.png")


# ---------------------------------------------- fig 54.3: prior moves the boundary
def fig_prior():
    # deliberately synthetic 1D model: N(0,1) vs N(2,1)
    pi1 = 0.02
    shift = -0.5 * math.log(pi1 / (1 - pi1))
    x_star = 1.0 + shift
    lr_at_2 = math.exp(2 * 2 - 2)
    odds = lr_at_2 * pi1 / (1 - pi1)
    post_at_2 = odds / (1 + odds)
    post_at_2_bal = math.exp(2) / (1 + math.exp(2))
    FACTS["prior_shift_log"] = round(math.log(pi1 / (1 - pi1)), 2)
    FACTS["x_star"] = round(x_star, 2)
    FACTS["post_x2_rare"] = round(post_at_2 * 100, 1)
    FACTS["post_x2_balanced"] = round(post_at_2_bal * 100, 1)
    # false alarms if the balanced boundary x=1 is shipped into a 2% population
    tail0 = 0.5 * math.erfc((1.0 - 0.0) / math.sqrt(2))          # P(X>1 | class 0)
    tail1 = 0.5 * math.erfc((1.0 - 2.0) / math.sqrt(2))          # P(X>1 | class 1)
    ppv_bal = pi1 * tail1 / (pi1 * tail1 + (1 - pi1) * tail0)
    t0 = 0.5 * math.erfc((x_star - 0.0) / math.sqrt(2))
    t1 = 0.5 * math.erfc((x_star - 2.0) / math.sqrt(2))
    ppv_cor = pi1 * t1 / (pi1 * t1 + (1 - pi1) * t0)
    rec_bal, rec_cor = tail1, t1
    FACTS["ppv_balanced"] = round(ppv_bal * 100, 1)
    FACTS["ppv_corrected"] = round(ppv_cor * 100, 1)
    FACTS["recall_balanced"] = round(rec_bal * 100, 1)
    FACTS["recall_corrected"] = round(rec_cor * 100, 1)
    print("fig3: x*", x_star, "post(2)", post_at_2, "ppv bal", ppv_bal, "ppv cor", ppv_cor,
          "recall", rec_bal, rec_cor)
    assert abs(x_star - 2.9459) < 5e-05
    assert 12.0 < post_at_2 * 100 < 14.0
    assert ppv_cor > 4 * ppv_bal

    xs = np.linspace(-3.5, 6.5, 600)
    d0 = np.exp(-xs ** 2 / 2) / math.sqrt(2 * math.pi)
    d1 = np.exp(-(xs - 2) ** 2 / 2) / math.sqrt(2 * math.pi)
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.15]})
    ax = axes[0]
    ax.fill_between(xs, 0, (1 - pi1) * d0, color=BLUE, alpha=0.18)
    ax.fill_between(xs, 0, pi1 * d1, color=RED, alpha=0.30)
    ax.plot(xs, (1 - pi1) * d0, color=BLUE, lw=2.0, label="частый класс: $0{,}98\\cdot N(0,1)$")
    ax.plot(xs, pi1 * d1, color=RED, lw=2.0, label="редкий класс: $0{,}02\\cdot N(2,1)$")
    ax.axvline(1.0, color=MUTED, lw=1.6, ls=(0, (5, 3)))
    ax.axvline(x_star, color=INK, lw=2.0)
    ax.text(1.0, 0.30, " граница при\n равных prior\n $x=1$", color=MUTED, fontsize=10, ha="left")
    ax.text(x_star + 0.1, 0.12, f" реальный prior:\n $x_*={x_star:.2f}$".replace(".", "{,}"), color=INK, fontsize=10)
    ax.set_ylabel("взвешенная плотность")
    ax.set_title("Точки не двигались — двигался prior")
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    ax2 = axes[1]
    post_bal = 1 / (1 + np.exp(-(2 * xs - 2)))
    post_rare = 1 / (1 + np.exp(-(2 * xs - 2 + math.log(pi1 / (1 - pi1)))))
    ax2.plot(xs, post_bal, color=MUTED, lw=1.8, ls=(0, (5, 3)), label="posterior при равных prior")
    ax2.plot(xs, post_rare, color=INK, lw=2.2, label="posterior при prior 2%")
    ax2.axhline(0.5, color=GRID, lw=1.2)
    ax2.plot([2.0], [post_at_2], marker="o", ms=8, color=RED, mec=PAPER, mew=1.2, zorder=6)
    ax2.text(2.1, post_at_2 + 0.06, f"$x=2$: {post_at_2*100:.1f}% вместо {post_at_2_bal*100:.1f}%".replace(".", ","),
             color=RED, fontsize=10)
    ax2.set_xlabel("значение признака $x$"); ax2.set_ylabel("$P(y=1\\mid x)$")
    ax2.legend(loc="upper left", frameon=False, fontsize=10)
    ax2.grid(True, color=GRID, lw=0.4, alpha=0.4); ax2.set_axisbelow(True)
    save(fig, OUT / "prior_shift.png")


# ---------------------------------------------- fig 54.4: naive Bayes on real spam
def spam_corpus():
    labels, texts = [], []
    with open(SMS, encoding="utf8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            labels.append(1 if parts[0].strip() == "spam" else 0)
            texts.append(parts[1])
    return np.array(labels), texts


def fig_naive_bayes():
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    y, texts = spam_corpus()
    FACTS["sms_n"] = int(len(y))
    FACTS["sms_spam_share"] = round(float(y.mean()) * 100, 1)
    Xtr_t, Xte_t, ytr, yte = train_test_split(texts, y, test_size=0.3, random_state=54, stratify=y)
    vec = CountVectorizer(lowercase=True, min_df=2)
    Xtr = vec.fit_transform(Xtr_t); Xte = vec.transform(Xte_t)
    FACTS["sms_vocab"] = int(Xtr.shape[1])
    nb = MultinomialNB(alpha=1.0).fit(Xtr, ytr)
    p = nb.predict_proba(Xte)[:, 1]
    pred = (p > 0.5).astype(int)
    acc = float((pred == yte).mean())
    tp = int(((pred == 1) & (yte == 1)).sum()); fp = int(((pred == 1) & (yte == 0)).sum())
    fn = int(((pred == 0) & (yte == 1)).sum())
    prec = tp / (tp + fp); rec = tp / (tp + fn)
    lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    plr = lr.predict_proba(Xte)[:, 1]
    acc_lr = float(((plr > 0.5).astype(int) == yte).mean())
    extreme = float(np.mean((p > 0.99) | (p < 0.01)))
    extreme_lr = float(np.mean((plr > 0.99) | (plr < 0.01)))
    # calibration in 10 bins
    bins = np.linspace(0, 1, 11)
    xs_c, ys_c, ws = [], [], []
    for i in range(10):
        m = (p >= bins[i]) & (p < bins[i + 1]) if i < 9 else (p >= bins[i]) & (p <= 1)
        if m.sum() >= 5:
            xs_c.append(float(p[m].mean())); ys_c.append(float(yte[m].mean())); ws.append(int(m.sum()))
    mid = [(i, x, yv, w) for i, (x, yv, w) in enumerate(zip(xs_c, ys_c, ws)) if 0.05 < x < 0.95]
    FACTS["nb_acc"] = round(acc * 100, 1)
    FACTS["nb_precision"] = round(prec * 100, 1)
    FACTS["nb_recall"] = round(rec * 100, 1)
    FACTS["lr_acc"] = round(acc_lr * 100, 1)
    FACTS["nb_extreme"] = round(extreme * 100, 1)
    FACTS["lr_extreme"] = round(extreme_lr * 100, 1)
    FACTS["nb_test_n"] = int(len(yte))
    print("fig4: n", len(y), "vocab", Xtr.shape[1], "acc", acc, "prec", prec, "rec", rec,
          "LR acc", acc_lr, "extreme NB", extreme, "LR", extreme_lr, "mid bins", mid)
    assert acc > 0.95 and prec > 0.85 and rec > 0.80
    assert extreme > 0.85 and extreme > extreme_lr + 0.15

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    ax = axes[0]
    ax.hist(np.clip(p, 0, 1), bins=np.linspace(0, 1, 41), color=BLUE, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("апостериорная вероятность спама"); ax.set_ylabel("число писем (лог. шкала)")
    ax.set_title(f"Naive Bayes уверен почти всегда\n{extreme*100:.1f}% ответов вне [0,01; 0,99]".replace(".", ",", 1), fontsize=13)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    ax2 = axes[1]
    ax2.plot([0, 1], [0, 1], color=MUTED, lw=1.4, ls=(0, (5, 3)), label="идеальная калибровка")
    ax2.plot(xs_c, ys_c, marker="o", ms=8, color=RED, lw=2.0, mec=PAPER, mew=1.2,
             label="naive Bayes на тесте")
    for x0, y0, w in zip(xs_c, ys_c, ws):
        ax2.annotate(str(w), (x0, y0), textcoords="offset points", xytext=(6, -12),
                     color=MUTED, fontsize=9)
    ax2.set_xlabel("заявленная вероятность"); ax2.set_ylabel("доля спама на самом деле")
    ax2.set_title(f"Калибровка: {acc*100:.1f}% верных ответов\nи почти пустая середина шкалы".replace(".", ",", 1), fontsize=13)
    ax2.legend(loc="upper left", frameon=False, fontsize=10)
    ax2.grid(True, color=GRID, lw=0.4, alpha=0.4); ax2.set_axisbelow(True)
    fig.suptitle("Хорошее ранжирование не значит честную вероятность (реальный SMS-корпус)",
                 fontsize=15, y=1.03)
    save(fig, OUT / "naive_bayes_calibration.png")


# ---------------------------------------------- fig 54.5: QDA collapses with dimension
def fig_dimension():
    from sklearn.datasets import load_breast_cancer
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = load_breast_cancer()
    X, y = data.data, data.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=60, random_state=54, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr = sc.transform(Xtr); Xte = sc.transform(Xte)
    ds = [2, 5, 8, 12, 16, 20, 25, 30]
    res = {"LDA": [], "QDA": [], "shrink": []}
    for d in ds:
        A, B = Xtr[:, :d], Xte[:, :d]
        res["LDA"].append(float(LDA(solver="lsqr", shrinkage=None).fit(A, ytr).score(B, yte)))
        res["QDA"].append(float(QDA(reg_param=0.0).fit(A, ytr).score(B, yte)))
        res["shrink"].append(float(QDA(reg_param=0.5).fit(A, ytr).score(B, yte)))
    FACTS["bc_train_n"] = int(len(ytr)); FACTS["bc_test_n"] = int(len(yte))
    FACTS["bc_lda_d30"] = round(res["LDA"][-1] * 100, 1)
    FACTS["bc_qda_d30"] = round(res["QDA"][-1] * 100, 1)
    FACTS["bc_shrink_d30"] = round(res["shrink"][-1] * 100, 1)
    FACTS["bc_qda_d5"] = round(res["QDA"][ds.index(5)] * 100, 1)
    FACTS["bc_lda_d5"] = round(res["LDA"][ds.index(5)] * 100, 1)
    FACTS["qda_params_d30"] = int(2 * 30 * 31 // 2)
    majority = float(max(yte.mean(), 1 - yte.mean()))
    FACTS["bc_majority"] = round(majority * 100, 1)
    shrink_from5 = [v for d, v in zip(ds, res["shrink"]) if d >= 5]
    FACTS["bc_shrink_min5"] = round(min(shrink_from5) * 100, 1)
    print("fig5:", ds, res, "majority", majority)
    assert res["QDA"][-1] < res["LDA"][-1]
    assert res["shrink"][-1] > res["QDA"][-1]
    # QDA does not merely drop: it degenerates into the constant "majority class" answer
    assert abs(res["QDA"][-1] - majority) < 1e-9
    assert min(shrink_from5) > 0.91

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(ds, np.array(res["LDA"]) * 100, marker="o", ms=7, color=BLUE, lw=2.2,
            label="LDA (одна общая $\\Sigma$)")
    ax.plot(ds, np.array(res["QDA"]) * 100, marker="s", ms=7, color=RED, lw=2.2,
            label="QDA (две матрицы $\\Sigma_k$)")
    ax.plot(ds, np.array(res["shrink"]) * 100, marker="^", ms=7, color=GREEN, lw=2.2,
            label="QDA со стягиванием ($\\gamma=0{,}5$)")
    ax.axhline(majority * 100, color=MUTED, lw=1.4, ls=(0, (4, 3)))
    ax.text(13.5, majority * 100 + 0.9,
            f"ответ «всегда частый класс»: {majority*100:.1f}%".replace(".", ","), color=MUTED, fontsize=10)
    ax.set_xlabel("число признаков $d$"); ax.set_ylabel("точность на тесте, %")
    ax.set_title(f"Обучающих объектов всего {len(ytr)}: гибкость становится обузой")
    ax.legend(loc="center left", frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "dimension.png")


# ---------------------------------------------- fig 54.6: marginalising a missing feature
def fig_missing():
    X, y = load_iris_petals()
    mu, cov, pooled, n = gauss_params(X, y)

    def post2(P):
        a = log_gauss(np.array(P), mu[1], pooled)
        b = log_gauss(np.array(P), mu[0], pooled)
        return 1 / (1 + math.exp(b - a))

    # marginal over petal width -> 1D normal in petal length
    m0, m1 = mu[0][0], mu[1][0]
    s2 = pooled[0, 0]

    def post1(x1):
        a = -(x1 - m1) ** 2 / (2 * s2)
        b = -(x1 - m0) ** 2 / (2 * s2)
        return 1 / (1 + math.exp(b - a))

    x_probe = np.array([5.0, 1.5])
    p_full = post2(x_probe); p_marg = post1(x_probe[0])
    x_boundary_1d = (m0 + m1) / 2
    FACTS["probe_full"] = round(p_full * 100, 1)
    FACTS["probe_marg"] = round(p_marg * 100, 1)
    FACTS["marg_boundary_len"] = round(float(x_boundary_1d), 2)
    FACTS["pooled_var_len"] = round(float(s2), 3)
    # how many objects flip verdict when the width is dropped
    flips = sum(1 for row in X if (post2(row) > 0.5) != (post1(row[0]) > 0.5))
    FACTS["flip_count"] = int(flips)
    acc_full = float(((np.array([post2(r) for r in X]) > 0.5).astype(int) == y).mean())
    acc_marg = float(((np.array([post1(r[0]) for r in X]) > 0.5).astype(int) == y).mean())
    FACTS["acc_full"] = round(acc_full * 100, 1)
    FACTS["acc_marg"] = round(acc_marg * 100, 1)
    print("fig6: probe", p_full, p_marg, "flips", flips, "acc", acc_full, acc_marg)
    assert p_full < 0.5 < p_marg
    assert acc_full > acc_marg and flips >= 3

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.0), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    cols = [BLUE, RED]
    for k in (0, 1):
        ax.scatter(X[y == k, 0], X[y == k, 1], s=26, color=cols[k], alpha=0.6,
                   edgecolors=PAPER, linewidths=0.5, zorder=3)
        ex, ey = ellipse(mu[k], pooled, 0.9)
        ax.plot(ex, ey, color=cols[k], lw=1.4, alpha=0.7, zorder=3)
    boundary(ax, lambda P: log_gauss(P, mu[1], pooled) - log_gauss(P, mu[0], pooled),
             (2.6, 7.4), (0.8, 2.8), INK, lw=2.0)
    ax.plot(*x_probe, marker="*", ms=20, color=GOLD, mec=INK, mew=0.9, zorder=8)
    ax.annotate(f"объект (5{chr(44)}0; 1{chr(44)}5)\nпо двум признакам virginica лишь на {p_full*100:.0f}%",
                x_probe, textcoords="offset points", xytext=(-4, -48), color=GOLD, fontsize=10,
                ha="center")
    ax.axvline(x_probe[0], color=GOLD, lw=1.0, ls=(0, (3, 3)), alpha=0.8)
    ax.set_xlabel("длина лепестка, см"); ax.set_ylabel("ширина лепестка, см")
    ax.set_title("Полная плотность в плоскости", fontsize=13)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    ax2 = axes[1]
    xs = np.linspace(2.6, 7.4, 400)
    for k, m in ((0, m0), (1, m1)):
        dens = np.exp(-(xs - m) ** 2 / (2 * s2)) / math.sqrt(2 * math.pi * s2)
        ax2.plot(xs, dens, color=cols[k], lw=2.2)
        ax2.fill_between(xs, 0, dens, color=cols[k], alpha=0.15)
    ax2.axvline(x_boundary_1d, color=INK, lw=2.0)
    ax2.axvline(x_probe[0], color=GOLD, lw=1.6, ls=(0, (3, 3)))
    ax2.text(x_boundary_1d - 0.12, ax2.get_ylim()[1] * 0.95,
             f"граница $x_1={x_boundary_1d:.2f}$".replace(".", "{,}"), color=INK, fontsize=10, ha="right")
    ax2.text(x_probe[0] + 0.12, ax2.get_ylim()[1] * 0.72,
             f"по одной длине\nvirginica на {p_marg*100:.0f}%", color=GOLD, fontsize=10, ha="left")
    ax2.set_xlabel("длина лепестка, см"); ax2.set_ylabel("плотность после маргинализации")
    ax2.set_title("Ширина неизвестна: интегрируем её", fontsize=13)
    ax2.grid(True, color=GRID, lw=0.4, alpha=0.4); ax2.set_axisbelow(True)
    fig.suptitle("Пропуск признака меняет не только уверенность, но и приговор", fontsize=15, y=1.02)
    save(fig, OUT / "missing.png")


# ---------------------------------------------- sidenote 1: Mahalanobis
def side_mahalanobis():
    S = np.diag([4.0, 1.0])
    A, B = np.array([2.0, 0.0]), np.array([0.0, 2.0])
    dA = math.sqrt(A @ np.linalg.inv(S) @ A)
    dB = math.sqrt(B @ np.linalg.inv(S) @ B)
    FACTS["mahal_A"] = round(dA, 2); FACTS["mahal_B"] = round(dB, 2)
    print("side1: dA", dA, "dB", dB)
    assert abs(dA - 1.0) < 0.05 and abs(dB - 2.0) < 1e-9

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for r, alpha in ((1, 0.9), (2, 0.5)):
        ex, ey = ellipse(np.zeros(2), S * r ** 2, 1 - math.exp(-0.5))
        ax.plot(ex, ey, color=BLUE, lw=1.6, alpha=alpha)
    ax.plot(*A, marker="o", ms=9, color=RED, mec=PAPER, mew=1.0)
    ax.plot(*B, marker="o", ms=9, color=GOLD, mec=PAPER, mew=1.0)
    ax.annotate("A (2;0)\n$d_M=1$", A, textcoords="offset points", xytext=(6, 8), color=RED, fontsize=10)
    ax.annotate("B (0;2)\n$d_M=2$", B, textcoords="offset points", xytext=(6, 4), color=GOLD, fontsize=10)
    ax.plot(0, 0, marker="P", ms=10, color=INK)
    ax.set_xlim(-5, 5); ax.set_ylim(-4, 4); ax.set_aspect("equal")
    ax.set_title("Одинаковы по Евклиду,\nразные по Махаланобису", fontsize=12)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "mahalanobis.png")


# ---------------------------------------------- sidenote 2: parameter counts
def side_params():
    ds = np.array([2, 5, 10, 20, 50, 100])
    K = 3
    lda = K * ds + ds * (ds + 1) / 2
    qda = K * ds + K * ds * (ds + 1) / 2
    nb = 2 * K * ds
    FACTS["params_d100_qda"] = int(K * 100 + K * 100 * 101 // 2)
    FACTS["params_d100_lda"] = int(K * 100 + 100 * 101 // 2)
    FACTS["params_d100_nb"] = int(2 * K * 100)
    print("side2:", FACTS["params_d100_lda"], FACTS["params_d100_qda"], FACTS["params_d100_nb"])
    assert FACTS["params_d100_qda"] == 15450

    fig, ax = plt.subplots(figsize=(4.8, 3.9))
    ax.plot(ds, qda, marker="s", ms=6, color=RED, lw=2.0, label="QDA")
    ax.plot(ds, lda, marker="o", ms=6, color=BLUE, lw=2.0, label="LDA")
    ax.plot(ds, nb, marker="^", ms=6, color=GREEN, lw=2.0, label="naive Bayes")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("признаков $d$"); ax.set_ylabel("параметров")
    ax.set_title("Цена гибкости, 3 класса", fontsize=12)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "params.png")


# ---------------------------------------------- sidenote 3: naive Bayes double counting
def side_double():
    # exact 2-class Gaussian model with correlated features; no randomness
    rho = 0.9
    S = np.array([[1.0, rho], [rho, 1.0]])
    mu1 = np.array([1.0, 1.0]); mu0 = -mu1
    ts = np.linspace(-1.6, 1.6, 200)
    true_p, naive_p = [], []
    for t in ts:
        x = np.array([t, t])
        lo_true = (log_gauss(x, mu1, S) - log_gauss(x, mu0, S))
        Sd = np.diag(np.diag(S))
        lo_naive = (log_gauss(x, mu1, Sd) - log_gauss(x, mu0, Sd))
        true_p.append(1 / (1 + math.exp(-lo_true)))
        naive_p.append(1 / (1 + math.exp(-lo_naive)))
    x_ref = np.array([0.5, 0.5])
    p_true = 1 / (1 + math.exp(-(log_gauss(x_ref, mu1, S) - log_gauss(x_ref, mu0, S))))
    p_naive = 1 / (1 + math.exp(-(log_gauss(x_ref, mu1, np.diag(np.diag(S)))
                                  - log_gauss(x_ref, mu0, np.diag(np.diag(S))))))
    FACTS["double_true"] = round(p_true * 100, 1)
    FACTS["double_naive"] = round(p_naive * 100, 1)
    FACTS["double_rho"] = rho
    print("side3: true", p_true, "naive", p_naive)
    assert p_naive > p_true + 0.1

    fig, ax = plt.subplots(figsize=(4.8, 3.9))
    ax.plot(ts, true_p, color=BLUE, lw=2.2, label="верная модель")
    ax.plot(ts, naive_p, color=RED, lw=2.2, ls=(0, (5, 3)), label="naive Bayes")
    ax.axhline(0.5, color=GRID, lw=1.2)
    ax.plot([0.5], [p_true], marker="o", ms=7, color=BLUE, mec=PAPER, mew=1.0)
    ax.plot([0.5], [p_naive], marker="o", ms=7, color=RED, mec=PAPER, mew=1.0)
    ax.set_xlabel("$x_1=x_2=t$"); ax.set_ylabel("$P(y=1\\mid x)$")
    ax.set_title(f"Два почти одинаковых\nсвидетельства ($\\rho={rho}$)".replace("0.9", "0{,}9"), fontsize=12)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "double_counting.png")


# ---------------------------------------------- sidenote 4: cost matrix
def side_risk():
    # condition risk for two actions as posterior varies, c(alarm|healthy)=1, c(miss|sick)=20
    c_fp, c_fn = 1.0, 20.0
    p = np.linspace(0, 1, 300)
    r_alarm = c_fp * (1 - p)
    r_calm = c_fn * p
    p_star = c_fp / (c_fp + c_fn)
    FACTS["risk_threshold"] = round(float(p_star), 4)
    print("side4: p*", p_star)
    assert abs(p_star - 1 / 21) < 1e-9

    fig, ax = plt.subplots(figsize=(4.8, 3.9))
    ax.plot(p, r_alarm, color=BLUE, lw=2.2, label="объявить тревогу")
    ax.plot(p, r_calm, color=RED, lw=2.2, label="промолчать")
    ax.axvline(p_star, color=INK, lw=1.6, ls=(0, (4, 3)))
    ax.text(p_star + 0.02, 12, f"$p_*={p_star:.3f}$".replace(".", "{,}"), color=INK, fontsize=10)
    ax.set_xlabel("$P(y=\\mathrm{болен}\\mid x)$"); ax.set_ylabel("условный риск")
    ax.set_title("Порог задаёт не модель,\nа матрица цен", fontsize=12)
    ax.legend(frameon=False, fontsize=10, loc="upper center")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "risk.png")


def main():
    fig_clouds()
    fig_lda_qda()
    fig_prior()
    fig_naive_bayes()
    fig_dimension()
    fig_missing()
    side_mahalanobis()
    side_params()
    side_double()
    side_risk()
    print(json.dumps(FACTS, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
