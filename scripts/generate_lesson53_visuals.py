"""Deterministic figures for lesson 53: geometry of a linear classifier.

Real data: SMS Spam Collection (scripts/data/sms-spam-collection.tsv) with two honest
features — the share of digits in the message and its length — plus the iris set for the
multiclass panel. Every number quoted in the lesson text is computed and asserted here.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

ROOT = Path(__file__).resolve().parents[1]
SMS = ROOT / "scripts" / "data" / "sms-spam-collection.tsv"
OUT = ROOT / "public" / "figures" / "lessons" / "53"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "53"

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


# ------------------------------------------------------------------ real data
def load_sms():
    y, texts = [], []
    with open(SMS, encoding="utf8") as f:
        for raw in f:
            parts = raw.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            y.append(1 if parts[0] == "spam" else 0)
            texts.append(parts[1])
    return np.array(y), texts


def two_features(texts):
    rows = []
    for t in texts:
        n = max(len(t), 1)
        rows.append([sum(c.isdigit() for c in t) / n, len(t) / 1000.0])
    return np.array(rows)


Y_ALL, TEXTS = load_sms()
X_ALL = two_features(TEXTS)
IDX = np.arange(len(Y_ALL))
itr, ite = train_test_split(IDX, test_size=0.3, random_state=53, stratify=Y_ALL)
XTR, XTE, YTR, YTE = X_ALL[itr], X_ALL[ite], Y_ALL[itr], Y_ALL[ite]

MODEL = LogisticRegression(max_iter=1000).fit(XTR, YTR)
W = MODEL.coef_[0]
B = float(MODEL.intercept_[0])
S_TE = XTE @ W + B
P_TE = 1.0 / (1.0 + np.exp(-S_TE))
ACC = accuracy_score(YTE, P_TE >= 0.5)
AUC = roc_auc_score(YTE, P_TE)


def register_basics():
    FACTS["n_total"] = len(Y_ALL)
    FACTS["n_spam"] = int(Y_ALL.sum())
    FACTS["spam_share"] = float(Y_ALL.mean())
    FACTS["n_train"] = len(YTR)
    FACTS["n_test"] = len(YTE)
    FACTS["w1"] = float(W[0]); FACTS["w2"] = float(W[1]); FACTS["b"] = B
    FACTS["w_norm"] = float(np.hypot(*W))
    FACTS["acc"] = float(ACC); FACTS["auc"] = float(AUC)
    assert FACTS["n_total"] == 5574 and FACTS["n_spam"] == 747
    assert abs(FACTS["spam_share"] - 0.134) < 0.001
    assert FACTS["n_train"] == 3901 and FACTS["n_test"] == 1673
    assert 19.3 < FACTS["w1"] < 19.7, FACTS["w1"]
    assert 7.0 < FACTS["w2"] < 7.4, FACTS["w2"]
    assert -3.3 < FACTS["b"] < -3.1, FACTS["b"]
    assert 20.6 < FACTS["w_norm"] < 21.0, FACTS["w_norm"]
    assert 0.93 < FACTS["acc"] < 0.94, FACTS["acc"]
    assert 0.970 < FACTS["auc"] < 0.976, FACTS["auc"]


def counts_at(thr):
    pred = P_TE >= thr
    tp = int(np.sum(pred & (YTE == 1))); fp = int(np.sum(pred & (YTE == 0)))
    fn = int(np.sum(~pred & (YTE == 1))); tn = int(np.sum(~pred & (YTE == 0)))
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn)
    return tp, fp, fn, tn, prec, rec


# ------------------------------------------- fig 53.1: plane, normal, distance
def fig_plane():
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ham = XTR[YTR == 0]; spam = XTR[YTR == 1]
    rng = np.random.default_rng(53)
    sub = rng.choice(len(ham), 700, replace=False)
    ax.scatter(ham[sub, 0], ham[sub, 1], s=13, color=BLUE, alpha=0.35, label="обычные письма (ham)")
    ax.scatter(spam[:, 0], spam[:, 1], s=15, color=RED, alpha=0.5, label="спам")
    xs = np.linspace(-0.01, 0.42, 100)
    ys = -(W[0] * xs + B) / W[1]
    ax.plot(xs, ys, color=INK, lw=2.4, label="граница $s(x)=0$")
    # normal arrow from a point on the boundary
    x0 = 0.06; y0 = -(W[0] * x0 + B) / W[1]
    u = W / FACTS["w_norm"] * 0.09
    ax.annotate("", xy=(x0 + u[0], y0 + u[1]), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=2.6))
    ax.text(x0 + u[0] + 0.008, y0 + u[1], "$w$", color=GOLD, fontsize=15)
    # signed distance of one concrete message
    px, py = 0.30, 0.140
    s_pt = W[0] * px + W[1] * py + B
    d_pt = s_pt / FACTS["w_norm"]
    t = -s_pt / (W @ W)
    fx, fy = px + W[0] * t, py + W[1] * t
    ax.plot([px, fx], [py, fy], color=GREEN, lw=1.8, ls=(0, (4, 3)))
    ax.scatter([px], [py], s=70, color=GREEN, zorder=6, edgecolor=PAPER, lw=1.2)
    ax.text(px + 0.01, py + 0.02, f"$s={s_pt:.2f}$,\nрасстояние ${d_pt:.3f}$", color=GREEN, fontsize=10.5)
    FACTS["pt_score"] = float(s_pt); FACTS["pt_dist"] = float(d_pt)
    FACTS["pt_prob"] = float(1 / (1 + np.exp(-s_pt)))
    FACTS["pt_dist_08"] = float((s_pt - np.log(4)) / FACTS["w_norm"])
    FACTS["odds_digits_1pp"] = float(np.exp(W[0] * 0.01))
    FACTS["odds_len_100"] = float(np.exp(W[1] * 0.1))
    assert 3.55 < s_pt < 3.75 and 0.17 < d_pt < 0.18, (s_pt, d_pt)
    assert 0.970 < FACTS["pt_prob"] < 0.978, FACTS["pt_prob"]
    assert 0.105 < FACTS["pt_dist_08"] < 0.115, FACTS["pt_dist_08"]
    assert 1.20 < FACTS["odds_digits_1pp"] < 1.23, FACTS["odds_digits_1pp"]
    assert 2.00 < FACTS["odds_len_100"] < 2.10, FACTS["odds_len_100"]
    ax.set_xlim(-0.01, 0.42); ax.set_ylim(0, 0.34)
    ax.set_aspect("equal")
    ax.set_xlabel("доля цифр в сообщении $x_1$")
    ax.set_ylabel("длина / 1000 символов $x_2$")
    ax.set_title(f"Реальный спам: где проходит прямая (accuracy {ACC*100:.1f}%)")
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "sms_plane.png")


# --------------------------------- fig 53.2: score distribution and the sigmoid
def fig_score_to_prob():
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    ax = axes[0]
    bins = np.linspace(-8, 10, 46)
    ax.hist(S_TE[YTE == 0], bins=bins, color=BLUE, alpha=0.55, label="ham")
    ax.hist(S_TE[YTE == 1], bins=bins, color=RED, alpha=0.6, label="спам")
    ax.set_yscale("log")
    for p, c in [(0.2, GREEN), (0.5, INK), (0.8, GOLD)]:
        s = np.log(p / (1 - p))
        ax.axvline(s, color=c, lw=1.8, ls=(0, (4, 3)))
        ax.text(s, 900, f"p*={p}", rotation=90, color=c, fontsize=9.5, ha="right", va="top")
    ax.set_xlabel("score $s(x)=w^\\top x+b$"); ax.set_ylabel("число писем (лог. шкала)")
    ax.set_title("Один score, три порога")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)

    ax = axes[1]
    ss = np.linspace(-8, 8, 400)
    ax.plot(ss, 1 / (1 + np.exp(-ss)), color=VIOLET, lw=2.6)
    for p, c in [(0.2, GREEN), (0.5, INK), (0.8, GOLD)]:
        s = np.log(p / (1 - p))
        ax.plot([s, s], [0, p], color=c, lw=1.4, ls=(0, (3, 3)))
        ax.plot([-8, s], [p, p], color=c, lw=1.4, ls=(0, (3, 3)))
        ax.text(-7.8, p + 0.03, f"logit({p})={s:+.2f}", color=c, fontsize=9.5)
    ax.set_xlim(-8, 8); ax.set_ylim(0, 1)
    ax.set_xlabel("score $s$"); ax.set_ylabel("$\\sigma(s)$")
    ax.set_title("Порог вероятности — это порог по score")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    FACTS["logit02"] = float(np.log(0.2 / 0.8)); FACTS["logit08"] = float(np.log(4))
    assert abs(FACTS["logit08"] - 1.3863) < 1e-3
    fig.tight_layout()
    save(fig, OUT / "score_to_prob.png")


# ------------------------------ fig 53.3: three thresholds = parallel transport
def fig_thresholds():
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ham = XTE[YTE == 0]; spam = XTE[YTE == 1]
    rng = np.random.default_rng(531)
    sub = rng.choice(len(ham), 500, replace=False)
    ax.scatter(ham[sub, 0], ham[sub, 1], s=12, color=BLUE, alpha=0.3)
    ax.scatter(spam[:, 0], spam[:, 1], s=14, color=RED, alpha=0.45)
    xs = np.linspace(-0.01, 0.42, 100)
    rows = []
    for p, c in [(0.2, GREEN), (0.5, INK), (0.8, GOLD)]:
        thr_s = np.log(p / (1 - p))
        ys = (thr_s - W[0] * xs - B) / W[1]
        tp, fp, fn, tn, prec, rec = counts_at(p)
        rows.append((p, tp, fp, fn, prec, rec))
        ax.plot(xs, ys, color=c, lw=2.2,
                label=f"$p_*={p}$: precision {prec*100:.0f}%, recall {rec*100:.0f}%")
    ax.set_xlim(-0.01, 0.42); ax.set_ylim(0, 0.34); ax.set_aspect("equal")
    ax.set_xlabel("доля цифр $x_1$"); ax.set_ylabel("длина / 1000 $x_2$")
    ax.set_title("Веса те же, политика разная: пороги двигают прямую параллельно")
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "thresholds_plane.png")
    for p, tp, fp, fn, prec, rec in rows:
        key = str(p).replace(".", "")
        FACTS[f"prec_{key}"] = prec; FACTS[f"rec_{key}"] = rec
        FACTS[f"tp_{key}"] = tp; FACTS[f"fp_{key}"] = fp; FACTS[f"fn_{key}"] = fn
    assert FACTS["rec_02"] > FACTS["rec_05"] > FACTS["rec_08"]
    assert FACTS["prec_02"] < FACTS["prec_05"] < FACTS["prec_08"]


# ------------------------------------------- fig 53.4: cost curve, optimal p*
def fig_cost():
    c_fp, c_fn = 1.0, 10.0
    thrs = np.linspace(0.01, 0.99, 197)
    costs, precs, recs = [], [], []
    for t in thrs:
        tp, fp, fn, tn, prec, rec = counts_at(t)
        costs.append((c_fp * fp + c_fn * fn) / len(YTE))
        precs.append(prec); recs.append(rec)
    costs = np.array(costs); precs = np.array(precs); recs = np.array(recs)
    best = float(thrs[int(np.argmin(costs))])
    theo = c_fp / (c_fp + c_fn)
    cost_at_half = float(costs[np.argmin(np.abs(thrs - 0.5))])
    cost_best = float(costs.min())
    FACTS["thr_best"] = best; FACTS["thr_theory"] = theo
    FACTS["cost_half"] = cost_at_half; FACTS["cost_best"] = cost_best
    FACTS["cost_drop"] = cost_at_half - cost_best
    assert abs(theo - 0.0909) < 1e-3
    assert 0.03 < best < 0.20, best
    assert cost_best < cost_at_half, (cost_best, cost_at_half)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))
    ax = axes[0]
    ax.plot(thrs, costs, color=RED, lw=2.4)
    ax.axvline(theo, color=GREEN, lw=1.8, ls=(0, (4, 3)),
               label=f"теория $p_*={theo:.3f}$")
    ax.axvline(best, color=GOLD, lw=1.8, ls=(0, (2, 2)),
               label=f"минимум на тесте {best:.2f}")
    ax.axvline(0.5, color=MUTED, lw=1.4, ls=(0, (1, 3)), label="привычные 0,5")
    ax.set_xlabel("порог вероятности $p_*$")
    ax.set_ylabel("средние потери на письмо")
    ax.set_title("Цена ошибок 1 : 10 сдвигает порог влево")
    ax.legend(frameon=False, fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)

    ax = axes[1]
    ax.plot(recs, precs, color=BLUE, lw=2.4)
    for t, c in [(0.2, GREEN), (0.5, INK), (0.8, GOLD)]:
        tp, fp, fn, tn, prec, rec = counts_at(t)
        ax.scatter([rec], [prec], s=60, color=c, zorder=5)
        ax.text(rec - 0.02, prec + 0.02, f"$p_*={t}$", color=c, fontsize=10, ha="right")
    ax.axhline(float(YTE.mean()), color=MUTED, lw=1.2, ls=(0, (3, 3)))
    ax.text(0.02, float(YTE.mean()) + 0.02, "доля спама (случайный выбор)", color=MUTED, fontsize=9)
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title("Одна кривая — все пороги")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "cost_curve.png")


# --------------------------------------------------- fig 53.5: calibration
def fig_calibration():
    texts_tr = [TEXTS[i] for i in itr]; texts_te = [TEXTS[i] for i in ite]
    vec = CountVectorizer(min_df=2)
    Atr = vec.fit_transform(texts_tr); Ate = vec.transform(texts_te)
    nb = MultinomialNB().fit(Atr, YTR)
    p_nb = nb.predict_proba(Ate)[:, 1]
    p_lr = P_TE

    def reliability(p, k=10, nmin=10):
        edges = np.linspace(0, 1, k + 1)
        xs, ys = [], []
        for i in range(k):
            m = (p >= edges[i]) & (p < edges[i + 1] if i < k - 1 else p <= 1.0)
            if m.sum() >= nmin:
                xs.append(float(p[m].mean())); ys.append(float(YTE[m].mean()))
        return np.array(xs), np.array(ys)

    def ece(p, k=10):
        edges = np.linspace(0, 1, k + 1); tot = 0.0
        for i in range(k):
            m = (p >= edges[i]) & (p < edges[i + 1] if i < k - 1 else p <= 1.0)
            if m.sum():
                tot += m.sum() * abs(p[m].mean() - YTE[m].mean())
        return float(tot / len(p))

    xs_lr, ys_lr = reliability(p_lr)
    xs_nb, ys_nb = reliability(p_nb)
    ece_lr, ece_nb = ece(p_lr), ece(p_nb)
    band_lr = float(np.mean((p_lr > 0.02) & (p_lr < 0.98)))
    band_nb = float(np.mean((p_nb > 0.02) & (p_nb < 0.98)))
    mid = (p_lr > 0.4) & (p_lr < 0.6)
    mid_n = int(mid.sum()); mid_obs = float(YTE[mid].mean())
    auc_nb = float(roc_auc_score(YTE, p_nb))

    def rec_at(p, t):
        pred = p >= t
        return float(np.sum(pred & (YTE == 1)) / np.sum(YTE == 1))

    nb_rec_lo, nb_rec_hi = rec_at(p_nb, 0.1), rec_at(p_nb, 0.9)
    lr_rec_lo, lr_rec_hi = rec_at(p_lr, 0.1), rec_at(p_lr, 0.9)
    FACTS.update({
        "ece_lr": ece_lr, "ece_nb": ece_nb, "band_lr": band_lr, "band_nb": band_nb,
        "mid_n": mid_n, "mid_obs": mid_obs, "auc_nb": auc_nb,
        "nb_rec_01": nb_rec_lo, "nb_rec_09": nb_rec_hi,
        "lr_rec_01": lr_rec_lo, "lr_rec_09": lr_rec_hi,
    })
    assert 0.06 < ece_lr < 0.08 and ece_nb < 0.02, (ece_lr, ece_nb)
    assert band_lr > 0.99 and band_nb < 0.08, (band_lr, band_nb)
    assert mid_n == 68 and 0.88 < mid_obs < 0.94, (mid_n, mid_obs)
    assert auc_nb > AUC
    assert nb_rec_hi > 0.85 and lr_rec_hi < 0.10, (nb_rec_hi, lr_rec_hi)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.4, ls=(0, (3, 3)), label="идеальная калибровка")
    ax.plot(xs_lr, ys_lr, "o-", color=BLUE, lw=2.2, ms=6,
            label=f"логрег, 2 признака (ECE {ece_lr:.3f})")
    ax.plot(xs_nb, ys_nb, "s-", color=RED, lw=2.2, ms=6,
            label=f"наивный Байес по словам (ECE {ece_nb:.3f})")
    ax.set_xlabel("объявленная вероятность"); ax.set_ylabel("наблюдённая доля спама")
    ax.set_title("Кривая надёжности: обещание против факта")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)

    ax = axes[1]
    ax.hist(p_nb, bins=np.linspace(0, 1, 26), color=RED, alpha=0.6, label="наивный Байес")
    ax.hist(p_lr, bins=np.linspace(0, 1, 26), color=BLUE, alpha=0.55, label="логрег")
    ax.set_yscale("log")
    ax.set_xlabel("объявленная вероятность спама"); ax.set_ylabel("число писем (лог)")
    ax.set_title(f"В середине шкалы у Байеса лишь {band_nb*100:.1f}% писем")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "calibration.png")


# ------------------------------------- fig 53.6: feature map bends the boundary
def fig_lift():
    rng = np.random.default_rng(53)
    n = 220
    r_in = rng.uniform(0.2, 1.0, n); r_out = rng.uniform(1.8, 2.8, n)
    a_in = rng.uniform(0, 2 * np.pi, n); a_out = rng.uniform(0, 2 * np.pi, n)
    Xi = np.c_[r_in * np.cos(a_in), r_in * np.sin(a_in)]
    Xo = np.c_[r_out * np.cos(a_out), r_out * np.sin(a_out)]
    X = np.vstack([Xi, Xo]); y = np.r_[np.zeros(n), np.ones(n)]
    lin = LogisticRegression(max_iter=1000).fit(X, y)
    acc_lin = float(accuracy_score(y, lin.predict(X)))
    Z = np.c_[X, (X ** 2).sum(1)]
    lift = LogisticRegression(max_iter=1000).fit(Z, y)
    acc_lift = float(accuracy_score(y, lift.predict(Z)))
    FACTS["acc_rings_linear"] = acc_lin; FACTS["acc_rings_lifted"] = acc_lift
    FACTS["n_rings"] = int(X.shape[0])
    assert FACTS["n_rings"] == 440, FACTS["n_rings"]
    assert acc_lin < 0.62 and acc_lift == 1.0, (acc_lin, acc_lift)
    r_cut = float((1.0 + 1.8) / 2)

    fig = plt.figure(figsize=(10.6, 4.8))
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(Xi[:, 0], Xi[:, 1], s=14, color=BLUE, alpha=0.7, label="класс 0")
    ax.scatter(Xo[:, 0], Xo[:, 1], s=14, color=RED, alpha=0.6, label="класс 1")
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(r_cut * np.cos(th), r_cut * np.sin(th), color=INK, lw=2.2)
    xs = np.linspace(-3, 3, 10)
    ax.plot(xs, -(lin.coef_[0][0] * xs + lin.intercept_[0]) / lin.coef_[0][1],
            color=GOLD, lw=2.0, ls=(0, (4, 3)), label=f"лучшая прямая: {acc_lin*100:.0f}%")
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal")
    ax.set_title("В плоскости прямая бессильна")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.scatter(Xi[:, 0], Xi[:, 1], (Xi ** 2).sum(1), s=12, color=BLUE, alpha=0.8)
    ax.scatter(Xo[:, 0], Xo[:, 1], (Xo ** 2).sum(1), s=12, color=RED, alpha=0.6)
    gx, gy = np.meshgrid(np.linspace(-3, 3, 2), np.linspace(-3, 3, 2))
    ax.plot_surface(gx, gy, np.full_like(gx, r_cut ** 2), color=GREEN, alpha=0.25)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_zlabel("$\\phi_3=x_1^2+x_2^2$")
    ax.set_title(f"После подъёма плоскость разделяет: {acc_lift*100:.0f}%")
    ax.set_facecolor(PAPER)
    ax.view_init(elev=16, azim=-58)
    fig.tight_layout()
    save(fig, OUT / "lift_circle.png")


# ------------------------------------------- fig 53.7: multiclass softmax, iris
def fig_multiclass():
    data = load_iris()
    X = data.data[:, 2:4]; y = data.target
    clf = LogisticRegression(max_iter=1000).fit(X, y)
    acc = float(accuracy_score(y, clf.predict(X)))
    FACTS["iris_acc"] = acc
    assert acc > 0.94, acc
    gx, gy = np.meshgrid(np.linspace(0.5, 7.5, 400), np.linspace(-0.2, 3.0, 400))
    G = np.c_[gx.ravel(), gy.ravel()]
    pred = clf.predict(G).reshape(gx.shape)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.contourf(gx, gy, pred, levels=[-0.5, 0.5, 1.5, 2.5],
                colors=["#e9eef5", "#fdf2ee", "#eaf2ee"])
    ax.contour(gx, gy, pred, levels=[0.5, 1.5], colors=[INK], linewidths=2.0)
    names = ["setosa", "versicolor", "virginica"]
    for k, c in enumerate([BLUE, RED, GREEN]):
        m = y == k
        ax.scatter(X[m, 0], X[m, 1], s=26, color=c, alpha=0.8, label=names[k])
    ax.set_xlabel("длина лепестка, см"); ax.set_ylabel("ширина лепестка, см")
    ax.set_title(f"Три score, argmax и прямые границы (accuracy {acc*100:.1f}%)")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "multiclass_iris.png")


# ------------------------------------------------------------ sidenote images
def side_odds():
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    s = np.linspace(-4, 4, 300)
    ax.plot(s, 1 / (1 + np.exp(-s)), color=VIOLET, lw=2.4)
    for v, c in [(-1, BLUE), (0, INK), (1, GOLD), (2, RED)]:
        p = 1 / (1 + np.exp(-v))
        ax.plot([v, v], [0, p], color=c, lw=1.2, ls=(0, (3, 3)))
        ax.scatter([v], [p], s=32, color=c, zorder=5)
        ax.text(v, p + 0.04, f"{p:.2f}", color=c, fontsize=9, ha="center")
    ax.set_xlabel("score $s$"); ax.set_ylabel("$\\sigma(s)$")
    ax.set_title("шаг +1 умножает шансы на $e$", fontsize=11.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "odds.png")
    FACTS["sigma1"] = float(1 / (1 + np.exp(-1)))
    FACTS["sigma2"] = float(1 / (1 + np.exp(-2)))
    assert abs(FACTS["sigma1"] - 0.7311) < 1e-3 and abs(FACTS["sigma2"] - 0.8808) < 1e-3
    # inline exercise: p=0,9 -> s=ln 9; the same increment again -> sigma(2 ln 9)
    FACTS["logit09"] = float(np.log(9.0))
    FACTS["dx_digits_for_09"] = float(np.log(9.0) / 19.495051505221575)
    FACTS["sigma_2logit09"] = float(1 / (1 + np.exp(-2 * np.log(9.0))))
    assert abs(FACTS["logit09"] - 2.20) < 0.01, FACTS["logit09"]
    assert abs(FACTS["dx_digits_for_09"] - 0.113) < 0.001, FACTS["dx_digits_for_09"]
    assert abs(FACTS["sigma_2logit09"] - 0.988) < 5e-4, FACTS["sigma_2logit09"]


def side_scaling():
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    s = np.linspace(-4, 4, 300)
    for k, c, lab in [(0.5, BLUE, "$0{,}5w$"), (1.0, INK, "$w$"), (4.0, RED, "$4w$")]:
        ax.plot(s, 1 / (1 + np.exp(-k * s)), color=c, lw=2.2, label=lab)
    ax.axvline(0, color=GOLD, lw=1.6, ls=(0, (3, 3)))
    ax.text(0.1, 0.06, "граница не двигается", color=GOLD, fontsize=9)
    ax.set_xlabel("$w^\\top x+b$ при $k=1$"); ax.set_ylabel("вероятность")
    ax.set_title("масштаб весов = крутизна", fontsize=11.5)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "scaling.png")


def side_confusion():
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.9))
    for ax, thr in zip(axes, [0.5, 0.1]):
        tp, fp, fn, tn, prec, rec = counts_at(thr)
        M = np.array([[tn, fp], [fn, tp]], float)
        ax.imshow(np.log1p(M), cmap="Blues", alpha=0.55)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(M[i, j])}", ha="center", va="center", color=INK, fontsize=11)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["не спам", "спам"], fontsize=9)
        ax.set_yticklabels(["ham", "спам"], fontsize=9)
        ax.set_xlabel("решение", fontsize=9.5); ax.set_ylabel("истина", fontsize=9.5)
        ax.set_title(f"$p_*={thr}$: recall {rec*100:.0f}%", fontsize=10.5)
        key = str(thr).replace(".", "")
        FACTS[f"rec_{key}"] = rec; FACTS[f"prec_{key}"] = prec
        FACTS[f"fp_{key}"] = fp; FACTS[f"fn_{key}"] = fn; FACTS[f"tp_{key}"] = tp
    fig.tight_layout()
    save(fig, SIDE / "confusion.png")
    assert FACTS["fn_01"] < FACTS["fn_05"] and FACTS["fp_01"] > FACTS["fp_05"]


def side_margin():
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    xs = np.linspace(-1, 4, 10)
    ax.plot(xs, 2.2 - 0.6 * xs, color=INK, lw=2.2)
    for (px, py), c, eps in [((1.0, 1.0), GREEN, 0.55), ((2.4, 0.9), RED, 0.55)]:
        ax.scatter([px], [py], s=60, color=c, zorder=6)
        th = np.linspace(0, 2 * np.pi, 120)
        ax.plot(px + eps * np.cos(th), py + eps * np.sin(th), color=c, lw=1.4, ls=(0, (3, 3)))
    ax.text(0.2, 0.3, "запас > $\\varepsilon$:\nрешение устойчиво", color=GREEN, fontsize=9)
    ax.text(2.1, 1.7, "шар пересёк\nграницу", color=RED, fontsize=9)
    ax.set_xlim(-0.5, 4); ax.set_ylim(-0.2, 3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("запас = страховка от шума", fontsize=11.5)
    save(fig, SIDE / "margin.png")


def main():
    register_basics()
    fig_plane(); fig_score_to_prob(); fig_thresholds(); fig_cost()
    fig_calibration(); fig_lift(); fig_multiclass()
    side_odds(); side_scaling(); side_confusion(); side_margin()
    print("--- lesson 53 facts ---")
    for k, v in FACTS.items():
        print(f"{k:20s} {v}")


if __name__ == "__main__":
    main()
