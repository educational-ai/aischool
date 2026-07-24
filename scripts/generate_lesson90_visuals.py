"""Deterministic figures for lesson 90: the capstone project as a protocol.

Everything is computed on the REAL UCI Bike Sharing hourly file
(scripts/data/bike-sharing-hour.csv, 17 379 rows, 2011-01-01..2012-12-31):

  * a ladder of baselines (constant / hour-by-daytype mean / lag-168 / gradient boosting);
  * two protocols -- temporal split vs random shuffle -- reported error vs the error on a
    genuinely future block (October-December 2012);
  * a feature leak (casual + registered == cnt) that no split can catch;
  * error slices by hour of day and weather;
  * threshold choice for a "peak hour" alarm (precision/recall);
  * a learning curve and a paired day-level bootstrap interval.

Every number quoted in content/lessons/90.md is produced and asserted here.
Also dumps scripts/data/lesson90_facts.json for the capstone-protocol-lab widget.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
BIKE = ROOT / "scripts" / "data" / "bike-sharing-hour.csv"
OUT = ROOT / "public" / "figures" / "lessons" / "90"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "90"
FACTS = ROOT / "scripts" / "data" / "lesson90_facts.json"

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

MAX_ITER = 300
SEED = 2026
HONEST = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday",
          "weathersit", "temp", "atemp", "hum", "windspeed"]


def nf(v, d=1):
    return f"{v:.{d}f}".replace(".", ",")


def save(fig, path, *, dpi=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------- data
def load():
    rows = list(csv.DictReader(open(BIKE)))
    date = np.array([dt.date.fromisoformat(r["dteday"]) for r in rows])
    cnt = np.array([int(r["cnt"]) for r in rows], float)
    casual = np.array([int(r["casual"]) for r in rows], float)
    registered = np.array([int(r["registered"]) for r in rows], float)
    hr = np.array([int(r["hr"]) for r in rows])
    workday = np.array([int(r["workingday"]) for r in rows])
    weather = np.array([int(r["weathersit"]) for r in rows])
    X = np.array([[float(r[c]) for c in HONEST] for r in rows])
    XL = np.column_stack([X, casual, registered])
    return dict(rows=rows, date=date, cnt=cnt, casual=casual, registered=registered,
                hr=hr, workday=workday, weather=weather, X=X, XL=XL)


def fit(X, cnt, idx, *, max_iter=MAX_ITER):
    model = HistGradientBoostingRegressor(random_state=SEED, max_iter=max_iter,
                                          early_stopping=False)
    model.fit(X[idx], cnt[idx])
    return model


D = load()
cnt = D["cnt"]; date = D["date"]; X = D["X"]; XL = D["XL"]
N = len(cnt)
assert N == 17379, N
assert date.min() == dt.date(2011, 1, 1) and date.max() == dt.date(2012, 12, 31)

MEAN_CNT = cnt.mean(); MEDIAN_CNT = float(np.median(cnt)); MAX_CNT = cnt.max()
print(f"rows={N} mean={MEAN_CNT:.1f} median={MEDIAN_CNT:.0f} max={MAX_CNT:.0f}")
assert abs(MEAN_CNT - 189.5) < 0.05 and MEDIAN_CNT == 142.0 and MAX_CNT == 977

LEAK_EXACT = bool(np.all(D["casual"] + D["registered"] == cnt))
assert LEAK_EXACT

TR = np.where(date <= dt.date(2012, 6, 30))[0]
VA = np.where((date >= dt.date(2012, 7, 1)) & (date <= dt.date(2012, 9, 30)))[0]
TE = np.where(date >= dt.date(2012, 10, 1))[0]
POOL = np.concatenate([TR, VA])
print(f"train={len(TR)} val={len(VA)} test={len(TE)}")
assert (len(TR), len(VA), len(TE)) == (13003, 2208, 2168)
TEST_DAYS = sorted(set(date[TE]))
assert len(TEST_DAYS) == 92

y_te = cnt[TE]
hr_te = D["hr"][TE]; wk_te = D["workday"][TE]; we_te = D["weather"][TE]

# ---------------------------------------------------------------- models
model_honest = fit(X, cnt, TR)
pred_val = model_honest.predict(X[VA])
pred_te = model_honest.predict(X[TE])
MAE_VAL = mean_absolute_error(cnt[VA], pred_val)
MAE_TEST = mean_absolute_error(y_te, pred_te)
R2_TEST = r2_score(y_te, pred_te)
print(f"honest: val MAE={MAE_VAL:.1f} test MAE={MAE_TEST:.1f} R2={R2_TEST:.3f}")
assert abs(MAE_VAL - 40.4) < 0.05 and abs(MAE_TEST - 45.5) < 0.3
assert abs(R2_TEST - 0.878) < 0.0005

model_leaky = fit(XL, cnt, TR)
pred_leaky = model_leaky.predict(XL[TE])
MAE_LEAK = mean_absolute_error(y_te, pred_leaky)
R2_LEAK = r2_score(y_te, pred_leaky)
print(f"leaky: test MAE={MAE_LEAK:.2f} R2={R2_LEAK:.4f}")
assert abs(MAE_LEAK - 3.4489) < 5e-05 and R2_LEAK > 0.99

# baselines
MED_TR = float(np.median(cnt[TR]))
MAE_CONST = mean_absolute_error(y_te, np.full(len(TE), MED_TR))
hour_mean = {}
for h in range(24):
    for w in (0, 1):
        m = (D["hr"][TR] == h) & (D["workday"][TR] == w)
        hour_mean[(h, w)] = cnt[TR][m].mean()
pred_hour = np.array([hour_mean[(hr_te[i], wk_te[i])] for i in range(len(TE))])
MAE_HOUR = mean_absolute_error(y_te, pred_hour)

pos = {(date[i], D["hr"][i]): i for i in range(N)}
lag_ok, lag_val = [], []
for i in TE:
    j = pos.get((date[i] - dt.timedelta(days=7), D["hr"][i]))
    lag_ok.append(j is not None)
    lag_val.append(cnt[j] if j is not None else np.nan)
lag_ok = np.array(lag_ok); lag_val = np.array(lag_val, float)
LAG_COVER = lag_ok.mean()
MAE_LAG = mean_absolute_error(y_te[lag_ok], lag_val[lag_ok])
print(f"const={MAE_CONST:.1f} hour={MAE_HOUR:.1f} lag168={MAE_LAG:.1f} cover={LAG_COVER:.3f}")
assert abs(MAE_CONST - 161.1) < 0.05 and abs(MAE_HOUR - 88.5) < 0.3
assert abs(MAE_LAG - 70.6) < 0.05 and abs(LAG_COVER - 0.982) < 0.003
print(f"median train target = {MED_TR:.0f}")
assert MED_TR == 124.0

# ---------------------------------------------------------------- paired bootstrap by days
err_model = np.abs(y_te - pred_te)
err_base = np.abs(y_te - pred_hour)
DELTA = err_base.mean() - err_model.mean()
day_of = date[TE]
by_day = {d: np.where(day_of == d)[0] for d in TEST_DAYS}
rng = np.random.default_rng(9090)
reps = np.empty(2000)
for b in range(2000):
    pick = rng.integers(0, len(TEST_DAYS), len(TEST_DAYS))
    ii = np.concatenate([by_day[TEST_DAYS[k]] for k in pick])
    reps[b] = err_base[ii].mean() - err_model[ii].mean()
CI_LO, CI_HI = np.quantile(reps, [0.025, 0.975])
print(f"delta={DELTA:.1f} CI=[{CI_LO:.1f}, {CI_HI:.1f}]")
assert abs(DELTA - 43.0) < 0.05 and abs(CI_LO - 35.8) < 0.6 and abs(CI_HI - 50.4) < 0.6

# ---------------------------------------------------------------- protocol grid (widget + fig 2)
FRACS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
rng_split = np.random.default_rng(9090)
perm = rng_split.permutation(POOL)
RTR, RVA = perm[:len(TR)], perm[len(TR):]
sub = np.random.default_rng(4242)
grid = {}
for feat, Xa in (("honest", X), ("leaky", XL)):
    for split in ("time", "random"):
        reported, future = [], []
        base = TR if split == "time" else RTR
        vi = VA if split == "time" else RVA
        for f in FRACS:
            k = int(len(base) * f)
            idx = base[-k:] if split == "time" else np.sort(sub.choice(base, k, replace=False))
            g = fit(Xa, cnt, idx)
            reported.append(round(float(mean_absolute_error(cnt[vi], g.predict(Xa[vi]))), 1))
            future.append(round(float(mean_absolute_error(y_te, g.predict(Xa[TE]))), 1))
        grid[f"{split}-{feat}"] = {"reported": reported, "future": future}
        print(f"{split}-{feat}: reported={reported} future={future}")

REP_TIME = grid["time-honest"]["reported"][-1]; FUT_TIME = grid["time-honest"]["future"][-1]
REP_RAND = grid["random-honest"]["reported"][-1]; FUT_RAND = grid["random-honest"]["future"][-1]
assert abs(REP_TIME - 40.4) < 0.05 and abs(FUT_TIME - 45.5) < 0.5
assert abs(REP_RAND - 23.1) < 0.05 and abs(FUT_RAND - 43.1) < 1.2
RATIO_RAND = FUT_RAND / REP_RAND
RATIO_TIME = FUT_TIME / REP_TIME
print(f"optimism: random x{RATIO_RAND:.2f}, temporal x{RATIO_TIME:.2f}")
assert 1.7 < RATIO_RAND < 1.9 and 1.0 < RATIO_TIME < 1.2

# ---------------------------------------------------------------- slices
SLICES_HOUR = []
for lo, hi, name in ((0, 5, "ночь 0–5"), (6, 11, "утро 6–11"),
                     (12, 17, "день 12–17"), (18, 23, "вечер 18–23")):
    m = (hr_te >= lo) & (hr_te <= hi)
    SLICES_HOUR.append((name, err_model[m].mean(), y_te[m].mean(), int(m.sum())))
    print(f"{name}: MAE={err_model[m].mean():.1f} mean={y_te[m].mean():.0f} n={m.sum()}")
assert abs(SLICES_HOUR[0][1] - 11.9) < 0.05 and abs(SLICES_HOUR[2][1] - 67.4) < 0.4
assert abs(SLICES_HOUR[1][1] - 53.8) < 0.05 and abs(SLICES_HOUR[3][1] - 48.5) < 0.4

SLICES_W = []
for w, name in ((1, "ясно"), (2, "туман/облачно"), (3, "дождь/снег")):
    m = we_te == w
    SLICES_W.append((name, err_model[m].mean(), int(m.sum())))
    print(f"weather {name}: MAE={err_model[m].mean():.1f} n={m.sum()}")
assert abs(SLICES_W[0][1] - 42.7) < 0.05 and abs(SLICES_W[2][1] - 72.0) < 0.6
assert SLICES_W[2][2] == 160

MAE_WORK = err_model[wk_te == 1].mean(); MAE_FREE = err_model[wk_te == 0].mean()
print(f"workday MAE={MAE_WORK:.1f} weekend MAE={MAE_FREE:.1f}")
assert abs(MAE_WORK - MAE_FREE) < 0.5

# ---------------------------------------------------------------- alarm thresholds
PEAK = 500.0
label = y_te > PEAK
BASE_RATE = label.mean()
print(f"base rate peak>{PEAK:.0f}: {BASE_RATE:.3f}")
assert abs(BASE_RATE - 0.108) < 0.0005
THRESH = []
for t in (300, 400, 500, 600):
    alarm = pred_te > t
    tp = int((alarm & label).sum())
    prec = tp / max(int(alarm.sum()), 1)
    rec = tp / int(label.sum())
    THRESH.append((t, prec, rec, int(alarm.sum())))
    print(f"threshold {t}: precision={prec:.3f} recall={rec:.3f} alarms={alarm.sum()}")
assert abs(THRESH[1][1] - 0.723) < 0.0005 and abs(THRESH[1][2] - 0.923) < 0.02
assert abs(THRESH[2][1] - 0.938) < 0.0005 and abs(THRESH[2][2] - 0.643) < 0.02
assert THRESH[3][1] == 1.0

# ---------------------------------------------------------------- learning curve
LC = []
lc_rng = np.random.default_rng(4242)
for f in FRACS:
    k = int(len(TR) * f)
    idx = np.sort(lc_rng.choice(TR, k, replace=False))
    g = fit(X, cnt, idx)
    LC.append((k, float(mean_absolute_error(y_te, g.predict(X[TE])))))
    print(f"learning curve n={k}: MAE={LC[-1][1]:.1f}")
assert abs(LC[0][1] - 59.9) < 0.05 and abs(LC[-1][1] - MAE_TEST) < 0.3
assert abs(LC[3][1] - 43.7) < 0.05 and LC[3][1] < LC[-1][1]


# ================================================================ figures
def fig1_ladder():
    names = ["константа\n(медиана train)", "среднее по\nчасу × типу дня",
             "тот же час\nнеделю назад", "градиентный\nбустинг"]
    vals = [MAE_CONST, MAE_HOUR, MAE_LAG, MAE_TEST]
    colors = [FAINT, GOLD, GREEN, BLUE]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bars = ax.barh(range(4), vals, color=colors, height=0.58)
    for i, (b, v) in enumerate(zip(bars, vals)):
        ax.text(v + 2.5, i, nf(v), va="center", color=INK, fontsize=12)
    ax.set_yticks(range(4)); ax.set_yticklabels(names, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("MAE на тесте октябрь–декабрь 2012, поездок в час")
    ax.set_xlim(0, 185)
    ax.set_title("Лестница baseline: сколько добавляет каждая ступень")
    ax.axvline(MAE_TEST, color=BLUE, lw=0.9, ls=(0, (4, 3)), alpha=0.7)
    ax.annotate("выигрыш у часового baseline\n" + nf(DELTA, 0) + " поездок/час, 95% ДИ [" + nf(CI_LO, 0) + "; " + nf(CI_HI, 0) + "]",
                xy=(MAE_HOUR, 1), xytext=(105, 2.55), color=MUTED, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=LINE, lw=1.0))
    ax.grid(True, axis="x", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, OUT / "baseline_ladder.png")


def fig2_protocol():
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    labels = ["случайное перемешивание", "разбиение по времени"]
    rep = [REP_RAND, REP_TIME]; fut = [FUT_RAND, FUT_TIME]
    xs = np.arange(2); w = 0.32
    ax.bar(xs - w / 2, rep, w, color=GOLD, label="ошибка, о которой отчитались")
    ax.bar(xs + w / 2, fut, w, color=BLUE, label="ошибка на будущем блоке (окт–дек)")
    for x, v in zip(xs - w / 2, rep):
        ax.text(x, v + 1.2, nf(v), ha="center", color=INK, fontsize=12)
    for x, v in zip(xs + w / 2, fut):
        ax.text(x, v + 1.2, nf(v), ha="center", color=INK, fontsize=12)
    ax.annotate("", xy=(-w / 2, REP_RAND + 6), xytext=(w / 2, FUT_RAND + 6),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.4))
    ax.text(0, FUT_RAND + 9, "разрыв ×" + nf(RATIO_RAND, 2), ha="center", color=RED, fontsize=12)
    ax.text(1, FUT_TIME + 9, "разрыв ×" + nf(RATIO_TIME, 2), ha="center", color=GREEN, fontsize=12)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("MAE, поездок в час"); ax.set_ylim(0, 62)
    ax.set_title("Протокол решает, во что вы поверите")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, OUT / "protocol_gap.png")


def fig3_leak():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.6),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    step = 7
    ax1.scatter(D["casual"][::step] + D["registered"][::step], cnt[::step], s=6,
                color=RED, alpha=0.35, linewidths=0)
    ax1.plot([0, 1000], [0, 1000], color=INK, lw=1.0, ls=(0, (4, 3)))
    ax1.set_xlabel("casual + registered"); ax1.set_ylabel("cnt (target)")
    ax1.set_title("Тождество, а не признак", fontsize=13)
    ax1.text(40, 880, "совпадает во всех\n17 379 строках", color=MUTED, fontsize=10)
    ax1.grid(True, color=GRID, lw=0.4, alpha=0.5); ax1.set_axisbelow(True)

    ax2.bar([0, 1], [MAE_LEAK, MAE_TEST], 0.5, color=[RED, BLUE])
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["с casual\nи registered", "только честные\nпризнаки"], fontsize=11)
    ax2.set_ylabel("MAE на тесте, поездок в час")
    for x, v in zip([0, 1], [MAE_LEAK, MAE_TEST]):
        ax2.text(x, v + 1.2, nf(v), ha="center", color=INK, fontsize=12)
    ax2.set_ylim(0, 56)
    ax2.set_title("«Слишком хорошо» — это симптом", fontsize=13)
    ax2.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax2.set_axisbelow(True)
    for ax in (ax1, ax2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Утечка признака не ловится никаким разбиением", fontsize=15, y=1.02)
    save(fig, OUT / "leak.png")


def fig4_slices():
    per_hour = np.array([err_model[hr_te == h].mean() for h in range(24)])
    mean_hour = np.array([y_te[hr_te == h].mean() for h in range(24)])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.6),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    ax1.bar(range(24), per_hour, color=BLUE, width=0.72, label="MAE модели")
    ax1.plot(range(24), mean_hour * 0.2, color=GOLD, lw=1.8,
             label="20% среднего спроса этого часа")
    ax1.set_xticks(range(0, 24, 3)); ax1.set_xlabel("час суток")
    ax1.set_ylabel("MAE, поездок в час")
    ax1.set_title("Средняя ошибка 45,5 — это среднее по очень разным часам", fontsize=13)
    ax1.legend(frameon=False, fontsize=10, loc="upper left")
    ax1.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax1.set_axisbelow(True)

    names = [s[0] for s in SLICES_W]; vals = [s[1] for s in SLICES_W]; ns = [s[2] for s in SLICES_W]
    ax2.bar(range(3), vals, 0.55, color=[GREEN, GOLD, RED])
    for i, (v, n) in enumerate(zip(vals, ns)):
        ax2.text(i, v + 1.2, nf(v), ha="center", color=INK, fontsize=12)
        ax2.text(i, 2.5, f"n={n}", ha="center", color=PAPER, fontsize=10)
    ax2.set_xticks(range(3)); ax2.set_xticklabels(names, fontsize=11)
    ax2.set_ylim(0, 84); ax2.set_ylabel("MAE, поездок в час")
    ax2.set_title("Погода: редкий режим — худший", fontsize=13)
    ax2.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax2.set_axisbelow(True)
    for ax in (ax1, ax2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    save(fig, OUT / "slices.png")


def fig5_threshold():
    ts = np.linspace(50, 750, 200)
    prec, rec = [], []
    for t in ts:
        alarm = pred_te > t
        tp = (alarm & label).sum()
        prec.append(tp / max(alarm.sum(), 1))
        rec.append(tp / label.sum())
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(ts, prec, color=BLUE, lw=2.2, label="точность (precision)")
    ax.plot(ts, rec, color=RED, lw=2.2, label="полнота (recall)")
    ax.axhline(BASE_RATE, color=MUTED, lw=1.0, ls=(0, (4, 3)))
    ax.text(58, BASE_RATE + 0.02, "доля пиковых часов " + nf(BASE_RATE, 3), color=MUTED, fontsize=10)
    for t, p, r, n in THRESH:
        ax.plot([t, t], [0, max(p, r)], color=LINE, lw=0.8)
        ax.scatter([t, t], [p, r], s=28, color=[BLUE, RED], zorder=5)
        ax.text(t, 1.03, f"{t}\n{n} тревог", ha="center", color=MUTED, fontsize=9)
    ax.set_xlabel("порог тревоги по прогнозу, поездок в час")
    ax.set_ylabel("доля"); ax.set_ylim(0, 1.16); ax.set_xlim(50, 750)
    ax.set_title("Метрика продолжается в интерфейс: цена порога")
    ax.legend(frameon=False, fontsize=10, loc="center left")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.45); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, OUT / "threshold.png")


def fig6_learning():
    ns = [p[0] for p in LC]; vals = [p[1] for p in LC]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(ns, vals, color=BLUE, lw=2.2, marker="o", ms=6)
    for n, v in zip(ns, vals):
        ax.text(n, v + 1.3, nf(v), ha="center", color=MUTED, fontsize=10)
    ax.axhline(MAE_HOUR, color=GOLD, lw=1.4, ls=(0, (5, 3)))
    ax.text(1400, MAE_HOUR + 1.5, "часовой baseline " + nf(MAE_HOUR), color=GOLD, fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel("размер обучающей выборки, часов (лог-шкала)")
    ax.set_ylabel("MAE на тесте окт–дек")
    ax.set_ylim(35, 96)
    ax.set_title("Кривая обучения: где кончается польза новых данных")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.45); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, OUT / "learning_curve.png")


# ---------------------------------------------------------------- sidenote figures
def side_calendar():
    fig, ax = plt.subplots(figsize=(4.6, 2.3))
    blocks = [(dt.date(2011, 1, 1), dt.date(2012, 6, 30), "train", BLUE),
              (dt.date(2012, 7, 1), dt.date(2012, 9, 30), "val", GOLD),
              (dt.date(2012, 10, 1), dt.date(2012, 12, 31), "test", RED)]
    t0 = dt.date(2011, 1, 1)
    for a, b, name, c in blocks:
        x0 = (a - t0).days; w = (b - a).days + 1
        ax.barh(0, w, left=x0, height=0.5, color=c)
        ax.text(x0 + w / 2, 0, name, ha="center", va="center", color=PAPER, fontsize=11)
    ax.set_xlim(0, 731); ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([]); ax.set_xticks([0, 365, 731])
    ax.set_xticklabels(["01.2011", "01.2012", "01.2013"], fontsize=9)
    ax.set_title("13 003 / 2 208 / 2 168 часов", fontsize=11)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    save(fig, SIDE / "split.png")


def side_bootstrap():
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.hist(reps, bins=36, color=WASH, edgecolor=LINE, lw=0.6)
    ax.axvline(CI_LO, color=RED, lw=1.4); ax.axvline(CI_HI, color=RED, lw=1.4)
    ax.axvline(0, color=INK, lw=1.0, ls=(0, (3, 3)))
    ax.set_xlabel("выигрыш у baseline, поездок/час", fontsize=10)
    ax.set_yticks([])
    ax.set_title("2000 реплик по дням\n95%: [" + nf(CI_LO) + "; " + nf(CI_HI) + "]", fontsize=11)
    ax.set_xlim(-2, 58)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    save(fig, SIDE / "bootstrap.png")


def side_skew():
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.hist(cnt, bins=50, color=WASH, edgecolor=LINE, lw=0.5)
    ax.axvline(MEDIAN_CNT, color=BLUE, lw=1.6)
    ax.axvline(MEAN_CNT, color=RED, lw=1.6)
    ax.text(MEDIAN_CNT + 15, 2400, "медиана " + nf(MEDIAN_CNT, 0), color=BLUE, fontsize=10)
    ax.text(MEAN_CNT + 15, 1900, "среднее " + nf(MEAN_CNT, 0), color=RED, fontsize=10)
    ax.set_xlabel("поездок в час", fontsize=10); ax.set_yticks([])
    ax.set_title("Скошенный target", fontsize=11)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    save(fig, SIDE / "skew.png")


def side_alarm():
    t, prec, rec, nal = THRESH[1]
    tp = int(round(prec * nal)); fp = nal - tp
    fn = int(label.sum()) - tp
    tn = len(TE) - tp - fp - fn
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    M = np.array([[tp, fp], [fn, tn]], float)
    ax.imshow(np.log1p(M), cmap="Blues", alpha=0.55)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(M[i, j])}", ha="center", va="center", color=INK, fontsize=13)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["пик", "не пик"], fontsize=10)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["тревога", "тишина"], fontsize=10)
    assert tp == 217 and fp == 83 and fn == 18 and tn == 1850, (tp, fp, fn, tn)
    ax.set_title(f"Порог {t}: точность " + nf(prec, 2) + ", полнота " + nf(rec, 2), fontsize=11)
    save(fig, SIDE / "confusion.png")


fig1_ladder(); fig2_protocol(); fig3_leak(); fig4_slices(); fig5_threshold(); fig6_learning()
side_calendar(); side_bootstrap(); side_skew(); side_alarm()

facts = {
    "rows": N, "mean": round(MEAN_CNT, 1), "median": MEDIAN_CNT, "max": int(MAX_CNT),
    "sizes": [len(TR), len(VA), len(TE)], "test_days": len(TEST_DAYS),
    "mae": {"const": round(MAE_CONST, 1), "hour": round(MAE_HOUR, 1),
            "lag168": round(MAE_LAG, 1), "model_val": round(MAE_VAL, 1),
            "model_test": round(MAE_TEST, 1), "leaky": round(MAE_LEAK, 2)},
    "delta": round(DELTA, 1), "ci": [round(CI_LO, 1), round(CI_HI, 1)],
    "fracs": FRACS, "grid": grid,
    "slices_hour": [[s[0], round(s[1], 1), round(s[2], 0), s[3]] for s in SLICES_HOUR],
    "slices_weather": [[s[0], round(s[1], 1), s[2]] for s in SLICES_W],
    "thresholds": [[t, round(p, 3), round(r, 3), n] for t, p, r, n in THRESH],
    "base_rate": round(float(BASE_RATE), 3),
    "learning_curve": [[n, round(v, 1)] for n, v in LC],
    "r2": {"honest": round(float(R2_TEST), 3), "leaky": round(float(R2_LEAK), 4)},
}
FACTS.write_text(json.dumps(facts, ensure_ascii=False, indent=1))
print("facts ->", FACTS)
print("OK")
