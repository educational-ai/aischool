"""Deterministic figures for lesson 51: robust loss, ridge and lasso.

Two different kinds of instability, two different cures. A single spiked row on the REAL
bike-sharing data drags the least-squares slope down while Huber holds; a high-leverage point
breaks both. On the REAL diabetes table the correlated pair s1/s2 splits into +-38, ridge
lifts the eigenvalues and collapses the condition number, lasso zeroes coordinates, elastic
net keeps the pair together. Every number quoted in the lesson is computed here and asserted.

SEED DISCIPLINE (rule P5): the anchor number of the opening is the slope on ALL 730 evening
hours (14.74), which depends on no seed at all. The 60-row subsample exists only because the
scatter of 730 points is an unreadable smear and because n=60 is the regime the rest of the
lesson works in (the bias-variance section trains on 60 patients). Whenever a subsample number
is quoted, n and the seed are named in the prose and SEED_SWEEP below reports the range of the
effect over six seeds.

IMPORTANT (normalization trap): the diabetes numbers reproduce only with an explicit z-score
StandardScaler applied to the RAW features, load_diabetes(scaled=False). The default
load_diabetes() returns columns already divided by n**0.5 * std, which rescales every
coefficient by a common factor and destroys the quoted values. The guard asserts below fail
loudly if that normalization is ever changed.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
BIKE = ROOT / "scripts" / "data" / "bike-sharing-hour.csv"
OUT = ROOT / "public" / "figures" / "lessons" / "51"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "51"

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


# ======================================================================== data

def diabetes_z():
    """RAW diabetes features, then explicit z-score. See the normalization note above."""
    d = load_diabetes(scaled=False)
    z = StandardScaler().fit_transform(d.data)
    return z, d.target, list(d.feature_names)


Z, Y, NAMES = diabetes_z()
I_S1, I_S2 = NAMES.index("s1"), NAMES.index("s2")

CORR_S1S2 = float(np.corrcoef(Z[:, I_S1], Z[:, I_S2])[0, 1])
OLS_W = LinearRegression().fit(Z, Y).coef_
RIDGE10 = Ridge(alpha=10).fit(Z, Y).coef_
RIDGE100 = Ridge(alpha=100).fit(Z, Y).coef_

print(f"corr(s1,s2) = {CORR_S1S2:.3f}")
print(f"OLS      s1={OLS_W[I_S1]:.1f}  s2={OLS_W[I_S2]:.1f}")
print(f"ridge10  s1={RIDGE10[I_S1]:.1f}  s2={RIDGE10[I_S2]:.1f}")
print(f"ridge100 s1={RIDGE100[I_S1]:.1f}  s2={RIDGE100[I_S2]:.1f}")

# --- normalization guards + printed-precision guards (rule P2: tolerance strictly below half
# --- of the last printed digit of the number that appears in the lesson)
assert abs(CORR_S1S2 - 0.897) < 0.0005, CORR_S1S2                 # prose "0,897"
assert abs(OLS_W[I_S1] + 37.7) < 0.04, OLS_W[I_S1]              # prose "-37,7"
assert abs(OLS_W[I_S2] - 22.7) < 0.05, OLS_W[I_S2]              # prose "+22,7"
assert abs(RIDGE10[I_S1] + 11.3) < 0.04 and abs(RIDGE10[I_S2] - 1.8) < 0.04
assert abs(RIDGE100[I_S1] + 2.1) < 0.04 and abs(RIDGE100[I_S2] + 3.7) < 0.04
assert abs(OLS_W[NAMES.index("bmi")] - 24.7) < 0.05

# corr is invariant to the z-transform (lesson item 22): same number on the raw columns
CORR_RAW = float(np.corrcoef(load_diabetes(scaled=False).data[:, I_S1],
                             load_diabetes(scaled=False).data[:, I_S2])[0, 1])
assert abs(CORR_RAW - CORR_S1S2) < 1e-12, (CORR_RAW, CORR_S1S2)

KF = KFold(5, shuffle=True, random_state=0)


def cv_r2(model):
    return float(cross_val_score(make_pipeline(StandardScaler(), model),
                                 load_diabetes(scaled=False).data, Y,
                                 cv=KF, scoring="r2").mean())


CV_OLS = cv_r2(LinearRegression())
CV_RIDGE100 = cv_r2(Ridge(alpha=100))
CV_LASSO1 = cv_r2(Lasso(alpha=1.0, max_iter=100000))
print(f"5-fold CV R2: OLS {CV_OLS:.5f}  ridge100 {CV_RIDGE100:.5f}  lasso1 {CV_LASSO1:.5f}")
# rule P9/P2: prose prints three decimals, so the tolerance must be below 5e-4
assert abs(CV_OLS - 0.489) < 0.0005, CV_OLS
assert abs(CV_RIDGE100 - 0.483) < 0.0005, CV_RIDGE100      # 0.48252 rounds to 0.483, not 0.482
assert abs(CV_LASSO1 - 0.490) < 0.0005, CV_LASSO1

EIG = np.linalg.eigvalsh(Z.T @ Z)
KAPPA0 = EIG.max() / EIG.min()
KAPPA_X = float(np.sqrt(KAPPA0))                     # kappa(X) = sqrt(kappa(X^T X))
KAPPA_A = {a: (EIG.max() + a) / (EIG.min() + a) for a in (10, 100, 4420, 44200)}
SPECTRUM = EIG[::-1]
print("spectrum: " + "  ".join(f"{v:.2f}" for v in SPECTRUM))
print(f"eig min={EIG.min():.4f} max={EIG.max():.4f} trace={EIG.sum():.2f} "
      f"kappa(XtX)={KAPPA0:.4f} kappa(X)={KAPPA_X:.4f}")
print("kappa(alpha): " + "  ".join(f"{a}:{v:.4f}" for a, v in KAPPA_A.items()))
assert abs(EIG.max() - 1778.7) < 0.05 and abs(EIG.min() - 3.78) < 0.004
assert abs(EIG.sum() - 442 * 10) < 1e-6, EIG.sum()   # tr(Z^T Z) = n*p for z-scored columns
assert abs(KAPPA0 - 470) < 0.4
assert abs(KAPPA_X - 21.7) < 0.05
assert abs(KAPPA_A[10] - 129.8) < 0.05 and abs(KAPPA_A[100] - 18.1) < 0.04
assert abs(KAPPA_A[4420] - 1.40) < 0.005 and abs(KAPPA_A[44200] - 1.04) < 0.004
# lesson problem "5 points": the four numbers the reader is asked to work with are the two
# ends and two interior values of THIS spectrum, and the shrink factors at alpha=100
SHRINK100 = SPECTRUM / (SPECTRUM + 100)
print("shrink at alpha=100: " + "  ".join(f"{v:.3f}" for v in SHRINK100))
assert abs(SHRINK100[0] - 0.947) < 0.0005 and abs(SHRINK100[-1] - 0.036) < 5e-4

# entry threshold of every feature along a fine grid
FINE = np.logspace(np.log10(60), np.log10(0.02), 300)
ENTRY = {}
for a in FINE:
    w = Lasso(alpha=a, max_iter=200000).fit(Z, Y).coef_
    for j, v in enumerate(w):
        if abs(v) > 1e-8 and NAMES[j] not in ENTRY:
            ENTRY[NAMES[j]] = a
ORDER = sorted(ENTRY, key=lambda k: -ENTRY[k])
print("entry order:", [(k, round(ENTRY[k], 2)) for k in ORDER])
assert ORDER[:3] == ["bmi", "s5", "bp"], ORDER
assert ORDER[-2:] == ["s2", "age"], ORDER          # s2 enters last of the blood-chemistry group
print(f"entry thresholds: s1={ENTRY['s1']:.4f}  s2={ENTRY['s2']:.5f}")
assert abs(ENTRY["s1"] - 3.24) < 0.005, ENTRY["s1"]      # prose and label print "3,2"
assert abs(ENTRY["s2"] - 0.2546) < 5e-05, ENTRY["s2"]     # prose and label print "0,25"
assert round(ENTRY["s1"], 1) == 3.2 and round(ENTRY["s2"], 2) == 0.25

# ---------------------------------------- lasso vs elastic net at EQUAL L1 strength (rule P10)
# ElasticNet(alpha=1, l1_ratio=0.5) puts 0.5 in front of ||w||_1, so the honest comparison
# partner is Lasso(alpha=0.5), not Lasso(alpha=1). Both are reported.
LASSO1 = Lasso(alpha=1.0, max_iter=200000).fit(Z, Y).coef_
LASSO05 = Lasso(alpha=0.5, max_iter=200000).fit(Z, Y).coef_
ENET = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=200000).fit(Z, Y).coef_
NNZ_L1, NNZ_L05 = int(np.sum(np.abs(LASSO1) > 1e-8)), int(np.sum(np.abs(LASSO05) > 1e-8))
NNZ_EN = int(np.sum(np.abs(ENET) > 1e-8))
print(f"lasso1   s1={LASSO1[I_S1]:.3f} s2={LASSO1[I_S2]:.3f} nnz={NNZ_L1}")
print(f"lasso0.5 s1={LASSO05[I_S1]:.3f} s2={LASSO05[I_S2]:.3f} nnz={NNZ_L05} "
      f"zeros={[NAMES[j] for j in range(10) if abs(LASSO05[j]) < 1e-8]}")
print(f"enet     s1={ENET[I_S1]:.3f} s2={ENET[I_S2]:.3f} nnz={NNZ_EN}")
assert abs(LASSO1[I_S2]) < 1e-8 and abs(LASSO1[I_S1] + 4.84) < 0.004
assert abs(LASSO05[I_S2]) < 1e-8 and abs(LASSO05[I_S1] + 7.76) < 0.004
assert [NAMES[j] for j in range(10) if abs(LASSO05[j]) < 1e-8] == ["age", "s2"]
assert [NAMES[j] for j in range(10) if abs(LASSO1[j]) < 1e-8] == ["age", "s2", "s4"]
assert round(ENET[I_S1], 2) == -0.24 and round(ENET[I_S2], 2) == -2.37
assert abs(ENET[I_S1] + 0.2413) < 5e-4 and abs(ENET[I_S2] + 2.3664) < 5e-4
assert NNZ_EN == 10 and NNZ_L05 == 8 and NNZ_L1 == 7
# the pair is kept, but NOT brought together: ten-fold apart. The Zou-Hastie bound is an
# upper bound, not a promise of equality (lesson item 14/15).
assert abs(ENET[I_S2] / ENET[I_S1]) > 9.0, (ENET[I_S1], ENET[I_S2])

# ------------------------------- exactly duplicated column: the clean limiting case (item 13)
ZDUP = np.column_stack([Z, Z[:, I_S1]])
R_DUP0 = Ridge(alpha=1e-8).fit(ZDUP, Y).coef_          # lambda -> 0+: minimum-norm point
R_DUP10 = Ridge(alpha=10).fit(ZDUP, Y).coef_           # finite lambda: sum shrinks further
EN_DUP = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=500000).fit(ZDUP, Y).coef_
LA_DUP = Lasso(alpha=1.0, max_iter=500000).fit(ZDUP, Y).coef_
print(f"dup col: ridge(0+) {R_DUP0[I_S1]:.3f}/{R_DUP0[10]:.3f} sum {R_DUP0[I_S1] + R_DUP0[10]:.3f} "
      f"(OLS s1 {OLS_W[I_S1]:.3f}); ridge(10) {R_DUP10[I_S1]:.3f}/{R_DUP10[10]:.3f} "
      f"sum {R_DUP10[I_S1] + R_DUP10[10]:.3f}")
print(f"dup col: enet {EN_DUP[I_S1]:.4f}/{EN_DUP[10]:.4f}; lasso {LA_DUP[I_S1]:.3f}/{LA_DUP[10]:.3f}")
assert abs(R_DUP0[I_S1] - R_DUP0[10]) < 1e-3                       # equal halves
assert abs(R_DUP0[I_S1] + R_DUP0[10] - OLS_W[I_S1]) < 5e-3         # sum = s* at lambda -> 0+
assert abs(R_DUP10[I_S1] - R_DUP10[10]) < 1e-6                     # still equal at finite lambda
assert abs(R_DUP10[I_S1] + R_DUP10[10]) < abs(OLS_W[I_S1]) - 20    # but the SUM shrank a lot
assert abs(R_DUP0[I_S1] + 18.84) < 0.004 and abs(R_DUP10[I_S1] + 6.94) < 0.004
assert abs(EN_DUP[I_S1] - EN_DUP[10]) < 0.01                       # elastic net splits evenly
assert abs(LA_DUP[I_S1] - LA_DUP[10]) > 4.0                        # lasso splits arbitrarily
# exact figures quoted in the prose and in problem 8
assert round(OLS_W[I_S1], 2) == -37.68
assert round(R_DUP10[I_S1] + R_DUP10[10], 2) == -13.89
assert round(EN_DUP[I_S1], 2) == -0.17 and round(EN_DUP[10], 2) == -0.17
assert round(LA_DUP[I_S1], 2) == -0.12 and round(LA_DUP[10], 2) == -4.72

# ------------------------------------------------------------------ bike data
def bike_evening_all():
    """ALL evening (17:00) hours of the bike-sharing table. No seed, no subsample."""
    t, c = [], []
    with open(BIKE) as f:
        for row in csv.DictReader(f):
            if row["hr"] == "17":
                t.append(47 * float(row["temp"]) - 8)   # temp is min-max scaled: T = 47*temp - 8
                c.append(float(row["cnt"]))
    return np.array(t), np.array(c)


FX, FY = bike_evening_all()
assert len(FX) == 730, len(FX)


def subsample(n=60, seed=51):
    """A readable slice of the 730 hours.

    Why n=60 and not the full table: (a) 730 overlapping dots is an unreadable smear on a
    figure of this size; (b) n=60 is exactly the regime the bias-variance section of the
    lesson works in (ridge trained on 60 patients), so the opening and the closing example
    speak about samples of the same size. The seed is named in the prose and SEED_SWEEP
    below quotes the spread of the effect over six seeds - rule P5.
    """
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(FX), n, replace=False))
    return FX[idx], FY[idx]


def huber_fit(x, y, eps=1.35):
    m = HuberRegressor(epsilon=eps, max_iter=2000).fit(x[:, None], y)
    return float(m.coef_[0]), float(m.intercept_)


# --- the seed-free anchor of the opening: all 730 hours -----------------------
OLS_FULL = np.polyfit(FX, FY, 1)
SPIKE_FULL_J = int(np.argmin(np.abs(FX - np.quantile(FX, 0.05))))
# the row the lesson tells the story about: T = 2,34 C, 159 rides, spoiled into 2159
STORY_J = int(np.flatnonzero((np.abs(FX - 2.34) < 1e-9) & (FY == 159.0))[0])
FY_SPIKED = FY.copy(); FY_SPIKED[STORY_J] += 2000
OLS_FULL_SPIKED = np.polyfit(FX, FY_SPIKED, 1)
DROP_FULL = 1 - OLS_FULL_SPIKED[0] / OLS_FULL[0]
print(f"FULL 730 evening hours: slope {OLS_FULL[0]:.4f} intercept {OLS_FULL[1]:.4f}")
print(f"FULL with the same row spoiled (+2000): slope {OLS_FULL_SPIKED[0]:.4f} "
      f"-> drop {100 * DROP_FULL:.2f} %")
assert round(OLS_FULL[0], 2) == 14.74 and abs(OLS_FULL[0] - 14.7449) < 5e-4
assert round(OLS_FULL_SPIKED[0], 2) == 14.24 and abs(OLS_FULL_SPIKED[0] - 14.2436) < 5e-4
assert abs(100 * DROP_FULL - 3.4) < 0.05, DROP_FULL
assert abs(FX[STORY_J] - 2.34) < 0.005 and FY[STORY_J] == 159.0

BX, BY = subsample()
SPIKE_J = int(np.argmin(np.abs(BX - np.quantile(BX, 0.05))))   # a cold evening
assert abs(BX[SPIKE_J] - 2.34) < 0.005 and BY[SPIKE_J] == 159.0  # same row as in the full table
BY_SPIKED = BY.copy(); BY_SPIKED[SPIKE_J] += 2000

OLS_CLEAN = np.polyfit(BX, BY, 1)
OLS_SPIKE = np.polyfit(BX, BY_SPIKED, 1)
HUB_CLEAN = huber_fit(BX, BY)
HUB_SPIKE = huber_fit(BX, BY_SPIKED)
DROP_60 = 1 - OLS_SPIKE[0] / OLS_CLEAN[0]
print(f"n=60 seed=51 OLS   {OLS_CLEAN[0]:.4f} -> {OLS_SPIKE[0]:.4f} "
      f"(drop {100 * DROP_60:.2f} %), intercept {OLS_CLEAN[1]:.2f} -> {OLS_SPIKE[1]:.2f}")
print(f"n=60 seed=51 Huber {HUB_CLEAN[0]:.4f} -> {HUB_SPIKE[0]:.4f}   spike at T={BX[SPIKE_J]:.2f}")
assert abs(OLS_CLEAN[0] - 13.704) < 0.0005 and round(OLS_CLEAN[0], 2) == 13.70
assert abs(OLS_SPIKE[0] - 6.9747) < 5e-05 and round(OLS_SPIKE[0], 2) == 6.97
# sidenote prints whole rides: 220 -> 377
assert round(OLS_CLEAN[1]) == 220 and round(OLS_SPIKE[1]) == 377
assert abs(OLS_CLEAN[1] - 219.85) < 0.005 and abs(OLS_SPIKE[1] - 376.78) < 5e-3
assert abs(HUB_CLEAN[0] - 14.797) < 0.0005 and abs(HUB_SPIKE[0] - 13.507) < 5e-4
assert abs(100 * DROP_60 - 49) < 0.4, DROP_60

# --- rule P5: the same experiment over six seeds, range quoted in the prose ----
SEED_SWEEP = []
for _s in (51, 1, 2, 3, 4, 5):
    _x, _y = subsample(60, _s)
    _j = int(np.argmin(np.abs(_x - np.quantile(_x, 0.05))))
    _y2 = _y.copy(); _y2[_j] += 2000
    _a, _b = np.polyfit(_x, _y, 1)[0], np.polyfit(_x, _y2, 1)[0]
    SEED_SWEEP.append((_s, _a, _b, 100 * (1 - _b / _a)))
print("seed sweep (seed, clean, spiked, drop %): "
      + "  ".join(f"{s}:{a:.2f}->{b:.2f} ({d:.0f}%)" for s, a, b, d in SEED_SWEEP))
DROP_MIN = min(d for *_, d in SEED_SWEEP); DROP_MAX = max(d for *_, d in SEED_SWEEP)
print(f"drop over six seeds: {DROP_MIN:.1f} % .. {DROP_MAX:.1f} %")
assert abs(DROP_MIN - 30) < 0.4 and abs(DROP_MAX - 72) < 0.4, (DROP_MIN, DROP_MAX)

RESID = BY - np.polyval(OLS_CLEAN, BX)
MAD = float(np.median(np.abs(RESID - np.median(RESID))))
SIGMA_HAT = 1.4826 * MAD
print(f"n=60 MAD={MAD:.2f}  sigma_hat=1.4826*MAD={SIGMA_HAT:.2f}  "
      f"delta=1.35*sigma={1.35 * SIGMA_HAT:.2f}  2*sigma={2 * SIGMA_HAT:.2f}")
assert abs(MAD - 130) < 0.5 and abs(SIGMA_HAT - 193) < 0.5
assert abs(1.35 * SIGMA_HAT - 261) < 0.5

# ------------------------------------------------------- leverage (rule P1/P11)
# The added point is NOT quiet: its residual is the second largest of the 61, it IS damped
# by Huber (weight ~ 0.53), and damping still fails - because its (x - xbar)^2 is an order of
# magnitude above the typical one, and the pull on the slope is proportional to w_i*(x_i-xbar)^2.
LEV_X, LEV_Y = 45.0, 1400.0
LX = np.append(BX, LEV_X); LY = np.append(BY, LEV_Y)
OLS_LEV = np.polyfit(LX, LY, 1)
HUB_LEV = huber_fit(LX, LY)
LEV_H = 1 / len(LX) + (LX - LX.mean()) ** 2 / np.sum((LX - LX.mean()) ** 2)
LEV_RES_OLS = float((LY - np.polyval(OLS_LEV, LX))[-1])
LEV_RES_HUB = float((LY - (HUB_LEV[1] + HUB_LEV[0] * LX))[-1])
LEV_RANK = int(np.sum(np.abs(LY - np.polyval(OLS_LEV, LX)) >= abs(LEV_RES_OLS)))
LEV_W = min(1.0, 1.35 * SIGMA_HAT / abs(LEV_RES_HUB))
LEV_DX2 = float((LX[-1] - LX.mean()) ** 2 / np.median((LX - LX.mean()) ** 2))
LEV_CLEAN_Y = float(np.polyval(OLS_CLEAN, LEV_X))
COOK_FACTOR = 1 / (1 - LEV_H[-1]) ** 2
print(f"leverage OLS {OLS_CLEAN[0]:.4f} -> {OLS_LEV[0]:.4f}, Huber {HUB_CLEAN[0]:.4f} -> {HUB_LEV[0]:.4f}")
print(f"leverage h={LEV_H[-1]:.4f} (median {np.median(LEV_H):.4f}, ratio {LEV_H[-1] / np.median(LEV_H):.2f}); "
      f"OLS residual {LEV_RES_OLS:.1f} (rank {LEV_RANK} of {len(LX)}, 2*sigma={2 * SIGMA_HAT:.0f}); "
      f"Huber residual {LEV_RES_HUB:.1f}, weight delta/|e| = {LEV_W:.3f}")
print(f"leverage (x-xbar)^2 is {LEV_DX2:.4f}x the median; Cook factor 1/(1-h)^2 = {COOK_FACTOR:.3f}; "
      f"clean line at x=45 would be y={LEV_CLEAN_Y:.1f}")
assert abs(OLS_LEV[0] - 16.407) < 0.0005 and round(OLS_LEV[0], 2) == 16.41
assert abs(HUB_LEV[0] - 16.285) < 0.0005 and round(HUB_LEV[0], 2) == 16.29
assert round(LEV_RES_HUB) == 491 and round(1.35 * SIGMA_HAT) == 261
assert abs(LEV_H[-1] - 0.142) < 0.0005 and round(LEV_H[-1], 2) == 0.14
assert abs(np.median(LEV_H) - 0.0255) < 5e-05 and round(float(np.median(LEV_H)), 3) == 0.025
assert abs(LEV_RES_OLS - 483.4) < 0.05 and round(LEV_RES_OLS) == 483
assert LEV_RES_OLS > 2 * SIGMA_HAT                  # it DOES stand out: 483 > 387
assert LEV_RANK == 2                                # second largest of the 61 residuals
assert abs(LEV_W - 0.531) < 0.0005                    # Huber damps it roughly by half
assert round(LEV_DX2, 1) == 13.9 and 13 < LEV_DX2 < 15                  # ... and the leverage term is 14x typical
assert abs(COOK_FACTOR - 1.359) < 0.0005
assert abs(LEV_CLEAN_Y - 836.5) < 0.05


# ======================================================== fig 51.1 loss shapes
def fig_losses():
    delta = 2.0
    e = np.linspace(-6, 6, 601)
    sq = 0.5 * e ** 2
    ab = np.abs(e)
    hub = np.where(np.abs(e) <= delta, 0.5 * e ** 2, delta * (np.abs(e) - 0.5 * delta))
    assert abs(hub[np.argmin(np.abs(e - 6))] - delta * (6 - 0.5 * delta)) < 1e-9
    d_sq, d_ab = e, np.sign(e)
    d_hub = np.clip(e, -delta, delta)
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(8.6, 6.4), sharex=True)
    a0.plot(e, sq, color=RED, lw=2.2, label=r"квадрат $\frac{1}{2}e^2$")
    a0.plot(e, ab, color=BLUE, lw=2.0, label=r"модуль $|e|$")
    a0.plot(e, hub, color=GREEN, lw=2.6, label=r"Хьюбер, $\delta=2$")
    a0.set_ylim(0, 12); a0.set_ylabel("штраф")
    a0.set_title("Три штрафа за один и тот же промах")
    a0.legend(loc="upper center", frameon=False, fontsize=10, ncol=3)
    a1.plot(e, d_sq, color=RED, lw=2.2)
    a1.plot(e, d_ab, color=BLUE, lw=2.0)
    a1.plot(e, d_hub, color=GREEN, lw=2.6)
    a1.axhline(delta, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    a1.axhline(-delta, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    a1.text(5.6, delta + 0.35, r"$+\delta$", color=MUTED, fontsize=10, ha="right")
    a1.set_ylabel("сила тяги $\\rho'(e)$"); a1.set_xlabel("остаток $e$")
    a1.set_title("Производная: с какой силой точка тянет прямую", fontsize=13)
    a1.set_ylim(-6.5, 6.5)
    a1.annotate("у квадрата сила растёт без границ", xy=(5.2, 5.2), xytext=(1.4, 5.4),
                color=RED, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
    for ax in (a0, a1):
        ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "loss_shapes.png")


# ================================================ fig 51.2 mean vs median pull
def fig_breakdown():
    ms = np.linspace(0, 60, 400)
    mean = (0 + 0 + 0 + 0 + ms) / 5
    med = np.zeros_like(ms)
    assert abs(mean[np.argmin(np.abs(ms - 50))] - 10) < 0.05      # M=50 -> mean 10, median 0
    assert med[np.argmin(np.abs(ms - 50))] == 0
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.4, 4.0))
    a0.plot(ms, mean, color=RED, lw=2.4, label="среднее (минимум суммы квадратов)")
    a0.plot(ms, med, color=GREEN, lw=2.4, label="медиана (минимум суммы модулей)")
    a0.set_xlabel("значение выброса $M$"); a0.set_ylabel("оценка центра")
    a0.set_title("Набор $0,0,0,0,M$", fontsize=13)
    a0.legend(loc="upper left", frameon=False, fontsize=9.5)
    a0.grid(True, color=GRID, lw=0.4, alpha=0.5); a0.set_axisbelow(True)

    # breakdown point: share of contamination each estimator survives
    share = np.arange(0, 11) / 20
    a1.bar([0, 1], [0, 0.5], width=0.5, color=[RED, GREEN])
    a1.set_xticks([0, 1]); a1.set_xticklabels(["среднее", "медиана"])
    a1.set_ylabel("предельная доля загрязнения")
    a1.set_ylim(0, 0.62)
    a1.text(0, 0.03, "0 %", ha="center", color=RED, fontsize=11)
    a1.text(1, 0.53, "50 %", ha="center", color=GREEN, fontsize=11)
    a1.set_title("Точка отказа оценки", fontsize=13)
    a1.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); a1.set_axisbelow(True)
    assert share[-1] == 0.5
    fig.suptitle("Одна строка может унести среднее и не сдвинуть медиану", y=1.02, fontsize=14.5)
    fig.tight_layout()
    save(fig, OUT / "breakdown.png")


# ============================================ fig 51.3 bike: spike breaks OLS
def fig_bike_robust():
    xs = np.linspace(BX.min() - 1, BX.max() + 1, 50)
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.scatter(BX, BY, s=42, color=INK, alpha=0.75, zorder=4,
               label="вечер 17:00, 60 случайных часов (seed 51)")
    ax.scatter([BX[SPIKE_J]], [BY_SPIKED[SPIKE_J]], s=130, color=GOLD,
               edgecolor=INK, lw=1.0, zorder=6, label="сломанная строка: +2000")
    ax.plot(xs, np.polyval(OLS_CLEAN, xs), color=MUTED, lw=1.8, ls=(0, (5, 3)),
            label=f"МНК без выброса: {OLS_CLEAN[0]:.2f} поездки/°C".replace(".", ","), zorder=7)
    ax.plot(xs, np.polyval(OLS_SPIKE, xs), color=RED, lw=2.6,
            label=f"МНК с выбросом: {OLS_SPIKE[0]:.2f}".replace(".", ","))
    ax.plot(xs, HUB_SPIKE[1] + HUB_SPIKE[0] * xs, color=GREEN, lw=2.6,
            label=f"Хьюбер с выбросом: {HUB_SPIKE[0]:.2f}".replace(".", ","))
    ax.annotate("", xy=(BX[SPIKE_J], BY_SPIKED[SPIKE_J] - 60), xytext=(BX[SPIKE_J], BY[SPIKE_J] + 40),
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.4))
    ax.set_xlabel("температура, °C"); ax.set_ylabel("поездок за час")
    ax.set_title(f"Случайные 60 вечерних часов (seed 51): наклон потерял "
                 f"{100 * DROP_60:.0f} %, Хьюбер устоял")
    ax.set_ylim(-120, 2400)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "bike_robust.png")


# ==================================== fig 51.4 leverage breaks Huber too
def fig_leverage():
    xs = np.linspace(-2, 48, 60)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.2, 4.6))
    a0.scatter(BX, BY, s=34, color=INK, alpha=0.7, zorder=4)
    a0.scatter([LEV_X], [LEV_Y], s=140, color=VIOLET, edgecolor=INK, lw=1.0, zorder=6)
    a0.text(LEV_X, LEV_Y - 130, "высокий рычаг", color=VIOLET, fontsize=10, ha="center")
    a0.plot(xs, np.polyval(OLS_CLEAN, xs), color=MUTED, lw=1.6, ls=(0, (5, 3)),
            label=f"без точки: {OLS_CLEAN[0]:.1f}".replace(".", ","))
    a0.plot(xs, np.polyval(OLS_LEV, xs), color=RED, lw=2.4, label=f"МНК: {OLS_LEV[0]:.1f}".replace(".", ","))
    a0.plot(xs, HUB_LEV[1] + HUB_LEV[0] * xs, color=GREEN, lw=2.4, label=f"Хьюбер: {HUB_LEV[0]:.1f}".replace(".", ","))
    a0.set_xlabel("температура, °C"); a0.set_ylabel("поездок за час")
    a0.set_title("Точка с редким $x$ ломает обе прямые", fontsize=13)
    a0.legend(loc="upper left", frameon=False, fontsize=9.5)

    # hat values of the augmented sample
    h = LEV_H
    res = LY - np.polyval(OLS_LEV, LX)
    assert h[-1] > 5 * np.median(h), (h[-1], np.median(h))
    a1.scatter(h[:-1], np.abs(res[:-1]), s=34, color=INK, alpha=0.7)
    a1.scatter([h[-1]], [abs(res[-1])], s=140, color=VIOLET, edgecolor=INK, lw=1.0, zorder=6)
    a1.axhline(2 * SIGMA_HAT, color=MUTED, lw=0.9, ls=(0, (2, 2)))
    a1.text(h.max() * 0.98, 2 * SIGMA_HAT + 18, r"$2\hat\sigma$", color=MUTED, fontsize=10, ha="right")
    a1.set_xlabel("рычаг $h_{ii}$"); a1.set_ylabel("|остаток|")
    a1.set_title(f"Рычаг в {h[-1] / np.median(h):.0f} раз выше типичного, "
                 f"остаток {abs(res[-1]):.0f} — второй по модулю", fontsize=12)
    a1.annotate((f"h = {h[-1]:.2f} при медиане {np.median(h):.3f},\n"
                 f"остаток {abs(res[-1]):.0f} > $2\\hat\\sigma$ = {2 * SIGMA_HAT:.0f}").replace(".", ","),
                xy=(h[-1], abs(res[-1])), xytext=(h[-1] * 0.30, abs(res[-1]) + 70),
                color=VIOLET, fontsize=10, arrowprops=dict(arrowstyle="->", color=VIOLET, lw=1.1))
    a1.set_ylim(-20, 680)
    for ax in (a0, a1):
        ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.suptitle("Робастность по остатку не защищает от необычного признака", y=1.02, fontsize=14.5)
    fig.tight_layout()
    save(fig, OUT / "leverage.png")


# ================================ fig 51.5 delta in MAD units on real residuals
def fig_delta_mad():
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.0, 4.3))
    a0.hist(RESID, bins=18, color=BLUE, alpha=0.55, edgecolor=PAPER)
    for k, c, lab in [(1.0, GOLD, r"$1\hat\sigma$"), (1.35, GREEN, r"$\delta=1{,}35\hat\sigma$"),
                      (3.0, RED, r"$3\hat\sigma$")]:
        a0.axvline(k * SIGMA_HAT, color=c, lw=1.8)
        a0.axvline(-k * SIGMA_HAT, color=c, lw=1.8)
        a0.text(k * SIGMA_HAT + 8, a0.get_ylim()[1] * (0.92 - 0.12 * k), lab, color=c, fontsize=10)
    a0.set_xlabel("остаток, поездок"); a0.set_ylabel("число часов")
    a0.set_title(f"MAD $={MAD:.0f}$, $\\hat\\sigma=1{{,}}4826\\cdot$MAD$={SIGMA_HAT:.0f}$", fontsize=12.5)

    ks = np.linspace(0.2, 4.0, 200)
    share = np.array([np.mean(np.abs(RESID) > k * SIGMA_HAT) for k in ks])
    at135 = float(np.mean(np.abs(RESID) > 1.35 * SIGMA_HAT))
    print(f"share treated linearly at delta=1.35*sigma: {100 * at135:.2f} %")
    assert abs(100 * at135 - 13.3) < 0.05, at135      # prose and caption print "13 %"
    a1.plot(ks, 100 * share, color=VIOLET, lw=2.4)
    a1.axvline(1.35, color=GREEN, lw=1.6, ls=(0, (4, 3)))
    a1.plot([1.35], [100 * at135], "o", color=GREEN, markersize=9)
    a1.annotate(f"при $\\delta=1{{,}}35\\hat\\sigma$ линейно\nобрабатываются {100 * at135:.0f} % точек",
                xy=(1.35, 100 * at135), xytext=(1.75, 100 * at135 + 16),
                color=GREEN, fontsize=10, arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.1))
    a1.set_xlabel(r"порог $\delta$ в единицах $\hat\sigma$")
    a1.set_ylabel("доля точек в линейной зоне, %")
    a1.set_title("Порог задают в единицах робастного масштаба", fontsize=12.5)
    for ax in (a0, a1):
        ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "delta_mad.png")


# ============================== fig 51.6 correlated pair and the ridge descent
def fig_collinear():
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.4, 4.8))
    a0.scatter(Z[:, I_S1], Z[:, I_S2], s=16, color=BLUE, alpha=0.45)
    lim = np.array([-3.2, 4.6])
    a0.plot(lim, lim * CORR_S1S2, color=RED, lw=2.0)
    a0.set_xlabel("s1 (общий холестерин), z-оценка")
    a0.set_ylabel("s2 (ЛПНП), z-оценка")
    a0.set_title(f"Два почти одинаковых признака: $r={CORR_S1S2:.3f}$".replace(".", ","), fontsize=13)

    # error surface in the (w_s1, w_s2) plane, other coefficients fixed at OLS
    g = np.linspace(-55, 45, 220)
    W1, W2 = np.meshgrid(g, g)
    base = Z @ OLS_W - Z[:, I_S1] * OLS_W[I_S1] - Z[:, I_S2] * OLS_W[I_S2]
    r0 = Y - base
    A = np.array([[Z[:, I_S1] @ Z[:, I_S1], Z[:, I_S1] @ Z[:, I_S2]],
                  [Z[:, I_S1] @ Z[:, I_S2], Z[:, I_S2] @ Z[:, I_S2]]]) / len(Y)
    b = np.array([Z[:, I_S1] @ r0, Z[:, I_S2] @ r0]) / len(Y)
    L = (A[0, 0] * W1 ** 2 + 2 * A[0, 1] * W1 * W2 + A[1, 1] * W2 ** 2
         - 2 * (b[0] * W1 + b[1] * W2))
    L = L - L.min()
    a1.contour(W1, W2, L, levels=np.array([0.5, 2, 6, 15, 40, 100, 250]) * 2,
               colors=[LINE], linewidths=0.9)
    for w, c, lab, off in [(OLS_W, RED, "МНК", (12, 12)),
                           (RIDGE10, GOLD, r"ridge $\alpha=10$", (14, 16)),
                           (RIDGE100, GREEN, r"ridge $\alpha=100$", (14, -30))]:
        a1.plot([w[I_S1]], [w[I_S2]], "o", color=c, markersize=11, zorder=6)
        a1.annotate(f"{lab}\n({w[I_S1]:.1f}; {w[I_S2]:.1f})".replace(".", ","), (w[I_S1], w[I_S2]),
                    textcoords="offset points", xytext=off, color=c, fontsize=10)
    a1.plot([0], [0], "+", color=INK, markersize=12)
    a1.axhline(0, color=GRID, lw=0.8); a1.axvline(0, color=GRID, lw=0.8)
    a1.set_xlabel("вес s1"); a1.set_ylabel("вес s2")
    a1.set_title("Долина ошибки и путь к началу координат", fontsize=13)
    for ax in (a0, a1):
        ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    fig.suptitle("Корреляция 0,897 разводит коэффициенты на −37,7 и +22,7", y=1.02, fontsize=14.5)
    fig.tight_layout()
    save(fig, OUT / "collinear.png")


# ======================== fig 51.7 eigenvalue lift and the condition number
def fig_conditioning():
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.2, 4.5))
    k = np.arange(1, 11)
    a0.semilogy(k, EIG[::-1], "o-", color=BLUE, lw=2.0, label=r"$X^\top X$")
    a0.semilogy(k, EIG[::-1] + 100, "s--", color=GOLD, lw=1.8, label=r"$+\,100\cdot I$")
    a0.semilogy(k, EIG[::-1] + 4420, "^--", color=GREEN, lw=1.8, label=r"$+\,4420\cdot I$")
    a0.set_xlabel("номер собственного значения"); a0.set_ylabel(r"$\lambda_k$ (лог)")
    a0.set_title(f"$\\lambda_{{\\min}}$ = {EIG.min():.1f} поднимается добавкой".replace(".", ","), fontsize=12.5)
    a0.legend(frameon=False, fontsize=9.5)
    a0.set_xticks(k)

    als = np.logspace(-1, 5, 200)
    kap = (EIG.max() + als) / (EIG.min() + als)
    a1.loglog(als, kap, color=VIOLET, lw=2.4)
    for a, c in [(10, GOLD), (100, GREEN), (4420, RED)]:
        a1.plot([a], [KAPPA_A[a]], "o", color=c, markersize=10)
        a1.annotate(f"$\\alpha$ = {a}\n$\\kappa$ = {KAPPA_A[a]:.1f}".replace(".", ","),
                    (a, KAPPA_A[a]), textcoords="offset points", xytext=(8, 8),
                    color=c, fontsize=10)
    a1.axhline(KAPPA0, color=LINE, lw=1.0, ls=(0, (3, 3)))
    a1.text(0.12, KAPPA0 * 1.15, f"$\\kappa(X^\\top X)$ = {KAPPA0:.0f}", color=MUTED, fontsize=10)
    a1.set_xlabel(r"штраф $\alpha$"); a1.set_ylabel(r"число обусловленности $\kappa$")
    a1.set_title("Обусловленность падает почти до единицы", fontsize=12.5)
    for ax in (a0, a1):
        ax.grid(True, color=GRID, lw=0.4, alpha=0.5, which="both"); ax.set_axisbelow(True)
    fig.suptitle("Ridge поднимает малые собственные значения", y=1.02, fontsize=14.5)
    fig.tight_layout()
    save(fig, OUT / "conditioning.png")


# ================================================== fig 51.8 lasso path (new angle)
def fig_lasso_path():
    als = np.logspace(np.log10(60), np.log10(0.03), 90)
    W = np.array([Lasso(alpha=a, max_iter=200000).fit(Z, Y).coef_ for a in als])
    nnz = np.sum(np.abs(W) > 1e-8, axis=1)
    # ---- rule P3: the caption is asserted on THIS grid, the one the figure is drawn from.
    # The counter passes through every value from 0 to 10, and it is NOT monotone: near the
    # weakest penalty it falls back from 10 to 9 once, when s3 crosses zero.
    assert sorted(set(int(v) for v in nnz)) == list(range(11)), sorted(set(nnz))
    assert nnz[0] == 0 and nnz[-1] == 10, (nnz[0], nnz[-1])
    dips = [i for i in range(1, len(nnz)) if nnz[i] < nnz[i - 1]]
    assert len(dips) == 1, dips
    _di = dips[0]
    _dropped = [NAMES[j] for j in range(10) if abs(W[_di][j]) < 1e-8]
    # the one non-monotone island: a narrow band of alpha where nnz falls back to 9
    _band = als[(nnz == 9) & (als < 0.2)]
    print(f"nnz on the drawn 90-point grid: 0 -> 10 through every value; one dip 10->9 "
          f"on alpha in [{_band.min():.4f}, {_band.max():.4f}], zeroed feature {_dropped}")
    assert _dropped == ["s3"], _dropped
    assert all(abs(W[i][NAMES.index("s3")]) < 1e-8 for i in np.flatnonzero((nnz == 9) & (als < 0.2)))
    assert round(_band.min(), 3) == 0.065 and round(_band.max(), 3) == 0.099
    assert nnz[als > 0.11][-1] == 10 and nnz[als < 0.06][0] == 10
    # the first three features to appear are bmi, s5, bp on this very grid
    _first_alpha = min(i for i in range(len(als)) if nnz[i] == 3)
    assert {NAMES[j] for j in range(10) if abs(W[_first_alpha][j]) > 1e-8} == {"bmi", "s5", "bp"}
    # the two dashed entry markers are drawn from ENTRY (a finer grid); check they agree with
    # the curves actually plotted here
    for nm in ("s1", "s2"):
        j = NAMES.index(nm)
        assert np.all(np.abs(W[als > ENTRY[nm] * 1.02, j]) < 1e-8), nm
        assert np.any(np.abs(W[als < ENTRY[nm] * 0.98, j]) > 1e-8), nm
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(9.0, 6.6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
    palette = {"bmi": RED, "s5": GREEN, "bp": BLUE, "s1": VIOLET, "s2": GOLD}
    for j, nm in enumerate(NAMES):
        c = palette.get(nm, LINE)
        a0.semilogx(als, W[:, j], color=c, lw=2.4 if nm in palette else 1.2,
                    alpha=1.0 if nm in palette else 0.7,
                    label=nm if nm in palette else None)
    a0.legend(loc="lower right", frameon=False, fontsize=10, ncol=2)
    for nm, c in [("s1", VIOLET), ("s2", GOLD)]:
        a0.axvline(ENTRY[nm], color=c, lw=1.0, ls=(0, (3, 3)))
    a0.text(ENTRY["s1"] * 1.12, 28, f"s1 входит\nпри $\\alpha$ = {ENTRY['s1']:.1f}".replace(".", ","),
            color=VIOLET, fontsize=9.5)
    a0.text(ENTRY["s2"] * 1.12, -32, f"s2 входит\nпри $\\alpha$ = {ENTRY['s2']:.2f}".replace(".", ","),
            color=GOLD, fontsize=9.5)
    a0.axhline(0, color=MUTED, lw=0.8)
    a0.set_ylabel("коэффициент")
    a0.set_title("Кто из пары s1/s2 представляет группу")
    a1.semilogx(als, nnz, color=INK, lw=2.0, drawstyle="steps-post")
    a1.set_xlabel(r"штраф $\alpha$ (слева слабый, справа сильный)"); a1.set_ylabel("ненулевых")
    a1.set_yticks([0, 5, 10])
    for ax in (a0, a1):
        ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "lasso_path.png")


# ============================================ fig 51.9 elastic net keeps the pair
def fig_elastic():
    x = np.arange(len(NAMES))
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    # rule P10: EQUAL L1 strength. ElasticNet(alpha=1, l1_ratio=0.5) carries 0.5*||w||_1,
    # so the comparison partner is Lasso(alpha=0.5), not Lasso(alpha=1).
    ax.bar(x - 0.22, LASSO05, width=0.42, color=RED, alpha=0.85,
           label=r"lasso, $\alpha=0{,}5$ ($L_1$-сила $0{,}5$)")
    ax.bar(x + 0.22, ENET, width=0.42, color=GREEN, alpha=0.85,
           label=r"elastic net, $\alpha=1$, $\gamma=0{,}5$ ($L_1$-сила $0{,}5$)")
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(NAMES)
    ax.set_ylabel("коэффициент")
    for j in (I_S1, I_S2):
        ax.axvspan(j - 0.5, j + 0.5, color=WASH, zorder=0)
    ax.annotate(f"lasso: s2 = 0\nelastic net: {ENET[I_S2]:.2f}".replace(".", ","),
                xy=(I_S2 + 0.22, ENET[I_S2]), xytext=(I_S2 + 1.5, -8.5),
                fontsize=10, color=GREEN, arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.1))
    ax.set_title("При одинаковой $L_1$-силе ромб теряет s2, а смесь его удерживает",
                 fontsize=13.5)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "elastic.png")


# ============================ fig 51.10 bias-variance vs lambda with the CV min
BV = {}


def fig_bias_variance():
    raw = load_diabetes(scaled=False).data
    rng = np.random.default_rng(51)
    perm = rng.permutation(len(Y)); tr, te = perm[:60], perm[60:]
    Xte, yte = raw[te], Y[te]
    als = np.logspace(-1, 4, 26)
    bias2, var, mse = [], [], []
    for a in als:
        preds = []
        for _ in range(200):
            i = rng.choice(tr, 60, replace=True)
            sc = StandardScaler().fit(raw[i])
            m = Ridge(alpha=a).fit(sc.transform(raw[i]), Y[i])
            preds.append(m.predict(sc.transform(Xte)))
        P = np.array(preds); mp = P.mean(0)
        bias2.append(np.mean((mp - yte) ** 2)); var.append(np.mean(P.var(0)))
        mse.append(np.mean((P - yte) ** 2))
    bias2, var, mse = map(np.array, (bias2, var, mse))
    k = int(np.argmin(mse))
    BV.update(alpha=float(als[k]), mse=float(mse[k]), var0=float(var[0]), varK=float(var[k]),
              mse0=float(mse[0]), mseEnd=float(mse[-1]))
    # rule P1: the prose says "at alpha=63 BOTH the variance and the bias fall; the price
    # arrives later". Assert exactly the quantities the prose names - the two MSE - Var
    # differences, and the growth of bias^2 at the right end.
    b0, bk, bend = mse[0] - var[0], mse[k] - var[k], mse[-1] - var[-1]
    print(f"bias-variance: best alpha={als[k]:.2f}; "
          f"MSE {mse[0]:.2f} -> {mse[k]:.2f} -> {mse[-1]:.2f}; "
          f"Var {var[0]:.2f} -> {var[k]:.2f} -> {var[-1]:.2f}; "
          f"MSE-Var (bias^2+noise) {b0:.2f} -> {bk:.2f} -> {bend:.2f}")
    assert abs(als[k] - 63) < 0.4, als[k]
    assert round(mse[0]) == 4623 and round(mse[k]) == 3829 and round(mse[-1]) == 5971
    assert abs(mse[0] - 4622.91) < 0.005 and abs(mse[k] - 3828.58) < 5e-3
    assert round(var[0]) == 523 and round(var[k]) == 124
    assert round(b0) == 4100, b0                 # prose "4100"
    assert round(bk) == 3704, bk                 # prose "3704" - it FELL, it did not grow
    assert round(bend) == 5867, bend             # prose "5867" - the price arrives here
    assert bk < b0, (bk, b0)                        # the narrative claim, asserted
    assert bend > b0, (bend, b0)
    assert abs(var[0] / var[k] - 4.2) < 0.05        # prose "в 4,2 раза"
    # alt-text claim: bias^2+noise dips before it climbs
    assert bias2.argmin() > 0 and bias2[-1] == bias2.max()
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.semilogx(als, bias2, color=BLUE, lw=2.2, label="смещение$^2$ + шум")
    ax.semilogx(als, var, color=GOLD, lw=2.2, label="разброс оценки")
    ax.semilogx(als, mse, color=RED, lw=2.8, label="полная ошибка на отложенных данных")
    ax.plot([als[k]], [mse[k]], "o", color=RED, markersize=11)
    ax.annotate(f"минимум: $\\alpha\\approx{als[k]:.0f}$\nошибка {mse[k]:.0f} против {mse[0]:.0f}",
                (als[k], mse[k]), textcoords="offset points", xytext=(30, -130),
                color=RED, fontsize=10.5, arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
    ax.set_xlabel(r"штраф $\alpha$"); ax.set_ylabel("средний квадрат ошибки")
    ax.set_title("Обучение по 60 больным: сначала падает и разброс, и смещение")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5, which="both"); ax.set_axisbelow(True)
    save(fig, OUT / "bias_variance.png")


# ================================================================== sidenotes
def side_influence():
    e = np.linspace(-5, 5, 400)
    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    ax.plot(e, e, color=RED, lw=2.0, label="квадрат")
    ax.plot(e, np.sign(e), color=BLUE, lw=1.8, label="модуль")
    ax.plot(e, np.clip(e, -1.5, 1.5), color=GREEN, lw=2.2, label="Хьюбер")
    ax.set_xlabel("остаток", fontsize=9); ax.set_ylabel("влияние", fontsize=9)
    ax.set_title("функция влияния", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "influence.png")


def side_diamond():
    """Same error contours, two constraint regions: the corner of the diamond zeroes a weight."""
    c = np.array([2.4, 0.6])
    R = np.array([[np.cos(-0.6), -np.sin(-0.6)], [np.sin(-0.6), np.cos(-0.6)]])
    A = R @ np.diag([1.0, 3.0]) @ R.T

    def loss(w):
        d = w - c
        return np.einsum("...i,ij,...j->...", d, A, d)

    t = np.linspace(0, 2 * np.pi, 4000)
    circle = np.stack([np.cos(t), np.sin(t)], axis=1)
    u = np.linspace(-1, 1, 2000)
    diamond = np.concatenate([np.stack([u, 1 - np.abs(u)], 1), np.stack([u, np.abs(u) - 1], 1)])
    w_ridge = circle[np.argmin(loss(circle))]
    w_lasso = diamond[np.argmin(loss(diamond))]
    print(f"margin geometry: ridge ({w_ridge[0]:.2f}; {w_ridge[1]:.2f}), "
          f"lasso ({w_lasso[0]:.2f}; {w_lasso[1]:.2f})")
    assert abs(w_ridge[0]) > 0.5 and abs(w_ridge[1]) > 0.2      # both weights alive
    assert abs(w_lasso[1]) < 0.02 and abs(abs(w_lasso[0]) - 1) < 0.02   # corner: one weight is 0

    g = np.linspace(-1.9, 3.4, 260)
    G1, G2 = np.meshgrid(g, g)
    L = loss(np.stack([G1, G2], axis=-1))
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.5))
    for ax, w, col, ttl in ((axes[0], w_ridge, BLUE, "круг: оба веса живы"),
                            (axes[1], w_lasso, GREEN, "угол ромба: вес обнулён")):
        lv = loss(w[None])[0]
        ax.contour(G1, G2, L, levels=[lv * 0.12, lv * 0.45, lv], colors=[LINE], linewidths=0.8)
        if col is BLUE:
            ax.plot(circle[:, 0], circle[:, 1], color=col, lw=1.8)
        else:
            ax.plot([1, 0, -1, 0, 1], [0, 1, 0, -1, 0], color=col, lw=1.8)
        ax.plot([c[0]], [c[1]], "x", color=MUTED, markersize=7)
        ax.plot([w[0]], [w[1]], "o", color=RED, markersize=6)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_xlim(-1.7, 3.3); ax.set_ylim(-1.7, 2.6)
        ax.set_title(ttl, fontsize=8)
    fig.tight_layout()
    save(fig, SIDE / "diamond.png")


def side_prior():
    w = np.linspace(-4, 4, 400)
    gauss = np.exp(-w ** 2 / 2) / np.sqrt(2 * np.pi)
    lap = np.exp(-np.abs(w) * np.sqrt(2)) * np.sqrt(2) / 2
    assert abs(np.trapezoid(gauss, w) - 1) < 5e-3 and abs(np.trapezoid(lap, w) - 1) < 5e-3
    assert lap[np.argmin(np.abs(w))] > gauss[np.argmin(np.abs(w))]   # sharper peak at zero
    fig, ax = plt.subplots(figsize=(4.0, 2.3))
    ax.plot(w, gauss, color=BLUE, lw=2.0, label="нормальный → ridge")
    ax.plot(w, lap, color=GREEN, lw=2.0, label="Лапласа → lasso")
    ax.set_xlabel("вес $w_j$", fontsize=9); ax.set_yticks([])
    ax.set_title("штраф — это prior", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "prior.png")


def side_scale():
    """Same physical effect, two units: the L2 penalty is not unit-free."""
    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    labels = ["км", "м", "мм"]
    pen = [1.0 ** 2, 1000.0 ** 2 * 1e-6 * 1e6, 0]
    coef = [1.0, 1e-3, 1e-6]
    ax.bar(range(3), [c ** 2 for c in coef], color=[BLUE, GOLD, RED])
    ax.set_yscale("log"); ax.set_xticks(range(3)); ax.set_xticklabels(labels)
    ax.set_ylabel("$w^2$ (лог)", fontsize=9)
    ax.set_title("один эффект, три единицы", fontsize=9.5)
    assert pen[0] == 1.0
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, SIDE / "units.png")


# ============================================================ problems (rule P13)
def check_problems():
    """Every problem of the lesson solved numerically, every printed answer asserted."""
    # --- 3 points: mean vs median on 1,2,2,3,100 and 1,2,2,3,1000
    a = np.array([1, 2, 2, 3, 100], float); b = np.array([1, 2, 2, 3, 1000], float)
    print(f"P3: mean {a.mean():.2f} -> {b.mean():.2f}, median {np.median(a):.1f} -> {np.median(b):.1f}")
    assert abs(a.mean() - 21.6) < 0.05 and abs(b.mean() - 201.6) < 0.004
    assert np.median(a) == 2 and np.median(b) == 2

    # --- exercise "six delays": 2,3,3,4,4,60 and the same with 600
    d = np.array([2, 3, 3, 4, 4, 60], float); d2 = np.array([2, 3, 3, 4, 4, 600], float)
    print(f"EX: mean {d.mean():.3f} -> {d2.mean():.3f}, median {np.median(d):.1f}")
    assert abs(d.mean() - 12.67) < 0.005 and abs(d2.mean() - 102.7) < 0.04
    assert np.median(d) == 3.5 and np.median(d2) == 3.5

    # --- 4 points: robust vs non-robust scale on the residual set (ddof stated explicitly)
    r = np.array([-210, -160, -40, 10, 55, 90, 140, 2000], float)
    s0, s1 = float(r.std(ddof=0)), float(r.std(ddof=1))
    mad_p = float(np.median(np.abs(r - np.median(r)))); sig_p = 1.4826 * mad_p
    d_rob, d_nonrob = 1.35 * sig_p, 1.35 * s1
    w_rob, w_nonrob = d_rob / 2000, d_nonrob / 2000
    print(f"P4: std(ddof=0) {s0:.2f}  std(ddof=1) {s1:.2f}  MAD {mad_p:.1f}  "
          f"sigma_hat {sig_p:.2f}  ratio s1/sigma {s1 / sig_p:.3f}")
    print(f"P4: delta_rob {d_rob:.2f} (weight {w_rob:.3f}), "
          f"delta_nonrob {d_nonrob:.2f} (weight {w_nonrob:.3f}); "
          f"damped by the robust threshold: {[float(v) for v in r if abs(v) > d_rob]}")
    assert abs(s0 - 676.20) < 0.005 and abs(s1 - 722.89) < 0.004
    assert mad_p == 90.0 and abs(sig_p - 133.43) < 0.004
    assert abs(s1 / sig_p - 5.42) < 0.005
    assert round(d_rob, 1) == 180.1 and round(d_nonrob, 1) == 975.9
    assert abs(d_rob - 180.136) < 0.0005 and abs(d_nonrob - 975.902) < 5e-3
    assert abs(w_rob - 0.090) < 0.0005 and abs(w_nonrob - 0.488) < 5e-4
    # the honest statement: BOTH thresholds damp the outlier - the non-robust one only twofold.
    assert d_nonrob < 2000 and d_rob < 2000
    assert [float(v) for v in r if abs(v) > d_rob] == [-210.0, 2000.0]
    assert [float(v) for v in r if abs(v) > d_nonrob] == [2000.0]
    # Checked: enlarging the outlier does NOT make the non-robust threshold harmless.
    for big in (6000.0, 1e5, 1e7):
        rb = np.array([-210, -160, -40, 10, 55, 90, 140, big], float)
        assert 1.35 * rb.std(ddof=1) < big          # delta ~ 0.45*M < M for every M
    print("P4: for n=8 the non-robust delta is ~0.45*M for ANY outlier M -> never harmless")

    # --- 5 points: soft threshold, zero segment length 2*lambda
    lam = 1.7
    soft = lambda t: np.sign(t) * np.maximum(np.abs(t) - lam, 0.0)
    grid = np.linspace(-5, 5, 100001)
    zero_len = float(np.sum(np.abs(soft(grid)) < 1e-12)) * (grid[1] - grid[0])
    assert abs(zero_len - 2 * lam) < 1e-3, zero_len
    # ridge counterpart under the SAME 1/2n normalization: min 0.5(w-a)^2 + lam*w^2
    a3 = 3.0
    ws = np.linspace(-5, 5, 2000001)
    w_star = ws[np.argmin(0.5 * (ws - a3) ** 2 + lam * ws ** 2)]
    assert abs(w_star - a3 / (1 + 2 * lam)) < 1e-4, (w_star, a3 / (1 + 2 * lam))
    print(f"P5: zero segment length {zero_len:.4f} = 2*lambda; "
          f"ridge analogue a/(1+2*lambda) = {a3 / (1 + 2 * lam):.4f} (numeric {w_star:.4f})")

    # --- 5 points: the real spectrum, condition numbers and shrink factors
    print("P-spectrum: " + "  ".join(f"{v:.2f}" for v in SPECTRUM))
    print(f"P-spectrum: kappa {KAPPA0:.2f} -> {KAPPA_A[100]:.2f} (alpha=100) "
          f"-> {KAPPA_A[4420]:.4f} (alpha=4420); shrink max {SHRINK100[0]:.3f}, "
          f"min {SHRINK100[-1]:.3f}")

    # --- 5 points: units. sd 100 km vs sd 2 C
    print("P-units: an effect of one sd costs w1*100 vs w2*2; to make the penalty comparable "
          "the columns are divided by their sd, and w_raw = w_z / sd")
    assert abs(100 / 2 - 50) < 1e-12       # a 50-fold difference in what lambda*w^2 costs

    # --- 5 points: two identical columns (asserted above on the real duplicated table)
    print(f"P8: ridge at lambda->0+ splits {OLS_W[I_S1]:.2f} into two equal halves "
          f"{R_DUP0[I_S1]:.2f}; at alpha=10 the halves are {R_DUP10[I_S1]:.2f} each and the "
          f"SUM {R_DUP10[I_S1] + R_DUP10[10]:.2f} is shrunk further")

    # --- 6 points: the CV numbers quoted in the problem statement
    print(f"P9: CV R2 OLS {CV_OLS:.3f}, ridge100 {CV_RIDGE100:.3f}, lasso1 {CV_LASSO1:.3f}")

    # --- numbers of the widget paragraph, reproduced with sklearn (rule P12).
    # The widget's own solvers were additionally run in node; see the header of
    # public/interactive/widgets/shrinkage-lab.js for the transcript.
    w3 = Lasso(alpha=10 ** 1.25, max_iter=200000).fit(Z, Y)
    r2_3 = float(w3.score(Z, Y)); r2_full = float(LinearRegression().fit(Z, Y).score(Z, Y))
    print(f"widget: lasso at alpha=10^1.25={10 ** 1.25:.1f} keeps "
          f"{int(np.sum(np.abs(w3.coef_) > 1e-8))} features, train R2 {r2_3:.4f} "
          f"vs full {r2_full:.4f}")
    assert int(np.sum(np.abs(w3.coef_) > 1e-8)) == 3
    assert abs(r2_3 - 0.392) < 0.0005 and abs(r2_full - 0.518) < 5e-4
    assert round(10 ** 1.25) == 18
    assert round(2 * SIGMA_HAT) == 387       # prose and caption of fig 51.5

    # --- ridge counterpart of the soft threshold under the 1/2n normalization
    assert abs(3 / (1 + 2 * 3) - 0.43) < 0.005 and abs(3 / (1 + 2 * 100) - 0.015) < 5e-4

    # --- the m^2 -> ft^2 conversion quoted in the collinearity section (rule P14)
    FT_PER_M = 0.3048
    assert abs(1 / FT_PER_M ** 2 - 10.7639) < 5e-05
    print(f"units: 1 m^2 = {1 / FT_PER_M ** 2:.4f} ft^2")


def fig_prox_vs_sub():
    """Разреженность делает не функционал, а последний шаг метода.

    Одна и та же задача lasso на z-стандартизованном диабете решается двумя
    способами: субградиентным спуском и проксимальным шагом. Значение цели у них
    почти одинаково, но точные нули появляются только у второго.
    """
    Xz, y_raw, _names = diabetes_z()
    y = y_raw - y_raw.mean()          # свободный член не штрафуется: убираем его центрированием
    n, p = Xz.shape
    G = Xz.T @ Xz
    Xty = Xz.T @ y
    L = float(np.linalg.eigvalsh(G / n).max())
    lam = 1.0

    def grad(w):
        return (G @ w - Xty) / n

    def obj(w):
        return 0.5 * float(np.sum((Xz @ w - y) ** 2)) / n + lam * float(np.abs(w).sum())

    def soft(v, k):
        return np.sign(v) * np.maximum(np.abs(v) - k, 0.0)

    steps = 400
    w = np.zeros(p); zeros_prox = []; obj_prox = []
    for _ in range(steps):
        w = soft(w - grad(w) / L, lam / L)
        zeros_prox.append(int(np.sum(w == 0))); obj_prox.append(obj(w))
    w_prox = w.copy()

    w = np.zeros(p); zeros_sub = []; obj_sub = []
    for k in range(steps):
        a = 1.0 / (L * np.sqrt(k + 1))
        w = w - a * (grad(w) + lam * np.sign(w))
        zeros_sub.append(int(np.sum(w == 0))); obj_sub.append(obj(w))
    w_sub = w.copy()

    min_abs_sub = float(np.min(np.abs(w_sub)))

    assert zeros_prox[-1] == 3, zeros_prox[-1]
    assert zeros_sub[-1] == 0, zeros_sub[-1]
    assert abs(obj_prox[-1] - 1533.77) < 0.005, obj_prox[-1]
    assert abs(obj_sub[-1] - 1535.30) < 0.005, obj_sub[-1]
    assert abs(min_abs_sub - 0.00038) < 5e-6, min_abs_sub
    assert float(np.min(np.abs(w_prox))) == 0.0

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.0))

    it = np.arange(1, steps + 1)
    a1.plot(it, zeros_prox, color=GREEN, lw=2.4, label="проксимальный шаг")
    a1.plot(it, zeros_sub, color=RED, lw=2.4, label="субградиентный спуск")
    a1.set_xscale("log")
    a1.set_ylim(-0.3, 4.3); a1.set_yticks([0, 1, 2, 3, 4])
    a1.set_xlabel("шаг"); a1.set_ylabel("сколько весов ровно нулевые")
    a1.set_title("Нули появляются только у одного из методов", fontsize=12.5)
    a1.grid(True, color=GRID, lw=0.6, alpha=0.6); a1.set_axisbelow(True)
    a1.legend(frameon=False, fontsize=10, loc="upper left")

    idx = np.argsort(np.abs(w_prox))
    xs = np.arange(p)
    a2.semilogy(xs, np.maximum(np.abs(w_sub[idx]), 1e-6), color=RED, lw=0, marker="o",
                ms=6, label="субградиент: нулей нет")
    vis = np.maximum(np.abs(w_prox[idx]), 1e-6)
    a2.semilogy(xs, vis, color=GREEN, lw=0, marker="s", ms=6, label="проксимальный: три нуля")
    a2.axhline(1e-6, color=LINE, lw=1.0, ls=(0, (4, 3)))
    a2.text(0.15, 1.35e-6, "уровень «ровно ноль»", color=MUTED, fontsize=9.5)
    a2.set_xticks(xs); a2.set_xticklabels([NAMES[i] for i in idx], fontsize=9)
    a2.set_ylabel("$|w_j|$"); a2.set_ylim(5e-7, 1e3)
    a2.set_title(f"Наименьший вес субградиента: {min_abs_sub:.5f}".replace(".", ","), fontsize=12.5)
    a2.grid(True, color=GRID, lw=0.6, alpha=0.6, axis="y"); a2.set_axisbelow(True)
    a2.legend(frameon=False, fontsize=10, loc="upper left")

    fig.suptitle("Разреженность делает последний шаг метода, а не функционал", y=1.03, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "prox_vs_sub.png")
    print(f"prox_vs_sub: zeros {zeros_sub[-1]} vs {zeros_prox[-1]}, obj {obj_sub[-1]:.2f} vs {obj_prox[-1]:.2f}, min|w_sub|={min_abs_sub:.5f}")


check_problems()

fig_losses()
fig_breakdown()
fig_bike_robust()
fig_leverage()
fig_delta_mad()
fig_collinear()
fig_conditioning()
fig_lasso_path()
fig_prox_vs_sub()
fig_elastic()
fig_bias_variance()
side_influence()
side_diamond()
side_prior()
side_scale()
print("lesson 51 figures written")
