"""Deterministic figures for lesson 05: statistics, ML and AI systems.

Main figures live in public/figures/lessons/05, margin schemes in
public/figures/sidenotes/05. Figures 4-6 are built from the real MTA daily
ridership extract stored in scripts/data/mta-daily-ridership.csv.
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "05"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "05"
DATA = ROOT / "scripts" / "data" / "mta-daily-ridership.csv"

PAPER = "#fffef9"
INK = "#171915"
MUTED = "#6e726a"
FAINT = "#969990"
GRID = "#deddd4"
LINE = "#c9c8be"
BLUE = "#315f8c"
RED = "#b94a3b"
GREEN = "#38735d"
GOLD = "#a57920"
VIOLET = "#6f5a8f"
WASH = "#f5f3ea"

mpl.rcParams.update(
    {
        "font.family": "PT Sans",
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "axes.edgecolor": LINE,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.16,
        "mathtext.fontset": "dejavuserif",
    }
)


def save(fig: plt.Figure, path: Path, *, dpi: int = 160) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    width: float = 1.35,
    rad: float = 0.0,
    mutation: float = 13,
    style: str = "-|>",
    linestyle: str = "solid",
    shrink_a: float = 0.0,
    shrink_b: float = 0.0,
    zorder: int = 3,
) -> FancyArrowPatch:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        connectionstyle=f"arc3,rad={rad}",
        color=color,
        linewidth=width,
        linestyle=linestyle,
        mutation_scale=mutation,
        shrinkA=shrink_a,
        shrinkB=shrink_b,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    size: tuple[float, float],
    *,
    face: str = PAPER,
    edge: str = LINE,
    linewidth: float = 1.2,
    rounding: float = 0.018,
    zorder: int = 2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        *size,
        boxstyle=f"round,pad=0.012,rounding_size={rounding}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def load_series() -> tuple[list[date], np.ndarray]:
    days: list[date] = []
    buses: list[int] = []
    with DATA.open() as handle:
        for row in csv.DictReader(handle):
            raw = row["buses_total_estimated_ridersip"]
            if not raw:
                continue
            days.append(date.fromisoformat(row["date"][:10]))
            buses.append(int(raw))
    order = np.argsort(np.array(days))
    days = [days[i] for i in order]
    return days, np.array(buses, dtype=float)[order]


DAYS, BUSES = load_series()
BY_DATE = dict(zip(DAYS, BUSES))


# ---------------------------------------------------------------- figure 5.1
def layers_map() -> None:
    fig, ax = plt.subplots(figsize=(14.2, 5.9))
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 5.9)
    ax.axis("off")

    fig.text(
        0.055,
        0.955,
        "Одна задача, три звена, три вида гарантий",
        fontsize=19,
        fontweight="bold",
    )

    nodes = [
        (0.55, "Наблюдения", "таблица дней:\nответы, поездки,\nпогода, календарь", MUTED, "", ""),
        (4.05, "Оценка", "что верно об\nисточнике данных", BLUE, "статистика", "провал: смещённая\nрамка выборки"),
        (7.55, "Прогноз", "как правило ошибётся\nна новых данных", GREEN, "машинное обучение", "провал: утечка будущего,\nслабая базовая линия"),
        (11.05, "Действие", "что произойдёт\nпри развёртывании", RED, "система ИИ", "провал: кольцо обратной\nсвязи, неверные цены"),
    ]

    box_w, box_h = 2.6, 1.62
    box_y = 2.95
    for x, title, sub, color, tag, fail in nodes:
        rounded_box(ax, (x, box_y), (box_w, box_h), face=WASH if tag else PAPER, edge=color, linewidth=1.5, rounding=0.09)
        ax.text(x + box_w / 2, box_y + box_h - 0.42, title, ha="center", va="center", fontsize=15.5, fontweight="bold", color=INK)
        ax.text(x + box_w / 2, box_y + 0.52, sub, ha="center", va="center", fontsize=10.2, color=MUTED, linespacing=1.35)
        if tag:
            rounded_box(ax, (x + 0.28, box_y + box_h + 0.18), (box_w - 0.56, 0.5), face=color, edge=color, rounding=0.07)
            ax.text(x + box_w / 2, box_y + box_h + 0.435, tag, ha="center", va="center", fontsize=10.6, color=PAPER, fontweight="bold")
        if fail:
            ax.text(x + box_w / 2, box_y - 0.62, fail, ha="center", va="center", fontsize=9.6, color=RED, linespacing=1.35)

    mid = box_y + box_h / 2
    for left, right in [(0.55, 4.05), (4.05, 7.55), (7.55, 11.05)]:
        arrow(ax, (left + box_w + 0.06, mid), (right - 0.06, mid), color=INK, width=1.7, mutation=16)

    arrow(
        ax,
        (11.05 + box_w / 2, box_y - 1.12),
        (0.55 + box_w / 2, box_y - 1.12 + 0.001),
        color=GOLD,
        width=1.7,
        rad=0.0,
        mutation=16,
    )
    ax.plot([11.05 + box_w / 2, 11.05 + box_w / 2], [box_y - 1.02, box_y - 1.12], color=GOLD, lw=1.7)
    ax.plot([0.55 + box_w / 2, 0.55 + box_w / 2], [box_y - 1.12, box_y + 0.0 - 0.06], color=GOLD, lw=1.7)
    ax.text(
        (0.55 + 11.05 + box_w) / 2,
        box_y - 1.38,
        "завтрашние данные приходят из мира, изменённого сегодняшним действием",
        ha="center",
        va="center",
        fontsize=10.6,
        color=GOLD,
    )

    save(fig, OUT / "layers-map.png")


# ---------------------------------------------------------------- figure 5.2
def literary_digest() -> None:
    fig = plt.figure(figsize=(12.8, 5.8))
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.80])
    ax.set_xlim(38, 68)
    ax.set_ylim(0, 5.8)
    ax.axis("off")

    # Visually circular bubbles: data-units-per-inch ratio between axes.
    x_per_in = 30 / (12.8 * 0.92)
    y_per_in = 5.8 / (5.8 * 0.80)
    circle_w = x_per_in / y_per_in

    fact = 60.8
    axis_y = 1.15

    ax.axhline(axis_y, color=INK, lw=1.2, zorder=2)
    for tick in range(40, 66, 5):
        ax.plot([tick, tick], [axis_y - 0.09, axis_y + 0.09], color=INK, lw=1.0, zorder=2)
        ax.text(tick, axis_y + 0.16, str(tick), fontsize=10, ha="center", va="bottom", color=MUTED)
    ax.text(67.6, axis_y - 0.35, "доля голосов\nза Рузвельта, %", fontsize=9.8, ha="right",
            va="top", color=MUTED, linespacing=1.3)

    scale = 1.55 / np.sqrt(2_380_000)

    # Literary Digest: huge biased sample, label fits inside the bubble.
    r_d = scale * np.sqrt(2_380_000)
    c_d = axis_y + 0.62 + r_d
    ax.add_patch(mpl.patches.Ellipse((42.9, c_d), 2 * r_d * circle_w, 2 * r_d,
                 facecolor=RED, alpha=0.22, edgecolor=RED, lw=1.6, zorder=3))
    ax.plot([42.9, 42.9], [axis_y, axis_y + 0.52], color=RED, lw=1.8, zorder=4)
    ax.plot(42.9, axis_y, "o", color=RED, ms=6, zorder=5)
    ax.text(42.9, c_d + 0.34, "Literary Digest\n42,9%", fontsize=12.4, fontweight="bold",
            color=RED, ha="center", va="center", linespacing=1.4, zorder=6)
    ax.text(42.9, c_d - 0.78, "2 380 000 ответов;\nрамка: телефонные книги\nи списки автовладельцев",
            fontsize=9.8, color=MUTED, ha="center", va="center", linespacing=1.4, zorder=6)

    # Gallup: small honest sample, labels beside the bubble.
    r_g = scale * np.sqrt(50_000)
    c_g = axis_y + 0.62 + r_g
    ax.add_patch(mpl.patches.Ellipse((55.7, c_g), 2 * r_g * circle_w, 2 * r_g,
                 facecolor=BLUE, alpha=0.30, edgecolor=BLUE, lw=1.6, zorder=3))
    ax.plot([55.7, 55.7], [axis_y, axis_y + 0.52], color=BLUE, lw=1.8, zorder=4)
    ax.plot(55.7, axis_y, "o", color=BLUE, ms=6, zorder=5)
    ax.text(55.7, c_g + r_g + 0.28, "Гэллап: 55,7%", fontsize=12.4, fontweight="bold",
            color=BLUE, ha="center", va="bottom", zorder=6)
    ax.text(55.7, c_g + r_g + 0.78, "около 50 000 ответов", fontsize=9.8,
            color=MUTED, ha="center", va="bottom", zorder=6)

    ax.plot([fact, fact], [axis_y - 0.28, axis_y + 3.55], color=INK, lw=1.6, zorder=4)
    ax.text(fact + 0.4, axis_y + 3.4, "факт: 60,8%", fontsize=11.6, fontweight="bold",
            color=INK, ha="left", va="center")

    err_y = axis_y - 0.55
    arrow(ax, (42.9, err_y), (fact, err_y), color=RED, width=1.2, mutation=11, style="<|-|>", zorder=5)
    ax.plot([42.9, 42.9], [axis_y - 0.09, err_y], color=RED, lw=0.9, linestyle=(0, (2, 3)), zorder=2)
    ax.plot([fact, fact], [axis_y - 0.28, err_y], color=RED, lw=0.9, linestyle=(0, (2, 3)), zorder=2)
    ax.text(51.85, err_y - 0.22, "промах 17,9 пункта; случайная ошибка такой выборки — 0,03 пункта",
            fontsize=10.2, color=RED, ha="center", va="top")

    ax.text(67.6, 4.5, "площадь круга —\nчисло ответов", fontsize=9.8, color=MUTED,
            ha="right", va="center", linespacing=1.35)

    fig.text(0.04, 0.925, "Выборы 1936 года: гигантская смещённая выборка против маленькой честной",
             fontsize=15, fontweight="bold")
    save(fig, OUT / "literary-digest.png")


# ---------------------------------------------------------------- figure 5.3
def interval_coverage() -> None:
    rng = np.random.default_rng(1936)
    n = 500
    p_true = 0.55
    p_frame = 0.62
    reps = 100

    fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.4), sharex=False)
    for ax, p_sample, title, note in [
        (
            axes[0],
            p_true,
            "Честная случайная выборка: промахивается примерно каждый двадцатый интервал",
            "покрытие",
        ),
        (
            axes[1],
            p_frame,
            "Та же формула при смещённой рамке: интервалы не расширились, а дружно легли мимо",
            "покрытие",
        ),
    ]:
        hits = 0
        for k in range(reps):
            phat = rng.binomial(n, p_sample) / n
            se = np.sqrt(phat * (1 - phat) / n)
            lo, hi = phat - 1.96 * se, phat + 1.96 * se
            covered = lo <= p_true <= hi
            hits += covered
            color = FAINT if covered else RED
            lw = 1.1 if covered else 1.8
            ax.plot([k, k], [lo, hi], color=color, lw=lw, zorder=2)
            ax.plot(k, phat, "o", color=color, ms=2.4, zorder=3)
        ax.axhline(p_true, color=BLUE, lw=1.5, zorder=4)
        ax.text(100.8, p_true, "истинная доля\np = 0,55", fontsize=10.2, color=BLUE, ha="left", va="center", linespacing=1.3)
        ax.set_xlim(-2, 113)
        ax.set_ylim(0.44, 0.72)
        ax.set_yticks([0.45, 0.5, 0.55, 0.6, 0.65, 0.7])
        ax.set_yticklabels(["0,45", "0,50", "0,55", "0,60", "0,65", "0,70"])
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=10)
        ax.text(
            0,
            0.455,
            f"{note}: {hits} из {reps}",
            fontsize=11.5,
            fontweight="bold",
            color=GREEN if hits > 80 else RED,
            ha="left",
        )
    axes[1].set_xlabel("номер повтора опроса (n = 500 в каждом)")
    fig.subplots_adjust(hspace=0.42, top=0.94, bottom=0.09, left=0.07, right=0.88)
    save(fig, OUT / "interval-coverage.png")


# ---------------------------------------------------------------- figure 5.4
def weekly_rhythm() -> None:
    start, end = date(2024, 9, 23), date(2024, 12, 1)
    window = [(d, v) for d, v in zip(DAYS, BUSES) if start <= d <= end]
    xs = [d for d, _ in window]
    ys = np.array([v for _, v in window]) / 1000

    year = [(d, v) for d, v in zip(DAYS, BUSES) if d.year == 2024]
    profile = [np.mean([v for d, v in year if d.weekday() == wd]) / 1000 for wd in range(7)]

    fig, (ax, axp) = plt.subplots(
        1, 2, figsize=(14.2, 5.6), gridspec_kw={"width_ratios": [2.6, 1.0], "wspace": 0.16}
    )

    for d, v in window:
        if d.weekday() >= 5:
            ax.axvspan(d, d + timedelta(days=1), color=WASH, zorder=0)
    ax.plot(xs, ys, color=BLUE, lw=1.6, zorder=3)
    ax.plot(xs, ys, "o", color=BLUE, ms=3, zorder=4)

    tg = date(2024, 11, 28)
    ax.annotate(
        "28 ноября:\nДень благодарения",
        xy=(tg, BY_DATE[tg] / 1000),
        xytext=(date(2024, 11, 2), 480),
        fontsize=10.2,
        color=RED,
        ha="left",
        va="center",
        linespacing=1.3,
        arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 1.2, "shrinkB": 4},
    )
    ax.set_ylabel("тыс. поездок в день")
    ax.set_ylim(380, 1580)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Десять недель осени 2024 года", loc="left", fontweight="bold", pad=10)
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%d.%m"))
    ax.tick_params(axis="x", labelsize=10)
    ax.text(xs[1], 452, "серые полосы — выходные", fontsize=10.2, color=MUTED)

    labels = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    colors = [BLUE] * 5 + [GOLD] * 2
    axp.bar(range(7), profile, color=colors, width=0.66, zorder=3)
    for i, v in enumerate(profile):
        axp.text(i, v + 26, f"{v:.0f}", ha="center", fontsize=9.6, color=MUTED)
    axp.set_xticks(range(7), labels)
    axp.set_ylim(0, 1580)
    axp.grid(axis="y", color=GRID, lw=0.7)
    axp.spines["top"].set_visible(False)
    axp.spines["right"].set_visible(False)
    axp.set_title("Средний профиль недели, 2024", loc="left", fontweight="bold", pad=10)
    save(fig, OUT / "mta-weekly-rhythm.png")


# ---------------------------------------------------------------- figure 5.5
def naive_errors() -> None:
    year = [(d, v) for d, v in zip(DAYS, BUSES) if d.year == 2024]
    errors = []
    for d, v in year:
        prev = BY_DATE.get(d - timedelta(days=7))
        if prev is not None:
            errors.append((d, (prev - v) / 1000))
    xs = [d for d, _ in errors]
    ys = np.array([e for _, e in errors])

    fig, ax = plt.subplots(figsize=(14.2, 5.9))
    ax.axhline(0, color=LINE, lw=1.1)
    small = np.abs(ys) < 320
    ax.plot(np.array(xs)[small], ys[small], "o", ms=3.2, color=FAINT, zorder=2)
    ax.plot(np.array(xs)[~small], ys[~small], "o", ms=5.2, color=RED, zorder=3)

    notes = [
        (date(2024, 12, 25), "25.12  Рождество", (-18, 26)),
        (date(2024, 12, 5), "5.12  неделя после\nДня благодарения", (-102, -2)),
        (date(2024, 11, 28), "28.11  День\nблагодарения", (-88, 8)),
        (date(2024, 9, 9), "9.09  неделя после\nДня труда", (-88, -4)),
        (date(2024, 1, 8), "8.01  неделя после\nНового года", (18, -14)),
        (date(2024, 2, 13), "13.02  снежный шторм:\nпризнак «погода»", (22, 22)),
        (date(2024, 6, 3), "3.06  неделя после\nДня поминовения", (-96, -4)),
        (date(2024, 5, 27), "27.05  День\nпоминовения", (16, 28)),
    ]
    for when, label, (dx, dy) in notes:
        val = dict(errors)[when]
        ax.annotate(
            label,
            xy=(when, val),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9.8,
            color=INK,
            linespacing=1.3,
            ha="left" if dx > 0 else "right",
            va="center",
            arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.9, "shrinkB": 3},
        )

    ax.set_ylabel("ошибка правила «как неделю назад», тыс. поездок")
    ax.set_ylim(-1150, 1150)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    month_starts = [date(2024, m, 1) for m in range(1, 13)]
    month_names = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]
    ax.set_xticks(month_starts, month_names)
    ax.set_title(
        "2024 год: крупные ошибки наивного прогноза не случайны — они читают календарь",
        loc="left",
        fontweight="bold",
        pad=12,
    )
    ax.text(
        date(2024, 1, 2),
        -1090,
        "выше нуля — перелёт (праздник тише прогноза), ниже нуля — недолёт (обычный день после праздника)",
        fontsize=10.4,
        color=MUTED,
        va="center",
    )
    save(fig, OUT / "naive-errors-calendar.png")


# ---------------------------------------------------------------- figure 5.6
def regime_shift() -> None:
    values = BUSES / 1e6
    kernel = np.ones(7) / 7
    smooth = np.convolve(values, kernel, mode="valid")
    smooth_days = DAYS[3 : 3 + len(smooth)]

    fig, ax = plt.subplots(figsize=(14.2, 5.4))
    free_a, free_b = date(2020, 3, 23), date(2020, 8, 31)
    ax.axvspan(free_a, free_b, color=WASH, zorder=0)
    ax.plot(smooth_days, smooth, color=BLUE, lw=1.7, zorder=3)

    ax.annotate(
        "первая неделя марта 2020:\nв будни около 2,2 млн поездок",
        xy=(date(2020, 3, 7), 1.86),
        xytext=(date(2020, 10, 20), 2.12),
        fontsize=10.4,
        color=INK,
        linespacing=1.35,
        va="center",
        arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.1, "shrinkB": 4},
    )
    ax.text(
        date(2020, 4, 12),
        0.30,
        "проезд бесплатный,\nоценка почти не видит\nпассажиров",
        fontsize=10.2,
        color=GOLD,
        ha="left",
        va="bottom",
        linespacing=1.35,
    )
    ax.annotate(
        "новое плато: около 1,3 млн,\nна 40% ниже прежнего уровня",
        xy=(date(2023, 6, 1), 1.34),
        xytext=(date(2022, 3, 10), 1.86),
        fontsize=10.4,
        color=INK,
        linespacing=1.35,
        va="center",
        arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.1, "shrinkB": 4},
    )

    ax.set_ylabel("млн поездок в день, сглаживание 7 дней")
    ax.set_ylim(0, 2.4)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        "1776 дней автобусов Нью-Йорка: сдвиг мира и поломка прибора в одной колонке",
        loc="left",
        fontweight="bold",
        pad=12,
    )
    save(fig, OUT / "regime-shift.png")


# ---------------------------------------------------------------- figure 5.7
def feedback_loop() -> None:
    fig, ax = plt.subplots(figsize=(12.4, 6.2))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    bw, bh = 1.95, 1.25
    box_y = 3.05
    nodes = [
        (0.85, "Признаки дня\n$x_t$", MUTED),
        (3.85, "Прогноз\n$\\hat y_t$", GREEN),
        (6.85, "Действие\n$a_t$", RED),
        (9.85, "Следующий день\n$x_{t+1}$", MUTED),
    ]
    for x, label, color in nodes:
        rounded_box(ax, (x, box_y), (bw, bh), face=PAPER, edge=color, linewidth=1.5, rounding=0.09)
        ax.text(x + bw / 2, box_y + bh / 2, label, ha="center", va="center", fontsize=12.6, linespacing=1.4)

    mid = box_y + bh / 2
    for left, right in [(0.85, 3.85), (3.85, 6.85), (6.85, 9.85)]:
        arrow(ax, (left + bw + 0.06, mid), (right - 0.06, mid), color=INK, width=1.6, mutation=15)

    arrow(
        ax,
        (9.85 + bw / 2, box_y - 0.07),
        (0.85 + bw / 2, box_y - 0.07),
        color=INK,
        width=1.6,
        rad=-0.30,
        mutation=15,
    )
    ax.text(6.15, 1.06, "новый день становится новыми признаками", fontsize=10.6, color=MUTED, ha="center")

    table = (5.35, 5.42)
    rounded_box(ax, (table[0] - 1.55, table[1] - 0.37), (3.1, 0.78), face=WASH, edge=GOLD, linewidth=1.3, rounding=0.08)
    ax.text(table[0], table[1], "таблица переполнений,\nпо которой модель переобучается",
            ha="center", va="center", fontsize=10.2, color=INK, linespacing=1.35)
    arrow(
        ax,
        (6.85 + bw / 2, box_y + bh + 0.07),
        (table[0] + 1.62, table[1] - 0.16),
        color=GOLD,
        width=1.5,
        rad=0.22,
        mutation=13,
        linestyle=(0, (5, 4)),
    )
    arrow(
        ax,
        (table[0] - 1.62, table[1] - 0.16),
        (3.85 + bw / 2, box_y + bh + 0.07),
        color=GOLD,
        width=1.5,
        rad=0.22,
        mutation=13,
        linestyle=(0, (5, 4)),
    )
    ax.text(
        12.05,
        5.42,
        "выпущенный резерв стирает\nпереполнения из данных:\nметрика меряет саму систему",
        fontsize=10.0,
        color=GOLD,
        ha="right",
        va="center",
        linespacing=1.35,
    )

    fig.text(0.055, 0.93, "Кольцо диспетчерской системы", fontsize=18, fontweight="bold")
    save(fig, OUT / "feedback-loop.png")


# ------------------------------------------------------------- margin schemes
def sampling_frame() -> None:
    rng = np.random.default_rng(1936)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.set_xlim(0, 6.4)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    rounded_box(ax, (0.25, 0.35), (5.9, 3.6), face=PAPER, edge=INK, linewidth=1.4, rounding=0.1)
    ax.text(0.45, 4.13, "все избиратели 1936 года", fontsize=11.5, fontweight="bold", ha="left", va="bottom")

    rounded_box(ax, (3.35, 0.62), (2.52, 3.02), face=WASH, edge=GOLD, linewidth=1.5, rounding=0.09)
    ax.text(4.61, 3.32, "рамка Digest:\nтелефон или автомобиль", fontsize=9.6, color=GOLD, ha="center", va="center", linespacing=1.3, zorder=4)

    pts = rng.uniform([0.5, 0.62], [6.0, 3.72], size=(240, 2))
    for x, y in pts:
        if 3.42 <= x <= 5.8 and y > 2.95:
            continue
        inside = 3.42 <= x <= 5.8 and 0.7 <= y <= 2.95
        landon = rng.random() < (0.55 if inside else 0.30)
        ax.plot(x, y, "o", ms=2.6, color=RED if landon else BLUE, alpha=0.85, zorder=3)

    ax.plot([], [], "o", color=BLUE, label="за Рузвельта")
    ax.plot([], [], "o", color=RED, label="за Лэндона")
    ax.legend(loc="lower left", fontsize=9.3, frameon=False, handletextpad=0.2, borderaxespad=0.1)
    save(fig, SIDE / "sampling-frame.png")


def stratified_split() -> None:
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.set_xlim(0, 6.4)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    strata = [
        (0.3, 2.7, "пешком, 50%", BLUE),
        (3.1, 1.7, "транспортом, 30%", GREEN),
        (4.9, 1.2, "на автомобиле, 20%", GOLD),
    ]
    for x0, width, label, color in strata:
        rounded_box(ax, (x0, 2.0), (width, 2.1), face=PAPER, edge=color, linewidth=1.5, rounding=0.07)
        ax.text(x0 + width / 2, 4.36, label, fontsize=9.8, color=color, ha="center", va="bottom", fontweight="bold")
        pts = rng.uniform([x0 + 0.14, 2.14], [x0 + width - 0.14, 3.96], size=(int(width * 13), 2))
        ax.plot(pts[:, 0], pts[:, 1], "o", ms=2.2, color=FAINT, zorder=2)
        chosen = pts[rng.choice(len(pts), size=max(3, int(width * 3)), replace=False)]
        ax.plot(chosen[:, 0], chosen[:, 1], "o", ms=3.6, color=color, zorder=3)
        arrow(ax, (x0 + width / 2, 1.92), (x0 + width / 2, 1.28), color=color, width=1.3, mutation=11)
        ax.text(x0 + width / 2, 1.02, f"$n_{{{strata.index((x0, width, label, color)) + 1}}}$", fontsize=11, ha="center", color=color)

    ax.text(3.2, 0.38, "своя случайная выборка в каждой страте,\nвеса задают состав итоговой оценки", fontsize=9.8, color=MUTED, ha="center", linespacing=1.35)
    save(fig, SIDE / "stratified-split.png")


def rolling_origin() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.set_xlim(0, 6.4)
    ax.set_ylim(0, 4.0)
    ax.axis("off")

    for row, train_len in enumerate([2.4, 3.1, 3.8]):
        y = 3.0 - row * 1.05
        ax.plot([0.5, 0.5 + train_len], [y, y], color=BLUE, lw=7, solid_capstyle="butt", zorder=2)
        ax.plot(0.5 + train_len + 0.28, y, "s", ms=9, color=RED, zorder=3)
        ax.text(0.5 + train_len + 0.58, y, f"проверка: день $T_{row + 1}+1$", fontsize=9.8, va="center", color=MUTED)
        ax.plot([0.5 + train_len + 0.02, 0.5 + train_len + 0.02], [y - 0.17, y + 0.17], color=INK, lw=1.2)
        ax.text(0.5 + train_len + 0.02, y + 0.30, f"$T_{row + 1}$", fontsize=10, ha="center", color=INK)

    ax.text(0.5, 3.52, "обучение: только прошлое до границы", fontsize=10, color=BLUE, ha="left")
    ax.text(0.5, 0.42, "граница сдвигается, будущее никогда\nне попадает в обучение", fontsize=9.8, color=MUTED, ha="left", linespacing=1.35)
    save(fig, SIDE / "rolling-origin.png")


if __name__ == "__main__":
    layers_map()
    literary_digest()
    interval_coverage()
    weekly_rhythm()
    naive_errors()
    regime_shift()
    feedback_loop()
    sampling_frame()
    stratified_split()
    rolling_origin()
    print("lesson 05 figures written to", OUT, "and", SIDE)
