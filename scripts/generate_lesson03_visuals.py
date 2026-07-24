from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "03"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "03"

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


CO_48 = np.array(
    [
        2.7,
        2.8,
        2.7,
        4.5,
        3.5,
        2.6,
        1.7,
        1.3,
        1.3,
        1.0,
        0.7,
        0.7,
        0.5,
        0.5,
        0.6,
        1.1,
        2.7,
        3.5,
        2.3,
        1.6,
        1.3,
        2.0,
        1.9,
        1.9,
        2.2,
        2.0,
        2.9,
        5.2,
        4.6,
        2.5,
        1.5,
        1.2,
        1.7,
        1.4,
        1.2,
        0.6,
        0.7,
        0.8,
        0.9,
        1.6,
        3.4,
        3.8,
        3.1,
        2.7,
        2.0,
        2.3,
        1.9,
        1.3,
    ]
)


def lcg_uniforms(seed: int, size: int) -> np.ndarray:
    """Return the same sequence as the browser's unsigned 32-bit LCG."""
    state = int(seed) % 2**32
    values: list[float] = []
    for _ in range(size):
        state = (1_664_525 * state + 1_013_904_223) % 2**32
        values.append((state + 0.5) / 2**32)
    return np.asarray(values)


def calibrated_probabilities(raw_scores: np.ndarray, target_observed: float) -> np.ndarray:
    """Shift scores so their clipped mean equals the target observation rate."""
    lower, upper = -2.0, 2.0
    for _ in range(80):
        middle = (lower + upper) / 2
        mean_probability = float(np.clip(raw_scores + middle, 0.08, 0.97).mean())
        if mean_probability < target_observed:
            lower = middle
        else:
            upper = middle
    return np.clip(raw_scores + (lower + upper) / 2, 0.08, 0.97)


def missingness_protocol(target_missing: float = 0.36, seed: int = 23) -> list[dict[str, object]]:
    indices = np.arange(CO_48.size)
    local_hours = (16 + indices) % 24
    normalized_co = (CO_48 - CO_48.min()) / (CO_48.max() - CO_48.min())
    raw_scores = [
        np.zeros(CO_48.size),
        np.where((local_hours >= 7) & (local_hours <= 21), 0.84, 0.34),
        0.93 - 0.72 * normalized_co,
    ]
    uniforms = lcg_uniforms(seed, CO_48.size)
    result: list[dict[str, object]] = []
    for raw in raw_scores:
        probability = calibrated_probabilities(raw, 1 - target_missing)
        observed = uniforms < probability
        result.append(
            {
                "probability": probability,
                "observed": observed,
                "expected_missing": 1 - float(probability.mean()),
                "realized_missing": 1 - float(observed.mean()),
            }
        )
    return result


def clean(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRID, linewidth=0.65, alpha=0.8)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, path: Path, *, dpi: int = 160) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    detail: str,
    *,
    edge: str,
    fill: str,
    title_size: float = 12,
    detail_size: float = 9.5,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=fill,
        edgecolor=edge,
        linewidth=1.3,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.64,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
    )
    ax.text(
        x + width / 2,
        y + height * 0.31,
        detail,
        ha="center",
        va="center",
        fontsize=detail_size,
        color=MUTED,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    rad: float = 0,
    width: float = 1.4,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            connectionstyle=f"arc3,rad={rad}",
            color=color,
            linewidth=width,
            mutation_scale=13,
            shrinkA=0,
            shrinkB=0,
            zorder=2,
        )
    )


def sensor_to_row() -> None:
    fig, ax = plt.subplots(figsize=(14.4, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.04,
        0.94,
        "Строка 11/03/2004 04:00 хранит путь измерения, а не сам воздух",
        fontsize=18,
        fontweight="bold",
    )
    ax.text(
        0.04,
        0.895,
        "Реальный фрагмент UCI Air Quality: эталон CO отсутствует, ответ сенсора сохранён.",
        fontsize=11.5,
        color=MUTED,
    )

    box(
        ax,
        0.04,
        0.49,
        0.15,
        0.22,
        "Воздух у дороги",
        "один объект\nв один час",
        edge=BLUE,
        fill="#edf3f7",
    )
    ax.text(
        0.235,
        0.845,
        "ДВЕ ПАРАЛЛЕЛЬНЫЕ ВЕТВИ",
        ha="center",
        fontsize=9.5,
        color=MUTED,
        fontweight="bold",
    )
    ax.plot([0.19, 0.235], [0.60, 0.60], color=INK, linewidth=1.4)
    ax.plot([0.235, 0.235], [0.405, 0.735], color=INK, linewidth=1.4)
    arrow(ax, (0.235, 0.735), (0.285, 0.735), color=GREEN)
    arrow(ax, (0.235, 0.405), (0.285, 0.405), color=RED)

    box(
        ax,
        0.285,
        0.645,
        0.16,
        0.18,
        "MOS-сенсор",
        "чувствительный слой\nменяет сопротивление",
        edge=GREEN,
        fill="#edf4ef",
    )
    box(
        ax,
        0.505,
        0.645,
        0.16,
        0.18,
        "Сигнал и среднее",
        "напряжение за час\nPT08.S1(CO)=1011",
        edge=VIOLET,
        fill="#f1edf6",
    )
    arrow(ax, (0.445, 0.735), (0.495, 0.735), color=GREEN)

    box(
        ax,
        0.285,
        0.315,
        0.16,
        0.18,
        "Анализатор",
        "отдельный эталонный\nканал CO",
        edge=RED,
        fill="#f7ece9",
    )
    box(
        ax,
        0.505,
        0.315,
        0.16,
        0.18,
        "Результат канала",
        "измерения нет\nCO(GT)=−200",
        edge=RED,
        fill="#f7ece9",
    )
    arrow(ax, (0.445, 0.405), (0.495, 0.405), color=RED)

    ax.plot([0.665, 0.715], [0.735, 0.735], color=GREEN, linewidth=1.4)
    ax.plot([0.665, 0.715], [0.405, 0.405], color=RED, linewidth=1.4)
    ax.plot([0.715, 0.715], [0.405, 0.735], color=INK, linewidth=1.4)
    arrow(ax, (0.715, 0.57), (0.755, 0.57), color=INK)
    box(
        ax,
        0.755,
        0.46,
        0.19,
        0.22,
        "Строка и парсер",
        "1011 остаётся числом\n−200 становится NA",
        edge=INK,
        fill=WASH,
    )
    ax.text(
        0.475,
        0.265,
        "Ветви встречаются только при сборке одной строки CSV.",
        ha="center",
        fontsize=10.5,
        color=MUTED,
    )

    ax.add_patch(
        FancyBboxPatch(
            (0.04, 0.08),
            0.92,
            0.13,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=WASH,
            edgecolor=LINE,
            linewidth=1,
        )
    )
    ax.text(0.06, 0.18, "СЫРАЯ СТРОКА", fontsize=9.5, color=MUTED, fontweight="bold")
    ax.text(
        0.06,
        0.13,
        "11/03/2004 ; 04.00.00 ;",
        fontsize=12.5,
        family="monospace",
    )
    ax.text(0.31, 0.13, "−200", color=RED, fontsize=13.5, family="monospace", fontweight="bold")
    ax.text(
        0.357,
        0.13,
        "; 1011 ; 14 ; 1,3 ; 527 ; …",
        fontsize=12.5,
        family="monospace",
    )
    ax.text(0.315, 0.095, "CO(GT)", fontsize=8.5, ha="center", color=RED)
    ax.text(0.42, 0.095, "PT08.S1(CO)", fontsize=8.5, ha="center", color=VIOLET)
    save(fig, OUT / "sensor-to-row.png")


def missingness_mechanisms() -> None:
    indices = np.arange(CO_48.size)
    experiments = missingness_protocol(target_missing=0.36, seed=23)
    titles = [
        "MCAR · случайный сбой",
        "MAR · зависит от местного часа",
        "MNAR · теряются высокие значения",
    ]
    subtitles = [
        "$a_i=0$",
        "$a_i=0{,}84$ в 07–21, иначе $0{,}34$",
        "$a_i=0{,}93-0{,}72z_i$",
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14.4, 8), sharex=True, gridspec_kw={"hspace": 0.25})
    full_mean = float(CO_48.mean())
    for ax, experiment, title, subtitle in zip(axes, experiments, titles, subtitles):
        probability = experiment["probability"]
        observed = experiment["observed"]
        assert isinstance(probability, np.ndarray)
        assert isinstance(observed, np.ndarray)
        observed_mean = float(CO_48[observed].mean())
        ipw_mean = float(np.sum(CO_48[observed] / probability[observed]) / np.sum(1 / probability[observed]))
        expected_missing = float(experiment["expected_missing"])
        realized_missing = float(experiment["realized_missing"])
        ax.plot(indices, CO_48, color=LINE, linewidth=1.2)
        ax.scatter(indices[observed], CO_48[observed], s=28, color=BLUE, zorder=4)
        ax.scatter(
            indices[~observed],
            CO_48[~observed],
            s=48,
            facecolor=PAPER,
            edgecolor=RED,
            linewidth=1.3,
            zorder=4,
        )
        ax.axhline(full_mean, color=INK, linewidth=1, linestyle=(0, (5, 4)))
        ax.axhline(observed_mean, color=RED, linewidth=1.2)
        ax.set_ylim(0, 7.5)
        ax.set_ylabel("CO, мг/м³")
        ax.text(0.01, 0.89, title, transform=ax.transAxes, fontsize=13, fontweight="bold")
        ax.text(
            0.01,
            0.70,
            f"{subtitle} · ожид. пропусков {expected_missing:.1%} · вышло {realized_missing:.1%}",
            transform=ax.transAxes,
            fontsize=10.2,
            color=MUTED,
        )
        ax.text(
            0.99,
            0.87,
            f"полное среднее {full_mean:.2f}  ·  наблюдаемое {observed_mean:.2f}  ·  IPW {ipw_mean:.2f}",
            transform=ax.transAxes,
            ha="right",
            fontsize=10.5,
            color=INK,
            bbox={"boxstyle": "round,pad=0.18", "fc": PAPER, "ec": "none", "alpha": 0.92},
        )
        clean(ax)
    axes[-1].set_xlabel("час от 24/03/2004 16:00 · реальные 48 значений UCI")
    fig.suptitle(
        "Одна ожидаемая доля пропусков, три механизма и общие случайные числа",
        x=0.08,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.94,
        0.955,
        "$p_i=\\mathrm{clip}(a_i+c,0{,}08,0{,}97)$ · $\\overline{p}=0{,}64$ · LCG seed 23",
        ha="right",
        fontsize=10,
        color=MUTED,
    )
    save(fig, OUT / "missingness-mechanisms.png")


def pseudoreplication() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 8), gridspec_kw={"wspace": 0.18})
    titles = [
        "Файл: 48 кадров",
        "Случайный split строк",
        "Split по рейсам",
    ]
    for ax, title in zip(axes, titles):
        ax.set_xlim(-0.5, 12.8)
        ax.set_ylim(-0.8, 4.8)
        ax.axis("off")
        ax.set_title(title, loc="left", pad=16, fontsize=15, fontweight="bold")

    labels = ["рейс A", "рейс B", "рейс C", "рейс D"]
    for row, label in enumerate(labels):
        y = 3.5 - row
        axes[0].add_patch(
            FancyBboxPatch(
                (-0.25, y - 0.28),
                12.4,
                0.56,
                boxstyle="round,pad=0.01,rounding_size=0.08",
                facecolor=WASH,
                edgecolor=LINE,
            )
        )
        axes[0].text(-0.35, y, label, ha="right", va="center", fontsize=10, color=MUTED)
        axes[0].scatter(np.arange(12), np.full(12, y), s=38, color=[BLUE, GREEN, GOLD, VIOLET][row])

    axes[0].text(5.7, -0.3, "48 строк  ≠  48 независимых поездок", ha="center", fontsize=11.5, color=INK)
    axes[0].text(5.7, -0.62, "эффективная единица — рейс", ha="center", fontsize=10.5, color=MUTED)

    rng = np.random.default_rng(12)
    for row, label in enumerate(labels):
        y = 3.5 - row
        mask = rng.random(12) < 0.25
        mask[row + 1] = True
        mask[(row + 5) % 12] = False
        colors = np.where(mask, RED, BLUE)
        axes[1].scatter(np.arange(12), np.full(12, y), s=42, color=colors)
        axes[1].text(-0.35, y, label, ha="right", va="center", fontsize=10, color=MUTED)
        axes[1].plot([0, 11], [y, y], color=GRID, linewidth=1, zorder=0)
    axes[1].text(0.2, -0.08, "train", color=BLUE, fontsize=10.5, fontweight="bold")
    axes[1].text(4.0, -0.08, "test", color=RED, fontsize=10.5, fontweight="bold")
    axes[1].text(5.7, -0.44, "Каждый тестовый рейс\nпредставлен и в train", ha="center", fontsize=11, color=RED)

    for row, label in enumerate(labels):
        y = 3.5 - row
        color = RED if row == 3 else BLUE
        axes[2].scatter(np.arange(12), np.full(12, y), s=42, color=color)
        axes[2].text(-0.35, y, label, ha="right", va="center", fontsize=10, color=MUTED)
        axes[2].plot([0, 11], [y, y], color=GRID, linewidth=1, zorder=0)
    axes[2].text(0.2, -0.08, "train: A–C", color=BLUE, fontsize=10.5, fontweight="bold")
    axes[2].text(5.3, -0.08, "test: D", color=RED, fontsize=10.5, fontweight="bold")
    axes[2].text(5.7, -0.44, "Тест проверяет новый рейс,\nа не новый номер кадра", ha="center", fontsize=11, color=GREEN)

    fig.suptitle(
        "Строки наследуют зависимость объекта, из которого были нарезаны",
        x=0.08,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    save(fig, OUT / "pseudoreplication.png")


def kappa_agreement() -> None:
    matrix = np.array([[42, 8, 2], [6, 28, 5], [1, 7, 31]])
    total = int(matrix.sum())
    observed = float(np.trace(matrix) / total)
    row = matrix.sum(axis=1)
    column = matrix.sum(axis=0)
    expected = float(np.sum(row * column) / total**2)
    kappa = (observed - expected) / (1 - expected)
    labels = ["обычно", "спорно", "опасно"]

    fig = plt.figure(figsize=(14.4, 8))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.0], wspace=0.30)
    ax = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[0, 2])

    image = ax.imshow(matrix, cmap=mpl.colors.LinearSegmentedColormap.from_list("quiet", [PAPER, "#d7e3eb", BLUE]), vmin=0, vmax=45)
    del image
    for i in range(3):
        for j in range(3):
            color = PAPER if matrix[i, j] > 30 else INK
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=18, color=color, fontweight="bold")
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set_xlabel("разметчик B")
    ax.set_ylabel("разметчик A")
    ax.set_title(f"Матрица голосов · n = {total}", loc="left", pad=14)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.text(0.02, 0.86, "Наблюдаемое согласие", fontsize=12, color=MUTED)
    ax2.text(0.02, 0.77, r"$p_o=(42+28+31)/130$", fontsize=15)
    ax2.text(0.02, 0.68, f"$p_o={observed:.3f}$", fontsize=19, color=BLUE, fontweight="bold")
    ax2.text(0.02, 0.51, "Ожидаемое по частотам классов", fontsize=12, color=MUTED)
    ax2.text(0.02, 0.42, r"$p_e=\sum_k p_{A,k}p_{B,k}$", fontsize=15)
    ax2.text(0.02, 0.33, f"$p_e={expected:.3f}$", fontsize=19, color=GOLD, fontweight="bold")
    ax2.add_patch(FancyBboxPatch((0.0, 0.08), 0.93, 0.15, boxstyle="round,pad=0.015,rounding_size=0.02", facecolor=WASH, edgecolor=LINE))
    ax2.text(0.04, 0.16, rf"$\kappa=(p_o-p_e)/(1-p_e)={kappa:.3f}$", fontsize=16, color=INK)

    ax3.set_xlim(0, 1)
    ax3.set_ylim(-0.2, 2.8)
    ax3.barh([1.8], [observed], color=BLUE, height=0.42)
    ax3.barh([0.9], [expected], color=GOLD, height=0.42)
    ax3.barh([0.0], [kappa], color=GREEN, height=0.42)
    for y, value, label in [(1.8, observed, "$p_o$"), (0.9, expected, "$p_e$"), (0.0, kappa, "$\\kappa$")]:
        ax3.text(0.02, y, label, va="center", color=PAPER, fontsize=13, fontweight="bold")
        ax3.text(value + 0.025, y, f"{value:.3f}", va="center", fontsize=12)
    ax3.set_yticks([])
    ax3.set_xticks([0, 0.5, 1])
    ax3.set_xlabel("доля")
    ax3.set_title("Согласие сверх случайного", loc="left", pad=14)
    clean(ax3)

    fig.suptitle(
        "Доля совпадений и согласие сверх частот классов — разные числа",
        x=0.08,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    save(fig, OUT / "kappa-agreement.png")


def leakage_timeline() -> None:
    fig, ax = plt.subplots(figsize=(14.4, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.05, 0.93, "Можно ли вычислить признак в момент прогноза?", fontsize=18, fontweight="bold")
    ax.text(0.05, 0.885, "Задача: в 08:00 предсказать задержку прибытия автобуса к конечной.", color=MUTED, fontsize=11.5)

    y = 0.48
    ax.plot([0.08, 0.92], [y, y], color=INK, linewidth=1.4)
    arrow(ax, (0.90, y), (0.94, y), color=INK)
    ticks = [
        (0.13, "07:50", "расписание\nи маршрут", BLUE),
        (0.31, "08:00", "момент\nпрогноза", GREEN),
        (0.56, "08:15", "текущее\nдвижение", GOLD),
        (0.79, "08:27", "фактическое\nприбытие", RED),
    ]
    for x, time, label, color in ticks:
        ax.plot([x, x], [y - 0.025, y + 0.025], color=color, linewidth=2)
        ax.text(x, y - 0.07, time, ha="center", color=color, fontsize=12, fontweight="bold")
        ax.text(x, y - 0.16, label, ha="center", color=MUTED, fontsize=10.5)

    box(ax, 0.07, 0.66, 0.17, 0.12, "Доступно", "план 08:20\nstop_id", edge=BLUE, fill="#edf3f7")
    box(
        ax,
        0.25,
        0.66,
        0.17,
        0.12,
        "Доступно",
        "наблюдение погоды\nполучено к 08:00",
        edge=GREEN,
        fill="#edf4ef",
        detail_size=8.8,
    )
    box(ax, 0.68, 0.66, 0.22, 0.12, "Будущее", "actual_arrival = 08:27", edge=RED, fill="#f7ece9")
    arrow(ax, (0.155, 0.66), (0.29, 0.54), color=BLUE, rad=0.06)
    arrow(ax, (0.335, 0.66), (0.31, 0.54), color=GREEN)
    arrow(ax, (0.79, 0.66), (0.34, 0.54), color=RED, rad=0.25, width=2)
    ax.text(0.56, 0.77, "утечка: стрелка идёт из будущего", ha="center", color=RED, fontsize=11.5, fontweight="bold")

    ax.add_patch(
        FancyBboxPatch(
            (0.20, 0.17),
            0.22,
            0.12,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=WASH,
            edgecolor=INK,
            linewidth=1.3,
        )
    )
    ax.text(0.31, 0.23, "модель в 08:00", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.add_patch(
        FancyBboxPatch(
            (0.62, 0.17),
            0.25,
            0.12,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor="#f7ece9",
            edgecolor=RED,
            linewidth=1.3,
        )
    )
    ax.text(0.745, 0.23, "$y=08{:}27-08{:}20=7$ мин", ha="center", va="center", fontsize=13)
    arrow(ax, (0.42, 0.23), (0.61, 0.23), color=INK)
    arrow(ax, (0.79, 0.47), (0.745, 0.30), color=RED)
    ax.text(
        0.5,
        0.065,
        "Низкая тестовая ошибка бессмысленна, если столбец actual_arrival появляется только после решения.",
        ha="center",
        fontsize=11.5,
        color=MUTED,
    )
    save(fig, OUT / "leakage-timeline.png")


def data_lineage() -> None:
    fig, ax = plt.subplots(figsize=(14.4, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.04, 0.94, "Паспорт связывает вывод с конкретной версией данных", fontsize=18, fontweight="bold")
    ax.text(
        0.04,
        0.895,
        "Проверенный снимок от 18.07.2026: байты, правила чтения и границы вывода записаны отдельно.",
        color=MUTED,
        fontsize=11.5,
    )

    nodes = [
        (0.04, "Сырой CSV", "не меняется", BLUE, "#edf3f7"),
        (0.235, "Парсер", "«;» + запятая\n−200 становится NA", GOLD, "#f7f1df"),
        (0.43, "Таблица", "9357 строк\nлокальное время", GREEN, "#edf4ef"),
        (0.625, "Временной split", "только целые недели\nраньше / позже", VIOLET, "#f1edf6"),
        (0.82, "Отчёт", "метрика + пределы\nприменимости", RED, "#f7ece9"),
    ]
    y = 0.64
    for index, (x, title, detail, edge, fill) in enumerate(nodes):
        box(ax, x, y, 0.145, 0.16, title, detail, edge=edge, fill=fill, title_size=11.5, detail_size=8.9)
        if index < len(nodes) - 1:
            arrow(ax, (x + 0.145, y + 0.08), (nodes[index + 1][0] - 0.01, y + 0.08))

    ax.text(0.04, 0.595, "ПАСПОРТ БАЙТОВ", fontsize=9.5, color=BLUE, fontweight="bold")
    ax.text(0.04, 0.515, "AirQualityUCI.csv  ·  785 065 bytes", family="monospace", fontsize=10.5)
    ax.text(
        0.04,
        0.468,
        "sha256 13277ae5d8581e80b7be09d47c7d3d06",
        family="monospace",
        fontsize=9.8,
    )
    ax.text(
        0.04,
        0.425,
        "       fe9b8e957078f2cf6e859f955e62f996",
        family="monospace",
        fontsize=9.8,
    )

    ax.text(0.54, 0.595, "ВРЕМЯ В БАЙТАХ", fontsize=9.5, color=GREEN, fontweight="bold")
    ax.text(0.54, 0.515, "10.03.2004 18:00 — 04.04.2005 14:00", family="monospace", fontsize=10.2)
    ax.text(0.54, 0.468, "local time; timezone unspecified", family="monospace", fontsize=10.2)
    ax.text(0.54, 0.425, "перенос: целые недели, без station split", fontsize=10, color=MUTED)

    ax.add_patch(
        FancyBboxPatch(
            (0.04, 0.15),
            0.92,
            0.20,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=WASH,
            edgecolor=LINE,
            linewidth=1,
        )
    )
    ax.text(0.06, 0.315, "ДВЕ КОЛЛИЗИИ МЕТАДАННЫХ", fontsize=9.5, color=RED, fontweight="bold")
    ax.text(
        0.06,
        0.265,
        "Карточка: «март 2004 — февраль 2005»",
        fontsize=10.6,
    )
    ax.text(0.44, 0.265, "≠", color=RED, fontsize=12, fontweight="bold")
    ax.text(0.48, 0.265, "CSV заканчивается 4 апреля 2005", fontsize=10.6)
    ax.text(
        0.06,
        0.215,
        "Старое примечание: research only",
        fontsize=10.6,
    )
    ax.text(0.44, 0.215, "≠", color=RED, fontsize=12, fontweight="bold")
    ax.text(0.48, 0.215, "поле License: CC BY 4.0", fontsize=10.6)
    ax.text(
        0.5,
        0.075,
        "Конфликт не «исправляют» догадкой: обе записи остаются в паспорте.",
        ha="center",
        fontsize=11,
        color=MUTED,
    )
    save(fig, OUT / "data-lineage.png")


def side_sensor() -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.12, 0.25), 0.43, 0.52, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor="#edf3f7", edgecolor=BLUE, linewidth=1.3))
    ax.add_patch(Rectangle((0.20, 0.36), 0.27, 0.28, facecolor=WASH, edgecolor=LINE))
    for y in np.linspace(0.39, 0.61, 5):
        ax.plot([0.23, 0.44], [y, y], color=GOLD, linewidth=1.2)
    for x, y in [(0.05, 0.62), (0.075, 0.48), (0.04, 0.34)]:
        ax.add_patch(Circle((x, y), 0.018, facecolor=RED, edgecolor=RED))
        arrow(ax, (x + 0.02, y), (0.11, 0.50), color=RED, width=1)
    arrow(ax, (0.55, 0.51), (0.72, 0.51), color=BLUE)
    ax.plot([0.73, 0.73, 0.84, 0.84, 0.94], [0.45, 0.62, 0.62, 0.38, 0.38], color=GREEN, linewidth=1.8)
    ax.text(0.335, 0.83, "чувствительный слой", ha="center", fontsize=10.5, fontweight="bold")
    ax.text(0.82, 0.76, "напряжение", ha="center", fontsize=10.5, color=GREEN)
    ax.text(0.50, 0.08, "газ меняет сопротивление; прибор выдаёт сигнал,\nа не готовую концентрацию", ha="center", fontsize=9.5, color=MUTED)
    save(fig, SIDE / "sensor-cutaway.png", dpi=200)


def side_missing_code() -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.05, 0.58), 0.90, 0.24, boxstyle="round,pad=0.012,rounding_size=0.02", facecolor=WASH, edgecolor=LINE))
    ax.text(0.09, 0.74, "11/03/2004  04:00", family="monospace", fontsize=10.2)
    ax.text(0.09, 0.63, "CO(GT) =", family="monospace", fontsize=10.5)
    ax.text(0.38, 0.63, "−200", family="monospace", fontsize=12, color=RED, fontweight="bold")
    ax.text(0.56, 0.63, "PT08 = 1011", family="monospace", fontsize=10.5, color=BLUE)
    arrow(ax, (0.44, 0.55), (0.44, 0.40), color=RED)
    box(ax, 0.17, 0.18, 0.25, 0.16, "число −200", "ошибка смысла", edge=RED, fill="#f7ece9", title_size=10.5, detail_size=8.8)
    box(ax, 0.58, 0.18, 0.25, 0.16, "NA", "пропуск", edge=GREEN, fill="#edf4ef", title_size=10.5, detail_size=8.8)
    arrow(ax, (0.47, 0.40), (0.69, 0.35), color=GREEN, rad=-0.1)
    ax.text(0.5, 0.93, "код пропуска не принадлежит шкале CO", ha="center", fontsize=10.5, fontweight="bold")
    ax.text(0.5, 0.06, "решение принимает парсер и фиксирует в паспорте", ha="center", fontsize=9.3, color=MUTED)
    save(fig, SIDE / "missing-code-card.png", dpi=200)


def side_groups() -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    colors = [BLUE, GREEN, GOLD]
    for row in range(3):
        y = 0.72 - row * 0.23
        ax.text(0.05, y, f"пациент {row + 1}", va="center", fontsize=9.5, color=MUTED)
        for col in range(8):
            ax.add_patch(Rectangle((0.30 + col * 0.075, y - 0.055), 0.055, 0.11, facecolor=PAPER, edgecolor=colors[row], linewidth=1))
    ax.text(0.5, 0.95, "24 снимка, но 3 независимых человека", ha="center", fontsize=10.5, fontweight="bold")
    ax.text(0.5, 0.08, "split делают по пациенту,\nа не по файлу изображения", ha="center", fontsize=9.5, color=MUTED)
    save(fig, SIDE / "grouped-images.png", dpi=200)


def main() -> None:
    sensor_to_row()
    missingness_mechanisms()
    pseudoreplication()
    kappa_agreement()
    leakage_timeline()
    data_lineage()
    side_sensor()
    side_missing_code()
    side_groups()


if __name__ == "__main__":
    main()
