from __future__ import annotations

import math
from pathlib import Path
from textwrap import fill

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "04"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "04"

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
    rad: float = 0,
    mutation: float = 13,
    zorder: int = 2,
) -> FancyArrowPatch:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        connectionstyle=f"arc3,rad={rad}",
        color=color,
        linewidth=width,
        mutation_scale=mutation,
        shrinkA=0,
        shrinkB=0,
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


def binary_increment_trace() -> None:
    """Eight executed rules; the last panel also displays the resulting tape."""
    frames = [
        ("q_scan", 0, "1011", "(q_scan, 1) -> (q_scan, 1, R)"),
        ("q_scan", 1, "1011", "(q_scan, 0) -> (q_scan, 0, R)"),
        ("q_scan", 2, "1011", "(q_scan, 1) -> (q_scan, 1, R)"),
        ("q_scan", 3, "1011", "(q_scan, 1) -> (q_scan, 1, R)"),
        ("q_scan", 4, "1011", "(q_scan, ·) -> (q_carry, ·, L)"),
        ("q_carry", 3, "1011", "(q_carry, 1) -> (q_carry, 0, L)"),
        ("q_carry", 2, "1010", "(q_carry, 1) -> (q_carry, 0, L)"),
        ("q_carry", 1, "1000", "(q_carry, 0) -> (q_halt, 1, S)"),
    ]

    fig = plt.figure(figsize=(14.4, 9.2))
    grid = fig.add_gridspec(
        4,
        2,
        left=0.055,
        right=0.97,
        top=0.83,
        bottom=0.07,
        wspace=0.12,
        hspace=0.28,
    )
    fig.text(
        0.055,
        0.945,
        "Восемь шагов прибавляют единицу: 1011₂  ->  1100₂",
        fontsize=19,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.895,
        "Каждый кадр показывает конфигурацию до шага; золотая строка — единственное правило, которое сейчас исполняется.",
        fontsize=12,
        color=MUTED,
    )
    fig.text(
        0.945,
        0.945,
        "головка сначала ищет правый край,\nзатем переносит единицу влево",
        ha="right",
        va="top",
        fontsize=10.5,
        color=MUTED,
    )

    for index, (state, head, tape, rule) in enumerate(frames):
        ax = fig.add_subplot(grid[index // 2, index % 2])
        ax.set_xlim(-0.45, 6.2)
        ax.set_ylim(-0.95, 1.65)
        ax.axis("off")

        rounded_box(
            ax,
            (-0.36, -0.88),
            (6.38, 2.44),
            face=PAPER,
            edge=LINE,
            linewidth=0.9,
            rounding=0.10,
            zorder=0,
        )
        ax.text(
            -0.20,
            1.32,
            f"{index + 1}",
            fontsize=13,
            fontweight="bold",
            color=BLUE if state == "q_scan" else RED,
            ha="left",
            va="center",
        )
        ax.text(
            0.28,
            1.32,
            state,
            fontsize=11.5,
            fontweight="bold",
            family="monospace",
            ha="left",
            va="center",
        )
        ax.text(
            5.76,
            1.32,
            (
                "после шага: 1100· · q_halt"
                if index == len(frames) - 1
                else ("поиск края" if state == "q_scan" else "перенос")
            ),
            fontsize=9.5,
            color=GREEN if index == len(frames) - 1 else MUTED,
            ha="right",
            va="center",
            fontweight="bold" if index == len(frames) - 1 else "normal",
        )

        cells = ["·"] + list(tape) + ["·"]
        for cell_index, symbol in enumerate(cells):
            x = cell_index
            active = cell_index == head + 1
            ax.add_patch(
                Rectangle(
                    (x, 0.25),
                    0.88,
                    0.72,
                    facecolor="#f7f1df" if active else PAPER,
                    edgecolor=GOLD if active else LINE,
                    linewidth=2.2 if active else 1.15,
                    zorder=2,
                )
            )
            ax.text(
                x + 0.44,
                0.61,
                symbol,
                ha="center",
                va="center",
                fontsize=17,
                family="monospace",
                fontweight="bold" if active else "normal",
                zorder=3,
            )
        active_x = head + 1 + 0.44
        ax.plot(active_x, 0.10, marker="^", markersize=8, color=GOLD, clip_on=False)
        ax.text(
            active_x,
            -0.04,
            "читает",
            ha="center",
            va="top",
            fontsize=8.5,
            color=GOLD,
        )
        rounded_box(
            ax,
            (0.18, -0.73),
            (5.47, 0.40),
            face="#f7f1df",
            edge=GOLD,
            linewidth=1.1,
            rounding=0.07,
            zorder=1,
        )
        ax.text(
            2.92,
            -0.53,
            rule,
            ha="center",
            va="center",
            fontsize=10.6,
            family="monospace",
            color=INK,
            zorder=3,
        )
    save(fig, OUT / "binary-increment-trace.png")


def halting_diagonal() -> None:
    fig, ax = plt.subplots(figsize=(14.4, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.04,
        0.94,
        "Тотальный предсказатель проигрывает на программе, которая делает наоборот",
        fontsize=19,
        fontweight="bold",
    )
    ax.text(
        0.04,
        0.89,
        "Предположение: H(P, x) всегда останавливается и безошибочно отвечает, остановится ли P(x).",
        fontsize=12,
        color=MUTED,
    )

    rounded_box(ax, (0.19, 0.72), (0.62, 0.10), face=WASH, edge=LINE)
    ax.text(
        0.50,
        0.775,
        "D(P):  если H(P, P) = ДА, то цикл навсегда;  иначе остановка",
        ha="center",
        va="center",
        fontsize=13,
        family="monospace",
        fontweight="bold",
    )
    ax.text(
        0.83,
        0.775,
        "запускаем D(D)",
        ha="left",
        va="center",
        fontsize=11,
        color=RED,
        fontweight="bold",
    )
    arrow(ax, (0.805, 0.775), (0.815, 0.775), color=RED)

    cases = [
        {
            "x": 0.06,
            "edge": BLUE,
            "wash": "#edf3f7",
            "title": "СЛУЧАЙ 1 · H(D, D) = ДА",
            "middle": "D доверяет ответу\nи входит в цикл",
            "behavior": "фактически: не останавливается",
            "contradiction": "H сказал «остановится»",
        },
        {
            "x": 0.53,
            "edge": RED,
            "wash": "#f7ece9",
            "title": "СЛУЧАЙ 2 · H(D, D) = НЕТ",
            "middle": "D доверяет ответу\nи сразу останавливается",
            "behavior": "фактически: останавливается",
            "contradiction": "H сказал «не остановится»",
        },
    ]
    for case in cases:
        x = case["x"]
        edge = case["edge"]
        rounded_box(ax, (x, 0.58), (0.41, 0.085), face=case["wash"], edge=edge, linewidth=1.5)
        ax.text(
            x + 0.205,
            0.624,
            case["title"],
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=edge,
        )
        arrow(ax, (x + 0.205, 0.575), (x + 0.205, 0.515), color=edge)
        rounded_box(ax, (x + 0.055, 0.39), (0.30, 0.12), face=PAPER, edge=LINE)
        ax.text(
            x + 0.205,
            0.45,
            case["middle"],
            ha="center",
            va="center",
            fontsize=12.5,
            fontweight="bold",
        )
        arrow(ax, (x + 0.205, 0.385), (x + 0.205, 0.325), color=edge)
        ax.text(
            x + 0.205,
            0.292,
            case["behavior"],
            ha="center",
            va="center",
            fontsize=11,
            color=GREEN,
        )
        rounded_box(ax, (x + 0.055, 0.11), (0.30, 0.105), face="#fff1ee", edge=RED, linewidth=1.6)
        ax.text(
            x + 0.205,
            0.164,
            "ПРОТИВОРЕЧИЕ",
            ha="center",
            va="center",
            fontsize=11,
            color=RED,
            fontweight="bold",
        )
        ax.text(
            x + 0.205,
            0.122,
            case["contradiction"],
            ha="center",
            va="center",
            fontsize=9.7,
            color=MUTED,
        )

    ax.plot([0.50, 0.50], [0.10, 0.68], color=GRID, linewidth=1.0)
    ax.text(
        0.50,
        0.045,
        "Оба возможных окончательных ответа H неверны. Значит, обещание «всегда остановиться и решить для всех P, x» невозможно.",
        ha="center",
        va="center",
        fontsize=11.5,
        color=INK,
        fontweight="bold",
    )
    save(fig, OUT / "halting-diagonal-trace.png")


def computability_landscape() -> None:
    fig, ax = plt.subplots(figsize=(14.4, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.04,
        0.94,
        "Три разных обещания алгоритма: решить, подтвердить «да», быть невозможным",
        fontsize=19,
        fontweight="bold",
    )
    ax.text(
        0.04,
        0.89,
        "Ключевой вопрос не «быстро ли?», а «завершается ли процедура на каждом допустимом входе?».",
        fontsize=12,
        color=MUTED,
    )

    rows = [
        {
            "y": 0.64,
            "edge": GREEN,
            "face": "#edf4ef",
            "status": "РАЗРЕШИМО",
            "question": "Есть ли d: 1 < d < n?",
            "trace": ["2", "3", "4", "…", "√n"],
            "detail": "91: найден 7  ·  97: диапазон исчерпан",
            "result": "«да» и «нет»\nполучены за конечное время",
        },
        {
            "y": 0.37,
            "edge": GOLD,
            "face": "#f7f1df",
            "status": "ПОЛУРАЗРЕШИМО",
            "question": "Остановится ли P(x)?",
            "trace": ["шаг 1", "2", "3", "…", "t"],
            "detail": "если P остановится, конечная трасса подтвердит «да»",
            "result": "ожидание без конца\nне подтверждает «нет»",
        },
        {
            "y": 0.10,
            "edge": RED,
            "face": "#f7ece9",
            "status": "НЕРАЗРЕШИМО",
            "question": "Тотальный H(P, x)?",
            "trace": ["ДА", "или", "НЕТ", "для", "всех"],
            "detail": "D(D) обращает любой окончательный ответ против H",
            "result": "общего безошибочного\nрешателя не существует",
        },
    ]

    for row in rows:
        y = row["y"]
        rounded_box(ax, (0.04, y), (0.92, 0.205), face=PAPER, edge=LINE, linewidth=0.9)
        ax.add_patch(
            Rectangle(
                (0.04, y),
                0.012,
                0.205,
                facecolor=row["edge"],
                edgecolor="none",
                zorder=3,
            )
        )
        ax.text(
            0.075,
            y + 0.158,
            row["status"],
            fontsize=11,
            fontweight="bold",
            color=row["edge"],
        )
        ax.text(
            0.075,
            y + 0.085,
            row["question"],
            fontsize=14,
            fontweight="bold",
        )

        start_x = 0.35
        for index, label in enumerate(row["trace"]):
            x = start_x + index * 0.065
            rounded_box(
                ax,
                (x, y + 0.10),
                (0.055, 0.055),
                face=row["face"],
                edge=row["edge"],
                linewidth=1.0,
                rounding=0.012,
            )
            ax.text(x + 0.0275, y + 0.1275, label, ha="center", va="center", fontsize=9.5)
            if index < len(row["trace"]) - 1:
                arrow(
                    ax,
                    (x + 0.057, y + 0.1275),
                    (x + 0.063, y + 0.1275),
                    color=MUTED,
                    width=0.9,
                    mutation=8,
                )
        ax.text(
            0.35,
            y + 0.058,
            row["detail"],
            fontsize=10,
            color=MUTED,
            ha="left",
            va="center",
        )
        arrow(ax, (0.69, y + 0.102), (0.735, y + 0.102), color=row["edge"])
        ax.text(
            0.845,
            y + 0.105,
            row["result"],
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color=row["edge"],
        )

    ax.text(
        0.50,
        0.045,
        "Поиск делителя конечен благодаря границе √n. Симуляция произвольной программы такой границы не имеет.",
        ha="center",
        va="center",
        fontsize=11.3,
        color=MUTED,
    )
    save(fig, OUT / "computability-landscape.png")


def exponential_time_scale() -> None:
    fig, ax = plt.subplots(figsize=(14.4, 8.0))
    n = np.arange(10, 91)
    seconds = np.power(2.0, n) / 1e9

    ax.semilogy(n, seconds, color=RED, linewidth=2.7, zorder=4)
    ax.fill_between(n, 1e-8, seconds, color=RED, alpha=0.055, zorder=1)
    ax.set_xlim(10, 90)
    ax.set_ylim(1e-7, 1e18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRID, linewidth=0.7, alpha=0.85, which="major")
    ax.set_axisbelow(True)
    ax.set_xlabel("размер входа n")
    ax.set_ylabel(r"время полного перебора $2^n$ вариантов")
    ax.set_title(
        "Алгоритм существует, но каждые +10 бит умножают время примерно на 1000",
        loc="left",
        fontsize=19,
        fontweight="bold",
        pad=25,
    )
    ax.text(
        0.0,
        1.018,
        r"Расчёт при постоянной скорости $10^9$ операций/с; накладные расходы намеренно не учитываются.",
        transform=ax.transAxes,
        fontsize=11.5,
        color=MUTED,
    )

    year = 365.25 * 24 * 3600
    ticks = [1e-6, 1e-3, 1, 3600, year, 1e3 * year, 1e6 * year, 1e9 * year]
    labels = ["1 мкс", "1 мс", "1 с", "1 ч", "1 год", "1000 лет", "1 млн лет", "1 млрд лет"]
    ax.set_yticks(ticks, labels)
    ax.set_xticks(np.arange(10, 91, 10))

    universe_seconds = 13.8e9 * year
    ax.axhline(universe_seconds, color=VIOLET, linewidth=1.3, linestyle=(0, (5, 4)), zorder=2)
    ax.text(
        89.2,
        universe_seconds * 3.1,
        "возраст Вселенной ≈ 13,8 млрд лет",
        ha="right",
        va="bottom",
        fontsize=10,
        color=VIOLET,
    )

    annotations = {
        20: ("1,05 мс", (-2, 24)),
        40: ("18,3 мин", (-14, 27)),
        60: ("36,5 года", (-18, 28)),
        80: ("38,3 млн лет", (-64, -34)),
    }
    for value, (label, offset) in annotations.items():
        sec = 2.0**value / 1e9
        ax.scatter([value], [sec], s=44, color=BLUE, edgecolor=PAPER, linewidth=1.2, zorder=5)
        ax.annotate(
            f"n={value}\n{label}",
            xy=(value, sec),
            xytext=offset,
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color=INK,
            arrowprops={"arrowstyle": "-", "color": BLUE, "linewidth": 1.1},
            bbox={"boxstyle": "round,pad=0.25", "fc": PAPER, "ec": LINE, "lw": 0.8},
        )

    ax.text(
        12,
        1.8e15,
        "Это ресурсная трудность:\nответ вычислим, но может быть недоступен практически.",
        fontsize=11,
        color=MUTED,
        va="top",
    )
    save(fig, OUT / "exponential-time-scale.png")


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    z = 1.96
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return center - half, center + half


def imitation_protocol() -> None:
    fig, ax = plt.subplots(figsize=(14.4, 8.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.04,
        0.94,
        "Имитационная игра становится экспериментом только после фиксации протокола",
        fontsize=19,
        fontweight="bold",
    )
    ax.text(
        0.04,
        0.89,
        "Вопросы, случайный порядок, слепое решение и правило подсчёта задают до раскрытия меток.",
        fontsize=12,
        color=MUTED,
    )

    steps = [
        ("1", "Банк вопросов", "4 темы × 10 заданий\nзафиксированы заранее", BLUE, "#edf3f7"),
        ("2", "Случайный порядок", "seed = 1950\nA/B перемешаны", VIOLET, "#f1edf6"),
        ("3", "Слепой судья", "решение записано\nдо раскрытия", GOLD, "#f7f1df"),
        ("4", "Открытие меток", "человек / машина\nи пропуски", RED, "#f7ece9"),
        ("5", "Отчёт", "матрица ошибок\n+ интервал Уилсона", GREEN, "#edf4ef"),
    ]
    x_positions = np.linspace(0.045, 0.795, 5)
    box_w = 0.15
    for index, (number, title, detail, edge, face) in enumerate(steps):
        x = float(x_positions[index])
        rounded_box(ax, (x, 0.60), (box_w, 0.19), face=face, edge=edge, linewidth=1.35)
        ax.text(
            x + 0.02,
            0.747,
            number,
            fontsize=13,
            fontweight="bold",
            color=edge,
            ha="left",
            va="center",
        )
        ax.text(
            x + box_w / 2,
            0.705,
            title,
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
        )
        ax.text(
            x + box_w / 2,
            0.645,
            detail,
            fontsize=9.6,
            color=MUTED,
            ha="center",
            va="center",
        )
        if index < len(steps) - 1:
            arrow(
                ax,
                (x + box_w + 0.006, 0.695),
                (float(x_positions[index + 1]) - 0.008, 0.695),
                color=MUTED,
                width=1.2,
            )

    ax.text(
        0.04,
        0.53,
        "Пример одного заранее выбранного отчёта: 40 диалогов, по 20 каждого источника",
        fontsize=11.5,
        fontweight="bold",
    )

    matrix = np.array([[14, 6], [8, 12]])
    cell_x, cell_y = 0.12, 0.18
    cell_w, cell_h = 0.11, 0.105
    ax.text(cell_x + cell_w, 0.45, "РЕШЕНИЕ СУДЬИ", ha="center", va="center", fontsize=9.5, color=MUTED)
    ax.text(cell_x + cell_w * 0.5, 0.413, "машина", ha="center", fontsize=9.5)
    ax.text(cell_x + cell_w * 1.5, 0.413, "человек", ha="center", fontsize=9.5)
    ax.text(0.052, cell_y + cell_h, "ИСТОЧНИК", rotation=90, ha="center", va="center", fontsize=9.5, color=MUTED)
    for row in range(2):
        ax.text(
            0.104,
            cell_y + (1 - row) * cell_h + cell_h / 2,
            "машина" if row == 0 else "человек",
            ha="right",
            va="center",
            fontsize=9.5,
        )
        for column in range(2):
            x = cell_x + column * cell_w
            y = cell_y + (1 - row) * cell_h
            correct = row == column
            ax.add_patch(
                Rectangle(
                    (x, y),
                    cell_w,
                    cell_h,
                    facecolor="#edf4ef" if correct else "#f7ece9",
                    edgecolor=LINE,
                    linewidth=1,
                )
            )
            ax.text(
                x + cell_w / 2,
                y + cell_h / 2,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=17,
                fontweight="bold",
                color=GREEN if correct else RED,
            )

    successes = int(np.trace(matrix))
    total = int(matrix.sum())
    lower, upper = wilson_interval(successes, total)
    ax.text(
        0.43,
        0.415,
        f"точность = ({matrix[0,0]} + {matrix[1,1]}) / {total} = {successes / total:.2f}",
        fontsize=13,
        fontweight="bold",
    )
    ax.text(
        0.43,
        0.365,
        f"95%-интервал Уилсона: [{lower:.3f}; {upper:.3f}]",
        fontsize=11,
        color=MUTED,
    )
    line_y = 0.255
    ax.plot([0.43, 0.90], [line_y, line_y], color=LINE, linewidth=2)
    for tick in np.linspace(0, 1, 6):
        x = 0.43 + 0.47 * tick
        ax.plot([x, x], [line_y - 0.012, line_y + 0.012], color=LINE, linewidth=1)
        ax.text(x, line_y - 0.038, f"{tick:.1f}", ha="center", va="top", fontsize=9, color=MUTED)
    x_lower = 0.43 + 0.47 * lower
    x_upper = 0.43 + 0.47 * upper
    x_point = 0.43 + 0.47 * (successes / total)
    x_chance = 0.43 + 0.47 * 0.5
    ax.plot([x_lower, x_upper], [line_y, line_y], color=BLUE, linewidth=7, solid_capstyle="round")
    ax.scatter([x_point], [line_y], s=90, color=INK, edgecolor=PAPER, linewidth=1.5, zorder=5)
    ax.plot([x_chance, x_chance], [line_y - 0.06, line_y + 0.06], color=GOLD, linestyle=(0, (4, 3)), linewidth=1.4)
    ax.text(x_chance, line_y + 0.075, "случай = 0,5", ha="center", fontsize=9.5, color=GOLD)
    ax.text(
        0.665,
        0.115,
        "0,5 остаётся внутри интервала:\n40 наблюдений ещё недостаточно для уверенного вывода «лучше случайного угадывания».",
        ha="center",
        va="center",
        fontsize=10.5,
        color=INK,
    )
    save(fig, OUT / "imitation-game-protocol.png")


def response_rubric_scatter() -> None:
    # Editorially constructed ratings, intentionally not empirical measurements.
    points = [
        ("M1", 5.0, 1.3, 4.4, "machine"),
        ("M2", 4.6, 3.7, 3.9, "machine"),
        ("M3", 4.2, 4.5, 2.8, "machine"),
        ("M4", 3.9, 4.7, 4.2, "machine"),
        ("M5", 2.4, 4.6, 3.7, "machine"),
        ("M6", 1.1, 4.4, 4.3, "machine"),
        ("M7", 3.2, 2.0, 1.5, "machine"),
        ("M8", 4.8, 3.1, 4.8, "machine"),
        ("M9", 1.8, 3.4, 1.2, "machine"),
        ("M10", 3.7, 3.9, 2.2, "machine"),
        ("H1", 4.7, 4.8, 4.1, "human"),
        ("H2", 4.0, 4.4, 2.9, "human"),
        ("H3", 3.5, 4.6, 3.4, "human"),
        ("H4", 1.2, 4.9, 2.0, "human"),
        ("H5", 2.7, 4.3, 4.0, "human"),
        ("H6", 4.9, 3.8, 3.5, "human"),
        ("H7", 3.3, 4.0, 1.7, "human"),
        ("H8", 2.0, 3.7, 2.5, "human"),
        ("H9", 4.4, 4.6, 4.7, "human"),
        ("H10", 1.5, 4.2, 3.8, "human"),
    ]

    fig, ax = plt.subplots(figsize=(14.4, 8.4))
    ax.set_xlim(0, 5.35)
    ax.set_ylim(0, 5.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRID, linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    ax.set_xticks(range(0, 6))
    ax.set_yticks(range(0, 6))
    ax.set_xlabel("фактическая точность · независимая проверка, 0–5")
    ax.set_ylabel("сходство с человеческим стилем · слепые судьи, 0–5")
    ax.set_title(
        "Человеческий стиль и истинность — разные координаты ответа",
        loc="left",
        fontsize=19,
        fontweight="bold",
        pad=27,
    )
    ax.text(
        0.0,
        1.02,
        "СИНТЕТИЧЕСКИЙ ПРИМЕР: 20 редакционных оценок, не результаты реального исследования. Цвет кодирует качество объяснения.",
        transform=ax.transAxes,
        fontsize=10.8,
        color=RED,
        fontweight="bold",
    )

    ax.axvspan(4, 5.35, color=GREEN, alpha=0.045)
    ax.axhspan(4, 5.35, color=VIOLET, alpha=0.04)
    ax.axvline(4, color=LINE, linewidth=1, linestyle=(0, (4, 4)))
    ax.axhline(4, color=LINE, linewidth=1, linestyle=(0, (4, 4)))

    def explanation_color(value: float) -> str:
        if value < 2.5:
            return RED
        if value < 4:
            return GOLD
        return GREEN

    for label, factual, human_like, explanation, source in points:
        marker = "o" if source == "machine" else "^"
        ax.scatter(
            factual,
            human_like,
            s=56 + explanation * 15,
            marker=marker,
            facecolor=explanation_color(explanation),
            edgecolor=PAPER,
            linewidth=1.2,
            alpha=0.90,
            zorder=4,
        )

    callouts = [
        ("M1", 5.0, 1.3, "точный, но нарочито\n«машинный» ответ", (-118, 26)),
        ("H4", 1.2, 4.9, "уверенный человеческий\nответ с ошибкой", (24, -58)),
        ("M6", 1.1, 4.4, "правдоподобная\nмашинная ошибка", (30, -88)),
        ("H9", 4.4, 4.6, "точность, стиль\nи объяснение высоки", (-142, -74)),
    ]
    for _label, x, y, text, offset in callouts:
        ax.annotate(
            text,
            xy=(x, y),
            xytext=offset,
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=9.8,
            arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 1},
            bbox={"boxstyle": "round,pad=0.25", "fc": PAPER, "ec": LINE, "lw": 0.8},
        )

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PAPER, markeredgecolor=INK, markersize=9, label="условная машина"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=PAPER, markeredgecolor=INK, markersize=9, label="условный человек"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=RED, markeredgecolor=PAPER, markersize=8, label="объяснение 0–2,4"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GOLD, markeredgecolor=PAPER, markersize=8, label="объяснение 2,5–3,9"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN, markeredgecolor=PAPER, markersize=8, label="объяснение 4–5"),
    ]
    ax.legend(handles=legend, loc="lower left", ncol=2, frameon=False, fontsize=9.5)
    ax.text(
        5.22,
        0.24,
        "Верхний правый угол желателен,\nно одна ось не заменяет другую.",
        ha="right",
        va="bottom",
        fontsize=10,
        color=MUTED,
    )
    save(fig, OUT / "response-rubric-scatter.png")


def markov_rewrite() -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.04, 0.94, "Нормальная подстановка переносит «+» влево", fontsize=14, fontweight="bold")
    ax.text(
        0.04,
        0.885,
        "Упорядоченные правила: 1) 1+ -> +1,  2) + -> пусто.",
        fontsize=9.6,
        color=MUTED,
    )

    states = [
        ("111+11", "1", (2, 4)),
        ("11+111", "1", (1, 3)),
        ("1+1111", "1", (0, 2)),
        ("+11111", "2", (0, 1)),
        ("11111", "стоп", None),
    ]
    y_positions = np.linspace(0.75, 0.14, len(states))
    for index, ((text, rule, span), y) in enumerate(zip(states, y_positions)):
        if span:
            char_width = 0.034
            x_start = 0.15
            ax.add_patch(
                Rectangle(
                    (x_start + span[0] * char_width - 0.006, y - 0.027),
                    (span[1] - span[0]) * char_width + 0.012,
                    0.075,
                    facecolor="#f7f1df",
                    edgecolor=GOLD,
                    linewidth=0.8,
                )
            )
        ax.text(0.15, y, text, fontsize=15, family="monospace", ha="left", va="center")
        ax.text(0.56, y, rule, fontsize=9.5, color=GREEN if rule == "стоп" else GOLD, ha="center", va="center")
        if index < len(states) - 1:
            arrow(ax, (0.69, y - 0.025), (0.69, y_positions[index + 1] + 0.035), color=MUTED, width=1.0, mutation=9)
    ax.text(
        0.95,
        0.51,
        "сначала применяется\nпервое подходящее\nправило",
        fontsize=9,
        color=MUTED,
        ha="right",
        va="center",
    )
    ax.text(
        0.04,
        0.035,
        "Итог: 3 + 2 единицы превращаются в 5 единиц. Это схема переписывания, не портрет учёного.",
        fontsize=8.8,
        color=MUTED,
        va="bottom",
    )
    save(fig, SIDE / "markov-string-rewrite.png")


def spreadsheet_universality() -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.04, 0.94, "Выразительность среды ≠ интеллект файла", fontsize=14, fontweight="bold")
    ax.text(
        0.04,
        0.885,
        "При итеративных ссылках и неограниченной сетке формулы могут кодировать универсальное вычисление.",
        fontsize=8.9,
        color=MUTED,
    )

    x0, y0 = 0.06, 0.43
    cols, rows = 5, 5
    cell_w, cell_h = 0.105, 0.075
    for row in range(rows):
        for col in range(cols):
            ax.add_patch(
                Rectangle(
                    (x0 + col * cell_w, y0 + (rows - 1 - row) * cell_h),
                    cell_w,
                    cell_h,
                    facecolor="#edf3f7" if (row, col) in {(1, 1), (2, 2), (3, 3)} else PAPER,
                    edgecolor=LINE,
                    linewidth=0.7,
                )
            )
    labels = {(1, 1): "q", (2, 2): "0", (3, 3): "1"}
    for (row, col), label in labels.items():
        ax.text(
            x0 + col * cell_w + cell_w / 2,
            y0 + (rows - 1 - row) * cell_h + cell_h / 2,
            label,
            ha="center",
            va="center",
            fontsize=12,
            family="monospace",
            fontweight="bold",
            color=BLUE,
        )
    ax.text(x0, y0 + rows * cell_h + 0.04, "=IF(B2=0; … ; …)", fontsize=10.5, family="monospace", color=BLUE)

    arrow(ax, (0.60, 0.62), (0.72, 0.62), color=GOLD, width=1.4)
    rounded_box(ax, (0.73, 0.48), (0.22, 0.27), face="#f7f1df", edge=GOLD)
    ax.text(0.84, 0.68, "МОЖЕТ", ha="center", fontsize=10, color=GOLD, fontweight="bold")
    ax.text(
        0.84,
        0.585,
        "симулировать\nзаписанный\nалгоритм",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
    )

    ax.plot([0.08, 0.93], [0.31, 0.31], color=GRID, linewidth=1)
    ax.text(0.07, 0.23, "ИЗ ЭТОГО НЕ СЛЕДУЕТ", fontsize=9.5, color=RED, fontweight="bold")
    ax.text(
        0.07,
        0.12,
        "что конкретная таблица знает факты,\nпонимает цель или выбирает хорошую программу.",
        fontsize=10.4,
        color=INK,
        va="center",
    )
    ax.text(
        0.95,
        0.055,
        "Тьюринг-полнота — свойство языка, не личности.",
        fontsize=9.1,
        color=MUTED,
        ha="right",
    )
    save(fig, SIDE / "spreadsheet-universality.png")


def wilson_mini() -> None:
    successes, total = 26, 40
    p = successes / total
    lower, upper = wilson_interval(successes, total)

    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    ax.set_xlim(0.35, 0.9)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.35, 0.94, "26 верных решений из 40 — ещё не приговор", fontsize=14, fontweight="bold")
    ax.text(
        0.35,
        0.875,
        "Доля 0,65; 95%-интервал Уилсона учитывает малый объём выборки.",
        fontsize=9.2,
        color=MUTED,
    )

    y = 0.56
    ax.plot([0.4, 0.85], [y, y], color=LINE, linewidth=2)
    for tick in np.arange(0.4, 0.91, 0.1):
        ax.plot([tick, tick], [y - 0.025, y + 0.025], color=LINE, linewidth=1)
        ax.text(tick, y - 0.065, f"{tick:.1f}", ha="center", va="top", fontsize=8.8, color=MUTED)
    ax.plot([lower, upper], [y, y], color=BLUE, linewidth=9, solid_capstyle="round")
    ax.scatter([p], [y], s=100, color=INK, edgecolor=PAPER, linewidth=1.5, zorder=5)
    ax.plot([0.5, 0.5], [y - 0.14, y + 0.14], color=GOLD, linewidth=1.4, linestyle=(0, (4, 3)))
    ax.text(0.5, y + 0.17, "случайное угадывание", ha="center", fontsize=8.7, color=GOLD)
    ax.text(lower, y + 0.08, f"{lower:.3f}", ha="center", fontsize=9.2, color=BLUE, fontweight="bold")
    ax.text(p, y + 0.08, f"{p:.2f}", ha="center", fontsize=10.5, color=INK, fontweight="bold")
    ax.text(upper, y + 0.08, f"{upper:.3f}", ha="center", fontsize=9.2, color=BLUE, fontweight="bold")

    ax.text(
        0.35,
        0.235,
        r"$\hat p_W \pm h_W$",
        fontsize=13,
        family="serif",
        color=BLUE,
    )
    ax.text(
        0.46,
        0.235,
        "не симметричен вокруг 0,65\nи остаётся внутри [0; 1]",
        fontsize=9.2,
        color=MUTED,
        va="center",
    )
    ax.text(
        0.35,
        0.055,
        "Поскольку 0,5 попадает в интервал, такой результат сам по себе не отделяет судью от случайности.",
        fontsize=9.3,
        color=INK,
        va="bottom",
    )
    save(fig, SIDE / "wilson-interval-mini.png")


def main() -> None:
    binary_increment_trace()
    halting_diagonal()
    computability_landscape()
    exponential_time_scale()
    imitation_protocol()
    response_rubric_scatter()
    markov_rewrite()
    spreadsheet_universality()
    wilson_mini()


if __name__ == "__main__":
    main()
