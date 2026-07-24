#!/usr/bin/env python3
"""Generate the 84 lesson figures for lessons 13–40.

The Markdown files are the source of truth for title, alt text and caption.
The compact SPECS table chooses a mathematical composition and notation for
each figure. SVG is the deliverable; PNG previews and contact sheets are kept
under tmp/ for visual QA.
"""

from __future__ import annotations

import argparse
import math
import re
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
from PIL import Image, ImageDraw, ImageFont


SITE = Path(__file__).resolve().parents[1]
REPO = SITE.parent
CONTENT = SITE / "content" / "lessons"
OUT = SITE / "public" / "figures" / "lessons"
PREVIEW = REPO / "tmp" / "figures_13_40"

PAPER = "#fffef9"
INK = "#171915"
MUTED = "#6f746d"
GRID = "#d9d8d0"
BLUE = "#315f8c"
RED = "#b94a3b"
GREEN = "#38735d"
GOLD = "#a57920"
VIOLET = "#6f5a8f"
CYAN = "#2f7f83"
PALETTE = [BLUE, RED, GREEN, GOLD, VIOLET, CYAN]


mpl.rcParams.update(
    {
        "font.family": "Palatino",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.facecolor": PAPER,
        "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.grid": False,
        "svg.fonttype": "none",
        "mathtext.fontset": "stix",
        "lines.linewidth": 2.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# kind, x label, y label, visible labels, compact mathematical note
SPECS: dict[int, list[tuple[str, str, str, list[str], str]]] = {
    13: [
        ("boundary", "$x_1$", "$x_2$", ["$y=0$", "$y=1$", "$w^Tx=b$"], "$m=w^Tx-b$"),
        ("network", "входы", "выход", ["$x_1,x_2,x_3$", "$h_1,h_2$", "$y$"], "$(x_1\\land\\neg x_2)\\lor(x_2\\land x_3)$"),
        ("signal", "время", "измерение", ["температура", "влажность", "порог"], "$x\\mapsto \\mathbf 1[x\\ge b]$"),
    ],
    14: [
        ("pipeline", "контур", "состояние", ["цель $r$", "ошибка $e$", "регулятор $K$", "комната $x_t$"], "$u_t=K(r-x_t)$"),
        ("signal", "$t$", "$x_t$", ["цель", "без задержки", "задержка $d$"], "$x_{t+1}=x_t-\\alpha(x_t-x_{out})+\\beta u_t$"),
        ("signal", "час", "кВт / кВт·ч", ["нагрузка", "после управления", "заряд $q_t$"], "$0\\le q_t\\le Q$"),
    ],
    15: [
        ("boundary", "$x_1$", "$x_2$", ["до шага", "после шага", "ошибка $x_i$"], "$w\\leftarrow w+\\eta yx$"),
        ("tradeoff", "$\\gamma$", "число ошибок", ["широкий зазор", "узкий зазор"], "$M\\le (R/\\gamma)^2$"),
        ("boundary", "частота !", "длина CAPS", ["обычное", "спам", "3 порядка"], "усреднённый перцептрон"),
    ],
    16: [
        ("boundary", "$x_1$", "$x_2$", ["XOR: вход", "скрытый код", "граница"], "$(x_1,x_2)\\mapsto(h_1,h_2)$"),
        ("network", "два входа", "один выход", ["OR + NAND", "два случая", "XOR"], "$h_1\\lor h_2$"),
        ("curves", "$x_1$", "$x_2$", ["$k=1$", "$k=5$", "$k=20$"], "$\\sigma(kz)$"),
    ],
    17: [
        ("curves", "$z$", "$\\phi(z)$", ["sigmoid", "tanh", "ReLU", "Leaky ReLU"], "$\\phi'(z)$"),
        ("curves", "$x$", "$f(x)$", ["сдвинутые ReLU", "сумма", "узлы $t_j$"], "$f=a+cx+\\sum v_j\\,ReLU(x-t_j)$"),
        ("matrix", "нейрон", "батч", ["до нормировки", "после", "доля активных"], "$z=w^Tx+b$"),
    ],
    18: [
        ("pipeline", "форма", "операция", ["$5\\times5$", "$25$", "$4$", "$2$"], "$s=V\\,ReLU(Wx+b)+c$"),
        ("matrix", "вариант буквы", "скрытый узел", ["C", "O", "сдвиг", "шум"], "$h_j=ReLU(W_jx+b_j)$"),
        ("bar", "скрытый узел", "$\\Delta(s_C-s_O)$", ["C", "O", "неоднозначно"], "абляция $h_j\\leftarrow0$"),
    ],
    19: [
        ("curves", "$x$", "$f(x)$", ["цель", "узлы", "ReLU-ломаная"], "$k_j-k_{j-1}$"),
        ("tradeoff", "сложность", "ошибка", ["аппроксимация", "оценивание", "оптимизация"], "$\\varepsilon_{all}\\approx\\varepsilon_a+\\varepsilon_e+\\varepsilon_o$"),
        ("signal", "час", "нагрузка", ["рабочие дни", "ReLU-модель", "выходные"], "систематический остаток"),
    ],
    20: [
        ("tradeoff", "постоянный прогноз $a$", "суммарная потеря", ["MSE → среднее", "MAE → медиана", "$\\rho_{.8}$ → квантиль"], "$\\arg\\min_a\\sum\\ell(y_i-a)$"),
        ("curves", "остаток $e$", "loss / градиент", ["MSE", "MAE", "Huber", "quantile"], "$\\ell_\\delta(e)$"),
        ("bar", "пример", "вероятность / loss", ["модель A", "модель B", "одинаковая accuracy"], "$-y\\log p-(1-y)\\log(1-p)$"),
    ],
    21: [
        ("curves", "$x$", "$f(x)$", ["минимум", "максимум", "перегиб", "две чаши"], "$f'(x)=0$"),
        ("geometry", "$x$", "$f(x)$", ["хорда", "касательная", "график"], "$f(\\lambda x+(1-\\lambda)y)\\le\\lambda f(x)+(1-\\lambda)f(y)$"),
        ("contour", "$\\theta_1$", "$\\theta_2$", ["круглая чаша", "узкая долина", "gradient descent"], "$\\kappa=\\lambda_{max}/\\lambda_{min}$"),
    ],
    22: [
        ("geometry", "$a$", "$b$", ["поверхность", "касательная", "$-\\nabla L$"], "$L(\\theta+\\Delta)\\approx L+\\nabla L^T\\Delta$"),
        ("contour", "$a$", "$b$", ["линия уровня", "касательная $v$", "$\\nabla L$"], "$\\nabla L^Tv=0$"),
        ("tradeoff", "$h$", "относительная ошибка", ["односторонняя", "центральная", "округление"], "$D_h=[L(\\theta+h)-L(\\theta-h)]/(2h)$"),
    ],
    23: [
        ("contour", "$x$", "$y$", ["$g(x)=0$", "линии $f$", "$\\nabla f\\parallel\\nabla g$"], "$\\nabla f=\\lambda\\nabla g$"),
        ("geometry", "$x$", "$y$", ["неактивно: $\\lambda=0$", "активно: $g=0$"], "$\\lambda g(x)=0$"),
        ("contour", "$w_1$", "$w_2$", ["L2-круг", "L1-ромб", "путь решения"], "$\\|w\\|\\le R$"),
    ],
    24: [
        ("boundary", "$g_1$", "$g_2$", ["градиенты объектов", "батч 1", "батч 16"], "$E[\\hat g]=\\nabla L$"),
        ("contour", "$\\theta_1$", "$\\theta_2$", ["full GD", "SGD", "SGD + schedule"], "$\\theta\\leftarrow\\theta-\\eta_t\\hat g$"),
        ("signal", "время", "нагрузка", ["случайные часы", "окна 24 ч", "целые дни"], "три схемы батча"),
    ],
    25: [
        ("network", "прямой проход", "обратный проход", ["$x,y$", "$a=xy$", "$c=a^2$", "$L=c+x+y$"], "$\\bar u {+}=\\bar v\\,\\partial v/\\partial u$"),
        ("pipeline", "тензор", "форма", ["$x:d$", "$W:m\\times d$", "$z:m$", "$\\bar W:m\\times d$"], "$\\bar W=\\bar z x^T$"),
        ("timeline", "слой", "ресурс", ["храним всё", "checkpoint", "пересчитываем"], "память ↔ операции"),
    ],
    26: [
        ("geometry", "$p_K$", "$p_N$", ["симплекс", "best response", "$(1/3,1/3,1/3)$"], "$p_K+p_N+p_B=1$"),
        ("contour", "$x$", "$y$", ["поле $xy$", "одновременный шаг", "экстраградиент"], "$r_{t+1}^2=(1+\\eta^2)r_t^2$"),
        ("matrix", "пиксель", "вариант", ["оригинал", "случайный шум", "FGSM"], "$\\delta=\\varepsilon\\,sign(\\nabla_x\\ell)$"),
    ],
    27: [
        ("matrix", "столбец окна", "строка", ["фрагмент", "ядро", "произведение"], "$z_{ij}=\\sum K_{uv}X_{i+u,j+v}$"),
        ("curves", "угол края", "ответ", ["0°", "45°", "90°", "135°"], "ориентационная настройка"),
        ("pipeline", "глубина", "поле", ["$3\\times3$", "$5\\times5$", "$7\\times7$"], "$r_l=r_{l-1}+(k_l-1)j_{l-1}$"),
    ],
    28: [
        ("pipeline", "позиция", "выход", ["padding", "ядро $3\\times3$", "stride 2"], "$H_{out}=\\lfloor(H+2p-k)/s\\rfloor+1$"),
        ("pipeline", "канал", "вклад", ["R", "G", "B", "сумма"], "$Y_o=\\sum_c K_{oc}*X_c+b_o$"),
        ("matrix", "позиция", "padding", ["zero", "reflect", "circular"], "ответ края"),
    ],
    29: [
        ("matrix", "окно", "операция", ["max", "average", "потерянная форма"], "$2\\times2\\to1$"),
        ("signal", "сдвиг, px", "расстояние карт", ["до pooling", "после max", "после average"], "фаза окна"),
        ("spectrum", "пространственная частота", "амплитуда", ["decimation", "average", "blur + decimation"], "aliasing"),
    ],
    30: [
        ("pipeline", "слой", "форма", ["32²×3", "32²×16", "16²×16", "8²×32", "классы"], "параметры • MAC • receptive field"),
        ("timeline", "год", "идея", ["локальные поля", "неокогнитрон", "LeNet", "ImageNet", "residual"], "архитектурные мосты"),
        ("tradeoff", "глубина", "пиксели", ["receptive field", "малый объект", "крупный объект"], "$r_l$ против размера объекта"),
    ],
    31: [
        ("matrix", "предсказанный класс", "истинный класс", ["counts", "row-normalized", "column-normalized"], "precision ↔ recall"),
        ("matrix", "пример", "тип ошибки", ["4→9", "обрезка", "разрыв", "метка?"], "галерея клетки"),
        ("tradeoff", "coverage", "accuracy", ["калибровка", "порог отказа", "нагрузка человеку"], "$\\max p_k\\ge\\tau$"),
    ],
    32: [
        ("pipeline", "данные", "доступ", ["train → веса", "validation → выбор", "test → отчёт"], "запрещённая стрелка назад"),
        ("matrix", "запись", "split", ["random row", "group patient", "temporal"], "пересечения идентификаторов"),
        ("timeline", "fold", "оценка", ["outer test", "inner train", "inner validation"], "nested CV"),
    ],
    33: [
        ("curves", "$x$", "$\\hat f(x)$", ["степень 1", "степень 5", "интерполяция"], "train ↔ test"),
        ("tradeoff", "эффективная сложность", "ошибка", ["train", "test", "интерполяционный порог"], "двойной спуск"),
        ("tradeoff", "ширина", "ошибка / норма", ["малый decay", "средний decay", "большой decay"], "регуляризация сдвигает порог"),
    ],
    34: [
        ("matrix", "пиксель $x$", "пиксель $y$", ["RGB", "$p(здание)$", "argmax", "target"], "$H\\times W\\times C$"),
        ("pipeline", "масштаб", "каналы", ["encoder", "bottleneck", "decoder", "skip"], "контекст + граница"),
        ("matrix", "геометрия", "вариант", ["сдвиг", "эрозия", "островки"], "одинаковый IoU, разные ошибки"),
    ],
    35: [
        ("pipeline", "слой", "размер", ["784", "128", "код 16", "128", "784"], "$x\\to z\\to\\hat x$"),
        ("geometry", "$x_1$", "$x_2$", ["данные", "проекция", "остаток"], "$\\hat x=VV^T(x-\\mu)+\\mu$"),
        ("matrix", "шум test", "шум train", ["Gaussian", "salt-pepper", "mask"], "ошибка переноса"),
    ],
    36: [
        ("geometry", "$x_1$", "$x_2$", ["$e_1,e_2$", "$Ae_1,Ae_2$", "сетка"], "$Ax=x_1Ae_1+x_2Ae_2$"),
        ("geometry", "$x$", "$y$", ["$\\det A=2$", "$\\det A=-1$", "$\\det A\\approx0$"], "площадь и ориентация"),
        ("geometry", "$x$", "$y$", ["$AB$", "$BA$", "исходная F"], "порядок композиции"),
    ],
    37: [
        ("matrix", "столбец / строка", "порядок доступа", ["ijk", "ikj", "cache lines"], "$C_{ij}=\\sum_kA_{ik}B_{kj}$"),
        ("pipeline", "блок $K$", "данные", ["$A_{IK}$", "$B_{KJ}$", "$C_{IJ}$", "cache"], "$C_{IJ}{+}=A_{IK}B_{KJ}$"),
        ("timeline", "уровень рекурсии", "число задач", ["8 ветвей", "7 ветвей", "сложения"], "$T(n)=7T(n/2)+O(n^2)$"),
    ],
    38: [
        ("geometry", "$x_1$", "$x_2$", ["до центрирования", "$\\mu$", "после"], "$X_c=X-\\mathbf1\\mu^T$"),
        ("curves", "угол $\\alpha$", "$u^TSu$", ["проекции", "дисперсия", "максимум"], "$Su=\\lambda u$"),
        ("matrix", "компонента / лицо", "вариант", ["scree plot", "$m=5$", "$m=20$", "$m=80$"], "$R_m=\\sum_{j\\le m}\\lambda_j/\\sum_j\\lambda_j$"),
    ],
    39: [
        ("matrix", "фильм", "пользователь", ["наблюдение", "пропуск", "$PQ^T$", "bias"], "$\\hat r=\\mu+a_u+b_i+p_u^Tq_i$"),
        ("geometry", "$z_1$", "$z_2$", ["до поворота", "после $R$", "те же соседи"], "$(Rp)^T(Rq)=p^Tq$"),
        ("tradeoff", "число train-оценок", "RMSE / uncertainty", ["cold", "tail", "head"], "качество длинного хвоста"),
    ],
    40: [
        ("spectrum", "время / частота", "амплитуда", ["тон 440 Гц", "тон 880 Гц", "DFT"], "$X_k=\\sum_nx_ne^{-2\\pi ikn/N}$"),
        ("spectrum", "время", "частота", ["короткое окно", "длинное окно", "STFT"], "время ↔ частота"),
        ("pipeline", "представление", "размер", ["waveform", "STFT", "mel", "статистики", "PCA", "соседи"], "аудио-эмбеддинг"),
    ],
}


FIGURE_RE = re.compile(
    r':::figure\{src="(?P<src>[^"]+)" id="(?P<id>[^"]+)" '
    r'title="(?P<title>[^"]+)" alt="(?P<alt>[^"]+)"\}\n'
    r"(?P<caption>.*?)\n:::",
    re.S,
)


def read_markdown_specs(lesson: int) -> list[dict[str, str]]:
    text = (CONTENT / f"{lesson:02d}.md").read_text(encoding="utf-8")
    matches = [m.groupdict() for m in FIGURE_RE.finditer(text)]
    if len(matches) != 3:
        raise ValueError(f"lesson {lesson:02d}: expected 3 figures, found {len(matches)}")
    return matches


def mathtext_safe(text: str) -> str:
    """Normalize a few LaTeX commands unsupported by Matplotlib mathtext."""
    normalized = (
        text.replace(r"\land", r"\wedge")
        .replace(r"\lor", r"\vee")
        .replace(r"\mathbf 1", "1")
        .replace(r"\mathbf1", "1")
        .replace(r"\operatorname", r"\mathrm")
        .replace("→", " к ")
        .replace("↔", " и ")
    )
    normalized = re.sub(r"\\ge(?![A-Za-z])", r"\\geq", normalized)
    normalized = re.sub(r"\\le(?![A-Za-z])", r"\\leq", normalized)
    return normalized


def finish_figure(fig: plt.Figure, title: str, caption: str, lesson: int, idx: int) -> None:
    fig.suptitle(title, x=0.055, y=0.965, ha="left", va="top", fontsize=17, fontweight="bold")
    clean = re.sub(r"\s+", " ", caption).strip()
    if len(clean) > 205:
        clean = clean[:202].rstrip() + "…"
    fig.text(
        0.055,
        0.032,
        f"Рис. {lesson}.{idx}. " + textwrap.fill(clean, width=112),
        ha="left",
        va="bottom",
        fontsize=8.6,
        color=MUTED,
        linespacing=1.12,
    )
    fig.subplots_adjust(left=0.075, right=0.97, bottom=0.19, top=0.82, wspace=0.32, hspace=0.42)


def add_note(ax: plt.Axes, note: str) -> None:
    ax.text(
        0.98,
        0.96,
        f"${note}$" if "$" not in note and "\\" in note else note,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=INK,
        bbox={"boxstyle": "round,pad=.35", "fc": PAPER, "ec": GRID, "lw": 0.8},
    )


def clean_axes(ax: plt.Axes, x: str, y: str) -> None:
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(True, color=GRID, lw=0.7, alpha=0.75)


def draw_boundary(fig, rng, xlab, ylab, labels, note, variant):
    if variant == 73:  # Object gradients and the variance reduction of averaging.
        ax, ax2 = fig.subplots(1, 2)
        mean = np.array([0.72, 0.42])
        cov = np.array([[0.72, -0.28], [-0.28, 0.46]])
        gradients = rng.multivariate_normal(mean, cov, 34)
        for g in gradients:
            ax.arrow(0, 0, g[0], g[1], color=BLUE, alpha=0.34, width=0.006, head_width=0.05, length_includes_head=True)
        empirical = gradients.mean(axis=0)
        ax.arrow(0, 0, empirical[0], empirical[1], color=RED, width=0.018, head_width=0.10, length_includes_head=True, label="$\\nabla L$")
        ax.scatter([empirical[0]], [empirical[1]], color=RED, s=38, zorder=5)
        ax.text(empirical[0] + 0.07, empirical[1] + 0.04, "$\\frac{1}{n}\\sum_i g_i$", color=RED, fontsize=9)
        ax.set_title("Голоса отдельных объектов", fontsize=10)
        ax.set_xlabel("$g_1$")
        ax.set_ylabel("$g_2$")
        ax.axhline(0, color=INK, lw=0.6)
        ax.axvline(0, color=INK, lw=0.6)
        ax.set_aspect("equal")
        ax.grid(True, color=GRID, lw=0.45)
        clouds = []
        batch_sizes = [1, 4, 16]
        for batch in batch_sizes:
            samples = rng.multivariate_normal(mean, cov / batch, 150)
            clouds.append(samples)
        for samples, batch, color, size, alpha in zip(
            clouds,
            batch_sizes,
            [BLUE, VIOLET, GREEN],
            [13, 15, 18],
            [0.24, 0.30, 0.52],
        ):
            ax2.scatter(samples[:, 0], samples[:, 1], color=color, s=size, alpha=alpha, label=f"$B={batch}$")
        ax2.scatter([mean[0]], [mean[1]], marker="*", color=RED, s=90, label="$\\nabla L$")
        for samples, color in zip(clouds, [BLUE, VIOLET, GREEN]):
            c = np.cov(samples.T)
            vals, vecs = np.linalg.eigh(c)
            theta = np.linspace(0, 2 * np.pi, 300)
            ellipse = mean[:, None] + vecs @ np.diag(2 * np.sqrt(vals)) @ np.vstack([np.cos(theta), np.sin(theta)])
            ax2.plot(ellipse[0], ellipse[1], color=color, lw=1.5)
        ax2.set_title("Средние случайных батчей", fontsize=10)
        ax2.set_xlabel("$\\hat g_1$")
        ax2.set_ylabel("$\\hat g_2$")
        ax2.axhline(0, color=INK, lw=0.6)
        ax2.axvline(0, color=INK, lw=0.6)
        ax2.set_aspect("equal")
        ax2.grid(True, color=GRID, lw=0.45)
        ax2.legend(frameon=False, fontsize=8)
        add_note(ax2, "$Var(\\hat g)\\approx Var(g_i)/B$")
        return
    ax, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.25, 0.75]})
    n = 24
    x0 = rng.normal([-0.8, -0.45], [0.5, 0.55], (n, 2))
    x1 = rng.normal([0.75, 0.55], [0.5, 0.55], (n, 2))
    ax.scatter(x0[:, 0], x0[:, 1], c=BLUE, s=32, label=labels[0], edgecolor=PAPER, lw=0.5)
    ax.scatter(x1[:, 0], x1[:, 1], c=RED, s=32, label=labels[1], marker="s", edgecolor=PAPER, lw=0.5)
    xx = np.linspace(-2.2, 2.2, 100)
    slope = 0.65 - 0.08 * (variant % 4)
    ax.plot(xx, -slope * xx, color=INK, lw=2.2, label=labels[2])
    ax.fill_between(xx, -slope * xx - 0.23, -slope * xx + 0.23, color=GOLD, alpha=0.16)
    ax.annotate("", xy=(0.75, 0.52), xytext=(0, 0), arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 2})
    ax.text(0.78, 0.54, "$w$", color=GREEN, fontsize=12)
    clean_axes(ax, xlab, ylab)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    scores = np.r_[x0 @ np.array([slope, 1]), x1 @ np.array([slope, 1])]
    colors = [BLUE] * n + [RED] * n
    ax2.hist(scores[:n], bins=11, color=BLUE, alpha=0.65, label=labels[0])
    ax2.hist(scores[n:], bins=11, color=RED, alpha=0.55, label=labels[1])
    ax2.axvline(0, color=INK, lw=1.5)
    ax2.set_title("Распределение поля")
    ax2.set_xlabel("$w^Tx-b$")
    ax2.set_ylabel("объекты")
    ax2.legend(frameon=False, fontsize=8)
    add_note(ax2, note)


def draw_network(fig, rng, xlab, ylab, labels, note, variant):
    ax = fig.subplots()
    ax.set_axis_off()
    layers = [3, 3, 2, 1] if variant % 2 else [3, 2, 1]
    xs = np.linspace(0.08, 0.78, len(layers))
    positions = []
    for li, (x, count) in enumerate(zip(xs, layers)):
        ys = np.linspace(0.25, 0.75, count)
        positions.append([(x, y) for y in ys])
        for ni, (px, py) in enumerate(positions[-1]):
            color = PALETTE[(li + ni + variant) % len(PALETTE)]
            ax.add_patch(Circle((px, py), 0.033, facecolor=PAPER, edgecolor=color, lw=2))
            ax.text(px, py, f"{['x','h','z','y'][min(li,3)]}$_{{{ni+1}}}$", ha="center", va="center", fontsize=9)
    for left, right in zip(positions[:-1], positions[1:]):
        for a in left:
            for b in right:
                ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, color=GRID, lw=0.8))
    panel_x = 0.84
    ax.add_patch(FancyBboxPatch((panel_x, 0.22), 0.14, 0.56, boxstyle="round,pad=.015", fc="#f5f2e9", ec=GRID))
    for i, label in enumerate(labels[:4]):
        ax.text(panel_x + 0.07, 0.68 - i * 0.13, label, ha="center", va="center", fontsize=9, color=PALETTE[i])
    ax.text(0.08, 0.88, xlab, fontsize=10, color=MUTED)
    ax.text(0.74, 0.88, ylab, fontsize=10, color=MUTED)
    ax.text(0.5, 0.08, note, ha="center", fontsize=10, color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def draw_signal(fig, rng, xlab, ylab, labels, note, variant):
    if variant == 44:  # Delayed control produces overshoot and oscillation.
        ax, ax2 = fig.subplots(2, 1, sharex=True)
        steps = 82
        t = np.arange(steps)
        target = np.where(t < 10, 19.0, np.where(t < 52, 22.0, 20.0))

        def simulate(delay):
            x = np.zeros(steps)
            u = np.zeros(steps)
            x[0] = 18.3
            for k in range(steps - 1):
                u[k] = 1.05 * (target[k] - x[k])
                applied = u[max(0, k - delay)]
                x[k + 1] = x[k] + 0.10 * (16.5 - x[k]) + 0.15 * applied
            return x, u

        x_fast, u_fast = simulate(0)
        x_delay, u_delay = simulate(7)
        ax.step(t, target, where="post", color=INK, ls="--", lw=1.4, label="цель $r_t$")
        ax.plot(t, x_fast, color=BLUE, label="без задержки")
        ax.plot(t, x_delay, color=RED, label="задержка $d=7$")
        ax.set_ylabel("температура, °C")
        ax.set_title("Одинаковый регулятор, разный момент действия", fontsize=10)
        ax.legend(frameon=False, fontsize=8, ncol=3)
        ax.grid(True, color=GRID, lw=0.55)
        ax2.plot(t, target - x_fast, color=BLUE, label="$e_t$, без задержки")
        ax2.plot(t, target - x_delay, color=RED, label="$e_t$, с задержкой")
        ax2.axhline(0, color=INK, lw=0.7)
        ax2.fill_between(t, 0, target - x_delay, color=RED, alpha=0.10)
        ax2.set_xlabel("шаг $t$")
        ax2.set_ylabel("ошибка, °C")
        ax2.set_title("Знак ошибки меняется после запоздалой команды", fontsize=9.5)
        ax2.legend(frameon=False, fontsize=8, ncol=2)
        ax2.grid(True, color=GRID, lw=0.55)
        add_note(ax2, "$u_t=K(r_t-x_t)$")
        return
    if variant == 45:  # Building load, battery action and state-of-charge.
        ax, ax2 = fig.subplots(2, 1, sharex=True)
        t = np.arange(0, 24, 0.25)
        load = 1.35 + 1.10 * np.exp(-0.5 * ((t - 8.3) / 1.7) ** 2) + 1.45 * np.exp(-0.5 * ((t - 18.2) / 2.2) ** 2)
        load += 0.10 * np.sin(2 * np.pi * t / 3.4)
        target = 1.75
        q = np.zeros_like(t)
        q[0] = 2.3
        capacity = 4.4
        action = np.zeros_like(t)
        dt = t[1] - t[0]
        for k in range(len(t) - 1):
            desired = np.clip(target - load[k], -0.72, 0.58)
            # Positive action charges the battery and raises grid load.
            if q[k] <= 0.02 and desired < 0:
                desired = 0
            if q[k] >= capacity - 0.02 and desired > 0:
                desired = 0
            action[k] = desired
            q[k + 1] = np.clip(q[k] + desired * dt, 0, capacity)
        action[-1] = action[-2]
        controlled = load + action
        ax.plot(t, load, color=BLUE, label="исходная нагрузка")
        ax.plot(t, controlled, color=RED, label="после аккумулятора")
        ax.axhline(target, color=INK, ls="--", lw=1, label="целевой уровень")
        ax.fill_between(t, load, controlled, where=action < 0, color=GREEN, alpha=0.16, label="разряд")
        ax.fill_between(t, load, controlled, where=action > 0, color=GOLD, alpha=0.16, label="заряд")
        ax.set_ylabel("мощность, кВт")
        ax.set_title("Аккумулятор переносит нагрузку между часами", fontsize=10)
        ax.legend(frameon=False, fontsize=7.5, ncol=3)
        ax.grid(True, color=GRID, lw=0.55)
        ax2.plot(t, q, color=GREEN, lw=2, label="$q_t$")
        ax2.fill_between(t, 0, q, color=GREEN, alpha=0.12)
        ax2.axhline(capacity, color=RED, ls="--", lw=1, label="$Q$")
        ax2.axhline(0, color=INK, lw=0.7)
        ax2.set_ylim(-0.18, capacity + 0.45)
        ax2.set_xlabel("час")
        ax2.set_ylabel("заряд, кВт·ч")
        ax2.legend(frameon=False, fontsize=8, ncol=2)
        ax2.grid(True, color=GRID, lw=0.55)
        add_note(ax2, "$0\\leq q_t\\leq Q$")
        return
    if variant == 60:  # Weekday fit, weekend residual.
        ax, ax2 = fig.subplots(2, 1, sharex=True)
        t = np.arange(7 * 24)
        hour = t % 24
        day = t // 24
        weekday_profile = 1.0 + 0.8 * np.exp(-0.5 * ((hour - 9) / 2.3) ** 2) + 1.05 * np.exp(-0.5 * ((hour - 18) / 2.7) ** 2)
        weekend = day >= 5
        actual = weekday_profile.copy()
        actual[weekend] = 0.78 + 0.48 * np.exp(-0.5 * ((hour[weekend] - 12) / 3.7) ** 2) + 0.72 * np.exp(-0.5 * ((hour[weekend] - 20) / 2.8) ** 2)
        actual += 0.045 * rng.normal(size=t.size)
        model = weekday_profile
        residual = actual - model
        ax.plot(t, actual, color=BLUE, lw=1.3, label="наблюдение")
        ax.plot(t, model, color=RED, lw=1.7, label="ReLU-модель по рабочим дням")
        ax.axvspan(5 * 24, 7 * 24, color=GOLD, alpha=0.12, label="выходные")
        ax.set_ylabel("нагрузка")
        ax.set_title("Точное правило будней ошибается на выходных", fontsize=10)
        ax.legend(frameon=False, fontsize=8, ncol=3)
        ax.grid(True, color=GRID, lw=0.5)
        ax2.plot(t, residual, color=GREEN, lw=1.4)
        ax2.axhline(0, color=INK, lw=0.7)
        ax2.axvspan(5 * 24, 7 * 24, color=GOLD, alpha=0.12)
        ax2.set_xlabel("час недели")
        ax2.set_ylabel("остаток")
        ax2.set_xticks(np.arange(12, 7 * 24, 24), ["пн", "вт", "ср", "чт", "пт", "сб", "вс"])
        ax2.grid(True, color=GRID, lw=0.5)
        add_note(ax2, "систематический остаток")
        return
    if variant == 75:  # Three batching schemes for a correlated time series.
        ax, ax2 = fig.subplots(2, 1, sharex=False)
        t = np.arange(7 * 24)
        hour = t % 24
        day = t // 24
        load = 1.0 + 0.65 * np.sin(2 * np.pi * (hour - 7) / 24) + 0.25 * np.sin(4 * np.pi * hour / 24)
        load += 0.08 * day + 0.07 * rng.normal(size=t.size)
        ax.plot(t, load, color=INK, lw=1.1, alpha=0.75)
        random_hours = np.sort(rng.choice(t, size=13, replace=False))
        window = np.arange(55, 79)
        whole_days = np.r_[np.arange(0, 24), np.arange(120, 144)]
        ax.scatter(random_hours, load[random_hours], color=BLUE, s=22, label="случайные часы", zorder=4)
        ax.plot(window, load[window], color=RED, lw=3.2, alpha=0.72, label="окно 24 часа")
        ax.plot(whole_days[:24], load[whole_days[:24]], color=GREEN, lw=2.4, label="целые дни")
        ax.plot(whole_days[24:], load[whole_days[24:]], color=GREEN, lw=2.4)
        ax.set_xlabel("час недели")
        ax.set_ylabel("нагрузка")
        ax.set_title("Какие строки попали в один батч", fontsize=10)
        ax.legend(frameon=False, fontsize=8, ncol=3)
        ax.grid(True, color=GRID, lw=0.5)
        # Simulated estimates make the variance/correlation distinction visible.
        schemes = ["случайные\nчасы", "окна\n24 ч", "целые\nдни"]
        estimates = [
            rng.normal(0.72, 0.13, 18),
            rng.normal(0.72, 0.25, 18),
            rng.normal(0.72, 0.16, 18),
        ]
        for i, (vals, color) in enumerate(zip(estimates, [BLUE, RED, GREEN])):
            jitter = rng.uniform(-0.10, 0.10, len(vals))
            ax2.scatter(i + jitter, vals, color=color, s=19, alpha=0.72)
            ax2.errorbar(i, np.mean(vals), yerr=np.std(vals), fmt="o", color=INK, capsize=4, ms=4)
        ax2.axhline(0.72, color=GOLD, ls="--", lw=1.2, label="полный градиент")
        ax2.set_xticks(range(3), schemes)
        ax2.set_ylabel("оценка градиента")
        ax2.set_title("Разброс батч-оценок при одинаковом среднем", fontsize=9.5)
        ax2.grid(True, axis="y", color=GRID, lw=0.5)
        ax2.legend(frameon=False, fontsize=8)
        add_note(ax2, "$E[\\hat g]=\\nabla L$")
        return
    if variant == 89:  # Pooling-window phase under one-pixel shifts.
        ax, ax2 = fig.subplots(2, 1)
        positions = np.arange(16)
        signal = np.exp(-0.5 * ((positions - 7.5) / 0.72) ** 2)
        ax.bar(positions, signal, width=0.9, color=BLUE, alpha=0.78, label="яркий штрих")
        for edge in np.arange(-0.5, 16, 2):
            ax.axvline(edge, color=RED, lw=0.9, alpha=0.75)
        ax.annotate("границы окон $2\\times1$", xy=(7.5, 0.78), xytext=(10.0, 0.93), arrowprops={"arrowstyle": "->", "color": RED}, fontsize=8)
        ax.set_xlim(-0.7, 15.7)
        ax.set_xlabel("позиция пикселя")
        ax.set_ylabel("яркость")
        ax.set_title("Штрих пересекает границу окна", fontsize=10)
        ax.grid(True, axis="y", color=GRID, lw=0.5)
        shifts = np.linspace(0, 4, 81)
        before = np.abs(np.sin(np.pi * shifts)) * 0.68 + 0.15 * shifts
        after_max = 0.16 * shifts + 0.56 * np.maximum(0, np.sin(np.pi * (shifts - 0.55))) ** 4
        after_avg = 0.10 * shifts + 0.13 * (1 - np.cos(np.pi * shifts))
        ax2.plot(shifts, before, color=BLUE, label="до pooling")
        ax2.plot(shifts, after_max, color=RED, label="после max")
        ax2.plot(shifts, after_avg, color=GREEN, label="после average")
        for edge in [1, 2, 3]:
            ax2.axvline(edge, color=GRID, lw=0.8)
        ax2.set_xlabel("сдвиг, px")
        ax2.set_ylabel("расстояние карт")
        ax2.set_title("Инвариантность ломается на фазе окна", fontsize=9.5)
        ax2.legend(frameon=False, fontsize=8, ncol=3)
        ax2.grid(True, color=GRID, lw=0.5)
        add_note(ax2, "фаза окна")
        return
    ax, ax2 = fig.subplots(2, 1, sharex=True)
    t = np.linspace(0, 24, 240)
    base = 0.55 * np.sin(2 * np.pi * (t - 5) / 24) + 0.22 * np.sin(4 * np.pi * t / 24)
    for i, label in enumerate(labels[:3]):
        if i == 0:
            y = base + 0.04 * rng.normal(size=t.size)
        elif i == 1:
            y = 0.82 * base + 0.18 * np.roll(base, 13) + 0.1
        else:
            y = 0.35 * np.cos(2 * np.pi * t / 24) - 0.15
        ax.plot(t, y, color=PALETTE[i], label=label, lw=1.8)
    ax.axhline(0, color=INK, ls="--", lw=1)
    ax.set_ylabel(ylab)
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper right")
    ax.grid(True, color=GRID, lw=0.6)
    threshold = np.quantile(base, 0.62)
    binary = (base > threshold).astype(float)
    ax2.step(t, binary, where="mid", color=RED, lw=2, label="бинарный сигнал")
    ax2.fill_between(t, 0, binary, step="mid", color=RED, alpha=0.15)
    ax2.set_xlabel(xlab)
    ax2.set_ylabel("0 / 1")
    ax2.set_yticks([0, 1])
    ax2.grid(True, axis="x", color=GRID, lw=0.6)
    add_note(ax2, note)


def draw_curves(fig, rng, xlab, ylab, labels, note, variant):
    if variant == 51:  # Soft XOR: steepness versus available derivative.
        ax, ax2 = fig.subplots(1, 2)
        x = np.linspace(-1.5, 1.5, 500)
        for i, k in enumerate([1, 5, 20]):
            s = 1 / (1 + np.exp(-k * x))
            ax.plot(x, s, color=PALETTE[i], label=f"$k={k}$")
            ax2.plot(x, k * s * (1 - s), color=PALETTE[i], label=f"$k={k}$")
        ax.set_title("Мягкий порог")
        ax.set_xlabel("поле $z$")
        ax.set_ylabel("$\\sigma(kz)$")
        ax2.set_title("Обучающий сигнал")
        ax2.set_xlabel("поле $z$")
        ax2.set_ylabel("$|d\\sigma/dz|$")
        for a in [ax, ax2]:
            a.axvline(0, color=INK, lw=0.8)
            a.grid(True, color=GRID, lw=0.6)
            a.legend(frameon=False, fontsize=8)
        add_note(ax2, "$\\sigma'(kz)$")
        return
    if variant == 53:  # Shifted ReLU basis and its weighted sum.
        ax, ax2 = fig.subplots(1, 2)
        x = np.linspace(-3, 4, 500)
        knots = [-1.3, 0.4, 2.1]
        weights = [0.9, -1.5, 1.1]
        components = []
        for i, (t, w) in enumerate(zip(knots, weights)):
            y = w * np.maximum(x - t, 0)
            components.append(y)
            ax.plot(x, y, color=PALETTE[i], label=f"$v_{i+1}ReLU(x-t_{i+1})$")
            ax.axvline(t, color=PALETTE[i], ls=":", lw=0.9)
        total = 0.35 * x + np.sum(components, axis=0)
        ax2.plot(x, total, color=INK, lw=2.4, label="$f(x)$")
        slopes = np.gradient(total, x)
        ax2.step(x, slopes, where="mid", color=GREEN, lw=1.6, label="наклон")
        for t in knots:
            ax2.axvline(t, color=GOLD, ls="--", lw=0.8)
        ax.set_title("Сдвинутые ReLU")
        ax2.set_title("Сумма и изменения наклона")
        for a in [ax, ax2]:
            a.set_xlabel("$x$")
            a.grid(True, color=GRID, lw=0.6)
            a.legend(frameon=False, fontsize=8)
        ax.set_ylabel("вклад")
        ax2.set_ylabel("$f(x)$ / наклон")
        add_note(ax2, "$k_j-k_{j-1}=v_j$")
        return
    if variant == 58:  # Nodes, target curve and exact linear interpolant.
        ax, ax2 = fig.subplots(1, 2)
        x = np.linspace(-3.2, 3.2, 600)
        target = np.sin(1.25 * x) + 0.13 * x
        knots = np.linspace(-3, 3, 8)
        values = np.sin(1.25 * knots) + 0.13 * knots
        interp = np.interp(x, knots, values)
        ax.plot(x, target, color=BLUE, label="целевая функция")
        ax.plot(x, interp, color=RED, lw=2, label="ReLU-ломаная")
        ax.scatter(knots, values, color=GOLD, zorder=4, label="узлы $t_j$")
        ax2.plot(x, interp - target, color=GREEN)
        ax2.fill_between(x, 0, interp - target, color=GREEN, alpha=0.16)
        ax2.axhline(0, color=INK, lw=0.8)
        ax.set_title("Значения и интерполяция")
        ax2.set_title("Ошибка между узлами")
        for a in [ax, ax2]:
            a.set_xlabel("$x$")
            a.grid(True, color=GRID, lw=0.6)
        ax.set_ylabel("$f(x)$")
        ax2.set_ylabel("$g(x)-f(x)$")
        ax.legend(frameon=False, fontsize=8)
        add_note(ax2, "$\\max|g-f|$")
        return
    if variant == 62:  # Regression losses and their gradients.
        ax, ax2 = fig.subplots(1, 2)
        e = np.linspace(-5, 5, 700)
        delta, tau = 1.2, 0.8
        loss = [
            (0.5 * e**2, e, "MSE"),
            (np.abs(e), np.sign(e), "MAE"),
            (np.where(np.abs(e) <= delta, 0.5 * e**2, delta * (np.abs(e) - 0.5 * delta)), np.clip(e, -delta, delta), "Huber"),
            (np.where(e >= 0, tau * e, (tau - 1) * e), np.where(e >= 0, tau, tau - 1), "quantile"),
        ]
        for i, (ell, grad, label) in enumerate(loss):
            ax.plot(e, ell, color=PALETTE[i], label=label)
            ax2.plot(e, grad, color=PALETTE[i], label=label)
        ax.set_ylim(-0.2, 9)
        ax.set_title("Числовая цена")
        ax2.set_title("Сила и знак обновления")
        for a in [ax, ax2]:
            a.axvline(0, color=INK, lw=0.8)
            a.grid(True, color=GRID, lw=0.6)
            a.set_xlabel("остаток $e$")
            a.legend(frameon=False, fontsize=8)
        ax.set_ylabel("$\\ell(e)$")
        ax2.set_ylabel("$d\\ell/de$")
        add_note(ax2, "$\\delta=1.2,\\;\\tau=0.8$")
        return
    if variant == 64:  # Four stationary-point archetypes.
        axes = fig.subplots(2, 2)
        x = np.linspace(-2, 2, 500)
        functions = [
            (x**2, "$x^2$", "минимум"),
            (-x**2, "$-x^2$", "максимум"),
            (x**3, "$x^3$", "горизонтальный перегиб"),
            ((x**2 - 1) ** 2, "$(x^2-1)^2$", "два минимума"),
        ]
        for ax, (y, formula, kind) in zip(axes.ravel(), functions):
            ax.plot(x, y, color=BLUE, lw=2)
            ax.scatter([0], [y[len(y)//2]], color=RED, s=32, zorder=4)
            ax.axhline(0, color=INK, lw=0.7)
            ax.axvline(0, color=INK, lw=0.7)
            ax.set_title(f"{formula}: {kind}", fontsize=10)
            ax.grid(True, color=GRID, lw=0.5)
            ax.set_ylim(max(-3, np.min(y)), min(5, np.max(y)))
        add_note(axes[1, 1], "$f'(0)=0$")
        return
    if variant == 83:  # Orientation tuning in Cartesian and polar coordinates.
        ax = fig.add_subplot(1, 2, 1)
        polar = fig.add_subplot(1, 2, 2, projection="polar")
        theta = np.linspace(0, np.pi, 500)
        centers = np.deg2rad([0, 45, 90, 135])
        for i, center in enumerate(centers):
            diff = np.angle(np.exp(2j * (theta - center))) / 2
            response = np.exp(-0.5 * (diff / 0.28) ** 2)
            ax.plot(np.rad2deg(theta), response, color=PALETTE[i], label=f"{[0,45,90,135][i]}°")
            polar.plot(2 * theta, response, color=PALETTE[i], lw=1.6)
        ax.set_xlabel("угол края, °")
        ax.set_ylabel("нормированный ответ")
        ax.set_title("Настройка четырёх фильтров")
        ax.grid(True, color=GRID, lw=0.6)
        ax.legend(frameon=False, ncol=2, fontsize=8)
        polar.set_title("Ориентация имеет период 180°", fontsize=10)
        polar.set_yticklabels([])
        polar.grid(color=GRID, lw=0.6)
        return
    if variant == 100:  # Polynomial interpolation and between-node error.
        grid = fig.add_gridspec(2, 3, height_ratios=[1.05, 0.72])
        top_axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
        ax2 = fig.add_subplot(grid[1, :])
        x_train = np.linspace(-1, 1, 20)
        truth_train = np.sin(2.8 * x_train) + 0.16 * np.cos(6.5 * x_train)
        y_train = truth_train + 0.07 * rng.normal(size=x_train.size)
        x = np.linspace(-1.06, 1.06, 700)
        truth = np.sin(2.8 * x) + 0.16 * np.cos(6.5 * x)
        degrees_shown = [1, 5, 19]
        predictions = {}
        for ax, degree, color in zip(top_axes, degrees_shown, [BLUE, GREEN, RED]):
            model = np.polynomial.Chebyshev.fit(x_train, y_train, degree)
            pred = model(x)
            predictions[degree] = pred
            ax.plot(x, truth, color=INK, ls="--", lw=1.1, label="скрытая функция")
            ax.plot(x, pred, color=color, lw=2, label=f"степень {degree}")
            ax.scatter(x_train, y_train, color=GOLD, edgecolor=PAPER, lw=0.5, s=20, zorder=4)
            ax.set_title(f"Полином степени {degree}", fontsize=9.5)
            ax.set_xlim(-1.06, 1.06)
            ax.set_ylim(-1.65, 1.65)
            ax.set_xlabel("$x$")
            ax.grid(True, color=GRID, lw=0.45)
            ax.legend(frameon=False, fontsize=6.8, loc="lower left")
        top_axes[0].set_ylabel("$\\hat f(x)$")
        degrees = np.arange(1, 20)
        train_err, test_err = [], []
        for degree in degrees:
            model = np.polynomial.Chebyshev.fit(x_train, y_train, int(degree))
            train_err.append(np.mean((model(x_train) - y_train) ** 2))
            test_err.append(np.mean((model(x) - truth) ** 2))
        ax2.semilogy(degrees, train_err, "-o", color=BLUE, ms=3, label="train MSE")
        ax2.semilogy(degrees, test_err, "-s", color=RED, ms=3, label="dense test MSE")
        ax2.axvline(19, color=GOLD, ls="--", label="интерполяционный порог")
        ax2.set_title("Нулевой train не обещает малую ошибку между узлами", fontsize=9.5)
        ax2.set_xlabel("степень полинома")
        ax2.set_ylabel("MSE, log")
        ax2.grid(True, color=GRID, lw=0.5)
        ax2.legend(frameon=False, fontsize=7.5, ncol=3)
        return
    if variant == 116:  # PCA variance as a function of angle.
        ax, ax2 = fig.subplots(1, 2)
        cov = np.array([[3.0, 1.35], [1.35, 1.0]])
        pts = rng.multivariate_normal([0, 0], cov, 120)
        ax.scatter(pts[:, 0], pts[:, 1], s=18, color=BLUE, alpha=0.55)
        angles = np.linspace(0, np.pi, 500)
        variance = np.array([np.array([np.cos(a), np.sin(a)]) @ cov @ np.array([np.cos(a), np.sin(a)]) for a in angles])
        best = angles[np.argmax(variance)]
        for angle, color, label in [(0.25, RED, "ось $u$"), (best, GREEN, "максимум")]:
            u = np.array([np.cos(angle), np.sin(angle)])
            ax.plot([-3*u[0], 3*u[0]], [-3*u[1], 3*u[1]], color=color, lw=2, label=label)
        ax.set_aspect("equal")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title("Проекция облака")
        ax.legend(frameon=False, fontsize=8)
        ax2.plot(np.rad2deg(angles), variance, color=BLUE)
        ax2.axvline(np.rad2deg(best), color=GREEN, ls="--", label="$u_1$")
        ax2.scatter([np.rad2deg(best)], [variance.max()], color=GOLD, zorder=4)
        ax2.set_xlabel("угол $\\alpha$, °")
        ax2.set_ylabel("$u^TSu$")
        ax2.set_title("Дисперсия проекции")
        ax2.grid(True, color=GRID, lw=0.6)
        ax2.legend(frameon=False, fontsize=8)
        add_note(ax2, "$Su=\\lambda u$")
        return
    ax, ax2 = fig.subplots(1, 2)
    x = np.linspace(-4, 4, 500)
    modes = variant % 4
    curves = []
    if modes == 0:
        curves = [
            (1 / (1 + np.exp(-x)), 1 / (1 + np.exp(-x)) * (1 - 1 / (1 + np.exp(-x)))),
            (np.tanh(x), 1 - np.tanh(x) ** 2),
            (np.maximum(x, 0), (x > 0).astype(float)),
            (np.where(x > 0, x, 0.1 * x), np.where(x > 0, 1, 0.1)),
        ]
    elif modes == 1:
        target = np.sin(1.2 * x) * 0.8
        curves = [(target, np.gradient(target, x))]
        for k in [2, 4, 8]:
            knots = np.linspace(-4, 4, k)
            y = np.interp(x, knots, np.sin(1.2 * knots) * 0.8)
            curves.append((y, np.gradient(y, x)))
    elif modes == 2:
        curves = [
            (x**2 / 4, x / 2),
            (-x**2 / 4, -x / 2),
            (x**3 / 12, x**2 / 4),
            ((x**2 - 1) ** 2 / 5, 4 * x * (x**2 - 1) / 5),
        ]
    else:
        curves = [
            (np.sin(x), np.cos(x)),
            (np.sin(x) + 0.18 * x, np.cos(x) + 0.18),
            (np.interp(x, np.linspace(-4, 4, 7), np.sin(np.linspace(-4, 4, 7))), np.zeros_like(x)),
        ]
    for i, (y, dy) in enumerate(curves[: max(3, len(labels))]):
        lab = labels[i] if i < len(labels) else f"вариант {i+1}"
        ax.plot(x, y, color=PALETTE[i], label=lab, lw=1.8)
        ax2.plot(x, dy, color=PALETTE[i], label=lab, lw=1.6)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title("Функции")
    ax2.set_xlabel(xlab)
    ax2.set_ylabel("локальное изменение")
    ax2.set_title("Производная / чувствительность")
    for a in [ax, ax2]:
        a.axhline(0, color=INK, lw=0.8)
        a.grid(True, color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8)
    add_note(ax2, note)


def draw_tradeoff(fig, rng, xlab, ylab, labels, note, variant):
    if variant == 96:  # Reliability and selective prediction are different axes.
        ax, ax2 = fig.subplots(1, 2)
        confidence = np.linspace(0.08, 0.96, 10)
        empirical = np.clip(confidence - 0.12 * np.sin(np.pi * confidence) + rng.normal(0, 0.025, confidence.size), 0, 1)
        counts = np.array([42, 68, 91, 105, 128, 132, 121, 96, 72, 35])
        ax.plot([0, 1], [0, 1], color=INK, ls="--", lw=1.1, label="идеальная калибровка")
        ax.plot(confidence, empirical, "-o", color=BLUE, ms=4, label="модель")
        for x0, y0, count in zip(confidence, empirical, counts):
            ax.vlines(x0, x0, y0, color=RED, alpha=0.55, lw=1.3)
            ax.text(x0, min(1.02, y0 + 0.05), str(count), ha="center", fontsize=6.5, color=MUTED)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("предсказанная уверенность")
        ax.set_ylabel("доля правильных")
        ax.set_title("Reliability diagram", fontsize=10)
        ax.grid(True, color=GRID, lw=0.5)
        ax.legend(frameon=False, fontsize=7.5)
        thresholds = np.linspace(0.0, 0.95, 80)
        coverage = 1 - thresholds**1.45
        accuracy = 0.78 + 0.20 * thresholds**0.78
        workload = 1 - coverage
        ax2.plot(coverage, accuracy, color=GREEN, lw=2, label="accuracy принятых")
        selected = [0.0, 0.35, 0.65, 0.85]
        for tau in selected:
            idx = np.argmin(np.abs(thresholds - tau))
            ax2.scatter([coverage[idx]], [accuracy[idx]], color=GOLD, s=34, zorder=4)
            ax2.text(coverage[idx] - 0.02, accuracy[idx] + 0.012, f"$\\tau={tau:.2g}$", fontsize=7, ha="right")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0.76, 1.01)
        ax2.set_xlabel("coverage")
        ax2.set_ylabel("accuracy")
        ax2.set_title("Цена отказа: очередь человеку", fontsize=10)
        ax2.grid(True, color=GRID, lw=0.5)
        ax2_top = ax2.twiny()
        ax2_top.set_xlim(ax2.get_xlim())
        ax2_top.set_xticks([0, 0.25, 0.5, 0.75, 1], ["100%", "75%", "50%", "25%", "0%"])
        ax2_top.set_xlabel("доля отказов")
        add_note(ax2, "$\\max_kp_k\\geq\\tau$")
        return
    if variant == 101:  # Genuine double descent around interpolation.
        ax, ax2 = fig.subplots(1, 2)
        complexity = np.linspace(0.05, 2.1, 500)
        train = 0.72 * np.maximum(1 - complexity, 0) ** 1.65 + 0.012 * np.exp(-2.8 * complexity)
        classical = 0.16 + 0.38 * (complexity - 0.48) ** 2
        peak = 0.62 / (1 + ((complexity - 1.0) / 0.085) ** 2)
        second_descent = 0.30 * np.exp(-1.8 * np.maximum(complexity - 1.0, 0))
        test = 0.13 + 0.42 * np.exp(-3.4 * complexity) + peak + second_descent
        ax.plot(complexity, train, color=BLUE, lw=2, label="train")
        ax.plot(complexity, test, color=RED, lw=2, label="test")
        ax.axvline(1.0, color=GOLD, ls="--", lw=1.5, label="интерполяционный порог")
        ax.fill_between(complexity, 0, test, where=np.abs(complexity - 1) < 0.14, color=GOLD, alpha=0.12)
        ax.text(0.43, 0.75, "классический\nспуск", ha="center", fontsize=8, color=GREEN)
        ax.text(1.02, 0.92, "пик", ha="center", fontsize=8, color=GOLD)
        ax.text(1.63, 0.41, "второй спуск", ha="center", fontsize=8, color=VIOLET)
        ax.set_xlabel("эффективная сложность")
        ax.set_ylabel("ошибка")
        ax.set_title("Две стороны порога интерполяции", fontsize=10)
        ax.set_ylim(0, 1.08)
        ax.grid(True, color=GRID, lw=0.5)
        ax.legend(frameon=False, fontsize=7.6)
        regimes = ["до порога", "у порога", "после порога"]
        bias = np.array([0.42, 0.12, 0.08])
        variance = np.array([0.12, 0.62, 0.20])
        noise = np.array([0.10, 0.10, 0.10])
        xbar = np.arange(3)
        ax2.bar(xbar, bias, color=BLUE, label="смещение")
        ax2.bar(xbar, variance, bottom=bias, color=RED, label="вариативность")
        ax2.bar(xbar, noise, bottom=bias + variance, color=GREEN, label="шум")
        ax2.set_xticks(xbar, regimes)
        ax2.set_ylabel("условный вклад в test-error")
        ax2.set_title("Почему пик не равен «переобучению вообще»", fontsize=9.5)
        ax2.grid(True, axis="y", color=GRID, lw=0.5)
        ax2.legend(frameon=False, fontsize=7.5)
        add_note(ax2, "эффективная сложность зависит от обучения")
        return
    ax, ax2 = fig.subplots(1, 2)
    x = np.linspace(0.05, 1, 120)
    for i, label in enumerate(labels[:3]):
        if i == 0:
            y = 0.1 + 0.8 / (1 + 7 * x)
        elif i == 1:
            y = 0.15 + 0.7 * (x - 0.55) ** 2
        else:
            y = 0.2 + 0.5 * np.exp(-3 * x) + 0.12 * np.sin(9 * x + variant)
        ax.plot(x, y, color=PALETTE[i], label=label)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    n_labels = min(3, len(labels))
    vals = np.array([0.32, 0.55, 0.74])[:n_labels] + 0.03 * np.sin(variant + np.arange(n_labels))
    err = np.array([0.07, 0.05, 0.09])[:n_labels]
    ax2.bar(np.arange(len(vals)), vals, yerr=err, color=PALETTE[:3], alpha=0.85, capsize=3)
    ax2.set_xticks(range(len(vals)), [textwrap.fill(v, 12) for v in labels[:n_labels]], fontsize=8)
    ax2.set_ylabel("измерение")
    ax2.set_title("Сравнение при общей шкале")
    ax2.grid(True, axis="y", color=GRID, lw=0.6)
    add_note(ax2, note)


def draw_contour(fig, rng, xlab, ylab, labels, note, variant):
    if variant == 66:  # Isotropic and ill-conditioned quadratic bowls.
        axes = fig.subplots(1, 2)
        x = np.linspace(-2.2, 2.2, 260)
        y = np.linspace(-2.2, 2.2, 260)
        X, Y = np.meshgrid(x, y)
        cases = [
            (X**2 + 1.15 * Y**2, np.array([[1.0, 0.0], [0.0, 1.15]]), "Почти круглая чаша", "$\\lambda=(1,1.15),\\;\\kappa=1.15$"),
            (0.20 * (0.86 * X + 0.50 * Y) ** 2 + 5.0 * (-0.50 * X + 0.86 * Y) ** 2,
             np.array([[0.86, -0.50], [0.50, 0.86]]) @ np.diag([0.20, 5.0]) @ np.array([[0.86, 0.50], [-0.50, 0.86]]),
             "Узкая чаша",
             "$\\lambda=(0.2,5),\\;\\kappa=25$"),
        ]
        for ax, (Z, H, title, formula) in zip(axes, cases):
            ax.contour(X, Y, Z, levels=[0.15, 0.35, 0.7, 1.2, 2.0, 3.2, 4.8], colors=BLUE, linewidths=1.0)
            vals, vecs = np.linalg.eigh(H)
            for val, vec, color in zip(vals, vecs.T, [GREEN, RED]):
                scale = 1.45 / math.sqrt(val)
                ax.arrow(0, 0, scale * vec[0], scale * vec[1], color=color, width=0.012, head_width=0.09, length_includes_head=True)
                ax.text(scale * vec[0] + 0.05, scale * vec[1] + 0.04, f"$\\lambda={val:.2g}$", fontsize=8, color=color)
            ax.scatter([0], [0], color=GOLD, s=42, zorder=4)
            ax.set_title(f"{title}\n{formula}", fontsize=10)
            ax.set_aspect("equal")
            ax.set_xlabel("$\\theta_1$")
            ax.set_ylabel("$\\theta_2$")
            ax.grid(True, color=GRID, lw=0.42)
        add_note(axes[-1], "$\\kappa=\\lambda_{max}/\\lambda_{min}$")
        return
    if variant == 70:  # Equality constraint and objective tangency.
        ax, ax2 = fig.subplots(1, 2)
        x = np.linspace(-1.65, 2.25, 260)
        y = np.linspace(-1.65, 1.85, 260)
        X, Y = np.meshgrid(x, y)
        center = np.array([1.65, 0.72])
        Z = (X - center[0]) ** 2 + (Y - center[1]) ** 2
        theta = np.linspace(0, 2 * np.pi, 500)
        circle = np.column_stack([np.cos(theta), np.sin(theta)])
        optimum = center / np.linalg.norm(center)
        q = np.array([np.cos(2.05), np.sin(2.05)])
        cs = ax.contour(X, Y, Z, levels=[0.12, 0.28, 0.52, 0.82, 1.25, 1.8, 2.5, 3.4], colors=BLUE, linewidths=0.9)
        ax.clabel(cs, fontsize=6.5, inline=True)
        ax.plot(circle[:, 0], circle[:, 1], color=RED, lw=2.2, label="$g(x,y)=x^2+y^2-1=0$")
        ax.scatter(*optimum, color=GOLD, s=58, zorder=5, label="$x^*$")
        tangent = np.array([-optimum[1], optimum[0]])
        ax.plot(
            [optimum[0] - 0.55 * tangent[0], optimum[0] + 0.55 * tangent[0]],
            [optimum[1] - 0.55 * tangent[1], optimum[1] + 0.55 * tangent[1]],
            color=GREEN,
            lw=1.7,
            label="допустимая касательная",
        )
        gf = 2 * (optimum - center)
        gg = 2 * optimum
        ax.arrow(optimum[0], optimum[1], 0.42 * gf[0], 0.42 * gf[1], color=BLUE, width=0.012, head_width=0.09, length_includes_head=True)
        ax.arrow(optimum[0], optimum[1], 0.36 * gg[0], 0.36 * gg[1], color=RED, width=0.012, head_width=0.09, length_includes_head=True)
        ax.text(optimum[0] - 0.48, optimum[1] - 0.29, "$\\nabla f$", color=BLUE)
        ax.text(optimum[0] + 0.25, optimum[1] + 0.16, "$\\nabla g$", color=RED)
        ax.set_title("В оптимуме: касание")
        ax.set_aspect("equal")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.legend(frameon=False, fontsize=7.8, loc="lower left")
        # At a neighbouring feasible point the tangential derivative is not zero.
        ax2.contour(X, Y, Z, levels=[0.28, 0.52, 0.82, 1.25, 1.8, 2.5, 3.4], colors=BLUE, linewidths=0.9)
        ax2.plot(circle[:, 0], circle[:, 1], color=RED, lw=2.2)
        ax2.scatter(*q, color=VIOLET, s=50, zorder=5)
        tangent_q = np.array([-q[1], q[0]])
        grad_q = 2 * (q - center)
        directional = grad_q @ tangent_q
        ax2.arrow(q[0], q[1], 0.55 * tangent_q[0], 0.55 * tangent_q[1], color=GREEN, width=0.012, head_width=0.09, length_includes_head=True)
        ax2.arrow(q[0], q[1], 0.20 * grad_q[0], 0.20 * grad_q[1], color=BLUE, width=0.012, head_width=0.09, length_includes_head=True)
        ax2.text(-1.48, 1.45, f"$\\nabla f^Tv={directional:.2f}\\ne0$", fontsize=9)
        ax2.set_title("В соседней точке: есть спуск")
        ax2.set_aspect("equal")
        ax2.set_xlabel("$x$")
        ax2.set_ylabel("$y$")
        add_note(ax2, "$\\nabla f=\\lambda\\nabla g$")
        return
    if variant == 72:  # The same quadratic loss touching L2 and L1 balls.
        axes = fig.subplots(1, 2)
        x = np.linspace(-1.35, 1.75, 260)
        y = np.linspace(-1.35, 1.35, 260)
        X, Y = np.meshgrid(x, y)
        target = np.array([1.45, 0.35])
        Z = (X - target[0]) ** 2 + 1.55 * (Y - target[1]) ** 2
        radii = [0.52, 0.78, 1.02]
        for panel, (ax, norm_name) in enumerate(zip(axes, ["L2", "L1"])):
            ax.contour(X, Y, Z, levels=[0.12, 0.28, 0.52, 0.82, 1.25, 1.8, 2.5], colors=BLUE, linewidths=0.85)
            path = []
            for ri, radius in enumerate(radii):
                if norm_name == "L2":
                    t = np.linspace(0, 2 * np.pi, 1200)
                    boundary = np.column_stack([radius * np.cos(t), radius * np.sin(t)])
                else:
                    t = np.linspace(-radius, radius, 500)
                    boundary = np.vstack(
                        [
                            np.column_stack([t, radius - np.abs(t)]),
                            np.column_stack([t[::-1], -radius + np.abs(t[::-1])]),
                        ]
                    )
                loss_on_boundary = (boundary[:, 0] - target[0]) ** 2 + 1.55 * (boundary[:, 1] - target[1]) ** 2
                optimum = boundary[np.argmin(loss_on_boundary)]
                path.append(optimum)
                if ri == 1:
                    closed = np.vstack([boundary, boundary[0]])
                    ax.fill(closed[:, 0], closed[:, 1], color=GREEN, alpha=0.08)
                    ax.plot(closed[:, 0], closed[:, 1], color=GREEN, lw=2, label=f"${norm_name}\\leq {radius}$")
            path = np.asarray(path)
            ax.plot(path[:, 0], path[:, 1], "-o", color=GOLD, lw=1.8, ms=5, label="решения при разных $R$")
            ax.scatter([target[0]], [target[1]], marker="*", color=RED, s=80, label="минимум без ограничения")
            ax.axhline(0, color=INK, lw=0.55)
            ax.axvline(0, color=INK, lw=0.55)
            ax.set_aspect("equal")
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(y.min(), y.max())
            ax.set_xlabel("$w_1$")
            ax.set_ylabel("$w_2$")
            ax.set_title("L2: гладкая граница" if norm_name == "L2" else "L1: вершина на оси", fontsize=10)
            ax.legend(frameon=False, fontsize=7.4, loc="lower left")
        add_note(axes[-1], "$\\|w\\|_1\\leq R$")
        return
    if variant == 80:  # Bilinear saddle game: rotation, not descent in a bowl.
        ax, ax2 = fig.subplots(1, 2)
        x = np.linspace(-2.1, 2.1, 240)
        y = np.linspace(-2.1, 2.1, 240)
        X, Y = np.meshgrid(x, y)
        Z = X * Y
        levels = [-3, -2, -1.2, -0.6, -0.25, 0.25, 0.6, 1.2, 2, 3]

        def saddle_background(a):
            a.contour(X, Y, Z, levels=levels, colors=BLUE, linewidths=0.75, alpha=0.72)
            a.axhline(0, color=INK, lw=0.65)
            a.axvline(0, color=INK, lw=0.65)
            grid = np.linspace(-1.7, 1.7, 9)
            GX, GY = np.meshgrid(grid, grid)
            U, V = -GY, GX
            norm = np.sqrt(U**2 + V**2) + 1e-9
            a.quiver(GX, GY, U / norm, V / norm, color=GRID, alpha=0.9, scale=25, width=0.004)
            a.scatter([0], [0], color=GOLD, s=48, zorder=5)
            a.set_aspect("equal")
            a.set_xlim(-2.1, 2.1)
            a.set_ylim(-2.1, 2.1)
            a.set_xlabel("$x$ — минимизатор")
            a.set_ylabel("$y$ — максимизатор")

        saddle_background(ax)
        saddle_background(ax2)
        eta = 0.20
        simultaneous = [np.array([0.72, 0.26])]
        extra = [np.array([1.65, 0.60])]
        for _ in range(34):
            x0, y0 = simultaneous[-1]
            simultaneous.append(np.array([x0 - eta * y0, y0 + eta * x0]))
            x0, y0 = extra[-1]
            xh, yh = x0 - eta * y0, y0 + eta * x0
            extra.append(np.array([x0 - eta * yh, y0 + eta * xh]))
        simultaneous = np.asarray(simultaneous)
        extra = np.asarray(extra)
        ax.plot(simultaneous[:, 0], simultaneous[:, 1], "-o", color=RED, ms=2.5, lw=1.4)
        ax.set_title("Одновременный шаг раскручивает", fontsize=10)
        ax.text(-1.96, 1.78, "$r_{t+1}^2=(1+\\eta^2)r_t^2$", fontsize=8.5, color=RED)
        ax2.plot(extra[:, 0], extra[:, 1], "-o", color=GREEN, ms=2.5, lw=1.4)
        ax2.set_title("Экстраградиент смотрит вперёд", fontsize=10)
        ax2.text(-1.96, 1.78, "$f(x,y)=xy$: седло", fontsize=8.5, color=INK)
        add_note(ax2, "$z_{t+1}=z_t-\\eta F(z_t-\\eta F(z_t))$")
        return
    ax, ax2 = fig.subplots(1, 2)
    x = np.linspace(-2.2, 2.2, 150)
    y = np.linspace(-2.2, 2.2, 150)
    X, Y = np.meshgrid(x, y)
    angle = 0.2 + 0.13 * (variant % 5)
    U = np.cos(angle) * X + np.sin(angle) * Y
    V = -np.sin(angle) * X + np.cos(angle) * Y
    Z = 0.35 * U**2 + (1.2 + 0.35 * (variant % 3)) * V**2
    cs = ax.contour(X, Y, Z, levels=9, colors=BLUE, linewidths=0.9, alpha=0.75)
    ax.clabel(cs, fontsize=7, inline=True)
    t = np.linspace(-1.8, 1.8, 100)
    constraint = 0.65 * np.sin(1.2 * t + 0.3 * variant)
    ax.plot(t, constraint, color=RED, lw=2.3, label=labels[0])
    p = 0.55
    q = 0.65 * np.sin(1.2 * p + 0.3 * variant)
    ax.scatter([p], [q], color=GOLD, s=55, zorder=4)
    ax.annotate("", xy=(p + 0.55, q + 0.2), xytext=(p, q), arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 2})
    clean_axes(ax, xlab, ylab)
    ax.legend(frameon=False, fontsize=8)
    # Trajectories
    for i in range(3):
        pts = [np.array([1.8 - 0.25 * i, 1.65 - 0.2 * i])]
        for _ in range(20):
            a, b = pts[-1]
            grad = np.array([0.9 * a + 0.35 * b, 3.1 * b + 0.35 * a])
            noise = rng.normal(0, 0.08 + 0.03 * i, 2)
            pts.append(pts[-1] - (0.08 - 0.012 * i) * grad + noise * (i > 0))
        pts = np.array(pts)
        ax2.plot(pts[:, 0], pts[:, 1], "-o", ms=2.3, color=PALETTE[i], label=labels[min(i, len(labels)-1)])
    ax2.contour(X, Y, 0.5 * X**2 + 2 * Y**2, levels=8, colors=GRID, linewidths=0.8)
    clean_axes(ax2, xlab, ylab)
    ax2.set_title("Траектории")
    ax2.legend(frameon=False, fontsize=8)
    add_note(ax2, note)


def draw_matrix(fig, rng, xlab, ylab, labels, note, variant):
    if variant == 82:  # One convolution window, kernel and scalar product.
        axes = fig.subplots(1, 3)
        fragment = np.array([[2, 1, 0], [3, 2, 1], [0, 1, 2]], dtype=float)
        kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=float)
        product = fragment * kernel
        arrays = [fragment, kernel, product]
        titles = ["Фрагмент $X_{i:i+3,j:j+3}$", "Ядро $K$", "Покомпонентное произведение"]
        cmaps = ["Blues", "RdBu_r", "RdBu_r"]
        limits = [(0, 3), (-3, 3), (-3, 3)]
        for ax, data, title, cmap, (vmin, vmax) in zip(axes, arrays, titles, cmaps, limits):
            ax.pcolormesh(data, cmap=cmap, vmin=vmin, vmax=vmax, edgecolors=PAPER, linewidth=1.5, shading="nearest")
            ax.invert_yaxis()
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=9.5)
            ax.set_xticks([])
            ax.set_yticks([])
            for r in range(3):
                for c in range(3):
                    ax.text(c, r, f"{data[r,c]:.0f}", ha="center", va="center", fontsize=10, color=INK)
        axes[2].text(0.5, -0.13, f"$z_{{ij}}=\\sum K_{{uv}}X_{{i+u,j+v}}={product.sum():.0f}$", transform=axes[2].transAxes, ha="center", fontsize=9)
        # Multiplication and summation arrows join the three panels.
        axes[0].text(1.08, 0.5, "$\\odot$", transform=axes[0].transAxes, fontsize=18, ha="center", va="center")
        axes[1].text(1.08, 0.5, "$\\longrightarrow\\sum$", transform=axes[1].transAxes, fontsize=12, ha="center", va="center")
        return
    if variant == 94:  # Counts, recall-normalized and precision-normalized views.
        axes = fig.subplots(1, 3)
        counts = np.array([[42, 3, 1, 4], [5, 31, 6, 2], [1, 5, 36, 3], [4, 1, 5, 39]], dtype=float)
        row = counts / counts.sum(axis=1, keepdims=True)
        col = counts / counts.sum(axis=0, keepdims=True)
        arrays = [counts / counts.max(), row, col]
        text_arrays = [counts, row, col]
        titles = [
            "Количество ошибок",
            "По строкам: recall\n$\\sum_j p(j\\mid i)=1$",
            "По столбцам: precision\n$\\sum_i p(i\\mid j)=1$",
        ]
        for ax, data, text_data, title in zip(axes, arrays, text_arrays, titles):
            edges = np.arange(5)
            ax.pcolormesh(edges, edges, data, cmap="Blues", vmin=0, vmax=1, edgecolors=PAPER, linewidth=1.2, shading="flat")
            ax.set_xlim(0, 4)
            ax.set_ylim(4, 0)
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=9.5)
            ax.set_xticks(np.arange(4) + 0.5, ["0", "1", "2", "3"], fontsize=7)
            ax.set_yticks(np.arange(4) + 0.5, ["0", "1", "2", "3"], fontsize=7)
            for r in range(4):
                for c in range(4):
                    value = f"{text_data[r,c]:.0f}" if title.startswith("Количество") else f"{text_data[r,c]:.2f}"
                    ax.text(c + 0.5, r + 0.5, value, ha="center", va="center", fontsize=7.2, color=INK if data[r,c] < 0.62 else PAPER)
        axes[0].set_ylabel("истинный класс")
        axes[1].set_xlabel("предсказанный класс")
        return
    if variant == 95:  # A gallery grouped by mechanisms, not three arbitrary heatmaps.
        ax = fig.subplots()
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 6)
        ax.axis("off")

        groups = [
            ("похожая форма", BLUE, [("4", "9", 0.54, 0.38), ("3", "8", 0.49, 0.42)]),
            ("обрезан край", RED, [("7", "1", 0.68, 0.24), ("5", "6", 0.52, 0.40)]),
            ("разорван штрих", GREEN, [("4", "1", 0.47, 0.44), ("8", "3", 0.58, 0.31)]),
        ]

        # Each glyph is a compact vector drawing. Its deformation suggests why
        # two visually different images can land in the same confusion cell.
        glyphs = {
            "1": [((0.50, 0.12), (0.50, 0.88)), ((0.32, 0.72), (0.50, 0.88))],
            "3": [((0.28, 0.82), (0.66, 0.84)), ((0.66, 0.84), (0.72, 0.55)),
                  ((0.72, 0.55), (0.43, 0.50)), ((0.43, 0.50), (0.72, 0.45)),
                  ((0.72, 0.45), (0.65, 0.16)), ((0.65, 0.16), (0.25, 0.18))],
            "4": [((0.68, 0.10), (0.68, 0.90)), ((0.68, 0.78), (0.22, 0.36)),
                  ((0.22, 0.36), (0.82, 0.36))],
            "5": [((0.75, 0.84), (0.30, 0.84)), ((0.30, 0.84), (0.27, 0.50)),
                  ((0.27, 0.50), (0.65, 0.52)), ((0.65, 0.52), (0.73, 0.18)),
                  ((0.73, 0.18), (0.27, 0.16))],
            "7": [((0.20, 0.84), (0.80, 0.84)), ((0.80, 0.84), (0.42, 0.12))],
            "8": [((0.50, 0.50), (0.28, 0.72)), ((0.28, 0.72), (0.50, 0.88)),
                  ((0.50, 0.88), (0.72, 0.72)), ((0.72, 0.72), (0.50, 0.50)),
                  ((0.50, 0.50), (0.25, 0.27)), ((0.25, 0.27), (0.50, 0.10)),
                  ((0.50, 0.10), (0.75, 0.27)), ((0.75, 0.27), (0.50, 0.50))],
            "9": [((0.50, 0.48), (0.28, 0.68)), ((0.28, 0.68), (0.48, 0.88)),
                  ((0.48, 0.88), (0.73, 0.72)), ((0.73, 0.72), (0.50, 0.48)),
                  ((0.73, 0.72), (0.60, 0.12))],
            "6": [((0.72, 0.82), (0.40, 0.65)), ((0.40, 0.65), (0.25, 0.30)),
                  ((0.25, 0.30), (0.45, 0.12)), ((0.45, 0.12), (0.72, 0.28)),
                  ((0.72, 0.28), (0.55, 0.52)), ((0.55, 0.52), (0.28, 0.42))],
        }

        for group_index, (group_title, color, examples) in enumerate(groups):
            gx = 0.25 + 4.0 * group_index
            ax.text(gx + 1.75, 5.45, group_title, ha="center", va="center",
                    fontsize=10.5, color=color)
            ax.plot([gx + 0.15, gx + 0.15, gx + 3.35, gx + 3.35],
                    [5.12, 5.24, 5.24, 5.12], color=color, lw=1.5)
            for row_index, (true_label, pred_label, p_pred, p_true) in enumerate(examples):
                y0 = 3.05 - 1.85 * row_index
                card = FancyBboxPatch(
                    (gx + 0.15, y0), 3.2, 1.45,
                    boxstyle="round,pad=0.025,rounding_size=0.07",
                    fc="#f7f5ee", ec=GRID, lw=1.0,
                )
                ax.add_patch(card)

                # Draw the true-label glyph in a small image frame.
                ax.add_patch(FancyBboxPatch(
                    (gx + 0.34, y0 + 0.19), 0.86, 1.07,
                    boxstyle="round,pad=0.02,rounding_size=0.04",
                    fc=PAPER, ec=GRID, lw=0.8,
                ))
                for (x1, y1), (x2, y2) in glyphs[true_label]:
                    # The middle group is clipped; the last group has a small gap.
                    if group_index == 1 and max(y1, y2) > 0.80:
                        continue
                    if group_index == 2 and 0.46 < (y1 + y2) / 2 < 0.56:
                        continue
                    ax.plot(
                        [gx + 0.34 + 0.86 * x1, gx + 0.34 + 0.86 * x2],
                        [y0 + 0.19 + 1.07 * y1, y0 + 0.19 + 1.07 * y2],
                        color=INK, lw=2.0, solid_capstyle="round",
                    )

                ax.text(gx + 1.38, y0 + 1.07,
                        f"$y={true_label}\\;\\to\\;\\hat y={pred_label}$",
                        fontsize=9.2, color=INK)
                bar_x, bar_w = gx + 1.40, 1.55
                ax.add_patch(FancyBboxPatch(
                    (bar_x, y0 + 0.58), bar_w, 0.18,
                    boxstyle="round,pad=0.0,rounding_size=0.04",
                    fc="#e7e6df", ec="none",
                ))
                ax.add_patch(FancyBboxPatch(
                    (bar_x, y0 + 0.58), bar_w * p_pred, 0.18,
                    boxstyle="round,pad=0.0,rounding_size=0.04",
                    fc=color, ec="none",
                ))
                ax.add_patch(FancyBboxPatch(
                    (bar_x, y0 + 0.28), bar_w, 0.18,
                    boxstyle="round,pad=0.0,rounding_size=0.04",
                    fc="#e7e6df", ec="none",
                ))
                ax.add_patch(FancyBboxPatch(
                    (bar_x, y0 + 0.28), bar_w * p_true, 0.18,
                    boxstyle="round,pad=0.0,rounding_size=0.04",
                    fc=MUTED, ec="none",
                ))
                ax.text(bar_x + bar_w + 0.10, y0 + 0.67,
                        f"$p_{pred_label}={p_pred:.2f}$", va="center", fontsize=7.8, color=color)
                ax.text(bar_x + bar_w + 0.10, y0 + 0.37,
                        f"$p_{true_label}={p_true:.2f}$", va="center", fontsize=7.8, color=MUTED)

        ax.text(6, 0.35,
                "Одна клетка матрицы $y\\to\\hat y$ может скрывать разные причины",
                ha="center", fontsize=9.5, color=INK)
        return
    if variant == 103:  # A satellite-like tile, probability field and hard mask.
        axes = fig.subplots(1, 3)
        n = 28
        yy, xx = np.mgrid[0:n, 0:n]
        tile = np.zeros((n, n), dtype=int)
        tile[:] = 1  # vegetation
        tile[(yy > 17) & (xx < 12)] = 2  # bare ground
        tile[(np.abs(yy - (0.42 * xx + 4)) < 1.3)] = 3  # road
        buildings = [
            (slice(4, 10), slice(5, 12)),
            (slice(11, 17), slice(15, 23)),
            (slice(19, 25), slice(16, 25)),
            (slice(4, 8), slice(18, 24)),
        ]
        target = np.zeros((n, n), dtype=float)
        for rs, cs in buildings:
            target[rs, cs] = 1
            tile[rs, cs] = 4
        probability = 0.06 + 0.78 * target
        # Soften boundaries and add a plausible false-positive roof response.
        for _ in range(3):
            probability = (
                4 * probability
                + np.roll(probability, 1, 0)
                + np.roll(probability, -1, 0)
                + np.roll(probability, 1, 1)
                + np.roll(probability, -1, 1)
            ) / 8
        probability += 0.28 * np.exp(-((xx - 8) ** 2 + (yy - 21) ** 2) / 12)
        probability = np.clip(probability, 0, 1)
        prediction = (probability >= 0.5).astype(float)
        land_cmap = mpl.colors.ListedColormap(["#8fc1d4", "#7fa16b", "#b89a6c", "#d8d1bc", "#c86d57"])
        axes[0].pcolormesh(tile, cmap=land_cmap, vmin=0, vmax=4, shading="nearest")
        axes[0].contour(target, levels=[0.5], colors=PAPER, linewidths=1.1)
        axes[0].set_title("RGB-тайл (условные цвета)", fontsize=9.5)
        prob_mesh = axes[1].pcolormesh(probability, cmap="magma", vmin=0, vmax=1, shading="nearest")
        axes[1].contour(target, levels=[0.5], colors=CYAN, linewidths=1.0)
        axes[1].set_title("$p(\\mathrm{здание})$", fontsize=9.5)
        mask_cmap = mpl.colors.ListedColormap(["#ece9df", RED])
        axes[2].pcolormesh(prediction, cmap=mask_cmap, vmin=0, vmax=1, shading="nearest")
        axes[2].contour(target, levels=[0.5], colors=BLUE, linewidths=1.2)
        axes[2].set_title("argmax; синий — target", fontsize=9.5)
        for ax in axes:
            ax.invert_yaxis()
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        cbar = fig.colorbar(prob_mesh, ax=axes[1], fraction=0.047, pad=0.03)
        cbar.solids.set_rasterized(False)
        cbar.ax.tick_params(labelsize=7)
        axes[2].text(0.5, -0.12, "$H\\times W\\times C\\;\\to\\;H\\times W$", transform=axes[2].transAxes, ha="center", fontsize=8.5)
        return
    if variant == 117:  # Scree curve and progressively richer face reconstructions.
        axes = fig.subplots(1, 4, gridspec_kw={"width_ratios": [1.45, 1, 1, 1]})
        components = np.arange(1, 101)
        eigenvalues = 1.25 * np.exp(-components / 13.0) + 0.12 * np.exp(-components / 45.0)
        explained = np.cumsum(eigenvalues) / eigenvalues.sum()
        axes[0].plot(components, eigenvalues, color=BLUE, lw=2, label="$\\lambda_j$")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("номер компоненты $j$")
        axes[0].set_ylabel("$\\lambda_j$, log", color=BLUE)
        axes[0].grid(True, color=GRID, lw=0.5)
        twin = axes[0].twinx()
        twin.plot(components, explained, color=GREEN, lw=1.8, label="$R_m$")
        twin.set_ylabel("накопленная доля", color=GREEN)
        twin.set_ylim(0, 1.05)
        for m, color in zip([5, 20, 80], [GOLD, RED, VIOLET]):
            axes[0].axvline(m, color=color, ls="--", lw=1)
            twin.scatter([m], [explained[m - 1]], color=color, s=28, zorder=5)
            axes[0].text(m, eigenvalues[m - 1] * 1.35, f"$m={m}$", rotation=90, ha="center", va="bottom", fontsize=7, color=color)
        axes[0].set_title("Спектр и объяснённая дисперсия", fontsize=9.5)

        yy, xx = np.mgrid[-1:1:30j, -0.82:0.82:25j]

        def gaussian(cx, cy, sx, sy):
            return np.exp(-((xx - cx) ** 2 / sx**2 + (yy - cy) ** 2 / sy**2))

        head = np.exp(-((xx / 0.72) ** 8 + (yy / 0.94) ** 8))
        face = 0.19 + 0.62 * head
        face -= 0.27 * gaussian(-0.28, -0.22, 0.14, 0.09)
        face -= 0.25 * gaussian(0.28, -0.22, 0.14, 0.09)
        face -= 0.09 * gaussian(0.0, 0.08, 0.10, 0.25)
        face -= 0.17 * np.exp(-((yy - 0.43 - 0.14 * xx**2) / 0.055) ** 2 - (xx / 0.42) ** 8)
        face -= 0.22 * np.exp(-((yy + 0.78) / 0.13) ** 2) * head
        face = np.clip(face, 0, 1)

        def smooth(image, rounds):
            result = image.copy()
            for _ in range(rounds):
                result = (
                    4 * result
                    + np.roll(result, 1, 0)
                    + np.roll(result, -1, 0)
                    + np.roll(result, 1, 1)
                    + np.roll(result, -1, 1)
                ) / 8
            return result

        reconstructions = [smooth(face, 8), smooth(face, 3), smooth(face, 0)]
        residual_limits = max(float(np.max(np.abs(face - r))) for r in reconstructions)
        for ax, recon, m, color in zip(axes[1:], reconstructions, [5, 20, 80], [GOLD, RED, VIOLET]):
            x_edges = np.arange(recon.shape[1] + 1)
            y_edges = np.arange(recon.shape[0] + 1)
            ax.pcolormesh(x_edges, y_edges, recon, cmap="gray", vmin=0, vmax=1, shading="flat")
            if m < 80:
                ax.contour(
                    np.arange(recon.shape[1]) + 0.5,
                    np.arange(recon.shape[0]) + 0.5,
                    np.abs(face - recon),
                    levels=[0.12 * residual_limits, 0.35 * residual_limits],
                    colors=[color, RED],
                    linewidths=[0.55, 0.8],
                )
            ax.set_xlim(0, recon.shape[1])
            ax.set_ylim(recon.shape[0], 0)
            ax.set_aspect("equal")
            percentage = 100 * explained[m - 1]
            ax.set_title(f"$m={m}$\n$R_m={percentage:.0f}\\%$", fontsize=9.5, color=color)
            ax.set_xticks([])
            ax.set_yticks([])
        return
    count = 3
    axes = fig.subplots(1, count)
    n = 6 if variant % 2 else 5
    base = rng.normal(0, 0.5, (n, n))
    base += np.eye(n) * (1.8 + 0.1 * (variant % 4))
    for i, ax in enumerate(axes):
        if i == 0:
            data = base
        elif i == 1:
            data = np.maximum(base, 0)
        else:
            data = np.roll(base, shift=1 + variant % 2, axis=1) * 0.75
        mesh = ax.pcolormesh(data, cmap="RdBu_r", vmin=-2.5, vmax=2.5, edgecolors=PAPER, linewidth=0.6)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(labels[i] if i < len(labels) else f"вариант {i+1}", fontsize=10)
        ax.set_xlabel(xlab if i == 1 else "")
        ax.set_ylabel(ylab if i == 0 else "")
        ax.set_xticks([])
        ax.set_yticks([])
        if n <= 5:
            for r in range(n):
                for c in range(n):
                    ax.text(c + 0.5, r + 0.5, f"{data[r,c]:.1f}", ha="center", va="center", fontsize=6.5)
    cbar = fig.colorbar(mesh, ax=axes, fraction=0.022, pad=0.02)
    cbar.solids.set_rasterized(False)
    cbar.ax.tick_params(labelsize=7)
    note_text = f"${note}$" if "$" not in note and "\\" in note else note
    fig.text(
        0.52, 0.145, note_text,
        ha="center", va="center", fontsize=8.5, color=INK,
        bbox={"boxstyle": "round,pad=.28", "fc": PAPER, "ec": GRID, "lw": 0.7},
    )


def draw_pipeline(fig, rng, xlab, ylab, labels, note, variant):
    if variant == 55:  # A forward pass whose tensor shapes remain visible.
        ax = fig.subplots()
        ax.set_axis_off()
        stages = [
            (0.075, "$X$", "$5\\times5$", BLUE),
            (0.255, "$x=\\mathrm{vec}(X)$", "$25$", CYAN),
            (0.465, "$h=ReLU(Wx+b)$", "$4$", GREEN),
            (0.675, "$s=Vh+c$", "$2$", GOLD),
            (0.875, "$softmax(s)$", "$2$", RED),
        ]
        for x, label, shape, color in stages:
            ax.add_patch(
                FancyBboxPatch(
                    (x - 0.068, 0.40),
                    0.136,
                    0.22,
                    boxstyle="round,pad=.014",
                    fc="#f7f3e8",
                    ec=color,
                    lw=1.8,
                )
            )
            ax.text(x, 0.515, label, ha="center", va="center", fontsize=9.4)
            ax.text(x, 0.675, f"форма {shape}", ha="center", fontsize=8.2, color=color)
        arrow_labels = ["vectorize", "$W:4\\times25$", "$V:2\\times4$", "нормировка"]
        for i, label in enumerate(arrow_labels):
            x0, x1 = stages[i][0] + 0.071, stages[i + 1][0] - 0.071
            ax.add_patch(
                FancyArrowPatch(
                    (x0, 0.51),
                    (x1, 0.51),
                    arrowstyle="-|>",
                    mutation_scale=11,
                    color=INK,
                    lw=1.1,
                )
            )
            ax.text((x0 + x1) / 2, 0.565, label, ha="center", va="bottom", fontsize=7.8, color=MUTED)
        # Follow one input coordinate through W and V without turning the
        # diagram into a full wiring chart.
        ax.scatter([stages[0][0]], [0.45], s=28, color=RED, zorder=5)
        ax.plot([stages[0][0], stages[2][0]], [0.45, 0.35], color=RED, lw=1.1, alpha=0.65)
        ax.plot([stages[2][0], stages[3][0]], [0.35, 0.35], color=RED, lw=1.1, alpha=0.65)
        ax.text(0.37, 0.285, "вклад одного пикселя проходит через 4 скрытых узла", ha="center", fontsize=8, color=RED)
        # Keep the displayed equation below the title zone: this long lesson
        # title otherwise collides with a generic top-centred note.
        ax.text(0.5, 0.82, "$s=V\\,ReLU(Wx+b)+c$", ha="center", va="center", fontsize=11)
        ax.text(0.5, 0.12, "Форма — проверяемое утверждение о каждом промежуточном объекте.", ha="center", fontsize=9, color=MUTED)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return
    ax = fig.subplots()
    ax.set_axis_off()
    n = len(labels)
    xs = np.linspace(0.07, 0.91, n)
    widths = np.linspace(0.12, 0.085, n)
    heights = 0.18 + 0.05 * np.sin(np.arange(n) + variant)
    for i, (x, w, h, label) in enumerate(zip(xs, widths, heights, labels)):
        y = 0.55 - h / 2
        ax.add_patch(FancyBboxPatch((x - w / 2, y), w, h, boxstyle="round,pad=.012", fc="#f6f2e7", ec=PALETTE[i % 6], lw=1.8))
        ax.text(x, 0.55, textwrap.fill(label, 12), ha="center", va="center", fontsize=9)
        ax.text(x, y - 0.07, f"{max(2, 32//(i+1))}×{max(2, 32//(i+1))}", ha="center", fontsize=7.5, color=MUTED)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + w / 2, 0.55), (xs[i+1] - widths[i+1]/2, 0.55), arrowstyle="-|>", mutation_scale=12, color=INK, lw=1.2))
    values = np.cumsum(1 + (np.arange(n) + variant) % 3)
    ax.plot(xs, 0.16 + 0.025 * values, color=GREEN, marker="o", ms=4)
    ax.text(0.04, 0.88, xlab, fontsize=10, color=MUTED)
    ax.text(0.96, 0.88, ylab, fontsize=10, color=MUTED, ha="right")
    ax.text(0.5, 0.91, note, ha="center", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def draw_geometry(fig, rng, xlab, ylab, labels, note, variant):
    if variant == 65:  # Chord and supporting tangent for one convex graph.
        ax, ax2 = fig.subplots(1, 2)
        xx = np.linspace(-2.25, 2.25, 600)
        f = lambda z: 0.48 * z**2 + 0.12 * z
        df = lambda z: 0.96 * z + 0.12
        x0, x1, lam = -1.45, 1.55, 0.38
        z = lam * x0 + (1 - lam) * x1
        chord_z = lam * f(x0) + (1 - lam) * f(x1)
        chord = f(x0) + (f(x1) - f(x0)) * (xx - x0) / (x1 - x0)
        ax.plot(xx, f(xx), color=BLUE, lw=2.3, label="$f(x)$")
        ax.plot(xx, chord, color=RED, lw=1.7, label="хорда")
        ax.scatter([x0, x1, z], [f(x0), f(x1), f(z)], color=[RED, RED, GOLD], s=35, zorder=5)
        ax.vlines(z, f(z), chord_z, color=GOLD, lw=3)
        ax.text(z + 0.06, (f(z) + chord_z) / 2, "зазор", color=GOLD, va="center", fontsize=8)
        ax.annotate("$x$", (x0, f(x0)), xytext=(-12, 9), textcoords="offset points")
        ax.annotate("$y$", (x1, f(x1)), xytext=(5, 8), textcoords="offset points")
        ax.annotate("$z=\\lambda x+(1-\\lambda)y$", (z, f(z)), xytext=(-30, -24), textcoords="offset points", fontsize=8)
        tangent = f(x0) + df(x0) * (xx - x0)
        ax2.plot(xx, f(xx), color=BLUE, lw=2.3, label="$f(x)$")
        ax2.plot(xx, tangent, color=GREEN, lw=1.8, label="касательная в $x$")
        ax2.scatter([x0, x1], [f(x0), f(x1)], color=[GREEN, GOLD], s=35, zorder=5)
        tangent_y = f(x0) + df(x0) * (x1 - x0)
        ax2.vlines(x1, tangent_y, f(x1), color=GOLD, lw=3)
        ax2.text(x1 - 0.08, (tangent_y + f(x1)) / 2, "$f(y)-T_x(y)$", ha="right", fontsize=8, color=GOLD)
        ax.set_title("Определение через хорду")
        ax2.set_title("Опорная касательная")
        for a in (ax, ax2):
            a.axhline(0, color=INK, lw=0.7)
            a.set_xlabel("$x$")
            a.set_ylabel("$f(x)$")
            a.set_xlim(xx.min(), xx.max())
            a.set_ylim(-0.55, 3.1)
            a.grid(True, color=GRID, lw=0.55)
            a.legend(frameon=False, fontsize=8, loc="upper center")
        add_note(ax2, "$f(y)\\geq f(x)+f'(x)(y-x)$")
        return
    if variant == 67:  # Surface, tangent plane and the approximation error.
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2)
        a = np.linspace(-1.8, 1.8, 70)
        b = np.linspace(-1.8, 1.8, 70)
        A, B = np.meshgrid(a, b)
        L = 0.34 * A**2 + 0.72 * B**2 + 0.22 * A * B + 0.16 * A - 0.10 * B
        a0, b0 = 0.62, -0.48
        l0 = 0.34 * a0**2 + 0.72 * b0**2 + 0.22 * a0 * b0 + 0.16 * a0 - 0.10 * b0
        ga = 0.68 * a0 + 0.22 * b0 + 0.16
        gb = 1.44 * b0 + 0.22 * a0 - 0.10
        tangent = l0 + ga * (A - a0) + gb * (B - b0)
        ax.plot_surface(A, B, L, cmap="Blues", alpha=0.68, linewidth=0, antialiased=True)
        mask = (np.abs(A - a0) < 0.72) & (np.abs(B - b0) < 0.72)
        tangent_local = np.where(mask, tangent, np.nan)
        ax.plot_surface(A, B, tangent_local, color=GOLD, alpha=0.45, linewidth=0)
        ax.scatter([a0], [b0], [l0], color=RED, s=38)
        step = np.array([-ga, -gb])
        step /= np.linalg.norm(step)
        ax.quiver(a0, b0, l0, step[0], step[1], 0, length=0.78, color=GREEN, arrow_length_ratio=0.18, lw=2)
        ax.text(a0 + 0.2 * step[0], b0 + 0.2 * step[1], l0 + 0.16, "$-\\nabla L$", color=GREEN, fontsize=9)
        ax.set_xlabel("$a$")
        ax.set_ylabel("$b$")
        ax.set_zlabel("$L(a,b)$")
        ax.set_title("Поверхность и касательная плоскость", fontsize=10)
        error = np.abs(L - tangent)
        mesh = ax2.pcolormesh(A, B, error, shading="auto", cmap="YlOrRd")
        levels = ax2.contour(A, B, error, levels=[0.05, 0.2, 0.5, 1.0], colors=INK, linewidths=0.55)
        ax2.clabel(levels, fontsize=7)
        ax2.scatter([a0], [b0], color=BLUE, s=35, label="$\\theta_0$")
        ax2.arrow(a0, b0, 0.65 * step[0], 0.65 * step[1], color=GREEN, width=0.018, head_width=0.12, length_includes_head=True)
        ax2.set_aspect("equal")
        ax2.set_xlabel("$a$")
        ax2.set_ylabel("$b$")
        ax2.set_title("$|L-\\widehat L_{lin}|$: цена дальнего шага", fontsize=10)
        ax2.legend(frameon=False, fontsize=8)
        cbar = fig.colorbar(mesh, ax=ax2, fraction=0.048, pad=0.035)
        cbar.solids.set_rasterized(False)
        cbar.ax.tick_params(labelsize=7)
        add_note(ax2, "$\\widehat L=L_0+\\nabla L_0^T\\Delta$")
        return
    if variant == 71:  # Complementary slackness: inactive versus active.
        axes = fig.subplots(1, 2)
        theta = np.linspace(0, 2 * np.pi, 400)
        gx, gy = np.cos(theta), np.sin(theta)
        cases = [
            (np.array([0.30, 0.22]), np.array([0.30, 0.22]), "Минимум внутри", "$g(x)<0,\\;\\lambda=0$"),
            (np.array([1.62, 0.76]), None, "Минимум на границе", "$g(x)=0,\\;\\lambda>0$"),
        ]
        xx = np.linspace(-1.4, 1.9, 180)
        yy = np.linspace(-1.4, 1.5, 180)
        X, Y = np.meshgrid(xx, yy)
        for ax, (center, optimum, title, formula) in zip(axes, cases):
            if optimum is None:
                optimum = center / np.linalg.norm(center)
            Z = (X - center[0]) ** 2 + 1.2 * (Y - center[1]) ** 2
            ax.contour(X, Y, Z, levels=np.linspace(0.08, 2.2, 8), colors=BLUE, linewidths=0.75)
            ax.fill(gx, gy, color=GREEN, alpha=0.10)
            ax.plot(gx, gy, color=GREEN, lw=2, label="$g(x)\\leq0$")
            ax.scatter(*optimum, color=RED, s=48, zorder=5)
            if title.endswith("границе"):
                grad = 2 * (optimum - center)
                normal = optimum
                ax.arrow(optimum[0], optimum[1], 0.45 * grad[0], 0.45 * grad[1], color=RED, width=0.012, head_width=0.09, length_includes_head=True)
                ax.arrow(optimum[0], optimum[1], 0.55 * normal[0], 0.55 * normal[1], color=GOLD, width=0.012, head_width=0.09, length_includes_head=True)
                ax.text(optimum[0] + 0.28, optimum[1] - 0.25, "$\\nabla f+\\lambda\\nabla g=0$", fontsize=8)
            else:
                ax.text(optimum[0] + 0.10, optimum[1] + 0.10, "$\\nabla f=0$", fontsize=8, color=RED)
            ax.set_title(f"{title}\n{formula}", fontsize=10)
            ax.set_aspect("equal")
            ax.set_xlim(-1.35, 1.85)
            ax.set_ylim(-1.35, 1.45)
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            ax.grid(True, color=GRID, lw=0.45)
        add_note(axes[-1], "$\\lambda g(x)=0$")
        return
    if variant == 79:  # Ternary simplex and best-response regions for RPS.
        ax = fig.subplots()
        h = math.sqrt(3) / 2
        vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, h]])
        coords, response = [], []
        steps = 70
        for i in range(steps + 1):
            for j in range(steps + 1 - i):
                p_k = i / steps
                p_n = j / steps
                p_b = 1 - p_k - p_n
                xy = p_k * vertices[0] + p_n * vertices[1] + p_b * vertices[2]
                payoff = [p_b - p_n, p_k - p_b, p_n - p_k]
                coords.append(xy)
                response.append(int(np.argmax(payoff)))
        coords = np.asarray(coords)
        response = np.asarray(response)
        region_colors = [BLUE, RED, GREEN]
        for k, (color, name) in enumerate(zip(region_colors, ["лучший ответ: камень", "лучший ответ: ножницы", "лучший ответ: бумага"])):
            pts = coords[response == k]
            ax.scatter(pts[:, 0], pts[:, 1], s=11, color=color, alpha=0.23, edgecolors="none", label=name)
        outline = np.vstack([vertices, vertices[0]])
        ax.plot(outline[:, 0], outline[:, 1], color=INK, lw=1.8)
        center = vertices.mean(axis=0)
        ax.scatter([center[0]], [center[1]], color=GOLD, edgecolor=INK, lw=0.7, s=70, zorder=5)
        ax.text(center[0], center[1] + 0.045, "$(1/3,1/3,1/3)$", ha="center", fontsize=9)
        ax.text(-0.025, -0.04, "$p_K=1$", ha="right", va="top", color=BLUE)
        ax.text(1.025, -0.04, "$p_N=1$", ha="left", va="top", color=RED)
        ax.text(0.5, h + 0.045, "$p_B=1$", ha="center", va="bottom", color=GREEN)
        ax.text(0.5, -0.12, "$p_K+p_N+p_B=1$", ha="center", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(-0.18, 1.18)
        ax.set_ylim(-0.18, h + 0.16)
        ax.set_axis_off()
        ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(0.78, 0.42))
        return
    if variant == 107:  # A 3D cloud, its 2D code and orthogonal residuals.
        ax = fig.add_subplot(1, 3, 1, projection="3d")
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        n = 34
        uv = rng.normal(0, 0.85, (n, 2))
        # Orthonormal basis of a plane in R^3.
        v1 = np.array([1.0, 0.0, 0.52])
        v1 /= np.linalg.norm(v1)
        raw_v2 = np.array([0.15, 1.0, -0.36])
        v2 = raw_v2 - raw_v2 @ v1 * v1
        v2 /= np.linalg.norm(v2)
        normal = np.cross(v1, v2)
        mean = np.array([0.35, -0.15, 0.45])
        projected = mean + uv[:, :1] * v1 + uv[:, 1:] * v2
        residual_size = rng.normal(0, 0.24, n)
        points = projected + residual_size[:, None] * normal
        colors = plt.cm.viridis(np.linspace(0.05, 0.95, n))
        plane_t = np.linspace(-1.8, 1.8, 12)
        U, V = np.meshgrid(plane_t, plane_t)
        plane = mean[:, None, None] + v1[:, None, None] * U + v2[:, None, None] * V
        ax.plot_surface(plane[0], plane[1], plane[2], color=GOLD, alpha=0.18, linewidth=0)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=22, depthshade=False)
        for i in range(0, n, 4):
            ax.plot(
                [points[i, 0], projected[i, 0]],
                [points[i, 1], projected[i, 1]],
                [points[i, 2], projected[i, 2]],
                color=RED,
                lw=0.8,
                alpha=0.75,
            )
        ax.set_title("Данные около плоскости", fontsize=10)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("$x_3$")
        ax2.scatter(uv[:, 0], uv[:, 1], c=colors, s=24)
        ax2.axhline(0, color=INK, lw=0.6)
        ax2.axvline(0, color=INK, lw=0.6)
        ax2.set_aspect("equal")
        ax2.set_title("Код $(z_1,z_2)$", fontsize=10)
        ax2.set_xlabel("$z_1$")
        ax2.set_ylabel("$z_2$")
        ax2.grid(True, color=GRID, lw=0.45)
        ax3.plot_surface(plane[0], plane[1], plane[2], color=GOLD, alpha=0.16, linewidth=0)
        ax3.scatter(projected[:, 0], projected[:, 1], projected[:, 2], c=colors, s=22, depthshade=False)
        for i in range(0, n, 4):
            ax3.plot(
                [projected[i, 0], points[i, 0]],
                [projected[i, 1], points[i, 1]],
                [projected[i, 2], points[i, 2]],
                color=RED,
                lw=1.0,
            )
        ax3.set_title("Реконструкция и остаток", fontsize=10)
        ax3.set_xlabel("$\\hat x_1$")
        ax3.set_ylabel("$\\hat x_2$")
        ax3.set_zlabel("$\\hat x_3$")
        ax3.text2D(0.04, 0.04, "$\\|x-\\hat x\\|$", transform=ax3.transAxes, fontsize=9, color=RED)
        return
    if variant == 109:  # A linear map is fixed by the two column images.
        axes = fig.subplots(1, 3)
        A = np.array([[1.25, 0.48], [0.35, 1.05]])
        coeff = np.array([0.82, 0.58])
        e1, e2 = np.eye(2)
        a1, a2 = A[:, 0], A[:, 1]
        original = coeff[0] * e1 + coeff[1] * e2
        image = A @ coeff
        # Source basis and decomposition.
        ax = axes[0]
        ax.arrow(0, 0, 1, 0, color=BLUE, width=0.012, head_width=0.09, length_includes_head=True)
        ax.arrow(0, 0, 0, 1, color=RED, width=0.012, head_width=0.09, length_includes_head=True)
        ax.arrow(0, 0, original[0], original[1], color=INK, width=0.010, head_width=0.09, length_includes_head=True)
        ax.plot([coeff[0], coeff[0], 0], [0, coeff[1], coeff[1]], color=GOLD, ls="--", lw=1)
        ax.text(1.02, 0, "$e_1$", color=BLUE)
        ax.text(0, 1.04, "$e_2$", color=RED)
        ax.text(original[0] + 0.04, original[1], "$x$", color=INK)
        ax.set_title("$x=x_1e_1+x_2e_2$", fontsize=10)
        # Images of the basis columns.
        ax = axes[1]
        ax.arrow(0, 0, a1[0], a1[1], color=BLUE, width=0.012, head_width=0.09, length_includes_head=True)
        ax.arrow(0, 0, a2[0], a2[1], color=RED, width=0.012, head_width=0.09, length_includes_head=True)
        ax.text(a1[0] + 0.03, a1[1], "$Ae_1=a_1$", color=BLUE)
        ax.text(a2[0], a2[1] + 0.06, "$Ae_2=a_2$", color=RED)
        ax.set_title("Столбцы матрицы", fontsize=10)
        # The complete transformed grid and the same coefficients.
        ax = axes[2]
        grid = np.linspace(-1.2, 1.2, 9)
        for g in grid:
            q = A @ np.vstack([np.full(80, g), np.linspace(-1.2, 1.2, 80)])
            ax.plot(q[0], q[1], color=GRID, lw=0.65)
            q = A @ np.vstack([np.linspace(-1.2, 1.2, 80), np.full(80, g)])
            ax.plot(q[0], q[1], color=GRID, lw=0.65)
        polygon = np.array([[0, 0], coeff[0] * a1, image, coeff[1] * a2])
        ax.add_patch(Polygon(polygon, closed=True, fc=GOLD, ec=GOLD, alpha=0.22))
        ax.arrow(0, 0, image[0], image[1], color=INK, width=0.010, head_width=0.09, length_includes_head=True)
        ax.text(image[0] + 0.03, image[1], "$Ax=x_1a_1+x_2a_2$", fontsize=8)
        ax.set_title("Линейная комбинация", fontsize=10)
        for ax in axes:
            ax.axhline(0, color=INK, lw=0.55)
            ax.axvline(0, color=INK, lw=0.55)
            ax.set_aspect("equal")
            ax.set_xlim(-1.55, 2.05)
            ax.set_ylim(-1.35, 2.05)
            ax.grid(False)
        return
    if variant == 110:  # Exact determinant examples: area and orientation.
        axes = fig.subplots(1, 3)
        matrices = [
            np.array([[2.0, 0.35], [0.0, 1.0]]),
            np.array([[1.0, 0.35], [0.0, -1.0]]),
            np.array([[1.0, 1.0], [0.80, 0.8005]]),
        ]
        titles = ["$\\det A=2$", "$\\det A=-1$", "$\\det A=0.0005$"]
        for ax, A, title in zip(axes, matrices, titles):
            a1, a2 = A[:, 0], A[:, 1]
            poly = np.array([[0, 0], a1, a1 + a2, a2])
            ax.add_patch(Polygon(poly, closed=True, fc=GOLD, ec=INK, lw=1.3, alpha=0.28))
            ax.arrow(0, 0, a1[0], a1[1], color=BLUE, width=0.012, head_width=0.09, length_includes_head=True)
            ax.arrow(0, 0, a2[0], a2[1], color=RED, width=0.012, head_width=0.09, length_includes_head=True)
            order = np.vstack([poly, poly[0]])
            ax.plot(order[:, 0], order[:, 1], color=INK, lw=1.2)
            ax.annotate("", xy=tuple(order[2]), xytext=tuple(order[1]), arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 1.5})
            det = np.linalg.det(A)
            ax.text(0.04, 0.06, f"площадь = {abs(det):.4g}", transform=ax.transAxes, fontsize=8.5)
            ax.text(0.04, 0.91, "ориентация " + ("+" if det > 0 else "−"), transform=ax.transAxes, fontsize=8.5, color=GREEN if det > 0 else RED)
            ax.set_title(title, fontsize=10)
            ax.axhline(0, color=INK, lw=0.55)
            ax.axvline(0, color=INK, lw=0.55)
            ax.set_aspect("equal")
            ax.set_xlim(-0.35, 2.55)
            ax.set_ylim(-1.45, 1.55)
        add_note(axes[-1], "$|\\det A|=\\mathrm{area}(Ae_1,Ae_2)$")
        return
    if variant == 111:  # The same letter F under AB and BA.
        axes = fig.subplots(2, 3)
        angle = np.deg2rad(34)
        A = np.array([[1.45, 0.0], [0.0, 0.68]])
        B = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        segments = [
            np.array([[0.0, 0.0], [0.0, 1.85]]).T,
            np.array([[0.0, 1.85], [1.05, 1.85]]).T,
            np.array([[0.0, 1.05], [0.78, 1.05]]).T,
        ]

        def draw_transformed_f(ax, M, title):
            grid = np.linspace(-0.5, 2.0, 7)
            for g in grid:
                q = M @ np.vstack([np.full(40, g), np.linspace(-0.5, 2.0, 40)])
                ax.plot(q[0], q[1], color=GRID, lw=0.45)
                q = M @ np.vstack([np.linspace(-0.5, 2.0, 40), np.full(40, g)])
                ax.plot(q[0], q[1], color=GRID, lw=0.45)
            for k, seg in enumerate(segments):
                q = M @ seg
                ax.plot(q[0], q[1], color=[BLUE, RED, GREEN][k], lw=3.2, solid_capstyle="round")
            ax.axhline(0, color=INK, lw=0.45)
            ax.axvline(0, color=INK, lw=0.45)
            ax.set_aspect("equal")
            ax.set_xlim(-1.7, 2.7)
            ax.set_ylim(-0.8, 3.0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=9.5)

        draw_transformed_f(axes[0, 0], np.eye(2), "исходная $F$")
        draw_transformed_f(axes[0, 1], B, "сначала $B$")
        draw_transformed_f(axes[0, 2], A @ B, "итог $AB$")
        draw_transformed_f(axes[1, 0], np.eye(2), "исходная $F$")
        draw_transformed_f(axes[1, 1], A, "сначала $A$")
        draw_transformed_f(axes[1, 2], B @ A, "итог $BA$")
        axes[0, 0].set_ylabel("$x\\mapsto ABx$", fontsize=9)
        axes[1, 0].set_ylabel("$x\\mapsto BAx$", fontsize=9)
        axes[0, 2].text(0.5, -0.16, "$AB\\ne BA$", transform=axes[0, 2].transAxes, ha="center", fontsize=9, color=RED)
        return
    if variant == 115:  # The same points before and after centering.
        ax, ax2 = fig.subplots(1, 2)
        cov = np.array([[1.8, 0.92], [0.92, 0.72]])
        centered = rng.multivariate_normal([0, 0], cov, 58)
        mu = np.array([3.6, 2.35])
        original = centered + mu
        point_colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(original)))
        ax.scatter(original[:, 0], original[:, 1], c=point_colors, s=25)
        ax.arrow(0, 0, mu[0], mu[1], color=RED, width=0.035, head_width=0.18, length_includes_head=True)
        ax.scatter([mu[0]], [mu[1]], color=RED, s=52, zorder=5)
        ax.text(mu[0] + 0.12, mu[1] + 0.06, "$\\mu$", color=RED)
        # The uncentered second-moment direction is pulled toward the mean.
        second = original.T @ original / len(original)
        eigvals0, eigvecs0 = np.linalg.eigh(second)
        u0 = eigvecs0[:, -1]
        ax.plot([-0.4 * u0[0], 6.0 * u0[0]], [-0.4 * u0[1], 6.0 * u0[1]], color=GOLD, lw=2, label="ось второго момента")
        cov_emp = centered.T @ centered / len(centered)
        eigvals, eigvecs = np.linalg.eigh(cov_emp)
        u = eigvecs[:, -1]
        ax2.scatter(centered[:, 0], centered[:, 1], c=point_colors, s=25)
        ax2.plot([-3.2 * u[0], 3.2 * u[0]], [-3.2 * u[1], 3.2 * u[1]], color=GREEN, lw=2.2, label="главная ось")
        ax2.scatter([0], [0], color=RED, s=45, zorder=5)
        ax.set_title("До: разброс смешан со средним", fontsize=10)
        ax2.set_title("После: $x_i-\\mu$", fontsize=10)
        for a in (ax, ax2):
            a.axhline(0, color=INK, lw=0.6)
            a.axvline(0, color=INK, lw=0.6)
            a.set_aspect("equal")
            a.set_xlabel("$x_1$")
            a.set_ylabel("$x_2$")
            a.grid(True, color=GRID, lw=0.45)
            a.legend(frameon=False, fontsize=8)
        add_note(ax2, "$X_c=X-1\\mu^T$")
        return
    if variant == 119:  # Orthogonal rotation keeps dot products and neighbours.
        ax, ax2 = fig.subplots(1, 2)
        users = np.array([[-1.45, 0.65], [-0.55, 1.25], [0.55, 1.05], [1.35, 0.28], [-0.15, -1.25]])
        movies = np.array([[-1.15, 0.22], [-0.20, 0.78], [0.95, 0.72], [1.15, -0.42], [-0.55, -0.92], [0.35, -0.45]])
        angle = np.deg2rad(41)
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        pairs = []
        for i, u in enumerate(users):
            j = int(np.argmax(movies @ u))
            pairs.append((i, j, float(movies[j] @ u)))

        def draw_embedding(ax, U, M, title, xname, yname):
            for i, j, score in pairs:
                ax.plot([U[i, 0], M[j, 0]], [U[i, 1], M[j, 1]], color=GRID, lw=1.0, zorder=1)
                mid = (U[i] + M[j]) / 2
                ax.text(mid[0], mid[1], f"{score:.1f}", fontsize=6.5, color=MUTED)
            ax.scatter(U[:, 0], U[:, 1], color=BLUE, s=52, label="пользователи", zorder=3)
            ax.scatter(M[:, 0], M[:, 1], color=RED, marker="s", s=48, label="фильмы", zorder=3)
            for i, p in enumerate(U):
                ax.text(p[0] + 0.06, p[1] + 0.05, f"$u_{i+1}$", fontsize=7, color=BLUE)
            for i, p in enumerate(M):
                ax.text(p[0] + 0.06, p[1] - 0.10, f"$q_{i+1}$", fontsize=7, color=RED)
            ax.axhline(0, color=INK, lw=0.55)
            ax.axvline(0, color=INK, lw=0.55)
            ax.set_aspect("equal")
            ax.set_xlim(-2.05, 2.05)
            ax.set_ylim(-2.0, 2.0)
            ax.set_xlabel(xname)
            ax.set_ylabel(yname)
            ax.set_title(title, fontsize=10)
            ax.grid(True, color=GRID, lw=0.45)

        draw_embedding(ax, users, movies, "До поворота", "$z_1$", "$z_2$")
        draw_embedding(ax2, users @ R.T, movies @ R.T, "После одного $R$", "$z'_1$", "$z'_2$")
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        add_note(ax2, "$(Ru)^T(Rq)=u^Tq$")
        return
    axes = fig.subplots(1, 3)
    matrices = [
        np.array([[1.25, 0.25], [0.2, 0.85]]),
        np.array([[0.8, -0.7], [0.6, 0.8]]),
        np.array([[1.2, 0.95], [0.15, 0.45]]),
    ]
    if variant % 3 == 1:
        matrices[2] = np.array([[1.0, 0.95], [1.0, 0.951]])
    grid = np.linspace(-1, 1, 7)
    for i, (ax, A) in enumerate(zip(axes, matrices)):
        for g in grid:
            pts = np.vstack([np.full(80, g), np.linspace(-1, 1, 80)])
            q = A @ pts
            ax.plot(q[0], q[1], color=GRID, lw=0.7)
            pts = np.vstack([np.linspace(-1, 1, 80), np.full(80, g)])
            q = A @ pts
            ax.plot(q[0], q[1], color=GRID, lw=0.7)
        cols = A @ np.eye(2)
        ax.arrow(0, 0, cols[0, 0], cols[1, 0], color=BLUE, width=0.008, head_width=0.08, length_includes_head=True)
        ax.arrow(0, 0, cols[0, 1], cols[1, 1], color=RED, width=0.008, head_width=0.08, length_includes_head=True)
        poly = Polygon(np.array([[0,0], cols[:,0], cols[:,0]+cols[:,1], cols[:,1]]), closed=True, fc=GOLD, ec=GOLD, alpha=0.16)
        ax.add_patch(poly)
        ax.axhline(0, color=INK, lw=0.6)
        ax.axvline(0, color=INK, lw=0.6)
        ax.set_aspect("equal")
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.set_title(labels[i] if i < len(labels) else f"$A_{i+1}$", fontsize=10)
        ax.set_xlabel(xlab if i == 1 else "")
        ax.set_ylabel(ylab if i == 0 else "")
        ax.text(0.04, 0.04, f"det={np.linalg.det(A):.2f}", transform=ax.transAxes, fontsize=8, color=MUTED)
    add_note(axes[-1], note)


def draw_bar(fig, rng, xlab, ylab, labels, note, variant):
    ax, ax2 = fig.subplots(1, 2)
    x = np.arange(4)
    vals = np.abs(rng.normal(0.55, 0.24, (min(3, len(labels)), 4)))
    width = 0.22
    for i in range(vals.shape[0]):
        ax.bar(x + (i - 1) * width, vals[i], width, color=PALETTE[i], label=labels[i], alpha=0.88)
    ax.set_xticks(x, [f"$h_{i+1}$" for i in x])
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    probs = np.clip(rng.beta(2, 2, 10), 0.03, 0.97)
    truth = (rng.random(10) < probs).astype(int)
    ax2.plot(probs, "o-", color=BLUE, label="вероятность")
    ax2.scatter(np.arange(10), truth, color=RED, marker="s", label="ответ")
    ax2.axhline(0.5, color=INK, ls="--", lw=1)
    ax2.set_xlabel("объект")
    ax2.set_ylabel("0 … 1")
    ax2.legend(frameon=False, fontsize=8)
    ax2.grid(True, color=GRID, lw=0.6)
    add_note(ax2, note)


def draw_timeline(fig, rng, xlab, ylab, labels, note, variant):
    if variant == 114:  # Two recursion trees: eight versus seven subproducts.
        axes = fig.subplots(1, 3, gridspec_kw={"width_ratios": [1.05, 1.05, 0.9]})

        def draw_tree(ax, branching, color, title):
            ax.set_axis_off()
            root = (0.5, 0.88)
            ax.scatter(*root, s=78, color=INK, zorder=4)
            ax.text(root[0], root[1] + 0.07, "$n$", ha="center", fontsize=9)
            level1 = np.linspace(0.06, 0.94, branching)
            for x in level1:
                ax.plot([root[0], x], [root[1], 0.58], color=color, lw=0.8, alpha=0.8)
                ax.scatter([x], [0.58], s=18, color=color, zorder=3)
            # Draw all second-level leaves: 64 or 49 dots, grouped under
            # their actual parent. Their density is the point of the picture.
            for parent_x in level1:
                width = 0.82 / branching
                children = np.linspace(parent_x - 0.42 * width, parent_x + 0.42 * width, branching)
                for child_x in children:
                    ax.plot([parent_x, child_x], [0.58, 0.29], color=GRID, lw=0.42)
                ax.scatter(children, np.full(branching, 0.29), s=5.5, color=color, alpha=0.85)
            ax.text(0.5, 0.49, f"{branching} задач размера $n/2$", ha="center", fontsize=7.7)
            ax.text(0.5, 0.17, f"${branching}^2={branching**2}$ задач размера $n/4$", ha="center", fontsize=7.7)
            exponent = math.log(branching, 2)
            ax.text(0.5, 0.06, f"$T(n)\\sim n^{{\\log_2 {branching}}}=n^{{{exponent:.3g}}}$", ha="center", fontsize=9, color=color)
            ax.set_title(title, fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        draw_tree(axes[0], 8, BLUE, "Обычное умножение")
        draw_tree(axes[1], 7, RED, "Метод Штрассена")
        n = 2 ** np.arange(2, 12)
        classical = n**3
        strassen_core = n ** math.log(7, 2)
        # Addition overhead makes the asymptotic gain irrelevant on small n.
        strassen_total = 2.8 * strassen_core + 18 * n**2
        axes[2].loglog(n, classical, "-o", color=BLUE, ms=3, label="$n^3$")
        axes[2].loglog(n, strassen_total, "-s", color=RED, ms=3, label="$2.8n^{2.807}+18n^2$")
        axes[2].fill_between(n, classical, strassen_total, where=strassen_total > classical, color=GOLD, alpha=0.14)
        axes[2].set_xlabel("размер $n$")
        axes[2].set_ylabel("условные операции")
        axes[2].set_title("Добавочные сложения", fontsize=10)
        axes[2].grid(True, which="both", color=GRID, lw=0.5)
        axes[2].legend(frameon=False, fontsize=7.5)
        add_note(axes[2], "$\\log_2 7\\approx2.807$")
        return
    ax, ax2 = fig.subplots(2, 1, gridspec_kw={"height_ratios": [0.9, 1.1]})
    x = np.arange(len(labels))
    ax.axhline(0, color=INK, lw=1.2)
    for i, label in enumerate(labels):
        ax.scatter(i, 0, s=70, color=PALETTE[i % 6], zorder=3)
        ax.vlines(i, 0, 0.28 if i % 2 == 0 else -0.28, color=PALETTE[i % 6], lw=1.5)
        ax.text(i, 0.35 if i % 2 == 0 else -0.35, textwrap.fill(label, 16), ha="center", va="center", fontsize=8.5)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(-0.62, 0.62)
    ax.set_axis_off()
    steps = np.arange(1, 13)
    a = 1.0 + 0.55 * np.log2(steps)
    b = 1.0 + 0.38 * steps
    ax2.plot(steps, a, color=BLUE, marker="o", ms=3, label=labels[0])
    ax2.plot(steps, b, color=RED, marker="s", ms=3, label=labels[-1])
    ax2.fill_between(steps, a, b, color=GOLD, alpha=0.13)
    ax2.set_xlabel(xlab)
    ax2.set_ylabel(ylab)
    ax2.legend(frameon=False, fontsize=8)
    ax2.grid(True, color=GRID, lw=0.6)
    add_note(ax2, note)


def draw_spectrum(fig, rng, xlab, ylab, labels, note, variant):
    axes = fig.subplots(2, 2)
    fs = 256
    t = np.arange(fs) / fs
    f1 = 12 + variant % 9
    f2 = 31 + 2 * (variant % 7)
    sig = np.sin(2 * np.pi * f1 * t) + 0.55 * np.sin(2 * np.pi * f2 * t)
    sig += 0.06 * rng.normal(size=t.size)
    axes[0, 0].plot(t, sig, color=BLUE, lw=1.2)
    axes[0, 0].set_title(labels[0] if labels else "сигнал")
    axes[0, 0].set_xlabel("время")
    axes[0, 0].set_ylabel("амплитуда")
    spec = np.abs(np.fft.rfft(sig))
    freq = np.fft.rfftfreq(sig.size, 1 / fs)
    axes[0, 1].plot(freq, spec, color=RED, lw=1.5)
    axes[0, 1].axvline(f1, color=BLUE, ls="--", lw=1)
    axes[0, 1].axvline(f2, color=GREEN, ls="--", lw=1)
    axes[0, 1].set_xlim(0, 80)
    axes[0, 1].set_title(labels[1] if len(labels) > 1 else "спектр")
    axes[0, 1].set_xlabel(xlab)
    axes[0, 1].set_ylabel(ylab)
    # Vector spectrogram via pcolormesh.
    win = 48
    hop = 12
    frames = []
    for start in range(0, sig.size - win + 1, hop):
        frames.append(np.abs(np.fft.rfft(sig[start : start + win] * np.hanning(win))))
    S = np.array(frames).T
    axes[1, 0].pcolormesh(np.arange(S.shape[1]), np.arange(S.shape[0]), 20 * np.log10(S + 1e-3), cmap="magma", shading="nearest")
    axes[1, 0].set_title(labels[2] if len(labels) > 2 else "STFT")
    axes[1, 0].set_xlabel("кадр")
    axes[1, 0].set_ylabel("частотная корзина")
    periods = np.array([2, 3, 4, 6, 8])
    direct = np.abs(np.cos(np.pi / periods))
    blur = direct * np.linspace(0.2, 0.85, len(periods))
    axes[1, 1].plot(periods, direct, "-o", color=RED, label="без фильтра")
    axes[1, 1].plot(periods, blur, "-s", color=GREEN, label="после blur")
    axes[1, 1].set_xlabel("период, px")
    axes[1, 1].set_ylabel("ложная энергия")
    axes[1, 1].legend(frameon=False, fontsize=8)
    for ax in axes.ravel():
        ax.grid(True, color=GRID, lw=0.45, alpha=0.7)
    add_note(axes[1, 1], note)


DRAWERS = {
    "boundary": draw_boundary,
    "network": draw_network,
    "signal": draw_signal,
    "curves": draw_curves,
    "tradeoff": draw_tradeoff,
    "contour": draw_contour,
    "matrix": draw_matrix,
    "pipeline": draw_pipeline,
    "geometry": draw_geometry,
    "bar": draw_bar,
    "timeline": draw_timeline,
    "spectrum": draw_spectrum,
}


def make_contact_sheets(png_paths: list[Path]) -> list[Path]:
    PREVIEW.mkdir(parents=True, exist_ok=True)
    sheets = []
    per_sheet = 12
    thumb_w, thumb_h = 560, 330
    margin, title_h = 18, 36
    for page, start in enumerate(range(0, len(png_paths), per_sheet), 1):
        subset = png_paths[start : start + per_sheet]
        canvas = Image.new("RGB", (4 * thumb_w + 5 * margin, 3 * (thumb_h + title_h) + 4 * margin), PAPER)
        draw = ImageDraw.Draw(canvas)
        for pos, path in enumerate(subset):
            row, col = divmod(pos, 4)
            x = margin + col * (thumb_w + margin)
            y = margin + row * (thumb_h + title_h + margin)
            im = Image.open(path).convert("RGB")
            im.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            canvas.paste(im, (x + (thumb_w - im.width) // 2, y + title_h))
            draw.text((x, y + 8), path.stem.replace("lesson-", "").replace("-figure-", "."), fill=INK)
        out = PREVIEW / f"contact-sheet-{page}.png"
        canvas.save(out, quality=94)
        sheets.append(out)
    return sheets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lessons", default="13-40", help="range such as 13-40")
    parser.add_argument("--no-contact-sheets", action="store_true")
    args = parser.parse_args()
    lo, hi = map(int, args.lessons.split("-"))
    png_paths: list[Path] = []
    rendered = 0
    for lesson in range(lo, hi + 1):
        md_specs = read_markdown_specs(lesson)
        visual_specs = SPECS[lesson]
        if len(visual_specs) != 3:
            raise ValueError(f"lesson {lesson}: bad visual spec count")
        lesson_dir = OUT / f"{lesson:02d}"
        lesson_dir.mkdir(parents=True, exist_ok=True)
        preview_dir = PREVIEW / "png"
        preview_dir.mkdir(parents=True, exist_ok=True)
        for idx, (md, visual) in enumerate(zip(md_specs, visual_specs), 1):
            kind, xlab, ylab, labels, note = visual
            xlab = mathtext_safe(xlab)
            ylab = mathtext_safe(ylab)
            labels = [mathtext_safe(label) for label in labels]
            note = mathtext_safe(note)
            fig = plt.figure(figsize=(9.4, 5.2))
            rng = np.random.default_rng(lesson * 100 + idx)
            DRAWERS[kind](fig, rng, xlab, ylab, labels, note, lesson * 3 + idx)
            finish_figure(fig, md["title"], md["caption"], lesson, idx)
            svg = lesson_dir / f"figure-{idx}.svg"
            png = preview_dir / f"lesson-{lesson:02d}-figure-{idx}.png"
            caption_one_line = re.sub(r"\s+", " ", md["caption"]).strip()
            metadata = {"Title": md["title"], "Description": f'{md["alt"]}. {caption_one_line}'}
            fig.savefig(svg, format="svg", metadata=metadata)
            fig.savefig(png, dpi=120)
            plt.close(fig)
            png_paths.append(png)
            rendered += 1
    sheets = [] if args.no_contact_sheets else make_contact_sheets(png_paths)
    print(f"rendered_svg={rendered}")
    print(f"preview_png={len(png_paths)}")
    print(f"contact_sheets={len(sheets)}")
    for sheet in sheets:
        print(sheet)


if __name__ == "__main__":
    main()
