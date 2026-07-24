"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type SideCalculationProps = {
  kind: string;
  options?: Record<string, string>;
};

const basePoints = [
  { x: 0.08, y: 0 },
  { x: 0.18, y: 0 },
  { x: 0.29, y: 0 },
  { x: 0.41, y: 1 },
  { x: 0.55, y: 0 },
  { x: 0.66, y: 1 },
  { x: 0.78, y: 1 },
  { x: 0.91, y: 1 },
];

function ThresholdShiftCalculation() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [shifted, setShifted] = useState(false);

  const points = useMemo(
    () =>
      basePoints.map((point, index) =>
        shifted && index === 4 ? { ...point, y: 1 } : point,
      ),
    [shifted],
  );

  const mistakes = points.filter((point) => Number(point.x >= threshold) !== point.y);
  const empiricalRisk = mistakes.length / points.length;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(220, Math.round(rect.width));
    const height = 150;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const left = 18;
    const right = width - 14;
    const axisY = 104;
    const x = (value: number) => left + value * (right - left);

    ctx.strokeStyle = "#c9c8bf";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(left, axisY);
    ctx.lineTo(right, axisY);
    ctx.stroke();

    [0, 0.5, 1].forEach((tick) => {
      ctx.beginPath();
      ctx.moveTo(x(tick), axisY - 4);
      ctx.lineTo(x(tick), axisY + 4);
      ctx.stroke();
      ctx.fillStyle = "#6e726a";
      ctx.font = "11px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(tick.toFixed(1), x(tick), axisY + 18);
    });

    ctx.strokeStyle = "#a57920";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x(threshold), 18);
    ctx.lineTo(x(threshold), axisY + 1);
    ctx.stroke();
    ctx.fillStyle = "#7a5a17";
    ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = threshold > 0.76 ? "right" : "left";
    ctx.fillText(`t = ${threshold.toFixed(2)}`, x(threshold) + (threshold > 0.76 ? -4 : 4), 14);

    points.forEach((point) => {
      const predicted = Number(point.x >= threshold);
      const wrong = predicted !== point.y;
      const py = point.y ? 45 : 82;
      ctx.beginPath();
      ctx.arc(x(point.x), py, wrong ? 5.5 : 4.2, 0, Math.PI * 2);
      ctx.fillStyle = point.y ? "#315f8c" : "#b94a3b";
      ctx.fill();
      if (wrong) {
        ctx.strokeStyle = "#171915";
        ctx.lineWidth = 1.4;
        ctx.stroke();
      }
    });

    ctx.fillStyle = "#315f8c";
    ctx.textAlign = "left";
    ctx.fillText("спам", left, 43);
    ctx.fillStyle = "#b94a3b";
    ctx.fillText("обычное", left, 80);
  }, [points, threshold]);

  return (
    <div className="side-calculation">
      <canvas
        ref={canvasRef}
        className="side-calculation__canvas"
        role="img"
        aria-label={`Порог ${threshold.toFixed(2)} даёт ${mistakes.length} ошибок из ${points.length}`}
      />
      <label className="side-calculation__label">
        <span>Порог</span>
        <output>{threshold.toFixed(2)}</output>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={threshold}
          onChange={(event) => setThreshold(Number(event.target.value))}
        />
      </label>
      <button
        type="button"
        className="side-calculation__toggle"
        aria-pressed={shifted}
        onClick={() => setShifted((value) => !value)}
      >
        {shifted ? "Вернуть письмо" : "Заменить одну метку"}
      </button>
      <p aria-live="polite">
        Ошибок: <strong>{mistakes.length}</strong> из {points.length};
        {" "}эмпирический риск <strong>{empiricalRisk.toFixed(3)}</strong>.
      </p>
    </div>
  );
}

function TerminalVelocityCalculation({
  options,
}: {
  options?: Record<string, string>;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mass = Number(options?.mass ?? 80);
  const density = Number(options?.density ?? 1.2);
  const observedVelocity = Number(options?.velocity ?? 5);
  const gravity = 9.81;
  const product = (2 * mass * gravity) / (density * observedVelocity ** 2);
  const [dragCoefficient, setDragCoefficient] = useState(1.2);
  const area = product / dragCoefficient;
  const velocity = Math.sqrt(
    (2 * mass * gravity) / (density * dragCoefficient * area),
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(220, Math.round(rect.width));
    const height = 150;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const left = 34;
    const right = width - 10;
    const top = 15;
    const bottom = 119;
    const x = (value: number) =>
      left + ((value - 0.6) / (1.8 - 0.6)) * (right - left);
    const y = (value: number) =>
      bottom - ((value - 25) / (90 - 25)) * (bottom - top);

    ctx.strokeStyle = "#c9c8bf";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(left, top);
    ctx.lineTo(left, bottom);
    ctx.lineTo(right, bottom);
    ctx.stroke();

    ctx.fillStyle = "#6e726a";
    ctx.font = "10px system-ui, sans-serif";
    ctx.textAlign = "center";
    [0.6, 1.2, 1.8].forEach((tick) => {
      ctx.beginPath();
      ctx.moveTo(x(tick), bottom - 3);
      ctx.lineTo(x(tick), bottom + 3);
      ctx.stroke();
      ctx.fillText(tick.toFixed(1), x(tick), bottom + 15);
    });
    ctx.save();
    ctx.translate(10, (top + bottom) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("A, м²", 0, 0);
    ctx.restore();
    ctx.fillText("C_d", (left + right) / 2, 148);

    ctx.strokeStyle = "#315f8c";
    ctx.lineWidth = 1.7;
    ctx.beginPath();
    for (let index = 0; index <= 120; index += 1) {
      const coefficient = 0.6 + (1.2 * index) / 120;
      const currentArea = product / coefficient;
      if (index === 0) ctx.moveTo(x(coefficient), y(currentArea));
      else ctx.lineTo(x(coefficient), y(currentArea));
    }
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(x(dragCoefficient), y(area), 5, 0, Math.PI * 2);
    ctx.fillStyle = "#b94a3b";
    ctx.fill();
    ctx.strokeStyle = "#fffef9";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }, [area, dragCoefficient, product]);

  return (
    <div className="side-calculation">
      <canvas
        ref={canvasRef}
        className="side-calculation__canvas"
        role="img"
        aria-label={`Коэффициент сопротивления ${dragCoefficient.toFixed(2)}, площадь ${area.toFixed(1)} квадратного метра и скорость ${velocity.toFixed(2)} метра в секунду`}
      />
      <label className="side-calculation__label">
        <span>
          Коэффициент <i>C</i><sub>d</sub>
        </span>
        <output>{dragCoefficient.toFixed(2)}</output>
        <input
          type="range"
          min="0.6"
          max="1.8"
          step="0.01"
          value={dragCoefficient}
          aria-label="Коэффициент аэродинамического сопротивления C d"
          onChange={(event) => setDragCoefficient(Number(event.target.value))}
        />
      </label>
      <p aria-live="polite">
        Площадь <strong>{area.toFixed(1)} м²</strong>; произведение{" "}
        <strong>
          <i>C</i><sub>d</sub>
          <i>A</i> = {product.toFixed(2)} м²
        </strong>
        ; скорость остаётся <strong>{velocity.toFixed(2)} м/с</strong>.
      </p>
    </div>
  );
}

function KappaMatrixCalculation() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [corrected, setCorrected] = useState(0);
  const matrix = useMemo(
    () => [
      [42 + corrected, 8 - corrected, 2],
      [6, 28, 5],
      [1, 7, 31],
    ],
    [corrected],
  );
  const total = 130;
  const observedAgreement =
    (matrix[0][0] + matrix[1][1] + matrix[2][2]) / total;
  const rowTotals = matrix.map((row) =>
    row.reduce((sum, value) => sum + value, 0),
  );
  const columnTotals = matrix[0].map((_, column) =>
    matrix.reduce((sum, row) => sum + row[column], 0),
  );
  const expectedAgreement =
    rowTotals.reduce(
      (sum, rowTotal, index) =>
        sum + rowTotal * columnTotals[index],
      0,
    ) /
    total ** 2;
  const kappa =
    (observedAgreement - expectedAgreement) / (1 - expectedAgreement);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(220, Math.round(rect.width));
    const height = 228;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const size = Math.min(42, (width - 78) / 3);
    const gridWidth = size * 3;
    const left = Math.max(48, (width - gridWidth) / 2 + 8);
    const top = 48;
    const classLabels = ["1", "2", "3"];

    ctx.fillStyle = "#6e726a";
    ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Эксперт B", left + gridWidth / 2, 14);
    classLabels.forEach((label, index) => {
      ctx.fillText(label, left + (index + 0.5) * size, top - 10);
    });
    ctx.save();
    ctx.translate(12, top + gridWidth / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Эксперт A", 0, 0);
    ctx.restore();

    matrix.forEach((row, rowIndex) => {
      ctx.fillStyle = "#6e726a";
      ctx.textAlign = "right";
      ctx.fillText(
        classLabels[rowIndex],
        left - 10,
        top + (rowIndex + 0.5) * size + 4,
      );
      row.forEach((value, columnIndex) => {
        const diagonal = rowIndex === columnIndex;
        const intensity = Math.min(1, value / 44);
        ctx.fillStyle = diagonal
          ? `rgba(49, 95, 140, ${0.10 + 0.42 * intensity})`
          : `rgba(185, 74, 59, ${0.05 + 0.35 * intensity})`;
        ctx.fillRect(
          left + columnIndex * size,
          top + rowIndex * size,
          size,
          size,
        );
        ctx.strokeStyle = "#c9c8bf";
        ctx.lineWidth = 1;
        ctx.strokeRect(
          left + columnIndex * size,
          top + rowIndex * size,
          size,
          size,
        );
        ctx.fillStyle = "#171915";
        ctx.font = "600 13px ui-monospace, monospace";
        ctx.textAlign = "center";
        ctx.fillText(
          String(value),
          left + (columnIndex + 0.5) * size,
          top + (rowIndex + 0.5) * size + 5,
        );
      });
    });

    ctx.fillStyle = "#6e726a";
    ctx.font = "10px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(
      "диагональ: совпадения",
      width / 2,
      top + gridWidth + 19,
    );
    ctx.fillText(
      "вне диагонали: разногласия",
      width / 2,
      top + gridWidth + 33,
    );
  }, [matrix]);

  return (
    <div className="side-calculation">
      <canvas
        ref={canvasRef}
        className="side-calculation__canvas side-calculation__canvas--matrix"
        role="img"
        aria-label={`Матрица согласия трёх классов. Исправлено разногласий: ${corrected}. Наблюдаемое согласие ${observedAgreement.toFixed(3)}, случайно ожидаемое ${expectedAgreement.toFixed(3)}, каппа Коэна ${kappa.toFixed(3)}.`}
      />
      <label className="side-calculation__label">
        <span>Перенести из (1, 2) в (1, 1)</span>
        <output>{corrected} из 8</output>
        <input
          type="range"
          min="0"
          max="8"
          step="1"
          value={corrected}
          aria-label="Число исправленных разногласий между экспертами"
          onChange={(event) => setCorrected(Number(event.target.value))}
        />
      </label>
      <p aria-live="polite">
        <i>p</i><sub>o</sub> = <strong>{observedAgreement.toFixed(3)}</strong>;
        {" "}<i>p</i><sub>e</sub> = <strong>{expectedAgreement.toFixed(3)}</strong>;
        {" "}&kappa; = <strong>{kappa.toFixed(3)}</strong>.
      </p>
    </div>
  );
}

export function SideCalculation({ kind, options }: SideCalculationProps) {
  if (kind === "threshold-shift") {
    return <ThresholdShiftCalculation />;
  }

  if (kind === "terminal-velocity") {
    return <TerminalVelocityCalculation options={options} />;
  }

  if (kind === "kappa-matrix") {
    return <KappaMatrixCalculation />;
  }

  return (
    <p className="side-calculation__fallback">
      Вычислительный пример временно недоступен.
    </p>
  );
}
