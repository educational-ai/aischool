// Lesson 02: validation-led fitting of a polynomial residual over a fixed time grid.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("mechanism-fit-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 1040;
      var H = 560;
      var RIDGE = 1e-9;
      var MASTER_TIMES = Array.from({ length: 31 }, function (_unused, index) {
        return index / 10;
      });
      var defaults = {
        degree: Number(options.degree == null ? 3 : options.degree),
        trainEnd: Number(options.trainEnd == null ? 1.5 : options.trainEnd),
        noiseCm: Number(options.noise == null ? 7 : options.noise),
        cubic: Number(
          options.cubic == null
            ? (options.drag == null ? 0.28 : Number(options.drag) * 0.0048)
            : options.cubic,
        ),
        seed: Number(options.seed == null ? 2026 : options.seed),
      };
      var state = {
        degree: defaults.degree,
        trainEnd: defaults.trainEnd,
        noiseCm: defaults.noiseCm,
        cubic: defaults.cubic,
        seed: defaults.seed,
        oracleOpen: false,
        committedDegree: null,
      };

      K.hint(
        root,
        "Степень выбирают по validation RMSE. Новый режим открой только после того, как степень записана: oracle-метрика не участвует в настройке.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "На узком экране сдвиньте полотно влево, чтобы увидеть панель остатка. Мастер-сетка фиксирована: t=0,0; 0,1; …; 3,0 с. Каждая пятая доступная точка отложена для validation. Генератор: h*=20−4,905t²+βt³, ε=σz; σ дано в сантиметрах. Для точки i код берёт uₛ=frac[43758,5453·sin(91,173(i+1)+17,731(seed+s))] при s=3 и 11, заменяет u₃ на max(10⁻⁷,u₃), затем считает z=√(−2 ln u₃)·cos(2πu₁₁). Oracle сравнивает прогноз с h* вне диапазона и не заменяет закрытый тест.",
      );

      var canvasState;
      var commit = null;
      var trajectoryBox = { x: 66, y: 48, w: 625, h: 350 };
      var residualBox = { x: 752, y: 48, w: 244, h: 350 };

      function clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
      }

      function uniform(index, salt) {
        var x = Math.sin((index + 1) * 91.173 + (state.seed + salt) * 17.731) * 43758.5453;
        return x - Math.floor(x);
      }

      function normalNoise(index) {
        var u1 = Math.max(1e-7, uniform(index, 3));
        var u2 = uniform(index, 11);
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      }

      function trueHeight(t) {
        return 20 - 4.905 * t * t + state.cubic * t * t * t;
      }

      function physicalHeight(t) {
        return 20 - 4.905 * t * t;
      }

      function solve(matrix, vector) {
        var n = matrix.length;
        if (!n) return [];
        var a = matrix.map(function (row, index) {
          return row.slice().concat([vector[index]]);
        });
        for (var col = 0; col < n; col += 1) {
          var pivot = col;
          for (var row = col + 1; row < n; row += 1) {
            if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
          }
          var swap = a[col];
          a[col] = a[pivot];
          a[pivot] = swap;
          var diagonal = Math.abs(a[col][col]) < 1e-12 ? 1e-12 : a[col][col];
          for (var j = col; j <= n; j += 1) a[col][j] /= diagonal;
          for (var i = 0; i < n; i += 1) {
            if (i === col) continue;
            var factor = a[i][col];
            for (var k = col; k <= n; k += 1) a[i][k] -= factor * a[col][k];
          }
        }
        return a.map(function (row) { return row[n]; });
      }

      function inverse(matrix) {
        return matrix.map(function (_unused, column) {
          var unit = matrix.map(function (_row, row) {
            return row === column ? 1 : 0;
          });
          return solve(matrix, unit);
        }).reduce(function (result, column) {
          column.forEach(function (value, row) {
            result[row] = result[row] || [];
            result[row].push(value);
          });
          return result;
        }, []);
      }

      function normInfinity(matrix) {
        if (!matrix.length) return 1;
        return Math.max.apply(null, matrix.map(function (row) {
          return row.reduce(function (sum, value) {
            return sum + Math.abs(value);
          }, 0);
        }));
      }

      function buildSamples() {
        var sigma = state.noiseCm / 100;
        return MASTER_TIMES.map(function (t, index) {
          var eligible = t <= state.trainEnd + 1e-9;
          var role = "new";
          if (eligible) role = index % 5 === 4 ? "validation" : "train";
          return {
            index: index,
            t: t,
            height: trueHeight(t) + sigma * normalNoise(index),
            role: role,
          };
        });
      }

      function fit() {
        var samples = buildSamples();
        var train = samples.filter(function (sample) {
          return sample.role === "train";
        });
        var validation = samples.filter(function (sample) {
          return sample.role === "validation";
        });
        if (state.degree === 0) {
          return {
            samples: samples,
            train: train,
            validation: validation,
            coefficients: [],
            condition: null,
          };
        }
        var matrix = [];
        var target = [];
        for (var row = 0; row < state.degree; row += 1) {
          matrix[row] = [];
          target[row] = 0;
          for (var col = 0; col < state.degree; col += 1) {
            matrix[row][col] = train.reduce(function (sum, sample) {
              return sum + Math.pow(sample.t, row + col + 2);
            }, 0);
          }
          matrix[row][row] += RIDGE;
          target[row] = train.reduce(function (sum, sample) {
            var residual = sample.height - physicalHeight(sample.t);
            return sum + Math.pow(sample.t, row + 1) * residual;
          }, 0);
        }
        var coefficients = solve(matrix, target);
        var condition = normInfinity(matrix) * normInfinity(inverse(matrix));
        return {
          samples: samples,
          train: train,
          validation: validation,
          coefficients: coefficients,
          condition: condition,
        };
      }

      function learnedResidual(t, coefficients) {
        return coefficients.reduce(function (sum, value, index) {
          return sum + value * Math.pow(t, index + 1);
        }, 0);
      }

      function hybridHeight(t, coefficients) {
        return physicalHeight(t) + learnedResidual(t, coefficients);
      }

      function x(box, t) {
        return box.x + t / 3 * box.w;
      }

      function yTrajectory(value) {
        return trajectoryBox.y + trajectoryBox.h - (value + 15) / 37 * trajectoryBox.h;
      }

      function yResidual(value) {
        return residualBox.y + residualBox.h - (value + 2) / 15 * residualBox.h;
      }

      function line(ctx, points, color, width, dash) {
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = width || 2;
        ctx.setLineDash(dash || []);
        ctx.beginPath();
        points.forEach(function (point, index) {
          if (index) ctx.lineTo(point.x, point.y);
          else ctx.moveTo(point.x, point.y);
        });
        ctx.stroke();
        ctx.restore();
      }

      function axes(ctx, box, yValues, yMap, yLabel) {
        ctx.font = "11px system-ui, sans-serif";
        ctx.fillStyle = C.muted;
        ctx.textAlign = "right";
        yValues.forEach(function (value) {
          var py = yMap(value);
          ctx.strokeStyle = C.grid;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(box.x, py);
          ctx.lineTo(box.x + box.w, py);
          ctx.stroke();
          ctx.fillText(String(value), box.x - 9, py + 4);
        });
        [0, 1, 2, 3].forEach(function (value) {
          var px = x(box, value);
          ctx.strokeStyle = C.grid;
          ctx.beginPath();
          ctx.moveTo(px, box.y);
          ctx.lineTo(px, box.y + box.h);
          ctx.stroke();
          ctx.textAlign = "center";
          ctx.fillText(String(value), px, box.y + box.h + 22);
        });
        ctx.strokeStyle = C.axis;
        ctx.strokeRect(box.x, box.y, box.w, box.h);
        ctx.save();
        ctx.translate(box.x - 42, box.y + box.h / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = "center";
        ctx.fillText(yLabel, 0, 0);
        ctx.restore();
      }

      function rmse(values) {
        if (!values.length) return null;
        return Math.sqrt(values.reduce(function (sum, value) {
          return sum + value * value;
        }, 0) / values.length);
      }

      function formatRmse(value) {
        return value == null ? "—" : value.toFixed(3) + " м";
      }

      function coefficientText(coefficients) {
        if (!coefficients.length) return "r(t)=0";
        var parts = coefficients.map(function (value, index) {
          var sign = value >= 0 && index ? "+" : "";
          return sign + value.toFixed(2) + "t" + (index ? "^" + (index + 1) : "");
        });
        return "r(t)=" + parts.join("");
      }

      function invalidateOracle() {
        state.oracleOpen = false;
        state.committedDegree = null;
      }

      function drawSample(ctx, sample, box, yMap, value, radius) {
        var validation = sample.role === "validation";
        ctx.beginPath();
        ctx.arc(x(box, sample.t), yMap(value), radius, 0, Math.PI * 2);
        ctx.fillStyle = validation ? C.paper : C.ink;
        ctx.fill();
        ctx.strokeStyle = validation ? C.gold : C.ink;
        ctx.lineWidth = validation ? 2 : 1;
        ctx.stroke();
      }

      function draw() {
        var fitted = fit();
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = C.paper;
        ctx.fillRect(0, 0, W, H);

        ctx.fillStyle = C.wash;
        ctx.fillRect(
          x(trajectoryBox, state.trainEnd),
          trajectoryBox.y,
          trajectoryBox.x + trajectoryBox.w - x(trajectoryBox, state.trainEnd),
          trajectoryBox.h,
        );
        ctx.fillRect(
          x(residualBox, state.trainEnd),
          residualBox.y,
          residualBox.x + residualBox.w - x(residualBox, state.trainEnd),
          residualBox.h,
        );
        axes(ctx, trajectoryBox, [-10, 0, 10, 20], yTrajectory, "высота, м");
        axes(ctx, residualBox, [0, 5, 10], yResidual, "остаток, м");

        var truthPoints = [];
        var physicsPoints = [];
        var hybridPoints = [];
        var trueResidualPoints = [];
        var learnedResidualPoints = [];
        var oracleErrors = [];
        for (var step = 0; step <= 240; step += 1) {
          var t = 3 * step / 240;
          var truth = trueHeight(t);
          var physics = physicalHeight(t);
          var hybrid = hybridHeight(t, fitted.coefficients);
          truthPoints.push({ x: x(trajectoryBox, t), y: yTrajectory(clamp(truth, -15, 22)) });
          physicsPoints.push({ x: x(trajectoryBox, t), y: yTrajectory(clamp(physics, -15, 22)) });
          hybridPoints.push({ x: x(trajectoryBox, t), y: yTrajectory(clamp(hybrid, -15, 22)) });
          trueResidualPoints.push({ x: x(residualBox, t), y: yResidual(clamp(truth - physics, -2, 13)) });
          learnedResidualPoints.push({ x: x(residualBox, t), y: yResidual(clamp(hybrid - physics, -2, 13)) });
          if (t > state.trainEnd + 1e-9) oracleErrors.push(hybrid - truth);
        }
        line(ctx, physicsPoints, C.blue, 2.0);
        line(ctx, hybridPoints, C.red, 2.35);
        line(ctx, learnedResidualPoints, C.red, 2.2);
        if (state.oracleOpen) {
          line(ctx, truthPoints, C.ink, 1.5, [6, 5]);
          line(ctx, trueResidualPoints, C.ink, 1.3, [6, 5]);
        }

        fitted.samples.forEach(function (sample) {
          if (sample.role === "new") return;
          drawSample(ctx, sample, trajectoryBox, yTrajectory, sample.height, 3.5);
          drawSample(
            ctx,
            sample,
            residualBox,
            yResidual,
            clamp(sample.height - physicalHeight(sample.t), -2, 13),
            3,
          );
        });

        var boundaryX = x(trajectoryBox, state.trainEnd);
        ctx.strokeStyle = C.ink;
        ctx.lineWidth = 1.25;
        ctx.beginPath();
        ctx.moveTo(boundaryX, trajectoryBox.y);
        ctx.lineTo(boundaryX, trajectoryBox.y + trajectoryBox.h);
        ctx.stroke();
        var boundaryResidualX = x(residualBox, state.trainEnd);
        ctx.beginPath();
        ctx.moveTo(boundaryResidualX, residualBox.y);
        ctx.lineTo(boundaryResidualX, residualBox.y + residualBox.h);
        ctx.stroke();

        ctx.font = "11px system-ui, sans-serif";
        ctx.fillStyle = C.muted;
        ctx.textAlign = "left";
        ctx.fillText(
          state.oracleOpen ? "oracle открыт, степень " + state.committedDegree : "новый режим закрыт",
          boundaryX + 8,
          trajectoryBox.y + 18,
        );
        ctx.fillStyle = C.blue;
        ctx.fillText("механика", trajectoryBox.x + 12, trajectoryBox.y + trajectoryBox.h - 53);
        ctx.fillStyle = C.red;
        ctx.fillText("механика + остаток", trajectoryBox.x + 12, trajectoryBox.y + trajectoryBox.h - 35);
        if (state.oracleOpen) {
          ctx.fillStyle = C.ink;
          ctx.fillText("скрытая траектория", trajectoryBox.x + 12, trajectoryBox.y + trajectoryBox.h - 17);
        }
        ctx.fillStyle = C.muted;
        ctx.textAlign = "center";
        ctx.fillText("время, с", trajectoryBox.x + trajectoryBox.w / 2, trajectoryBox.y + trajectoryBox.h + 44);
        ctx.fillText("время, с", residualBox.x + residualBox.w / 2, residualBox.y + residualBox.h + 44);
        ctx.fillStyle = C.ink;
        ctx.font = "600 14px system-ui, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("траектория", trajectoryBox.x, 25);
        ctx.fillText("что учит поправка", residualBox.x, 25);
        ctx.beginPath();
        ctx.arc(trajectoryBox.x + 118, 21, 3.4, 0, Math.PI * 2);
        ctx.fillStyle = C.ink;
        ctx.fill();
        ctx.fillStyle = C.muted;
        ctx.font = "11px system-ui, sans-serif";
        ctx.fillText("подгонка", trajectoryBox.x + 127, 25);
        ctx.beginPath();
        ctx.arc(trajectoryBox.x + 198, 21, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = C.paper;
        ctx.fill();
        ctx.strokeStyle = C.gold;
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = C.muted;
        ctx.fillText("validation", trajectoryBox.x + 207, 25);

        var formula = coefficientText(fitted.coefficients);
        ctx.fillStyle = C.wash;
        ctx.fillRect(66, 472, 930, 56);
        ctx.fillStyle = C.ink;
        ctx.font = "600 13px ui-monospace, monospace";
        ctx.textAlign = "left";
        ctx.fillText(formula.length > 115 ? formula.slice(0, 112) + "…" : formula, 84, 499);
        ctx.fillStyle = C.muted;
        ctx.font = "11px system-ui, sans-serif";
        ctx.fillText("коэффициенты найдены только по чёрным точкам; 10⁻⁹I — явно показанный численный стабилизатор", 84, 518);

        var trainErrors = fitted.train.map(function (sample) {
          return hybridHeight(sample.t, fitted.coefficients) - sample.height;
        });
        var validationErrors = fitted.validation.map(function (sample) {
          return hybridHeight(sample.t, fitted.coefficients) - sample.height;
        });
        output.set([
          { label: "seed", value: String(state.seed) },
          { label: "RMSE train (шумные точки)", value: formatRmse(rmse(trainErrors)), color: C.green },
          { label: "RMSE validation", value: formatRmse(rmse(validationErrors)), color: C.gold },
          {
            label: "RMSE new/oracle",
            value: state.oracleOpen ? formatRmse(rmse(oracleErrors)) : "закрыта",
            color: state.oracleOpen ? C.red : C.muted,
          },
          {
            label: "cond(XᵀX+10⁻⁹I)",
            value: fitted.condition == null
              ? "—"
              : (fitted.condition < 1e5 ? fitted.condition.toFixed(0) : fitted.condition.toExponential(1)),
            color: fitted.condition != null && fitted.condition > 1e7 ? C.red : C.muted,
          },
        ]);
        if (commit) {
          commit.textContent = state.oracleOpen
            ? "Открыто для степени " + state.committedDegree
            : "Зафиксировать степень и открыть";
          commit.setAttribute(
            "aria-label",
            state.oracleOpen
              ? "Новый режим открыт для степени " + state.committedDegree
              : "Зафиксировать текущую степень и открыть oracle-диагностику нового режима",
          );
        }
      }

      canvasState = K.makeCanvas(stage, W, H, {
        maxWidth: W,
        label: "Подгонка полиномиального остатка по train, выбор по validation и открываемая oracle-проверка нового режима",
        onResize: draw,
      });

      var degreeControl = K.segmented(controls, {
        label: "Степень остатка",
        value: String(state.degree),
        options: [0, 1, 2, 3, 5, 7].map(function (value) {
          return { value: String(value), label: String(value) };
        }),
      }, function (value) {
        state.degree = Number(value);
        invalidateOracle();
        draw();
      });
      var trainControl = K.slider(controls, {
        label: "Конец доступного диапазона",
        min: 0.7,
        max: 2.2,
        step: 0.1,
        value: state.trainEnd,
        unit: " с",
        format: function (value) { return value.toFixed(1); },
      }, function (value) {
        state.trainEnd = value;
        invalidateOracle();
        draw();
      });
      var noiseControl = K.slider(controls, {
        label: "σ шума высоты",
        min: 0,
        max: 20,
        step: 1,
        value: state.noiseCm,
        unit: " см",
      }, function (value) {
        state.noiseCm = value;
        invalidateOracle();
        draw();
      });
      var cubicControl = K.slider(controls, {
        label: "Кубический коэффициент β",
        min: 0,
        max: 0.48,
        step: 0.01,
        value: state.cubic,
        unit: " м/с³",
        format: function (value) { return value.toFixed(2); },
      }, function (value) {
        state.cubic = value;
        invalidateOracle();
        draw();
      });

      var validationAction = K.element("div", "kontur-int-control");
      validationAction.appendChild(K.element("span", "kontur-int-label-name", {
        text: "Новый режим",
      }));
      commit = K.element("button", "kontur-int-action", {
        type: "button",
        text: "Зафиксировать степень и открыть",
        "aria-label": "Зафиксировать текущую степень и открыть oracle-диагностику нового режима",
      });
      commit.addEventListener("click", function () {
        state.oracleOpen = true;
        state.committedDegree = state.degree;
        draw();
      });
      validationAction.appendChild(commit);
      controls.appendChild(validationAction);

      var repeatAction = K.element("div", "kontur-int-control");
      repeatAction.appendChild(K.element("span", "kontur-int-label-name", {
        text: "Повтор опыта",
      }));
      var repeat = K.element("button", "kontur-int-action", {
        type: "button",
        text: "Новый шум",
        "aria-label": "Сгенерировать новый шум измерений",
      });
      repeat.addEventListener("click", function () {
        state.seed += 1;
        invalidateOracle();
        draw();
      });
      var reset = K.element("button", "kontur-int-action", {
        type: "button",
        text: "Сбросить: seed 2026",
        "aria-label": "Вернуть исходные параметры и seed 2026",
      });
      reset.addEventListener("click", function () {
        state.degree = defaults.degree;
        state.trainEnd = defaults.trainEnd;
        state.noiseCm = defaults.noiseCm;
        state.cubic = defaults.cubic;
        state.seed = 2026;
        invalidateOracle();
        degreeControl.set(String(state.degree));
        trainControl.set(state.trainEnd);
        noiseControl.set(state.noiseCm);
        cubicControl.set(state.cubic);
        draw();
      });
      repeatAction.appendChild(repeat);
      repeatAction.appendChild(reset);
      controls.appendChild(repeatAction);

      draw();
      return function () {
        canvasState.destroy();
      };
    });
  }

  if (window.KonturInt) install();
  else window.addEventListener("kontur-int-ready", install, { once: true });
})();
