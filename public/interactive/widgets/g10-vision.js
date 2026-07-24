// Grade 10, modules 3–4. Vision and linear-algebra experiments.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;
    var K = window.KonturInt;
    var C = K.COLORS;

    function clamp(value, min, max) {
      return Math.max(min, Math.min(max, value));
    }

    function seeded(index, salt) {
      var value = Math.sin(index * 91.731 + salt * 37.113) * 43758.5453;
      return value - Math.floor(value);
    }

    function rgba(hex, alpha) {
      var value = hex.replace("#", "");
      var r = parseInt(value.slice(0, 2), 16);
      var g = parseInt(value.slice(2, 4), 16);
      var b = parseInt(value.slice(4, 6), 16);
      return "rgba(" + r + "," + g + "," + b + "," + alpha + ")";
    }

    function setup(root, hint, label, caption, height) {
      K.hint(root, hint);
      var stage = K.row(root);
      var redraw = function () {};
      var canvasState = K.makeCanvas(stage, 920, height || 460, {
        maxWidth: 920,
        label: label,
        onResize: function () { redraw(); },
      });
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(root, caption);
      return {
        canvas: canvasState,
        controls: controls,
        output: output,
        setDraw: function (fn) { redraw = fn; },
        destroy: function () { canvasState.destroy(); },
      };
    }

    function line(ctx, points, color, width, dash) {
      if (!points.length) return;
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

    function arrow(ctx, x1, y1, x2, y2, color, width) {
      var angle = Math.atan2(y2 - y1, x2 - x1);
      ctx.save();
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = width || 2;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x2, y2);
      ctx.lineTo(x2 - 9 * Math.cos(angle - 0.45), y2 - 9 * Math.sin(angle - 0.45));
      ctx.lineTo(x2 - 9 * Math.cos(angle + 0.45), y2 - 9 * Math.sin(angle + 0.45));
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }

    function title(ctx, text, x, y) {
      ctx.fillStyle = C.ink;
      ctx.font = "600 15px ET Book, Palatino, Georgia, serif";
      ctx.textAlign = "left";
      ctx.fillText(text, x, y);
    }

    function small(ctx, text, x, y, color, align) {
      ctx.fillStyle = color || C.muted;
      ctx.font = "12px system-ui, sans-serif";
      ctx.textAlign = align || "left";
      ctx.fillText(text, x, y);
    }

    function paintPaper(ctx, height) {
      ctx.clearRect(0, 0, 920, height);
      ctx.fillStyle = C.paper;
      ctx.fillRect(0, 0, 920, height);
    }

    function buildReceptiveField(root) {
      var ui = setup(
        root,
        "Поверни границу и фильтр. Локальный ответ велик, когда их направления совпадают.",
        "Ориентированный край и рецептивное поле",
        "Один локальный фильтр видит маленький фрагмент. Набор фильтров с разными углами превращает край в карту признаков.",
        460,
      );
      var state = { edge: 28, filter: 45, width: 0.12 };
      var image = { x: 60, y: 54, size: 310 };
      var kernel = { x: 427, y: 74, size: 170 };

      function response(angle) {
        var delta = (state.edge - angle) * Math.PI / 180;
        return Math.abs(Math.cos(delta));
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paintPaper(ctx, 460);
        title(ctx, "Локальный фрагмент", image.x, 32);
        var cells = 22;
        var cell = image.size / cells;
        var theta = state.edge * Math.PI / 180;
        for (var row = 0; row < cells; row += 1) {
          for (var col = 0; col < cells; col += 1) {
            var nx = (col + 0.5) / cells - 0.5;
            var ny = (row + 0.5) / cells - 0.5;
            var signed = nx * Math.cos(theta) + ny * Math.sin(theta);
            var level = signed > state.width ? 224 : signed < -state.width ? 65 : 145 + signed / state.width * 79;
            ctx.fillStyle = "rgb(" + level + "," + level + "," + Math.round(level * 0.96) + ")";
            ctx.fillRect(image.x + col * cell, image.y + row * cell, cell + 0.4, cell + 0.4);
          }
        }
        ctx.strokeStyle = C.axis;
        ctx.strokeRect(image.x, image.y, image.size, image.size);

        title(ctx, "Фильтр 7 × 7", kernel.x, 32);
        var kcell = kernel.size / 7;
        var phi = state.filter * Math.PI / 180;
        for (var kr = 0; kr < 7; kr += 1) {
          for (var kc = 0; kc < 7; kc += 1) {
            var kx = kc - 3;
            var ky = kr - 3;
            var projection = kx * Math.cos(phi) + ky * Math.sin(phi);
            var weight = Math.exp(-(kx * kx + ky * ky) / 11) * projection / 3;
            ctx.fillStyle = weight >= 0
              ? rgba(C.red, 0.12 + Math.min(0.78, Math.abs(weight) * 0.8))
              : rgba(C.blue, 0.12 + Math.min(0.78, Math.abs(weight) * 0.8));
            ctx.fillRect(kernel.x + kc * kcell, kernel.y + kr * kcell, kcell - 1, kcell - 1);
          }
        }
        ctx.strokeStyle = C.axis;
        ctx.strokeRect(kernel.x, kernel.y, kernel.size, kernel.size);
        small(ctx, "красный +   синий −", kernel.x, kernel.y + kernel.size + 23);

        var plot = { x: 630, y: 74, w: 235, h: 170 };
        title(ctx, "Ответ банка фильтров", plot.x, 32);
        ctx.fillStyle = C.wash;
        ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
        ctx.strokeStyle = C.grid;
        ctx.strokeRect(plot.x, plot.y, plot.w, plot.h);
        [0, 45, 90, 135, 180].forEach(function (degree) {
          var x = plot.x + degree / 180 * plot.w;
          line(ctx, [{ x: x, y: plot.y }, { x: x, y: plot.y + plot.h }], C.grid, 1);
          small(ctx, degree + "°", x, plot.y + plot.h + 19, C.muted, "center");
        });
        var curve = [];
        for (var degree = 0; degree <= 180; degree += 2) {
          curve.push({
            x: plot.x + degree / 180 * plot.w,
            y: plot.y + plot.h - response(degree) * plot.h,
          });
        }
        line(ctx, curve, C.blue, 2.5);
        var markerX = plot.x + state.filter / 180 * plot.w;
        var markerY = plot.y + plot.h - response(state.filter) * plot.h;
        ctx.beginPath();
        ctx.arc(markerX, markerY, 7, 0, Math.PI * 2);
        ctx.fillStyle = C.red;
        ctx.fill();

        var delta = Math.abs(state.edge - state.filter);
        delta = Math.min(delta, 180 - delta);
        ui.output.set([
          { label: "угол края", value: state.edge + "°" },
          { label: "угол фильтра", value: state.filter + "°" },
          { label: "расхождение", value: delta + "°" },
          { label: "отклик", value: response(state.filter).toFixed(3), color: C.blue },
        ]);
      }

      K.slider(ui.controls, {
        label: "Ориентация края", min: 0, max: 180, step: 1, value: state.edge,
        format: function (v) { return String(v); }, unit: "°",
      }, function (value) { state.edge = value; draw(); });
      K.slider(ui.controls, {
        label: "Ориентация фильтра", min: 0, max: 180, step: 1, value: state.filter,
        format: function (v) { return String(v); }, unit: "°",
      }, function (value) { state.filter = value; draw(); });
      K.slider(ui.controls, {
        label: "Толщина перехода", min: 0.02, max: 0.28, step: 0.01, value: state.width,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.width = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildConvolution(root) {
      var ui = setup(
        root,
        "Перемещай окно по условному спутниковому фрагменту. Справа строится вся карта откликов выбранного ядра.",
        "Свёртка 3 × 3 по изображению",
        "Одни и те же девять весов применяются в каждой позиции. Поэтому признак можно найти независимо от его координат.",
        480,
      );
      var state = { row: 4, col: 5, kernel: "edge" };
      var source = { x: 45, y: 60, size: 330 };
      var map = { x: 565, y: 60, size: 300 };
      var kernels = {
        edge: { label: "вертикальный край", values: [-1, 0, 1, -2, 0, 2, -1, 0, 1] },
        blur: { label: "усреднение", values: [1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9] },
        sharp: { label: "резкость", values: [0, -1, 0, -1, 5, -1, 0, -1, 0] },
      };

      function pixel(row, col) {
        var water = 0.18 + 0.035 * col;
        var field = ((row > 2 && row < 9 && col > 1 && col < 7) ? 0.28 : 0);
        var roof = ((row > 5 && row < 10 && col > 7 && col < 11) ? 0.48 : 0);
        var road = Math.abs(row - (0.58 * col + 1.3)) < 0.8 ? 0.32 : 0;
        return clamp(water + field + roof + road + (seeded(row * 17 + col, 8) - 0.5) * 0.05, 0, 1);
      }

      function conv(row, col) {
        var values = kernels[state.kernel].values;
        var sum = 0;
        for (var dy = 0; dy < 3; dy += 1) {
          for (var dx = 0; dx < 3; dx += 1) {
            sum += pixel(row + dy, col + dx) * values[dy * 3 + dx];
          }
        }
        return sum;
      }

      function responseRange() {
        var values = [];
        for (var row = 0; row < 10; row += 1) {
          for (var col = 0; col < 10; col += 1) values.push(conv(row, col));
        }
        return {
          min: Math.min.apply(null, values),
          max: Math.max.apply(null, values),
        };
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paintPaper(ctx, 480);
        title(ctx, "Фрагмент 12 × 12", source.x, 34);
        var sourceCell = source.size / 12;
        for (var row = 0; row < 12; row += 1) {
          for (var col = 0; col < 12; col += 1) {
            var value = pixel(row, col);
            ctx.fillStyle = "rgb(" + Math.round(36 + value * 165) + "," + Math.round(65 + value * 145) + "," + Math.round(78 + value * 105) + ")";
            ctx.fillRect(source.x + col * sourceCell, source.y + row * sourceCell, sourceCell - 0.7, sourceCell - 0.7);
          }
        }
        ctx.strokeStyle = C.red;
        ctx.lineWidth = 3;
        ctx.strokeRect(
          source.x + state.col * sourceCell,
          source.y + state.row * sourceCell,
          sourceCell * 3,
          sourceCell * 3,
        );

        var kernelX = 415;
        title(ctx, "Ядро", kernelX, 34);
        kernels[state.kernel].values.forEach(function (value, index) {
          var kx = kernelX + (index % 3) * 38;
          var ky = 86 + Math.floor(index / 3) * 38;
          ctx.fillStyle = value < 0 ? rgba(C.blue, 0.16 + Math.min(0.75, Math.abs(value) / 2)) : rgba(C.red, 0.12 + Math.min(0.75, Math.abs(value) / 5));
          ctx.fillRect(kx, ky, 36, 36);
          small(ctx, Number(value.toFixed(2)).toString(), kx + 18, ky + 23, C.ink, "center");
        });
        arrow(ctx, 523, 143, 550, 143, C.axis, 1.5);

        title(ctx, "Карта 10 × 10", map.x, 34);
        var range = responseRange();
        var mapCell = map.size / 10;
        for (var mr = 0; mr < 10; mr += 1) {
          for (var mc = 0; mc < 10; mc += 1) {
            var response = conv(mr, mc);
            var normalized = (response - range.min) / (range.max - range.min || 1);
            var red = Math.round(45 + normalized * 175);
            var blue = Math.round(210 - normalized * 145);
            ctx.fillStyle = "rgb(" + red + "," + Math.round(80 + normalized * 70) + "," + blue + ")";
            ctx.fillRect(map.x + mc * mapCell, map.y + mr * mapCell, mapCell - 0.8, mapCell - 0.8);
          }
        }
        ctx.strokeStyle = C.paper;
        ctx.lineWidth = 3;
        ctx.strokeRect(map.x + state.col * mapCell, map.y + state.row * mapCell, mapCell, mapCell);
        small(ctx, "минимум", map.x, map.y + map.size + 20, C.blue);
        small(ctx, "максимум", map.x + map.size, map.y + map.size + 20, C.red, "right");
        var current = conv(state.row, state.col);
        ui.output.set([
          { label: "позиция", value: "(" + state.row + ", " + state.col + ")" },
          { label: "ядро", value: kernels[state.kernel].label },
          { label: "сумма", value: current.toFixed(3), color: current >= 0 ? C.red : C.blue },
          { label: "размер выхода", value: "10 × 10" },
        ]);
      }

      K.segmented(ui.controls, {
        label: "Ядро", value: state.kernel,
        options: [
          { value: "edge", label: "край" },
          { value: "blur", label: "сглаживание" },
          { value: "sharp", label: "резкость" },
        ],
      }, function (value) { state.kernel = value; draw(); });
      K.slider(ui.controls, {
        label: "Строка окна", min: 0, max: 9, step: 1, value: state.row,
      }, function (value) { state.row = value; draw(); });
      K.slider(ui.controls, {
        label: "Столбец окна", min: 0, max: 9, step: 1, value: state.col,
      }, function (value) { state.col = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function digitValue(row, col, shift) {
      var x = col - shift;
      var y = row;
      var top = y > 1 && y < 4 && x > 2 && x < 9;
      var middle = y > 5 && y < 8 && x > 2 && x < 8;
      var bottom = y > 9 && y < 12 && x > 2 && x < 9;
      var upperLeft = x > 1 && x < 4 && y > 2 && y < 7;
      var lowerRight = x > 7 && x < 10 && y > 6 && y < 11;
      var base = top || middle || bottom || upperLeft || lowerRight ? 0.9 : 0.04;
      return clamp(base + (seeded(row * 19 + col, shift + 3) - 0.5) * 0.12, 0, 1);
    }

    function buildPooling(root) {
      var ui = setup(
        root,
        "Сдвигай цифру на один-два пикселя и сравни max pooling со средним. Выход уменьшается, но детали теряются.",
        "Pooling рукописной цифры",
        "Max pooling сохраняет самый сильный локальный отклик. Average pooling хранит средний уровень и сильнее размывает тонкие штрихи.",
        475,
      );
      var state = { method: "max", shift: 0, block: 2 };
      var input = { x: 52, y: 62, size: 326 };
      var output = { x: 535, y: 62, size: 326 };

      function pooled(row, col, shift) {
        var values = [];
        for (var dy = 0; dy < state.block; dy += 1) {
          for (var dx = 0; dx < state.block; dx += 1) {
            values.push(digitValue(row * state.block + dy, col * state.block + dx, shift));
          }
        }
        if (state.method === "max") return Math.max.apply(null, values);
        return values.reduce(function (sum, value) { return sum + value; }, 0) / values.length;
      }

      function pooledVector(shift) {
        var side = Math.floor(12 / state.block);
        var values = [];
        for (var row = 0; row < side; row += 1) {
          for (var col = 0; col < side; col += 1) values.push(pooled(row, col, shift));
        }
        return values;
      }

      function drawGrid(ctx, area, side, getter) {
        var cell = area.size / side;
        for (var row = 0; row < side; row += 1) {
          for (var col = 0; col < side; col += 1) {
            var value = getter(row, col);
            var level = Math.round(250 - value * 205);
            ctx.fillStyle = "rgb(" + level + "," + level + "," + Math.round(level * 0.96) + ")";
            ctx.fillRect(area.x + col * cell, area.y + row * cell, cell - 1, cell - 1);
          }
        }
        ctx.strokeStyle = C.axis;
        ctx.strokeRect(area.x, area.y, area.size, area.size);
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paintPaper(ctx, 475);
        title(ctx, "Вход 12 × 12", input.x, 34);
        drawGrid(ctx, input, 12, function (row, col) { return digitValue(row, col, state.shift); });
        arrow(ctx, 410, 225, 500, 225, C.axis, 2);
        small(ctx, state.method === "max" ? "максимум" : "среднее", 455, 207, C.muted, "center");
        small(ctx, state.block + " × " + state.block, 455, 246, C.muted, "center");
        var side = Math.floor(12 / state.block);
        title(ctx, "Выход " + side + " × " + side, output.x, 34);
        drawGrid(ctx, output, side, function (row, col) { return pooled(row, col, state.shift); });
        var reference = pooledVector(0);
        var shifted = pooledVector(state.shift);
        var mse = reference.reduce(function (sum, value, index) {
          return sum + Math.pow(value - shifted[index], 2);
        }, 0) / reference.length;
        ui.output.set([
          { label: "операция", value: state.method === "max" ? "max" : "average" },
          { label: "коэффициент сжатия", value: String(state.block * state.block) + "×" },
          { label: "сдвиг", value: state.shift + " px" },
          { label: "изменение выхода", value: mse.toFixed(3), color: C.blue },
        ]);
      }

      K.segmented(ui.controls, {
        label: "Правило", value: state.method,
        options: [{ value: "max", label: "max" }, { value: "avg", label: "average" }],
      }, function (value) { state.method = value; draw(); });
      K.slider(ui.controls, {
        label: "Сдвиг цифры", min: -2, max: 2, step: 1, value: state.shift,
        format: function (v) { return String(v); }, unit: " px",
      }, function (value) { state.shift = value; draw(); });
      K.segmented(ui.controls, {
        label: "Окно", value: "2",
        options: [{ value: "2", label: "2 × 2" }, { value: "3", label: "3 × 3" }],
      }, function (value) { state.block = Number(value); draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildArchitecture(root) {
      var ui = setup(
        root,
        "Меняй глубину, ядро и число каналов. Под каждой картой указаны размер, каналы и рецептивное поле.",
        "Конструктор небольшой CNN",
        "Глубина расширяет рецептивное поле. Число каналов увеличивает число признаков и параметров; пространственный размер здесь сокращается pooling-слоями.",
        470,
      );
      var state = { depth: 4, kernel: 3, channels: 16, skip: false };

      function model() {
        var layers = [];
        var size = 64;
        var inputChannels = 3;
        var receptive = 1;
        var jump = 1;
        var params = 0;
        for (var index = 0; index < state.depth; index += 1) {
          var outChannels = Math.min(128, state.channels * Math.pow(2, Math.floor(index / 2)));
          receptive += (state.kernel - 1) * jump;
          params += state.kernel * state.kernel * inputChannels * outChannels + outChannels;
          layers.push({
            size: size,
            channels: outChannels,
            receptive: receptive,
            params: params,
          });
          inputChannels = outChannels;
          if (index % 2 === 1 && index < state.depth - 1) {
            size /= 2;
            jump *= 2;
            receptive += jump / 2;
          }
        }
        return layers;
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paintPaper(ctx, 470);
        var layers = model();
        title(ctx, "Вход", 55, 34);
        var nodes = [{ x: 75, y: 190, w: 58, h: 150, label: "64² × 3", receptive: 1 }];
        var available = 650;
        layers.forEach(function (layer, index) {
          var x = 165 + index * available / Math.max(1, state.depth - 1);
          var scale = 150 * layer.size / 64;
          var width = 34 + Math.log2(layer.channels) * 5;
          nodes.push({
            x: x,
            y: 265 - scale / 2,
            w: width,
            h: Math.max(42, scale),
            label: layer.size + "² × " + layer.channels,
            receptive: layer.receptive,
          });
        });
        nodes.forEach(function (node, index) {
          if (index > 0) {
            var previous = nodes[index - 1];
            arrow(
              ctx,
              previous.x + previous.w + 7,
              previous.y + previous.h / 2,
              node.x - 10,
              node.y + node.h / 2,
              C.axis,
              1.4,
            );
          }
          ctx.fillStyle = index === 0 ? rgba(C.green, 0.18) : rgba(C.blue, 0.13 + index * 0.035);
          ctx.fillRect(node.x, node.y, node.w, node.h);
          ctx.strokeStyle = index === 0 ? C.green : C.blue;
          ctx.lineWidth = 1.6;
          ctx.strokeRect(node.x, node.y, node.w, node.h);
          small(ctx, node.label, node.x + node.w / 2, 342, C.ink, "center");
          small(ctx, "r = " + node.receptive, node.x + node.w / 2, 363, C.muted, "center");
        });
        if (state.skip && nodes.length > 4) {
          var from = nodes[1];
          var to = nodes[nodes.length - 1];
          ctx.save();
          ctx.strokeStyle = C.gold;
          ctx.lineWidth = 2;
          ctx.setLineDash([6, 4]);
          ctx.beginPath();
          ctx.moveTo(from.x + from.w / 2, from.y - 8);
          ctx.bezierCurveTo(from.x + 70, 64, to.x - 70, 64, to.x + to.w / 2, to.y - 8);
          ctx.stroke();
          ctx.restore();
          small(ctx, "skip connection", (from.x + to.x) / 2, 76, C.gold, "center");
        }
        small(ctx, "размер × каналы", 55, 415);
        small(ctx, "r — сторона рецептивного поля", 55, 438);
        var last = layers[layers.length - 1];
        ui.output.set([
          { label: "свёрточных слоёв", value: String(state.depth) },
          { label: "параметров", value: last.params.toLocaleString("ru-RU"), color: C.blue },
          { label: "последнее поле", value: last.receptive + " × " + last.receptive },
          { label: "выход", value: last.size + "² × " + last.channels },
        ]);
      }

      K.slider(ui.controls, {
        label: "Глубина", min: 1, max: 6, step: 1, value: state.depth,
      }, function (value) { state.depth = value; draw(); });
      K.segmented(ui.controls, {
        label: "Ядро", value: "3",
        options: [{ value: "3", label: "3 × 3" }, { value: "5", label: "5 × 5" }],
      }, function (value) { state.kernel = Number(value); draw(); });
      K.segmented(ui.controls, {
        label: "Стартовые каналы", value: "16",
        options: [
          { value: "8", label: "8" },
          { value: "16", label: "16" },
          { value: "32", label: "32" },
        ],
      }, function (value) { state.channels = Number(value); draw(); });
      K.segmented(ui.controls, {
        label: "Связь", value: "plain",
        options: [{ value: "plain", label: "цепочка" }, { value: "skip", label: "пропуск" }],
      }, function (value) { state.skip = value === "skip"; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function strokeTemplate(digit) {
      var data = new Array(14 * 14).fill(0);
      function mark(x, y, radius) {
        for (var row = 0; row < 14; row += 1) {
          for (var col = 0; col < 14; col += 1) {
            var distance = Math.hypot(col - x, row - y);
            data[row * 14 + col] = Math.max(data[row * 14 + col], clamp(1 - distance / radius, 0, 1));
          }
        }
      }
      function segment(x1, y1, x2, y2) {
        for (var step = 0; step <= 40; step += 1) {
          var t = step / 40;
          mark(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t, 1.45);
        }
      }
      if (digit === 0) {
        segment(4, 2, 9, 2); segment(3, 3, 3, 10); segment(10, 3, 10, 10); segment(4, 11, 9, 11);
      } else if (digit === 1) {
        segment(5, 4, 7, 2); segment(7, 2, 7, 11); segment(4, 11, 10, 11);
      } else if (digit === 3) {
        segment(3, 2, 9, 2); segment(9, 2, 10, 6); segment(4, 6, 9, 6); segment(10, 6, 9, 11); segment(3, 11, 9, 11);
      } else {
        segment(3, 2, 10, 2); segment(10, 2, 6, 11);
      }
      return data;
    }

    function buildDigit(root) {
      var ui = setup(
        root,
        "Рисуй в левой сетке. Простая модель сравнивает изображение с четырьмя прототипами и сразу меняет вероятности.",
        "Рукописная цифра и распределение классов",
        "Вероятность отвечает за выбор модели среди известных классов. Она не доказывает, что рисунок похож на настоящую цифру.",
        485,
      );
      var digits = [0, 1, 3, 7];
      var templates = {};
      digits.forEach(function (digit) { templates[digit] = strokeTemplate(digit); });
      var state = { pixels: templates[3].slice(), drawing: false };
      var grid = { x: 48, y: 60, size: 336 };
      var bars = { x: 525, y: 90, w: 300, h: 250 };

      function probabilities() {
        var scores = digits.map(function (digit) {
          var mse = state.pixels.reduce(function (sum, value, index) {
            return sum + Math.pow(value - templates[digit][index], 2);
          }, 0) / state.pixels.length;
          return -mse * 12;
        });
        var maximum = Math.max.apply(null, scores);
        var exp = scores.map(function (score) { return Math.exp(score - maximum); });
        var total = exp.reduce(function (sum, value) { return sum + value; }, 0);
        return exp.map(function (value, index) {
          return { digit: digits[index], value: value / total };
        }).sort(function (a, b) { return b.value - a.value; });
      }

      function paint(point) {
        var col = Math.floor((point.x - grid.x) / grid.size * 14);
        var row = Math.floor((point.y - grid.y) / grid.size * 14);
        if (row < 0 || row > 13 || col < 0 || col > 13) return;
        for (var dy = -1; dy <= 1; dy += 1) {
          for (var dx = -1; dx <= 1; dx += 1) {
            var rr = row + dy;
            var cc = col + dx;
            if (rr < 0 || rr > 13 || cc < 0 || cc > 13) continue;
            var strength = dx === 0 && dy === 0 ? 1 : 0.45;
            state.pixels[rr * 14 + cc] = Math.max(state.pixels[rr * 14 + cc], strength);
          }
        }
        draw();
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paintPaper(ctx, 485);
        title(ctx, "Поле 14 × 14", grid.x, 35);
        var cell = grid.size / 14;
        state.pixels.forEach(function (value, index) {
          var col = index % 14;
          var row = Math.floor(index / 14);
          var level = Math.round(250 - value * 220);
          ctx.fillStyle = "rgb(" + level + "," + level + "," + level + ")";
          ctx.fillRect(grid.x + col * cell, grid.y + row * cell, cell - 1, cell - 1);
        });
        ctx.strokeStyle = C.axis;
        ctx.strokeRect(grid.x, grid.y, grid.size, grid.size);
        title(ctx, "Вероятности", bars.x, 35);
        var probs = probabilities();
        digits.forEach(function (digit, index) {
          var probability = probs.find(function (item) { return item.digit === digit; }).value;
          var y = bars.y + index * 66;
          small(ctx, String(digit), bars.x - 25, y + 21, C.ink, "center");
          ctx.fillStyle = C.wash;
          ctx.fillRect(bars.x, y, bars.w, 30);
          ctx.fillStyle = index === digits.indexOf(probs[0].digit) ? C.blue : rgba(C.blue, 0.45);
          ctx.fillRect(bars.x, y, bars.w * probability, 30);
          small(ctx, (probability * 100).toFixed(1) + "%", bars.x + bars.w + 12, y + 21, C.ink);
        });
        var margin = probs[0].value - probs[1].value;
        ctx.fillStyle = margin < 0.18 ? rgba(C.gold, 0.12) : rgba(C.green, 0.1);
        ctx.fillRect(bars.x, 382, bars.w, 44);
        small(
          ctx,
          margin < 0.18 ? "соседние классы почти равны" : "лидер отделён от второго места",
          bars.x + 14,
          409,
          margin < 0.18 ? C.gold : C.green,
        );
        ui.output.set([
          { label: "ответ", value: String(probs[0].digit), color: C.blue },
          { label: "вероятность", value: (probs[0].value * 100).toFixed(1) + "%" },
          { label: "второе место", value: String(probs[1].digit) },
          { label: "зазор", value: (margin * 100).toFixed(1) + " п.п." },
        ]);
      }

      var removeDrag = K.drag(ui.canvas.canvas, { w: 920, h: 485 }, {
        down: function (point) { state.drawing = true; paint(point); },
        move: function (point) { if (state.drawing) paint(point); },
        up: function () { state.drawing = false; },
      });
      K.segmented(ui.controls, {
        label: "Заготовка", value: "3",
        options: [
          { value: "clear", label: "чисто" },
          { value: "3", label: "цифра 3" },
          { value: "7", label: "цифра 7" },
        ],
      }, function (value) {
        state.pixels = value === "clear" ? new Array(196).fill(0) : templates[Number(value)].slice();
        draw();
      });
      ui.setDraw(draw);
      draw();
      return function () {
        removeDrag();
        ui.destroy();
      };
    }

    function buildSplit(root) {
      var ui = setup(
        root,
        "Выбери правило разбиения. Цвет клетки показывает выборку; рамка объединяет записи одного пациента.",
        "Строки, пациенты и утечка данных",
        "Если один пациент встречается по обе стороны разбиения, тест измеряет узнавание пациента. Разбиение по группам проверяет перенос на новых людей.",
        485,
      );
      var state = { method: "rows", testShare: 0.3 };
      var panel = { x: 55, y: 72, w: 810, h: 330 };
      var patients = [
        "A-17", "B-04", "C-29", "D-11", "E-08", "F-31", "G-22", "H-15",
      ];

      function assignment(patient, visit) {
        if (state.method === "patients") {
          var testCount = Math.max(1, Math.round(patients.length * state.testShare));
          return patient >= patients.length - testCount ? "test" : "train";
        }
        if (state.method === "time") return visit === 2 ? "test" : "train";
        var threshold = 1 - state.testShare;
        return seeded(patient * 7 + visit, 41) < threshold ? "train" : "test";
      }

      function summary() {
        var leaking = 0;
        var train = 0;
        var test = 0;
        patients.forEach(function (_, patient) {
          var sides = new Set();
          for (var visit = 0; visit < 3; visit += 1) {
            var side = assignment(patient, visit);
            sides.add(side);
            if (side === "train") train += 1;
            else test += 1;
          }
          if (sides.size > 1) leaking += 1;
        });
        return { leaking: leaking, train: train, test: test };
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paintPaper(ctx, 485);
        title(ctx, "24 записи: 8 пациентов × 3 визита", panel.x, 36);
        small(ctx, "обучение", 620, 36, C.blue);
        small(ctx, "тест", 742, 36, C.red);
        ctx.fillStyle = C.blue;
        ctx.fillRect(597, 27, 14, 10);
        ctx.fillStyle = C.red;
        ctx.fillRect(719, 27, 14, 10);
        patients.forEach(function (patient, patientIndex) {
          var y = panel.y + patientIndex * 39;
          ctx.fillStyle = patientIndex % 2 ? C.wash : C.paper;
          ctx.fillRect(panel.x, y, panel.w, 33);
          ctx.strokeStyle = C.grid;
          ctx.strokeRect(panel.x, y, panel.w, 33);
          small(ctx, patient, panel.x + 18, y + 21, C.ink);
          for (var visit = 0; visit < 3; visit += 1) {
            var side = assignment(patientIndex, visit);
            var x = panel.x + 145 + visit * 175;
            ctx.fillStyle = side === "train" ? rgba(C.blue, 0.82) : rgba(C.red, 0.82);
            ctx.fillRect(x, y + 6, 140, 21);
            small(ctx, "визит " + (visit + 1), x + 70, y + 21, C.paper, "center");
          }
          var sides = [0, 1, 2].map(function (visit) { return assignment(patientIndex, visit); });
          if (new Set(sides).size > 1) {
            ctx.strokeStyle = C.gold;
            ctx.lineWidth = 2;
            ctx.strokeRect(panel.x - 5, y - 3, panel.w + 10, 39);
          }
        });
        small(ctx, "Золотая рамка: идентичность пациента попала в обе выборки.", panel.x, 438, C.gold);
        var totals = summary();
        var apparent = 0.71 + totals.leaking / 8 * 0.18;
        ui.output.set([
          { label: "train / test", value: totals.train + " / " + totals.test },
          { label: "пациентов в обеих", value: String(totals.leaking), color: totals.leaking ? C.gold : C.green },
          { label: "видимая accuracy", value: (apparent * 100).toFixed(1) + "%" },
          { label: "что проверяем", value: state.method === "patients" ? "новых пациентов" : state.method === "time" ? "будущие визиты" : "случайные строки" },
        ]);
      }

      K.segmented(ui.controls, {
        label: "Правило", value: state.method,
        options: [
          { value: "rows", label: "по строкам" },
          { value: "patients", label: "по пациентам" },
          { value: "time", label: "по времени" },
        ],
      }, function (value) { state.method = value; draw(); });
      K.slider(ui.controls, {
        label: "Доля теста", min: 0.15, max: 0.5, step: 0.05, value: state.testShare,
        format: function (v) { return Math.round(v * 100); }, unit: "%",
      }, function (value) { state.testShare = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildOverfit(root) {
      var ui = setup(
        root,
        "Сложность модели отмечена вертикальной линией. Меняй шум и объём выборки, чтобы увидеть сдвиг минимума тестовой ошибки.",
        "Train, test и точка нулевой ошибки",
        "Нулевая ошибка на обучении не выбирает модель. Нужна отдельная выборка, потому что тестовая кривая учитывает перенос на новые объекты.",
        475,
      );
      var state = { complexity: 42, noise: 0.22, samples: 80 };
      var plot = { x: 75, y: 62, w: 760, h: 320 };

      function errors(complexity) {
        var x = complexity / 100;
        var n = state.samples;
        var noise = state.noise;
        var train = clamp(0.62 * Math.exp(-6.2 * x) + noise * 0.12 - 0.07 * x, 0.003, 1);
        var bias = 0.47 * Math.exp(-4.8 * x);
        var variance = (0.22 + noise * 0.8) * Math.pow(x, 2.4) * Math.sqrt(100 / n);
        var secondDescent = x > 0.78 ? 0.17 * (x - 0.78) / 0.22 * Math.sqrt(n / 160) : 0;
        var test = clamp(0.08 + noise * 0.45 + bias + variance - secondDescent, 0.02, 0.95);
        return { train: train, test: test };
      }

      function sy(error) {
        return plot.y + plot.h - error / 0.9 * plot.h;
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paintPaper(ctx, 475);
        title(ctx, "Ошибка", plot.x, 34);
        ctx.fillStyle = C.wash;
        ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
        [0, 0.2, 0.4, 0.6, 0.8].forEach(function (error) {
          line(ctx, [{ x: plot.x, y: sy(error) }, { x: plot.x + plot.w, y: sy(error) }], C.grid, 1);
          small(ctx, error.toFixed(1), plot.x - 14, sy(error) + 4, C.muted, "right");
        });
        [0, 25, 50, 75, 100].forEach(function (complexity) {
          var x = plot.x + complexity / 100 * plot.w;
          line(ctx, [{ x: x, y: plot.y }, { x: x, y: plot.y + plot.h }], C.grid, 1);
          small(ctx, String(complexity), x, plot.y + plot.h + 22, C.muted, "center");
        });
        var trainLine = [];
        var testLine = [];
        var minimum = { complexity: 0, error: Infinity };
        for (var complexity = 0; complexity <= 100; complexity += 1) {
          var pair = errors(complexity);
          var x = plot.x + complexity / 100 * plot.w;
          trainLine.push({ x: x, y: sy(pair.train) });
          testLine.push({ x: x, y: sy(pair.test) });
          if (pair.test < minimum.error) minimum = { complexity: complexity, error: pair.test };
        }
        line(ctx, trainLine, C.blue, 2.7);
        line(ctx, testLine, C.red, 2.7);
        var markerX = plot.x + state.complexity / 100 * plot.w;
        line(ctx, [{ x: markerX, y: plot.y }, { x: markerX, y: plot.y + plot.h }], C.ink, 1.6, [5, 4]);
        var current = errors(state.complexity);
        ctx.beginPath();
        ctx.arc(markerX, sy(current.test), 7, 0, Math.PI * 2);
        ctx.fillStyle = C.red;
        ctx.fill();
        ctx.beginPath();
        ctx.arc(markerX, sy(current.train), 7, 0, Math.PI * 2);
        ctx.fillStyle = C.blue;
        ctx.fill();
        small(ctx, "train", plot.x + 18, plot.y + 24, C.blue);
        small(ctx, "test", plot.x + 78, plot.y + 24, C.red);
        small(ctx, "сложность модели →", plot.x + plot.w, 432, C.muted, "right");
        ui.output.set([
          { label: "train", value: current.train.toFixed(3), color: C.blue },
          { label: "test", value: current.test.toFixed(3), color: C.red },
          { label: "зазор", value: (current.test - current.train).toFixed(3) },
          { label: "минимум test", value: "сложность " + minimum.complexity },
        ]);
      }

      K.slider(ui.controls, {
        label: "Сложность", min: 0, max: 100, step: 1, value: state.complexity,
      }, function (value) { state.complexity = value; draw(); });
      K.slider(ui.controls, {
        label: "Шум меток", min: 0, max: 0.6, step: 0.01, value: state.noise,
        format: function (v) { return Math.round(v * 100); }, unit: "%",
      }, function (value) { state.noise = value; draw(); });
      K.slider(ui.controls, {
        label: "Объём выборки", min: 20, max: 400, step: 10, value: state.samples,
      }, function (value) { state.samples = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildingMask(row, col) {
      var main = row >= 3 && row <= 8 && col >= 3 && col <= 10;
      var wing = row >= 6 && row <= 10 && col >= 8 && col <= 14;
      var cut = row >= 7 && row <= 8 && col >= 8 && col <= 9;
      return (main || wing) && !cut;
    }

    function predictedMask(row, col, state) {
      var inside = false;
      for (var dy = -state.dilation; dy <= state.dilation; dy += 1) {
        for (var dx = -state.dilation; dx <= state.dilation; dx += 1) {
          if (buildingMask(row - state.dy + dy, col - state.dx + dx)) inside = true;
        }
      }
      return inside;
    }

    function buildSegmentation(root) {
      var ui = setup(
        root,
        "Сдвигай и расширяй прогноз. Цвет клетки показывает пересечение, пропуск или лишнюю область.",
        "Маска, граница и IoU",
        "IoU штрафует площадь несовпадения. Для тонких объектов и точных контуров к нему часто добавляют отдельную ошибку границы.",
        475,
      );
      var state = { dx: 1, dy: 0, dilation: 0 };
      var grid = { x: 64, y: 58, cell: 25, cols: 18, rows: 13 };

      function metrics() {
        var intersection = 0;
        var union = 0;
        var truth = 0;
        var prediction = 0;
        var boundaryMismatch = 0;
        function boundary(row, col, getter) {
          if (!getter(row, col)) return false;
          return [[1, 0], [-1, 0], [0, 1], [0, -1]].some(function (delta) {
            return !getter(row + delta[0], col + delta[1]);
          });
        }
        for (var row = 0; row < grid.rows; row += 1) {
          for (var col = 0; col < grid.cols; col += 1) {
            var a = buildingMask(row, col);
            var b = predictedMask(row, col, state);
            if (a) truth += 1;
            if (b) prediction += 1;
            if (a && b) intersection += 1;
            if (a || b) union += 1;
            if (boundary(row, col, buildingMask) !== boundary(row, col, function (r, c) {
              return predictedMask(r, c, state);
            })) boundaryMismatch += 1;
          }
        }
        return {
          intersection: intersection,
          union: union,
          truth: truth,
          prediction: prediction,
          iou: intersection / (union || 1),
          dice: 2 * intersection / (truth + prediction || 1),
          boundary: boundaryMismatch,
        };
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paintPaper(ctx, 475);
        title(ctx, "Пиксельная маска", grid.x, 33);
        for (var row = 0; row < grid.rows; row += 1) {
          for (var col = 0; col < grid.cols; col += 1) {
            var truth = buildingMask(row, col);
            var prediction = predictedMask(row, col, state);
            var color = C.wash;
            if (truth && prediction) color = rgba(C.green, 0.8);
            else if (truth) color = rgba(C.blue, 0.82);
            else if (prediction) color = rgba(C.red, 0.8);
            ctx.fillStyle = color;
            ctx.fillRect(
              grid.x + col * grid.cell,
              grid.y + row * grid.cell,
              grid.cell - 1,
              grid.cell - 1,
            );
          }
        }
        ctx.strokeStyle = C.axis;
        ctx.strokeRect(grid.x, grid.y, grid.cols * grid.cell, grid.rows * grid.cell);
        var stats = metrics();
        var chartX = 600;
        title(ctx, "Совпадение", chartX, 33);
        [
          { label: "IoU", value: stats.iou, color: C.green },
          { label: "Dice", value: stats.dice, color: C.blue },
        ].forEach(function (metric, index) {
          var y = 95 + index * 92;
          small(ctx, metric.label, chartX, y - 14, C.ink);
          ctx.fillStyle = C.wash;
          ctx.fillRect(chartX, y, 240, 28);
          ctx.fillStyle = metric.color;
          ctx.fillRect(chartX, y, 240 * metric.value, 28);
          small(ctx, (metric.value * 100).toFixed(1) + "%", chartX + 250, y + 20, metric.color);
        });
        small(ctx, "пересечение", chartX, 298, C.green);
        small(ctx, "пропуск", chartX, 328, C.blue);
        small(ctx, "лишний прогноз", chartX, 358, C.red);
        ctx.fillStyle = C.green; ctx.fillRect(chartX + 120, 288, 16, 12);
        ctx.fillStyle = C.blue; ctx.fillRect(chartX + 120, 318, 16, 12);
        ctx.fillStyle = C.red; ctx.fillRect(chartX + 120, 348, 16, 12);
        ui.output.set([
          { label: "IoU", value: stats.iou.toFixed(3), color: C.green },
          { label: "Dice", value: stats.dice.toFixed(3), color: C.blue },
          { label: "пикселей пересечения", value: String(stats.intersection) },
          { label: "ошибка границы", value: String(stats.boundary) },
        ]);
      }

      K.slider(ui.controls, {
        label: "Сдвиг по x", min: -4, max: 4, step: 1, value: state.dx,
        format: function (v) { return String(v); }, unit: " px",
      }, function (value) { state.dx = value; draw(); });
      K.slider(ui.controls, {
        label: "Сдвиг по y", min: -3, max: 3, step: 1, value: state.dy,
        format: function (v) { return String(v); }, unit: " px",
      }, function (value) { state.dy = value; draw(); });
      K.slider(ui.controls, {
        label: "Расширение", min: 0, max: 2, step: 1, value: state.dilation,
        format: function (v) { return String(v); }, unit: " px",
      }, function (value) { state.dilation = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function blurGrid(values, side, passes) {
      var current = values.slice();
      for (var pass = 0; pass < passes; pass += 1) {
        current = current.map(function (_, index) {
          var row = Math.floor(index / side);
          var col = index % side;
          var sum = 0;
          var count = 0;
          for (var dy = -1; dy <= 1; dy += 1) {
            for (var dx = -1; dx <= 1; dx += 1) {
              var rr = row + dy;
              var cc = col + dx;
              if (rr >= 0 && rr < side && cc >= 0 && cc < side) {
                sum += current[rr * side + cc];
                count += 1;
              }
            }
          }
          return sum / count;
        });
      }
      return current;
    }

    function buildAutoencoder(root) {
      var ui = setup(
        root,
        "Меняй размер кода и уровень шума. Реконструкция использует ограниченный код и знание формы цифры.",
        "Шум, латентный код и реконструкция",
        "Большой код легко копирует детали и шум. Узкий код теряет штрихи, зато вынуждает модель хранить устойчивую структуру.",
        480,
      );
      var state = { code: 8, noise: 0.28, seed: 2 };
      var clean = strokeTemplate(3);
      var panel = { y: 70, size: 260 };

      function noisy() {
        return clean.map(function (value, index) {
          var impulse = seeded(index + state.seed * 211, 77);
          if (impulse < state.noise * 0.16) return 1;
          if (impulse > 1 - state.noise * 0.16) return 0;
          return clamp(value + (seeded(index, state.seed + 83) - 0.5) * state.noise, 0, 1);
        });
      }

      function reconstruction(input) {
        var passes = Math.max(0, Math.round(3.4 - Math.log2(state.code) / 2));
        var smoothed = blurGrid(input, 14, passes);
        var priorWeight = clamp(0.68 - Math.log2(state.code) / 10, 0.08, 0.62);
        return smoothed.map(function (value, index) {
          return clamp(value * (1 - priorWeight) + clean[index] * priorWeight, 0, 1);
        });
      }

      function drawArray(ctx, values, x, label) {
        title(ctx, label, x, 35);
        var cell = panel.size / 14;
        values.forEach(function (value, index) {
          var level = Math.round(250 - value * 220);
          ctx.fillStyle = "rgb(" + level + "," + level + "," + level + ")";
          ctx.fillRect(x + (index % 14) * cell, panel.y + Math.floor(index / 14) * cell, cell - 0.7, cell - 0.7);
        });
        ctx.strokeStyle = C.axis;
        ctx.strokeRect(x, panel.y, panel.size, panel.size);
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paintPaper(ctx, 480);
        var input = noisy();
        var restored = reconstruction(input);
        drawArray(ctx, input, 45, "Повреждённый вход");
        arrow(ctx, 324, 200, 414, 200, C.axis, 2);
        ctx.fillStyle = rgba(C.violet, 0.16);
        ctx.fillRect(352, 154, 36, 92);
        ctx.strokeStyle = C.violet;
        ctx.strokeRect(352, 154, 36, 92);
        small(ctx, String(state.code), 370, 194, C.violet, "center");
        small(ctx, "чисел", 370, 215, C.violet, "center");
        drawArray(ctx, restored, 445, "Реконструкция");

        var latent = { x: 750, y: 92, w: 125, h: 215 };
        title(ctx, "Коды", latent.x, 35);
        ctx.fillStyle = C.wash;
        ctx.fillRect(latent.x, latent.y, latent.w, latent.h);
        ctx.strokeStyle = C.grid;
        ctx.strokeRect(latent.x, latent.y, latent.w, latent.h);
        for (var index = 0; index < 18; index += 1) {
          var angle = index / 18 * Math.PI * 2;
          var radius = 25 + seeded(index, 92) * 22;
          ctx.beginPath();
          ctx.arc(
            latent.x + latent.w / 2 + Math.cos(angle) * radius,
            latent.y + latent.h / 2 + Math.sin(angle) * radius * 1.5,
            4,
            0,
            Math.PI * 2,
          );
          ctx.fillStyle = index === state.seed % 18 ? C.red : rgba(C.violet, 0.5);
          ctx.fill();
        }
        small(ctx, "одна точка = один код", latent.x + latent.w / 2, 335, C.muted, "center");
        var mse = restored.reduce(function (sum, value, index) {
          return sum + Math.pow(value - clean[index], 2);
        }, 0) / restored.length;
        var noisyMse = input.reduce(function (sum, value, index) {
          return sum + Math.pow(value - clean[index], 2);
        }, 0) / input.length;
        ui.output.set([
          { label: "размер кода", value: String(state.code) },
          { label: "ошибка входа", value: noisyMse.toFixed(3), color: C.red },
          { label: "ошибка реконструкции", value: mse.toFixed(3), color: C.blue },
          { label: "сжатие", value: (196 / state.code).toFixed(1) + "×" },
        ]);
      }

      K.segmented(ui.controls, {
        label: "Код", value: "8",
        options: [
          { value: "2", label: "2" },
          { value: "8", label: "8" },
          { value: "32", label: "32" },
          { value: "64", label: "64" },
        ],
      }, function (value) { state.code = Number(value); draw(); });
      K.slider(ui.controls, {
        label: "Шум", min: 0, max: 0.8, step: 0.01, value: state.noise,
        format: function (v) { return Math.round(v * 100); }, unit: "%",
      }, function (value) { state.noise = value; draw(); });
      K.slider(ui.controls, {
        label: "Пример", min: 1, max: 20, step: 1, value: state.seed,
      }, function (value) { state.seed = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildMatrix(root) {
      var ui = setup(
        root,
        "Перетаскивай концы синего и красного столбцов. Справа вместе с ними меняется образ координатной сетки.",
        "Матрица как преобразование плоскости",
        "Столбцы матрицы задают образы базисных векторов. Определитель равен ориентированной площади параллелограмма на этих столбцах.",
        485,
      );
      var state = { a: 1.2, c: 0.25, b: -0.35, d: 1.05, dragging: null };
      var left = { x: 55, y: 75, w: 330, h: 330 };
      var right = { x: 535, y: 75, w: 330, h: 330 };

      function sx(panel, value) { return panel.x + (value + 2.3) / 4.6 * panel.w; }
      function sy(panel, value) { return panel.y + (2.3 - value) / 4.6 * panel.h; }

      function drawGrid(ctx, panel, transformed) {
        ctx.fillStyle = C.wash;
        ctx.fillRect(panel.x, panel.y, panel.w, panel.h);
        for (var value = -2; value <= 2; value += 0.5) {
          var vertical = [];
          var horizontal = [];
          for (var step = -2; step <= 2.001; step += 0.05) {
            var vx = value;
            var vy = step;
            var hx = step;
            var hy = value;
            if (transformed) {
              vertical.push({
                x: sx(panel, state.a * vx + state.b * vy),
                y: sy(panel, state.c * vx + state.d * vy),
              });
              horizontal.push({
                x: sx(panel, state.a * hx + state.b * hy),
                y: sy(panel, state.c * hx + state.d * hy),
              });
            } else {
              vertical.push({ x: sx(panel, vx), y: sy(panel, vy) });
              horizontal.push({ x: sx(panel, hx), y: sy(panel, hy) });
            }
          }
          line(ctx, vertical, value === 0 ? C.axis : C.grid, value === 0 ? 1.6 : 1);
          line(ctx, horizontal, value === 0 ? C.axis : C.grid, value === 0 ? 1.6 : 1);
        }
        ctx.strokeStyle = C.axis;
        ctx.strokeRect(panel.x, panel.y, panel.w, panel.h);
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paintPaper(ctx, 485);
        title(ctx, "До: единичная сетка", left.x, 35);
        title(ctx, "После: A-сетка", right.x, 35);
        drawGrid(ctx, left, false);
        drawGrid(ctx, right, true);
        arrow(ctx, sx(left, 0), sy(left, 0), sx(left, 1), sy(left, 0), C.blue, 3);
        arrow(ctx, sx(left, 0), sy(left, 0), sx(left, 0), sy(left, 1), C.red, 3);
        arrow(ctx, sx(right, 0), sy(right, 0), sx(right, state.a), sy(right, state.c), C.blue, 3);
        arrow(ctx, sx(right, 0), sy(right, 0), sx(right, state.b), sy(right, state.d), C.red, 3);
        var p1 = { x: sx(right, state.a), y: sy(right, state.c) };
        var p2 = { x: sx(right, state.b), y: sy(right, state.d) };
        ctx.beginPath();
        ctx.arc(p1.x, p1.y, 10, 0, Math.PI * 2);
        ctx.fillStyle = C.blue;
        ctx.fill();
        ctx.beginPath();
        ctx.arc(p2.x, p2.y, 10, 0, Math.PI * 2);
        ctx.fillStyle = C.red;
        ctx.fill();
        ctx.fillStyle = rgba(C.gold, 0.14);
        ctx.beginPath();
        ctx.moveTo(sx(right, 0), sy(right, 0));
        ctx.lineTo(p1.x, p1.y);
        ctx.lineTo(sx(right, state.a + state.b), sy(right, state.c + state.d));
        ctx.lineTo(p2.x, p2.y);
        ctx.closePath();
        ctx.fill();
        var determinant = state.a * state.d - state.b * state.c;
        ui.output.set([
          { label: "первый столбец", value: "(" + state.a.toFixed(2) + ", " + state.c.toFixed(2) + ")", color: C.blue },
          { label: "второй столбец", value: "(" + state.b.toFixed(2) + ", " + state.d.toFixed(2) + ")", color: C.red },
          { label: "det A", value: determinant.toFixed(3), color: Math.abs(determinant) < 0.08 ? C.gold : C.green },
          { label: "обратимость", value: Math.abs(determinant) < 0.08 ? "почти потеряна" : "есть" },
        ]);
      }

      var removeDrag = K.drag(ui.canvas.canvas, { w: 920, h: 485 }, {
        down: function (point) {
          var p1 = { x: sx(right, state.a), y: sy(right, state.c) };
          var p2 = { x: sx(right, state.b), y: sy(right, state.d) };
          state.dragging = Math.hypot(point.x - p1.x, point.y - p1.y)
            < Math.hypot(point.x - p2.x, point.y - p2.y) ? "first" : "second";
        },
        move: function (point) {
          if (!state.dragging) return;
          var x = clamp((point.x - right.x) / right.w * 4.6 - 2.3, -2.2, 2.2);
          var y = clamp(2.3 - (point.y - right.y) / right.h * 4.6, -2.2, 2.2);
          if (state.dragging === "first") {
            state.a = x; state.c = y;
          } else {
            state.b = x; state.d = y;
          }
          draw();
        },
        up: function () { state.dragging = null; },
      });
      K.segmented(ui.controls, {
        label: "Пример", value: "custom",
        options: [
          { value: "custom", label: "текущая" },
          { value: "rotation", label: "поворот" },
          { value: "shear", label: "сдвиг" },
          { value: "flat", label: "почти вырождена" },
        ],
      }, function (value) {
        if (value === "rotation") {
          state.a = 0.71; state.c = 0.71; state.b = -0.71; state.d = 0.71;
        } else if (value === "shear") {
          state.a = 1; state.c = 0; state.b = 0.9; state.d = 1;
        } else if (value === "flat") {
          state.a = 1.1; state.c = 0.55; state.b = 1; state.d = 0.48;
        }
        draw();
      });
      ui.setDraw(draw);
      draw();
      return function () {
        removeDrag();
        ui.destroy();
      };
    }

    function buildMatrixCost(root) {
      var ui = setup(
        root,
        "Меняй размер и способ вычисления. Кривые показывают число операций, схема справа — порядок чтения блоков.",
        "Стоимость матричного умножения",
        "Блочный алгоритм выполняет примерно столько же умножений, но чаще использует данные из быстрого кэша. Штрассен меняет асимптотику ценой сложения и сложной реализации.",
        475,
      );
      var state = { n: 512, method: "block", block: 64 };
      var plot = { x: 72, y: 62, w: 555, h: 310 };
      var methods = {
        plain: { label: "три цикла", color: C.ink },
        block: { label: "блоки", color: C.blue },
        strassen: { label: "Штрассен", color: C.red },
      };

      function operations(n, method) {
        if (method === "strassen") return 2.8 * Math.pow(n, Math.log2(7));
        return 2 * Math.pow(n, 3);
      }

      function timeFactor(method) {
        if (method === "plain") return 1.8;
        if (method === "block") return 0.72 + 0.12 * Math.abs(Math.log2(state.block / 64));
        return state.n < 512 ? 1.7 : 0.96;
      }

      function sx(n) {
        return plot.x + (Math.log2(n) - 6) / 5 * plot.w;
      }

      function sy(ops) {
        var log = Math.log10(ops);
        return plot.y + plot.h - (log - 5) / 6 * plot.h;
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paintPaper(ctx, 475);
        title(ctx, "Операции для n × n", plot.x, 34);
        ctx.fillStyle = C.wash;
        ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
        [6, 7, 8, 9, 10, 11].forEach(function (power) {
          var n = Math.pow(2, power);
          var x = sx(n);
          line(ctx, [{ x: x, y: plot.y }, { x: x, y: plot.y + plot.h }], C.grid, 1);
          small(ctx, String(n), x, plot.y + plot.h + 21, C.muted, "center");
        });
        [6, 7, 8, 9, 10, 11].forEach(function (power) {
          var y = sy(Math.pow(10, power));
          line(ctx, [{ x: plot.x, y: y }, { x: plot.x + plot.w, y: y }], C.grid, 1);
          small(ctx, "10^" + power, plot.x - 12, y + 4, C.muted, "right");
        });
        Object.keys(methods).forEach(function (method) {
          var points = [];
          for (var power = 6; power <= 11; power += 0.05) {
            var n = Math.pow(2, power);
            points.push({ x: sx(n), y: sy(operations(n, method)) });
          }
          line(ctx, points, methods[method].color, method === state.method ? 3 : 1.7, method === "block" ? [6, 4] : []);
        });
        var markerX = sx(state.n);
        line(ctx, [{ x: markerX, y: plot.y }, { x: markerX, y: plot.y + plot.h }], C.gold, 1.5, [5, 4]);
        small(ctx, "n", markerX, plot.y - 12, C.gold, "center");
        small(ctx, "три цикла", plot.x + 18, plot.y + 22, C.ink);
        small(ctx, "блоки", plot.x + 104, plot.y + 22, C.blue);
        small(ctx, "Штрассен", plot.x + 165, plot.y + 22, C.red);

        var matrix = { x: 682, y: 82, size: 180 };
        title(ctx, "Доступ к памяти", matrix.x, 34);
        var divisions = clamp(Math.round(state.n / state.block), 2, 8);
        var cell = matrix.size / divisions;
        for (var row = 0; row < divisions; row += 1) {
          for (var col = 0; col < divisions; col += 1) {
            var order = (row * divisions + col) / (divisions * divisions - 1 || 1);
            ctx.fillStyle = rgba(state.method === "plain" ? C.ink : C.blue, 0.08 + order * 0.65);
            ctx.fillRect(matrix.x + col * cell, matrix.y + row * cell, cell - 2, cell - 2);
          }
        }
        ctx.strokeStyle = C.axis;
        ctx.strokeRect(matrix.x, matrix.y, matrix.size, matrix.size);
        var workingSet = state.method === "plain" ? state.n * state.n * 8 * 2 : state.block * state.block * 8 * 3;
        small(ctx, "рабочий фрагмент", matrix.x, 296);
        small(ctx, (workingSet / 1024).toFixed(0) + " KiB", matrix.x, 320, C.blue);
        small(ctx, state.method === "block" ? divisions + " × " + divisions + " блоков" : "длинные проходы", matrix.x, 348);

        var ops = operations(state.n, state.method);
        var seconds = ops / 1.4e9 * timeFactor(state.method);
        ui.output.set([
          { label: "метод", value: methods[state.method].label },
          { label: "операций", value: ops.toExponential(2), color: methods[state.method].color },
          { label: "оценка времени", value: seconds < 1 ? Math.round(seconds * 1000) + " ms" : seconds.toFixed(1) + " s" },
          { label: "память матриц", value: (3 * state.n * state.n * 8 / 1024 / 1024).toFixed(1) + " MiB" },
        ]);
      }

      K.segmented(ui.controls, {
        label: "Алгоритм", value: state.method,
        options: [
          { value: "plain", label: "три цикла" },
          { value: "block", label: "блочный" },
          { value: "strassen", label: "Штрассен" },
        ],
      }, function (value) { state.method = value; draw(); });
      K.slider(ui.controls, {
        label: "Размер n", min: 64, max: 2048, step: 64, value: state.n,
      }, function (value) { state.n = value; draw(); });
      K.segmented(ui.controls, {
        label: "Блок", value: "64",
        options: [
          { value: "16", label: "16" },
          { value: "32", label: "32" },
          { value: "64", label: "64" },
          { value: "128", label: "128" },
        ],
      }, function (value) { state.block = Number(value); draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildPca(root) {
      var ui = setup(
        root,
        "Поворачивай направление ползунком или тяни его конец. Отрезки показывают проекции точек на выбранную ось.",
        "Дисперсия одномерной проекции",
        "Первая главная компонента проходит вдоль максимального разброса центрированных данных. Перпендикулярная ошибка реконструкции при этом минимальна.",
        485,
      );
      var state = { angle: 18, dragging: false, centered: true };
      var scatter = { x: 54, y: 65, w: 560, h: 340 };
      var curve = { x: 675, y: 80, w: 190, h: 210 };
      var cloudAngle = 37 * Math.PI / 180;

      function points() {
        var result = [];
        for (var index = 0; index < 42; index += 1) {
          var along = (seeded(index, 4) - 0.5) * 4.8;
          var across = (seeded(index, 7) - 0.5) * 1.15;
          var x = along * Math.cos(cloudAngle) - across * Math.sin(cloudAngle);
          var y = along * Math.sin(cloudAngle) + across * Math.cos(cloudAngle);
          if (!state.centered) {
            x += 0.8;
            y += 0.45;
          }
          result.push({ x: x, y: y });
        }
        return result;
      }

      function centerOf(data) {
        if (state.centered) return { x: 0, y: 0 };
        return data.reduce(function (sum, point) {
          return { x: sum.x + point.x / data.length, y: sum.y + point.y / data.length };
        }, { x: 0, y: 0 });
      }

      function sx(value) { return scatter.x + (value + 3.4) / 6.8 * scatter.w; }
      function sy(value) { return scatter.y + (2.6 - value) / 5.2 * scatter.h; }

      function varianceAt(angle, data, center) {
        var theta = angle * Math.PI / 180;
        var values = data.map(function (point) {
          return (point.x - center.x) * Math.cos(theta) + (point.y - center.y) * Math.sin(theta);
        });
        var mean = values.reduce(function (sum, value) { return sum + value; }, 0) / values.length;
        return values.reduce(function (sum, value) {
          return sum + Math.pow(value - mean, 2);
        }, 0) / values.length;
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paintPaper(ctx, 485);
        var data = points();
        var center = centerOf(data);
        title(ctx, "Облако и проекции", scatter.x, 35);
        ctx.fillStyle = C.wash;
        ctx.fillRect(scatter.x, scatter.y, scatter.w, scatter.h);
        ctx.strokeStyle = C.grid;
        ctx.strokeRect(scatter.x, scatter.y, scatter.w, scatter.h);
        line(ctx, [{ x: sx(0), y: scatter.y }, { x: sx(0), y: scatter.y + scatter.h }], C.grid, 1);
        line(ctx, [{ x: scatter.x, y: sy(0) }, { x: scatter.x + scatter.w, y: sy(0) }], C.grid, 1);
        var theta = state.angle * Math.PI / 180;
        var ux = Math.cos(theta);
        var uy = Math.sin(theta);
        data.forEach(function (point) {
          var relativeX = point.x - center.x;
          var relativeY = point.y - center.y;
          var t = relativeX * ux + relativeY * uy;
          var projection = { x: center.x + t * ux, y: center.y + t * uy };
          line(ctx, [
            { x: sx(point.x), y: sy(point.y) },
            { x: sx(projection.x), y: sy(projection.y) },
          ], rgba(C.red, 0.24), 1);
          ctx.beginPath();
          ctx.arc(sx(point.x), sy(point.y), 4.2, 0, Math.PI * 2);
          ctx.fillStyle = rgba(C.blue, 0.74);
          ctx.fill();
          ctx.beginPath();
          ctx.arc(sx(projection.x), sy(projection.y), 2.4, 0, Math.PI * 2);
          ctx.fillStyle = C.red;
          ctx.fill();
        });
        var axisLength = 3.1;
        line(ctx, [
          { x: sx(center.x - axisLength * ux), y: sy(center.y - axisLength * uy) },
          { x: sx(center.x + axisLength * ux), y: sy(center.y + axisLength * uy) },
        ], C.ink, 2.4);
        var handle = { x: sx(center.x + 2.4 * ux), y: sy(center.y + 2.4 * uy) };
        ctx.beginPath();
        ctx.arc(handle.x, handle.y, 9, 0, Math.PI * 2);
        ctx.fillStyle = C.gold;
        ctx.fill();

        title(ctx, "Дисперсия по углу", curve.x, 35);
        ctx.fillStyle = C.wash;
        ctx.fillRect(curve.x, curve.y, curve.w, curve.h);
        var variances = [];
        var maximum = 0;
        for (var angle = 0; angle <= 180; angle += 2) {
          var variance = varianceAt(angle, data, center);
          variances.push({ angle: angle, variance: variance });
          maximum = Math.max(maximum, variance);
        }
        line(ctx, variances.map(function (item) {
          return {
            x: curve.x + item.angle / 180 * curve.w,
            y: curve.y + curve.h - item.variance / maximum * curve.h,
          };
        }), C.blue, 2.4);
        var currentVariance = varianceAt(state.angle, data, center);
        var cx = curve.x + state.angle / 180 * curve.w;
        var cy = curve.y + curve.h - currentVariance / maximum * curve.h;
        ctx.beginPath();
        ctx.arc(cx, cy, 6.5, 0, Math.PI * 2);
        ctx.fillStyle = C.gold;
        ctx.fill();
        small(ctx, "0°", curve.x, curve.y + curve.h + 20);
        small(ctx, "90°", curve.x + curve.w / 2, curve.y + curve.h + 20, C.muted, "center");
        small(ctx, "180°", curve.x + curve.w, curve.y + curve.h + 20, C.muted, "right");
        var totalVariance = maximum + Math.min.apply(null, variances.map(function (item) { return item.variance; }));
        ui.output.set([
          { label: "направление", value: state.angle + "°" },
          { label: "дисперсия проекции", value: currentVariance.toFixed(3), color: C.blue },
          { label: "доля", value: (currentVariance / totalVariance * 100).toFixed(1) + "%" },
          { label: "ошибка реконструкции", value: Math.max(0, totalVariance - currentVariance).toFixed(3), color: C.red },
        ]);
      }

      var removeDrag = K.drag(ui.canvas.canvas, { w: 920, h: 485 }, {
        down: function (point) {
          if (point.x >= scatter.x && point.x <= scatter.x + scatter.w && point.y >= scatter.y && point.y <= scatter.y + scatter.h) {
            state.dragging = true;
          }
        },
        move: function (point) {
          if (!state.dragging) return;
          var x = (point.x - scatter.x) / scatter.w * 6.8 - 3.4;
          var y = 2.6 - (point.y - scatter.y) / scatter.h * 5.2;
          var angle = Math.atan2(y, x) * 180 / Math.PI;
          if (angle < 0) angle += 180;
          state.angle = Math.round(angle);
          angleControl.set(state.angle);
          draw();
        },
        up: function () { state.dragging = false; },
      });
      var angleControl = K.slider(ui.controls, {
        label: "Угол направления", min: 0, max: 180, step: 1, value: state.angle,
        format: function (v) { return String(v); }, unit: "°",
      }, function (value) { state.angle = value; draw(); });
      K.segmented(ui.controls, {
        label: "Среднее", value: "center",
        options: [{ value: "center", label: "вычесть" }, { value: "raw", label: "оставить" }],
      }, function (value) { state.centered = value === "center"; draw(); });
      ui.setDraw(draw);
      draw();
      return function () {
        removeDrag();
        ui.destroy();
      };
    }

    function buildRecommender(root) {
      var ui = setup(
        root,
        "Выбери пользователя и регуляризацию. Стрелка показывает его скрытый вектор, а справа меняются предсказанные оценки.",
        "Фильмы и пользователи в скрытом пространстве",
        "Координаты факторов допускают поворот и не обязаны иметь название. Практический смысл проверяют по соседям, ошибкам и рекомендациям.",
        485,
      );
      var state = { user: 0, lambda: 0.15 };
      var users = [
        { name: "Аня", x: 1.8, y: 0.35, bias: 0.1 },
        { name: "Борис", x: -0.5, y: 1.7, bias: -0.15 },
        { name: "Вика", x: -1.5, y: -0.6, bias: 0.25 },
        { name: "Глеб", x: 0.55, y: -1.75, bias: 0 },
      ];
      var films = [
        { name: "Матрица", short: "Матрица", x: 1.9, y: 0.4, bias: 0.2 },
        { name: "История игрушек", short: "Игрушки", x: -0.4, y: 1.8, bias: 0.35 },
        { name: "Фарго", short: "Фарго", x: -1.5, y: -0.8, bias: 0.05 },
        { name: "Контакт", short: "Контакт", x: 1.2, y: 1.1, bias: 0.12 },
        { name: "Титаник", short: "Титаник", x: -0.9, y: 0.9, bias: 0.18 },
        { name: "Криминальное чтиво", short: "Чтиво", x: 0.2, y: -1.7, bias: 0.08 },
      ];
      var map = { x: 52, y: 67, w: 520, h: 340 };
      function sx(value) { return map.x + (value + 2.5) / 5 * map.w; }
      function sy(value) { return map.y + (2.5 - value) / 5 * map.h; }

      function prediction(film) {
        var user = users[state.user];
        var shrink = 1 / (1 + state.lambda * 2.4);
        return clamp(3.1 + user.bias + film.bias + (user.x * film.x + user.y * film.y) * 0.58 * shrink, 0.5, 5);
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paintPaper(ctx, 485);
        title(ctx, "Два скрытых фактора", map.x, 35);
        ctx.fillStyle = C.wash;
        ctx.fillRect(map.x, map.y, map.w, map.h);
        line(ctx, [{ x: sx(0), y: map.y }, { x: sx(0), y: map.y + map.h }], C.grid, 1);
        line(ctx, [{ x: map.x, y: sy(0) }, { x: map.x + map.w, y: sy(0) }], C.grid, 1);
        ctx.strokeStyle = C.grid;
        ctx.strokeRect(map.x, map.y, map.w, map.h);
        films.forEach(function (film) {
          ctx.beginPath();
          ctx.arc(sx(film.x), sy(film.y), 7, 0, Math.PI * 2);
          ctx.fillStyle = C.blue;
          ctx.fill();
          small(ctx, film.short, sx(film.x) + 10, sy(film.y) - 8, C.ink);
        });
        var user = users[state.user];
        arrow(ctx, sx(0), sy(0), sx(user.x), sy(user.y), C.red, 3);
        ctx.beginPath();
        ctx.arc(sx(user.x), sy(user.y), 10, 0, Math.PI * 2);
        ctx.fillStyle = C.red;
        ctx.fill();
        small(ctx, user.name, sx(user.x) + 13, sy(user.y) + 17, C.red);

        var ranked = films.map(function (film) {
          return { film: film, rating: prediction(film) };
        }).sort(function (a, b) { return b.rating - a.rating; });
        var bars = { x: 640, y: 76, w: 210 };
        title(ctx, "Прогноз оценки", bars.x, 35);
        ranked.forEach(function (item, index) {
          var y = bars.y + index * 54;
          small(ctx, item.film.short, bars.x, y, C.ink);
          ctx.fillStyle = C.wash;
          ctx.fillRect(bars.x, y + 10, bars.w, 17);
          ctx.fillStyle = index === 0 ? C.red : C.blue;
          ctx.fillRect(bars.x, y + 10, bars.w * item.rating / 5, 17);
          small(ctx, item.rating.toFixed(2), bars.x + bars.w + 10, y + 24, index === 0 ? C.red : C.ink);
        });
        ui.output.set([
          { label: "пользователь", value: user.name, color: C.red },
          { label: "рекомендация", value: ranked[0].film.name, color: C.blue },
          { label: "прогноз", value: ranked[0].rating.toFixed(2) + " / 5" },
          { label: "норма вектора", value: (Math.hypot(user.x, user.y) / (1 + state.lambda)).toFixed(2) },
        ]);
      }

      K.segmented(ui.controls, {
        label: "Пользователь", value: "0",
        options: users.map(function (user, index) {
          return { value: String(index), label: user.name };
        }),
      }, function (value) { state.user = Number(value); draw(); });
      K.slider(ui.controls, {
        label: "Регуляризация λ", min: 0, max: 1, step: 0.01, value: state.lambda,
        format: function (v) { return v.toFixed(2); },
      }, function (value) { state.lambda = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    function buildAudio(root) {
      var ui = setup(
        root,
        "Смешай две частоты и меняй длину окна. Спектрограмма показывает, где во времени звучит каждая составляющая.",
        "Сигнал, спектрограмма и эмбеддинг",
        "Короткое окно точнее отмечает время, длинное лучше разделяет близкие частоты. Признаки спектра затем можно сжать и сравнивать как обычные векторы.",
        500,
      );
      var state = { f1: 220, f2: 520, window: 64, balance: 0.55 };
      var wave = { x: 65, y: 52, w: 790, h: 115 };
      var spec = { x: 65, y: 218, w: 560, h: 205 };
      var embed = { x: 690, y: 218, w: 165, h: 205 };

      function envelope(t, component) {
        if (component === 1) return t < 0.72 ? Math.sin(Math.min(1, t / 0.08) * Math.PI / 2) : Math.max(0, (1 - t) / 0.28);
        return t > 0.28 && t < 0.9 ? Math.sin(Math.min(1, (t - 0.28) / 0.09) * Math.PI / 2) : 0;
      }

      function signal(t) {
        return state.balance * envelope(t, 1) * Math.sin(2 * Math.PI * state.f1 * t)
          + (1 - state.balance) * envelope(t, 2) * Math.sin(2 * Math.PI * state.f2 * t);
      }

      function draw() {
        var ctx = ui.canvas.ctx;
        paintPaper(ctx, 500);
        title(ctx, "Осциллограмма, фрагмент 50 ms", wave.x, 30);
        ctx.fillStyle = C.wash;
        ctx.fillRect(wave.x, wave.y, wave.w, wave.h);
        line(ctx, [{ x: wave.x, y: wave.y + wave.h / 2 }, { x: wave.x + wave.w, y: wave.y + wave.h / 2 }], C.grid, 1);
        var wavePoints = [];
        for (var index = 0; index <= 720; index += 1) {
          var t = index / 720;
          var sampleTime = 0.38 + t * 0.05;
          wavePoints.push({
            x: wave.x + t * wave.w,
            y: wave.y + wave.h / 2 - signal(sampleTime) * wave.h * 0.43,
          });
        }
        line(ctx, wavePoints, C.blue, 1.8);
        small(ctx, "0", wave.x, wave.y + wave.h + 18);
        small(ctx, "50 ms", wave.x + wave.w, wave.y + wave.h + 18, C.muted, "right");

        title(ctx, "Спектрограмма", spec.x, 202);
        var timeBins = 48;
        var freqBins = 24;
        var cellW = spec.w / timeBins;
        var cellH = spec.h / freqBins;
        for (var row = 0; row < freqBins; row += 1) {
          var frequency = (freqBins - row - 0.5) / freqBins * 900;
          for (var col = 0; col < timeBins; col += 1) {
            var t = (col + 0.5) / timeBins;
            var resolution = 95 * 64 / state.window;
            var energy1 = envelope(t, 1) * Math.exp(-Math.pow(frequency - state.f1, 2) / (2 * resolution * resolution)) * state.balance;
            var energy2 = envelope(t, 2) * Math.exp(-Math.pow(frequency - state.f2, 2) / (2 * resolution * resolution)) * (1 - state.balance);
            var energy = clamp((energy1 + energy2) * 1.8, 0, 1);
            var red = Math.round(247 - energy * 108);
            var green = Math.round(245 - energy * 154);
            var blue = Math.round(237 - energy * 188);
            ctx.fillStyle = "rgb(" + red + "," + green + "," + blue + ")";
            ctx.fillRect(spec.x + col * cellW, spec.y + row * cellH, cellW + 0.3, cellH + 0.3);
          }
        }
        ctx.strokeStyle = C.axis;
        ctx.strokeRect(spec.x, spec.y, spec.w, spec.h);
        small(ctx, "900 Hz", spec.x - 10, spec.y + 4, C.muted, "right");
        small(ctx, "0", spec.x - 10, spec.y + spec.h, C.muted, "right");

        title(ctx, "Карта фрагментов", embed.x, 202);
        ctx.fillStyle = C.wash;
        ctx.fillRect(embed.x, embed.y, embed.w, embed.h);
        ctx.strokeStyle = C.grid;
        ctx.strokeRect(embed.x, embed.y, embed.w, embed.h);
        var candidates = [
          { name: "бас", x: 0.18, y: 0.76, f: 150 },
          { name: "гитара", x: 0.42, y: 0.48, f: 420 },
          { name: "голос", x: 0.64, y: 0.34, f: 610 },
          { name: "тарелка", x: 0.83, y: 0.18, f: 820 },
        ];
        var centroid = state.f1 * state.balance + state.f2 * (1 - state.balance);
        candidates.forEach(function (candidate) {
          ctx.beginPath();
          ctx.arc(embed.x + candidate.x * embed.w, embed.y + candidate.y * embed.h, 5, 0, Math.PI * 2);
          ctx.fillStyle = C.blue;
          ctx.fill();
          small(ctx, candidate.name, embed.x + candidate.x * embed.w + 7, embed.y + candidate.y * embed.h - 7);
        });
        var currentX = clamp(centroid / 900, 0.05, 0.95);
        var currentY = clamp(0.92 - Math.abs(state.f2 - state.f1) / 900, 0.08, 0.92);
        ctx.beginPath();
        ctx.arc(embed.x + currentX * embed.w, embed.y + currentY * embed.h, 9, 0, Math.PI * 2);
        ctx.fillStyle = C.red;
        ctx.fill();
        var nearest = candidates.slice().sort(function (a, b) {
          return Math.abs(a.f - centroid) - Math.abs(b.f - centroid);
        })[0];
        ui.output.set([
          { label: "частоты", value: state.f1 + " + " + state.f2 + " Hz" },
          { label: "спектральный центр", value: Math.round(centroid) + " Hz", color: C.red },
          { label: "окно", value: state.window + " отсчётов" },
          { label: "ближайший фрагмент", value: nearest.name, color: C.blue },
        ]);
      }

      K.slider(ui.controls, {
        label: "Первая частота", min: 80, max: 700, step: 10, value: state.f1,
        format: function (v) { return String(v); }, unit: " Hz",
      }, function (value) { state.f1 = value; draw(); });
      K.slider(ui.controls, {
        label: "Вторая частота", min: 120, max: 900, step: 10, value: state.f2,
        format: function (v) { return String(v); }, unit: " Hz",
      }, function (value) { state.f2 = value; draw(); });
      K.segmented(ui.controls, {
        label: "Окно STFT", value: "64",
        options: [
          { value: "32", label: "32" },
          { value: "64", label: "64" },
          { value: "128", label: "128" },
          { value: "256", label: "256" },
        ],
      }, function (value) { state.window = Number(value); draw(); });
      K.slider(ui.controls, {
        label: "Баланс первой", min: 0.1, max: 0.9, step: 0.01, value: state.balance,
        format: function (v) { return Math.round(v * 100); }, unit: "%",
      }, function (value) { state.balance = value; draw(); });
      ui.setDraw(draw);
      draw();
      return ui.destroy;
    }

    var builders = {
      "27": buildReceptiveField,
      "28": buildConvolution,
      "29": buildPooling,
      "30": buildArchitecture,
      "31": buildDigit,
      "32": buildSplit,
      "33": buildOverfit,
      "34": buildSegmentation,
      "35": buildAutoencoder,
      "36": buildMatrix,
      "37": buildMatrixCost,
      "38": buildPca,
      "39": buildRecommender,
      "40": buildAudio,
    };

    K.register("g10-vision", function (root, options) {
      var lesson = String(options.lesson || "");
      var builder = builders[lesson];
      if (!builder) {
        root.appendChild(K.element("p", "kontur-int-error", {
          text: "Для этого урока эксперимент ещё не собран.",
        }));
        return function () {};
      }
      return builder(root);
    });
  }

  if (window.KonturInt) install();
  else window.addEventListener("kontur-int-ready", install, { once: true });
})();
