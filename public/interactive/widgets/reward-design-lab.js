// Lesson 06: night rebalancing between three stations under an editable reward.
(function () {
  "use strict";

  function install() {
    if (!window.KonturInt) return;

    window.KonturInt.register("reward-design-lab", function (root, options, K) {
      var C = K.COLORS;
      var W = 1040;
      var H = 620;
      var DAYS = 28;
      var CAPACITY = [40, 40, 40];
      var TOTAL_BIKES = 45;
      // Morning demand rhythm per station: center, office district, suburb.
      var BASE_DEMAND = [
        [26, 30, 31, 29, 27, 12, 9],
        [21, 24, 25, 24, 22, 8, 6],
        [12, 13, 13, 13, 12, 16, 18],
      ];

      var state = {
        alpha: Number(options.alpha || 10),
        beta: Number(options.beta || 0),
        delta: Number(options.delta || 0),
        idle: false,
      };

      K.hint(
        root,
        "Агент каждую ночь развозит велосипеды жадно по своей награде. Начни с награды только за поездки и посмотри, что происходит с окраиной; затем закрывай лазейки коэффициентами.",
      );
      var stage = K.row(root);
      var controls = K.row(root, "controls");
      var output = K.readout(root);
      K.caption(
        root,
        "Три станции: центр, деловой район, спальная окраина; велосипедов 45, и на всех не хватает. Ночной агент перебирает допустимые развозки и выбирает лучшую по награде r = α·поездки − β·км − δ·часы пустых станций. Спрос детерминированный, seed = 28.",
      );

      var canvasState = K.makeCanvas(stage, W, H, {
        label: "Месяц ночных развозок и структура награды",
        onResize: draw,
        drag: false,
      });

      function mulberry(seed) {
        var a = seed >>> 0;
        return function () {
          a = (a + 0x6d2b79f5) >>> 0;
          var t = a;
          t = Math.imul(t ^ (t >>> 15), t | 1);
          t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
          return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
      }

      // Distances in km between stations 0-1, 0-2, 1-2.
      var DIST = [[0, 2, 6], [2, 0, 7], [6, 7, 0]];

      function demandFor(day) {
        var rnd = mulberry(28 + day * 97);
        return BASE_DEMAND.map(function (row) {
          var base = row[day % 7];
          return Math.max(0, Math.round(base + (rnd() - 0.5) * 6));
        });
      }

      // Enumerate allocations of 60 bikes over 3 stations (step 5, cap 40).
      var ALLOCATIONS = (function () {
        var list = [];
        for (var a = 0; a <= CAPACITY[0]; a += 5) {
          for (var b = 0; b <= CAPACITY[1]; b += 5) {
            var c = TOTAL_BIKES - a - b;
            if (c >= 0 && c <= CAPACITY[2]) list.push([a, b, c]);
          }
        }
        return list;
      })();

      function evaluate(allocation, previous, demand, weights) {
        var km = 0;
        var moved = allocation.map(function (v, i) { return v - previous[i]; });
        // Transport cost: move surpluses to deficits along shortest pairs (greedy).
        var surplus = [], deficit = [];
        moved.forEach(function (m, i) {
          if (m < 0) surplus.push({ i: i, amount: -m });
          if (m > 0) deficit.push({ i: i, amount: m });
        });
        surplus.forEach(function (s) {
          deficit.forEach(function (d) {
            if (!s.amount || !d.amount) return;
            var qty = Math.min(s.amount, d.amount);
            km += qty * DIST[s.i][d.i] / 10;
            s.amount -= qty;
            d.amount -= qty;
          });
        });
        var trips = 0;
        var emptyHours = 0;
        var leftover = [];
        for (var i = 0; i < 3; i += 1) {
          var served = Math.min(allocation[i], demand[i]);
          trips += served;
          leftover.push(allocation[i] - served);
          if (allocation[i] < demand[i]) {
            emptyHours += Math.min(4, Math.ceil((demand[i] - allocation[i]) / 5));
          }
        }
        return {
          reward: weights.alpha * trips - weights.beta * km - weights.delta * emptyHours,
          trips: trips,
          km: km,
          emptyHours: emptyHours,
          leftover: leftover,
        };
      }

      function runMonth(weights, idle) {
        var stock = [15, 15, 15];
        var days = [];
        var totals = { trips: 0, km: 0, emptyHours: 0, demand: 0 };
        for (var t = 0; t < DAYS; t += 1) {
          var demand = demandFor(t);
          var choice;
          if (idle) {
            choice = stock.slice();
          } else {
            var best = null;
            for (var k = 0; k < ALLOCATIONS.length; k += 1) {
              var trial = evaluate(ALLOCATIONS[k], stock, demand, weights);
              if (!best || trial.reward > best.reward + 1e-9) {
                best = trial;
                choice = ALLOCATIONS[k];
              }
            }
          }
          var outcome = evaluate(choice, stock, demand, weights);
          totals.trips += outcome.trips;
          totals.km += outcome.km;
          totals.emptyHours += outcome.emptyHours;
          totals.demand += demand[0] + demand[1] + demand[2];
          days.push({ demand: demand, allocation: choice, outcome: outcome });
          stock = outcome.leftover.map(function (v, i) {
            return Math.min(CAPACITY[i], v + Math.round((choice[i] ? choice[i] : 0) * 0));
          });
          // Bikes ride back during the day: everything returns to circulation.
          var total = stock[0] + stock[1] + stock[2];
          var deficitTotal = TOTAL_BIKES - total;
          stock[0] += deficitTotal; // returned bikes gather at the centre overnight
          if (stock[0] > CAPACITY[0]) {
            var spill = stock[0] - CAPACITY[0];
            stock[0] = CAPACITY[0];
            stock[1] = Math.min(CAPACITY[1], stock[1] + spill);
          }
        }
        return { days: days, totals: totals };
      }

      var strip = { x: 70, y: 60, w: 910, h: 300 };

      function draw() {
        var ctx = canvasState.ctx;
        ctx.clearRect(0, 0, W, H);
        ctx.font = "13px PT Sans, sans-serif";
        ctx.textBaseline = "middle";

        var weights = { alpha: state.alpha, beta: state.beta, delta: state.delta };
        var run = runMonth(weights, false);
        var idleRun = runMonth(weights, true);

        ctx.fillStyle = C.ink;
        ctx.font = "bold 14px PT Sans, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText("Утренний спрос и ночная развозка: 28 дней, три станции", strip.x, strip.y - 34);
        ctx.font = "13px PT Sans, sans-serif";

        var names = ["центр", "деловой район", "окраина"];
        var colors = [C.blue, C.green, C.gold];
        var laneH = strip.h / 3;
        var barW = strip.w / DAYS;
        for (var s = 0; s < 3; s += 1) {
          var laneY = strip.y + s * laneH;
          ctx.strokeStyle = C.grid;
          ctx.strokeRect(strip.x, laneY, strip.w, laneH - 12);
          ctx.fillStyle = colors[s];
          ctx.fillText(names[s], strip.x - 62, laneY + (laneH - 12) / 2);
          for (var t = 0; t < DAYS; t += 1) {
            var day = run.days[t];
            var maxVal = 45;
            var x = strip.x + t * barW;
            var demandH = day.demand[s] / maxVal * (laneH - 16);
            var allocH = day.allocation[s] / maxVal * (laneH - 16);
            ctx.fillStyle = C.grid;
            ctx.fillRect(x + barW * 0.16, laneY + laneH - 14 - demandH, barW * 0.3, demandH);
            var starves = day.allocation[s] < day.demand[s];
            ctx.fillStyle = starves ? C.red : colors[s];
            ctx.fillRect(x + barW * 0.52, laneY + laneH - 14 - allocH, barW * 0.3, allocH);
          }
        }
        ctx.fillStyle = C.muted;
        ctx.fillText("серый столбик — спрос; цветной — оставлено агентом; красный — станции не хватило", strip.x, strip.y + strip.h + 10);

        // Reward decomposition bar.
        var barY = strip.y + strip.h + 52;
        ctx.fillStyle = C.ink;
        ctx.font = "bold 14px PT Sans, sans-serif";
        ctx.fillText("Структура набранной награды за месяц", strip.x, barY - 18);
        ctx.font = "13px PT Sans, sans-serif";
        var gain = weights.alpha * run.totals.trips;
        var costKm = weights.beta * run.totals.km;
        var costEmpty = weights.delta * run.totals.emptyHours;
        var scale = 620 / Math.max(gain, 1);
        ctx.fillStyle = C.green;
        ctx.fillRect(strip.x, barY, gain * scale, 22);
        ctx.fillStyle = C.gold;
        ctx.fillRect(strip.x, barY + 30, costKm * scale, 22);
        ctx.fillStyle = C.red;
        ctx.fillRect(strip.x, barY + 60, costEmpty * scale, 22);
        ctx.fillStyle = C.ink;
        ctx.fillText("+" + Math.round(gain) + " за поездки", strip.x + gain * scale + 10, barY + 11);
        ctx.fillText("−" + Math.round(costKm) + " за километры", strip.x + costKm * scale + 10, barY + 41);
        ctx.fillText("−" + Math.round(costEmpty) + " за пустые станции", strip.x + costEmpty * scale + 10, barY + 71);

        var served = Math.round(run.totals.trips / run.totals.demand * 100);
        var servedIdle = Math.round(idleRun.totals.trips / idleRun.totals.demand * 100);
        output.set([
          {
            label: "спрос удовлетворён",
            value: served + "%",
            color: served > 92 ? C.green : served > 80 ? C.gold : C.red,
          },
          { label: "километров за месяц", value: String(Math.round(run.totals.km)) },
          {
            label: "часов пустых станций",
            value: String(run.totals.emptyHours),
            color: run.totals.emptyHours > 30 ? C.red : C.green,
          },
          { label: "без развозок было бы", value: servedIdle + "% спроса" },
        ]);
      }

      K.slider(
        controls,
        { label: "α — цена одной поездки", min: 1, max: 20, step: 1, value: state.alpha },
        function (value) { state.alpha = value; draw(); },
      );
      K.slider(
        controls,
        { label: "β — цена километра", min: 0, max: 30, step: 1, value: state.beta },
        function (value) { state.beta = value; draw(); },
      );
      K.slider(
        controls,
        { label: "δ — цена часа пустой станции", min: 0, max: 40, step: 2, value: state.delta },
        function (value) { state.delta = value; draw(); },
      );

      draw();
      return function () {
        canvasState.destroy();
      };
    });
  }

  install();
  window.addEventListener("kontur-int-ready", install);
})();
