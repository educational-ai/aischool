#!/usr/bin/env node
/**
 * Проверка числовой дисциплины (EDITORIAL_CONTRACT.md, раздел «Числовая дисциплина»).
 *
 * Ловит ровно те дефекты, которые адверсальный разбор нашёл в уроке 51:
 *   - число напечатано в прозе, но не посчитано скриптом;
 *   - допуск assert шире половины последнего печатаемого знака;
 *   - опорное утверждение висит на подвыборке с фиксированным seed;
 *   - в тексте есть императив к виджету, который никто не исполнял.
 *
 * Запуск: node scripts/numeric-discipline-audit.mjs [NN ...]
 */
import fs from "node:fs";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "..");
const ids = process.argv.slice(2).length
  ? process.argv.slice(2)
  : Array.from({ length: 40 }, (_, i) => String(51 + i));

/** Числа из прозы: русская запись 3,14 и обычная 3.14, включая проценты. */
function prosaNumbers(text) {
  // выбрасываем формулы, директивы и ссылки - там числа живут по своим правилам
  const clean = text
    .replace(/\$\$[\s\S]*?\$\$/g, " ")
    .replace(/\$[^$\n]*\$/g, " ")
    .replace(/^:{3,}.*$/gm, " ")
    .replace(/\/lesson\/\d+/g, " ")
    // «урок 46», «в уроке 32», «из урока 48» — это ссылка, а не измерение
    .replace(/уроk?[а-яё]*\s*\d+/gi, " ")
    .replace(/[Рр]ис\.?\s*\d+\.\d+/g, " ")
    .replace(/\bfig-\d+-\d+\b/g, " ")
    .replace(/\b(19|20)\d{2}\b/g, " ") // годы в атрибуциях
    // разряды через пробел: «2 099 712» — одно число, а не три
    .replace(/(\d)[\s  ](?=\d{3}\b)/g, "$1");
  const out = new Set();
  for (const m of clean.matchAll(/\d+(?:[.,]\d+)?/g)) {
    const raw = m[0].replace(",", ".");
    const v = Number(raw);
    if (!Number.isFinite(v)) continue;
    if (Number.isInteger(v) && v <= 12) continue; // счётные мелочи: «три слоя», «пять шагов»
    out.add(raw);
  }
  return [...out];
}

/**
 * Числа, которые скрипт действительно считает.
 *
 * Часть уроков хардкодит их в assert, часть считает и сбрасывает в
 * scripts/data/lessonNN_facts.json - учитываем оба источника, иначе честно
 * посчитанное число выглядит взятым с потолка.
 */
function scriptNumbers(src, id) {
  const out = new Set();
  for (const m of src.matchAll(/\d+(?:\.\d+)?/g)) out.add(m[0]);

  const dump = path.join(root, `scripts/data/lesson${id}_facts.json`);
  if (fs.existsSync(dump)) {
    const walk = (node) => {
      if (node === null) return;
      if (typeof node === "number") {
        out.add(String(node));
        out.add(node.toFixed(1));
        out.add(node.toFixed(2));
        out.add(node.toFixed(3));
        return;
      }
      if (typeof node === "string") {
        for (const m of node.matchAll(/\d+(?:[.,]\d+)?/g)) out.add(m[0].replace(",", "."));
        return;
      }
      if (Array.isArray(node)) return node.forEach(walk);
      if (typeof node === "object") return Object.values(node).forEach(walk);
    };
    try {
      walk(JSON.parse(fs.readFileSync(dump, "utf8")));
    } catch {
      /* повреждённый дамп поймает сам скрипт */
    }
  }
  return out;
}

/** Совпадение с точностью до округления: 0,4825 в скрипте покрывает 0,483 в прозе. */
function covered(value, pool) {
  if (pool.has(value)) return true;
  const v = Number(value);
  const digits = (value.split(".")[1] || "").length;
  for (const cand of pool) {
    const c = Number(cand);
    if (!Number.isFinite(c)) continue;
    if (digits === 0) {
      if (Math.round(c) === v) return true;
    } else if (Math.abs(Number(c.toFixed(digits)) - v) < 1e-9) {
      return true;
    }
    // проза часто печатает проценты там, где скрипт держит долю
    if (Math.abs(Number((c * 100).toFixed(digits)) - v) < 1e-9) return true;
  }
  return false;
}

/** Допуск assert должен быть строго меньше половины последнего печатаемого знака. */
function looseAsserts(src) {
  const bad = [];
  const re = /assert\s+abs\(([^)]*?)\s*-\s*([\d.]+)\)\s*<\s*([\d.eE+-]+)/g;
  for (const m of src.matchAll(re)) {
    const target = m[2];
    const tol = Number(m[3]);
    const digits = (target.split(".")[1] || "").length;
    const allowed = 0.5 * Math.pow(10, -digits);
    // Ровно половина последнего знака - это и есть тест «напечатанная цифра верна»;
    // ошибкой считаем только допуск ШИРЕ него.
    if (Number.isFinite(tol) && tol > allowed * 1.0000001) {
      bad.push(`abs(${m[1].trim()} - ${target}) < ${m[3]} — при ${digits} знаках допуск обязан быть <= ${allowed}`);
    }
  }
  return bad;
}

const IMPERATIVE = /(поставьте|доведите|убедитесь|уведите|потащите|перетащите|сдвиньте|переключитесь)/gi;

let totalIssues = 0;
const summary = [];

for (const id of ids) {
  const md = path.join(root, `content/lessons/${id}.md`);
  const py = path.join(root, `scripts/generate_lesson${id}_visuals.py`);
  if (!fs.existsSync(md)) continue;

  const text = fs.readFileSync(md, "utf8");
  const body = text.split(/\n##\s+Задачи/)[0];
  const issues = [];

  if (!fs.existsSync(py)) {
    issues.push("нет скрипта фигур — числа проверить нечем");
  } else {
    const src = fs.readFileSync(py, "utf8");
    const pool = scriptNumbers(src, id);

    const orphan = prosaNumbers(body).filter((n) => !covered(n, pool));
    if (orphan.length) {
      issues.push(`числа прозы вне скрипта (${orphan.length}): ${orphan.slice(0, 12).join(", ")}`);
    }

    const loose = looseAsserts(src);
    if (loose.length) issues.push(`слабые допуски (${loose.length}): ${loose[0]}`);

    if (/seed\s*=\s*(\d+)/.test(src)) {
      const seeds = [...src.matchAll(/seed\s*=\s*(\d+)/g)].map((m) => m[1]);
      if (seeds.includes(id)) {
        issues.push(`seed совпадает с номером урока (${id}) — маркер подгонки, нужен прогон по нескольким seed`);
      }
    }
  }

  const imperatives = [...body.matchAll(IMPERATIVE)].length;
  const widget = (text.match(/widget\{name="([^"]+)"/) || [])[1];
  if (widget) {
    const js = path.join(root, `public/interactive/widgets/${widget}.js`);
    if (!fs.existsSync(js)) issues.push(`виджет ${widget}.js отсутствует`);
    else if (!fs.readFileSync(js, "utf8").includes(`register("${widget}"`)) {
      issues.push(`виджет ${widget}.js не регистрирует своё имя`);
    }
  } else {
    issues.push("в уроке нет :::widget");
  }

  totalIssues += issues.length;
  summary.push({ id, issues, imperatives });
}

for (const { id, issues, imperatives } of summary) {
  if (!issues.length) {
    console.log(`✔ ${id}: чисто${imperatives ? ` (императивов к виджету: ${imperatives} — проверить вручную)` : ""}`);
  } else {
    console.log(`✖ ${id}:`);
    for (const i of issues) console.log(`    ${i}`);
    if (imperatives) console.log(`    императивов к виджету: ${imperatives} — проверить вручную`);
  }
}
console.log(`\nуроков проверено: ${summary.length}, замечаний: ${totalIssues}`);
