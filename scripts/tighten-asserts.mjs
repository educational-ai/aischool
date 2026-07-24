#!/usr/bin/env node
/**
 * Ужесточает допуски вида `assert abs(X - T) < TOL` до половины последнего
 * печатаемого знака T (с запасом 0,4 вместо 0,5).
 *
 * Смысл операции: допуск шире напечатанной точности молча пропускает ошибку
 * ровно в той цифре, которую видит читатель. После ужесточения падение assert
 * означает, что в уроке напечатано не то число, которое считает скрипт.
 *
 * Запуск: node scripts/tighten-asserts.mjs [--apply] [NN ...]
 */
import fs from "node:fs";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "..");
const argv = process.argv.slice(2);
const apply = argv.includes("--apply");
const ids = argv.filter((a) => /^\d+$/.test(a));
const targets = ids.length ? ids : Array.from({ length: 40 }, (_, i) => String(51 + i));

const RE = /(assert\s+abs\((?:[^()]|\([^()]*\))*?\s*-\s*)([\d]+\.[\d]+)(\)\s*<\s*)([\d.eE+-]+)/g;

let touched = 0;
let rewritten = 0;

for (const id of targets) {
  const file = path.join(root, `scripts/generate_lesson${id}_visuals.py`);
  if (!fs.existsSync(file)) continue;
  const src = fs.readFileSync(file, "utf8");
  const changes = [];

  const next = src.replace(RE, (full, head, target, mid, tol) => {
    const digits = (target.split(".")[1] || "").length;
    if (!digits) return full;
    const allowed = 0.5 * Math.pow(10, -digits);
    const current = Number(tol);
    if (!Number.isFinite(current) || current < allowed) return full;
    // 0,4 последнего знака: строго внутри допустимого, но не впритык к нулю
    const tightened = Number((0.4 * Math.pow(10, -digits)).toPrecision(2));
    changes.push(`${target}: ${tol} -> ${tightened}`);
    return `${head}${target}${mid}${tightened}`;
  });

  if (changes.length) {
    touched += 1;
    rewritten += changes.length;
    console.log(`${id}: ${changes.length} допусков`);
    for (const c of changes.slice(0, 4)) console.log(`    ${c}`);
    if (apply) fs.writeFileSync(file, next);
  }
}

console.log(`\nуроков затронуто: ${touched}, допусков ужесточено: ${rewritten}${apply ? "" : " (пробный прогон, нужен --apply)"}`);
