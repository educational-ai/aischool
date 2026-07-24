#!/usr/bin/env node
/**
 * Сводит фрагменты editorial/pending-contracts/NN.json в общий контракт
 * editorial/lesson-quality-v2.json.
 *
 * Уроки 52–90 перерабатываются параллельно, и каждый агент кладёт свой порог
 * отдельным файлом: писать в общий контракт одновременно нельзя — правки
 * затирают друг друга. Слияние выполняется одним проходом здесь.
 */
import fs from "node:fs";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "..");
const contractPath = path.join(root, "editorial/lesson-quality-v2.json");
const pendingDir = path.join(root, "editorial/pending-contracts");

if (!fs.existsSync(pendingDir)) {
  console.log("нет каталога pending-contracts — сливать нечего");
  process.exit(0);
}

const contract = JSON.parse(fs.readFileSync(contractPath, "utf8"));
const files = fs.readdirSync(pendingDir).filter((f) => f.endsWith(".json")).sort();

const merged = [];
const skipped = [];

for (const file of files) {
  const raw = JSON.parse(fs.readFileSync(path.join(pendingDir, file), "utf8"));
  // Фрагмент может быть либо {"57": {...}}, либо сразу телом порогов.
  const id = file.replace(/\.json$/, "");
  const entry = raw[id] ?? raw;

  if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
    skipped.push(`${file}: не объект`);
    continue;
  }
  if (!Array.isArray(entry.requiredWidgets) || entry.requiredWidgets.length === 0) {
    skipped.push(`${file}: пустой requiredWidgets`);
    continue;
  }

  const widget = entry.requiredWidgets[0];
  const widgetFile = path.join(root, "public/interactive/widgets", `${widget}.js`);
  if (!fs.existsSync(widgetFile)) {
    skipped.push(`${file}: виджет ${widget}.js не найден`);
    continue;
  }

  contract[id] = entry;
  merged.push(`${id} → ${widget}`);
}

fs.writeFileSync(contractPath, `${JSON.stringify(contract, null, 2)}\n`);

console.log(`слито записей: ${merged.length}`);
for (const line of merged) console.log(`  ${line}`);
if (skipped.length) {
  console.log(`\nпропущено: ${skipped.length}`);
  for (const line of skipped) console.log(`  ${line}`);
}
