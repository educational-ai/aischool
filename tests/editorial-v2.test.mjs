import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);
const contract = JSON.parse(
  await readFile(new URL("editorial/lesson-quality-v2.json", root), "utf8"),
);

function words(source) {
  return source
    .replace(/:::.*$/gm, " ")
    .replace(/[$`*_#|{}[\]():]/g, " ")
    .trim()
    .split(/\s+/)
    .filter(Boolean).length;
}

function count(source, expression) {
  return [...source.matchAll(expression)].length;
}

test("уроки, переведённые на стандарт Лагутина, не могут незаметно обеднеть", async () => {
  for (const [id, rules] of Object.entries(contract)) {
    const source = await readFile(
      new URL(`content/lessons/${id}.md`, root),
      "utf8",
    );
    const marker = "\n## Задачи\n";
    const markerIndex = source.indexOf(marker);
    assert.ok(markerIndex > 0, `${id}.md: нет самостоятельного раздела задач`);
    const body = source.slice(0, markerIndex);
    const homework = source.slice(markerIndex + marker.length);

    assert.ok(
      words(body) >= rules.minimumBodyWords,
      `${id}.md: основной текст короче редакционного минимума`,
    );
    assert.ok(
      words(homework) >= rules.minimumHomeworkWords,
      `${id}.md: условия домашней работы слишком краткие`,
    );
    assert.ok(
      count(body, /^## /gm) >= rules.minimumSections,
      `${id}.md: статья недостаточно расчленена`,
    );
    assert.ok(
      count(body, /^:{3,}figure\{/gm) >= rules.minimumFigures,
      `${id}.md: недостаточно самостоятельных рисунков`,
    );
    assert.ok(
      count(body, /^:{3,}sidenote\{/gm) >= rules.minimumSidenotes,
      `${id}.md: поля слишком редкие`,
    );
    assert.ok(
      count(body, /^:{3,}sidenote\{[^}\n]*\bsrc="/gm) >= rules.minimumSidenoteImages,
      `${id}.md: на полях не хватает изображений`,
    );
    assert.ok(
      count(body, /^:{3,}sidenote\{[^}\n]*\bmode="quote"/gm) >= rules.minimumQuoteSidenotes,
      `${id}.md: нет авторского голоса на полях`,
    );
    assert.ok(
      count(body, /^:{3,}exercise\{/gm) >= rules.minimumInlineExercises,
      `${id}.md: мало математических остановок внутри статьи`,
    );
    assert.ok(
      count(homework, /^:{3,}problem\{/gm) >= rules.minimumProblems,
      `${id}.md: домашняя работа недостаточно полна`,
    );
    assert.ok(
      count(body, /^\$\$$/gm) / 2 >= rules.minimumDisplayMathBlocks,
      `${id}.md: математика сведена к упоминаниям`,
    );

    for (const kind of rules.requiredSideCalculations) {
      assert.match(
        body,
        new RegExp(`^::sidecalc\\{[^}\\n]*kind="${kind}"`, "m"),
        `${id}.md: отсутствует вычисление на полях ${kind}`,
      );
    }
    for (const name of rules.requiredWidgets) {
      assert.match(
        body,
        new RegExp(`^:::widget\\{[^}\\n]*name="${name}"`, "m"),
        `${id}.md: отсутствует главная лаборатория ${name}`,
      );
    }
    for (const name of rules.forbiddenWidgets) {
      assert.doesNotMatch(
        body,
        new RegExp(`^:::widget\\{[^}\\n]*name="${name}"`, "m"),
        `${id}.md: остался шаблонный интерактив ${name}`,
      );
    }
  }
});
