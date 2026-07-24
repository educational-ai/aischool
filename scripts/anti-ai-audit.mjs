import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";

const lessonDir = new URL("../content/lessons/", import.meta.url);

const patterns = [
  ["является", /\bявля(?:ется|ются|лся|лась|лись)\b/giu],
  // Plural forms "данные/данных" mean data throughout this course, not the
  // bureaucratic demonstrative. Only unambiguous singular forms are markers.
  ["данный", /\bданн(?:ый|ая|ое|ого|ой|ому)\b/giu],
  ["указанный", /\bуказанн\p{L}*\b/giu],
  ["вышеупомянутый", /\bвышеупомянут\p{L}*\b/giu],
  ["обеспечивает", /\bобеспечива\p{L}*\b/giu],
  ["демонстрирует", /\bдемонстрир\p{L}*\b/giu],
  ["содействует", /\bсодейств\p{L}*\b/giu],
  ["соответствует", /\bсоответств\p{L}*\b/giu],
  ["подчёркивает", /\bподч[её]ркива\p{L}*\b/giu],
  ["символизирует", /\bсимволизир\p{L}*\b/giu],
  ["выступает в роли", /\bвыступа\p{L}*(?:\s+в\s+роли)?\b/giu],
  ["осуществляет", /\bосуществ\p{L}*\b/giu],
  ["представляет собой", /\bпредставля\p{L}*\s+собой\b/giu],
  ["затрагивает", /\bзатрагива\p{L}*\b/giu],
  ["способствует", /\bспособству\p{L}*\b/giu],
  ["погружаться", /\bпогружа\p{L}*\b/giu],
  ["гобелен", /\bгобелен\p{L}*\b/giu],
  ["переплетение", /\bпереплетени\p{L}*\b/giu],
  ["нюансы", /\bнюанс\p{L}*\b/giu],
  ["в рамках", /\bв\s+рамках\b/giu],
  ["важно отметить", /\bважно\s+отметить\b/giu],
  ["стоит отметить", /\bстоит\s+отметить\b/giu],
  ["стоит подчеркнуть", /\bстоит\s+подч[её]ркнуть\b/giu],
  ["необходимо учитывать", /\bнеобходимо\s+учитывать\b/giu],
  ["на основании вышеизложенного", /\bна\s+основании\s+вышеизложенного\b/giu],
  ["в контексте", /\bв\s+контексте\b/giu],
  ["таким образом", /\bтаким\s+образом\b/giu],
  ["в заключение следует отметить", /\bв\s+заключение\s+следует\s+отметить\b/giu],
  ["играет роль", /\bигра\p{L}*\s+(?:важную|ключевую|значительную)?\s*роль\b/giu],
  ["оказывает влияние", /\bоказыва\p{L}*\s+влияние\b/giu],
  ["носит характер", /\bноси\p{L}*\s+характер\b/giu],
  ["это не просто", /\bэто\s+не\s+просто\b/giu],
  ["не только… но и", /\bне\s+только\b[^.!?]{0,180}\bно\s+и\b/giu],
  ["давайте рассмотрим", /\bдавайте\s+рассмотрим\b/giu],
  ["конечно", /\bконечно[!,]/giu],
  ["безусловно", /\bбезусловно[!,]/giu],
  ["может похвастаться", /\bможе\p{L}*\s+похвастаться\b/giu],
  ["расположенный в самом сердце", /\bрасположенн\p{L}*\s+в\s+самом\s+сердце\b/giu],
  ["поворотный момент", /\bповоротн\p{L}*\s+момент\b/giu],
  ["служит напоминанием", /\bслужи\p{L}*\s+напоминанием\b/giu],
  ["в конце дня", /\bв\s+конце\s+дня\b/giu],
  ["за пределами коробки", /\bза\s+пределами\s+коробки\b/giu],
  ["на одной странице", /\bна\s+одной\s+странице\b/giu],
  ["уникальный", /\bуникальн(?:ый|ая|ое|ые|ого|ой|ому|ым|ыми|ых)\b/giu],
];

const allowedPerLesson = new Map([
  // These words often carry exact mathematical meaning. The skill treats
  // density, not a single justified occurrence, as the stronger tell.
  ["является", 2],
  ["соответствует", 2],
  ["обеспечивает", 1],
  ["не только… но и", 1],
]);

function stripMarkdown(source) {
  return source
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/\$\$[\s\S]*?\$\$/g, " ")
    .replace(/\$[^$\n]+\$/g, " ")
    .replace(/\[[^\]]+\]\([^)]+\)/g, (match) => match.slice(1, match.indexOf("]")))
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/^:::[^\n]*$/gm, "")
    .replace(/[*_`>#]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function lineOf(source, index) {
  return source.slice(0, index).split("\n").length;
}

const files = (await readdir(lessonDir))
  .filter((name) => /^\d{2}\.md$/.test(name))
  .sort();

const findings = [];
const sentenceOwners = new Map();
const headingOwners = new Map();

for (const file of files) {
  const source = await readFile(join(lessonDir.pathname, file), "utf8");
  const prose = stripMarkdown(source);

  for (const match of source.matchAll(/^##\s+(.+)$/gm)) {
    const heading = match[1].trim();
    if (heading === "Задачи") continue;
    const owners = headingOwners.get(heading) ?? [];
    owners.push(file);
    headingOwners.set(heading, owners);
  }

  for (const [label, pattern] of patterns) {
    // JavaScript's `\b` only understands ASCII word characters. Wrap the whole
    // lexical expression in Unicode-aware boundaries instead.
    const body = pattern.source.replaceAll("\\b", "");
    const unicodePattern = new RegExp(
      `(?<![\\p{L}\\p{N}_])(?:${body})(?![\\p{L}\\p{N}_])`,
      pattern.flags,
    );
    const matches = [...source.matchAll(unicodePattern)];
    const allowed = allowedPerLesson.get(label) ?? 0;
    for (const match of matches.slice(allowed)) {
      findings.push({
        file,
        line: lineOf(source, match.index ?? 0),
        kind: "marker",
        label,
        sample: match[0],
      });
    }
  }

  const dashCount = (prose.match(/—/g) ?? []).length;
  const dashDensity = prose.length ? dashCount * 1000 / prose.length : 0;
  if (dashDensity > 5) {
    findings.push({
      file,
      line: 1,
      kind: "dash-density",
      label: `${dashDensity.toFixed(1)} на 1000 знаков`,
      sample: `${dashCount} длинных тире`,
    });
  }

  for (const sentence of prose.split(/(?<=[.!?])\s+/u)) {
    const normalized = sentence
      .toLocaleLowerCase("ru")
      .replace(/[«»"':;,.!?()[\]{}]/g, "")
      .replace(/\s+/g, " ")
      .trim();
    if (normalized.length < 70) continue;
    const owners = sentenceOwners.get(normalized) ?? [];
    owners.push(file);
    sentenceOwners.set(normalized, owners);
  }
}

for (const [heading, owners] of headingOwners) {
  const uniqueOwners = [...new Set(owners)];
  if (uniqueOwners.length < 4) continue;
  findings.push({
    file: uniqueOwners.join(", "),
    line: 1,
    kind: "heading-template",
    label: `${uniqueOwners.length} уроков`,
    sample: heading,
  });
}

for (const [sentence, owners] of sentenceOwners) {
  const uniqueOwners = [...new Set(owners)];
  if (uniqueOwners.length < 2) continue;
  findings.push({
    file: uniqueOwners.join(", "),
    line: 1,
    kind: "repeated-sentence",
    label: `${uniqueOwners.length} урока`,
    sample: sentence.slice(0, 150),
  });
}

if (findings.length === 0) {
  console.log(`Anti-AI audit: ${files.length} файлов, 0 флагов.`);
  process.exit(0);
}

console.log(`Anti-AI audit: ${files.length} файлов, ${findings.length} флагов.\n`);
for (const finding of findings) {
  console.log(
    `[${finding.kind}] ${finding.file}:${finding.line} ${finding.label}: ${finding.sample}`,
  );
}
process.exitCode = 1;
