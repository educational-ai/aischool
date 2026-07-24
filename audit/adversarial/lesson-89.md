# Урок 89. Интерактивный ИИ, агенты и scaffolding

## Главная педагогическая идея

Agent capability belongs to the whole loop—model, state, tools, memory,
permissions and verifier. End-to-end failure compounds across steps; prompt
injection is best addressed by architectural trust boundaries, not prose.

## Что есть сейчас

1342 слова до задач, 12 разделов, 3 display-формулы, 1 определение,
0 встроенных упражнений, 4 sidenotes, 3 SVG, widget `g11-generative`
(`buildAgent`) and 5 tasks. Checked source, visuals and screenshot. Text
covers state/action loop, tool contracts, planner/executor/verifier,
compounding error, memory provenance, prompt injection, SWE-bench and stopping.

Ученик поймёт system-level view, least privilege and independent verification.
But none of the three process diagrams has visible directional arrows, and
widget is a reliability calculator, not an agent sandbox.

## Чего не хватает в рассуждении

1. Exact state machine with transition conditions, terminal states and
   invalid-action handling.
2. Reliability model with detected/undetected errors and retries, not only
   independent \(p^n\).
3. Capability lattice/authorization example read vs write vs send.
4. Counterexample verifier sharing same corrupted source.
5. Formal injection invariant: untrusted document cannot cause outbound action.
6. Budget as constrained policy with cost calculation.
7. Reproducible safe synthetic trace in article, not only huge homework.

## Рисунки и интерактив

### `figure-1.svg`

Caption calls it a state machine, but render has no visible arrowheads,
transition conditions or cycles. Boxes are merely arranged. Redraw with
states, guards, budget decrement, failure and terminal nodes.

### `figure-2.svg`

Evidence bundle/roles again lack direction; tests/parser/human are not linked
to claims they verify. Draw claim–evidence pairs and concrete feedback payload.

### `figure-3.svg`

Trust-boundary diagram is entirely arrowless; data/control/capability flows
cannot be distinguished. Use different line styles with prominent arrowheads,
policy gate and forbidden edge.

### `buildAgent`

Widget uses arbitrary independent reliability formulas and multiplies them.
Verifier catches a fixed fraction; budget does not constrain actions. There
are no tools, observations, injection text, permissions, action log,
run/step/reset. Replace with a safe finite agent simulator: goal, document
containing synthetic injection, search/read/calculate/send tools, capability
gate, planner choices, verifier evidence and trace. A user can replay same seed
with/without gate.

## Какие rich sidenotes нужны

- Norbert Wiener/cybernetics historical card and modern agent primary source;
- quote on least privilege from a security standard;
- JS reliability tree calculator;
- counterexample verifier using same wrong source;
- counterexample retry amplifying irreversible action;
- trust-boundary illustration;
- question model capability vs system capability;
- bridge back to MDP/alignment and forward to final project;
- warning never use real secrets in injection lab.

## Недостающие упражнения

1. Draw full probability tree with verifier/retry and compute outcomes.
2. Design typed read-only tool schema with four errors.
3. Prove architectural gate makes outbound attack success zero under stated
   capability model.
4. Construct correlated step failures showing \(p^n\) independence is wrong.
5. Browser-experiment seed 8905: 20 synthetic documents/injections, compare
   warning/separation/gate on task and attack success.

## План переписывания

1. Open with one complete safe task trace.
2. Formalize state machine and tool contract.
3. Compute reliability tree.
4. Separate planner/executor/verifier by evidence.
5. Rebuild widget as agent trace simulator.
6. Demonstrate injection and capability gate.
7. Add cost/stopping and reproducible artifact.
8. Reach 10 sidenotes, 4 inline exercises and 8 tasks.

## Вердикт

| Категория | Оценка |
|---|---|
| A. Глубина | FAIL |
| B. Математический ход | PARTIAL |
| C. Рисунки | FAIL |
| D. Интерактив | FAIL |
| E. Sidenotes | FAIL |
| F. Упражнения | PARTIAL |
| G. Данные | PARTIAL |
| H. Связность | PARTIAL |

**Общий вердикт: FAIL.**
