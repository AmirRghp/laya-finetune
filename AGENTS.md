# AGENTS.md

Binding rules for every agent and every human contributor working in this repository.
These rules are not advisory. A change that violates one is a defect, not a trade-off.

The project prepares fine-tuning data for **Laya Multilingual**
(`convaiinnovations/laya-multilingual`), a structured *decision* model, using the Persian
Markdown corpus in `corpus/`.

### Companion documents

These are **binding**, not background reading. They are referenced by section number
below, and a change that contradicts one is a defect.

| Document | What it fixes | Referenced by |
|---|---|---|
| [`docs/PRD.md`](docs/PRD.md) | What the project must achieve, and how we know it worked. Every functional requirement names the test that verifies it. | §6, §7, §9 |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Why each significant decision was made and what was rejected (ADR-001…ADR-015), the layering, the data model, and the 14 invariants. | §1–§6 |
| [`docs/TESTING.md`](docs/TESTING.md) | Test layers, focused commands, execution limits, and how to report results. | §7–§8 |

**When you change behaviour, update the companion document in the same commit.** A rule
here with no recorded reason is how the next contributor "fixes" it back.

**When you make an architectural decision, add an ADR to `docs/ARCHITECTURE.md`.** Number
them sequentially; never renumber or delete an existing one. If a decision is superseded,
mark the old one and point it at the new one.

---

## 1. Corpus handling — non-negotiable

1.1 `corpus/` is the **source of truth** and is **read-only** to every automated process.

1.2 Never modify, rename, move, delete, reformat, re-encode, or overwrite any file under
`corpus/`. Never write to `corpus/` for any reason, including scratch files, backups, or
temporary output.

1.3 Every corpus read opens files in binary or `encoding="utf-8"` read mode. Writes never target
a path resolved inside the corpus directory. `io.py` enforces this with an assertion, not a
convention.

1.4 `corpus/` is **not committed to Git**. `.gitignore` excludes it. Generated datasets that may
contain sensitive derived text are excluded too.

1.5 Document discovery is **recursive** over `*.md` and `*.markdown`, with extensions compared
case-insensitively.

1.6 Discovery order is **deterministic**: POSIX-relative path, sorted, with a stable Unicode
code-point comparison. Re-running discovery on an unchanged corpus yields an identical list.

1.7 **No filename assumptions.** The pipeline must never branch on a filename, prefix, numeric
separator, numbering scheme, extension-derived category, or path depth. Tests
(`test_corpus_discovery.py::test_filename_independence`) pin this by ingesting the same
documents under scrambled names and asserting identical chunk content and stable grouping.

1.8 The document collection is **not fixed**. Adding, removing or renaming documents under
`corpus/` must work without a code change. Incremental processing keys off content hashes, not
filenames, so a renamed file that keeps its content is not reprocessed.

---

## 2. Text handling

2.1 Persisted text is preserved exactly as read, apart from: BOM removal, CRLF→LF line-ending
normalisation, and Unicode NFC normalisation. Nothing else.

2.2 Aggressive cleaning — case folding, stripping punctuation, removing ZWNJ, transliteration,
collapsing whitespace, removing digits — is **forbidden** on stored text. Persian meaning,
punctuation, Persian and Arabic digits, dates and technical terminology are load-bearing.

2.3 Comparison-time folding (Arabic `ي`→`ی`, `ك`→`ک`, diacritic removal, zero-width removal)
is permitted **only** inside hashing, deduplication and similarity functions, and must be
applied to a derived value, never written back to the document or chunk text.

2.4 Tokenisation is Unicode-aware. Persian ZWNJ (`U+200C`) separates words and must not be
treated as whitespace during tokenisation.

2.5 Markdown must not be split at arbitrary character offsets when that would destroy meaning.
Chunk boundaries fall between structural elements only.

2.6 Structural elements — headings, fenced code blocks, list items, tables, blockquotes — are
preserved whole. A table is never split between its header and its body. A fenced code block is
never split inside the fence.

---

## 3. Laya API conformance

3.1 Use only the **official** Laya interfaces. No invented APIs, flags or options.

3.2 Fine-tuning entry points are `laya.train.finetune(...)` and `laya.train.dry_run(...)`
imported as `from laya.train import ...`. `laya.train` is a module, not a package.

3.3 The `laya-train` **CLI is not sufficient** for this project: it cannot express `amp`,
`gradient_checkpointing`, `calib_max`, `calib_frac` or `calib_seed`. Shelling out to it would
silently apply defaults the operator did not choose. Training goes through Python. → **ADR-009**

3.4 Training rows are exactly `{state, questions, expected}` (`choice`/`noul`/`score`), plus
optional `tags` and `language`, which `laya.evals` reads and the trainer ignores.

3.5 One row carries exactly one decision. `items_from_rows` discards every question on a row
whose `state` is empty, so a row with several chunks and one label is unrepresentable. Each
`(query, chunk)` pair is its own row. → **ADR-001**

3.6 Question `instructions` is mandatory and non-empty; `criteria` for a `choice` question is a
non-empty dict or list. Validate before writing, never after.

3.7 The question definition has one source of truth — `laya.questions` in the config. Builder,
validator and evaluation all read it from there. A question schema that differs between
training and inference is a defect. → **ADR-011**

3.8 The `state` is rendered from a config-driven Persian labelled template. The rendered form is
fingerprinted and compared between build time and evaluation time.

3.9 Pass `eval_data` for every real run. Without it Laya reports in-sample calibration fit
numbers and says so; those numbers are not held-out results and must not be presented as such.

3.10 Never reuse calibration data for model fitting. Validation is the calibration split.

3.11 Laya is a structured decision model, not a generative LM. SFT, QLoRA, TRL, causal-LM
cross-entropy on raw text, and any "prompt/completion" framing are **out of scope**.

3.12 `laya-multilingual` ships with `temperature: [1,1,1]` and empty
`temperature_by_options` — uncalibrated. Never gate on `min_confidence` or present a
confidence number as calibrated until temperatures have been refit and validated on held-out
data. Say so in every report that prints confidence.

3.13 The optimiser-update budget is
`epochs * ceil(ceil(train_items / micro_batch) / grad_accum)`. Print it before any run that
costs GPU time. A small dataset needs more epochs to reach a comparable update count. → **ADR-010**

3.14 Never fabricate a benchmark number. A metric that was not measured is absent from the
report, never zero and never estimated.

---

## 4. Determinism and reproducibility

4.1 One configured seed, threaded explicitly through every stage that samples, shuffles or
splits. No module-level `random`, no unseeded `random.shuffle`, no unseeded `np.random`.

4.2 Idempotence: re-running a stage on unchanged inputs produces byte-identical outputs.

4.3 Every artifact write is atomic — write to a sibling temp file, `os.replace`. A crashed run
never leaves a half-written dataset that the next run will trust. → **ADR-015**

4.4 Every artifact directory carries a `manifest.json` with input hashes, config hash, seed,
counts and stage version, so a result can be traced to the exact inputs that produced it.

4.5 Caches are content-addressed. A cache hit must produce the same value a fresh call would.

---

## 5. Code standards

5.1 Full type hints on every public function and every module-level constant.

5.2 Structured logging via the `logging` module with explicit named fields — never bare
`print()` for pipeline output, never `print()` for diagnostics in library code.

5.3 Exceptions are typed, defined in `errors.py`, and raised with a message naming the offending
path, id or value. Never swallow an exception silently; catch only to add context or to retry.

5.4 Configuration is dataclasses loaded from YAML with explicit validation. No hardcoded
absolute paths, credentials, API tokens or model endpoints anywhere in `src/`. Endpoints come
from config or environment.

5.5 Config defaults live in `configs/default.yaml`. Code holds only structural defaults and
raises on invalid values rather than silently repairing them.

5.6 Modules are small and single-purpose. `corpus/`, `tasks/`, `dataset/` and `evaluation/`
are packages with `__init__.py`; `chunking.py`, `retrieval.py`, `negatives.py`, `labeling.py`,
`training.py` and `io.py` are modules.

5.7 Prefer the standard library. Justified dependencies only: `pyyaml`, `httpx`, `numpy`, and
`pytest` for development. `laya` and `torch` are optional extras imported lazily inside
functions, never at module import time, so the data pipeline runs without them.

7.8 No databases, Docker orchestration, web dashboards, frontend frameworks, or service
infrastructure. This is a batch pipeline over files.

---

## 6. Dataset correctness rules

6.1 **Never automatically label a non-source chunk as irrelevant.** A negative must pass an
evidence check. Absence of the source document is not evidence of irrelevance. → **ADR-008**,
the four gates

6.2 Ambiguity is a discard, not a coin flip. Candidates answered *uncertain* by the teacher are
dropped, not assigned a label. → **ADR-007**

6.3 Never create a near-duplicate contradiction — a positive and a negative sharing content,
or a negative containing a key-point answer string.

6.4 Positive generation must not invent facts absent from the document. `key_points` are
extracted verbatim-grounded; the verbatim source span is stored alongside each one.

6.5 Sufficiency is never inferred from semantic similarity. It is judged against coverage of
the `key_points` by the retrieved set. → **ADR-002**

6.6 Routing labels derive from query form. The labeling teacher never receives the source
document's identity or domain. → **ADR-003**

6.7 Split **before** mining negatives, and mine within each split's own chunk pool, so no test
document's text can reach a training row. → **ADR-004**

6.8 Group splitting is by source document and by near-duplicate document family. No document
content, and no paraphrased query, may appear in two splits.

6.9 Prefer fewer correct examples over more unreliable ones. Volume is not a quality metric.

6.10 Human-reviewed labels always take precedence over automatically proposed ones. Both the
proposal and the accepted label are preserved.

6.11 Teacher confidence is not correctness. Confidence is recorded; it is never used as a
proxy for label validity.

---

## 7. Testing

7.1 Tests use small synthetic Persian fixtures written for the test suite. Never a production
corpus document, never real generated data.

7.2 Tests must not require network access, GPU access, or Laya weights. Every external model
call — Ollama generation, Ollama embedding, teacher labeling, Laya inference — is mocked.

7.3 External model calls go through a single client seam so a test can substitute a fake
transport without patching call sites. → **ADR-012**

7.4 Every module with logic gets tests covering its failure modes, not only its happy path:
empty input, malformed input, empty results, and threshold boundaries.

7.5 Naming independence, Persian handling, duplicate detection, split leakage and config
validation each have an explicit named test.

7.6 Do not weaken a test to make it pass. Removing an assertion, adding a skip, or loosening a
threshold to accommodate new behaviour is a regression in the suite.

---

## 8. Execution restrictions for agents

**Agents may implement and run focused, local tests. They do not run expensive model work.**

8.1 Agents must **not**: install dependencies, invoke Ollama, download model weights,
generate production datasets, start fine-tuning (including `train --yes`), start real
checkpoint evaluation jobs, push to GitHub, or modify any file under `corpus/`. Agents may
run focused `pytest` commands that use only the existing environment and synthetic,
mocked fixtures required by §7. A full test suite is reserved for broad shared-contract
changes, release work, or an explicit user request, as described in `docs/TESTING.md`.

8.2 Reading files, static inspection, and focused local tests are permitted. Reading
documentation over HTTP is permitted. Tests must not contact external services or load
model weights.

8.3 Never report a test as passing, a number as measured, or a run as successful. Only report
what was actually executed. Everything else is described as unverified.

8.4 Before declaring work complete, perform a static review pass for: missing imports, broken
module paths, interface mismatches between modules, schema inconsistencies, unguarded `None`
access, and unreachable error handling. Report what could not be verified by static review
or focused tests.

8.5 Never create placeholder implementations, fake results, stubbed-out features or `TODO`
markers in place of required behaviour. Required behaviour ships implemented or not at all.

8.6 Use the project-selected skills in [`docs/AGENT_SKILLS.md`](docs/AGENT_SKILLS.md).
Complete local copies live under `agent_skills/`. Select only the skill(s) relevant to the
current task and apply a selected skill when its trigger matches; AGENT_SKILLS.md ranks
them by project fit. User instructions, this file, the PRD and Architecture take precedence
over generic skill instructions. Do not load every skill into context or paste whole skill
files into a prompt; read the relevant local file just in time and keep the active context
to the task, the governing rules, and the needed source/test files.

8.7 Keep verification proportional to the change. For code or behavior changes, run the
smallest relevant focused test(s) once after the final edit when the existing environment
supports them; report the exact command and observed result. Do not install dependencies to
run tests. If unavailable, report the command as unrun. Documentation-only changes do not
need code tests. Run the full suite only for broad cross-stage or shared-contract changes,
release work, or an explicit user request. Never run main fine-tuning or real model jobs as
verification. Never weaken, skip, or delete a test to save time (§7.6).

---

## 9. Documentation

9.1 Every pipeline stage has a CLI command, documented in the README with its inputs, outputs,
runtime cost and whether it needs Ollama or Laya weights.

9.2 The README distinguishes four command classes explicitly: dataset preparation, commands
that call Ollama, commands that load Laya weights, commands that consume GPU time for training.

9.3 Document the manual execution order from a clean environment, flagging which steps are slow.

9.4 Record known limitations and unverified areas in the README rather than leaving them
implicit.

9.5 A non-obvious design decision gets a short note where it lives, explaining *why*.

9.6 [`docs/PRD.md`](docs/PRD.md), [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), and
[`docs/TESTING.md`](docs/TESTING.md) are **binding**, not background. Read the relevant
sections before changing behavior or choosing verification. A change that contradicts them
is a defect.

9.7 **A behavioural change updates the companion document in the same commit.** If a
rule here has no recorded reason, the next contributor will "fix" it back.

9.8 **A new architectural decision gets an ADR** in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), numbered sequentially. State the
context, the decision, the alternatives considered, and why each was rejected. Never
renumber or delete an existing ADR; supersede it and point the old one at the new one.

9.9 Every functional requirement in the PRD names the test that verifies it. A new
requirement without a named test is an unverified intention — either write the test or
move it to Open Questions.

### Rule-to-document map

Use this to find the reasoning behind a rule without reading the whole repository.

| Rule | Rationale |
|---|---|
| 1.7 no filename assumptions | ADR-005 · ARCHITECTURE §4 (content-derived identifiers) |
| 2.2 no aggressive cleaning | ARCHITECTURE §12 invariant 3 |
| 2.4 ZWNJ separates tokens | ARCHITECTURE §2 (`ids.py`) |
| 3.3 Python API, not the CLI | **ADR-009** |
| 3.5 one decision per row | **ADR-001** |
| 3.7 one question definition | **ADR-011** |
| 3.9 always pass `eval_data` | ADR-010 (print the budget, then prove it) |
| 3.13 print the update budget | **ADR-010** |
| 3.14 positive control | ARCHITECTURE §10 (cost of too few updates) |
| 4.1 one explicit seed | ARCHITECTURE §12 invariant 14 |
| 4.3 atomic writes | **ADR-015** |
| 4.4 manifest per artifact | **ADR-015** |
| 4.5 content-addressed caches | **ADR-015** |
| 5.5 defaults in YAML | ARCHITECTURE §7 (precedence chain) |
| 6.1 never auto-label irrelevant | **ADR-008** (four gates) |
| 6.2 ambiguity is a discard | **ADR-007** |
| 6.5 sufficiency ≠ similarity | **ADR-002** |
| 6.6 routing from query form | **ADR-003** |
| 6.7 split before mining | **ADR-004** |
| 6.8 group split by family | ADR-005 · ADR-006 (four leakage classes) |
| 6.11 confidence ≠ correctness | **ADR-007** |
| 7.3 one transport seam | **ADR-012** |
| 8.1 focused tests allowed; model work stays operator-run | `docs/TESTING.md` §4; ADR-012 defines the mock seam |
| 9.6–9.9 companion documents | this section |
