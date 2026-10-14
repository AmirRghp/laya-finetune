# Architecture

How this pipeline is put together, and — more usefully — **why each significant
decision was made and what was rejected**.

Code shows what was built. This document explains what else was considered.

| | |
|---|---|
| **Scope** | `src/laya_pipeline/` — 11,220 lines, 40 modules |
| **Binding rules** | [`AGENTS.md`](../AGENTS.md) |
| **Requirements** | [`PRD.md`](PRD.md) |
| **Testing** | [`TESTING.md`](TESTING.md) |
| **Runbook** | [README](../README.md#the-complete-runbook) |

---

## 1. Shape

```
corpus/*.md
   │
   ├─ scan ──────── discovery.py · markdown.py · models.py
   │                 recursive, filename-blind, deterministic order
   │
   ├─ process ───── chunking.py · tokenizer.py
   │                 structure-preserving; never splits a block
   │
   ├─ generate ──── tasks/relevance.py ─┐
   │                 questions + key points, then teacher verification
   │
   ├─ split ─────── dataset/split.py
   │                 document + family grouping; FOUR leakage checks
   │
   ├─ negatives ─── negatives.py · retrieval.py
   │                 3 strategies → 4 gates, inside the split's own pool
   │
   ├─ label ─────── labeling.py
   │                 teacher labels, contradictions, review queue
   │
   ├─ tasks ─────── tasks/sufficiency.py · tasks/routing.py
   │
   └─ build ─────── dataset/builder.py · validate.py · stats.py
                    Laya JSONL + structural + token-length validation
                       │
                       ▼
                    training.py ──► laya.train.finetune
                       │
                       ▼
                    evaluation/ ──► laya.evals
```

Everything above the `──►` is a **batch pass over files**. Nothing streams, nothing
serves, nothing keeps state between runs except content-addressed caches on disk
(AGENTS 7.8).

## 2. Layering

Dependencies point one way. A module never reaches upward.

```
cli.py           argparse only. No logic. Exits with a code.
   │
stages.py        orchestration: each stage reads the previous artifact
   │
   ├── corpus/        discovery → markdown → models
   ├── chunking.py    tokenizer.py    ids.py
   ├── llm/           ollama.py · prompts.py · schema.py · cache.py
   ├── tasks/         relevance · sufficiency · routing · base(Teacher)
   ├── negatives.py   retrieval.py    labeling.py
   ├── dataset/       records · split · builder · validate · stats
   ├── training.py    evaluation/
   │
   └── io.py  config.py  errors.py  logging_setup.py      ← leaf, no dependents
```

`io.py`, `config.py`, `errors.py` and `logging_setup.py` are leaves. They know nothing
about stages, tasks or Laya. That is what makes them testable in isolation and why the
corpus write-protection can be enforced in one place.

## 3. Decisions worth the explanation

### ADR-001: One `(query, chunk)` pair per row

**Context.** Laya's `items_from_rows` **discards every question on a row whose `state`
is empty**. A naive row — one chunk, three questions — would silently lose two.

**Decision.** One pair per row. A row with several chunks and one label is
*unrepresentable*, not merely awkward.

**Rejected.** Concatenating chunks into one state: it conflates "these passages
together are insufficient" with "this passage is irrelevant", which are different
tasks with different labels.

**Cost.** More rows, longer training set. Accepted.

### ADR-002: Sufficiency is judged against key points, never similarity

**Context.** The tempting shortcut is "if the top-k chunks are semantically close to
the query, they're sufficient". It is wrong in the most common Persian failure mode:
three paraphrases of the same policy clause, none of which contains the number the user
asked for.

**Decision.** Every generated question carries `key_points` — the facts a correct
answer must contain, each with its verbatim source span. Sufficiency is the coverage of
those points by the retrieved set.

**Consequence.** Sufficiency cannot be produced for a query whose key points were never
extracted. That is the intended failure: it is discarded, not guessed.

### ADR-003: Routing labels never see the source document

**Context.** The routing teacher could easily be handed the query *and* the document it
came from. Then it learns organisational domains — "HR queries are semantic" — and
degrades to a document classifier the moment a new domain appears.

**Decision.** The routing prompt receives query text and nothing else. No title, no
heading path, no source id.

**Verified by.** The prompt template has no placeholder for them, so passing one is a
structural error, not a discipline problem.

### ADR-004: Split before mining negatives

**Context.** Mining negatives first, then splitting, lets a test document's text reach a
training row as a *negative*. No query overlap is needed — the chunk text itself leaks.

**Decision.** `stages.stage_negatives` runs after `stage_split` and receives only the
split's own chunk pool (`_pool_for_split`). Stage order in `stages.py` *is* the
guarantee.

**Cost.** Smaller pools per split, so validation and test cannot fill the candidate
budget (measured: 69 unfilled negatives). Correctness over yield.

### ADR-005: Group splits by near-duplicate family

**Context.** Document-level splitting leaks when two documents are 95% the same. One
lands in train, one in test, and test is no longer held out.

**Decision.** Documents whose normalised content is ≥ `family_similarity_threshold`
shingle-Jaccard are merged into one family and land in the same split.

**Cost.** O(n²) over documents. Trivial at 20; needs an inverted index at tens of
thousands (known limitation 9).

### ADR-006: Four leakage classes, all reported

**Context.** "We checked for leakage" is worthless without saying what was checked.

**Decision.** Four classes, each named in the report:

| Kind | Meaning |
|---|---|
| `identical_decision` | the same decision appears in two splits |
| `source_document_split` | one document's chunks span two splits |
| `duplicate_query` | a normalised-identical query in two splits |
| `paraphrased_query` | queries above `query_leakage_threshold` shingle-Jaccard |

`leakage` must be `[]`. A non-empty list is a defect and must block acceptance of the
split and downstream dataset; emitting a warning while continuing is insufficient.

### ADR-007: Ambiguity is a discard

**Context.** A teacher asked "does this answer the query?" will sometimes be unsure.
Assigning `irrelevant` when the truth is "it does, partially" injects a wrong label;
assigning `relevant` injects a false positive.

**Decision.** `uncertain` is a third option offered to the teacher, and an uncertain
verdict **discards** the example. Teacher confidence is recorded and **never used as a
proxy for label validity** (AGENTS 6.11).

**Cost.** A weak teacher produces a small dataset rather than a wrong one. This is the
single most important failure mode in the whole project: a quiet pipeline is safe, a
confident wrong one is not.

### ADR-008: Four gates on every negative

**Context.** "This chunk is not from the source document, so it is irrelevant" is
false reasoning. An unrelated chunk *can* answer a query.

**Decision.** Four gates, in order:

1. **not the source chunk** — the weakest possible reason.
2. **not a near-duplicate of it** (shingle Jaccard ≥ `near_duplicate_threshold`).
3. **contains no key-point answer string** — checked as both the verbatim `span` and
   the `text`, because a paraphrase of the answer still leaks.
4. **the teacher says it does not answer the query** — the only gate that can say
   something about the *query* rather than about provenance.

If teacher verification is disabled, negative mining is skipped and no negative
rows are emitted; an unverified candidate is never promoted to a training label.

Everything rejected is written out with its reason, so the gates are auditable rather
than trusted.

**Cost.** Measured: 1,845 of 2,635 candidates rejected at gate 4. Twelve teacher calls
per negative kept. That is the pipeline's dominant cost and it is the correct price.

### ADR-009: Training through the Python API, not the CLI

**Context.** `laya-train` is the documented entry point, but it cannot express `amp`,
`gradient_checkpointing`, `calib_max`, `calib_frac` or `calib_seed`.

**Decision.** `training.py` calls `laya.train.dry_run` and `laya.train.finetune`
directly, so every `TrainConfig` field is reachable.

**Consequence.** Shelling out would have silently applied defaults the operator did not
choose — the most expensive kind of surprise, because the run *succeeds*.

### ADR-010: `train` prints the budget and refuses without `--yes`

**Context.** The optimizer-update budget is
`epochs × ceil(ceil(train_items / micro_batch) / grad_accum)`. A small dataset needs
**more** epochs than a large one to reach a comparable update count. A run that never
rises above the class prior is the classic symptom of too few updates.

**Decision.** Every `train` invocation prints the resolved plan and the budget, then
stops with exit 2 unless `--yes` is passed. The budget is printed **before** the model
loads, so reading it costs nothing.

### ADR-011: `laya.questions` is the single source of truth

**Context.** The dataset builder, the validator and the evaluation harness all need the
same question definitions. Any drift between them means the model is trained on wording
inference will never see — the most common way a fine-tune silently fails.

**Decision.** One definition in `configs/default.yaml`, read by all three. `state`
rendering is fingerprinted at build time and compared at evaluation time; `validate`
reports reworded instructions and changed option text as errors.

`label_roles` names which option is positive/negative so label *strings* live next to
the question instead of scattered through code.

### ADR-012: One transport seam for every model call

**Context.** Ollama calls appear in generation, evidence checking, labeling, retrieval
and embedding. Mocking them individually is the classic source of "passes in CI, fails
with the real API".

**Decision.** `llm/ollama.py` exposes `OllamaTransport` with `get` and `post`. Every
call goes through it. `tests/conftest.py::FakeTransport` implements both verbs and
**enforces** that `/api/tags` is GET-only — because a fake that accepts any verb hides
a wrong verb, which is exactly how `is_available()` came to report every reachable
server as unreachable for the entire life of the suite.

### ADR-013: Reasoning off by default

**Context.** Measured on the development machine, `qwen3.5:2b` with `think` enabled
spent **4060 eval tokens** on a one-line Persian question and returned
`content: ""` with `done_reason: "length"`. The pipeline reported that as
`model response contains no JSON object: ''` after three attempts and about five
minutes — accurate, but naming a parsing problem when the cause was a config flag.

**Decision.** `ollama.think` defaults to `false` and is sent on every generation call.
An empty reply now reports *which* fault occurred — reasoning exhausted, or context
exhausted — and a reasoning-exhausted reply is **not retried**, because the same budget
will be spent the same way.

### ADR-014: `preflight` is step 0

**Context.** A mistyped model name does not fail when the config loads. It fails on the
first API call, inside the retry loop, ~100 s later — repeated for every chunk.

**Decision.** `laya-pipeline preflight` compares the resolved config against
`/api/tags` in one second and exits non-zero on a blocking finding, so it can gate a
script. `scripts/check_env.sh` calls the same command rather than growing a second
implementation that can drift.

### ADR-015: Atomic writes and content-addressed caches

**Context.** A run killed mid-write must not leave a truncated dataset that the next run
trusts. And a run killed at 60% must not re-bill the model for the first 60%.

**Decision.**
- `io.py` writes to a sibling temp file and `os.replace`s it.
- `llm/cache.py` keys on the exact request: endpoint, model, prompt, system,
  temperature, top-p, seed, schema. A hit equals a fresh call.
- Every artifact directory carries `manifest.json` with input hashes, config hash,
  seed, counts and stage version.

**Cost.** Disk. Measured: 2,635 cache entries for one `negatives` run. Worth it — that
cache is what makes the stage resumable.

## 4. Data model

```python
# corpus/models.py
Block(kind, text, line_start)              # heading | paragraph | list_item | table
HeadingRef(level, text, anchor)
DocumentSource(path, relpath, content_id, byte_size, text)
ParsedDocument(source, frontmatter, blocks)
Chunk(chunk_id, source_id, content_id, ordinal, text, body,
      heading_path, block_kinds, token_count, oversized, metadata)

# dataset/records.py
KeyPoint(text, span, index)                # span is verbatim from the source
ChunkRef(chunk_id, text)
GeneratedExample(example_id, task, query, query_type, source_chunk_id, key_points)
LabeledExample(example_id, task, query, chunks, key_points, label, role,
               split, label_source, strategy, metadata)
```

`LabeledExample.role` is `positive` or `negative`; `label_source` is `human`,
`teacher` or `oracle`. Both are validated on construction — an example with an unknown
label source or an unknown split is refused, not defaulted.

**Identifiers are content-derived.** `content_id` hashes normalised content, so renaming
a file keeps its id and a renamed file is not reprocessed. `chunk_id` derives from
`(source, ordinal, content)`. No identifier depends on a path (AGENTS 1.7, 1.8).

## 5. The Laya row contract

What `dataset/builder.py` emits, and nothing else:

```json
{
  "state": "پرسش کاربر:\n...\n\nمتن بازیابی‌شده:\n...",
  "questions": {
    "relevance": {
      "type": "choice",
      "instructions": "آیا متن بازیابی‌شده برای پاسخ به پرسش کاربر مرتبط است؟",
      "criteria": {
        "relevant": "متن بازیابی‌شده اطلاعات لازم برای پاسخ به پرسش کاربر را در بر دارد.",
        "irrelevant": "متن بازیابی‌شده دربارهٔ موضوعی متفاوت است و به پرسش کاربر پاسخ نمی‌دهد."
      }
    }
  },
  "expected": { "relevance": "relevant" },
  "tags": ["task:relevance", "role:positive", "split:train", "source:teacher"],
  "language": "fa"
}
```

`tags` and `language` are read by `laya.evals` and ignored by the trainer. `criteria`
order is preserved on write — `write_jsonl` does **not** sort keys, because sorting
reorders `criteria` and silently rewrites the decision space of every choice question
while remaining schema-valid. That was a real defect, caught in review.

## 6. Stage contracts

| Stage | Reads | Writes | Cost |
|---|---|---|---|
| `preflight` | config | — | 1 GET |
| `scan` | `corpus/` | `scan/` | local |
| `process` | `scan/` | `chunks/` | local |
| `generate` | `chunks/` | `generated/` | ollama |
| `split` | `generated/` | `split/` (leakage blocks publication and every downstream consumer) | local |
| `negatives` | `split/` + `chunks/` | `negatives/` | ollama |
| `label` | `negatives/` | `labels/`, `review/` | ollama |
| `tasks` | `split/` + `chunks/` | `tasks/<task>/` | ollama |
| `build` | `labels/` + `tasks/` | `dataset/<task>/` | local |
| `validate` | `dataset/` | `validation.json` | local / weights |
| `stats` | `dataset/` | `stats.{json,md}` | local |
| `train` | `dataset/` | `checkpoints/` | GPU |
| `evaluate` | `dataset/` | `evaluation/` | weights |

Each stage reads the previous artifact rather than shared memory, so any stage is
independently invokable and the pipeline resumes from any point.
Generation resume additionally requires matching config, chunk-content and prompt
fingerprints; stale or legacy artifacts require an explicit `--no-resume` rebuild.
Every stage manifest records hashes for its inputs and the installed stage version.

## 7. Configuration

`configs/default.yaml` is the source of truth. 12 sections, all dataclass-backed with
explicit validation. `code holds only structural defaults` (AGENTS 5.5) — unknown keys
are errors, not warnings, so a typo like `question_per_chunk` cannot silently take a
default.

Precedence, lowest to highest:

```
dataclass default  →  configs/default.yaml  →  .env  →  real environment  →  --set
```

`.env` is read at the repo root (or one level up). A **real environment variable always
wins**, so a one-off override never requires editing a file:

```bash
LAYA_OLLAMA_TEACHER_MODEL=qwen2.5:14b laya-pipeline label   # this run only
```

Values are coerced to the declared dataclass type. An uncoercible value
(`LAYA_OLLAMA_NUM_CTX=soon`) raises at load time rather than becoming a string that
surfaces later as a confusing model failure.

## 8. Error taxonomy

20 typed exceptions in `errors.py`, all under `PipelineError`. The distinction that
matters:

```
RetryableError ── OllamaError ──┬── OllamaUnavailable   (server down: retry useless)
                               ├── OllamaTimeout       (retry with backoff)
                               └── OllamaResponseError (retry only if marked)
```

The retry policy reads `retryable`, so a schema violation does not consume three
attempts and a timeout does not give up after one. `LeakageError` extends `SplitError`
so a leak can be caught specifically without swallowing other split failures.

## 9. Testing strategy

The test suite uses synthetic Persian fixtures and mocked external calls. Focused pytest
commands are the default feedback loop; the full suite is reserved for broad shared-contract
changes, release work, or an explicit request. Agents may run short local tests but never
main fine-tuning, real Ollama jobs, production generation, or real checkpoint evaluation.
The operational commands and result-recording rules live in [`TESTING.md`](TESTING.md).

| Level | What it covers |
|---|---|
| Unit | `ids`, `io`, `config`, `errors`, `schema`, `cache`, `markdown` |
| Contract | Filename independence, determinism, idempotence, atomicity, split and row invariants |
| Mocked integration | Stage composition through `FakeTransport` |
| Regression | Review findings map to focused tests as they are addressed |

Rules that shape the suite:

- **No real corpus document.** Synthetic Persian fixtures only (AGENTS 7.1).
- **Every external call mocked** through the `FakeTransport` seam (AGENTS 7.2, 7.3).
- **Failure modes, not just happy paths.** Empty input, malformed input, empty results,
  threshold boundaries (AGENTS 7.4).
- **Hermetic by construction.** An autouse fixture runs each test outside the repo root
  so a developer's `.env` cannot leak in.
- **Never weakened.** Removing an assertion or loosening a threshold to make a test pass
  is a regression in the suite (AGENTS 7.6).

## 10. Performance

The previous document revision recorded one operator-run dataset pipeline on a development
machine with 20 documents, 80 chunks, and an RTX 3050 Laptop (CPU for data stages). The
operator recalls that the run was likely theirs. Current saved artifacts contain aggregate
counts but not stage timings, environment capture, or complete manifests, so the timings
below are historical operator-reported values, not independently verified measurements.

| Stage | Calls | Wall clock |
|---|---|---|
| `generate` | 320 (historical report) | 15 min (historical report) |
| `negatives` | 2,635 (historical report) | 85 min (historical report) |
| **total dataset** | ~3,700 (historical report) | **~2.5 h (historical report)** |

The model is the cost, not the pipeline. Two levers, in order of preference:

| Lever | Effect | Trade-off |
|---|---|---|
| `negatives.candidates_per_positive` 12 → 4 | ~3× faster | more positives fall short of `ratio` |
| `negatives.ratio` 1 → 2 | more efficient per negative | label balance skews toward `irrelevant` |
| `generation.questions_per_chunk` 3 → 2 | ~25% faster | smaller dataset |

The `negatives` stage verifies **every** surviving candidate and only then truncates to
`ratio`. That ordering is deliberate — you want the best surviving negative, and you
cannot know which of twelve will survive without checking — but it means the stage costs
12 teacher calls per negative kept. That is the pipeline's single dominant cost.

## 11. Extension points

| To add | Touch | Do not touch |
|---|---|---|
| A task | `tasks/`, `laya.questions`, `dataset/builder.py` | The Laya row contract |
| A negative strategy | `negatives.STRATEGIES` + `_draw` | The four gates |
| A model backend | `llm/` behind `OllamaTransport` | Call sites (the seam is the point) |
| A metric | `evaluation/report.py` | The "unmeasured means absent" rule |
| A chunker | `chunking.py` | Structural-element preservation |

Anything that changes `laya.questions` or `state_templates` invalidates every built
dataset. Rebuild and re-validate.

## 12. Invariants

If any of these is false, the pipeline is defective — regardless of what the metrics say:

1. `corpus/` is never written to.
2. No behaviour depends on a filename, prefix, numbering or path depth.
3. Stored text is byte-preserved apart from BOM, line endings, NFC.
4. Folding never touches stored text.
5. An uncertain verdict discards; it never assigns.
6. A non-source chunk is never auto-labelled irrelevant.
7. Splitting precedes mining; mining never crosses a split boundary.
8. `split.json.leakage` is empty; a non-empty leakage report blocks acceptance and build.
9. One `(query, chunk)` pair per row.
10. Question definitions come from one place.
11. An unmeasured metric is absent, never zero.
12. A confidence number is never presented as calibrated before temperatures are refit.
13. A crashed run leaves no artifact the next run will trust.
14. Re-running an unchanged stage produces byte-identical output.
