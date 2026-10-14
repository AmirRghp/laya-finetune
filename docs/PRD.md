# PRD — Persian RAG Decision-Model Pipeline for Laya Multilingual

| | |
|---|---|
| **Status** | Implemented; dataset stages verified against real software, training not yet run |
| **Version** | 1.0 |
| **Owner** | Amirreza Gholipour |
| **Last updated** | 2026-10-14 |
| **Binding rules** | [`AGENTS.md`](../AGENTS.md) — a violation is a defect, not a trade-off |
| **Architecture** | [`ARCHITECTURE.md`](ARCHITECTURE.md) |

---

## 1. Problem

Laya Multilingual (`convaiinnovations/laya-multilingual`) is a non-autoregressive
**decision** model. It does not generate text; in a single forward pass it answers typed
questions — `choice`, `noul`, `score` — about a state. A Persian retrieval-augmented
pipeline needs three of those decisions, and none of them is taught by the base
checkpoint for Persian organisational documents:

| Task | Decision | Input | Options |
|---|---|---|---|
| **relevance** (primary) | Does this retrieved passage answer the query? | query + one chunk | `relevant` / `irrelevant` |
| **sufficiency** | Do the retrieved passages together support a complete answer? | query + N chunks | `sufficient` / `partial` / `insufficient` |
| **routing** | Which retrieval strategy fits this query? | query alone | `semantic` / `keyword` / `hybrid` |

The failure this project exists to prevent: a Persian RAG system retrieves five
lexically similar passages, two of which do not contain the answer, and confidently
answers from them. Every decision model here is trained to catch exactly that.

## 2. Objective

Produce a reproducible, auditable, leak-free fine-tuning dataset for the three tasks
above from an arbitrary collection of Persian Markdown documents, fine-tune the base
checkpoint on it, and report an honest comparison against the untouched base.

**Success is:** a held-out evaluation that a reviewer can trust — one where every
number was measured, every claim is traceable to an input hash, and no test document's
text influenced a training row.

### Non-goals

| Not doing | Why |
|---|---|
| SFT, QLoRA, TRL, causal-LM next-token training | Laya is a decision model, not a generative LM (AGENTS 3.11) |
| Serving, APIs, databases, dashboards | Batch pipeline over files (AGENTS 7.8) |
| A Persian stemmer or morphological analyser | A wrong light stemmer costs more recall than it buys; lexical overlap is used as a conservative gate, never as the arbiter |
| Automatic labelling without human review | Only *proposals* are automatic; review is offered, not assumed (AGENTS 6.10) |

## 3. Users

| User | Need |
|---|---|
| **ML engineer** (primary) | run the pipeline, tune it, read the evaluation report |
| **Reviewer** | judge whether the labels are trustworthy, correct them in a CSV |
| **Future maintainer / agent** | understand *why* a rule exists before changing it |

## 4. Assumptions

Stated explicitly rather than silently resolved. Each is falsifiable by the person
running the pipeline.

1. **Corpus is Persian Markdown**, any filenames, any directory depth. Not fixed.
2. **`corpus/` is the source of truth and is never written to**, by any process.
3. **A local Ollama server** is available. Three models: generation, teacher, embedding.
4. **Training uses only the official `laya.train` Python API.** No invented flags.
5. **The checkpoint ships uncalibrated** — `temperature [1,1,1]`, no fitted
   `temperature_by_options`. Confidence is never presented as calibrated until refit.
6. **~20 documents is enough to prove the pipeline**, not enough to prove the model.
   Per-split metrics on a corpus this size carry a very wide confidence interval, and
   the pipeline says so rather than reporting a clean-looking number.

## 5. Functional requirements

Each requirement is testable and maps to at least one named test (AGENTS 7.5).

### FR-1 Corpus ingestion

| # | Requirement | Verified by |
|---|---|---|
| FR-1.1 | Recursive discovery of `*.md` / `*.markdown`, case-insensitive | `test_markdown.py`, `test_corpus_discovery.py` |
| FR-1.2 | Deterministic order: POSIX-relative path, stable code-point sort | `test_corpus_discovery.py` |
| FR-1.3 | **Filenames, prefixes, numbering and path depth are never interpreted** | `test_filename_independence` |
| FR-1.4 | Document collection is not fixed: add/remove/rename works with no code change | `test_corpus_discovery.py` |
| FR-1.5 | Text preserved exactly except BOM removal, CRLF→LF, Unicode NFC | `test_markdown.py`, `test_ids.py` |
| FR-1.6 | Markdown blocks (headings, lists, tables, fences, quotes) never split mid-block | `test_chunking.py::TestSizing` |

### FR-2 Dataset construction

| # | Requirement | Verified by |
|---|---|---|
| FR-2.1 | Persian questions generated per chunk, each with verbatim-grounded key points | `test_tasks_and_negatives.py::TestRelevanceGeneration` |
| FR-2.2 | Positives rest on a teacher *finding*, not a plausible generation | `verify_positives: true` + evidence gate |
| FR-2.3 | Hard negatives from three strategies, interleaved round-robin | `test_tasks_and_negatives.py::TestNegativeSampling` |
| FR-2.4 | **A non-source chunk is never auto-labelled irrelevant** | four gates, AGENTS 6.1 |
| FR-2.5 | **Ambiguity is a discard, never a coin flip** | `test_an_uncertain_verdict_is_a_discard_not_a_label` |
| FR-2.6 | Sufficiency judged against key-point coverage, never similarity | `tasks/sufficiency.py` |
| FR-2.7 | Routing labels derive from query form only — teacher never sees source identity | `tasks/routing.py`, AGENTS 6.6 |

### FR-3 Splits and leakage

| # | Requirement | Verified by |
|---|---|---|
| FR-3.1 | **Split before mining negatives** | stage order in `stages.py` |
| FR-3.2 | Negatives mined inside each split's *own* chunk pool | `stage_negatives` `_pool_for_split` |
| FR-3.3 | Group splitting by document and by near-duplicate family | `test_dataset.py::TestLeakage` |
| FR-3.4 | Four leakage classes detected: identical decision, source split, duplicate query, paraphrased query | `test_dataset.py` |
| FR-3.5 | Leakage report must be empty; non-empty is a defect | AGENTS 6.8 |

### FR-4 Dataset emission

| # | Requirement | Verified by |
|---|---|---|
| FR-4.1 | Rows are exactly `{state, questions, expected}` plus optional `tags`, `language` | `dataset/builder.py` |
| FR-4.2 | **One `(query, chunk)` pair per row** | AGENTS 3.5 |
| FR-4.3 | Question definitions have **one** source of truth | `configs/default.yaml::laya.questions` |
| FR-4.4 | `state` rendered from a config-driven Persian template; fingerprinted | `test_dataset.py::TestValidation` |
| FR-4.5 | Invalid labels, missing fields, duplicate states reported by row index | `test_dataset.py::TestValidation` |

### FR-5 Training

| # | Requirement | Verified by |
|---|---|---|
| FR-5.1 | Training through `laya.train.finetune` / `dry_run` in Python, not the CLI | AGENTS 3.3 |
| FR-5.2 | Optimizer-update budget printed **before** any GPU spend | `training.py`, AGENTS 3.13 |
| FR-5.3 | `train` refuses to start without `--yes` | `training.py` |
| FR-5.4 | `eval_data` passed for every real run | AGENTS 3.9 |
| FR-5.5 | Label-encoding positive control available as the chance-level diagnostic | AGENTS 3.14 |

### FR-6 Reproducibility

| # | Requirement | Verified by |
|---|---|---|
| FR-6.1 | One seed threaded through every stage; no module-level RNG | AGENTS 4.1 |
| FR-6.2 | Re-running a stage on unchanged inputs is byte-identical | AGENTS 4.2 |
| FR-6.3 | Every artifact write is atomic (`os.replace`) | `io.py`, `test_cli_and_io.py` |
| FR-6.4 | Every artifact directory carries `manifest.json` with input hashes, config hash, seed, counts, stage version | `io.write_manifest` |
| FR-6.5 | Caches content-addressed; a hit equals a fresh call | `llm/cache.py` |

### FR-7 Safety and diagnosis

| # | Requirement | Verified by |
|---|---|---|
| FR-7.1 | Writes under `corpus/` are refused by assertion, not convention | `io.resolve_output_path`, `test_corpus_protection` |
| FR-7.2 | `preflight` compares configured models against Ollama in one second | `test_diagnostics.py` |
| FR-7.3 | Structured logging with named fields; no bare `print()` in library code | AGENTS 5.2 |
| FR-7.4 | Typed exceptions naming the offending path, id or value | `errors.py` (20 classes) |

## 6. Non-functional requirements

| # | Requirement | Target | Status |
|---|---|---|---|
| NFR-1 | Dataset stage, cold, 20 docs / 80 chunks | ≤ 3 h on one local GPU-free box | **measured 2.5 h** |
| NFR-2 | Full test suite runs without network, GPU, weights or a real corpus doc | 0 external deps | **461 tests, 7.8 s** |
| NFR-3 | A crashed run never leaves a half-written artifact | 0 occurrences | atomic writes |
| NFR-4 | Re-running an interrupted stage does not re-bill the model | 0 duplicate calls | content-addressed cache |
| NFR-5 | Corpus is byte-identical before and after any pipeline run | 0 writes | verified |
| NFR-6 | Full data pipeline runs without `laya` or `torch` installed | 0 imports | lazy import |

## 7. Success criteria

Specific, testable, and currently honest about what has been executed.

### Data quality

- [x] `generate` yields >2 questions per chunk from a 4B teacher, all grounded
      *(measured: 225 examples from 80 chunks, 15 discarded)*
- [x] Every negative carries a recorded acceptance reason
      *(measured: 156 accepted, 1845 rejected by the teacher-says-answers gate)*
- [x] `split.json.leakage == []` *(measured)*
- [x] Split warnings surface the small-corpus problem rather than hiding it
      *(measured: 3 families in validation and test)*

### Engineering

- [x] 461 tests pass, no network/GPU/weights
- [x] Filename independence pinned by `test_filename_independence`
- [x] Corpus unmodified after every run
- [x] `preflight` catches a missing model in ~1 s instead of ~100 s per chunk

### Training and evaluation — **not yet executed**

- [ ] `train --dry-run` prints a budget above the chance-level threshold
- [ ] `train smoke --yes` completes on the target GPU without OOM
- [ ] `evaluate classify` beats the class prior on the held-out test split
- [ ] Report carries a calibration caveat and no unmeasured metric

**No number in this repository is a training or evaluation result.** Everything in the
first two blocks was executed; everything in the third block is a target.

## 8. Commands

Full per-stage detail: [README runbook](../README.md#the-complete-runbook).

```bash
# environment
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[laya,dev]'
cp .env.example .env
bash scripts/check_env.sh

# dataset (stages 0-12; nothing touches a GPU)
laya-pipeline preflight
laya-pipeline scan && laya-pipeline process
laya-pipeline generate          # SLOW
laya-pipeline split
laya-pipeline negatives         # SLOWEST
laya-pipeline label             # SLOW
laya-pipeline tasks --task sufficiency
laya-pipeline tasks --task routing
laya-pipeline build --task all && laya-pipeline validate --task all && laya-pipeline stats --task all

# training (GPU)
laya-pipeline train --task relevance --profile configs/dry_run.yaml --dry-run
laya-pipeline train --task relevance --profile configs/smoke.yaml --out outputs/checkpoints/smoke --yes
laya-pipeline train --task relevance --profile configs/full.yaml  --out outputs/checkpoints/relevance --yes

# evaluation
laya-pipeline evaluate classify  --checkpoint outputs/checkpoints/relevance --task relevance --out outputs/evaluation
laya-pipeline evaluate retrieval --split test --out outputs/evaluation

# tests
pytest
```

## 9. Project structure

```
corpus/          Source of truth. READ-ONLY. Not committed. Not in git history.
prompts/         6 editable Persian prompt templates
configs/         default.yaml (source of truth) + dry_run / smoke / full profiles
src/laya_pipeline/   11,220 lines — pipeline
tests/           4,077 lines — 461 tests, all external calls mocked
docs/            This PRD and ARCHITECTURE.md
outputs/         Generated artifacts. Not committed.
AGENTS.md        Binding rules for every agent and human contributor
```

## 10. Boundaries

**Always**
- Run `pytest` before committing.
- Read `configs/default.yaml` before changing behaviour; it is the source of truth.
- Print the optimizer-update budget before any run that costs GPU time.
- Record what was actually executed. Never report a target as a result.

**Ask first**
- Changing `laya.questions` or `state_templates` (invalidates every built dataset).
- Adding a dependency (AGENTS 5.7 justifies each existing one).
- Changing split ratios or family-similarity thresholds (invalidates every dataset).
- Weakening or deleting any test (AGENTS 7.6).

**Never**
- Write, move, rename or delete anything under `corpus/`.
- Let a filename, prefix or numbering scheme influence behaviour.
- Fold Arabic yeh/kaf, remove ZWNJ, or normalise digits in *stored* text.
- Auto-label a non-source chunk irrelevant; assign a label to an uncertain verdict.
- Present an unmeasured metric as zero.
- Present a confidence number as calibrated before temperatures are refit.
- Push to GitHub, run training, or invoke a production model as an agent (AGENTS 8.1).

## 11. Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R-1 | 4B teacher too weak → small, quiet dataset | **observed** | Medium | Safe failure by design; upgrade `LAYA_OLLAMA_TEACHER_MODEL` |
| R-2 | Corpus too small for held-out claims | **observed** | High | Pipeline warns; report must say it. Add documents, not questions |
| R-3 | GPU too small for `max_len: 1024` | **observed** (3.7 GiB) | High | Smoke profile first; lower `max_len` before `micro_batch` |
| R-4 | `EvalReport.cases` shape unpinned → macro-F1 unavailable | Medium | Low | Extraction is defensive; reports **unavailable with a reason**, never zero |
| R-5 | Uncalibrated confidences → over-trust | **certain** | High | Caveat in every report; no `min_confidence` gating |
| R-6 | Family merging is O(n²) | Low at this scale | Low | Inverted index needed only at tens of thousands of documents |

## 12. Open questions

| # | Question | Blocks | Who |
|---|---|---|---|
| Q-1 | Does relevance beat the class prior on the held-out test split? | Success criteria | Training run |
| Q-2 | What `laya.max_len` fits the 3.7 GiB card without OOM? | Training | Smoke run |
| Q-3 | What is the real `EvalReport.cases` shape? | Macro-F1, confusion matrix | First `evaluate classify` |
| Q-4 | Is a stronger teacher (7B+) worth the wall-clock? | Dataset size | Operator |
| Q-5 | Should `negatives.ratio` rise above 1.0 for per-negative efficiency? | Label balance | After first evaluation |

---

## Appendix — measured baseline

The one real end-to-end dataset run on the development machine, 2026-10-13. These are
the numbers the pipeline produced, not estimates.

| Metric | Value |
|---|---|
| Documents | 20 |
| Chunks | 80 (min 160 / median 180 / max 202 tokens, 0 oversized) |
| Positives generated | 225 |
| Questions discarded (unanswerable) | 15 |
| Split | train 160 / validation 33 / test 32, across 20 families (14/3/3) |
| Negatives accepted | 156 |
| Negatives unfilled (pool too small) | 69 |
| Teacher rejections in mining | 1845 |
| `generate` wall clock | 15 min (320 calls) |
| `negatives` wall clock | 85 min (2,635 verification calls) |
| Leakage findings | 0 |