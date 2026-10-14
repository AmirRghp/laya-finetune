# laya-finetune

A reproducible data-preparation and fine-tuning pipeline for **Laya Multilingual**
(`convaiinnovations/laya-multilingual`), a non-autoregressive *decision* model, using the
Persian Markdown documents in `corpus/`.

It builds a labelled decision dataset for three tasks, converts it to Laya's
typed-decisions JSONL, fine-tunes the checkpoint, and compares the result against the
untouched base checkpoint.

```
corpus/*.md
   → discover → parse → chunk
   → generate Persian questions + key points     (Ollama)
   → positives + document-level splits
   → mine hard negatives inside each split       (Ollama)
   → teacher labeling + human review queue       (Ollama)
   → sufficiency / routing labeling              (Ollama)
   → Laya JSONL + validation
   → fine-tune  →  evaluate  →  comparison report
```

Laya is not a generative language model. It never writes text; it answers typed
questions (`choice`, `noul`, `score`) in one forward pass. This project therefore
contains no SFT, QLoRA, TRL, or causal-LM next-token training — only Laya's own
`laya.train` API (see [Why the Python API](#why-the-python-api)).

---

## Quick start

Clean checkout to a validated training dataset:

```bash
cd laya-finetune

# 1. Python environment
python3 -m venv .venv
source .venv/bin/activate                    # Windows: .venv\Scripts\activate
pip install -e '.[laya,dev]'                 # pipeline + tests + training/eval

# 2. Tell the pipeline which models you actually have
cp .env.example .env
$EDITOR .env                                 # set the three LAYA_OLLAMA_*_MODEL lines

# 3. Verify before spending time
bash scripts/check_env.sh                    # must end "0 failures"
laya-pipeline preflight                      # or just this one

# 4. Put your documents in corpus/ — any filenames, any depth
cp ~/my-documents/*.md corpus/

# 5. Build the dataset (stages 0-12; resumable, safe to Ctrl-C)
bash scripts/run_pipeline.sh
```

Output:

```
outputs/dataset/
├── relevance/    train.jsonl  validation.jsonl  test.jsonl
├── sufficiency/  train.jsonl  validation.jsonl  test.jsonl
└── routing/      train.jsonl  validation.jsonl  test.jsonl
```

Then, and only then, training — see [the complete runbook](#the-complete-runbook):

```bash
laya-pipeline train --task relevance --profile configs/dry_run.yaml --dry-run  # prints the budget
laya-pipeline train --task relevance --profile configs/smoke.yaml \
                   --out outputs/checkpoints/smoke --yes
```

### Historical wall-clock report

The earlier runbook recorded these timings for 20 documents and 80 chunks, using
`qwen3.5:4b-q4_K_M`, `qwen3-embedding:4b`, and an RTX 3050 Laptop. The operator recalls
running the pipeline, but the saved artifacts do not include run attribution or stage
timings. Treat these as historical estimates, not a current benchmark.

| Stage | Class | Wall clock | Calls | What dominates |
|---|---|---|---|---|
| `preflight` | ollama | **1 s** | 1 GET | nothing |
| `scan` | local | <1 s | 0 | filesystem |
| `process` | local | ~10 s | 0 | parsing; tokenizer download on first run |
| `generate` | ollama | **15 min** | 320 | 80 question calls (6 s) + 240 evidence checks (2.4 s) |
| `split` | local | ~3 min | 0 | family merging, leakage checks |
| `negatives` | ollama | **85 min** | ~2,600 | 12 verification calls per positive kept |
| `label` | ollama | ~10 min | ~230 | 1 call per mined negative |
| `tasks` (both) | ollama | ~30 min | ~500 | retrieval + judging |
| `build` → `stats` | local | <1 min | 0 | arithmetic |
| **Total** | | **~2.5 h** | ~3,700 | |

Steps 1–12 cost **nothing but local CPU**. GPU time only appears at step 14.

Two properties that matter when you interrupt:

- **Model responses resume from the content-addressed cache.** Re-running a stage
  re-uses matching requests from `outputs/cache/ollama/`. Generation also checks its
  config, chunk, prompt, and stage-version fingerprints before reusing generated rows;
  stale or legacy rows require `generate --no-resume`.
- **Nothing is written until a stage finishes.** Artifacts are written atomically
  (`os.replace`), so a crash never leaves a half-written dataset that the next
  run would trust.

---

## Contents

| | |
|---|---|
| [Quick start](#quick-start) | clean checkout → validated dataset |
| [The complete runbook](#the-complete-runbook) | **every stage, with expected output and what to check** |
| [1. Purpose](#1-purpose) | the three decision tasks and why |
| [2. Architecture](#2-architecture) | module layout |
| [3. Prerequisites](#3-prerequisites) | what you need installed |
| [4. Environment setup](#4-environment-setup) | venv, models, GPU |
| [5. Configuration](#5-configuration) | `default.yaml`, `.env`, model choice |
| [6. Input documents](#6-input-documents) | what to put in `corpus/` |
| [7. Dataset generation](#7-dataset-generation) | commands by cost class |
| [8. Manual label review](#8-manual-label-review) | correcting teacher labels |
| [9. Dataset validation](#9-dataset-validation) | structural and token-length checks |
| [10. Fine-tuning commands](#10-fine-tuning-commands) | profiles, budget, OOM recovery |
| [11. Evaluation commands](#11-evaluation-commands) | classification and retrieval metrics |
| [12. Tests](#12-tests) | the suite |
| [13. Expected artifacts](#13-expected-artifacts) | what lands in `outputs/` |
| [14. Troubleshooting](#14-troubleshooting) | symptom → cause → fix |
| [15. Limitations and risks](#15-limitations-and-risks) | what is unverified |
| [16. Manual execution order](#16-manual-execution-order-condensed) | condensed table |

### Also in `docs/`

| Document | What it is |
|---|---|
| [`docs/PRD.md`](docs/PRD.md) | Requirements, success criteria, risks, and the saved artifact snapshot. Every requirement names the test that verifies it. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Why each significant decision was made and what was rejected - 15 ADRs, the layering, the data model, 14 invariants. |
| [`AGENTS.md`](AGENTS.md) | Binding rules. A violation is a defect, not a trade-off. |

The two documents under `docs/` are **binding**, not background: a change that contradicts
either is a defect. `AGENTS.md` section 9 carries the rule-to-document map.

---

## 1. Purpose

Improve Laya's decisions inside a Persian retrieval-augmented system. Three questions:

| Task | Question | Input | Options |
|---|---|---|---|
| **relevance** (primary) | Does this retrieved passage answer the query? | query + one chunk | `relevant` / `irrelevant` |
| **sufficiency** | Do the retrieved passages together support a complete answer? | query + N chunks | `sufficient` / `partial` / `insufficient` |
| **routing** | Which retrieval strategy fits this query? | query alone | `semantic` / `keyword` / `hybrid` |

Design decisions that matter, and why:

- **Relevance is the primary target.** One `(query, chunk)` pair per training row.
- **Sufficiency is never inferred from similarity.** Each question carries `key_points`
  — the facts a correct answer must contain. A teacher judges the retrieved set against
  those points. Three lexically similar passages that miss the answer are `insufficient`,
  not `sufficient`.
- **Routing labels come from query form, never from the source document.** The labeling
  prompt receives the query text and nothing else — no document title, no heading path,
  no source id — so the model cannot learn to route by organisational domain.
- **Splits happen before negative mining.** Negatives are drawn from each split's own
  chunk pool, so no test document's text can reach a training row.

---

## 2. Architecture

```
src/laya_pipeline/
├── cli.py            one subcommand per stage; argparse; no logic
├── stages.py         orchestration: each stage reads the previous artifact
├── config.py         dataclass config, YAML + env, validation
├── ids.py            hashing, Persian-aware tokenisation and similarity
├── io.py             atomic writes, corpus write-protection, JSONL/CSV
├── errors.py         every typed failure
├── tokenizer.py      Laya tokenizer (lazy) with a character-estimate fallback
├── chunking.py       heading-aware, structure-preserving chunking
├── retrieval.py      BM25 + embedding indexes; Recall@K / MRR / nDCG@K
├── negatives.py      three sampling strategies and four correctness gates
├── labeling.py       teacher labeling, review queue, conflict detection
├── training.py       laya.train bridge; budget calculation
├── corpus/           discovery.py, markdown.py, models.py
├── llm/              ollama.py, prompts.py, schema.py, cache.py
├── tasks/            relevance.py, sufficiency.py, routing.py, base.py
├── dataset/          records.py, split.py, builder.py, validate.py, stats.py
└── evaluation/       classification.py, retrieval.py, report.py
```

Dependencies: `pyyaml`, `httpx`, `numpy`. Dev: `pytest`. Optional: `laya` + `torch`.
**The entire data pipeline runs without `laya` or `torch` installed** — they are imported
lazily inside `training.py` and `evaluation/`.

Rules that bind every change are in [`AGENTS.md`](AGENTS.md).

---

## 3. Prerequisites

- Python **3.10+**
- Ollama running locally, with a generation model, a teacher model and an embedding model
- For training: an NVIDIA GPU with ≥16 GB VRAM and a PyTorch build matching your CUDA
- `pip install -e .` and, for training, `pip install -e '.[laya]'`

Check everything at once:

```bash
bash scripts/check_env.sh
```

It is read-only. It reports Python, every dependency, the Ollama endpoint and its models,
the GPU, and the project files.

---

## 4. Environment setup

```bash
cd /path/to/laya-finetune

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e '.[dev]'            # data pipeline + tests
pip install -e '.[laya,dev]'       # also training and evaluation
```

Pull the Ollama models the config expects:

```bash
ollama pull qwen2.5:7b-instruct    # generation + teacher
ollama pull bge-m3                 # embeddings
```

> **GPU note — measured on this machine.** `scripts/check_env.sh` reports
> `torch 2.14.1+cu130`, `cuda available: True`, and an **RTX 3050 Laptop GPU
> with 3.7 GiB VRAM (capability 8.6)**. That is well under the 16 GB listed
> above, so the training profiles here are tuned for a small card: start from
> `configs/smoke.yaml`, which trains the head only for one epoch. mmBERT-base
> with a 1024-token window in bf16 needs roughly 1.5 GiB of weights before any
> activations, so expect `micro_batch: 1`, `gradient_checkpointing: true`, and
> possibly a shorter `laya.max_len`. If a run OOMs, reduce `laya.max_len`
> before anything else — activations scale with it — and say in your report
> that the shorter window was used.

---

## The complete runbook

Every stage, in order: what it does, what it produces, how to check it worked, and
what to do if it looks wrong. Run them one at a time so you can inspect between.

To run steps 0–12 unattended instead:

```bash
bash scripts/run_pipeline.sh                # stops at the human review step
bash scripts/run_pipeline.sh --with-train   # also runs the smoke profile, after a prompt
```

---

### Step 0 — preflight *(1 second, `[ollama]`)*

```bash
laya-pipeline preflight              # compare the resolved config against /api/tags
laya-pipeline preflight --offline    # config only; Ollama may be stopped
```

Exits non-zero on a blocking finding, so it can gate a script:

```bash
laya-pipeline preflight && laya-pipeline generate
```

**Why this is step 0:** a mistyped model name does not fail when the config loads.
It fails on the first API call, inside the retry loop, about 100 seconds later —
repeated for every chunk. One second here saves an hour later.

---

### Step 1 — scan *(under a second, `[local]`)*

```bash
laya-pipeline scan
```

Walks `corpus/` recursively for `*.md` / `*.markdown`, hashes each file, writes
`outputs/scan/documents.jsonl` + `scan.json` + `manifest.json`.

```bash
wc -l outputs/scan/documents.jsonl     # 20 on the development corpus
```

**If it looks wrong:** `DiscoveryError: corpus directory does not exist` means you
are not in the repo root, or `paths.corpus_dir` points elsewhere. Filenames,
prefixes and directory depth are deliberately never interpreted (AGENTS.md 1.7), so
nothing is skipped because of what it is called.

---

### Step 2 — process *(~10 seconds, `[local]`)*

```bash
laya-pipeline process
```

Parses each document into structural blocks, then packs blocks into chunks without
splitting a heading, list, table, code fence or blockquote. Writes
`outputs/chunks/chunks.jsonl` + `process.json` + `manifest.json`.

```bash
python -c "
import json; cs=[json.loads(l) for l in open('outputs/chunks/chunks.jsonl')]
t=sorted(c['token_count'] for c in cs)
print(len(cs),'chunks  min',t[0],'median',t[len(t)//2],'max',t[-1],
      ' oversized',sum(c['oversized'] for c in cs))"
```

Expected here: `80 chunks  min 160 median 180 max 202  oversized 0`.

The first run downloads the Laya tokenizer; later runs use the cache.

**If it looks wrong:** a high `oversized` count means individual blocks exceed
`chunking.max_tokens`. Per AGENTS.md 2.5 the pipeline will not split them mid-block,
so it reports them rather than truncating them.

---

### Step 3 — generate *(~15 min, `[ollama]`)* — **the first slow stage**

```bash
laya-pipeline generate              # resumes
laya-pipeline generate --no-resume  # forces regeneration
```

Per chunk: one call asking for diverse Persian questions *plus the key points each
answer must contain*, then one teacher call per question confirming the chunk really
answers it. A question the teacher answers `uncertain` about is **discarded**,
never labelled — so a positive rests on a finding, not on a plausible generation.

```bash
python -c "
import json; r=json.load(open('outputs/generated/relevance.report.json'))
print(json.dumps(r['stats'], indent=2))"
```

Measured here: `chunks_processed: 80`, `teacher_calls: 240`,
`examples_generated: 225`, `questions_discarded_unanswerable: 15`, 320 calls total.

**Cost model:** roughly `chunks × (1 + questions_per_chunk)` calls. Lower
`generation.questions_per_chunk` to go faster; raise it for a richer dataset.

**If it looks wrong:**

- `examples_generated` far below `chunks_processed` → the teacher is discarding
  almost everything. That is a *safe* failure (nothing wrong gets in), but it means
  a weak teacher: pull a larger `LAYA_OLLAMA_TEACHER_MODEL`.
- `model returned an empty answer: it spent the whole context window reasoning` →
  set `LAYA_OLLAMA_THINK=false`. Already the default; something is overriding it.
- Slow → the model is the cost, not the pipeline.

---

### Step 4 — split *(~3 min, `[local]`)*

```bash
laya-pipeline split
```

Builds one positive per verified question, merges near-duplicate documents into
families, and assigns whole families to train/validation/test. Runs **before**
negative mining so no test document's text can reach a training row.

Reads `outputs/generated/`, writes `outputs/split/positives.jsonl`, `split.json`,
`dropped.json`.

```bash
python -c "
import json; s=json.load(open('outputs/split/split.json'))
print(json.dumps(s['stats'], indent=2)); print('leakage:', s['leakage'])"
```

Measured here: 225 examples → train 160 / validation 33 / test 32, across 20
families (14 / 3 / 3).

**Read `leakage` and `warnings` before going further.** `leakage` must be empty —
anything in it is a defect, not a warning. Warnings about few families per split are
the small-corpus signal: 3 families in test means per-document metrics there carry a
very wide confidence interval.

### Step 5 — negatives *(~85 min, `[ollama]`)* — **the most expensive stage**

```bash
laya-pipeline negatives
```

Draws candidates three ways — random, BM25 lexical, embedding retrieval —
interleaved round-robin, then puts each through four gates:

1. not the source chunk,
2. not a near-duplicate of it,
3. does not literally contain a key-point answer string,
4. **the teacher says it does not answer the query.**

Absence from the source document is never sufficient on its own (AGENTS.md 6.1).
Everything rejected is written out with its reason.
If `negatives.verify_with_teacher` is false, the stage skips mining and emits no negative
training rows.

```bash
python -c "
import json; r=json.load(open('outputs/negatives/negatives.report.json'))
print(json.dumps(r['stats'], indent=2))"
```

**Cost, so you are not surprised.** `negatives` verifies *every* surviving candidate
and only afterwards truncates to `negatives.ratio`. At the shipped
`candidates_per_positive: 12` and `ratio: 1.0` that is **12 teacher calls per
negative kept** — about 2,600 calls for 225 positives. To go faster:

```bash
laya-pipeline --set negatives.candidates_per_positive=4 negatives   # ~3x faster
```

Trade-off: fewer candidates means more positives fall short of `ratio`, so you get
fewer negatives. Raising `ratio` instead is *more* efficient per negative (12
verifications then yield 2 rather than 1) but skews the label balance toward
`irrelevant`. Check `positives_short_of_target` and `negatives_unfilled` afterwards.

**If it looks wrong:** a high `positives_short_of_target` in validation/test means
that split's chunk pool is too small to supply the budget — a corpus-size property,
not a bug. See [Limitation 5](#15-limitations-and-risks).

---

### Step 6 — label *(~10 min, `[ollama]`)*

```bash
laya-pipeline label
```

Teacher-labels the mined negatives, detects contradictions (the same decision
carrying two different labels — a data defect, not a modelling problem), and writes
the human review queue.

Reads `outputs/negatives/`; writes `outputs/labels/labeled.jsonl`, `labels.json`
and `outputs/review/review.csv`.

---

### Step 7 — human review *(your time, no cost)*

Open `outputs/review/review.csv` and correct the `final_label` column for rows whose
`status` is `needs_review` or `uncertain`. Leave every other column alone, save, then:

```bash
python - <<'PY'
from laya_pipeline.config import load_config
from laya_pipeline.io import read_jsonl, write_jsonl
from laya_pipeline.dataset.records import LabeledExample
from laya_pipeline.labeling import apply_corrections, read_review_csv

config = load_config("configs/default.yaml")
path = "outputs/labels/labeled.jsonl"
examples = [LabeledExample.from_dict(r) for r in read_jsonl(path)]
rows = read_review_csv("outputs/review/review.csv")

options = {t: config.laya.options(t) for t in config.laya.question_ids()}
updated, stats = apply_corrections(examples, rows, options_by_task=options, reviewer="you")

write_jsonl(path, (e.to_dict() for e in updated))
print(stats)
PY
```

Human labels always win; the automatic label is preserved under
`metadata.superseded_label` so the original decision stays auditable. Rebuild
afterwards, because splits are unchanged but labels are not:

```bash
laya-pipeline build --task all && laya-pipeline validate --task all
```

Skipping review is allowed — those labels stay as the teacher set them — but the
rows in the queue are precisely the ones a stronger model would have got wrong.

---

### Steps 8–9 — sufficiency and routing *(~30 min, `[ollama]`)*

```bash
laya-pipeline tasks --task sufficiency --top-k 3
laya-pipeline tasks --task routing
```

**Sufficiency** labels real retrieval output: the query is run against that split's
index, the top-k chunks are retrieved, and the teacher judges them against the key
points. It is never inferred from similarity — three lexically similar passages that
miss the answer are `insufficient`.

**Routing** labels the query alone. The prompt never sees the source document's
title, heading path or id, so the model cannot learn to route by organisational
domain.

### Steps 10–12 — build, validate, stats *(under a minute, `[local]`)*

```bash
laya-pipeline build    --task all     # writes the Laya JSONL
laya-pipeline validate --task all     # structural checks
laya-pipeline stats    --task all     # writes stats.md
```

```bash
cat outputs/dataset/stats.md
```

`validate` reports required fields, non-empty state, question type, instructions
matching the configured definition, option order, label ∈ options, duplicate states
and a split tag on every row — naming the offending row for each failure. An option
that never appears is a **warning**, not an error: a class the dataset never contains
is a class the model will never learn to produce.

Once training is installable, run it again against the real tokenizer:

```bash
laya-pipeline validate --task all --with-laya
```

Over-long states get truncated at training *and* at inference, silently destroying
the evidence at the end of the prompt, so this pass is worth running.

---

### Step 13 — dry run *(seconds, `[weights]`, no GPU, no training)*

```bash
laya-pipeline train --task relevance --profile configs/dry_run.yaml --dry-run
```

Builds every training item with the real Laya tokenizer and prints the
**optimizer-update budget**. Always do this before committing GPU time:

```
epochs               : 8
micro batch          : 4
grad accum           : 16
effective batch      : 64
estimated train items: 1120
optimizer updates    : 152

This is a REAL training run and it has NOT been started.
Re-run with --yes to begin.
```

That budget is `epochs × ceil(ceil(train_items / micro_batch) / grad_accum)`. A small
dataset needs **more** epochs than a large one to reach a comparable update count.
`train` refuses to start without `--yes`.

---

### Steps 14–15 — train *(GPU)*

```bash
# prove it fits and works: head only, one epoch
laya-pipeline train --task relevance --profile configs/smoke.yaml \
                   --out outputs/checkpoints/smoke --yes

# the real run
laya-pipeline train --task relevance --profile configs/full.yaml \
                   --out outputs/checkpoints/relevance --yes
```

| Profile | File | What it does |
|---|---|---|
| dry run | `configs/dry_run.yaml` | Builds every item, prints the budget. No weights, no training. |
| smoke | `configs/smoke.yaml` | One epoch, encoder frozen. Minutes. Proves the path and measures what fits. |
| full | `configs/full.yaml` | 8 epochs, full fine-tune. Hours. |

Do the smoke run first on a small card. If it OOMs, lower `laya.max_len` before
anything else, then `micro_batch`, then enable `gradient_checkpointing` — there is
no point discovering the memory ceiling on a four-hour run.

`--out` receives the layout `laya.load` reads (`model.safetensors`, `encoder/`,
`tokenizer/`, `rl_agent_config.json`) plus `questions.json`, `train_report.json` and
`checkpoint_latest/` rewritten after every epoch, so a crash costs at most one epoch.

**If a run stays at chance**, run the official label-encoding positive control. It
prepends `label=<answer>;` to every state, making the task trivially learnable, which
separates "too few updates" from "task genuinely hard":

```bash
laya-pipeline build --task relevance --positive-control
laya-pipeline train --task relevance --profile configs/smoke.yaml \
                   --positive-control --dry-run
```

Control fails → suspect the budget or the setup. Control passes but the real task
stays at the prior → the evidence points at the task or the representation.

---

### Steps 16–17 — evaluate *(minutes, `[weights]`)*

```bash
# classification: base versus fine-tuned, on the held-out test split
laya-pipeline evaluate classify --checkpoint outputs/checkpoints/relevance \
                              --task relevance --out outputs/evaluation

# retrieval: Recall@K, MRR, nDCG@K over a real index
laya-pipeline evaluate retrieval --split test --out outputs/evaluation

# reranking: retrieval with a Laya relevance decision on top
laya-pipeline evaluate retrieval --split test \
                                --rerank-with outputs/checkpoints/relevance
```

`classify` scores your checkpoint and, unless you pass the base model as
`--checkpoint`, scores `evaluation.base_model` too so the comparison exists. Reports
land in `outputs/evaluation/evaluation.{json,md}`.

Accuracy, ECE, Brier, AURC, selective accuracy and latency come from Laya's own
`laya.evals` harness — this project wraps it rather than reimplementing it. Added
here: macro-F1, per-class F1, a confusion matrix, and for the relevance task the
**recall of `relevant`** and the count of **useful evidence discarded** — the
asymmetry that matters in a RAG pipeline.

A metric that was not measured is **absent** — never zero, never estimated. An empty
table cell means "not measured".

> Confidence figures are NOT calibrated unless the checkpoint's training report
> records a temperature fitted on a held-out slice. `laya-multilingual` ships with
> `temperature [1, 1, 1]` and no fitted `temperature_by_options`.

So: do not gate on `min_confidence` until the calibration report says it was fitted.

---

### Recovering from interruption

Every stage is safe to Ctrl-C and safe to re-run:

```bash
laya-pipeline generate    # resumes from cached model responses
laya-pipeline negatives   # continues from where it stopped, not from zero
```

- **Model responses are cached** in `outputs/cache/ollama/`, keyed by the exact
  request. Changing a prompt, a temperature or a model invalidates only the affected
  entries — a re-run after a crash costs nothing.
- **Artifacts are written atomically.** A stage killed mid-write leaves the previous
  complete version intact, never a truncated one the next run would trust.
- **To redo one stage cleanly**, delete its output directory
  (`rm -rf outputs/negatives`). To invalidate every cached response,
  `rm -rf outputs/cache` — that re-bills you for all of it.

---

---

## 5. Configuration

Everything lives in [`configs/default.yaml`](configs/default.yaml). The keys worth reading
first:

| Section | What it controls |
|---|---|
| `paths` | corpus, output, prompts. Relative paths resolve against `project.root` (default: cwd) |
| `chunking` | token sizes, overlap, tokenizer. Sizes are in the checkpoint's tokens |
| `ollama` | endpoint, three model names, timeouts, retries, cache location |
| `generation` | questions per chunk, key-point cap, temperature, resume |
| `negatives` | ratio, strategies and weights, similarity window, near-duplicate threshold |
| `labeling` | auto-accept threshold and the uncertain band |
| `splitting` | ratios, family similarity, query-leakage threshold |
| `laya` | base checkpoint, token budgets, **and the question definitions** |
| `training` | mirrors `laya.train.TrainConfig` field for field |

Override any single key without editing the file:

```bash
laya-pipeline --set generation.questions_per_chunk=5 stats
```

Environment overrides are declared in the config's own `env_overrides` block — see
[`.env.example`](.env.example). No endpoint, credential or absolute path is hardcoded in
`src/`.

### `.env` is read

Copy `.env.example` to `.env` and edit it; the variables in `env_overrides` are read
from there, so the file is not decorative:

```bash
cp .env.example .env
```

Precedence: **the real environment wins over `.env`**. That is deliberate — a
one-off override must not require editing a file:

```bash
LAYA_OLLAMA_TEACHER_MODEL=qwen2.5:14b laya-pipeline label   # this run only
```

Values are coerced to the type the config declares, and an uncoercible value
(`LAYA_OLLAMA_NUM_CTX=soon`) is an error at load time rather than a string that
surfaces later as a confusing model failure. The same applies to `LAYA_OLLAMA_THINK`:
`true`, `false`, or `null` to leave the flag out entirely.

### Choosing models

The three models do different jobs, and the generation/teacher model is the one
that determines whether the dataset is any good.

| Role | Variable | What it does | Choose for |
|---|---|---|---|
| generation | `LAYA_OLLAMA_GENERATION_MODEL` | writes the Persian questions and key points | instruction-following in Persian, and **structured JSON output** |
| teacher | `LAYA_OLLAMA_TEACHER_MODEL` | verifies evidence, labels relevance, judges sufficiency | the same, but stronger — it gates what survives |
| embedding | `LAYA_OLLAMA_EMBEDDING_MODEL` | mines hard negatives | multilingual, Persian-aware |

Two measured constraints, both from direct probing on this machine:

1. **Turn reasoning off for generation.** The Qwen3.5 family emits a long chain
   of thought into a separate `thinking` field before answering. With
   `think: true` a 2B model spent 4060 tokens on a one-line question and
   returned `content: ""` with `done_reason: "length"` — an empty answer, ~100
   seconds, three times over. `ollama.think` now defaults to `false` and is sent
   on every generation call; an empty reply now reports that cause by name
   instead of saying "no JSON object: ''".
2. **A 2B model is too small to be the teacher.** It answers, but it answers
   inconsistently, and the pipeline correctly discards anything uncertain
   (AGENTS.md 6.2) — so a weak teacher produces a small, quiet dataset rather
   than a wrong one. If `generate` or `label` yields far fewer rows than the
   corpus size, the teacher is the first thing to change.

Embedding models are used only to rank candidate negatives, so a query/prompt
prefix is a quality nicety rather than a correctness requirement. `bge-m3` and
`qwen3-embedding` both work; the latter expects an `Instruct:` prefix on queries
for its best numbers, which this pipeline does not add.

### The questions are the contract

`laya.questions` and `laya.state_templates` are the **single source of truth** for what
the model is asked. The dataset builder, the validator and the evaluation harness all read
them from there. Training on wording inference never sees is the most common way a
fine-tune silently fails, so if you change a question, change it there and rebuild.

```bash
# always: run after any question edit
laya-pipeline validate --task all --with-laya
```

`laya.label_roles` names which option is the positive one, so label strings live next to
the question they belong to instead of in code.

---

## 6. Input documents

Drop `.md` or `.markdown` files anywhere under `corpus/`. Discovery is recursive and
**assumes nothing about filenames** — no prefixes, no numbering, no directory conventions.
Renaming a file does not break anything; adding documents needs no code change.

Recognised structure: YAML frontmatter, ATX and setext headings, paragraphs, lists, pipe
tables, fenced code blocks, blockquotes.

Text is preserved as written. The only transformations applied to stored text are BOM
removal, line-ending normalisation and Unicode NFC. Comparison-time folding (Arabic
`ي`→`ی`, digit normalisation, ZWNJ removal) is used for hashing and similarity only and is
never written back.

---

## 7. Dataset generation

### Commands, by cost class

| Command | Class | Needs |
|---|---|---|
| `preflight` | `[ollama]` | a running Ollama, one `GET /api/tags` |
| `scan` | `[local]` | nothing |
| `process` | `[local]` | nothing |
| `generate` | `[ollama]` | generation model |
| `split` | `[local]` | `generate` output |
| `negatives` | `[ollama]` | generation + embedding models |
| `label` | `[ollama]` | teacher model |
| `tasks --task sufficiency` | `[ollama]` | generation + embedding models |
| `tasks --task routing` | `[ollama]` | generation model |
| `build`, `validate`, `stats` | `[local]` | nothing |
| `train` | `[weights]/[gpu-train]` | Laya weights, then GPU hours |
| `evaluate classify` | `[weights]` | Laya weights |
| `evaluate retrieval` | `[weights]/[ollama]` | embedding model, optionally a checkpoint |

### Run `preflight` first

```bash
laya-pipeline preflight          # compares the resolved config against /api/tags
laya-pipeline preflight --offline   # reports the config only; Ollama may be stopped
```

It exits non-zero on a blocking finding, so it can gate a script:

```bash
laya-pipeline preflight && laya-pipeline generate
```

This exists because a mistyped model name does not fail when the configuration
is loaded — it fails on the first call, inside the retry loop, about a hundred
seconds later, repeated for every chunk. `preflight` turns that into a
one-second check. It also warns when a reasoning model is paired with
`think` enabled, and when `num_ctx` is too small for a Persian chunk plus a JSON
answer. `scripts/check_env.sh` calls the same command, so there is one
implementation rather than two that can drift.

Each help string is tagged. The tags mean:

| Tag | What it needs | Cost |
|---|---|---|
| `[local]` | files only | seconds |
| `[ollama]` | a local Ollama model | minutes to hours |
| `[weights]` | loads Laya weights | one-time checkpoint download, then inference |
| `[gpu-train]` | GPU | minutes to hours |

### Sequential quickstart from a clean environment

Full per-stage detail, with expected outputs and what to check at each step, is in
[the complete runbook](#the-complete-runbook). The commands, in order:

```bash
source .venv/bin/activate

laya-pipeline --config configs/default.yaml preflight   # [ollama] 1 second
laya-pipeline --config configs/default.yaml scan        # [local]
laya-pipeline --config configs/default.yaml process     # [local] downloads the tokenizer once
laya-pipeline --config configs/default.yaml generate      # [ollama] SLOW
laya-pipeline --config configs/default.yaml split         # [local]
laya-pipeline --config configs/default.yaml negatives   # [ollama] SLOWEST
laya-pipeline --config configs/default.yaml label       # [ollama]
laya-pipeline --config configs/default.yaml tasks --task sufficiency   # [ollama] SLOW
laya-pipeline --config configs/default.yaml tasks --task routing        # [ollama] SLOW
laya-pipeline --config configs/default.yaml build    --task all         # [local]
laya-pipeline --config configs/default.yaml validate --task all         # [local]
laya-pipeline --config configs/default.yaml stats    --task all         # [local]
```

Or in one pass:

```bash
bash scripts/run_pipeline.sh
```

`generate` reuses existing rows only when the generation config, chunk contents, prompt,
and stage version match the manifest. If any differ, rerun with `--no-resume` to generate
fresh rows explicitly.

Nothing before the training commands loads Laya weights or touches a GPU.

### What each slow stage does

**`generate` [ollama]** For every chunk, writes a prompt that asks for diverse Persian
questions *and* the key points each answer must contain. Then a teacher verifies that the
chunk really answers each question. A question the teacher answers `uncertain` about is
**discarded**, not labelled. Positives therefore rest on a finding, not on a generation
being plausible.

**`negatives` [ollama]** Three strategies — random, lexical (BM25), and embedding
retrieval — interleaved round-robin. Every candidate passes four gates: not the source
chunk, not a near-duplicate of it, does not literally contain a key-point answer string,
and the teacher says it does **not** answer the query. Absent from the source document is
never on its own sufficient. Everything rejected is written out with its reason.

**`label` [ollama]** Teacher-labels the mined negatives, builds the review queue, and
detects contradictions — the same decision carrying two different labels, which is a data
defect rather than a modelling problem.

**`tasks sufficiency` / `tasks routing` [ollama]** Sufficiency labels *real retrieval
output*: the query is run against that split's index, the top-k chunks are retrieved, and
the teacher judges them against the key points. Routing labels the query alone.

---

## 8. Manual label review

The full procedure is [Step 7 of the runbook](#step-7-human-review-your-time-no-cost).
In short: open `outputs/review/review.csv`, correct the `final_label` column for rows
whose `status` is `needs_review` or `uncertain`, save, apply with the snippet there, then
`build && validate`. Human labels always win and the automatic label is preserved under
`metadata.superseded_label`.

---

## 9. Dataset validation

```bash
laya-pipeline validate --task all                    # structural only
laya-pipeline validate --task all --with-laya        # also state token lengths
```

The structural pass checks, and names the offending row for each failure: required
fields, non-empty state, question type, instructions matching the configured definition,
option order and consistency, label ∈ options, duplicate states, and a split tag on every
row. It warns when an option never appears and when a split is empty.

`--with-laya` loads the real tokenizer and measures state length against `laya.max_len`.
Over-long states get truncated at training and at inference, which silently destroys the
evidence at the end of the prompt — so this is worth running.

The authoritative check is Laya's own:

```bash
laya-train --data outputs/dataset/relevance/train.jsonl \
           --base convaiinnovations/laya-multilingual --dry-run
```

That builds the training items with Laya's tokenizer and prints the optimizer-update
budget. The pipeline runs the same thing through the Python API in the next section.

---

## 10. Fine-tuning commands

Step-by-step commands, with the budget output and the OOM recovery order, are in
[the runbook](#steps-1415-train-gpu). What follows is the reasoning behind them.

### Why the Python API

`laya-train` cannot express `amp`, `gradient_checkpointing`, `calib_max` or `calib_frac`.
Shelling out to it would silently apply defaults you did not choose. This project calls
`laya.train.dry_run` and `laya.train.finetune` directly, so every `TrainConfig` field is
reachable.

### Three profiles

| Profile | File | What it does |
|---|---|---|
| dry run | `configs/dry_run.yaml` | Builds every item with the real tokenizer, prints the budget. **No weights, no training.** |
| smoke | `configs/smoke.yaml` | One epoch, encoder frozen. Minutes. Proves the path and measures what fits. |
| full | `configs/full.yaml` | 8 epochs, full fine-tune. Hours. |

**These are operator-run steps. Agents must not start fine-tuning or load a real
checkpoint; see `AGENTS.md` §8 and `docs/TESTING.md`.**

### Order

```bash
# 1. always do this first — no weights, no GPU
laya-pipeline train --task relevance --profile configs/dry_run.yaml --dry-run

# 2. prove it fits and works
laya-pipeline train --task relevance --profile configs/smoke.yaml \
                   --out outputs/checkpoints/smoke --yes

# 3. the real run
laya-pipeline train --task relevance --profile configs/full.yaml \
                   --out outputs/checkpoints/relevance --yes
```

`train` **always prints the resolved plan and the optimizer-update budget first**, and
**refuses to start without `--yes`**:

```
epochs               : 8
micro batch          : 4
grad accum           : 16
effective batch      : 64
estimated train items: 1120
optimizer updates    : 152

This is a REAL training run and it has NOT been started.
Re-run with --yes to begin.
```

That budget is `epochs × ceil(ceil(train_items / micro_batch) / grad_accum)`. A small
dataset needs **more** epochs than a large one to reach a comparable update count — the
official guidance is explicit about this, and it is why the default is 8 rather than the
notebook's 4.

**These settings have not been measured on your GPU.** The measured hardware is an
RTX 3050 Laptop with 3.7 GiB. If a run OOMs, lower `laya.max_len` first, then
`micro_batch`, then enable `gradient_checkpointing` — there is no point discovering
the memory ceiling on a four-hour run.

### When a run stays at chance

Run the official label-encoding positive control. It prepends `label=<answer>;` to every
state so the task becomes trivially learnable, which separates "too few updates" from
"task genuinely hard":

```bash
laya-pipeline build --task relevance --positive-control
laya-pipeline train --task relevance --profile configs/smoke.yaml \
                   --positive-control --dry-run
```

Control fails → suspect the budget or the setup. Control passes but the real task stays at
the prior → the evidence points at the task or the representation. See
[Limitations](#15-limitations-and-risks) for the seed-dependence caveat.

### Artifacts per run

`--out` receives the layout `laya.load` reads — `model.safetensors`, `encoder/`,
`tokenizer/`, `rl_agent_config.json` — plus `questions.json`, `train_report.json` and
`checkpoint_latest/` rewritten after every epoch, so a crash costs at most one epoch.

---

## 11. Evaluation commands

```bash
# classification: base versus fine-tuned, on the held-out test split
laya-pipeline evaluate classify \
    --checkpoint outputs/checkpoints/relevance \
    --task relevance \
    --out outputs/evaluation

# retrieval: Recall@K, MRR, nDCG@K over a real index
laya-pipeline evaluate retrieval --split test --out outputs/evaluation

# reranking: retrieval with a Laya relevance decision on top
laya-pipeline evaluate retrieval --split test \
    --rerank-with outputs/checkpoints/relevance
```

`classify` scores the fine-tuned checkpoint and, unless you pass the base model as
`--checkpoint`, scores `evaluation.base_model` too so the comparison exists. Reports land
in `outputs/evaluation/evaluation.{json,md}`.

### What is measured

Accuracy, ECE, Brier, AURC, selective accuracy and latency come from Laya's own
`laya.evals` harness — this project wraps it rather than reimplementing it. Added here:
macro-F1, per-class F1, a confusion matrix, and for the relevance task the
**recall of `relevant`** and the count of **useful evidence discarded**, which is the
asymmetry that matters in a RAG pipeline.

Retrieval metrics are a genuine retrieval measurement: a real index, a real ranked list,
and gold sets that are the chunks that actually answer the query. Classification accuracy
is never presented as retrieval quality. Queries with no gold document are excluded from
the mean and counted, rather than scored 1.0 and inflating it.

### Reading a report

A metric that was not measured is **absent**, never zero and never estimated. An empty
table cell means "not measured". Every report carries a caveats block; the standard one is:

> Confidence figures are NOT calibrated unless the checkpoint's training report records a
> temperature fitted on a held-out slice. `laya-multilingual` ships with `temperature
> [1, 1, 1]` and no fitted `temperature_by_options`.

So: do not gate on `min_confidence` until the calibration report says it was fitted.

---

## 12. Tests

See [`docs/TESTING.md`](docs/TESTING.md) for test layers, focused commands, and rules for
recording results. Run the smallest relevant test file or selector first; run the full suite
only for broad cross-stage changes, release work, or an explicit request.

```bash
pip install -e '.[dev]'
pytest -q --tb=short tests/test_corpus_discovery.py
pytest -q --tb=short tests/test_dataset.py -k validation
```

No test touches the network, a GPU, Laya weights or a production corpus document. Every
external model call goes through one transport seam and is replaced by a scripted fake.

The previous documentation recorded 461 passing tests, with different elapsed times in the
README and PRD. No current test log is stored with the artifacts, so treat that count and its
runtime as historical until the suite is run against the current tree. Agents may run
focused tests in the existing environment. They must not run main fine-tuning, production
generation, real Ollama calls, or real checkpoint evaluation.

```bash
pytest -q --tb=short    # full suite; see TESTING.md for when to use it
```

Coverage by area: recursive discovery, filename independence, Persian handling, Markdown
structure, chunk boundaries, identifier stability, duplicate detection, schema validation,
prompt rendering, reasoning-mode flags and empty-answer diagnostics, `.env` precedence,
retry and cache behaviour, synthetic output validation, negative-gate correctness, label
precedence and conflicts, group splitting, leakage detection, dataset conversion, config
validation, CLI parsing, and artifact-safety guards.

---

## 13. Expected artifacts

```
outputs/
├── scan/          documents.jsonl, scan.json, manifest.json
├── chunks/        documents.jsonl, chunks.jsonl, process.json, manifest.json
├── generated/     relevance.jsonl, relevance.report.json, manifest.json
├── split/         positives.jsonl, split.json, dropped.json, manifest.json
├── negatives/     negatives.jsonl, negatives.report.json, manifest.json
├── labels/        labeled.jsonl, labels.json, manifest.json
├── review/        review.csv, manifest.json
├── sufficiency/   sufficiency.report.json, manifest.json
├── routing/       routing.report.json, manifest.json
├── dataset/
│   ├── relevance/ train.jsonl, validation.jsonl, test.jsonl, all.jsonl,
│   │              questions.json, manifest.json
│   ├── sufficiency/ ...
│   ├── routing/     ...
│   ├── validation.json, validation.manifest.json
│   ├── stats.json, stats.md, stats.manifest.json
├── cache/ollama/  content-addressed model responses (safe to delete)
└── checkpoints/   smoke/, relevance/   (git-ignored)
```

`outputs/` and `corpus/` are git-ignored: the corpus is the source of truth and is not
committed, and generated datasets may embed derived document text.

---

## 14. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `DiscoveryError: corpus directory does not exist` | wrong path or run from the wrong directory | run from the repo root; check `paths.corpus_dir` |
| `OllamaUnavailable` | server not running, or wrong endpoint | `ollama serve`; check `LAYA_OLLAMA_ENDPOINT` |
| `OllamaResponseError ... 404` on `/api/embed` | older Ollama | handled automatically; it falls back to `/api/embeddings` |
| `model response contains no JSON object: ''` after minutes | a reasoning model spent its whole budget thinking and never answered | `think: false` is already the default; confirm `LAYA_OLLAMA_THINK` is not `true`, and prefer a non-reasoning model for generation |
| `model returned an empty answer: it spent the whole context window reasoning` | same cause, now named explicitly | set `LAYA_OLLAMA_THINK=false`; raise `LAYA_OLLAMA_NUM_CTX` only if the prompt itself is large |
| `.env` edits appear to do nothing | variables only take effect when exported | `.env` at the repo root is read now; a real environment variable still wins, so `LAYA_OLLAMA_GENERATION_MODEL=x laya-pipeline generate` overrides the file for one command |
| Teacher keeps returning `uncertain` | the model is too small for the Persian instructions | use a stronger generation/teacher model; the pipeline correctly discards rather than guessing |
| `fewer X candidates than negatives.ratio` | the corpus is small | lower `negatives.ratio`; this is a corpus property, not a bug |
| Split warnings about empty splits or few families | fewer documents than splits need | add documents, or accept a wider error bar and say so |
| `ValidationError ... exceeds laya.max_len` | Persian tokenises densely | lower `chunking.max_tokens`, or raise `laya.max_len` if the checkpoint allows |
| RTX 5060 Ti reports `cuda is not available` | torch build lacks `sm_120` | install a newer torch. Not applicable here: `check_env.sh` measured `cuda available: True` on torch 2.14.1+cu130 |
| Training OOM on a small card | `laya.max_len: 1024` drives activation memory | lower `laya.max_len` first, then `micro_batch`, then enable `gradient_checkpointing`. See the GPU note in section 4 |
| Accuracy high but the model always answers `irrelevant` | class imbalance | read `stats.md`; rebalance `negatives.ratio` |
| `training ... laya is not installed` | optional extra missing | `pip install -e '.[laya]'` — stages 1–10 work without it |

---

## 15. Limitations and risks

1. **Execution provenance.** Earlier documentation records a 461-test run and
   dataset-stage runs. The operator recalls running the pipeline, and local dataset
   artifacts are present. The saved artifacts predate per-stage manifests, so they do not
   provide complete attribution. Current stage code writes manifests on new runs. See
   `docs/TESTING.md` for how to record verification against the current tree.

   **Training and model evaluation:** no checkpoint or evaluation report was found in the
   inspected `outputs/` listing. Main fine-tuning and real checkpoint evaluation have not
   been established by the available artifacts; agents must leave those steps to the
   operator.

2. **`EvalReport.cases` shape is not fully pinned** in the official docs. Macro-F1 and the
   confusion matrix are extracted defensively across several plausible shapes; if none
   matches, those metrics are reported **unavailable with the reason**, never as zero.
   Verify on your first `evaluate classify` run.

3. **Training settings are unverified on your GPU.** The measured hardware is an
   RTX 3050 Laptop (3.7 GiB), not the 16 GB the profiles were sized for. The
   values in `configs/smoke.yaml` are a head-only starting point, not measured
   working settings. Read the optimizer-update budget printed by
   `train --dry-run` before committing GPU time (AGENTS.md 3.13), and expect to
   lower `laya.max_len` to fit.

4. **Laya's own findings on small datasets.** A small dataset needs more epochs for a
   comparable update count, and a run that never rises above the class prior may be
   seed-dependent rather than wrong. Use the positive control before changing anything
   else, and repeat a run with a different seed before concluding a task is unlearnable.

5. **The corpus is small.** 20 documents across several domains. With fewer than ~10
   document families per split, per-document metrics carry a very wide confidence interval,
   and the pipeline warns about it. More documents will help far more than more questions
   per document.

6. **`laya-multilingual` ships uncalibrated.** Its confidences are over-confident and no
   temperatures are fitted. Every report says so; honour it before gating on confidence.

7. **Negative labels depend on a teacher.** Four gates reduce the risk, but a mined
   negative that a different reviewer would call ambiguous is still a wrong label. The
   review queue exists for exactly these; use it.

8. **Persian similarity is lexical.** BM25 and the near-duplicate gate use token overlap
   with no stemming. Persian morphology is productive enough that a wrong light stemmer
   costs more recall than it buys. The embedding path is available for cases that need it.

9. **Family merging is O(n²)** over documents. Cheap at this corpus size; a corpus of tens
   of thousands of documents would need an inverted index and a different split strategy.

10. **The retrieval backend defaults to Ollama embeddings.** Quality for Persian hard
    negatives depends on the chosen model; if they are poor, try `retrieval_backend:
    laya_encoder` (no extra download, uses the checkpoint's own encoder) or a dedicated
    multilingual bi-encoder.

---

## 16. Manual execution order, condensed

Per-stage detail is in [the complete runbook](#the-complete-runbook). The table summarizes
historical operator-reported timings for 20 documents and 80 chunks; no current benchmark
log is available:

| # | Command | Class | Historical time | Dominated by |
|---|---|---|---|---|
| 0 | `preflight` | ollama | **1 s** | one `GET /api/tags` |
| 1 | `scan` | local | <1 s | filesystem |
| 2 | `process` | local | ~10 s | parsing; tokenizer download once |
| 3 | `generate` | ollama | **15 min** | 320 model calls |
| 4 | `split` | local | ~3 min | family merging, leakage checks |
| 5 | `negatives` | ollama | **85 min** | ~2,600 verification calls |
| 6 | `label` | ollama | ~10 min | 1 call per mined negative |
| 7 | *human review* | — | your time | reading `review.csv` |
| 8 | `tasks --task sufficiency` | ollama | ~25 min | retrieval + judging |
| 9 | `tasks --task routing` | ollama | ~5 min | 1 call per query |
| 10 | `build` | local | <1 s | arithmetic |
| 11 | `validate --with-laya` | local | seconds | tokenizer |
| 12 | `stats` | local | <1 s | arithmetic |
| 13 | `train --dry-run` | weights, no training | seconds | tokenizer |
| 14 | `train smoke --yes` | **GPU** | minutes | one epoch, head only |
| 15 | `train full --yes` | **GPU** | hours | 8 epochs |
| 16 | `evaluate classify` | weights | minutes | inference |
| 17 | `evaluate retrieval` | weights / ollama | minutes | ranking |

Steps 1–12 cost nothing but local CPU. Step 0 costs one second and can save step 3
entirely. Steps 3, 5 and 8 are where the wall-clock time goes. Step 15 is where the
money goes. Nothing touches a GPU before step 14.

Times scale roughly with `corpus documents × questions_per_chunk`. To cut the bill
without changing the dataset's shape, the cheapest knob is
`negatives.candidates_per_positive` (step 5 is 12 teacher calls per negative kept).
