# Testing Strategy

This document defines the short, repeatable checks for the Laya data pipeline. It keeps
ordinary changes quick and keeps expensive model work out of the agent test loop.

## 1. Goals

- Catch regressions in corpus handling, Persian text preservation, dataset schemas, split
  boundaries, and model-call handling with small synthetic fixtures.
- Run the smallest relevant test set after a code change.
- Keep tests independent of the production corpus, Ollama, Laya checkpoints, network,
  GPU, and fine-tuning.
- Record whether a result came from an agent, the operator, or CI. A file existing under
  `outputs/` proves an artifact exists; it does not by itself identify who ran the stage.

## 2. Test layers

| Layer | Scope | Typical files |
|---|---|---|
| Unit | One function or module, including malformed and boundary inputs | `test_ids.py`, `test_config.py`, `test_markdown.py`, `test_chunking.py` |
| Contract | Dataset rows, split safety, deterministic artifacts, corpus write protection | `test_dataset.py`, `test_cli_and_io.py`, `test_corpus_discovery.py` |
| Mocked integration | Stage and client interactions through synthetic fixtures and `FakeTransport` | `test_tasks_and_negatives.py`, `test_llm.py`, `test_diagnostics.py` |
| Model/training | No automated test loads weights or performs fine-tuning | Operator-only workflow, outside pytest |

Every regression test uses synthetic Persian text. External model calls use the transport
seam from `tests/conftest.py`; tests must not call Ollama or Laya directly.

## 3. Selecting tests

Run the narrowest test that covers the edited behavior. Keep output compact with `-q` and
`--tb=short`.

```bash
# Example: corpus discovery and filename independence
pytest -q --tb=short tests/test_corpus_discovery.py

# Example: only the negative-sampling cases in the relevant file
pytest -q --tb=short tests/test_tasks_and_negatives.py -k NegativeSampling

# Example: row building and validation
pytest -q --tb=short tests/test_dataset.py -k 'build or validation'

# Example: retrieval metric and report behavior
pytest -q --tb=short tests/test_retrieval_and_report.py
```

For a change spanning shared contracts or multiple pipeline stages, run the small set of
affected test files together. Run the full suite only when the operator requests it, a
change crosses multiple shared contracts, or a release needs it:

```bash
pytest -q --tb=short
```

Do not repeat a command after the relevant files have not changed. Do not run the full
suite as the default for a documentation-only edit.

## 4. Tests never perform model work

Pytest must not install dependencies, download weights, access the network, read a
production corpus document, start Ollama, load a Laya checkpoint, create production
datasets, or start a training/evaluation job. A test that needs an external model must use
a fake transport or a small injected function.

**Agents may run focused pytest commands in the existing project environment.** They must
not run the main fine-tuning job, any training command with `--yes`, a production dataset
generation run, a real Ollama job, or a real checkpoint evaluation. Those are operator-run
steps. `train --dry-run` may load tokenizer/model files and is also not part of routine
agent verification; use mocked tests for train-plan logic instead.

Do not install packages to make tests run. If pytest or a dependency is unavailable, report
the exact focused command as unrun and let the operator decide whether to prepare the
environment.

## 5. Coverage priorities

Give explicit tests to high-risk contracts:

- Filename-independent recursive discovery and stable ordering.
- Persian normalization, ZWNJ tokenization, and verbatim source spans.
- Empty/malformed configuration and unknown keys.
- Structural Markdown boundaries and oversized blocks.
- Negative evidence gates, uncertain answers, and answer-string leakage.
- Family-level splits, identical decisions, duplicate queries, paraphrase leakage, and
  split-local negative pools.
- Exact Laya row keys, one decision per row, configured question schema, and valid split
  tags.
- Atomic writes, output path protection, cache identity, and manifest contents.
- CLI safety gates for real training, including the required held-out validation data.
- Retrieval evaluation with both lexical and embedding backends, using fake embeddings.

New behavioral requirements must name their test in `docs/PRD.md`. Tests may not be
weakened, skipped, or deleted to reduce runtime.

## 6. Recording results

Record test results with the command, date, result, and executor. Distinguish these states:

- **Executed:** the command ran and its exit status/output was observed.
- **Operator-reported:** the operator says it ran; this agent did not verify it.
- **Unverified:** no result is available for the current source tree.
- **Not applicable:** documentation-only change with no code behavior changed.

Do not copy historical counts or elapsed times into a current result. A previous README
and PRD recorded 461 tests with different elapsed times; until a run is recorded against
the current tree, treat both timings as historical and unverified.

## 7. Definition of done

- New behavior has a focused synthetic regression test or a documented reason that a test
  is not applicable.
- The selected short test command passes, or its exact unrun/failed state is reported.
- No test triggered model training, production generation, external network access, or
  corpus reads.
- Final notes identify what ran and what remains unverified.
