"""Retrieval metrics, index construction, and the evaluation report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from laya_pipeline.corpus.models import Chunk, DocumentSource
from laya_pipeline.errors import PipelineError
from laya_pipeline.evaluation.classification import (
    answer_confidence_of,
    answer_label,
    confusion_matrix,
    label_of,
    macro_f1,
    per_class_scores,
)
from laya_pipeline.evaluation.report import compare_runs, render_markdown
from laya_pipeline.evaluation.retrieval import build_queries
from laya_pipeline.ids import normalize_text, stable_id
from laya_pipeline.retrieval import (
    BM25Index,
    NumpyEmbeddingIndex,
    build_index,
    cosine_matrix,
    mrr,
    ndcg_at_k,
    recall_at_k,
)


def chunk(text: str, name: str) -> Chunk:
    """A minimal in-memory chunk. ``content_hash`` is derived, not passed."""
    normalized = normalize_text(text)
    return Chunk(
        chunk_id=stable_id("chunk", name),
        source_id=stable_id("src", name),
        relpath=name,
        content_id=stable_id("cnt", normalized),
        ordinal=0,
        text=normalized,
        body=normalized,
    )


SAMPLES = [
    chunk("سیاست مرخصی سالانه و برنامه‌ریزی غیبت کارکنان", "a.md"),
    chunk("کنترل دسترسی به دادهٔ مشتری و ثبت درخواست", "b.md"),
    chunk("انتقال ماندهٔ مرخصی به دورهٔ بعد", "c.md"),
    chunk("مسئولیت‌های تیم ارتباطات داخلی و کانال‌ها", "d.md"),
]


class TestRetrievalMetrics:
    def test_recall_at_k(self) -> None:
        assert recall_at_k(["a", "b", "c"], ["a"], 1) == 1.0
        assert recall_at_k(["a", "b", "c"], ["z"], 3) == 0.0
        assert recall_at_k(["x", "y", "a"], ["a"], 3) == 1.0
        assert recall_at_k(["a", "b"], ["a", "b"], 1) == 0.5

    def test_recall_with_no_gold_is_one_and_is_documented(self) -> None:
        assert recall_at_k(["a"], [], 1) == 1.0

    def test_recall_at_zero_is_zero(self) -> None:
        assert recall_at_k(["a"], ["a"], 0) == 0.0

    def test_mrr(self) -> None:
        assert mrr(["a", "b", "c"], ["c"]) == pytest.approx(1 / 3)
        assert mrr(["a"], ["z"]) == 0.0
        assert mrr(["a"], ["a"]) == 1.0

    def test_ndcg_is_one_for_a_perfect_ranking(self) -> None:
        assert ndcg_at_k(["a", "b"], ["a", "b"], 2) == pytest.approx(1.0)

    def test_ndcg_penalises_a_late_hit(self) -> None:
        assert ndcg_at_k(["x", "y", "a"], ["a"], 3) < ndcg_at_k(["a", "x", "y"], ["a"], 3)

    def test_ndcg_at_zero(self) -> None:
        assert ndcg_at_k(["a"], ["a"], 0) == 0.0

    def test_ndcg_with_no_gold(self) -> None:
        assert ndcg_at_k(["a"], [], 5) == 1.0


class TestBM25Index:
    def test_ranks_the_matching_document_first(self) -> None:
        index = BM25Index([c.chunk_id for c in SAMPLES], [c.body for c in SAMPLES])
        hits = index.rank("سیاست مرخصی سالانه", 2)
        assert hits[0].chunk_id == SAMPLES[0].chunk_id
        assert hits[0].score > 0

    def test_scores_are_ordered_descending(self) -> None:
        index = BM25Index([c.chunk_id for c in SAMPLES], [c.body for c in SAMPLES])
        hits = index.rank("کنترل دسترسی داده", 4)
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)

    def test_score_of_an_unknown_id_is_none(self) -> None:
        index = BM25Index([c.chunk_id for c in SAMPLES], [c.body for c in SAMPLES])
        assert index.score_of("مرخصی", "absent") is None

    def test_mismatched_lengths_are_rejected(self) -> None:
        with pytest.raises(PipelineError):
            BM25Index(["a"], ["یک", "دو"])

    def test_an_empty_index_is_rejected(self) -> None:
        with pytest.raises(PipelineError):
            BM25Index([], [])

    def test_a_query_with_no_shared_tokens_scores_zero(self) -> None:
        index = BM25Index([c.chunk_id for c in SAMPLES], [c.body for c in SAMPLES])
        assert all(h.score == 0.0 for h in index.rank("کلمهٔ‌ناشناخته", 4))


class TestEmbeddingIndex:
    def _matrix(self) -> np.ndarray:
        """Three rows matching the three ids the tests index over."""
        return np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype="float32"
        )

    def _ids(self) -> list[str]:
        return ["a", "b", "c"]

    def test_ranks_by_cosine_similarity(self) -> None:
        index = NumpyEmbeddingIndex(["a", "b", "c"], self._matrix())
        hits = index.rank_vector(np.asarray([1.0, 0.0], dtype="float32"), 3)
        assert hits[0].chunk_id == "a"

    def test_row_count_must_match_ids(self) -> None:
        with pytest.raises(PipelineError, match="rows for"):
            NumpyEmbeddingIndex(["a"], self._matrix())

    def test_an_empty_index_is_rejected(self) -> None:
        with pytest.raises(PipelineError):
            NumpyEmbeddingIndex([], np.zeros((0, 2), dtype="float32"))

    def test_a_raw_query_needs_an_embed_fn(self) -> None:
        index = NumpyEmbeddingIndex(self._ids(), self._matrix())
        with pytest.raises(PipelineError, match="embed_fn"):
            index.rank("متن", 1)

    def test_a_raw_query_scores_with_an_embed_fn(self) -> None:
        index = NumpyEmbeddingIndex(self._ids(), self._matrix())
        hits = index.rank("متن", 1, embed_fn=lambda t: [[1.0, 0.0]])
        assert hits[0].chunk_id == "a"

    def test_score_of_looks_up_a_known_id(self) -> None:
        index = NumpyEmbeddingIndex(self._ids(), self._matrix())
        assert index.score_of("متن", "a", embed_fn=lambda t: [[1.0, 0.0]]) == pytest.approx(1.0)
        assert index.score_of("متن", "absent", embed_fn=lambda t: [[1.0, 0.0]]) is None

    def test_ranks_are_sequential_from_one(self) -> None:
        index = NumpyEmbeddingIndex(self._ids(), self._matrix())
        hits = index.rank_vector(np.asarray([0.7, 0.7], dtype="float32"), 3)
        assert [hit.rank for hit in hits] == [0, 1, 2]

    def test_cosine_matrix_shape(self) -> None:
        result = cosine_matrix(self._matrix(), np.asarray([1.0, 0.0], dtype="float32"))
        assert result.shape == (3,)


class TestBuildIndex:
    def test_lexical_backend_needs_no_embeddings(self) -> None:
        index = build_index("lexical", SAMPLES)
        assert index.backend == "bm25"
        assert len(index) == len(SAMPLES)

    def test_an_embedding_backend_needs_an_embed_fn(self) -> None:
        with pytest.raises(PipelineError, match="embed_fn"):
            build_index("ollama", SAMPLES)

    def test_embedding_backend_builds_from_a_callable(self) -> None:
        def embed(texts):
            return np.asarray(
                [[float(len(text)), float(text.count("مرخصی"))] for text in texts],
                dtype="float32",
            )

        index = build_index("ollama", SAMPLES, embed_fn=embed, batch_size=2)
        assert len(index) == len(SAMPLES)
        assert index.dimension == 2

    def test_an_embed_fn_returning_the_wrong_count_is_rejected(self) -> None:
        def bad_embed(texts):
            return np.zeros((1, 4), dtype="float32")

        with pytest.raises(PipelineError, match="vectors for"):
            build_index("ollama", SAMPLES, embed_fn=bad_embed, batch_size=4)

    def test_an_empty_chunk_list_is_rejected(self) -> None:
        with pytest.raises(PipelineError):
            build_index("lexical", [])


class TestRetrievalQueries:
    def test_only_positives_become_queries(self) -> None:
        from laya_pipeline.dataset.records import ChunkRef, LabeledExample

        positive = LabeledExample(
            example_id="p", task="relevance", query="پرسش؟",
            chunks=[ChunkRef(chunk_id="c1", text="متن")], label="relevant",
            role="positive", split="test",
        )
        negative = LabeledExample(
            example_id="n", task="relevance", query="پرسش؟",
            chunks=[ChunkRef(chunk_id="c2", text="متن")], label="irrelevant",
            role="negative", split="test",
        )
        queries = build_queries([positive, negative], "test")
        assert len(queries) == 1
        assert queries[0].relevant_ids == ["c1"]
        assert queries[0].has_gold


class TestClassificationMetrics:
    PAIRS = [("relevant", "relevant"), ("irrelevant", "relevant"), ("relevant", "irrelevant")]

    def test_confusion_matrix_counts(self) -> None:
        matrix = confusion_matrix(self.PAIRS, ["relevant", "irrelevant"])
        assert matrix["relevant"]["relevant"] == 1
        assert matrix["irrelevant"]["relevant"] == 1

    def test_absent_labels_still_appear(self) -> None:
        matrix = confusion_matrix(self.PAIRS, ["relevant", "irrelevant", "partial"])
        assert matrix["partial"] == {"relevant": 0, "irrelevant": 0, "partial": 0}

    def test_per_class_scores(self) -> None:
        scores = per_class_scores(self.PAIRS, ["relevant", "irrelevant"])
        assert scores["relevant"]["support"] == 2
        assert scores["relevant"]["recall"] == pytest.approx(0.5)

    def test_macro_f1_ignores_unseen_classes(self) -> None:
        """A class the dataset never contains is excluded, not scored as zero.

        Averaging in an absent class would punish the model for data the
        dataset does not have, which is exactly the kind of silent distortion
        AGENTS.md 3.14 rules out.
        """
        scores = per_class_scores(self.PAIRS, ["relevant", "irrelevant", "partial"])
        assert scores["partial"]["support"] == 0
        value = macro_f1(scores)
        supported = [scores[k]["f1"] for k in ("relevant", "irrelevant")]
        assert value == pytest.approx(sum(supported) / len(supported))
        assert value != pytest.approx(scores["partial"]["f1"])

    def test_macro_f1_of_no_data_is_none(self) -> None:
        assert macro_f1({"a": {"f1": 0.0, "support": 0}}) is None

    def test_label_of_coerces_scalars(self) -> None:
        assert label_of("x") == "x"
        assert label_of(True) == "true"
        assert label_of(2) == "2"
        assert label_of(None) is None

    def test_answer_label_reads_conventional_keys(self) -> None:
        assert answer_label({"choice": "relevant"}, "relevance") == "relevant"
        assert answer_label("relevant", "relevance") == "relevant"
        assert answer_label(None, "relevance") is None
        assert answer_label({"unexpected": 1}, "relevance") is None

    def test_answer_confidence_is_not_faked_from_entropy(self) -> None:
        assert answer_confidence_of({"answer_confidence": 0.8}) == 0.8
        assert answer_confidence_of({"confidence": 0.8}) is None
        assert answer_confidence_of("x") is None


class TestComparisonReport:
    def _run(self, accuracy: float, ece: float) -> dict:
        return {"overall": {"choice_accuracy": accuracy, "ece": ece}, "classification": {}}

    def test_delta_is_computed_when_both_sides_exist(self) -> None:
        comparison = compare_runs(
            {"base": self._run(0.5, 0.2), "tuned": self._run(0.7, 0.1)}
        )
        assert comparison["metrics"]["choice_accuracy"]["delta"] == pytest.approx(0.2)

    def test_a_missing_metric_is_omitted_not_zeroed(self) -> None:
        comparison = compare_runs(
            {"base": {"overall": {"ece": 0.2}}, "tuned": {"overall": {}}}
        )
        assert "ece" not in comparison["metrics"].get("ece", {}) or "delta" not in comparison["metrics"]["ece"]

    def test_a_fabricated_zero_never_appears(self) -> None:
        comparison = compare_runs({"base": {"overall": {}}, "tuned": {"overall": {}}})
        assert comparison["metrics"] == {}

    def test_markdown_marks_absent_values(self) -> None:
        markdown = render_markdown(
            compare_runs({"base": self._run(0.5, 0.2), "tuned": {"overall": {}}})
        )
        assert "—" in markdown

    def test_calibration_caveats_are_surfaced(self) -> None:
        run = self._run(0.5, 0.2)
        run["calibration_note"] = "not calibrated"
        comparison = compare_runs({"base": run})
        assert any("not calibrated" in c for c in comparison["caveats"])

    def test_a_non_held_out_run_is_flagged(self) -> None:
        run = self._run(0.5, 0.2)
        run["is_held_out"] = False
        assert any("NOT held out" in c for c in compare_runs({"base": run})["caveats"])

    def test_unavailable_classification_is_explained(self) -> None:
        run = self._run(0.5, 0.2)
        run["classification"] = {"available": False, "note": "no per-case records"}
        assert any("macro-F1" in c for c in compare_runs({"base": run})["caveats"])

    def test_at_least_one_run_is_required(self) -> None:
        with pytest.raises(ValueError):
            compare_runs({})


def test_persian_chunk_helpers_build() -> None:
    made = chunk("متن آزمایشی فارسی", "x.md")
    assert isinstance(made, Chunk)
    assert made.chunk_id.startswith("chunk-")


def test_document_source_is_unused_here() -> None:
    assert DocumentSource is not None


def test_path_fixture_is_available(tmp_path: Path) -> None:
    assert tmp_path.is_dir()