"""Task generators, negative mining and labeling."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from laya_pipeline.chunking import Chunker
from laya_pipeline.config import ChunkingConfig, Config
from laya_pipeline.corpus.markdown import parse_markdown
from laya_pipeline.corpus.models import Chunk, DocumentSource
from laya_pipeline.dataset.records import (
    ChunkRef,
    GeneratedExample,
    KeyPoint,
    LabeledExample,
    make_example_id,
    strongest_label_source,
)
from laya_pipeline.errors import NegativeSamplingError
from laya_pipeline.ids import normalize_text, stable_id
from laya_pipeline.labeling import (
    apply_corrections,
    build_review_queue,
    detect_contradictions,
    label_distribution,
    propose_labels,
)
from laya_pipeline.llm.cache import NullCache
from laya_pipeline.llm.ollama import OllamaClient
from laya_pipeline.llm.prompts import PromptLibrary
from laya_pipeline.negatives import NegativeSampler, supply_report
from laya_pipeline.tasks import GenerationContext, QuestionGenerator, RoutingLabeler
from laya_pipeline.tasks.base import UNCERTAIN, clean_question, grounded_key_points
from laya_pipeline.tokenizer import EstimateTokenCounter

from tests.conftest import (
    FakeTransport,
    evidence_response,
    label_response,
    questions_response,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_chunks(count: int = 6) -> list[Chunk]:
    chunker = Chunker(_chunking_config(), EstimateTokenCounter(2.6))
    blocks = []
    for index in range(count):
        text = (
            f"# سند شماره {index}\n\n"
            f"## بخش {index}\n\n"
            f"متن یکتا دربارهٔ موضوع شمارهٔ {index} و مقدار {index * 7} واحد. "
            f"این جمله فقط در سند {index} وجود دارد و تکراری نیست.\n"
        )
        blocks.append(
            parse_markdown(
                DocumentSource(
                    path=Path(f"d{index}.md"),
                    relpath=f"d{index}.md",
                    content_id=stable_id("cnt", text),
                    byte_size=len(text.encode("utf-8")),
                    text=normalize_text(text),
                )
            )
        )
    return chunker.chunk_documents(blocks)


def _chunking_config() -> ChunkingConfig:
    return ChunkingConfig(min_chunk_chars=0, target_tokens=200, max_tokens=300)


def make_context(config: Config, transport: FakeTransport) -> GenerationContext:
    client = OllamaClient(
        config.ollama, transport=transport, cache=NullCache(), sleep=lambda _s: None
    )
    return GenerationContext(
        config=config,
        client=client,
        prompts=PromptLibrary(config.prompt_dir),
        rng=random.Random(config.project.seed),
        task="relevance",
    )


def positive(chunk: Chunk, query: str = "چه مقداری تعیین شده است؟") -> LabeledExample:
    return LabeledExample(
        example_id="",
        task="relevance",
        query=query,
        chunks=[
            ChunkRef(chunk_id=chunk.chunk_id, text=chunk.text, source_id=chunk.source_id)
        ],
        label="relevant",
        label_source="rule",
        role="positive",
        source_id=chunk.source_id,
        relpath=chunk.relpath,
        # A span that appears only in this chunk, so it guards against an answer
        # leak without rejecting every other chunk as a duplicate.
        key_points=[
            KeyPoint(
                text=f"مقدار {chunk.ordinal} واحد",
                span=f"مقدار {chunk.ordinal * 7} واحد",
                index=0,
            )
        ],
    )


# --------------------------------------------------------------------------- #
# Task A: relevance
# --------------------------------------------------------------------------- #
class TestRelevanceGeneration:
    def test_questions_and_key_points_are_captured(self, config: Config) -> None:
        # One response carries both questions; the two evidence checks follow.
        transport = FakeTransport(
            [
                questions_response(
                    ("چه مقداری تعیین شده است؟", "direct", "مقدار ۵ واحد"),
                    ("چند روز فرصت هست؟", "paraphrase", "دو روز"),
                ),
                evidence_response("yes", 0.95),
                evidence_response("yes", 0.9),
            ]
        )
        generator = _generator(config, transport)
        report = generator.generate(make_chunks(1), make_context(config, transport))
        assert len(report.examples) == 2
        assert report.examples[0].task == "relevance"
        assert report.examples[0].key_points[0].text == "مقدار ۵ واحد"
        assert [e.query_type for e in report.examples] == ["direct", "paraphrase"]

    def test_an_unanswerable_question_is_discarded(self, config: Config) -> None:
        transport = FakeTransport(
            [
                questions_response(
                    ("پرسش اول؟", "direct", "نکتهٔ اول"),
                    ("پرسش دوم؟", "direct", "نکتهٔ دوم"),
                ),
                evidence_response("no", 0.9),
                evidence_response("yes", 0.9),
            ]
        )
        generator = _generator(config, transport)
        report = generator.generate(make_chunks(1), make_context(config, transport))
        assert len(report.examples) == 1
        assert report.stats["questions_discarded_unanswerable"] == 1

    def test_an_uncertain_verdict_is_a_discard_not_a_label(self, config: Config) -> None:
        transport = FakeTransport(
            [
                questions_response(("پرسش اول؟", "direct", "نکتهٔ اول")),
                evidence_response(UNCERTAIN, 0.5),
            ]
        )
        generator = _generator(config, transport)
        report = generator.generate(make_chunks(1), make_context(config, transport))
        assert report.examples == []
        assert report.stats["questions_discarded_uncertain_evidence"] == 1

    def test_resume_skips_already_generated_chunks(self, config: Config) -> None:
        """A chunk with stored examples is not re-billed to the model."""
        chunks = make_chunks(1)
        stored = GeneratedExample(
            example_id="ex-stored", task="relevance", query="پرسش ذخیره‌شده؟",
            query_type="direct", source_chunk_id=chunks[0].chunk_id,
        )
        transport = FakeTransport([])
        generator = _generator(config, transport)
        report = generator.generate(
            chunks, make_context(config, transport),
            existing={chunks[0].chunk_id: [stored]},
        )
        assert report.stats["chunks_resumed"] == 1
        assert report.stats["examples_resumed"] == 1
        assert report.examples[0].example_id == "ex-stored"
        assert transport.calls == []

    def test_a_generation_failure_is_recorded_not_raised(self, config: Config) -> None:
        transport = FakeTransport([], fail_times=99)
        generator = _generator(config, transport)
        report = generator.generate(make_chunks(1), make_context(config, transport))
        assert report.examples == []
        assert report.failures and report.stats["chunks_failed"] == 1

    def test_short_chunks_are_skipped(self, config: Config) -> None:
        config.generation.min_chunk_chars = 10_000
        generator = QuestionGenerator(config, PromptLibrary(config.prompt_dir), _client(config, FakeTransport()))
        report = generator.generate(make_chunks(2), make_context(config, FakeTransport()))
        assert report.stats["chunks_too_short"] == 2
        assert report.examples == []

    def test_a_question_with_no_key_points_is_rejected(self, config: Config) -> None:
        # Rejected on key points alone, so no evidence call is made at all.
        transport = FakeTransport([questions_response(("پرسش اول؟", "direct"))])
        generator = _generator(config, transport)
        report = generator.generate(make_chunks(1), make_context(config, transport))
        assert report.examples == []
        assert report.stats["questions_rejected_no_key_points"] == 1
        # Exactly one call: the generation. No evidence check, because the
        # question was discarded before there was anything to verify.
        assert len(transport.calls) == 1


# --------------------------------------------------------------------------- #
# Task C: routing
# --------------------------------------------------------------------------- #
class TestRoutingLabeling:
    def _generated(self) -> list[GeneratedExample]:
        return [
            GeneratedExample(
                example_id="", task="relevance",
                query="سند HR-001 بند ۳ دربارهٔ چیست؟",
                query_type="direct", source_chunk_id="c1", source_id="s1",
            )
        ]

    def test_a_label_is_produced_without_chunks(self, config: Config) -> None:
        transport = FakeTransport([label_response("keyword", 0.9)])
        labeler = RoutingLabeler(config, PromptLibrary(config.prompt_dir), _client(config, transport))
        report = labeler.label(self._generated(), make_context(config, transport))
        assert len(report.examples) == 1
        assert report.examples[0].label == "keyword"
        assert report.examples[0].chunks == []

    def test_the_teacher_never_sees_the_source_document(self, config: Config) -> None:
        transport = FakeTransport([label_response("keyword", 0.9)])
        labeler = RoutingLabeler(config, PromptLibrary(config.prompt_dir), _client(config, transport))
        labeler.label(self._generated(), make_context(config, transport))
        prompt = transport.calls[0]["payload"]["messages"][-1]["content"]
        assert "c1" not in prompt
        assert "s1" not in prompt

    def test_uncertain_is_discarded(self, config: Config) -> None:
        transport = FakeTransport([label_response(UNCERTAIN, 0.4)])
        labeler = RoutingLabeler(config, PromptLibrary(config.prompt_dir), _client(config, transport))
        report = labeler.label(self._generated(), make_context(config, transport))
        assert report.examples == []
        assert report.stats["discarded_uncertain"] == 1


# --------------------------------------------------------------------------- #
# Negative mining
# --------------------------------------------------------------------------- #
class TestNegativeSampling:
    def _sampler(self, config: Config, transport: FakeTransport) -> NegativeSampler:
        return NegativeSampler(
            config, PromptLibrary(config.prompt_dir), _client(config, transport)
        )

    def test_mining_produces_verified_negatives(self, config: Config) -> None:
        chunks = make_chunks(6)
        pool = {c.chunk_id: c for c in chunks}
        transport = FakeTransport([evidence_response("no", 0.9)] * 40)
        report = self._sampler(config, transport).mine(
            [positive(chunks[0])], pool, make_context(config, transport)
        )
        assert report.examples
        assert all(e.label == "irrelevant" for e in report.examples)
        assert all(e.role == "negative" for e in report.examples)
        assert all(e.strategy in ("random", "lexical", "retrieval") for e in report.examples)

    def test_a_candidate_the_teacher_says_answers_is_rejected(self, config: Config) -> None:
        chunks = make_chunks(4)
        pool = {c.chunk_id: c for c in chunks}
        transport = FakeTransport([evidence_response("yes", 0.9)] * 40)
        report = self._sampler(config, transport).mine(
            [positive(chunks[0])], pool, make_context(config, transport)
        )
        assert report.examples == []
        assert report.stats.get("rejected_teacher_positive", 0) > 0

    def test_uncertain_candidates_are_rejected(self, config: Config) -> None:
        chunks = make_chunks(4)
        pool = {c.chunk_id: c for c in chunks}
        transport = FakeTransport([evidence_response(UNCERTAIN, 0.5)] * 40)
        report = self._sampler(config, transport).mine(
            [positive(chunks[0])], pool, make_context(config, transport)
        )
        assert report.examples == []
        assert report.stats.get("rejected_teacher_uncertain", 0) > 0

    def test_a_candidate_containing_a_key_point_is_rejected(self, config: Config) -> None:
        chunks = make_chunks(4)
        pool = {c.chunk_id: c for c in chunks}
        example = positive(chunks[0])
        # A key point whose span appears verbatim in another chunk is an answer leak.
        example.key_points = [KeyPoint(text="متن", span="تکراری نیست", index=0)]
        target = next(c for c in chunks if "تکراری نیست" in c.text and c.chunk_id != chunks[0].chunk_id)
        transport = FakeTransport([evidence_response("no", 0.9)] * 40)
        report = self._sampler(config, transport).mine([example], pool, make_context(config, transport))
        assert all(e.chunk_ids[0] != target.chunk_id for e in report.examples)

    def test_the_source_chunk_is_never_a_negative(self, config: Config) -> None:
        chunks = make_chunks(3)
        pool = {c.chunk_id: c for c in chunks}
        transport = FakeTransport([evidence_response("no", 0.9)] * 40)
        example = positive(chunks[0])
        report = self._sampler(config, transport).mine([example], pool, make_context(config, transport))
        assert all(e.chunk_ids[0] != chunks[0].chunk_id for e in report.examples)

    def test_a_zero_ratio_mines_nothing(self, config: Config) -> None:
        config.negatives.ratio = 0
        chunks = make_chunks(3)
        report = self._sampler(config, FakeTransport()).mine(
            [positive(chunks[0])], {c.chunk_id: c for c in chunks},
            make_context(config, FakeTransport()),
        )
        assert report.examples == []
        assert report.stats["disabled_ratio_zero"] == 1

    def test_an_empty_pool_is_an_error(self, config: Config) -> None:
        chunks = make_chunks(1)
        with pytest.raises(NegativeSamplingError, match="pool is empty"):
            self._sampler(config, FakeTransport()).mine(
                [positive(chunks[0])], {}, make_context(config, FakeTransport())
            )

    def test_non_relevance_tasks_are_skipped(self, config: Config) -> None:
        chunks = make_chunks(3)
        example = positive(chunks[0])
        example.task = "routing"
        report = self._sampler(config, FakeTransport()).mine(
            [example], {c.chunk_id: c for c in chunks}, make_context(config, FakeTransport())
        )
        assert report.examples == []
        assert report.stats["skipped_non_relevance_task"] == 1

    def test_supply_report_describes_the_corpus(self, config: Config) -> None:
        chunks = make_chunks(4)
        pool = {c.chunk_id: c for c in chunks}
        supply = supply_report([positive(chunks[0])], pool, config.negatives.ratio)
        assert supply["pool_chunks"] == 4
        assert supply["min_candidates"] == 3

    def test_mining_is_reproducible(self, config: Config) -> None:
        chunks = make_chunks(6)
        pool = {c.chunk_id: c for c in chunks}
        runs = []
        for _ in range(2):
            transport = FakeTransport([evidence_response("no", 0.9)] * 60)
            ctx = make_context(config, transport)
            runs.append(
                [e.chunk_ids[0] for e in self._sampler(config, transport).mine([positive(chunks[0])], pool, ctx).examples]
            )
        assert runs[0] == runs[1]


# --------------------------------------------------------------------------- #
# Labeling and review
# --------------------------------------------------------------------------- #
class TestLabeling:
    def test_human_review_precedes_teacher(self) -> None:
        assert strongest_label_source("rule", "human") == "human"
        assert strongest_label_source("teacher", "human") == "human"
        assert strongest_label_source("rule", "teacher") == "teacher"
        assert strongest_label_source(None, "rule") == "rule"

    def test_unknown_source_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            strongest_label_source("rule", "oracle")

    def test_positives_are_not_teacher_relabelled(self, config: Config) -> None:
        chunks = make_chunks(1)
        transport = FakeTransport([])
        report = propose_labels(
            [positive(chunks[0])], config, _client(config, transport),
            PromptLibrary(config.prompt_dir),
        )
        assert report.stats["kept_rule_labeled"] == 1
        assert transport.calls == []

    def test_a_negative_gets_teacher_labeled(self, config: Config) -> None:
        chunks = make_chunks(1)
        example = LabeledExample(
            example_id="", task="relevance", query="پرسش؟",
            chunks=[ChunkRef(chunk_id="x", text="متن نامرتبط")],
            label="irrelevant", label_source="teacher", role="negative",
        )
        transport = FakeTransport([label_response("irrelevant", 0.95)])
        report = propose_labels(
            [example], config, _client(config, transport), PromptLibrary(config.prompt_dir)
        )
        assert report.examples[0].label == "irrelevant"
        assert report.stats["accepted_automatically"] == 1

    def test_low_confidence_queues_for_review(self, config: Config) -> None:
        chunks = make_chunks(1)
        example = LabeledExample(
            example_id="", task="relevance", query="پرسش؟",
            chunks=[ChunkRef(chunk_id="x", text="متن")],
            label="irrelevant", label_source="teacher", role="negative",
        )
        transport = FakeTransport([label_response("irrelevant", 0.5)])
        report = propose_labels(
            [example], config, _client(config, transport), PromptLibrary(config.prompt_dir)
        )
        assert report.queue and report.queue[0].status == "uncertain"

    def test_contradicting_labels_are_detected(self, config: Config) -> None:
        chunks = make_chunks(1)
        base = positive(chunks[0])
        flipped = LabeledExample(
            example_id="other", task="relevance", query=base.query,
            chunks=[ChunkRef(chunk_id=base.chunk_ids[0], text=base.chunks[0].text)],
            label="irrelevant", label_source="teacher", role="negative",
        )
        found = detect_contradictions([base, flipped])
        assert len(found) == 1
        assert found[0]["labels"] == ["irrelevant", "relevant"]

    def test_corrections_win_and_preserve_the_original(self, config: Config) -> None:
        chunks = make_chunks(1)
        example = LabeledExample(
            example_id="ex-1", task="relevance", query="پرسش؟",
            chunks=[ChunkRef(chunk_id="x", text="متن")],
            label="irrelevant", label_source="teacher", confidence=0.6, role="negative",
        )
        rows = [{"example_id": "ex-1", "final_label": "relevant"}]
        updated, stats = apply_corrections(
            [example], rows, options_by_task={"relevance": ["relevant", "irrelevant"]}
        )
        assert stats["changed"] == 1
        assert updated[0].label == "relevant"
        assert updated[0].label_source == "human"
        assert updated[0].metadata["superseded_label"]["label"] == "irrelevant"

    def test_an_out_of_range_correction_is_ignored(self, config: Config) -> None:
        example = LabeledExample(
            example_id="ex-1", task="relevance", query="پرسش؟",
            chunks=[ChunkRef(chunk_id="x", text="متن")],
            label="relevant", label_source="teacher", role="positive",
        )
        rows = [{"example_id": "ex-1", "final_label": "maybe"}]
        updated, stats = apply_corrections(
            [example], rows, options_by_task={"relevance": ["relevant", "irrelevant"]}
        )
        assert stats["invalid_label"] == 1
        assert updated[0].label == "relevant"

    def test_review_queue_asks_for_a_second_opinion_on_negatives(self, config: Config) -> None:
        chunks = make_chunks(1)
        negative = LabeledExample(
            example_id="", task="relevance", query="پرسش؟",
            chunks=[ChunkRef(chunk_id="x", text="متن")],
            label="irrelevant", label_source="teacher", confidence=0.99, role="negative",
        )
        queue = build_review_queue([negative], config)
        assert queue and queue[0].status == "needs_second_opinion"

    def test_distribution_shows_zero_for_absent_options(self, config: Config) -> None:
        distribution = label_distribution([], {"relevance": ["relevant", "irrelevant"]})
        assert distribution["relevance"]["by_label"]["relevant"] == 0


# --------------------------------------------------------------------------- #
# Small unit helpers
# --------------------------------------------------------------------------- #
class TestHelpers:
    def test_clean_question_rejects_out_of_range(self) -> None:
        # Persian is measured in characters, and five characters is already a
        # plausible short question, so the bound is set above that.
        assert clean_question("کوتاه", min_chars=10, max_chars=100) is None
        assert clean_question("خیلی بلندتر از حد مجاز است", min_chars=5, max_chars=10) is None
        assert clean_question("  پرسش خوب؟  ", min_chars=5, max_chars=100) == "پرسش خوب؟"

    def test_clean_question_strips_surrounding_quotes(self) -> None:
        assert clean_question("«پرسش خوب؟»", min_chars=5, max_chars=100) == "پرسش خوب؟"

    def test_grounding_is_checked_against_the_source(self) -> None:
        points = grounded_key_points(["نکتهٔ یک"], ["عبارتی که نیست"], limit=4)
        assert points and points[0].span == ""

    def test_a_verified_span_is_kept(self) -> None:
        points = grounded_key_points(["نکتهٔ یک"], ["نکتهٔ یک"], limit=4)
        assert points[0].span == "نکتهٔ یک"

    def test_example_ids_are_deterministic(self) -> None:
        assert make_example_id("t", "q", "c") == make_example_id("t", "q", "c")
        assert make_example_id("t", "q", "c") != make_example_id("t", "q2", "c")

    def test_an_example_must_carry_a_known_label_source(self) -> None:
        with pytest.raises(ValueError):
            LabeledExample(
                example_id="", task="relevance", query="پرسش؟",
                chunks=[ChunkRef(chunk_id="x", text="t")],
                label="relevant", label_source="oracle",
            )

    def test_an_example_must_carry_a_known_split(self) -> None:
        with pytest.raises(ValueError):
            LabeledExample(
                example_id="", task="relevance", query="پرسش؟",
                chunks=[ChunkRef(chunk_id="x", text="t")],
                label="relevant", split="holdout",
            )


def _client(config: Config, transport: FakeTransport) -> OllamaClient:
    """A real client wired to the scripted transport — no retries, no cache.

    Built on the shipped ``OllamaConfig``, so the tests exercise the same
    endpoint, model and retry settings a real run would use.
    """
    return OllamaClient(
        config.ollama, transport=transport, cache=NullCache(), sleep=lambda _s: None
    )


def _generator(config: Config, transport: FakeTransport) -> QuestionGenerator:
    return QuestionGenerator(config, PromptLibrary(config.prompt_dir), _client(config, transport))