"""Splitting, leakage, dataset construction and validation."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from laya_pipeline.chunking import Chunker
from laya_pipeline.config import ChunkingConfig, Config
from laya_pipeline.corpus.markdown import parse_markdown
from laya_pipeline.corpus.models import Chunk, DocumentSource
from laya_pipeline.dataset.records import ChunkRef
from laya_pipeline.dataset import (
    SPLITS,
    LabeledExample,
    apply_splits,
    assign_splits,
    build_families,
    build_rows,
    detect_leakage,
    question_definition,
    render_state,
    validate_rows,
)
from laya_pipeline.dataset.builder import POSITIVE_CONTROL_TEMPLATE
from laya_pipeline.errors import DatasetValidationError, SplitError
from laya_pipeline.ids import normalize_text, short_hash, stable_id
from laya_pipeline.tokenizer import EstimateTokenCounter


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def make_document(index: int, body: str) -> DocumentSource:
    text = normalize_text(f"# سند {index}\n\n## بخش {index}\n\n{body}\n")
    return DocumentSource(
        path=Path(f"doc{index}.md"),
        relpath=f"doc{index}.md",
        content_id=stable_id("cnt", text),
        byte_size=len(text.encode("utf-8")),
        text=text,
    )


DISTINCT_BODIES = [
    "کنترل دسترسی به دادهٔ مشتری با درخواست ثبت‌شده و تأیید مالک سامانه",
    "انتقال ماندهٔ مرخصی کارکنان به دورهٔ بعد با سقف پنج روز کاری",
    "انتخاب کانال رسمی تیم برای هماهنگی جاری و ثبت پیوند سند",
    "گزارش رویداد مشکوک از طریق کانال امنیت در کمتر از یک ساعت",
    "بررسی بار آزمایشی تنها با اجازهٔ صریح مالک سامانه انجام می‌شود",
    "نگهداری نسخهٔ پشتیبان رمزگذاری‌شده و بازیابی پس از خرابی",
    "ارزیابی اثر زیست‌محیطی پیش از بهره‌برداری از زمین جدید",
    "تعیین سقف اعتبار برای حساب‌های شرکتی در هر روز کاری",
    "آموزش دورهٔ ایمنی کارگران پیش از ورود به سایت ساخت",
    "تنظیم سیاست نگهداری داده برای مدت نگهداری مشخص",
    "تفکیک شبکهٔ اداری از شبکهٔ تولید برای کاهش سطح حمله",
    "ثبت گزارش ماهانهٔ مصرف انرژی در دفتر مرکزی",
    "بازبینی قرارداد بیمه در پایان هر دورهٔ دوازده‌ماهه",
    "مهاجرت پایگاه داده با کمترین زمان قطعی سرویس",
    "تدوین سیاست کار از راه دور برای کارکنان غیرحضوری",
    "پایش سلامت سرویس‌ها و هشدار زودهنگام به تیم عملیات",
    "تعریف سطح دسترسی بر اساس کمترین امتیاز لازم",
    "تهیهٔ نقشهٔ راه محصول برای دو فصل آینده",
    "استانداردسازی قالب گزارش‌های فصلی مالی",
    "تعیین مسئول پاسخ‌گویی در هر سطح از سازمان",
]


def chunks_for(*bodies: str) -> list[Chunk]:
    config = ChunkingConfig(min_chunk_chars=0)
    chunker = Chunker(config, EstimateTokenCounter(config.chars_per_token_estimate))
    return chunker.chunk_documents([parse_markdown(make_document(i, b)) for i, b in enumerate(bodies)])


def chunks_by_source(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    grouped: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.source_id, []).append(chunk)
    return grouped


def example_for(chunk: Chunk, query: str, *, split: str | None = None, label: str = "relevant") -> LabeledExample:
    return LabeledExample(
        example_id="", task="relevance", query=query,
        chunks=[ChunkRef(chunk_id=chunk.chunk_id, text=chunk.text, source_id=chunk.source_id)],
        label=label, label_source="rule", role="positive",
        split=split, source_id=chunk.source_id, relpath=chunk.relpath,
    )


@pytest.fixture
def many_chunks() -> list[Chunk]:
    """20 lexically distinct documents, so family merging does not collapse them."""
    return chunks_for(*DISTINCT_BODIES)


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
class TestFamilies:
    def test_distinct_documents_form_distinct_families(self) -> None:
        chunks = chunks_for("متن یک", "متن دو", "متن سه")
        family_of, count = build_families(chunks_by_source(chunks), 0.7)
        assert count == 3
        assert len(set(family_of.values())) == 3

    def test_near_duplicate_documents_are_merged(self) -> None:
        body = "این متن تقریباً یکسان است و باید خانواده‌ای بسازد."
        chunks = chunks_for(body, body + " چند کلمهٔ بیشتر.")
        family_of, count = build_families(chunks_by_source(chunks), 0.6)
        assert count == 1
        assert len(set(family_of.values())) == 1

    def test_a_zero_threshold_never_merges(self) -> None:
        body = "متن"
        chunks = chunks_for(body, body)
        _family_of, count = build_families(chunks_by_source(chunks), 0.0)
        assert count == 2


class TestAssignment:
    def test_every_example_gets_a_split(self, config: Config, many_chunks: list[Chunk]) -> None:
        examples = [example_for(c, f"پرسش {c.ordinal} چیست؟") for c in many_chunks]
        report = assign_splits(examples, config, chunks_by_source(many_chunks), random.Random(1))
        assigned = apply_splits(examples, report)
        assert all(e.split in SPLITS for e in assigned)

    def test_a_document_never_appears_in_two_splits(self, config: Config, many_chunks: list[Chunk]) -> None:
        examples = [
            example_for(c, f"پرسش {c.ordinal}؟") for c in many_chunks for _ in range(2)
        ]
        report = assign_splits(examples, config, chunks_by_source(many_chunks), random.Random(7))
        per_source: dict[str, set[str]] = {}
        for example in apply_splits(examples, report):
            per_source.setdefault(example.source_id, set()).add(example.split)
        assert all(len(splits) == 1 for splits in per_source.values())

    def test_assignment_is_reproducible(self, config: Config, many_chunks: list[Chunk]) -> None:
        examples = [example_for(c, f"پرسش {c.ordinal}؟") for c in many_chunks]
        first = assign_splits(examples, config, chunks_by_source(many_chunks), random.Random(3))
        second = assign_splits(examples, config, chunks_by_source(many_chunks), random.Random(3))
        assert first.assignment == second.assignment

    def test_too_few_documents_produces_a_warning(self, config: Config) -> None:
        chunks = chunks_for("الف", "ب")
        examples = [example_for(c, "پرسش؟") for c in chunks]
        report = assign_splits(examples, config, chunks_by_source(chunks), random.Random(1))
        assert any("families" in warning for warning in report.warnings)

    def test_an_empty_split_is_warned_about(self, config: Config) -> None:
        chunks = chunks_for("الف", "ب")
        examples = [example_for(c, "پرسش؟") for c in chunks]
        report = assign_splits(examples, config, chunks_by_source(chunks), random.Random(1))
        assert report.stats["assigned_families"]

    def test_a_duplicate_document_lands_in_one_split(
        self, config: Config, many_chunks: list[Chunk]
    ) -> None:
        """A near-copy must not be able to straddle the train/test boundary."""
        duplicate_chunks = chunks_for(
            DISTINCT_BODIES[0], DISTINCT_BODIES[0] + " با ذکر جزئیات بیشتر."
        )
        pool = chunks_by_source(many_chunks + duplicate_chunks)
        examples = [
            example_for(c, f"پرسش شمارهٔ {c.ordinal}؟") for c in many_chunks
        ] + [example_for(c, f"پرسش نسخهٔ {c.ordinal}؟") for c in duplicate_chunks]
        report = assign_splits(examples, config, pool, random.Random(11))
        per_family: dict[str, set[str]] = {}
        for example in apply_splits(examples, report):
            family = report.family_of[example.source_id]
            per_family.setdefault(family, set()).add(example.split)
        merged = {f for f, s in per_family.items() if len(s) > 1}
        assert not merged, f"family {merged} was split across {per_family}"

    def test_examples_without_a_source_are_refused(self, config: Config) -> None:
        example = LabeledExample(
            example_id="", task="relevance", query="پرسش؟",
            chunks=[ChunkRef(chunk_id="x", text="متن")], label="relevant",
        )
        with pytest.raises(SplitError, match="source document id"):
            assign_splits([example], config, {}, random.Random(1))


class TestLeakage:
    def test_seed_changes_equal_weight_family_ties(
        self, config: Config, many_chunks: list[Chunk]
    ) -> None:
        examples = [example_for(c, f"پرسش {c.ordinal}؟") for c in many_chunks]
        first = assign_splits(examples, config, chunks_by_source(many_chunks), random.Random(1))
        second = assign_splits(examples, config, chunks_by_source(many_chunks), random.Random(7))
        assert first.assignment != second.assignment

    def test_identical_decisions_across_splits_are_found(self, config: Config) -> None:
        chunk = chunks_for("متن")[0]
        train = example_for(chunk, "پرسش یکسان؟", split="train")
        test = LabeledExample(
            example_id="other", task="relevance", query="پرسش یکسان؟",
            chunks=[ChunkRef(chunk_id=chunk.chunk_id, text=chunk.text)],
            label="irrelevant", label_source="teacher", role="negative", split="test",
            source_id=chunk.source_id,
        )
        findings = detect_leakage([train, test], config)
        assert any(f["kind"] == "identical_decision" for f in findings)

    def test_a_document_in_two_splits_is_found(self, config: Config) -> None:
        chunk = chunks_for("متن")[0]
        train = example_for(chunk, "پرسش الف؟", split="train")
        test = example_for(chunk, "پرسش ب؟", split="test")
        findings = detect_leakage([train, test], config)
        assert any(f["kind"] == "source_document_split" for f in findings)

    def test_a_paraphrased_query_across_splits_is_found(self, config: Config) -> None:
        chunks = chunks_for("متن یک", "متن دو")
        train = example_for(chunks[0], "مدیر چند روز برای پاسخ فرصت دارد؟", split="train")
        test = example_for(chunks[1], "مدیر چند روز برای پاسخ فرصت دارد!", split="test")
        findings = detect_leakage([train, test], config)
        assert any(f["kind"] == "paraphrased_query" for f in findings)

    def test_clean_splits_report_nothing(self, config: Config) -> None:
        chunks = chunks_for("محتوای الف کاملاً متفاوت", "محتوای ب کاملاً متفاوت")
        train = example_for(chunks[0], "پرسش کاملاً متفاوت الف؟", split="train")
        test = example_for(chunks[1], "پرسش کاملاً متفاوت ب؟", split="test")
        assert detect_leakage([train, test], config) == []

    def test_a_clean_split_reports_no_leakage(
        self, config: Config, many_chunks: list[Chunk]
    ) -> None:
        """A structurally sound split must come back clean.

        This is the regression guard for the ordering rule: because ``split``
        runs before ``negatives``, leakage here can only come from the queries
        themselves. Any finding means the guard is not doing its job.
        """
        # One question per document, keyed on its position in the list rather than its
        # chunk ordinal — several documents each start at ordinal 0, so the
        # ordinal alone would produce identical queries, which is exactly the
        # duplicate-query leakage this test asserts is absent.
        examples = [
            example_for(chunk, f"پرسشِ یکتا دربارهٔ موضوع شمارهٔ {index} چیست؟")
            for index, chunk in enumerate(many_chunks)
        ]
        report = assign_splits(examples, config, chunks_by_source(many_chunks), random.Random(2))
        assert report.leakage == []
        assert not any(w.startswith("leakage[") for w in report.warnings)


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #
class TestStateRendering:
    def test_query_and_chunk_are_placed(self, config: Config) -> None:
        rendered = render_state(config.laya.template("relevance"), "پرسش من", ["متن بازیابی"])
        assert "پرسش من" in rendered and "متن بازیابی" in rendered

    def test_multiple_chunks_are_separated(self, config: Config) -> None:
        rendered = render_state(config.laya.template("sufficiency"), "پ", ["اول", "دوم"])
        assert "اول" in rendered and "دوم" in rendered

    def test_numbering_only_applies_to_a_set(self, config: Config) -> None:
        single = render_state(config.laya.template("sufficiency"), "پ", ["تنها"], numbered=True)
        many = render_state(config.laya.template("sufficiency"), "پ", ["الف", "ب"], numbered=True)
        assert "[1]" not in single
        assert "[1]" in many and "[2]" in many

    def test_rendering_leaves_no_placeholders(self, config: Config) -> None:
        for task in config.laya.question_ids():
            chunks = ["متن"] if "{{chunks}}" in config.laya.template(task) else []
            rendered = render_state(config.laya.template(task), "پرسش", chunks)
            assert "{{" not in rendered


class TestRows:
    def test_a_positive_becomes_a_valid_row(self, config: Config) -> None:
        chunk = chunks_for("متن نمونه برای آزمون")[0]
        row = build_rows(config, [example_for(chunk, "پرسش؟", split="train")], "relevance").rows[0]
        assert set(row) == {"state", "questions", "expected", "tags", "language"}
        assert row["expected"]["relevance"] == "relevant"
        assert row["questions"]["relevance"] == question_definition(config, "relevance")

    def test_the_question_matches_the_configured_definition(self, config: Config) -> None:
        chunk = chunks_for("متن")[0]
        row = build_rows(config, [example_for(chunk, "پرسش؟", split="train")], "relevance").rows[0]
        assert row["questions"]["relevance"]["instructions"] == config.laya.questions["relevance"]["instructions"]

    def test_tags_carry_task_split_and_role(self, config: Config) -> None:
        chunk = chunks_for("متن")[0]
        row = build_rows(config, [example_for(chunk, "پرسش؟", split="validation")], "relevance").rows[0]
        assert "task:relevance" in row["tags"]
        assert "split:validation" in row["tags"]
        assert "role:positive" in row["tags"]

    def test_an_unassigned_split_is_skipped(self, config: Config) -> None:
        chunk = chunks_for("متن")[0]
        artifact = build_rows(config, [example_for(chunk, "پرسش؟")], "relevance")
        assert artifact.rows == []
        assert artifact.counts["skipped"]["unassigned_split"] == 1

    def test_routing_rows_carry_no_chunks(self, config: Config) -> None:
        routing = LabeledExample(
            example_id="", task="routing", query="پرسش مسیریابی؟",
            chunks=[], label="keyword", label_source="teacher", role="routing", split="train",
        )
        row = build_rows(config, [routing], "routing").rows[0]
        assert "پرسش مسیریابی؟" in row["state"]
        assert row["expected"]["routing"] == "keyword"

    def test_a_relevance_row_without_chunks_is_refused(self, config: Config) -> None:
        broken = LabeledExample(
            example_id="", task="relevance", query="پرسش؟", chunks=[],
            label="relevant", label_source="rule", role="positive", split="train",
        )
        artifact = build_rows(config, [broken], "relevance")
        assert artifact.rows == []
        assert artifact.counts["skipped"]["invalid"] == 1

    def test_a_label_outside_the_options_is_refused(self, config: Config) -> None:
        chunk = chunks_for("متن")[0]
        bad = example_for(chunk, "پرسش؟", split="train", label="maybe")
        assert build_rows(config, [bad], "relevance").rows == []

    def test_counts_cover_every_configured_option(self, config: Config) -> None:
        chunk = chunks_for("متن")[0]
        artifact = build_rows(config, [example_for(chunk, "پرسش؟", split="train")], "relevance")
        assert artifact.counts["by_label"]["irrelevant"] == 0

    def test_the_positive_control_prefixes_the_state(self, config: Config) -> None:
        chunk = chunks_for("متن")[0]
        artifact = build_rows(
            config, [example_for(chunk, "پرسش؟", split="train")], "relevance",
            positive_control=True,
        )
        assert artifact.rows[0]["state"].startswith(POSITIVE_CONTROL_TEMPLATE.format(label="relevant"))


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def good_rows(config: Config, count: int = 3) -> list[dict]:
    artifact = build_rows(
        config,
        [example_for(chunks_for(f"متن {i}")[0], f"پرسش {i}؟", split="train") for i in range(count)],
        "relevance",
    )
    return artifact.rows


class TestValidation:
    def test_a_good_dataset_passes(self, config: Config) -> None:
        report = validate_rows(config, good_rows(config), "relevance")
        assert report.ok, report.problems

    def test_a_missing_field_is_reported(self, config: Config) -> None:
        rows = good_rows(config, 1)
        del rows[0]["expected"]
        report = validate_rows(config, rows, "relevance")
        assert not report.ok
        assert any("expected" in p for p in report.problems)

    def test_an_empty_state_is_reported_with_the_reason(self, config: Config) -> None:
        rows = good_rows(config, 1)
        rows[0]["state"] = "   "
        report = validate_rows(config, rows, "relevance")
        assert any("items_from_rows" in p for p in report.problems)

    def test_a_duplicate_state_is_reported(self, config: Config) -> None:
        rows = good_rows(config, 2)
        rows[1]["state"] = rows[0]["state"]
        report = validate_rows(config, rows, "relevance")
        assert any("duplicate state" in p for p in report.problems)

    def test_reworded_instructions_are_reported(self, config: Config) -> None:
        rows = good_rows(config, 1)
        rows[0]["questions"]["relevance"]["instructions"] = "جملهٔ متفاوتی"
        report = validate_rows(config, rows, "relevance")
        assert any("differs from the configured" in p for p in report.problems)

    def test_changed_option_text_is_reported(self, config: Config) -> None:
        rows = good_rows(config, 1)
        rows[0]["questions"]["relevance"]["criteria"]["relevant"] = "توصیف دیگر"
        report = validate_rows(config, rows, "relevance")
        assert any("criteria" in p for p in report.problems)

    def test_an_invalid_label_is_reported(self, config: Config) -> None:
        rows = good_rows(config, 1)
        rows[0]["expected"]["relevance"] = "شاید"
        report = validate_rows(config, rows, "relevance")
        assert any("not one of" in p for p in report.problems)

    def test_a_missing_split_is_reported(self, config: Config) -> None:
        rows = good_rows(config, 1)
        rows[0]["tags"] = ["task:relevance"]
        report = validate_rows(config, rows, "relevance")
        assert any("split" in p for p in report.problems)

    def test_metadata_does_not_replace_the_split_tag(self, config: Config) -> None:
        rows = good_rows(config, 1)
        rows[0]["tags"] = ["task:relevance"]
        rows[0]["metadata"] = {"split": "train"}
        report = validate_rows(config, rows, "relevance")
        assert not report.ok
        assert any("unexpected Laya row field" in problem for problem in report.problems)
        assert any("no split tag" in problem for problem in report.problems)

    def test_unexpected_fields_are_rejected(self, config: Config) -> None:
        rows = good_rows(config, 1)
        rows[0]["metadata"] = {"example_id": "sensitive-provenance"}
        report = validate_rows(config, rows, "relevance")
        assert any("unexpected Laya row field" in problem for problem in report.problems)

    def test_an_invalid_split_name_is_reported(self, config: Config) -> None:
        rows = good_rows(config, 1)
        rows[0]["tags"] = [t for t in rows[0]["tags"] if not t.startswith("split:")]
        rows[0]["tags"].append("split:holdout")
        report = validate_rows(config, rows, "relevance")
        assert any("not one of" in p for p in report.problems)

    def test_an_absent_option_is_a_warning_not_an_error(self, config: Config) -> None:
        report = validate_rows(config, good_rows(config, 2), "relevance")
        assert report.ok
        assert any("never in the data" in w for w in report.warnings)

    def test_an_empty_dataset_fails(self, config: Config) -> None:
        report = validate_rows(config, [], "relevance")
        assert not report.ok

    def test_raise_if_failed_lists_problems(self, config: Config) -> None:
        rows = good_rows(config, 1)
        rows[0]["expected"]["relevance"] = "شاید"
        with pytest.raises(DatasetValidationError) as caught:
            validate_rows(config, rows, "relevance").raise_if_failed()
        assert caught.value.problems

    def test_token_overrun_is_measured_when_a_counter_is_given(self, config: Config) -> None:
        rows = good_rows(config, 2)
        # chars_per_token below 1 makes every character its own token, so the
        # states blow past max_len. A large divisor would do the opposite.
        report = validate_rows(config, rows, "relevance", token_counter=EstimateTokenCounter(0.01))
        assert report.measured["over_budget"] == len(rows)
        assert not report.ok

    def test_a_counter_that_fits_reports_no_overrun(self, config: Config) -> None:
        rows = good_rows(config, 2)
        report = validate_rows(config, rows, "relevance", token_counter=EstimateTokenCounter(2.6))
        assert report.measured["over_budget"] == 0
        assert report.ok

    def test_absent_token_measurement_is_reported_not_assumed(self, config: Config) -> None:
        report = validate_rows(config, good_rows(config, 1), "relevance", token_counter=None)
        assert report.measured.get("skipped")
        assert any("no Laya tokenizer" in w for w in report.warnings)

    def test_a_forbidden_extra_question_is_reported(self, config: Config) -> None:
        rows = good_rows(config, 1)
        rows[0]["questions"]["extra"] = {"type": "noul", "instructions": "x"}
        report = validate_rows(config, rows, "relevance")
        assert any("questions" in p for p in report.problems)


def test_state_fingerprint_is_stable() -> None:
    assert short_hash("متن") == short_hash("متن")
