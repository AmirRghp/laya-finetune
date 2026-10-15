"""Typed configuration loaded from YAML.

Design notes
------------
* Config is plain stdlib dataclasses. An unknown key is an error, not a silent
  no-op — a typo in a key name is the most common way a pipeline quietly runs
  with the wrong settings (AGENTS.md 5.5).
* ``configs/default.yaml`` holds every value. ``src/`` holds only structural
  defaults and raises rather than repairs.
* Environment overrides are declared in the config itself (``env_overrides``),
  so there is one list rather than two that drift apart.
* Relative paths resolve against ``project.root``, which defaults to the current
  working directory. Run commands from the repository root.
"""

from __future__ import annotations

import json
import os
import re
import types
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

#: A dotenv key. Restricted on purpose: a line like ``export FOO=1`` or a stray
#: prose line is skipped rather than becoming a variable nobody can unset.
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
from urllib.parse import urlparse

from .errors import ConfigError
from .ids import content_hash

__all__ = [
    "Config",
    "ProjectConfig",
    "PathsConfig",
    "ChunkingConfig",
    "OllamaConfig",
    "GenerationConfig",
    "NegativesConfig",
    "LabelingConfig",
    "SplittingConfig",
    "LayaConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "load_training_profile_overrides",
    "load_config",
    "config_hash",
]

QUESTION_TYPES = ("choice", "noul", "score")
OPTION_LAYOUTS = ("sequential", "parallel")
NEGATIVE_STRATEGIES = ("random", "lexical", "retrieval")
RETRIEVAL_BACKENDS = ("ollama", "laya_encoder", "lexical")
LOSSES = ("rlcd", "soft-ce")
SPLIT_NAMES = ("train", "validation", "test")


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
@dataclass
class ProjectConfig:
    name: str = "laya-persian-rag"
    seed: int = 20260922
    root: Optional[str] = None


@dataclass
class PathsConfig:
    corpus_dir: str = "corpus"
    output_dir: str = "outputs"
    prompt_dir: str = "prompts"


@dataclass
class ChunkingConfig:
    tokenizer_model: str = "convaiinnovations/laya-multilingual"
    tokenizer_max_length: int = 8192
    target_tokens: int = 320
    max_tokens: int = 448
    min_tokens: int = 48
    overlap_blocks: int = 1
    include_heading_path: bool = True
    include_section_title: bool = True
    chars_per_token_estimate: float = 2.6
    min_chunk_chars: int = 120


@dataclass
class OllamaConfig:
    endpoint: str = "http://127.0.0.1:11434"
    generation_model: str = "qwen2.5:7b-instruct"
    embedding_model: str = "bge-m3"
    teacher_model: str = "qwen2.5:7b-instruct"
    timeout_seconds: float = 180.0
    connect_timeout_seconds: float = 10.0
    max_retries: int = 3
    retry_backoff_seconds: float = 1.5
    num_ctx: int = 8192
    keep_alive: str = "10m"
    #: Send ``think: false`` to the model.
    #:
    #: Reasoning models (the Qwen3 family and others) emit a long chain of thought
    #: into a separate field and only then answer. For a *reasoning* budget that
    #: can be a feature; for structured extraction it is fatal in practice — a
    #: 2B model burned 4060 tokens thinking about a one-line question and hit the
    #: context limit with ``done_reason: "length"`` and an **empty** content
    #: field, which surfaced as "model response contains no JSON object: ''".
    #:
    #: Turning it off costs nothing here: the prompts ask for extraction and
    #: classification, not for deliberation. Set to ``null`` to let the model
    #: think, which is the right choice for a large reasoning model on a task
    #: that genuinely needs deliberation.
    think: Optional[bool] = False
    cache_dir: str = "outputs/cache/ollama"


@dataclass
class GenerationConfig:
    questions_per_chunk: int = 3
    max_key_points: int = 4
    min_chunk_chars: int = 200
    min_question_chars: int = 15
    max_question_chars: int = 320
    temperature: float = 0.7
    top_p: float = 0.9
    seed: int = 20260922
    resume: bool = True
    workers: int = 1
    # Confirm with the teacher that each generated question is actually
    # answerable from its source chunk before it becomes a positive example.
    # Without this, "relevant" is an assumption rather than a finding.
    verify_positives: bool = True
    question_types: List[str] = field(
        default_factory=lambda: [
            "direct",
            "paraphrase",
            "conversational",
            "terminology",
            "alternative",
            "hard",
        ]
    )


@dataclass
class NegativesConfig:
    ratio: float = 1.0
    strategies: List[str] = field(default_factory=lambda: list(NEGATIVE_STRATEGIES))
    strategy_weights: List[float] = field(default_factory=lambda: [0.34, 0.33, 0.33])
    candidates_per_positive: int = 12
    min_similarity: float = 0.25
    max_similarity: float = 0.92
    near_duplicate_threshold: float = 0.85
    lexical_candidates: int = 8
    verify_with_teacher: bool = True


@dataclass
class LabelingConfig:
    review_threshold: float = 0.80
    uncertain_low: float = 0.35
    uncertain_high: float = 0.65
    temperature: float = 0.0


@dataclass
class SplittingConfig:
    seed: int = 20260922
    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15
    family_similarity_threshold: float = 0.70
    query_leakage_threshold: float = 0.90
    paraphrase_ngram: int = 3
    warn_below_groups: int = 10


@dataclass
class LayaConfig:
    base_model: str = "convaiinnovations/laya-multilingual"
    max_len: int = 1024
    head_max_len: int = 256
    max_options: int = 8
    language: str = "fa"
    state_templates: Dict[str, str] = field(default_factory=dict)
    questions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    #: Which option of a task's question plays which role when the pipeline
    #: builds a positive or a negative. Naming the role here rather than in code
    #: means the label strings live with the question definition they belong to.
    label_roles: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def question_ids(self) -> List[str]:
        return sorted(self.questions)

    def question(self, task: str) -> Dict[str, Any]:
        try:
            return self.questions[task]
        except KeyError:
            raise ConfigError(
                f"no Laya question defined for task {task!r}; "
                f"configured tasks: {', '.join(self.question_ids()) or '(none)'}"
            ) from None

    def template(self, task: str) -> str:
        try:
            return self.state_templates[task]
        except KeyError:
            raise ConfigError(
                f"no state template defined for task {task!r} under laya.state_templates"
            ) from None

    def options(self, task: str) -> List[str]:
        return list(self.question(task).get("criteria", {}).keys())

    def role_label(self, task: str, role: str) -> str:
        """The option string that plays ``role`` for ``task``.

        Raises rather than defaulting: a task whose positive label is not
        configured should stop the build, not silently emit inverted labels.
        """
        roles = self.label_roles.get(task) or {}
        label = roles.get(role)
        if not label:
            raise ConfigError(
                f"laya.label_roles.{task}.{role} is not configured; the pipeline needs "
                "to know which option is the positive one"
            )
        options = self.options(task)
        if label not in options:
            raise ConfigError(
                f"laya.label_roles.{task}.{role} is {label!r}, which is not an option "
                f"of the {task!r} question (options: {', '.join(options)})"
            )
        return label


@dataclass
class TrainingConfig:
    """Mirrors ``laya.train.TrainConfig``. See :meth:`to_train_config`."""

    epochs: int = 8
    micro_batch: int = 4
    grad_accum: int = 16
    encoder_lr: float = 2.5e-5
    head_lr: float = 1.0e-4
    min_lr: float = 1.0e-6
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    loss: str = "rlcd"
    rl_samples: int = 4
    sigma_start: float = 0.4
    sigma_end: float = 0.1
    w_sph: float = 0.75
    w_rps: float = 1.0
    shuffle_options: List[str] = field(default_factory=list)
    option_layout: Optional[str] = None
    freeze_encoder: bool = False
    calib_max: int = 200
    calib_frac: float = 0.1
    calib_seed: int = 20260922
    seed: int = 20260922
    max_len: Optional[int] = None
    head_max_len: Optional[int] = None
    amp: Optional[bool] = None
    gradient_checkpointing: Optional[bool] = None
    log_every: int = 50
    label_smoothing: float = 0.0
    target_error: float = 0.10
    min_abstain_n: int = 10
    device: str = "auto"
    # Not a TrainConfig field: set by the train stage to the validation split so
    # Laya reports genuinely held-out before/after metrics (AGENTS.md 3.9).
    eval_data: Optional[str] = None

    _LAYA_FIELDS = (
        "epochs",
        "micro_batch",
        "grad_accum",
        "encoder_lr",
        "head_lr",
        "min_lr",
        "weight_decay",
        "grad_clip",
        "loss",
        "rl_samples",
        "sigma_start",
        "sigma_end",
        "w_sph",
        "w_rps",
        "shuffle_options",
        "option_layout",
        "freeze_encoder",
        "calib_max",
        "calib_frac",
        "calib_seed",
        "seed",
        "max_len",
        "head_max_len",
        "amp",
        "gradient_checkpointing",
        "log_every",
        "label_smoothing",
        "target_error",
        "min_abstain_n",
        "eval_data",
    )

    def to_train_config(self, max_len: int, head_max_len: int) -> Dict[str, Any]:
        """Keyword arguments for ``laya.train.TrainConfig``.

        ``max_len``/``head_max_len`` default to the checkpoint's own budgets; the
        stage passes the values it validated against so training and inference
        agree on the token budgets.
        """
        kwargs = {name: getattr(self, name) for name in self._LAYA_FIELDS}
        kwargs["shuffle_options"] = tuple(self.shuffle_options)
        kwargs["max_len"] = self.max_len or max_len
        kwargs["head_max_len"] = self.head_max_len or head_max_len
        return kwargs

    def effective_batch(self) -> int:
        return self.micro_batch * self.grad_accum

    def optimizer_updates(self, train_items: int) -> int:
        """``epochs * ceil(ceil(items / micro_batch) / grad_accum)``.

        The same formula Laya documents, reimplemented so the budget can be
        printed *before* a run that costs GPU hours (AGENTS.md 3.13).
        """
        if train_items <= 0:
            return 0
        micro_batches = -(-train_items // self.micro_batch)
        updates_per_epoch = -(-micro_batches // self.grad_accum)
        return self.epochs * updates_per_epoch


@dataclass
class EvaluationConfig:
    batch_size: int = 16
    latency_repeats: int = 1
    retrieval_ks: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    retrieval_backend: str = "ollama"
    base_model: str = "convaiinnovations/laya-multilingual"
    fine_tuned_model: Optional[str] = None


@dataclass
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    negatives: NegativesConfig = field(default_factory=NegativesConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    splitting: SplittingConfig = field(default_factory=SplittingConfig)
    laya: LayaConfig = field(default_factory=LayaConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    env_overrides: Dict[str, str] = field(default_factory=dict)
    source_path: Optional[str] = None

    # -- paths ------------------------------------------------------------- #
    @property
    def root(self) -> Path:
        return Path(self.project.root).expanduser().resolve() if self.project.root else Path.cwd()

    @property
    def corpus_dir(self) -> Path:
        return self._resolve(self.paths.corpus_dir)

    @property
    def output_dir(self) -> Path:
        return self._resolve(self.paths.output_dir)

    @property
    def prompt_dir(self) -> Path:
        return self._resolve(self.paths.prompt_dir)

    def _resolve(self, value: str) -> Path:
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else (self.root / candidate).resolve()

    def stage_dir(self, name: str) -> Path:
        """Where a stage writes its artifacts. Always outside the corpus."""
        return self.output_dir / name

    # -- serialisation ----------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        payload = {
            f.name: _plain(getattr(self, f.name))
            for f in fields(self)
            if not f.name.startswith("_")
        }
        return payload

    def with_overrides(self, overrides: Mapping[str, Any]) -> "Config":
        """Return a copy with dotted-key overrides applied and revalidated."""
        merged = self.to_dict()
        for dotted, value in overrides.items():
            _set_dotted(merged, dotted, value)
        rebuilt = _build(Config, merged, "config")
        rebuilt.source_path = self.source_path
        rebuilt.validate()
        return rebuilt

    # -- validation -------------------------------------------------------- #
    def validate(self) -> None:
        validate_project(self.project)
        validate_paths(self.paths)
        validate_chunking(self.chunking, self.laya)
        validate_ollama(self.ollama)
        validate_generation(self.generation)
        validate_negatives(self.negatives)
        validate_labeling(self.labeling)
        validate_splitting(self.splitting)
        validate_laya(self.laya)
        validate_training(self.training, self.laya)
        validate_evaluation(self.evaluation)

    @property
    def config_hash(self) -> str:
        """Stable hash of the resolved configuration, recorded in every manifest."""
        return config_hash(self.to_dict())


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _positive(value: Any, name: str, *, allow_zero: bool = False) -> None:
    floor = 0 if allow_zero else 1
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= floor,
        f"{name} must be an integer >= {floor}, got {value!r}",
    )


def _fraction(value: Any, name: str, low: float = 0.0, high: float = 1.0) -> None:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool) and low <= value <= high,
        f"{name} must be in [{low}, {high}], got {value!r}",
    )


def validate_project(section: ProjectConfig) -> None:
    _require(bool(section.name.strip()), "project.name must not be empty")
    _positive(section.seed, "project.seed", allow_zero=True)


def validate_paths(section: PathsConfig) -> None:
    for name in ("corpus_dir", "output_dir", "prompt_dir"):
        value = getattr(section, name)
        _require(
            isinstance(value, str) and bool(value.strip()),
            f"paths.{name} must be a non-empty path",
        )


def validate_chunking(section: ChunkingConfig, laya: LayaConfig) -> None:
    _positive(section.tokenizer_max_length, "chunking.tokenizer_max_length")
    _positive(section.target_tokens, "chunking.target_tokens")
    _positive(section.max_tokens, "chunking.max_tokens")
    _positive(section.min_tokens, "chunking.min_tokens", allow_zero=True)
    _positive(section.overlap_blocks, "chunking.overlap_blocks", allow_zero=True)
    _require(
        section.min_tokens <= section.target_tokens <= section.max_tokens,
        "chunking requires min_tokens <= target_tokens <= max_tokens, got "
        f"{section.min_tokens} <= {section.target_tokens} <= {section.max_tokens}",
    )
    _require(
        section.max_tokens <= laya.max_len,
        f"chunking.max_tokens ({section.max_tokens}) exceeds laya.max_len "
        f"({laya.max_len}); chunks that do not fit the state budget get truncated",
    )
    _require(
        section.tokenizer_max_length >= laya.max_len,
        f"chunking.tokenizer_max_length ({section.tokenizer_max_length}) is below "
        f"laya.max_len ({laya.max_len})",
    )
    _require(
        section.chars_per_token_estimate > 0,
        "chunking.chars_per_token_estimate must be positive",
    )
    _positive(section.min_chunk_chars, "chunking.min_chunk_chars", allow_zero=True)


def validate_ollama(section: OllamaConfig) -> None:
    parsed = urlparse(section.endpoint)
    _require(
        parsed.scheme in ("http", "https") and bool(parsed.netloc),
        f"ollama.endpoint must be an http(s) URL with a host, got {section.endpoint!r}",
    )
    for name in ("generation_model", "embedding_model", "teacher_model"):
        value = getattr(section, name)
        _require(
            isinstance(value, str) and bool(value.strip()),
            f"ollama.{name} must name a model; none is hardcoded by design",
        )
    _require(section.timeout_seconds > 0, "ollama.timeout_seconds must be positive")
    _require(
        section.connect_timeout_seconds > 0,
        "ollama.connect_timeout_seconds must be positive",
    )
    _positive(section.max_retries, "ollama.max_retries")
    _require(section.retry_backoff_seconds >= 0, "ollama.retry_backoff_seconds must be >= 0")
    _positive(section.num_ctx, "ollama.num_ctx")
    if section.think is not None and not isinstance(section.think, bool):
        raise ConfigError(f"ollama.think must be true, false or null, got {section.think!r}")


def validate_generation(section: GenerationConfig) -> None:
    _positive(section.questions_per_chunk, "generation.questions_per_chunk")
    _positive(section.max_key_points, "generation.max_key_points")
    _require(section.min_chunk_chars >= 0, "generation.min_chunk_chars must be >= 0")
    _positive(section.workers, "generation.workers")
    _require(
        section.min_question_chars < section.max_question_chars,
        "generation requires min_question_chars < max_question_chars, got "
        f"{section.min_question_chars} >= {section.max_question_chars}",
    )
    _require(0.0 <= section.temperature <= 2.0, "generation.temperature must be in [0, 2]")
    _require(0.0 < section.top_p <= 1.0, "generation.top_p must be in (0, 1]")
    _positive(section.seed, "generation.seed", allow_zero=True)
    known = {
        "direct",
        "paraphrase",
        "conversational",
        "terminology",
        "alternative",
        "hard",
    }
    unknown = sorted(set(section.question_types) - known)
    _require(
        not unknown,
        f"generation.question_types has unknown entries: {', '.join(unknown)}; "
        f"known: {', '.join(sorted(known))}",
    )
    _require(
        len(section.question_types) > 0,
        "generation.question_types must not be empty",
    )


def validate_negatives(section: NegativesConfig) -> None:
    _require(section.ratio >= 0, "negatives.ratio must be >= 0")
    unknown = sorted(set(section.strategies) - set(NEGATIVE_STRATEGIES))
    _require(
        not unknown,
        f"negatives.strategies has unknown entries: {', '.join(unknown)}; "
        f"known: {', '.join(NEGATIVE_STRATEGIES)}",
    )
    _require(len(section.strategies) > 0, "negatives.strategies must not be empty")
    _require(
        len(section.strategy_weights) == len(section.strategies),
        f"negatives.strategy_weights has {len(section.strategy_weights)} entries but "
        f"strategies has {len(section.strategies)}",
    )
    for index, weight in enumerate(section.strategy_weights):
        _require(
            isinstance(weight, (int, float)) and not isinstance(weight, bool) and weight >= 0,
            f"negatives.strategy_weights[{index}] must be >= 0, got {weight!r}",
        )
    total = sum(section.strategy_weights)
    _require(
        abs(total - 1.0) < 1e-6,
        f"negatives.strategy_weights must sum to 1.0, got {total:.6f}",
    )
    _positive(section.candidates_per_positive, "negatives.candidates_per_positive")
    _fraction(section.min_similarity, "negatives.min_similarity")
    _fraction(section.max_similarity, "negatives.max_similarity")
    _require(
        section.min_similarity <= section.max_similarity,
        f"negatives requires min_similarity ({section.min_similarity}) <= "
        f"max_similarity ({section.max_similarity})",
    )
    _fraction(section.near_duplicate_threshold, "negatives.near_duplicate_threshold")
    _positive(section.lexical_candidates, "negatives.lexical_candidates", allow_zero=True)


def validate_labeling(section: LabelingConfig) -> None:
    _fraction(section.review_threshold, "labeling.review_threshold")
    _fraction(section.uncertain_low, "labeling.uncertain_low")
    _fraction(section.uncertain_high, "labeling.uncertain_high")
    _require(
        section.uncertain_low < section.uncertain_high,
        f"labeling requires uncertain_low ({section.uncertain_low}) < "
        f"uncertain_high ({section.uncertain_high})",
    )
    _require(
        section.uncertain_high <= section.review_threshold,
        f"labeling.uncertain_high ({section.uncertain_high}) must be <= "
        f"review_threshold ({section.review_threshold}) so confidence bands have "
        "stable review categories",
    )
    _require(0.0 <= section.temperature <= 2.0, "labeling.temperature must be in [0, 2]")


def validate_splitting(section: SplittingConfig) -> None:
    ratios = {name: getattr(section, name) for name in SPLIT_NAMES}
    for name, value in ratios.items():
        _fraction(value, f"splitting.{name}")
    total = sum(ratios.values())
    _require(
        abs(total - 1.0) < 1e-6,
        f"splitting ratios must sum to 1.0, got {total:.6f} "
        f"({', '.join(f'{k}={v}' for k, v in ratios.items())})",
    )
    _positive(section.seed, "splitting.seed", allow_zero=True)
    _fraction(section.family_similarity_threshold, "splitting.family_similarity_threshold")
    _fraction(section.query_leakage_threshold, "splitting.query_leakage_threshold")
    _positive(section.paraphrase_ngram, "splitting.paraphrase_ngram")
    _positive(section.warn_below_groups, "splitting.warn_below_groups")


def validate_laya(section: LayaConfig) -> None:
    _require(
        isinstance(section.base_model, str) and bool(section.base_model.strip()),
        "laya.base_model must name a checkpoint",
    )
    _positive(section.max_len, "laya.max_len")
    _positive(section.head_max_len, "laya.head_max_len")
    _require(
        section.head_max_len <= section.max_len,
        f"laya.head_max_len ({section.head_max_len}) must be <= laya.max_len "
        f"({section.max_len})",
    )
    _positive(section.max_options, "laya.max_options")
    _require(
        section.max_options >= 2,
        "laya.max_options must be at least 2 for a meaningful choice question",
    )
    _require(
        bool(section.questions),
        "laya.questions is empty; nothing can be built without at least one question",
    )
    for task, definition in section.questions.items():
        where = f"laya.questions.{task}"
        _require(isinstance(definition, dict), f"{where} must be a mapping")
        qtype = definition.get("type")
        _require(
            qtype in QUESTION_TYPES,
            f"{where}.type must be one of {QUESTION_TYPES}, got {qtype!r}",
        )
        instructions = definition.get("instructions")
        _require(
            isinstance(instructions, str) and bool(instructions.strip()),
            f"{where}.instructions must be a non-empty string; Laya rejects a question "
            "without it",
        )
        if qtype == "choice":
            criteria = definition.get("criteria")
            _require(
                isinstance(criteria, dict) and len(criteria) >= 1,
                f"{where}.criteria must be a non-empty mapping of label -> description "
                "for a choice question",
            )
            for label, description in criteria.items():
                _require(
                    isinstance(label, str) and bool(label.strip()),
                    f"{where}.criteria has a non-string or empty label: {label!r}",
                )
                _require(
                    isinstance(description, str) and bool(description.strip()),
                    f"{where}.criteria[{label!r}] must be a non-empty description; it is "
                    "the option text the model reads",
                )
            _require(
                len(criteria) <= section.max_options,
                f"{where} has {len(criteria)} options, above laya.max_options "
                f"({section.max_options})",
            )
        _require(
            task in section.state_templates,
            f"no laya.state_templates entry for task {task!r}",
        )
    orphan = sorted(set(section.state_templates) - set(section.questions))
    _require(
        not orphan,
        f"laya.state_templates has entries with no question: {', '.join(orphan)}",
    )
    valid_roles = {"positive", "negative"}
    for task, roles in section.label_roles.items():
        where = f"laya.label_roles.{task}"
        _require(task in section.questions, f"{where} has no matching question")
        _require(isinstance(roles, dict) and bool(roles), f"{where} must be a non-empty mapping")
        options = list(section.questions[task].get("criteria", {}).keys())
        for role, label in roles.items():
            _require(
                role in valid_roles,
                f"{where}.{role} must be one of {sorted(valid_roles)}, got {role!r}",
            )
            _require(
                label in options,
                f"{where}.{role} is {label!r}, which is not an option of the {task!r} "
                f"question (options: {', '.join(options)})",
            )
        _require(
            roles.get("positive") != roles.get("negative") if len(roles) == 2 else True,
            f"{where} gives positive and negative the same label",
        )


def validate_training(section: TrainingConfig, laya: LayaConfig) -> None:
    for name in ("epochs", "micro_batch", "grad_accum", "rl_samples", "log_every", "min_abstain_n", "calib_max"):
        _positive(getattr(section, name), f"training.{name}")
    _require(
        section.loss in LOSSES,
        f"training.loss must be one of {LOSSES}, got {section.loss!r}",
    )
    for name in ("encoder_lr", "head_lr", "min_lr"):
        value = getattr(section, name)
        _require(
            isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0,
            f"training.{name} must be >= 0, got {value!r}",
        )
    for name in ("weight_decay", "grad_clip", "sigma_start", "sigma_end", "w_sph", "w_rps"):
        value = getattr(section, name)
        _require(
            isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0,
            f"training.{name} must be >= 0, got {value!r}",
        )
    unknown = sorted(set(section.shuffle_options) - set(QUESTION_TYPES))
    _require(
        not unknown,
        f"training.shuffle_options names unknown question types: {', '.join(unknown)}",
    )
    _require(
        section.option_layout is None or section.option_layout in OPTION_LAYOUTS,
        f"training.option_layout must be null or one of {OPTION_LAYOUTS}, "
        f"got {section.option_layout!r}",
    )
    _require(
        0.0 <= section.calib_frac < 1.0,
        f"training.calib_frac must be in [0, 1), got {section.calib_frac}",
    )
    _fraction(section.label_smoothing, "training.label_smoothing")
    _fraction(section.target_error, "training.target_error")
    _require(
        section.seed >= 0, "training.seed must be >= 0",
    )
    for name in ("max_len", "head_max_len"):
        value = getattr(section, name)
        if value is not None:
            _positive(value, f"training.{name}")
    if section.max_len is not None:
        _require(
            section.max_len <= laya.max_len,
            f"training.max_len ({section.max_len}) exceeds laya.max_len ({laya.max_len})",
        )
    if section.head_max_len is not None:
        _require(
            section.head_max_len <= laya.max_len,
            f"training.head_max_len ({section.head_max_len}) exceeds laya.max_len "
            f"({laya.max_len})",
        )


def validate_evaluation(section: EvaluationConfig) -> None:
    _positive(section.batch_size, "evaluation.batch_size")
    _positive(section.latency_repeats, "evaluation.latency_repeats")
    _require(
        len(section.retrieval_ks) > 0,
        "evaluation.retrieval_ks must not be empty",
    )
    for k in section.retrieval_ks:
        _positive(k, "evaluation.retrieval_ks entries")
    _require(
        section.retrieval_ks == sorted(section.retrieval_ks),
        f"evaluation.retrieval_ks must be sorted ascending, got {section.retrieval_ks}",
    )
    _require(
        section.retrieval_backend in RETRIEVAL_BACKENDS,
        f"evaluation.retrieval_backend must be one of {RETRIEVAL_BACKENDS}, "
        f"got {section.retrieval_backend!r}",
    )
    _require(
        isinstance(section.base_model, str) and bool(section.base_model.strip()),
        "evaluation.base_model must name a checkpoint",
    )


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_training_profile_overrides(
    path: str | os.PathLike[str],
) -> Dict[str, Any]:
    """Load only supported training-profile values as dotted config overrides."""
    from .io import read_text

    profile_path = Path(path).expanduser()
    if not profile_path.is_file():
        raise ConfigError(f"training profile not found: {profile_path}")
    text = read_text(profile_path)
    try:
        import yaml

        loaded = yaml.safe_load(text)
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ConfigError("PyYAML is required to read a training profile") from exc
    except Exception as exc:  # yaml.YAMLError
        raise ConfigError(f"{profile_path} is not valid YAML: {exc}") from exc

    if not isinstance(loaded, Mapping):
        raise ConfigError(f"{profile_path} must contain a YAML mapping at the top level")
    if any(not isinstance(key, str) for key in loaded):
        raise ConfigError(f"{profile_path} has a non-string top-level key")
    unknown_sections = sorted(set(loaded) - {"project", "training"})
    if unknown_sections:
        raise ConfigError(
            f"{profile_path} has unsupported section(s): {', '.join(unknown_sections)}"
        )

    project = loaded.get("project", {})
    if not isinstance(project, Mapping):
        raise ConfigError(f"{profile_path} project section must be a mapping")
    if any(not isinstance(key, str) for key in project):
        raise ConfigError(f"{profile_path} project section has a non-string key")
    unknown_project = sorted(set(project) - {"seed"})
    if unknown_project:
        raise ConfigError(
            f"{profile_path} project section has unsupported key(s): "
            f"{', '.join(unknown_project)}"
        )

    training = loaded.get("training")
    if not isinstance(training, Mapping) or not training:
        raise ConfigError(f"{profile_path} must contain a non-empty training mapping")
    if any(not isinstance(key, str) for key in training):
        raise ConfigError(f"{profile_path} training section has a non-string key")

    overrides = {f"training.{key}": value for key, value in training.items()}
    if "seed" in project:
        overrides["project.seed"] = project["seed"]
    return overrides


def load_config(
    path: str | os.PathLike[str] | None = None,
    *,
    apply_env: bool = True,
    overrides: Mapping[str, Any] | None = None,
) -> Config:
    """Load, validate and return the configuration.

    ``path`` is the YAML file. Without it, the built-in dataclass defaults are
    used, which is enough for tests and for ``--help`` paths but not for a real
    run: ``laya.questions`` and ``laya.state_templates`` have no defaults.
    """
    data: Dict[str, Any] = {}
    source_path: Optional[str] = None
    if path is not None:
        from .io import read_text  # local import keeps --help working without the file layer

        config_path = Path(path).expanduser()
        if not config_path.is_file():
            raise ConfigError(f"config file not found: {config_path}")
        text = read_text(config_path)
        try:
            import yaml

            loaded = yaml.safe_load(text)
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ConfigError("PyYAML is required to read the config file") from exc
        except Exception as exc:  # yaml.YAMLError
            raise ConfigError(f"{config_path} is not valid YAML: {exc}") from exc
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"{config_path} must contain a YAML mapping at the top level")
        data = loaded
        source_path = str(config_path)

    config = _build(Config, data, "config")
    config.source_path = source_path
    if apply_env:
        _load_dotenv()
        _apply_env_overrides(config)
    if overrides:
        config = config.with_overrides(overrides)
    config.validate()
    return config


def _load_dotenv(project_root: Optional[str] = None) -> Optional[Path]:
    """Populate ``os.environ`` from ``.env`` at the project root.

    ``.env.example`` documents ``LAYA_*`` variables, so a ``.env`` file that
    nothing reads would be a trap: an operator edits it, sees no change, and has
    no way to tell whether the edit worked. It is read here.

    Two rules keep it predictable:

    * a real environment variable always wins, so a one-off
      ``LAYA_OLLAMA_TEACHER_MODEL=... laya-pipeline generate`` still overrides
      the file without editing it;
    * nothing is overwritten and nothing is logged, because a value in a dotenv
      file may be a credential.

    Returns the path that was read, or ``None`` if there was no file.
    """
    root = Path(project_root) if project_root else Path.cwd()
    candidates = [root / ".env"]
    # Also look one level up, so the file works from a subdirectory invocation.
    if root.parent != root:
        candidates.append(root.parent / ".env")
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            text = candidate.read_text(encoding="utf-8")
        except OSError:
            continue
        for lineno, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("export "):
                line = line[len("export ") :].strip()
            key, separator, value = line.partition("=")
            if not separator:
                continue
            key = key.strip()
            if not _ENV_NAME_RE.match(key):
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            else:
                value = value.split(" #", 1)[0].rstrip()
            if key not in os.environ:
                os.environ[key] = value
            del lineno  # the line number is not reported, to avoid echoing secrets
        return candidate
    return None


def _apply_env_overrides(config: Config) -> None:
    """Apply the ``env_overrides`` map declared in the config itself.

    Keys are either a bare field of the ``ollama`` section (``generation_model``)
    or a dotted path into any section (``paths.corpus_dir``). A bare key is kept
    working because most overrides concern the model endpoint.

    The declared type drives coercion, so a string from the environment becomes
    the same Python type the dataclass declares. A value that cannot be coerced
    raises rather than being stored as a string, because a silently mistyped
    ``num_ctx`` would surface much later as a confusing model failure.
    """
    mapping = getattr(config, "env_overrides", None) or {}
    if not mapping:
        return
    for target_path, env_name in mapping.items():
        raw = os.environ.get(env_name)
        if raw is None or raw == "":
            continue
        if "." not in target_path:
            # A bare key names the ollama section, which is where almost every
            # override concerns.
            target_path = f"ollama.{target_path}"
        section_name, _, field_name = target_path.rpartition(".")
        section = config if not section_name else getattr(config, section_name, None)
        if section is None:
            raise ConfigError(
                f"env_overrides.{target_path} does not name a section of the configuration"
            )
        if not hasattr(section, field_name):
            raise ConfigError(
                f"env_overrides.{target_path} does not name a key of the "
                f"{section_name or 'root'} section"
            )
        setattr(section, field_name, _coerce_env_value(raw, env_name, target_path, section, field_name))


def _coerce_env_value(
    raw: str, env_name: str, target_path: str, section: object, field_name: str
) -> Any:
    """Convert one environment string to the type its dataclass field declares."""
    current = getattr(section, field_name)
    optional_bool = _is_optional_bool(type(section), field_name)
    if optional_bool or isinstance(current, bool):
        # A tri-state field ("null" clears it) needs the annotation rather than
        # the loaded value, which is a plain False once the YAML has set it.
        lowered = raw.strip().lower()
        if lowered in ("null", "none", "unset", "default"):
            if not optional_bool:
                raise ConfigError(
                    f"{env_name}={raw!r} cannot clear {target_path}; it is a plain "
                    "boolean, not an optional one"
                )
            return None
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
        hint = (
            "use true, false, or null to leave the field unset"
            if optional_bool
            else "use true or false"
        )
        raise ConfigError(f"{env_name}={raw!r} is not a boolean for {target_path}; {hint}")
    if isinstance(current, int):
        try:
            return int(raw)
        except ValueError:
            raise ConfigError(
                f"{env_name}={raw!r} is not an integer for {target_path}"
            ) from None
    if isinstance(current, float):
        try:
            return float(raw)
        except ValueError:
            raise ConfigError(
                f"{env_name}={raw!r} is not a number for {target_path}"
            ) from None
    if isinstance(current, str) or current is None:
        return raw
    raise ConfigError(
        f"{env_name}={raw!r} cannot be applied to {target_path}: its declared type "
        f"{type(current).__name__} has no coercion from a string"
    )


def _is_optional_bool(dataclass_type: type, field_name: str) -> bool:
    """True when the field is declared ``Optional[bool]``.

    Needed because the YAML default of ``think`` is ``false``, so the loaded
    value alone cannot distinguish "a plain bool" from "a tri-state bool the
    file happens to have set to false". Only the declaration can.
    """
    try:
        hints = get_type_hints(dataclass_type)
    except Exception:  # pragma: no cover - defensive: unresolvable annotations
        return False
    annotation = hints.get(field_name)
    if annotation is None:
        return False
    args = get_args(annotation)
    return bool in args and type(None) in args


def _set_dotted(target: Dict[str, Any], dotted: str, value: Any) -> None:
    """Apply one ``a.b.c=value`` override in place."""
    parts = dotted.split(".")
    cursor: Any = target
    for part in parts[:-1]:
        if not isinstance(cursor, dict) or part not in cursor:
            raise ConfigError(
                f"override key {dotted!r} does not exist in the configuration "
                f"(unknown segment {part!r})"
            )
        cursor = cursor[part]
    if not isinstance(cursor, dict) or parts[-1] not in cursor:
        raise ConfigError(f"override key {dotted!r} does not exist in the configuration")
    cursor[parts[-1]] = value


def _plain(value: Any) -> Any:
    """Recursively convert dataclasses/tuples to JSON-friendly containers."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def config_hash(data: Mapping[str, Any]) -> str:
    """Stable hash of a configuration mapping."""
    canonical = json.dumps(_plain(data), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return content_hash(canonical, fold=False)


# --------------------------------------------------------------------------- #
# Reflection-based construction
# --------------------------------------------------------------------------- #
def _build(cls: type, data: Mapping[str, Any], where: str) -> Any:
    """Instantiate dataclass ``cls`` from ``data``, rejecting unknown keys."""
    if not isinstance(data, Mapping):
        raise ConfigError(f"{where} must be a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ConfigError(
            f"{where} has unknown key(s): {', '.join(unknown)}; "
            f"known: {', '.join(sorted(known))}"
        )
    kwargs: Dict[str, Any] = {}
    for name, value in data.items():
        if name.startswith("_"):
            continue
        kwargs[name] = _coerce(value, hints[name], f"{where}.{name}")
    return cls(**kwargs)


def _coerce(value: Any, annotation: Any, where: str) -> Any:
    """Coerce a YAML value to the annotated type, or raise naming the key."""
    if annotation is Any or annotation is None:
        return value

    origin = get_origin(annotation)

    # Optional[X] / X | None
    if origin in (Union, types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if value is None:
            return None
        if len(args) == 1:
            return _coerce(value, args[0], where)
        return value

    if origin in (list, List):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where} must be a list, got {type(value).__name__}")
        args = get_args(annotation)
        inner = args[0] if args else Any
        return [_coerce(v, inner, f"{where}[{i}]") for i, v in enumerate(value)]

    if origin in (tuple, Tuple):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where} must be a list, got {type(value).__name__}")
        return tuple(value)

    if origin in (dict, Dict):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{where} must be a mapping, got {type(value).__name__}")
        args = get_args(annotation)
        if len(args) == 2:
            return {
                str(_coerce(k, args[0], f"{where} key")): _coerce(v, args[1], f"{where}[{k}]")
                for k, v in value.items()
            }
        return dict(value)

    if annotation is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{where} must be true or false, got {value!r}")

    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where} must be an integer, got {value!r}")
        return value

    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where} must be a number, got {value!r}")
        return float(value)

    if annotation is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where} must be a string, got {value!r}")
        return value

    if is_dataclass(annotation):
        return _build(annotation, value, where)

    return value
