"""The three eval flows and how each one is graded.

    DuplicatesFlow   47 probes, each checked against the 30 corpus features
    CategoriesFlow   all 77 items, categorized against the frozen gold taxonomy
    SummariesFlow    the 22 duplicate probes, each merged into its true feature,
                     with the updated summary graded by a judge model

Each flow calls the same pipeline stage functions the app uses
(app/pipeline/stages.py), against its own throwaway SQLite database.

Every flow has the same shape:
    flow.name, flow.cases
    flow.tags_for(item)                              -> list of report tags
    flow.run_case(item, model, judge_model)          -> CaseOutcome
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from sqlalchemy import select

from app.ai.embeddings import build_text_to_embed, get_shared_embedder
from app.ai.vectors import vector_to_bytes
from app.config import Settings
from app.db.models import Category, Feature, Submission
from app.db.session import create_database_engine, create_session_factory, create_tables
from app.domain.enums import CategoryStatus, SubmissionStatus
from app.pipeline import stages
from evals.orbit.dataset import DATA_DIR, Item, load_corpus, load_probes, load_taxonomy
from evals.orbit.recording_model import RecordedCall, RecordingLanguageModel

JUDGE_MODEL = "claude-sonnet-5-5"


@dataclass
class CaseOutcome:
    grade: dict[str, float]
    prompt_text: str
    extra_trace_turns: list[dict] = field(default_factory=list)
    explanation: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    judge_calls: list[RecordedCall] = field(default_factory=list)


def describe_item_for_report(item: Item) -> str:
    return f"{item.title}\n\n{item.description}"


# --- Duplicates ---------------------------------------------------------------


def grade_duplicate_verdict(item: Item, predicted_corpus_id: str | None) -> dict[str, float]:
    """Correct means: the right existing feature, or correctly "not a duplicate"."""
    is_correct = predicted_corpus_id == item.duplicate_of
    merged_wrongly = predicted_corpus_id is not None and predicted_corpus_id != item.duplicate_of
    missed_duplicate = item.duplicate_of is not None and predicted_corpus_id is None
    return {
        "correct": float(is_correct),
        "false_merge": float(merged_wrongly),
        "missed_dup": float(missed_duplicate),
        "refused": 0.0,
    }


class DuplicatesFlow:
    """Each probe is compared with the 30 corpus features only, never with other probes."""

    name = "duplicates"

    def __init__(self, settings: Settings, temp_dir: Path):
        self.settings = settings
        self.corpus = load_corpus()
        self.cases = load_probes()
        self.embedder = get_shared_embedder(settings.embedding_model)

        engine = create_database_engine(f"sqlite:///{(temp_dir / 'duplicates.db').as_posix()}")
        create_tables(engine)
        self.session_factory = create_session_factory(engine)

        self.feature_id_by_corpus_id = self.store_corpus_as_features()
        self.corpus_id_by_feature_id = {}
        for corpus_id, feature_id in self.feature_id_by_corpus_id.items():
            self.corpus_id_by_feature_id[feature_id] = corpus_id

        # Embed every probe once, up front, so the threads don't share the model.
        probe_texts = [build_text_to_embed(probe.title, probe.description) for probe in self.cases]
        probe_vectors = self.embedder.embed_texts(probe_texts)
        self.embedding_by_probe_id = {}
        for probe, vector in zip(self.cases, probe_vectors):
            self.embedding_by_probe_id[probe.id] = vector

    def store_corpus_as_features(self) -> dict[str, int]:
        """Create one feature (with one embedded submission) per corpus item."""
        corpus_texts = [build_text_to_embed(item.title, item.description) for item in self.corpus]
        corpus_vectors = self.embedder.embed_texts(corpus_texts)

        feature_id_by_corpus_id = {}
        with self.session_factory() as session:
            for item, vector in zip(self.corpus, corpus_vectors):
                feature = Feature(title=item.title, summary=item.description)
                session.add(feature)
                session.flush()
                session.add(
                    Submission(
                        correlation_id=f"eval-{item.id}",
                        title=item.title,
                        description=item.description,
                        status=SubmissionStatus.NEW_FEATURE,
                        feature_id=feature.id,
                        embedding=vector_to_bytes(vector),
                        embedding_model=self.embedder.model_name,
                    )
                )
                feature_id_by_corpus_id[item.id] = feature.id
            session.commit()
        return feature_id_by_corpus_id

    def tags_for(self, item: Item) -> list[str]:
        return [item.kind]

    def run_case(self, item: Item, model: RecordingLanguageModel, judge_model: RecordingLanguageModel) -> CaseOutcome:
        with self.session_factory() as session:
            candidates = stages.find_duplicate_candidates(
                session,
                self.embedding_by_probe_id[item.id],
                max_candidates=self.settings.duplicate_top_k,
                min_similarity=self.settings.duplicate_similarity_threshold,
            )

            candidate_corpus_ids = []
            candidates_for_report = []
            for candidate in candidates:
                corpus_id = self.corpus_id_by_feature_id[candidate.feature_id]
                candidate_corpus_ids.append(corpus_id)
                candidates_for_report.append({"id": corpus_id, "similarity": round(candidate.similarity, 4)})

            meta = {
                "gold": item.duplicate_of,
                "candidates": candidates_for_report,
                "gold_in_candidates": None,
                "llm_called": bool(candidates),
            }
            if item.duplicate_of:
                meta["gold_in_candidates"] = item.duplicate_of in candidate_corpus_ids

            predicted_corpus_id = None
            if candidates:
                # Same as the app: only ask the model when the search found something.
                feature_ids = [candidate.feature_id for candidate in candidates]
                features_by_id = {}
                for feature in session.scalars(select(Feature).where(Feature.id.in_(feature_ids))):
                    features_by_id[feature.id] = feature
                candidates_with_features = []
                for candidate in candidates:
                    candidates_with_features.append((candidate, features_by_id[candidate.feature_id]))

                result, _ = stages.ask_model_if_duplicate(
                    model, item.title, item.description, candidates_with_features
                )
                judged_feature_id = result.output.duplicate_of_feature_id
                if judged_feature_id is not None:
                    predicted_corpus_id = self.corpus_id_by_feature_id.get(judged_feature_id)
                meta["confidence"] = result.output.confidence
            meta["predicted"] = predicted_corpus_id

        return CaseOutcome(
            grade=grade_duplicate_verdict(item, predicted_corpus_id),
            prompt_text=describe_item_for_report(item),
            meta=meta,
        )


# --- Categories -----------------------------------------------------------------


def grade_category_choice(item: Item, action: str, chosen_category_name: str | None) -> dict[str, float]:
    """Correct means the gold category or the item's also-acceptable category."""
    is_acceptable = chosen_category_name is not None and item.is_acceptable_category(chosen_category_name)
    is_exact_gold = chosen_category_name == item.category
    invented_new_category = action == "propose_new"
    return {
        "correct": float(is_acceptable),
        "exact": float(is_exact_gold),
        "invented": float(invented_new_category),
        "refused": 0.0,
    }


class CategoriesFlow:
    """Every corpus item and probe, categorized against the frozen gold taxonomy.

    The categories have no example features, so each item is judged on the
    category names and descriptions alone.
    """

    name = "categories"

    def __init__(self, settings: Settings, temp_dir: Path):
        self.cases = load_corpus() + load_probes()

        engine = create_database_engine(f"sqlite:///{(temp_dir / 'categories.db').as_posix()}")
        create_tables(engine)
        self.session_factory = create_session_factory(engine)

        self.category_name_by_id = {}
        with self.session_factory() as session:
            for gold_category in load_taxonomy():
                session.add(
                    Category(
                        name=gold_category["name"],
                        description=gold_category["description"],
                        status=CategoryStatus.ACTIVE,
                    )
                )
            session.commit()
            for category in session.scalars(select(Category)):
                self.category_name_by_id[category.id] = category.name

    def tags_for(self, item: Item) -> list[str]:
        tags = [item.category]
        if item.acceptable:
            tags.append("has-alt")
        return tags

    def run_case(self, item: Item, model: RecordingLanguageModel, judge_model: RecordingLanguageModel) -> CaseOutcome:
        with self.session_factory() as session:
            result, _ = stages.ask_model_for_category(model, session, item.title, item.description)
        assignment = result.output

        chosen_category_name = None
        if assignment.action == "assign":
            chosen_category_name = self.category_name_by_id.get(assignment.category_id)

        meta = {
            "gold": item.category,
            "acceptable": list(item.acceptable),
            "action": assignment.action,
            "predicted": chosen_category_name or assignment.new_category_name,
            "confidence": assignment.confidence,
        }
        return CaseOutcome(
            grade=grade_category_choice(item, assignment.action, chosen_category_name),
            prompt_text=describe_item_for_report(item),
            meta=meta,
        )


# --- Summaries ------------------------------------------------------------------


class SummaryCheck(BaseModel):
    passed: bool
    reasoning: str


class SummaryJudgment(BaseModel):
    keeps_existing: SummaryCheck
    adds_new_info: SummaryCheck
    no_fabrication: SummaryCheck
    flag_consistent: SummaryCheck


MENTIONS_REQUESTERS = re.compile(
    r"\b("
    r"requesters?"
    r"|(several|many|multiple|some) (users|people|customers|teams|requesters|ask)"
    r"|\d+ (users|people|customers|requests|requesters)"
    r"|(users|people|customers) have (asked|requested)"
    r")\b",
    re.IGNORECASE,
)


def split_into_sentences(text: str) -> list[str]:
    sentences = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        if sentence:
            sentences.append(sentence)
    return sentences


def check_summary_format(summary: str, existing_summary: str | None = None) -> tuple[bool, str]:
    """Return (passed, reason). A summary must be 2-4 sentences and not mention requesters.

    A summary returned unchanged skips the sentence count: the summary prompt
    tells the model to keep the existing summary when nothing is new, even if
    that summary is a single sentence.
    """
    match = MENTIONS_REQUESTERS.search(summary)
    if match:
        return False, f'mentions requesters: "{match.group(0)}"'

    if existing_summary is not None and summary.strip() == existing_summary.strip():
        return True, "unchanged from the existing summary (sentence count not checked)"

    sentence_count = len(split_into_sentences(summary))
    if sentence_count < 2 or sentence_count > 4:
        return False, f"{sentence_count} sentences (expected 2-4)"
    return True, f"{sentence_count} sentences, no requester mentions"


def build_judge_message(existing_summary: str, item: Item, updated_summary: str, changed_flag: bool) -> str:
    return (
        f"<existing_summary>\n{existing_summary}\n</existing_summary>\n\n"
        f"<new_request>\nTitle: {item.title}\nDescription: {item.description}\n</new_request>\n\n"
        f"<updated_summary>\n{updated_summary}\n</updated_summary>\n\n"
        f"<changed_flag>{str(changed_flag).lower()}</changed_flag>"
    )


@dataclass
class SummaryGrade:
    grade: dict[str, float]
    explanation: dict[str, str]
    judge_message: str


def grade_summary(
    judge_prompt: str,
    judge_model: RecordingLanguageModel,
    existing_summary: str,
    item: Item,
    updated_summary: str,
    changed_flag: bool,
) -> SummaryGrade:
    """Four checks by the judge model, plus the format check in code."""
    judge_message = build_judge_message(existing_summary, item, updated_summary, changed_flag)
    judgment = judge_model.request_structured_output(
        system_prompt=judge_prompt, user_message=judge_message, output_type=SummaryJudgment
    ).output

    format_passed, format_reason = check_summary_format(updated_summary, existing_summary)

    judge_checks = {
        "keeps_existing": judgment.keeps_existing,
        "adds_new_info": judgment.adds_new_info,
        "no_fabrication": judgment.no_fabrication,
        "flag_consistent": judgment.flag_consistent,
    }
    grade = {}
    explanation = {}
    for check_name, check in judge_checks.items():
        grade[check_name] = float(check.passed)
        explanation[check_name] = check.reasoning
    grade["format_ok"] = float(format_passed)
    explanation["format_ok"] = format_reason

    every_check_passed = all(score == 1.0 for score in grade.values())
    grade["all_pass"] = float(every_check_passed)
    grade["refused"] = 0.0
    return SummaryGrade(grade=grade, explanation=explanation, judge_message=judge_message)


class SummariesFlow:
    """The 22 duplicate probes, each paired with its true corpus feature.

    Pairing with the true feature (rather than whatever the duplicate check
    picked) keeps summary quality separate from duplicate-detection mistakes.
    """

    name = "summaries"

    def __init__(self, settings: Settings, temp_dir: Path):
        corpus_by_id = {item.id: item for item in load_corpus()}
        self.cases = []
        self.existing_feature_by_probe_id = {}
        for probe in load_probes():
            if probe.duplicate_of:
                self.cases.append(probe)
                self.existing_feature_by_probe_id[probe.id] = corpus_by_id[probe.duplicate_of]
        self.judge_prompt = (DATA_DIR / "judge_summary_v1.txt").read_text(encoding="utf-8")

    def tags_for(self, item: Item) -> list[str]:
        return [item.kind, item.duplicate_of]

    def run_case(self, item: Item, model: RecordingLanguageModel, judge_model: RecordingLanguageModel) -> CaseOutcome:
        existing = self.existing_feature_by_probe_id[item.id]
        feature = Feature(title=existing.title, summary=existing.description)  # never saved

        result, _ = stages.ask_model_for_merged_summary(model, feature, item.title, item.description)
        update = result.output

        summary_grade = grade_summary(
            self.judge_prompt, judge_model, existing.description, item, update.summary, update.changed
        )
        return CaseOutcome(
            grade=summary_grade.grade,
            explanation=summary_grade.explanation,
            prompt_text=describe_item_for_report(item),
            meta={"feature": existing.id, "changed": update.changed},
            extra_trace_turns=[
                {"role": "user", "name": "judge", "content": summary_grade.judge_message},
                {"role": "assistant", "name": "judge", "content": format_explanation(summary_grade.explanation)},
            ],
            judge_calls=list(judge_model.calls),
        )


def format_explanation(explanation: dict[str, str]) -> str:
    lines = []
    for check_name, reason in explanation.items():
        lines.append(f"{check_name}: {reason}")
    return "\n".join(lines)


FLOW_CLASSES = {
    "duplicates": DuplicatesFlow,
    "categories": CategoriesFlow,
    "summaries": SummariesFlow,
}
