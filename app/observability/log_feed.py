"""Turns decision log events into short, human-readable lines for the debug log box.

The debug box on the submit page polls for new events. Each poll passes the
position it has read up to, and gets back only the events written after it.
The position is simply the number of events already read from the log file.

Only a readable sentence and a few labels are sent to the browser. The raw
payload and any debug details (full prompts, raw model responses) are not.
"""

from dataclasses import asdict, dataclass

from app.observability.decision_log import DecisionEvent, DecisionLogger, EventType

# How many past events to show when the debug box is first opened.
EVENTS_SHOWN_ON_FIRST_LOAD = 100


@dataclass
class FeedLine:
    position: int
    time: str
    actor: str
    event_type: str
    text: str
    is_error: bool


@dataclass
class FeedPage:
    lines: list[FeedLine]
    next_position: int

    def to_json_dict(self) -> dict:
        line_dicts = []
        for line in self.lines:
            line_dicts.append(asdict(line))
        return {"lines": line_dicts, "next_position": self.next_position}


# --- Reading new events --------------------------------------------------------------


def read_feed_after(decision_log: DecisionLogger, position: int | None) -> FeedPage:
    """Events written after `position`, as readable lines.

    With no position (the box was just opened), the most recent events are
    returned. If the position is past the end of the log, the log file was
    replaced or cleared, so the feed starts again from the recent events.
    """
    all_events = decision_log.read_all_events()
    event_count = len(all_events)

    if position is None or position > event_count:
        first_index = max(0, event_count - EVENTS_SHOWN_ON_FIRST_LOAD)
    else:
        first_index = position

    lines = []
    for index in range(first_index, event_count):
        lines.append(build_feed_line(index, all_events[index]))
    return FeedPage(lines=lines, next_position=event_count)


def build_feed_line(position: int, event: DecisionEvent) -> FeedLine:
    return FeedLine(
        position=position,
        time=event.timestamp.isoformat(),
        actor=describe_actor(event),
        event_type=event.event_type.value,
        text=describe_event(event),
        is_error=event.event_type is EventType.PIPELINE_ERROR,
    )


# --- Describing events --------------------------------------------------------------


def submission_label(submission_id: int | None) -> str:
    if submission_id is None:
        return "a submission"
    return f"S-{submission_id:03d}"


def feature_label(feature_id: int | None) -> str:
    if feature_id is None:
        return "a feature"
    return f"F-{feature_id:03d}"


def percent(value: float | None) -> str:
    if value is None:
        return "?"
    return f"{round(value * 100)}%"


def describe_actor(event: DecisionEvent) -> str:
    actor = event.actor
    if actor.type == "human":
        return f"Reviewer {actor.reviewer}"
    if actor.type == "ai":
        return f"AI ({actor.model})"
    return "System"


def describe_event(event: DecisionEvent) -> str:
    """One sentence saying what happened. Unknown event types get a generic sentence."""
    describers = {
        EventType.SUBMISSION_RECEIVED: describe_submission_received,
        EventType.DUPLICATE_CANDIDATES: describe_duplicate_candidates,
        EventType.DUPLICATE_JUDGMENT: describe_duplicate_judgment,
        EventType.SUMMARY_PROPOSED: describe_summary_proposed,
        EventType.CATEGORY_PROPOSED: describe_category_proposed,
        EventType.MERGE_DECISION: describe_merge_decision,
        EventType.CATEGORY_DECISION: describe_category_decision,
        EventType.CATEGORY_CREATED: describe_category_created,
        EventType.CATEGORY_REASSIGNED: describe_category_reassigned,
        EventType.FEATURE_EDITED: describe_feature_edited,
        EventType.FEATURE_ARCHIVED: describe_feature_archived,
        EventType.FEATURE_RESTORED: describe_feature_restored,
        EventType.FEATURE_SUPPORTED: describe_feature_supported,
        EventType.SUBMISSION_UNMERGED: describe_submission_unmerged,
        EventType.SUBMISSION_MOVED: describe_submission_moved,
        EventType.SUBMISSION_ARCHIVED: describe_submission_archived,
        EventType.SUBMISSION_RESTORED: describe_submission_restored,
        EventType.TAXONOMY_PROPOSED: describe_taxonomy_proposed,
        EventType.TAXONOMY_DECISION: describe_taxonomy_decision,
        EventType.PIPELINE_ERROR: describe_pipeline_error,
    }
    describer = describers.get(event.event_type)
    if describer is None:
        return f"{event.event_type.value} recorded."
    return describer(event, event.payload)


def describe_submission_received(event: DecisionEvent, payload: dict) -> str:
    title = payload.get("title", "")
    return f'New submission {submission_label(event.submission_id)} received: "{title}".'


def describe_duplicate_candidates(event: DecisionEvent, payload: dict) -> str:
    candidates = payload.get("candidates", [])
    submission = submission_label(event.submission_id)
    if not candidates:
        return f"No similar features found for {submission}."
    candidate_texts = []
    for candidate in candidates:
        candidate_texts.append(
            f"{feature_label(candidate.get('feature_id'))} ({percent(candidate.get('similarity'))})"
        )
    return f"Similar features found for {submission}: {', '.join(candidate_texts)}."


def describe_duplicate_judgment(event: DecisionEvent, payload: dict) -> str:
    submission = submission_label(event.submission_id)
    duplicate_of = payload.get("duplicate_of")
    if duplicate_of is None:
        return f"AI judged {submission} is not a duplicate (confidence {percent(event.confidence)})."
    return (
        f"AI judged {submission} a duplicate of {feature_label(duplicate_of)} "
        f"(confidence {percent(event.confidence)})."
    )


def describe_summary_proposed(event: DecisionEvent, payload: dict) -> str:
    feature = feature_label(event.feature_id)
    if payload.get("changed"):
        return f"AI proposed an updated summary for {feature}."
    return f"AI kept the summary of {feature} unchanged."


def describe_category_proposed(event: DecisionEvent, payload: dict) -> str:
    submission = submission_label(event.submission_id)
    confidence = percent(event.confidence)
    if payload.get("action") == "propose_new":
        name = payload.get("new_category_name")
        return f'AI proposed a new category "{name}" for {submission} (confidence {confidence}).'
    return (
        f"AI proposed category #{payload.get('category_id')} for {submission} "
        f"(confidence {confidence})."
    )


def describe_merge_decision(event: DecisionEvent, payload: dict) -> str:
    submission = submission_label(event.submission_id)
    feature = feature_label(event.feature_id)
    if payload.get("decision") == "reject":
        return f"Merge of {submission} into {feature} rejected."
    return f"{submission} merged into {feature}."


def describe_category_decision(event: DecisionEvent, payload: dict) -> str:
    submission = submission_label(event.submission_id)
    feature = feature_label(event.feature_id)
    decision_words = {
        "approve": "approved",
        "edit": "edited",
        "reject": "rejected",
        "manual": "chosen by hand",
    }
    decision = decision_words.get(payload.get("decision"), "decided")
    category_id = payload.get("final_category_id")
    return f"Category {decision} for {submission}: {feature} filed under category #{category_id}."


def describe_category_created(event: DecisionEvent, payload: dict) -> str:
    return f'Category "{payload.get("name")}" created.'


def describe_category_reassigned(event: DecisionEvent, payload: dict) -> str:
    from_name = payload.get("from_category_name") or "Uncategorized"
    to_name = payload.get("to_category_name")
    text = f'{feature_label(event.feature_id)} moved from "{from_name}" to "{to_name}".'
    reason = payload.get("reason")
    if reason:
        text += f' Reason: "{reason}".'
    return text


def describe_feature_edited(event: DecisionEvent, payload: dict) -> str:
    changed_fields = " and ".join(payload.keys()) or "details"
    return f"{feature_label(event.feature_id)} edited ({changed_fields} changed)."


def describe_feature_archived(event: DecisionEvent, payload: dict) -> str:
    return f'{feature_label(event.feature_id)} "{payload.get("title", "")}" deleted.'


def describe_feature_restored(event: DecisionEvent, payload: dict) -> str:
    return f"{feature_label(event.feature_id)} restored."


def describe_feature_supported(event: DecisionEvent, payload: dict) -> str:
    count = payload.get("support_count")
    supporters = "supporter" if count == 1 else "supporters"
    return f"{feature_label(event.feature_id)} supported (now {count} {supporters})."


def describe_submission_unmerged(event: DecisionEvent, payload: dict) -> str:
    return (
        f"{submission_label(event.submission_id)} unmerged from "
        f"{feature_label(payload.get('from_feature_id'))} into new feature "
        f"{feature_label(payload.get('to_feature_id'))}."
    )


def describe_submission_moved(event: DecisionEvent, payload: dict) -> str:
    return (
        f"{submission_label(event.submission_id)} moved from "
        f"{feature_label(payload.get('from_feature_id'))} to "
        f"{feature_label(payload.get('to_feature_id'))}."
    )


def describe_submission_archived(event: DecisionEvent, payload: dict) -> str:
    return f"{submission_label(event.submission_id)} deleted from {feature_label(event.feature_id)}."


def describe_submission_restored(event: DecisionEvent, payload: dict) -> str:
    return f"{submission_label(event.submission_id)} restored to {feature_label(event.feature_id)}."


def describe_taxonomy_proposed(event: DecisionEvent, payload: dict) -> str:
    category_count = len(payload.get("categories", []))
    request_count = payload.get("request_count", 0)
    return f"AI proposed {category_count} categories from {request_count} requests."


def describe_taxonomy_decision(event: DecisionEvent, payload: dict) -> str:
    if payload.get("decision") == "reject":
        return "Proposed categories rejected."
    return "Proposed categories approved."


def describe_pipeline_error(event: DecisionEvent, payload: dict) -> str:
    stage = payload.get("stage", "an AI step")
    message = payload.get("message", "")
    return (
        f"Error during {stage} for {submission_label(event.submission_id)}: "
        f"{payload.get('error', 'Error')}: {message}"
    )
