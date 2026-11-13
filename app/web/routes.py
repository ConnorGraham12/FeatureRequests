"""HTML pages and form handlers.

Each form handler converts the form fields, calls one service function, and
redirects. Decision errors (a missing category, an already-decided proposal)
are shown on the page the reviewer came from.
"""

from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Category, Feature, Proposal, Submission
from app.db.session import get_request_db_session
from app.domain.enums import CategoryStatus, ProposalStatus, ProposalType, SubmissionStatus
from app.domain.proposals import InvalidDecisionError, ProposalAlreadyDecidedError
from app.observability.decision_log import EventType
from app.pipeline.stages import list_active_categories
from app.services import feature_management, review, taxonomy, triage
from app.services.context import TriageContext

DECISION_ERRORS = (InvalidDecisionError, ProposalAlreadyDecidedError)


# --- Template setup -----------------------------------------------------------


def format_submission_id(submission_id: int) -> str:
    return f"S-{submission_id:03d}"


def format_feature_id(feature_id: int) -> str:
    return f"F-{feature_id:03d}"


def format_score(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"


templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
templates.env.filters["sid"] = format_submission_id
templates.env.filters["fid"] = format_feature_id
templates.env.filters["pct"] = format_score

router = APIRouter()


# --- Helpers --------------------------------------------------------------------


def get_triage_context(request: Request) -> TriageContext:
    """FastAPI dependency: the shared TriageContext built at startup."""
    return request.app.state.triage


def redirect_to(url: str, error_message: str | None = None, notice: str | None = None) -> RedirectResponse:
    """Redirect after a form post, optionally carrying an error or a notice to show on the page."""
    if error_message:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}error={quote(error_message)}"
    if notice:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}notice={quote(notice)}"
    return RedirectResponse(url, status_code=303)


def messages_from_query(request: Request) -> dict:
    """The error and notice passed along by redirect_to, for the page template."""
    return {
        "error": request.query_params.get("error"),
        "notice": request.query_params.get("notice"),
    }


def parse_optional_int(form_value: str | None) -> int | None:
    """Form fields arrive as text; an empty or non-numeric value means "not chosen"."""
    text = (form_value or "").strip()
    if text.isdigit():
        return int(text)
    return None


def load_proposal_or_404(session: Session, proposal_id: int) -> Proposal:
    proposal = session.get(Proposal, proposal_id)
    if proposal is None or proposal.submission_id is None:
        raise HTTPException(404)
    return proposal


# --- Submitting a request -------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
def show_submission_form(request: Request):
    return templates.TemplateResponse(
        request, "index.html", {"error": request.query_params.get("error")}
    )


@router.post("/submissions")
def submit_feature_request(
    title: str = Form(""),
    description: str = Form(""),
    session: Session = Depends(get_request_db_session),
    context: TriageContext = Depends(get_triage_context),
):
    title = title.strip()
    description = description.strip()
    if not title or not description:
        return redirect_to("/", "Both a title and a description are required.")

    submission = triage.process_new_submission(session, context, title, description)
    return redirect_to(f"/submissions/{submission.id}")


# --- Reviewing a submission ------------------------------------------------------


@router.get("/review", response_class=HTMLResponse)
def show_review_queue(request: Request, session: Session = Depends(get_request_db_session)):
    query = (
        select(Submission)
        .where(Submission.status == SubmissionStatus.PENDING)
        .order_by(Submission.created_at)
    )
    pending_submissions = session.scalars(query).all()
    return templates.TemplateResponse(request, "queue.html", {"submissions": pending_submissions})


@router.get("/submissions/{submission_id}", response_class=HTMLResponse)
def show_submission_review(
    submission_id: int, request: Request, session: Session = Depends(get_request_db_session)
):
    submission = session.get(Submission, submission_id)
    if submission is None:
        raise HTTPException(404)
    already_triaged = submission.status is not SubmissionStatus.PENDING
    if already_triaged and submission.feature_id:
        return redirect_to(f"/features/{submission.feature_id}")

    pending_proposals_by_type = {}
    for proposal in submission.proposals:
        if proposal.status is ProposalStatus.PENDING:
            pending_proposals_by_type[proposal.type] = proposal

    merge_proposal = pending_proposals_by_type.get(ProposalType.DUPLICATE)
    summary_proposal = pending_proposals_by_type.get(ProposalType.SUMMARY)
    category_proposal = pending_proposals_by_type.get(ProposalType.CATEGORY)
    if category_proposal is None:
        category_proposal = pending_proposals_by_type.get(ProposalType.NEW_CATEGORY)

    merge_target_feature = None
    if merge_proposal is not None:
        merge_target_feature = session.get(Feature, merge_proposal.feature_id)

    categories = list_active_categories(session)
    categories_by_id = {category.id: category for category in categories}

    return templates.TemplateResponse(
        request,
        "review.html",
        {
            "submission": submission,
            "duplicate": merge_proposal,
            "summary": summary_proposal,
            "target": merge_target_feature,
            "category_proposal": category_proposal,
            "categories": categories,
            "categories_by_id": categories_by_id,
            "error": request.query_params.get("error"),
        },
    )


@router.post("/proposals/{proposal_id}/merge")
def handle_merge_decision(
    proposal_id: int,
    action: str = Form(...),
    summary: str = Form(""),
    session: Session = Depends(get_request_db_session),
    context: TriageContext = Depends(get_triage_context),
):
    proposal = load_proposal_or_404(session, proposal_id)
    review_page = f"/submissions/{proposal.submission_id}"
    try:
        if action == "approve":
            review.approve_merge(session, context, proposal_id, edited_summary=summary)
        elif action == "reject":
            review.reject_merge(session, context, proposal_id)
        else:
            return redirect_to(review_page, f"Unknown action: {action}")
    except review.NotFoundError:
        raise HTTPException(404)
    except DECISION_ERRORS as error:
        return redirect_to(review_page, str(error))
    return redirect_to(review_page)


@router.post("/proposals/{proposal_id}/category")
def handle_category_decision(
    proposal_id: int,
    decision: str = Form(...),
    category_id: str = Form(""),
    summary: str = Form(""),
    name: str = Form(""),
    description: str = Form(""),
    session: Session = Depends(get_request_db_session),
    context: TriageContext = Depends(get_triage_context),
):
    proposal = load_proposal_or_404(session, proposal_id)
    review_page = f"/submissions/{proposal.submission_id}"
    chosen_category_id = parse_optional_int(category_id)
    try:
        if decision == "reject":
            feature = review.reject_category_suggestion(
                session, context, proposal_id, chosen_category_id, edited_summary=summary
            )
        elif decision == "approve" and proposal.type is ProposalType.CATEGORY:
            feature = review.approve_suggested_category(
                session, context, proposal_id, edited_summary=summary
            )
        elif decision == "approve" and proposal.type is ProposalType.NEW_CATEGORY:
            feature = review.create_suggested_new_category(
                session, context, proposal_id, edited_summary=summary
            )
        elif decision == "edit" and proposal.type is ProposalType.NEW_CATEGORY:
            feature = review.create_edited_new_category(
                session, context, proposal_id, name, description, edited_summary=summary
            )
        else:
            return redirect_to(review_page, f"Unknown decision: {decision}")
    except review.NotFoundError:
        raise HTTPException(404)
    except DECISION_ERRORS as error:
        return redirect_to(review_page, str(error))
    return redirect_to(f"/features/{feature.id}")


@router.post("/submissions/{submission_id}/manual-category")
def handle_manual_category(
    submission_id: int,
    category_id: str = Form(""),
    summary: str = Form(""),
    session: Session = Depends(get_request_db_session),
    context: TriageContext = Depends(get_triage_context),
):
    review_page = f"/submissions/{submission_id}"
    chosen_category_id = parse_optional_int(category_id)
    if chosen_category_id is None:
        return redirect_to(review_page, "Choose a category.")
    try:
        feature = review.assign_category_manually(
            session, context, submission_id, chosen_category_id, summary
        )
    except review.NotFoundError:
        raise HTTPException(404)
    except DECISION_ERRORS as error:
        return redirect_to(review_page, str(error))
    return redirect_to(f"/features/{feature.id}")


# --- Features -------------------------------------------------------------------


@router.get("/features", response_class=HTMLResponse)
def show_feature_list(
    request: Request,
    category: str = "",
    archived: str = "",
    session: Session = Depends(get_request_db_session),
):
    selected_category_id = parse_optional_int(category)
    show_archived = archived == "1"

    active_submission_counts = (
        select(Submission.feature_id, func.count().label("submission_count"))
        .where(Submission.archived_at.is_(None))
        .group_by(Submission.feature_id)
        .subquery()
    )
    query = (
        select(Feature, active_submission_counts.c.submission_count)
        .join(active_submission_counts, active_submission_counts.c.feature_id == Feature.id)
        .order_by(active_submission_counts.c.submission_count.desc(), Feature.updated_at.desc())
    )
    if show_archived:
        query = query.where(Feature.archived_at.is_not(None))
    else:
        query = query.where(Feature.archived_at.is_(None))
    if selected_category_id is not None:
        query = query.where(Feature.category_id == selected_category_id)

    features_with_counts = session.execute(query).all()
    return templates.TemplateResponse(
        request,
        "features.html",
        {
            "rows": features_with_counts,
            "categories": list_active_categories(session),
            "selected": selected_category_id,
            "show_archived": show_archived,
        },
    )


def list_other_active_features(session: Session, feature: Feature) -> list[Feature]:
    """Features a submission could be moved to: every other non-deleted feature."""
    query = (
        select(Feature)
        .where(Feature.id != feature.id, Feature.archived_at.is_(None))
        .order_by(Feature.title)
    )
    return list(session.scalars(query))


@router.get("/features/{feature_id}", response_class=HTMLResponse)
def show_feature_detail(
    feature_id: int, request: Request, session: Session = Depends(get_request_db_session)
):
    feature = session.get(Feature, feature_id)
    if feature is None:
        raise HTTPException(404)

    approved_merges_query = select(Proposal).where(
        Proposal.feature_id == feature.id,
        Proposal.type == ProposalType.DUPLICATE,
        Proposal.status == ProposalStatus.APPROVED,
    )
    merge_proposal_by_submission_id = {}
    for proposal in session.scalars(approved_merges_query):
        merge_proposal_by_submission_id[proposal.submission_id] = proposal

    summary_updates_query = select(Proposal).where(
        Proposal.feature_id == feature.id,
        Proposal.type == ProposalType.SUMMARY,
        Proposal.status.in_([ProposalStatus.APPROVED, ProposalStatus.EDITED]),
    )
    submissions_that_updated_summary = set()
    for proposal in session.scalars(summary_updates_query):
        submissions_that_updated_summary.add(proposal.submission_id)

    other_categories = []
    for category in list_active_categories(session):
        if category.id != feature.category_id:
            other_categories.append(category)

    context: TriageContext = request.app.state.triage
    category_changes = context.decision_log.read_events_for_feature(
        feature.id, EventType.CATEGORY_REASSIGNED
    )

    return templates.TemplateResponse(
        request,
        "feature.html",
        {
            "feature": feature,
            "merges": merge_proposal_by_submission_id,
            "summary_updates": submissions_that_updated_summary,
            "categories": other_categories,
            "other_features": list_other_active_features(session, feature),
            "reassignments": category_changes,
            **messages_from_query(request),
        },
    )


@router.post("/features/{feature_id}/edit")
def handle_feature_edit(
    feature_id: int,
    title: str = Form(""),
    summary: str = Form(""),
    session: Session = Depends(get_request_db_session),
    context: TriageContext = Depends(get_triage_context),
):
    feature_page = f"/features/{feature_id}"

    try:
        feature_management.edit_feature_details(session, context, feature_id, title, summary)
    except review.NotFoundError:
        raise HTTPException(404)
    except DECISION_ERRORS as error:
        return redirect_to(feature_page, str(error))
    return redirect_to(feature_page, notice="Feature updated.")


@router.post("/features/{feature_id}/archive")
def handle_feature_archive(
    feature_id: int,
    session: Session = Depends(get_request_db_session),
    context: TriageContext = Depends(get_triage_context),
):
    feature_page = f"/features/{feature_id}"

    try:
        feature_management.archive_feature(session, context, feature_id)
    except review.NotFoundError:
        raise HTTPException(404)
    except DECISION_ERRORS as error:
        return redirect_to(feature_page, str(error))
    return redirect_to(feature_page, notice="Feature deleted. You can restore it from this page.")


@router.post("/features/{feature_id}/restore")
def handle_feature_restore(
    feature_id: int,
    session: Session = Depends(get_request_db_session),
    context: TriageContext = Depends(get_triage_context),
):
    feature_page = f"/features/{feature_id}"

    try:
        feature_management.restore_feature(session, context, feature_id)
    except review.NotFoundError:
        raise HTTPException(404)
    except DECISION_ERRORS as error:
        return redirect_to(feature_page, str(error))
    return redirect_to(feature_page, notice="Feature restored.")


@router.post("/features/{feature_id}/category")
def handle_feature_category_change(
    feature_id: int,
    category_id: str = Form(""),
    new_name: str = Form(""),
    new_description: str = Form(""),
    reason: str = Form(""),
    session: Session = Depends(get_request_db_session),
    context: TriageContext = Depends(get_triage_context),
):
    feature_page = f"/features/{feature_id}"
    chosen_category_id = parse_optional_int(category_id)
    wants_new_category = bool(new_name.strip())

    if wants_new_category and chosen_category_id is not None:
        return redirect_to(feature_page, "Choose an existing category or name a new one, not both")
    if not wants_new_category and chosen_category_id is None:
        return redirect_to(feature_page, "Choose an existing category or name a new one")

    try:
        if wants_new_category:
            feature_management.move_feature_to_new_category(
                session, context, feature_id, new_name, new_description, reason
            )
        else:
            feature_management.move_feature_to_existing_category(
                session, context, feature_id, chosen_category_id, reason
            )
    except review.NotFoundError:
        raise HTTPException(404)
    except DECISION_ERRORS as error:
        return redirect_to(feature_page, str(error))
    return redirect_to(feature_page, notice="Category changed.")


# --- Merged submissions inside a feature -------------------------------------------


def feature_page_for_submission(session: Session, submission_id: int) -> str:
    submission = session.get(Submission, submission_id)
    if submission is None or submission.feature_id is None:
        raise HTTPException(404)
    return f"/features/{submission.feature_id}"


@router.post("/submissions/{submission_id}/unmerge")
def handle_submission_unmerge(
    submission_id: int,
    session: Session = Depends(get_request_db_session),
    context: TriageContext = Depends(get_triage_context),
):
    feature_page = feature_page_for_submission(session, submission_id)
    try:
        new_feature = feature_management.unmerge_submission(session, context, submission_id)
    except review.NotFoundError:
        raise HTTPException(404)
    except DECISION_ERRORS as error:
        return redirect_to(feature_page, str(error))
    notice = (
        f"{format_submission_id(submission_id)} is now its own feature, "
        f"{format_feature_id(new_feature.id)}. You may want to edit this feature's summary."
    )
    return redirect_to(feature_page, notice=notice)


@router.post("/submissions/{submission_id}/move")
def handle_submission_move(
    submission_id: int,
    target_feature_id: str = Form(""),
    session: Session = Depends(get_request_db_session),
    context: TriageContext = Depends(get_triage_context),
):
    feature_page = feature_page_for_submission(session, submission_id)
    chosen_feature_id = parse_optional_int(target_feature_id)
    if chosen_feature_id is None:
        return redirect_to(feature_page, "Choose a feature to move it to.")
    try:
        target_feature = feature_management.move_submission_to_feature(
            session, context, submission_id, chosen_feature_id
        )
    except review.NotFoundError:
        raise HTTPException(404)
    except DECISION_ERRORS as error:
        return redirect_to(feature_page, str(error))
    notice = f"{format_submission_id(submission_id)} moved to {format_feature_id(target_feature.id)}."
    return redirect_to(feature_page, notice=notice)


@router.post("/submissions/{submission_id}/archive")
def handle_submission_archive(
    submission_id: int,
    session: Session = Depends(get_request_db_session),
    context: TriageContext = Depends(get_triage_context),
):
    feature_page = feature_page_for_submission(session, submission_id)

    try:
        feature_management.archive_submission(session, context, submission_id)
    except review.NotFoundError:
        raise HTTPException(404)
    except DECISION_ERRORS as error:
        return redirect_to(feature_page, str(error))
    notice = f"{format_submission_id(submission_id)} deleted. You can restore it below."
    return redirect_to(feature_page, notice=notice)


@router.post("/submissions/{submission_id}/restore")
def handle_submission_restore(
    submission_id: int,
    session: Session = Depends(get_request_db_session),
    context: TriageContext = Depends(get_triage_context),
):
    feature_page = feature_page_for_submission(session, submission_id)

    try:
        feature_management.restore_submission(session, context, submission_id)
    except review.NotFoundError:
        raise HTTPException(404)
    except DECISION_ERRORS as error:
        return redirect_to(feature_page, str(error))
    return redirect_to(feature_page, notice=f"{format_submission_id(submission_id)} restored.")


# --- Categories -----------------------------------------------------------------


@router.get("/categories", response_class=HTMLResponse)
def show_categories(request: Request, session: Session = Depends(get_request_db_session)):
    feature_count_query = (
        select(Feature.category_id, func.count())
        .where(Feature.archived_at.is_(None))
        .group_by(Feature.category_id)
    )
    feature_count_by_category_id = {}
    for category_id, feature_count in session.execute(feature_count_query):
        feature_count_by_category_id[category_id] = feature_count

    pending_taxonomy_query = select(Proposal).where(
        Proposal.type == ProposalType.TAXONOMY,
        Proposal.status == ProposalStatus.PENDING,
    )
    active_categories_query = (
        select(Category)
        .where(Category.status == CategoryStatus.ACTIVE)
        .order_by(Category.name)
    )
    return templates.TemplateResponse(
        request,
        "categories.html",
        {
            "categories": session.scalars(active_categories_query).all(),
            "counts": feature_count_by_category_id,
            "taxonomy_proposals": session.scalars(pending_taxonomy_query).all(),
            "error": request.query_params.get("error"),
        },
    )


def read_kept_categories_from_form(form) -> list[dict]:
    """The taxonomy form has numbered rows: include_0, name_0, description_0, ...

    Returns the rows whose "keep" checkbox is ticked.
    """
    kept_categories = []
    row_number = 0
    while f"name_{row_number}" in form:
        if form.get(f"include_{row_number}"):
            kept_categories.append(
                {
                    "name": str(form[f"name_{row_number}"]),
                    "description": str(form.get(f"description_{row_number}", "")),
                }
            )
        row_number += 1
    return kept_categories


@router.post("/proposals/{proposal_id}/taxonomy")
async def handle_taxonomy_decision(
    proposal_id: int,
    request: Request,
    session: Session = Depends(get_request_db_session),
    context: TriageContext = Depends(get_triage_context),
):
    form = await request.form()
    decision = form.get("decision", "")
    try:
        if decision == "reject":
            taxonomy.reject_taxonomy(session, context, proposal_id)
        else:
            kept_categories = read_kept_categories_from_form(form)
            taxonomy.approve_taxonomy(session, context, proposal_id, kept_categories)
    except review.NotFoundError:
        raise HTTPException(404)
    except DECISION_ERRORS as error:
        return redirect_to("/categories", str(error))
    return redirect_to("/categories")
