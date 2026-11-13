from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Category, Feature, Proposal, Submission
from app.db.session import get_session
from app.domain.enums import CategoryStatus, Decision, ProposalStatus, ProposalType, SubmissionStatus
from app.domain.proposals import InvalidDecisionError, ProposalAlreadyDecidedError
from app.pipeline.stages import active_categories
from app.services import review, taxonomy, triage
from app.services.context import TriageContext

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
templates.env.filters["sid"] = lambda i: f"S-{i:03d}"
templates.env.filters["fid"] = lambda i: f"F-{i:03d}"
templates.env.filters["pct"] = lambda x: "—" if x is None else f"{x:.2f}"

router = APIRouter()

DECISION_ERRORS = (InvalidDecisionError, ProposalAlreadyDecidedError)


def get_ctx(request: Request) -> TriageContext:
    return request.app.state.triage


def _redirect(url: str, error: str | None = None) -> RedirectResponse:
    if error:
        url += ("&" if "?" in url else "?") + "error=" + quote(error)
    return RedirectResponse(url, status_code=303)


def _optional_int(value: str | None) -> int | None:
    value = (value or "").strip()
    return int(value) if value.isdigit() else None


# --- Submit ------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"error": request.query_params.get("error")})


@router.post("/submissions")
def create_submission(
    title: str = Form(""),
    description: str = Form(""),
    session: Session = Depends(get_session),
    ctx: TriageContext = Depends(get_ctx),
):
    title, description = title.strip(), description.strip()
    if not title or not description:
        return _redirect("/", "Both a title and a description are required.")
    submission = triage.submit(session, ctx, title, description)
    return _redirect(f"/submissions/{submission.id}")


# --- Review ------------------------------------------------------------------


@router.get("/review", response_class=HTMLResponse)
def review_queue(request: Request, session: Session = Depends(get_session)):
    pending = session.scalars(
        select(Submission)
        .where(Submission.status == SubmissionStatus.PENDING)
        .order_by(Submission.created_at)
    ).all()
    return templates.TemplateResponse(request, "queue.html", {"submissions": pending})


@router.get("/submissions/{submission_id}", response_class=HTMLResponse)
def review_submission(submission_id: int, request: Request, session: Session = Depends(get_session)):
    submission = session.get(Submission, submission_id)
    if submission is None:
        raise HTTPException(404)
    if submission.status is not SubmissionStatus.PENDING and submission.feature_id:
        return _redirect(f"/features/{submission.feature_id}")

    pending = {
        p.type: p for p in submission.proposals if p.status is ProposalStatus.PENDING
    }
    duplicate = pending.get(ProposalType.DUPLICATE)
    category_proposal = pending.get(ProposalType.CATEGORY) or pending.get(ProposalType.NEW_CATEGORY)
    target = session.get(Feature, duplicate.feature_id) if duplicate else None
    categories = active_categories(session)

    return templates.TemplateResponse(
        request,
        "review.html",
        {
            "submission": submission,
            "duplicate": duplicate,
            "summary": pending.get(ProposalType.SUMMARY),
            "target": target,
            "category_proposal": category_proposal,
            "categories": categories,
            "categories_by_id": {c.id: c for c in categories},
            "error": request.query_params.get("error"),
        },
    )


@router.post("/proposals/{proposal_id}/merge")
def decide_merge(
    proposal_id: int,
    action: str = Form(...),
    summary: str = Form(""),
    session: Session = Depends(get_session),
    ctx: TriageContext = Depends(get_ctx),
):
    proposal = session.get(Proposal, proposal_id)
    if proposal is None or proposal.submission_id is None:
        raise HTTPException(404)
    back = f"/submissions/{proposal.submission_id}"
    try:
        review.decide_merge(session, ctx, proposal_id, approve=action == "approve", edited_summary=summary)
    except review.NotFoundError:
        raise HTTPException(404)
    except DECISION_ERRORS as error:
        return _redirect(back, str(error))
    return _redirect(back)


@router.post("/proposals/{proposal_id}/category")
def decide_category(
    proposal_id: int,
    decision: Decision = Form(...),
    category_id: str = Form(""),
    summary: str = Form(""),
    name: str = Form(""),
    description: str = Form(""),
    session: Session = Depends(get_session),
    ctx: TriageContext = Depends(get_ctx),
):
    proposal = session.get(Proposal, proposal_id)
    if proposal is None or proposal.submission_id is None:
        raise HTTPException(404)
    back = f"/submissions/{proposal.submission_id}"
    try:
        feature = review.decide_category(
            session,
            ctx,
            proposal_id,
            decision,
            category_id=_optional_int(category_id),
            summary=summary,
            name=name,
            description=description,
        )
    except review.NotFoundError:
        raise HTTPException(404)
    except DECISION_ERRORS as error:
        return _redirect(back, str(error))
    return _redirect(f"/features/{feature.id}")


@router.post("/submissions/{submission_id}/manual-category")
def manual_category(
    submission_id: int,
    category_id: str = Form(""),
    summary: str = Form(""),
    session: Session = Depends(get_session),
    ctx: TriageContext = Depends(get_ctx),
):
    back = f"/submissions/{submission_id}"
    chosen = _optional_int(category_id)
    if chosen is None:
        return _redirect(back, "Choose a category.")
    try:
        feature = review.assign_manually(session, ctx, submission_id, chosen, summary)
    except review.NotFoundError:
        raise HTTPException(404)
    except DECISION_ERRORS as error:
        return _redirect(back, str(error))
    return _redirect(f"/features/{feature.id}")


# --- Features ----------------------------------------------------------------


@router.get("/features", response_class=HTMLResponse)
def list_features(request: Request, category: str = "", session: Session = Depends(get_session)):
    category_id = _optional_int(category)
    counts = (
        select(Submission.feature_id, func.count().label("n"))
        .group_by(Submission.feature_id)
        .subquery()
    )
    query = (
        select(Feature, counts.c.n)
        .join(counts, counts.c.feature_id == Feature.id)
        .order_by(counts.c.n.desc(), Feature.updated_at.desc())
    )
    if category_id is not None:
        query = query.where(Feature.category_id == category_id)
    return templates.TemplateResponse(
        request,
        "features.html",
        {
            "rows": session.execute(query).all(),
            "categories": active_categories(session),
            "selected": category_id,
        },
    )


@router.get("/features/{feature_id}", response_class=HTMLResponse)
def feature_detail(feature_id: int, request: Request, session: Session = Depends(get_session)):
    feature = session.get(Feature, feature_id)
    if feature is None:
        raise HTTPException(404)
    merges = {
        p.submission_id: p
        for p in session.scalars(
            select(Proposal).where(
                Proposal.feature_id == feature.id,
                Proposal.type == ProposalType.DUPLICATE,
                Proposal.status == ProposalStatus.APPROVED,
            )
        )
    }
    summary_updates = {
        p.submission_id
        for p in session.scalars(
            select(Proposal).where(
                Proposal.feature_id == feature.id,
                Proposal.type == ProposalType.SUMMARY,
                Proposal.status.in_([ProposalStatus.APPROVED, ProposalStatus.EDITED]),
            )
        )
    }
    return templates.TemplateResponse(
        request,
        "feature.html",
        {"feature": feature, "merges": merges, "summary_updates": summary_updates},
    )


# --- Categories --------------------------------------------------------------


@router.get("/categories", response_class=HTMLResponse)
def list_categories(request: Request, session: Session = Depends(get_session)):
    counts = dict(
        session.execute(
            select(Feature.category_id, func.count()).group_by(Feature.category_id)
        ).all()
    )
    taxonomy_proposals = session.scalars(
        select(Proposal).where(
            Proposal.type == ProposalType.TAXONOMY, Proposal.status == ProposalStatus.PENDING
        )
    ).all()
    return templates.TemplateResponse(
        request,
        "categories.html",
        {
            "categories": session.scalars(
                select(Category).where(Category.status == CategoryStatus.ACTIVE).order_by(Category.name)
            ).all(),
            "counts": counts,
            "taxonomy_proposals": taxonomy_proposals,
            "error": request.query_params.get("error"),
        },
    )


@router.post("/proposals/{proposal_id}/taxonomy")
async def decide_taxonomy(
    proposal_id: int,
    request: Request,
    session: Session = Depends(get_session),
    ctx: TriageContext = Depends(get_ctx),
):
    form = await request.form()
    decision = Decision(form.get("decision", ""))
    edited = []
    index = 0
    while f"name_{index}" in form:
        if form.get(f"include_{index}"):
            edited.append(
                {"name": str(form[f"name_{index}"]), "description": str(form.get(f"description_{index}", ""))}
            )
        index += 1
    try:
        taxonomy.decide(session, ctx, proposal_id, decision, edited)
    except review.NotFoundError:
        raise HTTPException(404)
    except DECISION_ERRORS as error:
        return _redirect("/categories", str(error))
    return _redirect("/categories")
