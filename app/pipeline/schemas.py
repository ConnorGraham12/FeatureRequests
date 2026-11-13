"""Structured outputs the model must return, one schema per pipeline stage."""

from typing import Literal

from pydantic import BaseModel, Field


class RejectedCandidate(BaseModel):
    feature_id: int
    reason: str


class DuplicateJudgment(BaseModel):
    duplicate_of_feature_id: int | None = Field(
        description="ID of the candidate feature this request duplicates, or null if none."
    )
    confidence: float = Field(description="0 to 1: how sure you are of this verdict.")
    reasoning: str = Field(description="One or two sentences explaining the verdict.")
    rejected_candidates: list[RejectedCandidate] = Field(
        description="Each candidate you decided is NOT a duplicate, with a short reason."
    )


class SummaryUpdate(BaseModel):
    summary: str = Field(description="The full updated feature summary.")
    changed: bool = Field(description="True if the new request added information.")
    reasoning: str


class CategoryAssignment(BaseModel):
    action: Literal["assign", "propose_new"]
    category_id: int | None = Field(description="Existing category ID when action is assign.")
    new_category_name: str | None = Field(description="Name when action is propose_new.")
    new_category_description: str | None = Field(
        description="One-sentence description of what the new category covers."
    )
    feature_summary: str = Field(
        description="One or two neutral sentences summarizing what the user wants and why."
    )
    confidence: float = Field(description="0 to 1.")
    reasoning: str


class ProposedCategory(BaseModel):
    name: str
    description: str


class TaxonomyProposal(BaseModel):
    categories: list[ProposedCategory]
    reasoning: str
