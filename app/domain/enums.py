from enum import StrEnum


class SubmissionStatus(StrEnum):
    PENDING = "pending"  # awaiting human review
    NEW_FEATURE = "new_feature"  # created its own feature
    MERGED = "merged"  # consolidated into an existing feature


class FeatureStatus(StrEnum):
    TRIAGED = "triaged"


class CategoryStatus(StrEnum):
    PROPOSED = "proposed"  # suggested by the AI, not yet approved
    ACTIVE = "active"
    RETIRED = "retired"


class ProposalType(StrEnum):
    DUPLICATE = "duplicate"
    SUMMARY = "summary"
    CATEGORY = "category"
    NEW_CATEGORY = "new_category"
    TAXONOMY = "taxonomy"


class ProposalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited"  # approved with human changes


class Decision(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    EDIT = "edit"
