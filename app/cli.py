"""Command-line tools.

    python -m app.cli bootstrap-taxonomy requests.jsonl

Each line of the input file is a JSON object with "title" and "description".
The proposed categories then wait for approval on the /categories page.
"""

import argparse
import json
import sys
from pathlib import Path

from app.ai.llm import LLMError
from app.config import get_settings
from app.db.session import create_database_engine, create_session_factory, create_tables
from app.main import build_triage_context
from app.services import taxonomy


def read_requests_from_jsonl(path: Path) -> list[tuple[str, str]]:
    """Return (title, description) pairs. Stops with a clear message on a bad line."""
    requests = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if "title" not in record or "description" not in record:
            raise SystemExit(f"{path}:{line_number}: each line needs a title and a description")
        requests.append((record["title"], record["description"]))
    return requests


def bootstrap_taxonomy(path: Path) -> int:
    requests = read_requests_from_jsonl(path)
    if not requests:
        print(f"No requests found in {path}", file=sys.stderr)
        return 1

    settings = get_settings()
    engine = create_database_engine(settings.database_url)
    create_tables(engine)
    context = build_triage_context(settings)
    session_factory = create_session_factory(engine)

    with session_factory() as session:
        try:
            proposal = taxonomy.request_taxonomy_proposal(session, context, requests)
        except LLMError as error:
            print(f"Taxonomy proposal failed: {error}", file=sys.stderr)
            return 1

    proposed_categories = proposal.payload["categories"]
    print(f"Proposed {len(proposed_categories)} categories from {len(requests)} requests:")
    for category in proposed_categories:
        print(f"  - {category['name']}: {category['description']}")
    print(f"Correlation ID {proposal.correlation_id}. Approve or edit them at /categories.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    bootstrap_command = commands.add_parser(
        "bootstrap-taxonomy", help="Ask the AI to propose categories"
    )
    bootstrap_command.add_argument("file", type=Path, help="JSONL file of {title, description}")
    args = parser.parse_args()

    if args.command == "bootstrap-taxonomy":
        return bootstrap_taxonomy(args.file)
    return 1


if __name__ == "__main__":
    sys.exit(main())
