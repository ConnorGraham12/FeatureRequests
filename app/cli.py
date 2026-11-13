"""Command-line tools.

    python -m app.cli bootstrap-taxonomy requests.jsonl

Each line of the input file is a JSON object with "title" and "description".
The proposed taxonomy then waits for approval on the /categories page.
"""

import argparse
import json
import sys
from pathlib import Path

from app.ai.llm import LLMError
from app.config import get_settings
from app.db.session import init_db, make_engine, make_session_factory
from app.main import build_context
from app.services import taxonomy


def _read_requests(path: Path) -> list[tuple[str, str]]:
    requests = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        try:
            requests.append((record["title"], record["description"]))
        except KeyError as missing:
            raise SystemExit(f"{path}:{number}: missing field {missing}")
    return requests


def bootstrap_taxonomy(path: Path) -> int:
    requests = _read_requests(path)
    if not requests:
        print(f"No requests found in {path}", file=sys.stderr)
        return 1

    settings = get_settings()
    engine = make_engine(settings.database_url)
    init_db(engine)
    ctx = build_context(settings)
    with make_session_factory(engine)() as session:
        try:
            proposal = taxonomy.propose(session, ctx, requests)
        except LLMError as error:
            print(f"Taxonomy proposal failed: {error}", file=sys.stderr)
            return 1

    print(f"Proposed {len(proposal.payload['categories'])} categories from {len(requests)} requests:")
    for category in proposal.payload["categories"]:
        print(f"  - {category['name']}: {category['description']}")
    print(f"Correlation ID {proposal.correlation_id}. Approve or edit them at /categories.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    bootstrap = commands.add_parser("bootstrap-taxonomy", help="Ask the AI to propose categories")
    bootstrap.add_argument("file", type=Path, help="JSONL file of {title, description}")
    args = parser.parse_args()

    if args.command == "bootstrap-taxonomy":
        return bootstrap_taxonomy(args.file)
    return 1


if __name__ == "__main__":
    sys.exit(main())
