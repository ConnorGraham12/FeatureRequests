"""Loading versioned prompt files.

Prompts live in app/ai/prompt_templates/<name>_<version>.txt. Each decision log
entry records the prompt's version and a hash of its text, so an edit to a
prompt file shows up in the log even if nobody bumped the version.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

PROMPT_TEMPLATES_DIR = Path(__file__).parent / "prompt_templates"


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    text: str

    @property
    def versioned_name(self) -> str:
        """For example "categorize_v1"."""
        return f"{self.name}_{self.version}"

    @property
    def content_hash(self) -> str:
        """First 12 hex characters of the SHA-256 of the prompt text."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:12]


def load_prompt_template(
    name: str, version: str, templates_dir: Path = PROMPT_TEMPLATES_DIR
) -> PromptTemplate:
    path = templates_dir / f"{name}_{version}.txt"
    return PromptTemplate(name=name, version=version, text=path.read_text(encoding="utf-8"))
