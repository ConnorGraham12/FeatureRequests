"""Versioned prompt files.

Prompts live in app/ai/prompt_templates/<name>_<version>.txt and use $placeholders.
Each decision log entry records the prompt version and a hash of its text, so
a change to a prompt file is visible even if the version wasn't bumped.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from string import Template

PROMPTS_DIR = Path(__file__).parent / "prompt_templates"


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    text: str

    @property
    def id(self) -> str:
        return f"{self.name}_{self.version}"

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:12]

    def render(self, **values: str) -> str:
        # Template uses $name, so JSON braces in prompts need no escaping.
        return Template(self.text).substitute(**values)


def load_prompt(name: str, version: str, prompts_dir: Path = PROMPTS_DIR) -> Prompt:
    path = prompts_dir / f"{name}_{version}.txt"
    return Prompt(name=name, version=version, text=path.read_text(encoding="utf-8"))
