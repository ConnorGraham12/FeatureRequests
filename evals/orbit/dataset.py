"""Loads the Orbit mock dataset: the gold taxonomy, the existing features, and the probes."""

import json
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).parent


@dataclass(frozen=True)
class Item:
    """One feature request in the dataset, with its expected answers."""

    id: str
    title: str
    description: str
    category: str  # the gold category
    acceptable: tuple[str, ...] = ()  # other categories that also count as correct
    kind: str = "corpus"  # corpus, paraphrase, extension, hard_negative or easy_negative
    duplicate_of: str | None = None  # the corpus id this probe duplicates, if any
    confusable_with: str | None = None  # for hard negatives: the look-alike corpus id

    def is_acceptable_category(self, category_name: str | None) -> bool:
        return category_name == self.category or category_name in self.acceptable


def load_items_from_jsonl(path: Path) -> list[Item]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["acceptable"] = tuple(row.get("acceptable", ()))
        items.append(Item(**row))
    return items


def load_taxonomy(data_dir: Path = DATA_DIR) -> list[dict]:
    return json.loads((data_dir / "taxonomy.json").read_text(encoding="utf-8"))


def load_corpus(data_dir: Path = DATA_DIR) -> list[Item]:
    return load_items_from_jsonl(data_dir / "corpus.jsonl")


def load_probes(data_dir: Path = DATA_DIR) -> list[Item]:
    return load_items_from_jsonl(data_dir / "probes.jsonl")
