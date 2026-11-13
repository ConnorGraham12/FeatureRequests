"""Free, local checks over the dataset, and a CASES.md file for human review.

    python -m evals.orbit.review_cases

Checks the labels, reports how cases are balanced, and runs the embedding
search (no API calls) to show which probes would reach the model's duplicate
check at the configured similarity threshold.
"""

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from app.ai.embeddings import build_text_to_embed, get_shared_embedder
from app.config import Settings, get_settings
from evals.orbit.dataset import DATA_DIR, Item, load_corpus, load_probes, load_taxonomy


# --- Label checks ---------------------------------------------------------------


def find_label_problems(corpus: list[Item], probes: list[Item], category_names: set[str]) -> list[str]:
    problems = []
    all_items = corpus + probes
    corpus_ids = {item.id for item in corpus}

    id_counts = Counter(item.id for item in all_items)
    for item_id, count in id_counts.items():
        if count > 1:
            problems.append(f"duplicate id {item_id}")

    title_counts = Counter(item.title.lower() for item in all_items)
    for title, count in title_counts.items():
        if count > 1:
            problems.append(f"repeated title {title!r}")

    for item in all_items:
        for category_name in (item.category, *item.acceptable):
            if category_name not in category_names:
                problems.append(f"{item.id}: unknown category {category_name!r}")
        if item.duplicate_of and item.duplicate_of not in corpus_ids:
            problems.append(f"{item.id}: duplicate_of {item.duplicate_of} is not in the corpus")
    return problems


# --- Embedding search -----------------------------------------------------------


@dataclass
class ProbeSearchResult:
    probe: Item
    candidate_ids: list[str]
    similarity_to_target: float | None  # target = its duplicate, or the look-alike for a hard negative
    target_is_candidate: bool | None


def run_embedding_search(settings: Settings, corpus: list[Item], probes: list[Item]) -> list[ProbeSearchResult]:
    embedder = get_shared_embedder(settings.embedding_model)
    corpus_vectors = embedder.embed_texts([build_text_to_embed(c.title, c.description) for c in corpus])
    probe_vectors = embedder.embed_texts([build_text_to_embed(p.title, p.description) for p in probes])
    # Vectors are normalized, so the dot product is the cosine similarity.
    similarity_matrix = probe_vectors @ corpus_vectors.T
    corpus_ids = [item.id for item in corpus]

    results = []
    for probe, similarities in zip(probes, similarity_matrix):
        best_first = np.argsort(-similarities)[: settings.duplicate_top_k]
        candidate_ids = []
        for index in best_first:
            if similarities[index] >= settings.duplicate_similarity_threshold:
                candidate_ids.append(corpus_ids[index])

        target_id = probe.duplicate_of or probe.confusable_with
        similarity_to_target = None
        target_is_candidate = None
        if target_id:
            similarity_to_target = float(similarities[corpus_ids.index(target_id)])
            target_is_candidate = target_id in candidate_ids

        results.append(ProbeSearchResult(probe, candidate_ids, similarity_to_target, target_is_candidate))
    return results


def print_dataset_report(
    settings: Settings,
    category_names: set[str],
    corpus: list[Item],
    probes: list[Item],
    search_results: list[ProbeSearchResult],
    problems: list[str],
) -> None:
    all_items = corpus + probes
    duplicates = [r for r in search_results if r.probe.duplicate_of]
    hard_negatives = [r for r in search_results if r.probe.kind == "hard_negative"]
    easy_negatives = [r for r in search_results if r.probe.kind == "easy_negative"]

    duplicates_found = sum(1 for r in duplicates if r.target_is_candidate)
    look_alikes_found = sum(1 for r in hard_negatives if r.target_is_candidate)
    easy_with_candidates = sum(1 for r in easy_negatives if r.candidate_ids)
    reaching_the_model = sum(1 for r in search_results if r.candidate_ids)

    print(f"Taxonomy: {len(category_names)} categories. Corpus: {len(corpus)} features. Probes: {len(probes)}")
    print(f"  probes by kind: {dict(Counter(p.kind for p in probes))}")
    print(f"  categories over all {len(all_items)} items: {dict(Counter(i.category for i in all_items))}")
    print(f"  items with an acceptable second category: {sum(1 for i in all_items if i.acceptable)}")
    print(
        f"Embedding search ({settings.embedding_model}, "
        f"threshold {settings.duplicate_similarity_threshold}, k {settings.duplicate_top_k}):"
    )
    print(f"  duplicates whose true feature is a candidate: {duplicates_found}/{len(duplicates)}")
    print(f"  hard negatives whose look-alike is a candidate: {look_alikes_found}/{len(hard_negatives)}")
    print(f"  easy negatives with any candidate: {easy_with_candidates}/{len(easy_negatives)}")
    print(f"  probes that would reach the model's duplicate check: {reaching_the_model}/{len(probes)}")

    missed = []
    for result in duplicates:
        if not result.target_is_candidate:
            missed.append(f"{result.probe.id} ({result.similarity_to_target:.2f})")
    if missed:
        print(f"  duplicates the embedding search misses: {', '.join(missed)}")

    if problems:
        print("Problems:")
        for problem in problems:
            print(f"  {problem}")
    else:
        print("Problems: none")


# --- CASES.md ---------------------------------------------------------------------


def wrap_in_code_fence(text: str) -> str:
    """Fence text with more backticks than any run inside it, so it can't close the fence."""
    backtick_runs = re.findall(r"`+", text)
    longest_run = max((len(run) for run in backtick_runs), default=0)
    fence = "`" * max(3, longest_run + 1)
    return f"{fence}\n{text}\n{fence}"


def describe_category(item: Item) -> str:
    if item.acceptable:
        return f"{item.category} ({', '.join(item.acceptable)})"
    return item.category


def build_cases_markdown(
    settings: Settings, category_count: int, corpus: list[Item], search_results: list[ProbeSearchResult]
) -> str:
    corpus_by_id = {item.id: item for item in corpus}
    probes = [result.probe for result in search_results]
    lines = [
        "# Orbit eval cases",
        "",
        f"Generated by `python -m evals.orbit.review_cases`. {len(corpus)} existing features, "
        f"{len(probes)} probe submissions, {category_count} gold categories.",
        "",
        "Similarity = cosine similarity between the probe and its duplicate or look-alike "
        f"(threshold {settings.duplicate_similarity_threshold}).",
        "",
        "## Probes",
        "",
        "| id | kind | expected duplicate | look-alike | similarity | category (also OK) |",
        "|---|---|---|---|---|---|",
    ]
    for result in search_results:
        probe = result.probe
        if probe.duplicate_of:
            expected_duplicate = f"{probe.duplicate_of} {corpus_by_id[probe.duplicate_of].title}"
        else:
            expected_duplicate = "none"
        look_alike = ""
        if probe.confusable_with:
            look_alike = f"{probe.confusable_with} {corpus_by_id[probe.confusable_with].title}"
        similarity = ""
        if result.similarity_to_target is not None:
            similarity = f"{result.similarity_to_target:.2f}"
        lines.append(
            f"| {probe.id} | {probe.kind} | {expected_duplicate} | {look_alike} | {similarity} "
            f"| {describe_category(probe)} |"
        )

    lines += ["", "## Existing features (corpus)", "", "| id | title | category (also OK) |", "|---|---|---|"]
    for item in corpus:
        lines.append(f"| {item.id} | {item.title} | {describe_category(item)} |")

    lines += ["", "## Probe text", ""]
    for probe in probes:
        lines += [f"### {probe.id}: {probe.title}", "", wrap_in_code_fence(probe.description), ""]
    lines += ["## Corpus text", ""]
    for item in corpus:
        lines += [f"### {item.id}: {item.title}", "", wrap_in_code_fence(item.description), ""]
    return "\n".join(lines)


def main() -> None:
    settings = get_settings()
    category_names = {category["name"] for category in load_taxonomy()}
    corpus = load_corpus()
    probes = load_probes()

    problems = find_label_problems(corpus, probes, category_names)
    search_results = run_embedding_search(settings, corpus, probes)
    print_dataset_report(settings, category_names, corpus, probes, search_results, problems)

    output_path = DATA_DIR / "CASES.md"
    output_path.write_text(
        build_cases_markdown(settings, len(category_names), corpus, search_results), encoding="utf-8"
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
