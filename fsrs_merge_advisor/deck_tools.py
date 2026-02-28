from __future__ import annotations

from collections.abc import Sequence


def leaf_deck_entries(entries: Sequence[tuple[int, str]]) -> list[tuple[int, str]]:
    ancestor_names: set[str] = set()
    for _, name in entries:
        parts = name.split("::")
        for idx in range(1, len(parts)):
            ancestor_names.add("::".join(parts[:idx]))
    return [(deck_id, name) for deck_id, name in entries if name not in ancestor_names]


def count_relearning_steps_in_day(steps: Sequence[float]) -> int:
    count = 0
    accumulated_minutes = 0.0
    for raw_value in steps:
        value = float(raw_value)
        accumulated_minutes += value
        if accumulated_minutes >= 1440:
            break
        count += 1
    return count


def build_deck_search_query(
    *,
    deck_id: int,
    deck_name: str,
    include_children: bool,
) -> str:
    if not include_children:
        return f"did:{deck_id} -is:suspended"

    escaped_name = deck_name.replace("\\", "\\\\").replace('"', '\\"')
    return f'deck:"{escaped_name}" -is:suspended'


def optimization_progress_message(
    *,
    done: int,
    total: int,
    deck_name: str | None = None,
) -> str:
    remaining = max(total - done, 0)
    if deck_name:
        return (
            f'Optimizing "{deck_name}"\n'
            f"Completed: {done}/{total}\n"
            f"Remaining: {remaining}"
        )
    return f"Preparing deck optimizations...\nCompleted: {done}/{total}\nRemaining: {remaining}"


def similar_items_below_threshold(
    *,
    names: Sequence[str],
    distances_row: Sequence[float | None],
    self_index: int,
    threshold: float,
) -> list[str]:
    pairs: list[tuple[str, float]] = []
    for idx, value in enumerate(distances_row):
        if idx == self_index or value is None:
            continue
        if value < threshold:
            pairs.append((names[idx], value))
    pairs.sort(key=lambda item: (item[1], item[0].lower()))
    return [f"{name} ({distance:.4f})" for name, distance in pairs]
