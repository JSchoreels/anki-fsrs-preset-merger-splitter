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


def descendant_deck_ids(
    entries: Sequence[tuple[int, str]],
    root_deck_name: str,
) -> list[int]:
    prefix = f"{root_deck_name}::"
    return [
        deck_id
        for deck_id, deck_name in entries
        if deck_name == root_deck_name or deck_name.startswith(prefix)
    ]


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


def can_reuse_cached_params(
    *,
    cached_review_count: int | None,
    current_review_count: int,
    cached_params: Sequence[float] | None,
) -> bool:
    return cached_review_count == current_review_count and cached_params is not None


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


def similarity_groups_from_matrix(
    *,
    names: Sequence[str],
    distances: Sequence[Sequence[float | None]],
    threshold: float,
    min_group_size: int = 2,
) -> list[list[str]]:
    total = len(names)
    if total == 0:
        return []
    if len(distances) != total:
        raise ValueError("Distance matrix row count must match names count")
    if any(len(row) != total for row in distances):
        raise ValueError("Distance matrix must be square and match names count")

    def _merge_distance(left: list[int], right: list[int]) -> float | None:
        max_distance = 0.0
        for li in left:
            for ri in right:
                value = distances[li][ri]
                if value is None or value >= threshold:
                    return None
                if value > max_distance:
                    max_distance = value
        return max_distance

    clusters: list[list[int]] = [[idx] for idx in range(total)]
    while True:
        best_pair: tuple[int, int] | None = None
        best_distance: float | None = None
        best_tie_breaker: tuple[str, str] | None = None

        for left_idx in range(len(clusters)):
            for right_idx in range(left_idx + 1, len(clusters)):
                merge_distance = _merge_distance(clusters[left_idx], clusters[right_idx])
                if merge_distance is None:
                    continue

                left_name = min((names[idx] for idx in clusters[left_idx]), key=str.lower)
                right_name = min((names[idx] for idx in clusters[right_idx]), key=str.lower)
                tie_breaker = (left_name.lower(), right_name.lower())
                if best_pair is None:
                    best_pair = (left_idx, right_idx)
                    best_distance = merge_distance
                    best_tie_breaker = tie_breaker
                    continue

                assert best_distance is not None
                assert best_tie_breaker is not None
                if merge_distance < best_distance or (
                    merge_distance == best_distance and tie_breaker < best_tie_breaker
                ):
                    best_pair = (left_idx, right_idx)
                    best_distance = merge_distance
                    best_tie_breaker = tie_breaker

        if best_pair is None:
            break

        left_idx, right_idx = best_pair
        merged = sorted(clusters[left_idx] + clusters[right_idx])
        del clusters[right_idx]
        clusters[left_idx] = merged

    groups: list[list[str]] = []
    for cluster in clusters:
        if len(cluster) < min_group_size:
            continue
        group_names = sorted((names[idx] for idx in cluster), key=lambda name: name.lower())
        groups.append(group_names)

    groups.sort(key=lambda group: (-len(group), group[0].lower()))
    return groups


def grouped_names_by_label(pairs: Sequence[tuple[str, str]]) -> list[tuple[str, list[str]]]:
    grouped: dict[str, list[str]] = {}
    for label, name in pairs:
        grouped.setdefault(label, []).append(name)
    result = [
        (label, sorted(names, key=lambda value: value.lower()))
        for label, names in grouped.items()
    ]
    result.sort(key=lambda item: item[0].lower())
    return result


def max_pairwise_distance_for_group(
    *,
    group_indexes: Sequence[int],
    distances: Sequence[Sequence[float | None]],
) -> float | None:
    if len(group_indexes) < 2:
        return None
    values: list[float] = []
    for left in range(len(group_indexes)):
        for right in range(left + 1, len(group_indexes)):
            value = distances[group_indexes[left]][group_indexes[right]]
            if value is not None:
                values.append(float(value))
    return max(values) if values else None


def max_distance_to_group_for_item(
    *,
    item_index: int,
    group_indexes: Sequence[int],
    distances: Sequence[Sequence[float | None]],
) -> float | None:
    values = [
        float(distances[item_index][other_idx])
        for other_idx in group_indexes
        if other_idx != item_index and distances[item_index][other_idx] is not None
    ]
    if not values:
        return None
    return max(values)


def recommended_group_preset_name(group_number: int) -> str:
    return f"FSRS Preset Advisor : Group {group_number}"


def unique_name(base: str, existing: Sequence[str]) -> str:
    existing_set = set(existing)
    if base not in existing_set:
        return base
    suffix = 2
    while True:
        candidate = f"{base} ({suffix})"
        if candidate not in existing_set:
            return candidate
        suffix += 1
