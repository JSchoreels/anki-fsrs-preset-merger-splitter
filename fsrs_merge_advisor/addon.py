from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any, Callable

from aqt import mw
from aqt.qt import (
    QAbstractItemView,
    QAction,
    QColor,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    Qt,
    QVBoxLayout,
    QWidget,
)
from aqt.utils import showInfo, showWarning

from .analyzer import (
    FSRSProfile,
    analyze_profiles,
    extract_fsrs_weights,
    is_fsrs6_valid_params,
    pairwise_distance_matrix,
)
from .deck_tools import (
    build_deck_search_query,
    can_reuse_cached_params,
    count_relearning_steps_in_day,
    descendant_deck_ids,
    leaf_deck_entries,
    max_distance_to_group_for_item,
    max_pairwise_distance_for_group,
    optimization_progress_message,
    recommended_group_preset_name,
    similar_items_below_threshold,
    similarity_groups_from_matrix,
    unique_name,
)
from .reference_covariance import FSRS6_RECENCY_MAHALANOBIS_SHARED_PRESET_THRESHOLD

_ACTION_LABEL = "FSRS Preset Proximity"
_DECK_ACTION_LABEL = "FSRS Deck Proximity (Computed)"
_CACHE_FILE_NAME = "deck_params_cache.json"
_PRESET_BACKUP_FILE_NAME = "deck_preset_backup.json"


class _PaneScopedListWidget(QListWidget):
    def __init__(self, pane_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pane_name = pane_name

    def _same_pane_source(self, event: Any) -> bool:
        source = event.source()
        return (
            isinstance(source, _PaneScopedListWidget)
            and source._pane_name == self._pane_name
        )

    def dragEnterEvent(self, event: Any) -> None:
        if self._same_pane_source(event):
            super().dragEnterEvent(event)
        else:
            event.ignore()

    def dragMoveEvent(self, event: Any) -> None:
        if self._same_pane_source(event):
            super().dragMoveEvent(event)
        else:
            event.ignore()

    def dropEvent(self, event: Any) -> None:
        if self._same_pane_source(event):
            super().dropEvent(event)
        else:
            event.ignore()


def _deck_entries() -> list[tuple[int, str]]:
    entries = []
    for item in mw.col.decks.all_names_and_ids():
        if isinstance(item, Mapping):
            deck_id = int(item["id"])
            name = str(item["name"])
        else:
            deck_id = int(getattr(item, "id"))
            name = str(getattr(item, "name"))
        entries.append((deck_id, name))
    return entries


def _legacy_cache_file_path() -> Path:
    return Path(__file__).resolve().parent / _CACHE_FILE_NAME


def _preferred_cache_file_path() -> Path:
    pm = getattr(mw, "pm", None)
    profile_folder_getter = getattr(pm, "profileFolder", None)
    if callable(profile_folder_getter):
        try:
            profile_folder = profile_folder_getter()
            if isinstance(profile_folder, str) and profile_folder:
                return Path(profile_folder) / _CACHE_FILE_NAME
        except Exception:
            pass
    return _legacy_cache_file_path()


def _cache_file_candidates() -> list[Path]:
    preferred = _preferred_cache_file_path()
    legacy = _legacy_cache_file_path()
    if preferred == legacy:
        return [preferred]
    return [preferred, legacy]


def _legacy_preset_backup_file_path() -> Path:
    return Path(__file__).resolve().parent / _PRESET_BACKUP_FILE_NAME


def _preferred_preset_backup_file_path() -> Path:
    pm = getattr(mw, "pm", None)
    profile_folder_getter = getattr(pm, "profileFolder", None)
    if callable(profile_folder_getter):
        try:
            profile_folder = profile_folder_getter()
            if isinstance(profile_folder, str) and profile_folder:
                return Path(profile_folder) / _PRESET_BACKUP_FILE_NAME
        except Exception:
            pass
    return _legacy_preset_backup_file_path()


def _preset_backup_file_candidates() -> list[Path]:
    preferred = _preferred_preset_backup_file_path()
    legacy = _legacy_preset_backup_file_path()
    if preferred == legacy:
        return [preferred]
    return [preferred, legacy]


def _save_preset_backup(assignments: Mapping[int, int]) -> None:
    payload = {
        "version": 1,
        "entries": {str(int(deck_id)): int(conf_id) for deck_id, conf_id in assignments.items()},
    }
    path = _preferred_preset_backup_file_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    except Exception:
        pass


def _load_preset_backup() -> dict[int, int]:
    preferred = _preferred_preset_backup_file_path()
    for path in _preset_backup_file_candidates():
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, Mapping):
            continue
        entries: dict[int, int] = {}
        for raw_deck_id, raw_conf_id in raw_entries.items():
            deck_id = _as_int(raw_deck_id)
            conf_id = _as_int(raw_conf_id)
            if deck_id is None or conf_id is None:
                continue
            entries[deck_id] = conf_id
        if entries and path != preferred:
            _save_preset_backup(entries)
        return entries
    return {}


def _current_preset_assignments(deck_ids: Sequence[int]) -> dict[int, int]:
    assignments: dict[int, int] = {}
    for deck_id in deck_ids:
        try:
            deck = mw.col.decks.get(deck_id)
        except Exception:
            continue
        conf_id = _as_int(_field_any(deck, ("conf", "config_id")))
        if conf_id is not None:
            assignments[int(deck_id)] = conf_id
    return assignments


def _apply_preset_assignments(assignments: Mapping[int, int]) -> tuple[int, int]:
    changed = 0
    failed = 0
    setter = getattr(mw.col.decks, "set_config_id_for_deck_dict", None)
    for deck_id, preset_id in assignments.items():
        try:
            deck = mw.col.decks.get(deck_id)
            if not deck:
                failed += 1
                continue

            current_preset = _as_int(_field_any(deck, ("conf", "config_id")))
            if current_preset == preset_id:
                continue

            if setter is not None:
                setter(deck, preset_id)
            else:
                if isinstance(deck, Mapping):
                    deck["conf"] = preset_id
                else:
                    setattr(deck, "conf", preset_id)
                mw.col.decks.save(deck)
            changed += 1
        except Exception:
            failed += 1

    try:
        mw.reset()
    except Exception:
        pass
    return changed, failed


def _load_deck_param_cache() -> dict[str, dict[str, Any]]:
    preferred = _preferred_cache_file_path()
    for path in _cache_file_candidates():
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            continue
        normalized: dict[str, dict[str, Any]] = {}
        for key, value in entries.items():
            if isinstance(key, str) and isinstance(value, dict):
                normalized[key] = value

        if normalized and path != preferred:
            _save_deck_param_cache(normalized)
        return normalized
    return {}


def _save_deck_param_cache(entries: Mapping[str, Mapping[str, Any]]) -> None:
    payload = {"version": 1, "entries": dict(entries)}
    path = _preferred_cache_file_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    except Exception:
        # Ignore cache persistence errors; computation can proceed without cache.
        pass


def _cache_key(deck_id: int, *, include_children: bool) -> str:
    scope = "tree" if include_children else "single"
    return f"{scope}:{deck_id}"


def _review_count_for_deck_scope(
    *,
    deck_id: int,
    deck_name: str,
    include_children: bool,
    entries: Sequence[tuple[int, str]],
) -> int:
    scope_ids = (
        descendant_deck_ids(entries, deck_name)
        if include_children
        else [deck_id]
    )
    if not scope_ids:
        return 0
    placeholders = ",".join("?" for _ in scope_ids)
    sql = (
        "SELECT COUNT(*) "
        "FROM revlog r "
        "JOIN cards c ON c.id = r.cid "
        f"WHERE c.did IN ({placeholders})"
    )
    count = mw.col.db.scalar(sql, *scope_ids)
    return int(count or 0)


def _as_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _field_any(obj: Any, names: Sequence[str]) -> Any:
    for name in names:
        value = _field(obj, name)
        if value is not None:
            return value
    return None


def _config_name(conf_id: int, config: Any) -> str:
    name = _field(config, "name")
    return str(name) if isinstance(name, str) and name else f"Preset {conf_id}"


def _config_from_conf_id(conf_id: int) -> Any:
    getter_names = ["get_config", "get_config_dict", "dconf_for_update_dict", "getconf"]
    for getter_name in getter_names:
        getter = getattr(mw.col.decks, getter_name, None)
        if getter is None:
            continue
        try:
            value = getter(conf_id)
        except Exception:
            continue
        if value is not None:
            return value
    return None


def _config_for_deck(deck_id: int) -> Any:
    by_deck = getattr(mw.col.decks, "config_dict_for_deck_id", None)
    if by_deck is not None:
        try:
            value = by_deck(deck_id)
            if value is not None:
                return value
        except Exception:
            pass

    deck = mw.col.decks.get(deck_id)
    conf_id = _field_any(deck, ("conf", "config_id"))
    if isinstance(conf_id, int):
        return _config_from_conf_id(conf_id)

    return None


def _normalize_config(config: Any, fallback_id: Any = None) -> tuple[int, str, Any] | None:
    conf_id = _as_int(_field(config, "id"))
    if conf_id is None:
        conf_id = _as_int(fallback_id)
    if conf_id is None:
        return None
    conf_name = _config_name(conf_id, config)
    return conf_id, conf_name, config


def _all_preset_configs() -> list[tuple[int, str, Any]]:
    seen: dict[int, tuple[str, Any]] = {}

    getter_names = (
        "all_config",
        "all_configs",
        "all_config_dict",
        "all_config_dicts",
        "all_confs",
    )
    for getter_name in getter_names:
        getter = getattr(mw.col.decks, getter_name, None)
        if getter is None:
            continue
        try:
            raw = getter()
        except Exception:
            continue

        if isinstance(raw, Mapping):
            entries = list(raw.items())
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            entries = [(None, config) for config in raw]
        else:
            continue

        for key, config in entries:
            normalized = _normalize_config(config, fallback_id=key)
            if normalized is None:
                continue
            conf_id, conf_name, conf_obj = normalized
            seen[conf_id] = (conf_name, conf_obj)

    if seen:
        return sorted((conf_id, item[0], item[1]) for conf_id, item in seen.items())

    # Fallback for older API shapes: gather presets referenced by decks.
    for deck_id, _ in _deck_entries():
        deck = mw.col.decks.get(deck_id)
        conf_id = _as_int(_field_any(deck, ("conf", "config_id")))
        if conf_id is None:
            continue
        if conf_id in seen:
            continue

        config = _config_from_conf_id(conf_id)
        if config is None:
            continue
        name = _config_name(conf_id, config)
        seen[conf_id] = (name, config)

    return sorted((conf_id, item[0], item[1]) for conf_id, item in seen.items())


def _load_profiles() -> list[FSRSProfile]:
    profiles: list[FSRSProfile] = []

    for conf_id, conf_name, config in _all_preset_configs():
        weights = extract_fsrs_weights(config)
        if weights is None:
            continue

        profiles.append(FSRSProfile(profile_id=conf_id, profile_name=conf_name, weights=weights))

    return profiles


def _to_float_sequence(value: Any) -> tuple[float, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None


def _extract_relearning_steps(config: Any) -> tuple[float, ...]:
    direct = _to_float_sequence(_field_any(config, ("relearnSteps", "relearn_steps")))
    if direct is not None:
        return direct

    lapse = _field(config, "lapse")
    from_lapse = _to_float_sequence(_field_any(lapse, ("delays", "steps", "relearn_steps")))
    if from_lapse is not None:
        return from_lapse

    return ()


def _num_relearning_steps_in_day_for_deck(deck_id: int) -> int:
    config = _config_for_deck(deck_id)
    steps = _extract_relearning_steps(config)
    return count_relearning_steps_in_day(steps)


def _compute_fsrs_params_for_deck(
    *,
    search: str,
    current_params: tuple[float, ...],
    num_of_relearning_steps: int,
) -> tuple[tuple[float, ...], int]:
    backend = getattr(mw.col, "_backend", None)
    compute = getattr(backend, "compute_fsrs_params", None)
    if compute is None:
        raise RuntimeError("compute_fsrs_params backend endpoint is unavailable")

    base_kwargs = {
        "search": search,
        "ignore_revlogs_before_ms": 0,
        "health_check": False,
    }
    candidates = [
        {
            **base_kwargs,
            "current_params": list(current_params),
            "num_of_relearning_steps": num_of_relearning_steps,
        },
        {
            "search": base_kwargs["search"],
            "ignoreRevlogsBeforeMs": 0,
            "healthCheck": False,
            "currentParams": list(current_params),
            "numOfRelearningSteps": num_of_relearning_steps,
        },
    ]

    response = None
    last_type_error: Exception | None = None
    for kwargs in candidates:
        try:
            response = compute(**kwargs)
            break
        except TypeError as exc:
            last_type_error = exc
            continue

    if response is None:
        if last_type_error is not None:
            raise RuntimeError("Unable to call compute_fsrs_params with known argument names") from last_type_error
        raise RuntimeError("compute_fsrs_params failed without a result")

    params = _to_float_sequence(_field(response, "params"))
    if params is None:
        params = ()
    fsrs_items = _as_int(_field_any(response, ("fsrs_items", "fsrsItems"))) or 0
    return params, fsrs_items


def _load_computed_deck_profiles(
    *,
    include_middle_and_root: bool,
    progress_callback: Callable[[int, int, str | None], bool] | None = None,
) -> tuple[list[FSRSProfile], list[str], list[str], list[str], bool]:
    entries = sorted(_deck_entries(), key=lambda item: item[1].lower())
    selected_entries = entries if include_middle_and_root else leaf_deck_entries(entries)
    cache = _load_deck_param_cache()
    cache_dirty = False

    profiles: list[FSRSProfile] = []
    no_data_decks: list[str] = []
    invalid_param_decks: list[str] = []
    failed_decks: list[str] = []
    total = len(selected_entries)
    processed = 0

    def _save_cache_if_needed() -> None:
        if cache_dirty:
            _save_deck_param_cache(cache)

    if progress_callback is not None and not progress_callback(0, total, None):
        _save_cache_if_needed()
        return profiles, no_data_decks, invalid_param_decks, failed_decks, True

    for deck_id, deck_name in selected_entries:
        if progress_callback is not None and not progress_callback(processed, total, deck_name):
            _save_cache_if_needed()
            return profiles, no_data_decks, invalid_param_decks, failed_decks, True

        config = _config_for_deck(deck_id)
        current_params = extract_fsrs_weights(config) or ()
        num_of_relearning_steps = _num_relearning_steps_in_day_for_deck(deck_id)
        review_count = _review_count_for_deck_scope(
            deck_id=deck_id,
            deck_name=deck_name,
            include_children=include_middle_and_root,
            entries=entries,
        )
        search = build_deck_search_query(
            deck_id=deck_id,
            deck_name=deck_name,
            include_children=include_middle_and_root,
        )
        cache_key = _cache_key(deck_id, include_children=include_middle_and_root)
        cached_entry = cache.get(cache_key, {})
        cached_review_count = _as_int(cached_entry.get("review_count"))
        cached_params = _to_float_sequence(cached_entry.get("params"))
        cached_fsrs_items = _as_int(cached_entry.get("fsrs_items")) or 0

        if can_reuse_cached_params(
            cached_review_count=cached_review_count,
            current_review_count=review_count,
            cached_params=cached_params,
        ):
            params = cached_params
            fsrs_items = cached_fsrs_items
        else:
            try:
                params, fsrs_items = _compute_fsrs_params_for_deck(
                    search=search,
                    current_params=current_params,
                    num_of_relearning_steps=num_of_relearning_steps,
                )
            except Exception:
                failed_decks.append(deck_name)
                processed += 1
                continue
            cache[cache_key] = {
                "review_count": review_count,
                "params": list(params),
                "fsrs_items": fsrs_items,
            }
            cache_dirty = True

        if fsrs_items <= 0:
            no_data_decks.append(deck_name)
            processed += 1
            continue
        if not is_fsrs6_valid_params(params):
            invalid_param_decks.append(deck_name)
            processed += 1
            continue

        profiles.append(FSRSProfile(profile_id=deck_id, profile_name=deck_name, weights=params))
        processed += 1

    _save_cache_if_needed()
    if progress_callback is not None and not progress_callback(processed, total, None):
        return profiles, no_data_decks, invalid_param_decks, failed_decks, True
    return profiles, no_data_decks, invalid_param_decks, failed_decks, False


def _params_to_str(weights: tuple[float, ...]) -> str:
    return ", ".join(f"{w:.4f}" for w in weights)


def _show_pairwise_distances(
    profiles: list[FSRSProfile],
    *,
    title: str,
    empty_message: str,
) -> None:
    ordered_profiles, distances = pairwise_distance_matrix(profiles)
    if not ordered_profiles:
        showInfo(empty_message)
        return

    dialog = QDialog(mw)
    dialog.setWindowTitle(f"{title} - All Distances")
    layout = QVBoxLayout(dialog)

    table = QTableWidget(len(ordered_profiles), len(ordered_profiles), dialog)
    labels = [profile.profile_name for profile in ordered_profiles]
    table.setHorizontalHeaderLabels(labels)

    for row_idx, row_profile in enumerate(ordered_profiles):
        table.setVerticalHeaderItem(row_idx, QTableWidgetItem(row_profile.profile_name))
        for col_idx in range(len(ordered_profiles)):
            col_profile = ordered_profiles[col_idx]
            if not is_fsrs6_valid_params(row_profile.weights) or not is_fsrs6_valid_params(
                col_profile.weights
            ):
                display = "Not FSRS6 valid params"
                table.setItem(row_idx, col_idx, QTableWidgetItem(display))
                continue

            value = distances[row_idx][col_idx]
            display = "-" if value is None else f"{value:.4f}"
            table.setItem(row_idx, col_idx, QTableWidgetItem(display))

    table.resizeColumnsToContents()
    layout.addWidget(table)

    dialog.resize(1200, 700)
    dialog.exec()


def _show_results_for_profiles(
    profiles: list[FSRSProfile],
    *,
    title: str,
    item_label: str,
    similar_profiles_by_id: Mapping[int, str] | None = None,
    similarity_groups: Sequence[Sequence[str]] | None = None,
    similarity_labels: Sequence[str] | None = None,
    similarity_deck_ids: Sequence[int] | None = None,
    similarity_distances: Sequence[Sequence[float | None]] | None = None,
    preset_groups: Sequence[tuple[int, str, Sequence[int]]] | None = None,
) -> None:
    if not profiles:
        showWarning(f"No FSRS parameters were found on existing {item_label.lower()}s.")
        return

    results = analyze_profiles(profiles)
    if not results:
        showInfo("Nothing to display.")
        return

    dialog = QDialog(mw)
    dialog.setWindowTitle(title)
    layout = QVBoxLayout(dialog)

    table = QTableWidget(len(results), 5, dialog)
    similar_column_label = f"Similar {item_label}s" if similar_profiles_by_id is not None else f"Nearest {item_label}"
    table.setHorizontalHeaderLabels(
        [
            item_label,
            "FSRS Params",
            similar_column_label,
            "Distance",
            "Share Preset?",
        ]
    )

    for row, result in enumerate(results):
        if result.status_message:
            similar_or_nearest = "-"
            nearest_distance = result.status_message
            share_preset = "-"
        else:
            if similar_profiles_by_id is None:
                similar_or_nearest = result.nearest_profile_name or "-"
            else:
                similar_or_nearest = similar_profiles_by_id.get(result.profile.profile_id, "-")
            nearest_distance = (
                "-" if result.nearest_distance is None else f"{result.nearest_distance:.4f}"
            )
            if result.should_share_preset is None:
                share_preset = "-"
            else:
                share_preset = "Yes" if result.should_share_preset else "No"

        table.setItem(row, 0, QTableWidgetItem(result.profile.profile_name))
        table.setItem(row, 1, QTableWidgetItem(_params_to_str(result.profile.weights)))
        table.setItem(row, 2, QTableWidgetItem(similar_or_nearest))
        table.setItem(row, 3, QTableWidgetItem(nearest_distance))
        table.setItem(row, 4, QTableWidgetItem(share_preset))

    table.resizeColumnsToContents()
    layout.addWidget(table)

    matrix_button = QPushButton("Show All Distances", dialog)
    matrix_button.clicked.connect(
        lambda: _show_pairwise_distances(
            profiles,
            title=title,
            empty_message=f"No {item_label.lower()}s with FSRS parameters are available.",
        )
    )
    layout.addWidget(matrix_button)

    if (
        similarity_groups is not None
        and similarity_labels is not None
        and similarity_deck_ids is not None
        and similarity_distances is not None
    ):
        groups_button = QPushButton("Show Similarity Groups", dialog)
        groups_button.clicked.connect(
            lambda: _show_similarity_groups(
                title=title,
                item_label=item_label,
                groups=similarity_groups,
                labels=similarity_labels,
                deck_ids=similarity_deck_ids,
                distances=similarity_distances,
                preset_groups=preset_groups,
            )
        )
        layout.addWidget(groups_button)

    dialog.resize(1200, 500)
    dialog.exec()


def _show_similarity_groups(
    *,
    title: str,
    item_label: str,
    groups: Sequence[Sequence[str]],
    labels: Sequence[str],
    deck_ids: Sequence[int],
    distances: Sequence[Sequence[float | None]],
    preset_groups: Sequence[tuple[int, str, Sequence[int]]] | None,
) -> None:
    if not groups:
        showInfo(
            f'No similarity groups found ({item_label} distance < '
            f"{FSRS6_RECENCY_MAHALANOBIS_SHARED_PRESET_THRESHOLD:.1f})."
        )
        return

    dialog = QDialog(mw)
    dialog.setWindowTitle(f"{title} - Similarity Groups")
    layout = QVBoxLayout(dialog)
    dialog_closed = False

    def _mark_dialog_closed(*_args: Any) -> None:
        nonlocal dialog_closed
        dialog_closed = True

    dialog.finished.connect(_mark_dialog_closed)

    summary = QLabel(
        f"{len(groups)} groups found (pairwise {item_label.lower()} distance < "
        f"{FSRS6_RECENCY_MAHALANOBIS_SHARED_PRESET_THRESHOLD:.1f})",
        dialog,
    )
    layout.addWidget(summary)

    name_to_index = {name: idx for idx, name in enumerate(labels)}
    group_indexes: list[list[int]] = [
        [name_to_index[name] for name in group if name in name_to_index]
        for group in groups
    ]
    initial_group_indexes = [list(group) for group in group_indexes]
    item_data_role = getattr(Qt, "ItemDataRole", None)
    user_role = item_data_role.UserRole if item_data_role is not None else Qt.UserRole
    drop_action = getattr(Qt, "DropAction", None)
    move_action = drop_action.MoveAction if drop_action is not None else Qt.MoveAction
    drag_drop_mode_enum = getattr(QAbstractItemView, "DragDropMode", None)
    drag_drop_mode = (
        drag_drop_mode_enum.DragDrop
        if drag_drop_mode_enum is not None
        else QAbstractItemView.DragDrop
    )
    selection_mode_enum = getattr(QAbstractItemView, "SelectionMode", None)
    selection_mode = (
        selection_mode_enum.SingleSelection
        if selection_mode_enum is not None
        else QAbstractItemView.SingleSelection
    )
    green = "#1a7f37"
    red = "#b42318"
    neutral = "#667085"

    def _metric_text_and_color(value: float | None) -> tuple[str, str]:
        if value is None:
            return "n/a", neutral
        if value < FSRS6_RECENCY_MAHALANOBIS_SHARED_PRESET_THRESHOLD:
            return f"{value:.4f}", green
        if value > FSRS6_RECENCY_MAHALANOBIS_SHARED_PRESET_THRESHOLD:
            return f"{value:.4f}", red
        return f"{value:.4f}", neutral

    def _group_indexes_from_list(list_widget: QListWidget) -> list[int]:
        if dialog_closed:
            return []
        result: list[int] = []
        try:
            row_count = list_widget.count()
        except RuntimeError:
            return result
        for row in range(row_count):
            try:
                item = list_widget.item(row)
            except RuntimeError:
                return result
            if item is None:
                continue
            try:
                raw_idx = item.data(user_role)
            except RuntimeError:
                continue
            idx = _as_int(raw_idx)
            if idx is not None:
                result.append(idx)
        return result

    def _group_max_distance(indexes: Sequence[int]) -> float | None:
        return max_pairwise_distance_for_group(group_indexes=indexes, distances=distances)

    def _deck_max_distance(deck_idx: int, group_indexes: Sequence[int]) -> float | None:
        return max_distance_to_group_for_item(
            item_index=deck_idx,
            group_indexes=group_indexes,
            distances=distances,
        )

    split_layout = QHBoxLayout()

    left_scroll = QScrollArea(dialog)
    left_scroll.setWidgetResizable(True)
    left_container = QWidget(left_scroll)
    left_container_layout = QVBoxLayout(left_container)
    left_header = QLabel("Existing Preset Groups (drag decks between presets)", left_container)
    left_container_layout.addWidget(left_header)
    preset_widgets: list[tuple[int, str, QGroupBox, QLabel, QListWidget]] = []
    for preset_id, preset_name, member_indexes in preset_groups or []:
        box = QGroupBox(preset_name, left_container)
        box_layout = QVBoxLayout(box)
        stats_label = QLabel(box)
        box_layout.addWidget(stats_label)
        list_widget = _PaneScopedListWidget("left", box)
        list_widget.setSelectionMode(selection_mode)
        list_widget.setDragEnabled(True)
        list_widget.setAcceptDrops(True)
        list_widget.setDropIndicatorShown(True)
        list_widget.setDragDropMode(drag_drop_mode)
        list_widget.setDefaultDropAction(move_action)
        for deck_idx in member_indexes:
            item = QListWidgetItem(list_widget)
            item.setData(user_role, int(deck_idx))
        box_layout.addWidget(list_widget)
        left_container_layout.addWidget(box)
        preset_widgets.append((preset_id, preset_name, box, stats_label, list_widget))
    left_scroll.setWidget(left_container)
    split_layout.addWidget(left_scroll)

    right_scroll = QScrollArea(dialog)
    right_scroll.setWidgetResizable(True)
    container = QWidget(right_scroll)
    container_layout = QVBoxLayout(container)
    right_header = QLabel("Similarity Groups (drag decks between groups)", container)
    container_layout.addWidget(right_header)
    group_widgets: list[tuple[int, QGroupBox, QLabel, QListWidget]] = []

    for idx, group in enumerate(group_indexes):
        group_box = QGroupBox(f"Group {idx + 1}", container)
        group_layout = QVBoxLayout(group_box)
        stats_label = QLabel(group_box)
        group_layout.addWidget(stats_label)
        list_widget = _PaneScopedListWidget("right", group_box)
        list_widget.setSelectionMode(selection_mode)
        list_widget.setDragEnabled(True)
        list_widget.setAcceptDrops(True)
        list_widget.setDropIndicatorShown(True)
        list_widget.setDragDropMode(drag_drop_mode)
        list_widget.setDefaultDropAction(move_action)
        for deck_idx in group:
            item = QListWidgetItem(list_widget)
            item.setData(user_role, deck_idx)
        group_layout.addWidget(list_widget)
        container_layout.addWidget(group_box)
        group_widgets.append((idx + 1, group_box, stats_label, list_widget))

    def _populate_list_widget(list_widget: QListWidget, indexes: Sequence[int]) -> None:
        if dialog_closed:
            return
        try:
            list_widget.clear()
        except RuntimeError:
            return
        for deck_idx in indexes:
            try:
                item = QListWidgetItem(list_widget)
                item.setData(user_role, int(deck_idx))
            except RuntimeError:
                return

    def _populate_group_widgets(index_groups: Sequence[Sequence[int]]) -> None:
        for group_idx, (_, _, _, list_widget) in enumerate(group_widgets):
            indexes = index_groups[group_idx] if group_idx < len(index_groups) else ()
            _populate_list_widget(list_widget, indexes)

    def _refresh_similarity_views() -> None:
        if dialog_closed:
            return
        for group_number, group_box, stats_label, list_widget in group_widgets:
            current_indexes = _group_indexes_from_list(list_widget)
            group_max = _group_max_distance(current_indexes)
            max_text = "n/a" if group_max is None else f"{group_max:.4f}"
            try:
                group_box.setTitle(f"Group {group_number} (Max: {max_text})")
                stats_label.setText(f"{len(current_indexes)} {item_label.lower()}s")
                stats_label.setStyleSheet("")
            except RuntimeError:
                continue

            try:
                row_count = list_widget.count()
            except RuntimeError:
                continue
            for row in range(row_count):
                try:
                    item = list_widget.item(row)
                except RuntimeError:
                    break
                if item is None:
                    continue
                try:
                    raw_idx = item.data(user_role)
                except RuntimeError:
                    continue
                deck_idx = _as_int(raw_idx)
                if deck_idx is None or deck_idx < 0 or deck_idx >= len(labels):
                    try:
                        item.setText(str(item.text()))
                        item.setForeground(QColor(neutral))
                    except RuntimeError:
                        continue
                    continue

                deck_max = _deck_max_distance(deck_idx, current_indexes)
                max_value_text, max_value_color = _metric_text_and_color(deck_max)
                try:
                    item.setText(f"{labels[deck_idx]} (max: {max_value_text})")
                    item.setForeground(QColor(max_value_color))
                except RuntimeError:
                    continue

    def _refresh_preset_views() -> None:
        if dialog_closed:
            return
        for _, preset_name, group_box, stats_label, list_widget in preset_widgets:
            current_indexes = _group_indexes_from_list(list_widget)
            group_max = _group_max_distance(current_indexes)
            max_text = "n/a" if group_max is None else f"{group_max:.4f}"
            try:
                group_box.setTitle(f"{preset_name} (Max: {max_text})")
                stats_label.setText(f"{len(current_indexes)} {item_label.lower()}s")
                stats_label.setStyleSheet("")
            except RuntimeError:
                continue

            singleton = len(current_indexes) == 1
            try:
                row_count = list_widget.count()
            except RuntimeError:
                continue
            for row in range(row_count):
                try:
                    item = list_widget.item(row)
                except RuntimeError:
                    break
                if item is None:
                    continue
                try:
                    raw_idx = item.data(user_role)
                except RuntimeError:
                    continue
                deck_idx = _as_int(raw_idx)
                if deck_idx is None or deck_idx < 0 or deck_idx >= len(labels):
                    try:
                        item.setText(str(item.text()))
                        item.setForeground(QColor(neutral))
                    except RuntimeError:
                        continue
                    continue

                deck_max = _deck_max_distance(deck_idx, current_indexes)
                if singleton and deck_max is None:
                    max_value_text, max_value_color = ("n/a", green)
                else:
                    max_value_text, max_value_color = _metric_text_and_color(deck_max)
                try:
                    item.setText(f"{labels[deck_idx]} (max: {max_value_text})")
                    item.setForeground(QColor(max_value_color))
                except RuntimeError:
                    continue

    for _, _, _, list_widget in group_widgets:
        model = list_widget.model()
        model.rowsInserted.connect(lambda *_args: _refresh_similarity_views())
        model.rowsRemoved.connect(lambda *_args: _refresh_similarity_views())
        model.modelReset.connect(lambda *_args: _refresh_similarity_views())

    for _, _, _, _, list_widget in preset_widgets:
        model = list_widget.model()
        model.rowsInserted.connect(lambda *_args: _refresh_preset_views())
        model.rowsRemoved.connect(lambda *_args: _refresh_preset_views())
        model.modelReset.connect(lambda *_args: _refresh_preset_views())

    all_list_widgets = [row[3] for row in group_widgets] + [row[4] for row in preset_widgets]
    selection_syncing = False

    def _find_item_by_deck_index(list_widget: QListWidget, deck_idx: int) -> QListWidgetItem | None:
        try:
            row_count = list_widget.count()
        except RuntimeError:
            return None
        for row in range(row_count):
            try:
                item = list_widget.item(row)
            except RuntimeError:
                return None
            if item is None:
                continue
            try:
                raw_idx = item.data(user_role)
            except RuntimeError:
                continue
            if _as_int(raw_idx) == deck_idx:
                return item
        return None

    def _sync_selection(deck_idx: int) -> None:
        nonlocal selection_syncing
        if dialog_closed or selection_syncing:
            return
        selection_syncing = True
        try:
            for list_widget in all_list_widgets:
                try:
                    list_widget.blockSignals(True)
                except RuntimeError:
                    continue
            for list_widget in all_list_widgets:
                try:
                    list_widget.clearSelection()
                except RuntimeError:
                    continue
                item = _find_item_by_deck_index(list_widget, deck_idx)
                if item is None:
                    continue
                try:
                    item.setSelected(True)
                    list_widget.setCurrentItem(item)
                except RuntimeError:
                    continue
        finally:
            for list_widget in all_list_widgets:
                try:
                    list_widget.blockSignals(False)
                except RuntimeError:
                    continue
            selection_syncing = False

    def _sync_selection_from_item(item: QListWidgetItem | None) -> None:
        if item is None or dialog_closed:
            return
        try:
            raw_idx = item.data(user_role)
        except RuntimeError:
            return
        deck_idx = _as_int(raw_idx)
        if deck_idx is None:
            return
        _sync_selection(deck_idx)

    for list_widget in all_list_widgets:
        list_widget.itemClicked.connect(_sync_selection_from_item)

    right_scroll.setWidget(container)
    split_layout.addWidget(right_scroll)
    layout.addLayout(split_layout)

    buttons_layout = QHBoxLayout()
    reset_button = QPushButton("Reset Groups", dialog)
    reset_button.clicked.connect(
        lambda: (_populate_group_widgets(initial_group_indexes), _refresh_similarity_views())
    )
    buttons_layout.addWidget(reset_button)

    store_backup_button = QPushButton("Store Preset Backup", dialog)

    def _store_preset_backup_for_visible_decks() -> None:
        assignments = _current_preset_assignments(deck_ids)
        if not assignments:
            showWarning("No preset assignments found to store.")
            return
        _save_preset_backup(assignments)
        showInfo(f"Stored preset backup for {len(assignments)} decks.")

    store_backup_button.clicked.connect(_store_preset_backup_for_visible_decks)
    buttons_layout.addWidget(store_backup_button)

    revert_backup_button = QPushButton("Revert Preset Backup", dialog)

    def _revert_preset_backup() -> None:
        backup = _load_preset_backup()
        if not backup:
            showWarning("No stored preset backup found.")
            return
        changed, failed = _apply_preset_assignments(backup)
        if failed:
            showWarning(f"Reverted {changed} decks from backup, failed for {failed}.")
        else:
            showInfo(f"Reverted {changed} decks from backup.")
        dialog.accept()

    revert_backup_button.clicked.connect(_revert_preset_backup)
    buttons_layout.addWidget(revert_backup_button)

    recommended_button = QPushButton("Use Recommended Group", dialog)

    def _use_recommended_groups() -> None:
        create_config_id = getattr(mw.col.decks, "add_config_returning_id", None)
        create_config = getattr(mw.col.decks, "add_config", None)

        existing_names = [name for _, name, _ in _all_preset_configs()]
        created = 0
        failed = 0
        target_by_deck: dict[int, int] = {}

        for group_number, indexes in enumerate(initial_group_indexes, start=1):
            valid_indexes = [idx for idx in indexes if 0 <= idx < len(deck_ids)]
            if len(valid_indexes) < 2:
                continue

            base_name = recommended_group_preset_name(group_number)
            preset_name = unique_name(base_name, existing_names)

            clone_from = None
            for idx in valid_indexes:
                deck_id = deck_ids[idx]
                deck = mw.col.decks.get(deck_id)
                conf_id = _as_int(_field_any(deck, ("conf", "config_id")))
                if conf_id is None:
                    continue
                conf = _config_from_conf_id(conf_id)
                if conf is not None:
                    clone_from = conf
                    break

            try:
                if create_config_id is not None:
                    new_conf_id = int(create_config_id(preset_name, clone_from=clone_from))
                elif create_config is not None:
                    new_conf = create_config(preset_name, clone_from=clone_from)
                    new_conf_id = _as_int(_field(new_conf, "id"))
                    if new_conf_id is None:
                        raise RuntimeError("Created preset has no id")
                else:
                    raise RuntimeError("Preset creation API unavailable")
            except Exception:
                failed += len(valid_indexes)
                continue

            existing_names.append(preset_name)
            created += 1

            for idx in valid_indexes:
                target_by_deck[deck_ids[idx]] = int(new_conf_id)

        if target_by_deck:
            _save_preset_backup(_current_preset_assignments(list(target_by_deck.keys())))
            assigned, apply_failed = _apply_preset_assignments(target_by_deck)
            failed += apply_failed
        else:
            assigned = 0

        if created == 0 and assigned == 0 and failed == 0:
            showInfo("No recommended groups with at least 2 decks were found.")
            return
        if failed:
            showWarning(
                f"Created {created} preset groups and reassigned {assigned} decks, "
                f"with {failed} failures."
            )
        else:
            showInfo(
                f"Created {created} preset groups and reassigned {assigned} decks.\n\n"
                "Reopen this window to refresh existing preset groups."
            )
        dialog.accept()

    recommended_button.clicked.connect(_use_recommended_groups)
    buttons_layout.addWidget(recommended_button)

    save_button = QPushButton("Save Preset Changes", dialog)
    save_button.setEnabled(bool(preset_widgets))

    def _save_preset_changes() -> None:
        target_by_deck: dict[int, int] = {}
        for preset_id, _, _, _, list_widget in preset_widgets:
            for deck_idx in _group_indexes_from_list(list_widget):
                if 0 <= deck_idx < len(deck_ids):
                    target_by_deck[deck_ids[deck_idx]] = preset_id

        _save_preset_backup(_current_preset_assignments(list(target_by_deck.keys())))
        changed, failed = _apply_preset_assignments(target_by_deck)

        if failed:
            showWarning(f"Preset changes applied to {changed} decks, failed for {failed}.")
        elif changed:
            showInfo(f"Preset changes applied to {changed} decks.")
        else:
            showInfo("No preset changes to save.")

    save_button.clicked.connect(_save_preset_changes)
    buttons_layout.addWidget(save_button)
    layout.addLayout(buttons_layout)

    _refresh_similarity_views()
    _refresh_preset_views()

    dialog.resize(1000, 600)
    dialog.exec()


def _show_preset_results() -> None:
    profiles = _load_profiles()
    _show_results_for_profiles(profiles, title=_ACTION_LABEL, item_label="Preset")


def _show_deck_computed_results() -> None:
    choice = QMessageBox.question(
        mw,
        _DECK_ACTION_LABEL,
        "Default scope is leaf decks only.\n\n"
        "Do you also want to include middle/root decks for the computation?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    include_middle_and_root = choice == QMessageBox.StandardButton.Yes
    progress = QProgressDialog("Preparing deck optimizations...", "Cancel", 0, 100, mw)
    progress.setWindowTitle(_DECK_ACTION_LABEL)
    progress.setAutoClose(False)
    progress.setAutoReset(False)
    progress.setMinimumDuration(0)
    progress.setValue(0)

    def _progress_callback(done: int, total: int, deck_name: str | None) -> bool:
        progress.setMaximum(max(total, 0))
        progress.setValue(min(done, total))
        progress.setLabelText(
            optimization_progress_message(done=done, total=total, deck_name=deck_name)
        )
        progress.show()
        progress.repaint()
        mw.app.processEvents()
        progress.repaint()
        return not progress.wasCanceled()

    try:
        profiles, no_data_decks, invalid_param_decks, failed_decks, cancelled = (
            _load_computed_deck_profiles(
                include_middle_and_root=include_middle_and_root,
                progress_callback=_progress_callback,
            )
        )
    finally:
        progress.close()

    if cancelled and not profiles:
        showWarning("Deck optimization cancelled.")
        return
    if cancelled:
        showInfo("Deck optimization cancelled. Showing partial results.")

    if not profiles:
        showWarning("No deck produced usable FSRS6 computed parameters.")
        return

    notes: list[str] = []
    if no_data_decks:
        notes.append(f"Skipped (no FSRS items): {len(no_data_decks)}")
    if invalid_param_decks:
        notes.append(f"Skipped (non-FSRS6 params): {len(invalid_param_decks)}")
    if failed_decks:
        notes.append(f"Skipped (compute failure): {len(failed_decks)}")
    if notes:
        showInfo("Deck computation notes:\n" + "\n".join(notes))

    ordered, matrix = pairwise_distance_matrix(profiles)
    labels = [profile.profile_name for profile in ordered]
    ordered_deck_ids = [profile.profile_id for profile in ordered]
    preset_groups_map: dict[int, dict[str, Any]] = {}
    for idx, profile in enumerate(ordered):
        config = _config_for_deck(profile.profile_id)
        conf_id = _as_int(_field(config, "id"))
        if conf_id is None:
            deck = mw.col.decks.get(profile.profile_id)
            conf_id = _as_int(_field_any(deck, ("conf", "config_id")))
        if conf_id is None:
            continue

        preset_name = _config_name(conf_id, config) if config is not None else f"Preset {conf_id}"
        group = preset_groups_map.get(conf_id)
        if group is None:
            preset_groups_map[conf_id] = {"name": preset_name, "indexes": [idx]}
        else:
            group["indexes"].append(idx)

    preset_groups = sorted(
        [
            (
                conf_id,
                str(data["name"]),
                [int(v) for v in data["indexes"]],
            )
            for conf_id, data in preset_groups_map.items()
        ],
        key=lambda item: item[1].lower(),
    )
    similar_by_id: dict[int, str] = {}
    for idx, profile in enumerate(ordered):
        similar_items = similar_items_below_threshold(
            names=labels,
            distances_row=matrix[idx],
            self_index=idx,
            threshold=FSRS6_RECENCY_MAHALANOBIS_SHARED_PRESET_THRESHOLD,
        )
        similar_by_id[profile.profile_id] = ", ".join(similar_items) if similar_items else "-"
    similarity_groups = similarity_groups_from_matrix(
        names=labels,
        distances=matrix,
        threshold=FSRS6_RECENCY_MAHALANOBIS_SHARED_PRESET_THRESHOLD,
    )

    _show_results_for_profiles(
        profiles,
        title=_DECK_ACTION_LABEL,
        item_label="Deck",
        similar_profiles_by_id=similar_by_id,
        similarity_groups=similarity_groups,
        similarity_labels=labels,
        similarity_deck_ids=ordered_deck_ids,
        similarity_distances=matrix,
        preset_groups=preset_groups,
    )


def init_addon() -> None:
    preset_action = QAction(_ACTION_LABEL, mw)
    preset_action.triggered.connect(_show_preset_results)
    mw.form.menuTools.addAction(preset_action)

    deck_action = QAction(_DECK_ACTION_LABEL, mw)
    deck_action.triggered.connect(_show_deck_computed_results)
    mw.form.menuTools.addAction(deck_action)
