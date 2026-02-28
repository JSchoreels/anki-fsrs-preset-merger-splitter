from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

from aqt import mw
from aqt.qt import (
    QAction,
    QDialog,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
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
    count_relearning_steps_in_day,
    leaf_deck_entries,
    optimization_progress_message,
    similar_items_below_threshold,
)
from .reference_covariance import FSRS6_RECENCY_MAHALANOBIS_SHARED_PRESET_THRESHOLD

_ACTION_LABEL = "FSRS Preset Proximity"
_DECK_ACTION_LABEL = "FSRS Deck Proximity (Computed)"


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

    profiles: list[FSRSProfile] = []
    no_data_decks: list[str] = []
    invalid_param_decks: list[str] = []
    failed_decks: list[str] = []
    total = len(selected_entries)
    processed = 0

    if progress_callback is not None and not progress_callback(0, total, None):
        return profiles, no_data_decks, invalid_param_decks, failed_decks, True

    for deck_id, deck_name in selected_entries:
        if progress_callback is not None and not progress_callback(processed, total, deck_name):
            return profiles, no_data_decks, invalid_param_decks, failed_decks, True

        config = _config_for_deck(deck_id)
        current_params = extract_fsrs_weights(config) or ()
        num_of_relearning_steps = _num_relearning_steps_in_day_for_deck(deck_id)
        search = build_deck_search_query(
            deck_id=deck_id,
            deck_name=deck_name,
            include_children=include_middle_and_root,
        )

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

    dialog.resize(1200, 500)
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
    similar_by_id: dict[int, str] = {}
    for idx, profile in enumerate(ordered):
        similar_items = similar_items_below_threshold(
            names=labels,
            distances_row=matrix[idx],
            self_index=idx,
            threshold=FSRS6_RECENCY_MAHALANOBIS_SHARED_PRESET_THRESHOLD,
        )
        similar_by_id[profile.profile_id] = ", ".join(similar_items) if similar_items else "-"

    _show_results_for_profiles(
        profiles,
        title=_DECK_ACTION_LABEL,
        item_label="Deck",
        similar_profiles_by_id=similar_by_id,
    )


def init_addon() -> None:
    preset_action = QAction(_ACTION_LABEL, mw)
    preset_action.triggered.connect(_show_preset_results)
    mw.form.menuTools.addAction(preset_action)

    deck_action = QAction(_DECK_ACTION_LABEL, mw)
    deck_action.triggered.connect(_show_deck_computed_results)
    mw.form.menuTools.addAction(deck_action)
