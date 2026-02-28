from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from aqt import mw
from aqt.qt import QAction, QDialog, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout
from aqt.utils import showInfo, showWarning

from .analyzer import (
    FSRSProfile,
    analyze_profiles,
    extract_fsrs_weights,
    is_fsrs6_valid_params,
    pairwise_distance_matrix,
)

_ACTION_LABEL = "FSRS Preset Proximity"


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


def _params_to_str(weights: tuple[float, ...]) -> str:
    return ", ".join(f"{w:.4f}" for w in weights)


def _show_pairwise_distances(profiles: list[FSRSProfile]) -> None:
    ordered_profiles, distances = pairwise_distance_matrix(profiles)
    if not ordered_profiles:
        showInfo("No presets with FSRS parameters are available.")
        return

    dialog = QDialog(mw)
    dialog.setWindowTitle(f"{_ACTION_LABEL} - All Distances")
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


def _show_results() -> None:
    profiles = _load_profiles()
    if not profiles:
        showWarning("No FSRS parameters were found on existing presets.")
        return

    results = analyze_profiles(profiles)
    if not results:
        showInfo("Nothing to display.")
        return

    dialog = QDialog(mw)
    dialog.setWindowTitle(_ACTION_LABEL)
    layout = QVBoxLayout(dialog)

    table = QTableWidget(len(results), 5, dialog)
    table.setHorizontalHeaderLabels(
        ["Preset", "FSRS Params", "Nearest Preset", "Distance", "Share Preset?"]
    )

    for row, result in enumerate(results):
        if result.status_message:
            nearest_name = "-"
            nearest_distance = result.status_message
            share_preset = "-"
        else:
            nearest_name = result.nearest_profile_name or "-"
            nearest_distance = (
                "-" if result.nearest_distance is None else f"{result.nearest_distance:.4f}"
            )
            if result.should_share_preset is None:
                share_preset = "-"
            else:
                share_preset = "Yes" if result.should_share_preset else "No"

        table.setItem(row, 0, QTableWidgetItem(result.profile.profile_name))
        table.setItem(row, 1, QTableWidgetItem(_params_to_str(result.profile.weights)))
        table.setItem(row, 2, QTableWidgetItem(nearest_name))
        table.setItem(row, 3, QTableWidgetItem(nearest_distance))
        table.setItem(row, 4, QTableWidgetItem(share_preset))

    table.resizeColumnsToContents()
    layout.addWidget(table)

    matrix_button = QPushButton("Show All Distances", dialog)
    matrix_button.clicked.connect(lambda: _show_pairwise_distances(profiles))
    layout.addWidget(matrix_button)

    dialog.resize(1200, 500)
    dialog.exec()


def init_addon() -> None:
    action = QAction(_ACTION_LABEL, mw)
    action.triggered.connect(_show_results)
    mw.form.menuTools.addAction(action)
