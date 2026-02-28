from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from aqt import mw
from aqt.qt import QAction, QDialog, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout
from aqt.utils import showInfo, showWarning

from .analyzer import FSRSProfile, analyze_profiles, extract_fsrs_weights, pairwise_distance_matrix

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


def _config_from_conf_id(conf_id: int) -> Mapping[str, Any] | None:
    getter_names = ["get_config", "get_config_dict", "dconf_for_update_dict", "getconf"]
    for getter_name in getter_names:
        getter = getattr(mw.col.decks, getter_name, None)
        if getter is None:
            continue
        try:
            value = getter(conf_id)
        except Exception:
            continue
        if isinstance(value, Mapping):
            return value
    return None


def _config_for_deck(deck_id: int) -> Mapping[str, Any] | None:
    by_deck = getattr(mw.col.decks, "config_dict_for_deck_id", None)
    if by_deck is not None:
        try:
            value = by_deck(deck_id)
            if isinstance(value, Mapping):
                return value
        except Exception:
            pass

    deck = mw.col.decks.get(deck_id)
    if not isinstance(deck, Mapping):
        return None

    conf_id = deck.get("conf")
    if isinstance(conf_id, int):
        return _config_from_conf_id(conf_id)

    return None


def _normalize_config(config: Any) -> tuple[int, str, Mapping[str, Any]] | None:
    if not isinstance(config, Mapping):
        return None
    conf_id = config.get("id")
    if not isinstance(conf_id, int):
        return None
    name = config.get("name")
    conf_name = str(name) if isinstance(name, str) and name else f"Preset {conf_id}"
    return conf_id, conf_name, config


def _all_preset_configs() -> list[tuple[int, str, Mapping[str, Any]]]:
    seen: dict[int, tuple[str, Mapping[str, Any]]] = {}

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
            configs = raw.values()
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            configs = raw
        else:
            continue

        for config in configs:
            normalized = _normalize_config(config)
            if normalized is None:
                continue
            conf_id, conf_name, conf_dict = normalized
            seen[conf_id] = (conf_name, conf_dict)

    if seen:
        return sorted((conf_id, item[0], item[1]) for conf_id, item in seen.items())

    # Fallback for older API shapes: gather presets referenced by decks.
    for deck_id, _ in _deck_entries():
        deck = mw.col.decks.get(deck_id)
        if not isinstance(deck, Mapping):
            continue
        conf_id = deck.get("conf")
        if not isinstance(conf_id, int):
            continue
        if conf_id in seen:
            continue

        config = _config_from_conf_id(conf_id)
        if not isinstance(config, Mapping):
            continue
        conf_name = config.get("name")
        name = str(conf_name) if isinstance(conf_name, str) and conf_name else f"Preset {conf_id}"
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
        nearest_name = result.nearest_profile_name or "-"
        nearest_distance = "-" if result.nearest_distance is None else f"{result.nearest_distance:.4f}"
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
