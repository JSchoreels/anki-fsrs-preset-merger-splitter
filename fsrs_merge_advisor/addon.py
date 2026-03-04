from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import json
from pathlib import Path
from typing import Any, Callable

from aqt import mw
from aqt.qt import (
    QAbstractItemView,
    QAction,
    QColor,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSlider,
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
from .infra.decks_gateway import (
    all_preset_configs as _all_preset_configs,
    apply_preset_assignments as _apply_preset_assignments,
    as_int as _as_int,
    config_for_deck as _config_for_deck,
    config_from_conf_id as _config_from_conf_id,
    config_name as _config_name,
    current_preset_assignments as _current_preset_assignments,
    deck_entries as _deck_entries,
    field as _field,
    field_any as _field_any,
)
from .tools.cache import can_reuse_cached_params
from .tools.deck_scope import descendant_deck_ids, leaf_deck_entries
from .tools.fsrs_payload import set_fsrs_params_on_config_payload
from .tools.grouping import (
    max_distance_to_group_for_item,
    max_pairwise_distance_for_group,
    recommended_group_preset_name,
    similar_items_below_threshold,
    similarity_groups_from_matrix,
)
from .tools.progress_messages import (
    optimization_progress_message,
    preset_optimization_progress_message,
)
from .tools.relearning import count_relearning_steps_in_day
from .tools.search_queries import build_deck_search_query, build_multi_deck_search_query
from .use_cases.preset_reoptimization import (
    changed_preset_deck_groups_for_reoptimization,
    preset_optimization_summary_message,
)
from .use_cases.first_review_split import (
    FIRST_REVIEW_RATINGS,
    is_first_review_split_deck_name,
    normalize_card_ids_from_single_column_rows,
    split_card_ids_by_first_review,
    target_deck_names_for_first_review_split,
    target_preset_names_for_first_review_split,
)
from .use_cases.preset_cleanup import empty_advisor_preset_candidates
from .use_cases.evaluate_mergeability import (
    align_group_indexes_by_overlap,
    can_reuse_evaluate_cached_logloss,
    cluster_logloss_groups_agglomerative,
    evaluate_logloss_cache_key,
    mergeable_pair_count,
    symmetric_merge_score_matrix,
)
from .reference_covariance import FSRS6_RECENCY_MAHALANOBIS_SHARED_PRESET_THRESHOLD

_ACTION_LABEL = "FSRS Preset Proximity"
_DECK_ACTION_LABEL = "FSRS Deck Proximity (Computed)"
_EVALUATE_MERGEABILITY_LABEL = "FSRS Mergeability (Evaluate)"
_FIRST_REVIEW_SPLIT_LABEL = "FSRS Split Deck by First Review"
_FIRST_REVIEW_UNSPLIT_LABEL = "FSRS Merge Back First Review Split"
_CLEAN_EMPTY_PRESETS_LABEL = "FSRS Cleanup Empty Advisor Presets"
_CACHE_FILE_NAME = "deck_params_cache.json"
_PRESET_BACKUP_FILE_NAME = "deck_preset_backup.json"
_EVALUATE_LOGLOSS_CACHE_FILE_NAME = "evaluate_logloss_cache.json"


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


def _legacy_evaluate_logloss_cache_file_path() -> Path:
    return Path(__file__).resolve().parent / _EVALUATE_LOGLOSS_CACHE_FILE_NAME


def _preferred_evaluate_logloss_cache_file_path() -> Path:
    pm = getattr(mw, "pm", None)
    profile_folder_getter = getattr(pm, "profileFolder", None)
    if callable(profile_folder_getter):
        try:
            profile_folder = profile_folder_getter()
            if isinstance(profile_folder, str) and profile_folder:
                return Path(profile_folder) / _EVALUATE_LOGLOSS_CACHE_FILE_NAME
        except Exception:
            pass
    return _legacy_evaluate_logloss_cache_file_path()


def _evaluate_logloss_cache_file_candidates() -> list[Path]:
    preferred = _preferred_evaluate_logloss_cache_file_path()
    legacy = _legacy_evaluate_logloss_cache_file_path()
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


def _load_evaluate_logloss_cache() -> dict[str, dict[str, Any]]:
    preferred = _preferred_evaluate_logloss_cache_file_path()
    for path in _evaluate_logloss_cache_file_candidates():
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
            _save_evaluate_logloss_cache(normalized)
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


def _save_evaluate_logloss_cache(entries: Mapping[str, Mapping[str, Any]]) -> None:
    payload = {"version": 1, "entries": dict(entries)}
    path = _preferred_evaluate_logloss_cache_file_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    except Exception:
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


def _deck_id_for_name(deck_name: str) -> int | None:
    for did, name in _deck_entries():
        if name == deck_name:
            return int(did)
    return None


def _as_positive_int(value: Any) -> int | None:
    direct = _as_int(value)
    if direct is not None and direct > 0:
        return int(direct)
    try:
        converted = int(value)
    except (TypeError, ValueError):
        return None
    return converted if converted > 0 else None


def _ensure_deck_id_for_name(deck_name: str) -> int | None:
    existing = _deck_id_for_name(deck_name)
    if existing is not None:
        return existing

    deck_manager = getattr(mw.col, "decks", None)
    if deck_manager is None:
        return None

    candidates: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = [
        ("id_for_name", (deck_name,), {"create": True}),
        ("id_for_name", (deck_name, True), {}),
        ("id", (deck_name,), {"create": True}),
        ("id", (deck_name, True), {}),
        ("id", (deck_name,), {}),
        ("idForName", (deck_name,), {}),
        ("add_normal_deck_with_name", (deck_name,), {}),
    ]
    for method_name, args, kwargs in candidates:
        method = getattr(deck_manager, method_name, None)
        if method is None:
            continue
        try:
            created = method(*args, **kwargs)
        except Exception:
            continue
        deck_id = _as_positive_int(created)
        if deck_id is not None:
            return deck_id

    return _deck_id_for_name(deck_name)


def _cards_and_first_review_ease_for_decks(deck_ids: Sequence[int]) -> list[tuple[int, int | None]]:
    normalized_deck_ids = sorted({int(did) for did in deck_ids if int(did) > 0})
    if not normalized_deck_ids:
        return []

    placeholders = ",".join("?" for _ in normalized_deck_ids)
    sql = (
        "SELECT c.id, "
        "(SELECT r.ease FROM revlog r WHERE r.cid = c.id ORDER BY r.id ASC LIMIT 1) "
        "FROM cards c "
        f"WHERE c.did IN ({placeholders})"
    )
    rows = mw.col.db.all(sql, *normalized_deck_ids)

    normalized_rows: list[tuple[int, int | None]] = []
    for raw_card_id, raw_first_ease in rows:
        card_id = _as_positive_int(raw_card_id)
        if card_id is None:
            continue
        first_ease = _as_int(raw_first_ease)
        normalized_rows.append((card_id, first_ease))
    return normalized_rows


def _card_ids_for_decks(deck_ids: Sequence[int]) -> list[int]:
    normalized_deck_ids = sorted({int(did) for did in deck_ids if int(did) > 0})
    if not normalized_deck_ids:
        return []

    placeholders = ",".join("?" for _ in normalized_deck_ids)
    sql = f"SELECT id FROM cards WHERE did IN ({placeholders})"
    db_list = getattr(mw.col.db, "list", None)
    if callable(db_list):
        try:
            return normalize_card_ids_from_single_column_rows(
                db_list(sql, *normalized_deck_ids)
            )
        except Exception:
            pass
    return normalize_card_ids_from_single_column_rows(
        mw.col.db.all(sql, *normalized_deck_ids)
    )


def _deck_exists(deck_id: int) -> bool:
    try:
        return mw.col.decks.get(int(deck_id)) is not None
    except Exception:
        return False


def _delete_deck_by_id(deck_id: int) -> bool:
    if deck_id <= 0:
        return False
    deck_manager = getattr(mw.col, "decks", None)
    if deck_manager is None:
        return False

    method_calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = [
        ("remove", ([int(deck_id)],), {"cards_too": False}),
        ("remove", ([int(deck_id)],), {"cardsToo": False}),
        ("remove", ([int(deck_id)], False), {}),
        ("remove", ([int(deck_id)], False, True), {}),
        ("rem", ([int(deck_id)],), {"cardsToo": False, "childrenToo": True}),
        ("rem", ([int(deck_id)], False, True), {}),
        ("rem", ([int(deck_id)], False), {}),
    ]

    for method_name, args, kwargs in method_calls:
        method = getattr(deck_manager, method_name, None)
        if method is None:
            continue
        try:
            method(*args, **kwargs)
        except Exception:
            continue
        if not _deck_exists(int(deck_id)):
            return True

    return not _deck_exists(int(deck_id))


def _delete_preset_by_id(conf_id: int) -> bool:
    if conf_id <= 0:
        return False
    deck_manager = getattr(mw.col, "decks", None)
    if deck_manager is None:
        return False

    method_calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = [
        ("remove_config", (int(conf_id),), {}),
        ("remove_config", ([int(conf_id)],), {}),
        ("remove_config_id", (int(conf_id),), {}),
        ("rem_config", (int(conf_id),), {}),
        ("remConfig", (int(conf_id),), {}),
        ("remConf", (int(conf_id),), {}),
    ]

    def _preset_exists() -> bool:
        return any(existing_id == int(conf_id) for existing_id, _name, _cfg in _all_preset_configs())

    if not _preset_exists():
        return True

    for method_name, args, kwargs in method_calls:
        method = getattr(deck_manager, method_name, None)
        if method is None:
            continue
        try:
            method(*args, **kwargs)
        except Exception:
            continue
        if not _preset_exists():
            return True

    return not _preset_exists()


def _move_cards_to_deck(
    *,
    card_ids: Sequence[int],
    target_deck_id: int,
) -> tuple[int, int]:
    normalized = sorted({int(card_id) for card_id in card_ids if int(card_id) > 0})
    if not normalized:
        return 0, 0

    set_deck = getattr(mw.col, "set_deck", None)
    if callable(set_deck):
        call_patterns: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
            ((list(normalized), int(target_deck_id)), {}),
            ((), {"cids": list(normalized), "did": int(target_deck_id)}),
            ((), {"card_ids": list(normalized), "deck_id": int(target_deck_id)}),
        ]
        for args, kwargs in call_patterns:
            try:
                if kwargs:
                    set_deck(**kwargs)
                else:
                    set_deck(*args)
                return len(normalized), 0
            except Exception:
                continue

    get_card = getattr(mw.col, "get_card", None)
    if get_card is None:
        get_card = getattr(mw.col, "getCard", None)
    update_card = getattr(mw.col, "update_card", None)
    if update_card is None:
        update_card = getattr(mw.col, "updateCard", None)
    if callable(get_card) and callable(update_card):
        moved = 0
        failed = 0
        for card_id in normalized:
            try:
                card = get_card(card_id)
                if card is None:
                    failed += 1
                    continue
                current_did = _as_int(_field(card, "did"))
                if current_did == int(target_deck_id):
                    continue
                if isinstance(card, Mapping):
                    card["did"] = int(target_deck_id)
                else:
                    setattr(card, "did", int(target_deck_id))
                update_card(card)
                moved += 1
            except Exception:
                failed += 1
        return moved, failed

    try:
        placeholders = ",".join("?" for _ in normalized)
        mw.col.db.execute(
            f"UPDATE cards SET did = ? WHERE id IN ({placeholders})",
            int(target_deck_id),
            *normalized,
        )
        return len(normalized), 0
    except Exception:
        return 0, len(normalized)

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


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
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


def _evaluate_logloss_with_params(
    *,
    params: Sequence[float],
    search: str,
) -> float:
    backend = getattr(mw.col, "_backend", None)
    evaluate = getattr(backend, "evaluate_params_legacy", None)
    if evaluate is None:
        evaluate = getattr(backend, "evaluateParamsLegacy", None)
    if evaluate is None:
        raise RuntimeError("evaluate_params_legacy backend endpoint is unavailable")

    candidates = [
        {
            "params": list(params),
            "search": search,
            "ignore_revlogs_before_ms": 0,
        },
        {
            "params": list(params),
            "search": search,
            "ignoreRevlogsBeforeMs": 0,
        },
    ]

    response = None
    last_type_error: Exception | None = None
    for kwargs in candidates:
        try:
            response = evaluate(**kwargs)
            break
        except TypeError as exc:
            last_type_error = exc
            continue

    if response is None:
        if last_type_error is not None:
            raise RuntimeError("Unable to call evaluate_params_legacy with known argument names") from last_type_error
        raise RuntimeError("evaluate_params_legacy failed without a result")

    log_loss = _to_float(_field_any(response, ("log_loss", "logLoss")))
    if log_loss is None:
        raise RuntimeError("evaluate_params_legacy response did not include log_loss")
    return float(log_loss)


def _optimize_preset_configs(
    preset_deck_groups: Sequence[tuple[int, Sequence[int]]],
) -> tuple[int, int, int, int, bool]:
    update_config = getattr(mw.col.decks, "update_config", None)
    total = len(preset_deck_groups)
    if total == 0:
        return 0, 0, 0, 0, False

    progress = QProgressDialog("Preparing preset optimizations...", "Cancel", 0, total, mw)
    progress.setWindowTitle("Optimize Changed Presets")
    progress.setAutoClose(False)
    progress.setAutoReset(False)
    progress.setMinimumDuration(0)
    progress.setValue(0)

    optimized = 0
    no_data = 0
    invalid_params = 0
    failed = 0
    cancelled = False
    processed = 0

    def _set_progress(done: int, preset_name: str | None) -> None:
        progress.setMaximum(max(total, 0))
        progress.setValue(min(done, total))
        progress.setLabelText(
            preset_optimization_progress_message(
                done=done,
                total=total,
                preset_name=preset_name,
            )
        )
        progress.show()
        progress.repaint()
        mw.app.processEvents()
        progress.repaint()

    try:
        _set_progress(0, None)
        for conf_id, deck_ids in preset_deck_groups:
            if progress.wasCanceled():
                cancelled = True
                break

            config = _config_from_conf_id(conf_id)
            conf_name = _config_name(conf_id, config) if config is not None else f"Preset {conf_id}"
            _set_progress(processed, conf_name)

            if not callable(update_config) or config is None:
                failed += 1
                processed += 1
                _set_progress(processed, None)
                continue

            search = build_multi_deck_search_query(deck_ids)
            if search is None:
                failed += 1
                processed += 1
                _set_progress(processed, None)
                continue

            try:
                params, fsrs_items = _compute_fsrs_params_for_deck(
                    search=search,
                    current_params=extract_fsrs_weights(config) or (),
                    num_of_relearning_steps=count_relearning_steps_in_day(
                        _extract_relearning_steps(config)
                    ),
                )
            except Exception:
                failed += 1
                processed += 1
                _set_progress(processed, None)
                continue

            if fsrs_items <= 0:
                no_data += 1
                processed += 1
                _set_progress(processed, None)
                continue
            if not is_fsrs6_valid_params(params):
                invalid_params += 1
                processed += 1
                _set_progress(processed, None)
                continue

            try:
                cloned = copy.deepcopy(config)
                if not isinstance(cloned, Mapping):
                    raise RuntimeError("Invalid target preset payload")
                payload = dict(cloned)
                payload["id"] = int(conf_id)
                payload_name = _field(payload, "name")
                if not isinstance(payload_name, str) or not payload_name:
                    payload["name"] = conf_name
                payload = set_fsrs_params_on_config_payload(
                    config_payload=payload,
                    params=params,
                )
                update_config(payload)
                optimized += 1
            except Exception:
                failed += 1

            processed += 1
            _set_progress(processed, None)
    finally:
        progress.close()

    if optimized:
        try:
            mw.reset()
        except Exception:
            pass

    return optimized, no_data, invalid_params, failed, cancelled


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

    matrix_title = QLabel("Proximity Matrix", dialog)
    layout.addWidget(matrix_title)

    ordered_profiles, distances = pairwise_distance_matrix(profiles)
    matrix_table = QTableWidget(len(ordered_profiles), len(ordered_profiles), dialog)
    matrix_labels = [profile.profile_name for profile in ordered_profiles]
    matrix_table.setHorizontalHeaderLabels(matrix_labels)
    for row_idx, row_profile in enumerate(ordered_profiles):
        matrix_table.setVerticalHeaderItem(row_idx, QTableWidgetItem(row_profile.profile_name))
        for col_idx, col_profile in enumerate(ordered_profiles):
            if not is_fsrs6_valid_params(row_profile.weights) or not is_fsrs6_valid_params(
                col_profile.weights
            ):
                display = "Not FSRS6 valid params"
            else:
                value = distances[row_idx][col_idx]
                display = "-" if value is None else f"{value:.4f}"
            matrix_table.setItem(row_idx, col_idx, QTableWidgetItem(display))
    matrix_table.resizeColumnsToContents()
    layout.addWidget(matrix_table)

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

    dialog.resize(1300, 900)
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
    open_proximity_callback: Callable[[], None] | None = None,
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

    name_to_index = {name: idx for idx, name in enumerate(labels)}
    current_threshold = float(FSRS6_RECENCY_MAHALANOBIS_SHARED_PRESET_THRESHOLD)

    def _recommended_groups_from_threshold(threshold: float) -> list[list[int]]:
        grouped_names = similarity_groups_from_matrix(
            names=labels,
            distances=distances,
            threshold=threshold,
            min_group_size=1,
        )
        return [[name_to_index[name] for name in group if name in name_to_index] for group in grouped_names]

    recommended_group_indexes = _recommended_groups_from_threshold(current_threshold)

    summary = QLabel(dialog)
    layout.addWidget(summary)

    threshold_layout = QHBoxLayout()
    threshold_layout.addWidget(QLabel("Grouping Threshold:", dialog))
    orientation_enum = getattr(Qt, "Orientation", None)
    horizontal = orientation_enum.Horizontal if orientation_enum is not None else Qt.Horizontal
    threshold_slider = QSlider(horizontal, dialog)
    threshold_slider.setMinimum(1)
    threshold_slider.setMaximum(100)
    threshold_slider.setSingleStep(1)
    threshold_slider.setValue(int(round(current_threshold * 10)))
    threshold_layout.addWidget(threshold_slider)
    threshold_value_label = QLabel(dialog)
    threshold_layout.addWidget(threshold_value_label)
    layout.addLayout(threshold_layout)
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
        if value < current_threshold:
            return f"{value:.4f}", green
        if value > current_threshold:
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
    left_header = QLabel("Left Panel: Presets Editor (drag decks between presets)", left_container)
    left_container_layout.addWidget(left_header)
    preset_widgets: list[tuple[int, str, QGroupBox, QLabel, QListWidget]] = []
    initial_preset_indexes: list[list[int]] = []
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
        initial_preset_indexes.append([int(idx) for idx in member_indexes])
    left_scroll.setWidget(left_container)
    split_layout.addWidget(left_scroll)

    right_scroll = QScrollArea(dialog)
    right_scroll.setWidgetResizable(True)
    container = QWidget(right_scroll)
    container_layout = QVBoxLayout(container)
    right_header = QLabel(
        "Right Panel: Similarity Groups (drag decks between groups)", container
    )
    container_layout.addWidget(right_header)
    group_widgets: list[tuple[int, QGroupBox, QLabel, QListWidget]] = []

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

    def _populate_preset_widgets(index_groups: Sequence[Sequence[int]]) -> None:
        for group_idx, (_, _, _, _, list_widget) in enumerate(preset_widgets):
            indexes = index_groups[group_idx] if group_idx < len(index_groups) else ()
            _populate_list_widget(list_widget, indexes)

    def _update_summary() -> None:
        if dialog_closed:
            return
        summary.setText(
            f"{len(recommended_group_indexes)} groups found "
            f"(pairwise {item_label.lower()} distance < {current_threshold:.1f})"
        )

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

    for _, _, _, _, list_widget in preset_widgets:
        model = list_widget.model()
        model.rowsInserted.connect(lambda *_args: _refresh_preset_views())
        model.rowsRemoved.connect(lambda *_args: _refresh_preset_views())
        model.modelReset.connect(lambda *_args: _refresh_preset_views())

    selection_syncing = False

    def _all_list_widgets() -> list[QListWidget]:
        return [row[3] for row in group_widgets] + [row[4] for row in preset_widgets]

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
            for list_widget in _all_list_widgets():
                try:
                    list_widget.blockSignals(True)
                except RuntimeError:
                    continue
            for list_widget in _all_list_widgets():
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
            for list_widget in _all_list_widgets():
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

    for _, _, _, _, list_widget in preset_widgets:
        list_widget.itemClicked.connect(_sync_selection_from_item)

    def _clear_similarity_widgets() -> None:
        nonlocal group_widgets
        for _, group_box, _, _ in group_widgets:
            try:
                container_layout.removeWidget(group_box)
                group_box.deleteLater()
            except RuntimeError:
                continue
        group_widgets = []

    def _rebuild_similarity_widgets(index_groups: Sequence[Sequence[int]]) -> None:
        _clear_similarity_widgets()
        for idx, group in enumerate(index_groups):
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

            model = list_widget.model()
            model.rowsInserted.connect(lambda *_args: _refresh_similarity_views())
            model.rowsRemoved.connect(lambda *_args: _refresh_similarity_views())
            model.modelReset.connect(lambda *_args: _refresh_similarity_views())
            list_widget.itemClicked.connect(_sync_selection_from_item)

    def _on_threshold_changed(raw_value: int) -> None:
        nonlocal current_threshold, recommended_group_indexes
        current_threshold = float(raw_value) / 10.0
        threshold_value_label.setText(f"{current_threshold:.1f}")
        recommended_group_indexes = _recommended_groups_from_threshold(current_threshold)
        _rebuild_similarity_widgets(recommended_group_indexes)
        _update_summary()
        _refresh_similarity_views()
        _refresh_preset_views()

    threshold_slider.valueChanged.connect(_on_threshold_changed)
    _on_threshold_changed(threshold_slider.value())

    right_scroll.setWidget(container)
    split_layout.addWidget(right_scroll)
    layout.addLayout(split_layout)

    default_values_layout = QHBoxLayout()
    default_values_layout.addWidget(QLabel("Preset Default Values:", dialog))
    preset_defaults_combo = QComboBox(dialog)
    preset_defaults_combo.addItem("Auto (per-group source preset)", None)
    for conf_id, conf_name, _ in _all_preset_configs():
        preset_defaults_combo.addItem(conf_name, int(conf_id))
    default_values_layout.addWidget(preset_defaults_combo)
    layout.addLayout(default_values_layout)

    buttons_layout = QHBoxLayout()
    if open_proximity_callback is not None:
        proximity_button = QPushButton("See Proximity Matrix", dialog)
        proximity_button.clicked.connect(open_proximity_callback)
        buttons_layout.addWidget(proximity_button)

    reset_button = QPushButton("Reset Groups", dialog)
    reset_button.clicked.connect(
        lambda: (
            _populate_group_widgets(recommended_group_indexes),
            _populate_preset_widgets(initial_preset_indexes),
            _refresh_similarity_views(),
            _refresh_preset_views(),
        )
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

    def _maybe_offer_reoptimization_for_changed_presets(
        *,
        before_assignments: Mapping[int, int],
        target_by_deck: Mapping[int, int],
        candidate_preset_ids: Sequence[int],
    ) -> None:
        after_target_assignments = _current_preset_assignments(list(target_by_deck.keys()))
        all_deck_ids = [deck_id for deck_id, _deck_name in _deck_entries()]
        all_current_assignments = _current_preset_assignments(all_deck_ids)
        changed_preset_deck_groups = changed_preset_deck_groups_for_reoptimization(
            before_assignments=before_assignments,
            after_target_assignments=after_target_assignments,
            target_by_deck=target_by_deck,
            candidate_preset_ids=candidate_preset_ids,
            all_current_assignments=all_current_assignments,
        )
        if not changed_preset_deck_groups:
            return

        optimize_choice = QMessageBox.question(
            dialog,
            "Optimize Changed Presets?",
            "Do you want to optimize FSRS params for advisor presets whose deck composition changed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if optimize_choice != QMessageBox.StandardButton.Yes:
            return

        optimized, no_data, invalid_params, optimization_failed, cancelled = (
            _optimize_preset_configs(changed_preset_deck_groups)
        )
        message = preset_optimization_summary_message(
            optimized=optimized,
            no_data=no_data,
            invalid_params=invalid_params,
            failed=optimization_failed,
            cancelled=cancelled,
        )
        if optimization_failed:
            showWarning(message)
        else:
            showInfo(message)

    recommended_button = QPushButton("Use Recommended Group", dialog)

    def _use_recommended_groups() -> None:
        create_config_id = getattr(mw.col.decks, "add_config_returning_id", None)
        create_config = getattr(mw.col.decks, "add_config", None)
        update_config = getattr(mw.col.decks, "update_config", None)
        selected_default_conf_id = _as_int(preset_defaults_combo.currentData())
        selected_default_conf = (
            _config_from_conf_id(selected_default_conf_id)
            if selected_default_conf_id is not None
            else None
        )

        existing_by_name: dict[str, int] = {}
        for conf_id, conf_name, _ in _all_preset_configs():
            if conf_name not in existing_by_name:
                existing_by_name[conf_name] = int(conf_id)
        created = 0
        reused = 0
        copied_settings = 0
        settings_copy_failed = 0
        failed = 0
        target_by_deck: dict[int, int] = {}
        recommended_conf_ids: list[int] = []

        for group_number, indexes in enumerate(recommended_group_indexes, start=1):
            valid_indexes = [idx for idx in indexes if 0 <= idx < len(deck_ids)]
            if len(valid_indexes) < 1:
                continue

            base_name = recommended_group_preset_name(group_number)
            new_conf_id = existing_by_name.get(base_name)
            if new_conf_id is not None:
                reused += 1
                if selected_default_conf is not None:
                    try:
                        target_conf = _config_from_conf_id(new_conf_id)
                        if target_conf is not None and callable(update_config):
                            cloned = copy.deepcopy(selected_default_conf)
                            if not isinstance(cloned, Mapping):
                                raise RuntimeError("Invalid source preset payload")
                            cloned = dict(cloned)
                            cloned["id"] = int(new_conf_id)
                            target_name = _field(target_conf, "name")
                            if isinstance(target_name, str) and target_name:
                                cloned["name"] = target_name
                            else:
                                cloned["name"] = base_name
                            update_config(cloned)
                            copied_settings += 1
                    except Exception:
                        settings_copy_failed += 1
            else:
                clone_from = selected_default_conf
                if clone_from is None:
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
                        new_conf_id = int(create_config_id(base_name, clone_from=clone_from))
                    elif create_config is not None:
                        new_conf = create_config(base_name, clone_from=clone_from)
                        new_conf_id = _as_int(_field(new_conf, "id"))
                        if new_conf_id is None:
                            raise RuntimeError("Created preset has no id")
                    else:
                        raise RuntimeError("Preset creation API unavailable")
                except Exception:
                    failed += len(valid_indexes)
                    continue

                existing_by_name[base_name] = int(new_conf_id)
                created += 1
                if selected_default_conf is not None:
                    copied_settings += 1

            for idx in valid_indexes:
                target_by_deck[deck_ids[idx]] = int(new_conf_id)
            if int(new_conf_id) not in recommended_conf_ids:
                recommended_conf_ids.append(int(new_conf_id))

        before_assignments: dict[int, int] = {}
        if target_by_deck:
            before_assignments = _current_preset_assignments(list(target_by_deck.keys()))
            _save_preset_backup(before_assignments)
            assigned, apply_failed = _apply_preset_assignments(target_by_deck)
            failed += apply_failed
        else:
            assigned = 0

        if (
            created == 0
            and reused == 0
            and copied_settings == 0
            and assigned == 0
            and failed == 0
            and settings_copy_failed == 0
        ):
            showInfo("No recommended groups were found.")
            return
        if failed:
            showWarning(
                f"Created {created} groups, reused {reused} groups, "
                f"copied settings on {copied_settings} groups, "
                f"and reassigned {assigned} decks, "
                f"with {failed} assignment failures and {settings_copy_failed} settings copy failures."
            )
        else:
            showInfo(
                f"Created {created} groups, reused {reused} groups, "
                f"copied settings on {copied_settings} groups, "
                f"and reassigned {assigned} decks.\n\n"
                "Reopen this window to refresh existing preset groups."
            )

        _maybe_offer_reoptimization_for_changed_presets(
            before_assignments=before_assignments,
            target_by_deck=target_by_deck,
            candidate_preset_ids=recommended_conf_ids,
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

        before_assignments = _current_preset_assignments(list(target_by_deck.keys()))
        _save_preset_backup(before_assignments)
        changed, failed = _apply_preset_assignments(target_by_deck)

        if failed:
            showWarning(f"Preset changes applied to {changed} decks, failed for {failed}.")
        elif changed:
            showInfo(f"Preset changes applied to {changed} decks.")
        else:
            showInfo("No preset changes to save.")

        candidate_preset_ids = sorted({int(conf_id) for conf_id in target_by_deck.values()})
        _maybe_offer_reoptimization_for_changed_presets(
            before_assignments=before_assignments,
            target_by_deck=target_by_deck,
            candidate_preset_ids=candidate_preset_ids,
        )

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
        min_group_size=1,
    )

    def _open_proximity_screen() -> None:
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

    _show_similarity_groups(
        title=_DECK_ACTION_LABEL,
        item_label="Deck",
        groups=similarity_groups,
        labels=labels,
        deck_ids=ordered_deck_ids,
        distances=matrix,
        preset_groups=preset_groups,
        open_proximity_callback=_open_proximity_screen,
    )


def _split_deck_by_first_review() -> None:
    entries = sorted(_deck_entries(), key=lambda item: item[1].lower())
    if not entries:
        showWarning("No decks available.")
        return

    deck_names = [name for _deck_id, name in entries]
    selected_name, accepted = QInputDialog.getItem(
        mw,
        _FIRST_REVIEW_SPLIT_LABEL,
        "Select source deck:",
        deck_names,
        0,
        False,
    )
    if not accepted or not selected_name:
        return

    selected_deck_name = str(selected_name)
    selected_deck_id = next(
        (int(deck_id) for deck_id, name in entries if name == selected_deck_name),
        None,
    )
    if selected_deck_id is None:
        showWarning("Selected deck was not found.")
        return

    scope_deck_ids = descendant_deck_ids(entries, selected_deck_name)
    if not scope_deck_ids:
        scope_deck_ids = [selected_deck_id]

    rows = _cards_and_first_review_ease_for_decks(scope_deck_ids)
    if not rows:
        showInfo(f'No cards found in "{selected_deck_name}".')
        return

    buckets, no_first_review_count, unexpected_rating_count = split_card_ids_by_first_review(
        card_first_ease_rows=rows
    )
    target_deck_names = target_deck_names_for_first_review_split(selected_deck_name)
    target_preset_names = target_preset_names_for_first_review_split(selected_deck_name)
    split_counts = {
        label: len(buckets[rating])
        for rating, label in FIRST_REVIEW_RATINGS
    }
    total_split = sum(split_counts.values())

    confirmation_lines = [
        f'Source deck: "{selected_deck_name}"',
        f"Cards in scope: {len(rows)}",
        f"Again: {split_counts['Again']}",
        f"Hard: {split_counts['Hard']}",
        f"Good: {split_counts['Good']}",
        f"Easy: {split_counts['Easy']}",
    ]
    if no_first_review_count:
        confirmation_lines.append(f"No first review: {no_first_review_count} (left unchanged)")
    if unexpected_rating_count:
        confirmation_lines.append(
            f"Unexpected first rating: {unexpected_rating_count} (left unchanged)"
        )

    proceed = QMessageBox.question(
        mw,
        _FIRST_REVIEW_SPLIT_LABEL,
        "\n".join(confirmation_lines) + "\n\nProceed with split?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.Yes,
    )
    if proceed != QMessageBox.StandardButton.Yes:
        return

    moved_counts: dict[str, int] = {label: 0 for _rating, label in FIRST_REVIEW_RATINGS}
    failed_count = 0
    failed_target_decks: list[str] = []
    failed_target_presets: list[str] = []
    created_presets = 0
    reused_presets = 0
    target_preset_assignments: dict[int, int] = {}
    create_config_id = getattr(mw.col.decks, "add_config_returning_id", None)
    create_config = getattr(mw.col.decks, "add_config", None)
    source_config = _config_for_deck(selected_deck_id)
    existing_presets_by_name: dict[str, int] = {}
    for conf_id, conf_name, _config in _all_preset_configs():
        if conf_name not in existing_presets_by_name:
            existing_presets_by_name[conf_name] = int(conf_id)

    for rating, label in FIRST_REVIEW_RATINGS:
        target_name = target_deck_names[rating]
        target_deck_id = _ensure_deck_id_for_name(target_name)
        if target_deck_id is None:
            failed_target_decks.append(target_name)
            failed_count += len(buckets[rating])
            continue

        preset_name = target_preset_names[rating]
        preset_id = existing_presets_by_name.get(preset_name)
        if preset_id is None:
            try:
                if create_config_id is not None:
                    preset_id = int(create_config_id(preset_name, clone_from=source_config))
                elif create_config is not None:
                    new_config = create_config(preset_name, clone_from=source_config)
                    preset_id = _as_int(_field(new_config, "id"))
                else:
                    preset_id = None
            except Exception:
                preset_id = None

            if preset_id is None:
                failed_target_presets.append(preset_name)
                failed_count += len(buckets[rating])
                continue
            created_presets += 1
            existing_presets_by_name[preset_name] = int(preset_id)
        else:
            reused_presets += 1

        target_preset_assignments[int(target_deck_id)] = int(preset_id)
        moved, failed = _move_cards_to_deck(
            card_ids=buckets[rating],
            target_deck_id=target_deck_id,
        )
        moved_counts[label] = moved
        failed_count += failed

    preset_assign_changed = 0
    preset_assign_failed = 0
    if target_preset_assignments:
        preset_assign_changed, preset_assign_failed = _apply_preset_assignments(target_preset_assignments)
        failed_count += preset_assign_failed

    if total_split > 0:
        try:
            mw.reset()
        except Exception:
            pass

    result_lines = [
        f'Split completed for "{selected_deck_name}".',
        f"Moved Again: {moved_counts['Again']}",
        f"Moved Hard: {moved_counts['Hard']}",
        f"Moved Good: {moved_counts['Good']}",
        f"Moved Easy: {moved_counts['Easy']}",
        (
            f"Presets: created {created_presets}, reused {reused_presets}, "
            f"assigned {preset_assign_changed}"
        ),
    ]
    if no_first_review_count:
        result_lines.append(f"Unchanged (no first review): {no_first_review_count}")
    if unexpected_rating_count:
        result_lines.append(f"Unchanged (unexpected first rating): {unexpected_rating_count}")
    if failed_target_decks:
        result_lines.append("Failed to create/find target decks:")
        for deck_name in failed_target_decks:
            result_lines.append(f"- {deck_name}")
    if failed_target_presets:
        result_lines.append("Failed to create/find target presets:")
        for preset_name in failed_target_presets:
            result_lines.append(f"- {preset_name}")
    if failed_count:
        showWarning("\n".join(result_lines + [f"Move failures: {failed_count}"]))
        return
    showInfo("\n".join(result_lines))


def _merge_back_first_review_split() -> None:
    entries = sorted(_deck_entries(), key=lambda item: item[1].lower())
    if not entries:
        showWarning("No decks available.")
        return

    deck_names = [
        name for _deck_id, name in entries if not is_first_review_split_deck_name(name)
    ]
    if not deck_names:
        showWarning("No base deck available to merge into.")
        return

    selected_name, accepted = QInputDialog.getItem(
        mw,
        _FIRST_REVIEW_UNSPLIT_LABEL,
        "Select base deck:",
        deck_names,
        0,
        False,
    )
    if not accepted or not selected_name:
        return

    base_deck_name = str(selected_name)
    base_deck_id = next(
        (int(deck_id) for deck_id, name in entries if name == base_deck_name),
        None,
    )
    if base_deck_id is None:
        showWarning("Selected deck was not found.")
        return

    split_deck_names = target_deck_names_for_first_review_split(base_deck_name)
    per_label_cards: dict[str, list[int]] = {}
    existing_split_decks = 0
    for _rating, label in FIRST_REVIEW_RATINGS:
        split_name = split_deck_names[_rating]
        split_scope_ids = descendant_deck_ids(entries, split_name)
        if not split_scope_ids:
            direct_id = next(
                (int(deck_id) for deck_id, name in entries if name == split_name),
                None,
            )
            if direct_id is not None:
                split_scope_ids = [direct_id]
        if not split_scope_ids:
            per_label_cards[label] = []
            continue
        existing_split_decks += 1
        per_label_cards[label] = _card_ids_for_decks(split_scope_ids)

    if existing_split_decks == 0:
        showInfo(f'No "{base_deck_name} - Again/Hard/Good/Easy" decks found.')
        return

    total_to_merge = sum(len(card_ids) for card_ids in per_label_cards.values())
    if total_to_merge <= 0:
        showInfo("Split decks exist, but there are no cards to merge back.")
        return

    confirmation_lines = [
        f'Base deck: "{base_deck_name}"',
        f"Again -> base: {len(per_label_cards['Again'])}",
        f"Hard -> base: {len(per_label_cards['Hard'])}",
        f"Good -> base: {len(per_label_cards['Good'])}",
        f"Easy -> base: {len(per_label_cards['Easy'])}",
        f"Total cards to merge back: {total_to_merge}",
    ]
    proceed = QMessageBox.question(
        mw,
        _FIRST_REVIEW_UNSPLIT_LABEL,
        "\n".join(confirmation_lines) + "\n\nProceed with merge back?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.Yes,
    )
    if proceed != QMessageBox.StandardButton.Yes:
        return

    merged_counts: dict[str, int] = {label: 0 for _rating, label in FIRST_REVIEW_RATINGS}
    failed_count = 0
    for _rating, label in FIRST_REVIEW_RATINGS:
        moved, failed = _move_cards_to_deck(
            card_ids=per_label_cards[label],
            target_deck_id=base_deck_id,
        )
        merged_counts[label] = moved
        failed_count += failed

    try:
        mw.reset()
    except Exception:
        pass

    deleted_split_decks: list[str] = []
    delete_failed_split_decks: list[str] = []
    refreshed_entries = sorted(_deck_entries(), key=lambda item: item[1].lower())
    for _rating, label in FIRST_REVIEW_RATINGS:
        split_name = split_deck_names[_rating]
        split_scope_ids = descendant_deck_ids(refreshed_entries, split_name)
        if not split_scope_ids:
            continue
        if _card_ids_for_decks(split_scope_ids):
            continue
        root_id = next(
            (int(deck_id) for deck_id, name in refreshed_entries if name == split_name),
            None,
        )
        if root_id is None:
            continue
        if _delete_deck_by_id(root_id):
            deleted_split_decks.append(split_name)
        else:
            delete_failed_split_decks.append(split_name)

    result_lines = [
        f'Merged cards back into "{base_deck_name}".',
        f"Merged Again: {merged_counts['Again']}",
        f"Merged Hard: {merged_counts['Hard']}",
        f"Merged Good: {merged_counts['Good']}",
        f"Merged Easy: {merged_counts['Easy']}",
        f"Total merged: {sum(merged_counts.values())}",
    ]
    if deleted_split_decks:
        result_lines.append(f"Deleted empty split decks: {len(deleted_split_decks)}")
    if delete_failed_split_decks:
        result_lines.append("Failed to delete empty split decks:")
        for deck_name in delete_failed_split_decks:
            result_lines.append(f"- {deck_name}")
    if failed_count:
        showWarning("\n".join(result_lines + [f"Move failures: {failed_count}"]))
        return
    showInfo("\n".join(result_lines))


def _cleanup_empty_advisor_presets() -> None:
    all_presets = [(int(conf_id), str(conf_name)) for conf_id, conf_name, _cfg in _all_preset_configs()]
    all_deck_ids = [int(deck_id) for deck_id, _deck_name in _deck_entries()]
    current_assignments = _current_preset_assignments(all_deck_ids)
    used_preset_ids = sorted({int(conf_id) for conf_id in current_assignments.values()})

    candidates = empty_advisor_preset_candidates(
        presets=all_presets,
        used_preset_ids=used_preset_ids,
    )
    if not candidates:
        showInfo("No empty advisor presets to clean.")
        return

    preview_limit = 15
    preview = [f"- {name}" for _conf_id, name in candidates[:preview_limit]]
    remaining = len(candidates) - len(preview)
    if remaining > 0:
        preview.append(f"... and {remaining} more")

    proceed = QMessageBox.question(
        mw,
        _CLEAN_EMPTY_PRESETS_LABEL,
        (
            f"Found {len(candidates)} empty advisor preset(s) not assigned to any deck.\n\n"
            + "\n".join(preview)
            + "\n\nDelete them?"
        ),
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.Yes,
    )
    if proceed != QMessageBox.StandardButton.Yes:
        return

    deleted = 0
    failed: list[str] = []
    for conf_id, conf_name in candidates:
        if _delete_preset_by_id(conf_id):
            deleted += 1
        else:
            failed.append(conf_name)

    try:
        mw.reset()
    except Exception:
        pass

    if failed:
        lines = [
            f"Deleted {deleted} preset(s), failed to delete {len(failed)}.",
            "Failed presets:",
        ]
        lines.extend(f"- {name}" for name in failed)
        showWarning("\n".join(lines))
        return

    showInfo(f"Deleted {deleted} empty advisor preset(s).")


def _show_mergeability_evaluation() -> None:
    choice = QMessageBox.question(
        mw,
        _EVALUATE_MERGEABILITY_LABEL,
        "Default scope is leaf decks only.\n\n"
        "Do you also want to include middle/root decks for the computation?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    include_middle_and_root = choice == QMessageBox.StandardButton.Yes
    progress = QProgressDialog("Preparing deck optimizations...", "Cancel", 0, 100, mw)
    progress.setWindowTitle(_EVALUATE_MERGEABILITY_LABEL)
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
    if len(profiles) < 2:
        showWarning("At least 2 decks with usable FSRS6 computed parameters are required.")
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

    ordered_profiles = sorted(profiles, key=lambda profile: profile.profile_name.lower())
    labels = [profile.profile_name for profile in ordered_profiles]
    entries = sorted(_deck_entries(), key=lambda item: item[1].lower())
    searches = [
        build_deck_search_query(
            deck_id=profile.profile_id,
            deck_name=profile.profile_name,
            include_children=include_middle_and_root,
        )
        for profile in ordered_profiles
    ]
    review_counts = [
        _review_count_for_deck_scope(
            deck_id=profile.profile_id,
            deck_name=profile.profile_name,
            include_children=include_middle_and_root,
            entries=entries,
        )
        for profile in ordered_profiles
    ]

    total_evals = len(ordered_profiles) * len(ordered_profiles)
    eval_progress = QProgressDialog("Evaluating mergeability logloss...", "Cancel", 0, total_evals, mw)
    eval_progress.setWindowTitle(_EVALUATE_MERGEABILITY_LABEL)
    eval_progress.setAutoClose(False)
    eval_progress.setAutoReset(False)
    eval_progress.setMinimumDuration(0)
    eval_progress.setValue(0)
    evaluate_cache = _load_evaluate_logloss_cache()
    evaluate_cache_dirty = False

    losses: list[list[float | None]] = [
        [None for _ in range(len(ordered_profiles))] for _ in range(len(ordered_profiles))
    ]
    eval_failed = 0
    done = 0
    cancelled_eval = False
    try:
        for target_idx, target_profile in enumerate(ordered_profiles):
            for source_idx, source_profile in enumerate(ordered_profiles):
                eval_progress.setValue(done)
                eval_progress.setLabelText(
                    f'Evaluating "{target_profile.profile_name}" with params from '
                    f'"{source_profile.profile_name}"\nCompleted: {done}/{total_evals}'
                )
                eval_progress.show()
                eval_progress.repaint()
                mw.app.processEvents()
                eval_progress.repaint()
                if eval_progress.wasCanceled():
                    cancelled_eval = True
                    break
                cache_key = evaluate_logloss_cache_key(
                    target_deck_id=target_profile.profile_id,
                    include_children=include_middle_and_root,
                    source_params=source_profile.weights,
                )
                cached_entry = evaluate_cache.get(cache_key, {})
                cached_review_count = _as_int(cached_entry.get("review_count"))
                cached_log_loss = _to_float(cached_entry.get("log_loss"))
                if can_reuse_evaluate_cached_logloss(
                    cached_review_count=cached_review_count,
                    current_review_count=review_counts[target_idx],
                    cached_log_loss=cached_log_loss,
                ):
                    losses[target_idx][source_idx] = cached_log_loss
                    done += 1
                    continue
                try:
                    computed_log_loss = _evaluate_logloss_with_params(
                        params=source_profile.weights,
                        search=searches[target_idx],
                    )
                    losses[target_idx][source_idx] = computed_log_loss
                    evaluate_cache[cache_key] = {
                        "review_count": review_counts[target_idx],
                        "log_loss": computed_log_loss,
                    }
                    evaluate_cache_dirty = True
                except Exception:
                    eval_failed += 1
                    losses[target_idx][source_idx] = None
                done += 1
            if cancelled_eval:
                break
    finally:
        eval_progress.close()
    if evaluate_cache_dirty:
        _save_evaluate_logloss_cache(evaluate_cache)

    if cancelled_eval and done == 0:
        showWarning("Mergeability evaluation cancelled.")
        return
    if cancelled_eval:
        showInfo("Mergeability evaluation cancelled. Showing partial results.")
    if eval_failed:
        showWarning(f"Mergeability evaluation had {eval_failed} failed logloss evaluations.")

    score_matrix = symmetric_merge_score_matrix(losses=losses, review_counts=review_counts)
    maha_ordered_profiles, maha_raw_matrix = pairwise_distance_matrix(ordered_profiles)
    profile_index_by_id = {
        profile.profile_id: idx for idx, profile in enumerate(maha_ordered_profiles)
    }
    maha_matrix: list[list[float | None]] = [
        [None for _ in range(len(ordered_profiles))] for _ in range(len(ordered_profiles))
    ]
    for row_idx, row_profile in enumerate(ordered_profiles):
        mapped_row = profile_index_by_id.get(row_profile.profile_id)
        if mapped_row is None:
            continue
        for col_idx, col_profile in enumerate(ordered_profiles):
            mapped_col = profile_index_by_id.get(col_profile.profile_id)
            if mapped_col is None:
                continue
            maha_matrix[row_idx][col_idx] = maha_raw_matrix[mapped_row][mapped_col]

    preset_groups_map: dict[int, dict[str, Any]] = {}
    for idx, profile in enumerate(ordered_profiles):
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
    current_preset_groups = sorted(
        [
            (
                int(conf_id),
                str(data["name"]),
                [int(v) for v in data["indexes"]],
            )
            for conf_id, data in preset_groups_map.items()
        ],
        key=lambda item: item[1].lower(),
    )

    dialog = QDialog(mw)
    dialog.setWindowTitle(f"{_EVALUATE_MERGEABILITY_LABEL} - Group Comparison")
    layout = QVBoxLayout(dialog)

    summary_label = QLabel(dialog)
    summary_label.setWordWrap(True)
    layout.addWidget(summary_label)

    sliders_layout = QHBoxLayout()
    orientation_enum = getattr(Qt, "Orientation", None)
    horizontal = orientation_enum.Horizontal if orientation_enum is not None else Qt.Horizontal
    maha_slider_layout = QHBoxLayout()
    maha_slider_layout.addWidget(QLabel("Mahalanobis Threshold:", dialog))
    maha_slider = QSlider(horizontal, dialog)
    maha_slider.setMinimum(1)
    maha_slider.setMaximum(100)
    maha_slider.setSingleStep(1)
    maha_slider.setValue(int(round(FSRS6_RECENCY_MAHALANOBIS_SHARED_PRESET_THRESHOLD * 10)))
    maha_slider_layout.addWidget(maha_slider)
    maha_value_label = QLabel(dialog)
    maha_slider_layout.addWidget(maha_value_label)
    sliders_layout.addLayout(maha_slider_layout)

    logloss_slider_layout = QHBoxLayout()
    logloss_slider_layout.addWidget(QLabel("LogLoss Delta Threshold:", dialog))
    logloss_slider = QSlider(horizontal, dialog)
    logloss_slider.setMinimum(0)
    logloss_slider.setMaximum(50)
    logloss_slider.setSingleStep(1)
    logloss_slider.setValue(3)
    logloss_slider_layout.addWidget(logloss_slider)
    logloss_value_label = QLabel(dialog)
    logloss_slider_layout.addWidget(logloss_value_label)
    sliders_layout.addLayout(logloss_slider_layout)
    layout.addLayout(sliders_layout)

    panels_layout = QHBoxLayout()
    layout.addLayout(panels_layout)

    current_scroll = QScrollArea(dialog)
    current_scroll.setWidgetResizable(True)
    current_container = QWidget(current_scroll)
    current_layout = QVBoxLayout(current_container)
    current_header = QLabel("Current Preset", current_container)
    current_layout.addWidget(current_header)
    current_summary = QLabel(current_container)
    current_summary.setWordWrap(True)
    current_layout.addWidget(current_summary)
    current_scroll.setWidget(current_container)
    panels_layout.addWidget(current_scroll)

    maha_scroll = QScrollArea(dialog)
    maha_scroll.setWidgetResizable(True)
    maha_container = QWidget(maha_scroll)
    maha_layout = QVBoxLayout(maha_container)
    maha_header = QLabel("Mahalanobis Distance", maha_container)
    maha_layout.addWidget(maha_header)
    maha_summary = QLabel(maha_container)
    maha_summary.setWordWrap(True)
    maha_layout.addWidget(maha_summary)
    maha_scroll.setWidget(maha_container)
    panels_layout.addWidget(maha_scroll)

    logloss_scroll = QScrollArea(dialog)
    logloss_scroll.setWidgetResizable(True)
    logloss_container = QWidget(logloss_scroll)
    logloss_layout = QVBoxLayout(logloss_container)
    logloss_header = QLabel("LogLoss Proximity", logloss_container)
    logloss_layout.addWidget(logloss_header)
    logloss_summary = QLabel(logloss_container)
    logloss_summary.setWordWrap(True)
    logloss_layout.addWidget(logloss_summary)
    logloss_scroll.setWidget(logloss_container)
    panels_layout.addWidget(logloss_scroll)

    current_group_boxes: list[QGroupBox] = []
    maha_group_boxes: list[QGroupBox] = []
    logloss_group_boxes: list[QGroupBox] = []
    active_maha_groups: list[list[int]] = []
    active_logloss_groups: list[list[int]] = []
    maha_apply_button = QPushButton("Use this split", maha_container)
    logloss_apply_button = QPushButton("Use this split", logloss_container)
    maha_layout.addWidget(maha_apply_button)
    logloss_layout.addWidget(logloss_apply_button)

    def _clear_group_boxes(target: list[QGroupBox], panel_layout: QVBoxLayout, summary: QLabel) -> None:
        for group_box in target:
            panel_layout.removeWidget(group_box)
            group_box.deleteLater()
        target.clear()

    def _render_groups(
        *,
        panel_layout: QVBoxLayout,
        summary: QLabel,
        target: list[QGroupBox],
        groups: Sequence[Sequence[int]],
        header_text: str,
        distances: Sequence[Sequence[float | None]] | None = None,
        threshold: float | None = None,
        display_names_by_group: Sequence[Sequence[str]] | None = None,
        insert_before_widget: QWidget | None = None,
    ) -> None:
        _clear_group_boxes(target, panel_layout, summary)
        summary.setText(header_text)
        selection_mode_enum = getattr(QAbstractItemView, "SelectionMode", None)
        no_selection = (
            selection_mode_enum.NoSelection
            if selection_mode_enum is not None
            else QAbstractItemView.NoSelection
        )
        for group_number, group_indexes in enumerate(groups, start=1):
            max_text = "n/a"
            if distances is not None:
                max_value = max_pairwise_distance_for_group(
                    group_indexes=group_indexes,
                    distances=distances,
                )
                if max_value is not None:
                    max_text = f"{max_value:.4f}"
            title = f"Group {group_number} (Max: {max_text})"
            if threshold is not None and max_text != "n/a":
                title += f" / <= {threshold:.4f}"
            box = QGroupBox(title, panel_layout.parentWidget())
            box_layout = QVBoxLayout(box)
            deck_list = QListWidget(box)
            deck_list.setSelectionMode(no_selection)
            if display_names_by_group is not None and (group_number - 1) < len(display_names_by_group):
                for display_name in display_names_by_group[group_number - 1]:
                    QListWidgetItem(str(display_name), deck_list)
            else:
                for idx in group_indexes:
                    if 0 <= idx < len(labels):
                        QListWidgetItem(labels[idx], deck_list)
            box_layout.addWidget(deck_list)
            if insert_before_widget is not None:
                insert_index = panel_layout.indexOf(insert_before_widget)
                if insert_index < 0:
                    panel_layout.addWidget(box)
                else:
                    panel_layout.insertWidget(insert_index, box)
            else:
                panel_layout.addWidget(box)
            target.append(box)

    current_group_indexes = [group[2] for group in current_preset_groups if group[2]]
    _render_groups(
        panel_layout=current_layout,
        summary=current_summary,
        target=current_group_boxes,
        groups=current_group_indexes,
        header_text=f"{len(current_group_indexes)} current preset groups",
        distances=maha_matrix,
    )

    total_pairs = (len(labels) * (len(labels) - 1)) // 2

    def _group_indexes_from_names(group_names: Sequence[Sequence[str]]) -> list[list[int]]:
        index_by_name = {name: idx for idx, name in enumerate(labels)}
        return [[index_by_name[name] for name in group if name in index_by_name] for group in group_names]

    def _apply_split_to_presets(*, split_label: str, group_indexes: Sequence[Sequence[int]]) -> None:
        create_config_id = getattr(mw.col.decks, "add_config_returning_id", None)
        create_config = getattr(mw.col.decks, "add_config", None)

        existing_by_name: dict[str, int] = {}
        for conf_id, conf_name, _ in _all_preset_configs():
            if conf_name not in existing_by_name:
                existing_by_name[conf_name] = int(conf_id)

        created = 0
        reused = 0
        failed = 0
        target_by_deck: dict[int, int] = {}
        applied_conf_ids: list[int] = []

        for group_number, indexes in enumerate(group_indexes, start=1):
            valid_indexes = [idx for idx in indexes if 0 <= idx < len(ordered_profiles)]
            if not valid_indexes:
                continue

            base_name = f"FSRS Preset Advisor : {split_label} Group {group_number}"
            new_conf_id = existing_by_name.get(base_name)
            if new_conf_id is not None:
                reused += 1
            else:
                clone_from = None
                for idx in valid_indexes:
                    profile_deck_id = ordered_profiles[idx].profile_id
                    conf = _config_for_deck(profile_deck_id)
                    if conf is not None:
                        clone_from = conf
                        break
                try:
                    if create_config_id is not None:
                        new_conf_id = int(create_config_id(base_name, clone_from=clone_from))
                    elif create_config is not None:
                        new_conf = create_config(base_name, clone_from=clone_from)
                        new_conf_id = _as_int(_field(new_conf, "id"))
                        if new_conf_id is None:
                            raise RuntimeError("Created preset has no id")
                    else:
                        raise RuntimeError("Preset creation API unavailable")
                except Exception:
                    failed += len(valid_indexes)
                    continue
                created += 1
                existing_by_name[base_name] = int(new_conf_id)

            for idx in valid_indexes:
                deck_id = ordered_profiles[idx].profile_id
                target_by_deck[int(deck_id)] = int(new_conf_id)
            if int(new_conf_id) not in applied_conf_ids:
                applied_conf_ids.append(int(new_conf_id))

        if not target_by_deck:
            showInfo(f"No groups available to apply for {split_label}.")
            return

        before_assignments = _current_preset_assignments(list(target_by_deck.keys()))
        _save_preset_backup(before_assignments)
        assigned, apply_failed = _apply_preset_assignments(target_by_deck)
        failed += apply_failed

        if failed:
            showWarning(
                f"{split_label}: created {created} groups, reused {reused} groups, "
                f"reassigned {assigned} decks, with {failed} failures."
            )
        else:
            showInfo(
                f"{split_label}: created {created} groups, reused {reused} groups, "
                f"reassigned {assigned} decks."
            )

        after_target_assignments = _current_preset_assignments(list(target_by_deck.keys()))
        all_deck_ids = [deck_id for deck_id, _deck_name in _deck_entries()]
        all_current_assignments = _current_preset_assignments(all_deck_ids)
        changed_preset_deck_groups = changed_preset_deck_groups_for_reoptimization(
            before_assignments=before_assignments,
            after_target_assignments=after_target_assignments,
            target_by_deck=target_by_deck,
            candidate_preset_ids=applied_conf_ids,
            all_current_assignments=all_current_assignments,
        )
        if not changed_preset_deck_groups:
            return
        optimize_choice = QMessageBox.question(
            dialog,
            "Optimize Changed Presets?",
            "Do you want to optimize FSRS params for advisor presets whose deck composition changed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if optimize_choice != QMessageBox.StandardButton.Yes:
            return
        optimized, no_data, invalid_params, optimization_failed, cancelled = (
            _optimize_preset_configs(changed_preset_deck_groups)
        )
        message = preset_optimization_summary_message(
            optimized=optimized,
            no_data=no_data,
            invalid_params=invalid_params,
            failed=optimization_failed,
            cancelled=cancelled,
        )
        if optimization_failed:
            showWarning(message)
        else:
            showInfo(message)

    def _refresh_group_panels() -> None:
        nonlocal active_maha_groups, active_logloss_groups
        maha_threshold = float(maha_slider.value()) / 10.0
        logloss_threshold = float(logloss_slider.value()) / 1000.0
        maha_value_label.setText(f"{maha_threshold:.1f}")
        logloss_value_label.setText(f"{logloss_threshold:.3f}")

        maha_groups_names = similarity_groups_from_matrix(
            names=labels,
            distances=maha_matrix,
            threshold=maha_threshold,
            min_group_size=1,
        )
        maha_groups = _group_indexes_from_names(maha_groups_names)
        active_maha_groups = [list(group) for group in maha_groups]
        maha_mergeable = mergeable_pair_count(score_matrix=maha_matrix, threshold=maha_threshold)

        logloss_groups = cluster_logloss_groups_agglomerative(
            losses=losses,
            review_counts=review_counts,
            threshold=logloss_threshold,
        )
        active_logloss_groups = [list(group) for group in logloss_groups]
        logloss_mergeable = mergeable_pair_count(score_matrix=score_matrix, threshold=logloss_threshold)

        aligned_pairs = align_group_indexes_by_overlap(
            left_groups=maha_groups,
            right_groups=logloss_groups,
        )
        ordered_maha_groups = [
            maha_groups[left_idx] for left_idx, _right_idx in aligned_pairs if left_idx is not None
        ]
        ordered_logloss_groups = [
            logloss_groups[right_idx]
            for _left_idx, right_idx in aligned_pairs
            if right_idx is not None
        ]
        display_maha_groups: list[list[str]] = []
        display_logloss_groups: list[list[str]] = []
        for left_idx, right_idx in aligned_pairs:
            left_group = maha_groups[left_idx] if left_idx is not None else []
            right_group = logloss_groups[right_idx] if right_idx is not None else []
            common = set(left_group) & set(right_group)

            if left_idx is not None:
                display_maha_groups.append(
                    [
                        f'{"= " if idx in common else "- "}{labels[idx]}'
                        for idx in left_group
                    ]
                )
            if right_idx is not None:
                display_logloss_groups.append(
                    [
                        f'{"= " if idx in common else "+ "}{labels[idx]}'
                        for idx in right_group
                    ]
                )

        _render_groups(
            panel_layout=maha_layout,
            summary=maha_summary,
            target=maha_group_boxes,
            groups=ordered_maha_groups,
            header_text=f"{len(maha_groups)} groups, mergeable pairs: {maha_mergeable}/{total_pairs}",
            distances=maha_matrix,
            threshold=maha_threshold,
            display_names_by_group=display_maha_groups,
            insert_before_widget=maha_apply_button,
        )
        _render_groups(
            panel_layout=logloss_layout,
            summary=logloss_summary,
            target=logloss_group_boxes,
            groups=ordered_logloss_groups,
            header_text=(
                f"{len(logloss_groups)} groups, mergeable pairs: "
                f"{logloss_mergeable}/{total_pairs}"
            ),
            distances=score_matrix,
            threshold=logloss_threshold,
            display_names_by_group=display_logloss_groups,
            insert_before_widget=logloss_apply_button,
        )

        summary_label.setText(
            f"Decks: {len(labels)}. "
            f"Mahalanobis threshold: {maha_threshold:.1f}. "
            f"LogLoss delta threshold: {logloss_threshold:.3f}."
        )

    maha_apply_button.clicked.connect(
        lambda: _apply_split_to_presets(split_label="Mahalanobis", group_indexes=active_maha_groups)
    )

    logloss_apply_button.clicked.connect(
        lambda: _apply_split_to_presets(split_label="LogLoss", group_indexes=active_logloss_groups)
    )

    maha_slider.valueChanged.connect(_refresh_group_panels)
    logloss_slider.valueChanged.connect(_refresh_group_panels)
    _refresh_group_panels()

    dialog.resize(1300, 900)
    dialog.exec()


def init_addon() -> None:
    preset_action = QAction(_ACTION_LABEL, mw)
    preset_action.triggered.connect(_show_preset_results)
    mw.form.menuTools.addAction(preset_action)

    deck_action = QAction(_DECK_ACTION_LABEL, mw)
    deck_action.triggered.connect(_show_deck_computed_results)
    mw.form.menuTools.addAction(deck_action)

    first_review_split_action = QAction(_FIRST_REVIEW_SPLIT_LABEL, mw)
    first_review_split_action.triggered.connect(_split_deck_by_first_review)
    mw.form.menuTools.addAction(first_review_split_action)

    first_review_unsplit_action = QAction(_FIRST_REVIEW_UNSPLIT_LABEL, mw)
    first_review_unsplit_action.triggered.connect(_merge_back_first_review_split)
    mw.form.menuTools.addAction(first_review_unsplit_action)

    cleanup_presets_action = QAction(_CLEAN_EMPTY_PRESETS_LABEL, mw)
    cleanup_presets_action.triggered.connect(_cleanup_empty_advisor_presets)
    mw.form.menuTools.addAction(cleanup_presets_action)

    evaluate_action = QAction(_EVALUATE_MERGEABILITY_LABEL, mw)
    evaluate_action.triggered.connect(_show_mergeability_evaluation)
    mw.form.menuTools.addAction(evaluate_action)
