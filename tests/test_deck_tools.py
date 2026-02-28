from fsrs_merge_advisor.deck_tools import (
    build_deck_search_query,
    count_relearning_steps_in_day,
    leaf_deck_entries,
    optimization_progress_message,
    similar_items_below_threshold,
)


def test_leaf_deck_entries_only_returns_true_leaves():
    entries = [
        (1, "Languages"),
        (2, "Languages::French"),
        (3, "Languages::French::Vocabulary"),
        (4, "Languages::Spanish"),
        (5, "Math"),
    ]

    leaves = leaf_deck_entries(entries)

    assert leaves == [
        (3, "Languages::French::Vocabulary"),
        (4, "Languages::Spanish"),
        (5, "Math"),
    ]


def test_count_relearning_steps_in_day_stops_at_24h_limit():
    assert count_relearning_steps_in_day([10, 20, 30]) == 3
    assert count_relearning_steps_in_day([1440]) == 0
    assert count_relearning_steps_in_day([1000, 300, 200, 100]) == 2


def test_build_deck_search_query_without_children_uses_did():
    assert (
        build_deck_search_query(deck_id=42, deck_name="A::B", include_children=False)
        == "did:42 -is:suspended"
    )


def test_build_deck_search_query_with_children_uses_escaped_deck_name():
    assert (
        build_deck_search_query(
            deck_id=42,
            deck_name='A::"Quoted"\\Name',
            include_children=True,
        )
        == 'deck:"A::\\"Quoted\\"\\\\Name" -is:suspended'
    )


def test_optimization_progress_message_with_deck_name():
    assert optimization_progress_message(done=3, total=10, deck_name="Vocabulary") == (
        'Optimizing "Vocabulary"\nCompleted: 3/10\nRemaining: 7'
    )


def test_optimization_progress_message_without_deck_name():
    assert optimization_progress_message(done=10, total=10, deck_name=None) == (
        "Preparing deck optimizations...\nCompleted: 10/10\nRemaining: 0"
    )


def test_similar_items_below_threshold_filters_and_sorts():
    names = ["A", "B", "C", "D"]
    row = [0.0, 2.5, 4.2, 2.5]
    assert similar_items_below_threshold(
        names=names,
        distances_row=row,
        self_index=0,
        threshold=3.8,
    ) == ["B (2.5000)", "D (2.5000)"]


def test_similar_items_below_threshold_excludes_self_none_and_boundary():
    names = ["A", "B", "C"]
    row = [0.0, None, 3.8]
    assert similar_items_below_threshold(
        names=names,
        distances_row=row,
        self_index=0,
        threshold=3.8,
    ) == []
