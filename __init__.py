try:
    from .fsrs_merge_advisor.addon import init_addon
except ImportError:
    # Allows running unit tests outside Anki where `aqt` is unavailable.
    init_addon = None

if init_addon is not None:
    init_addon()
