"""Feature engineering for Poker44 v3.0 subject-session v2 payloads.

v3.0 replaces isolated hand chunks with `subject-session.v2` records: 1-3
poker hands plus sanitized interaction telemetry (click/pointer/scroll events
and aggregate timing statistics). See
https://github.com/Poker44/Poker44-subnet/blob/dev/docs/tournament-evaluation-workflow.md
for the normative contract.

Design notes (from the published migration checklist):
  - Features must be *behavioral*, not shape-based. The platform runs a
    provenance-leak canary and explicitly monitors "classifiers based only on
    payload byte length or raw event volume" as unstable/gameable signals.
    Every telemetry feature here is a ratio, rate, or dispersion statistic
    rather than a raw count, for that reason.
  - `session_id`, `window_id`, array order and request timing are metadata,
    not behavioral features, and are never used here.
  - v2 amounts are raw chip counts (no bb normalization field is exposed), so
    bet sizing is expressed relative to `pot_size`/`current_bet`/
    `player_stack` instead of assuming a known blind level.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

ACTION_TYPES = ("fold", "check", "call", "bet", "raise", "all_in")
AGG_ACTIONS = frozenset({"bet", "raise", "all_in"})
PASSIVE_ACTIONS = frozenset({"check", "call"})
COMMON_POT_FRACS = (0.33, 0.5, 0.66, 0.75, 1.0)

TELEMETRY_EVENT_TYPES = (
    "click",
    "pointer_down",
    "pointer_move",
    "scroll",
    "focus_in",
    "visibility",
)
TARGET_CATEGORIES = ("poker_action", "navigation", "control", "other")

HAND_KEYS = [
    "h_n_actions",
    "h_max_active_players",
    "h_n_phases",
    "h_ratio_fold",
    "h_ratio_check",
    "h_ratio_call",
    "h_ratio_bet",
    "h_ratio_raise",
    "h_ratio_all_in",
    "h_agg_ratio",
    "h_betpot_mean",
    "h_betpot_std",
    "h_betpot_snap_uniq",
    "h_stackpot_mean",
    "h_bigram_uniq",
    "h_bigram_ent",
    "h_reached_flop",
    "h_reached_turn",
    "h_reached_river",
    "h_decision_ms_mean",
    "h_decision_ms_cv",
    "h_gap_ms_mean",
    "h_gap_ms_cv",
    "h_heuristic",
]

AGG_SUFFIXES = ("mean", "std", "q25", "q75", "max")

TELEMETRY_KEYS = [
    "t_events_per_action",
    "t_offset_gap_cv",
    "t_offset_gap_mean_ms",
    "t_event_type_entropy",
    "t_target_entropy",
    "t_poker_action_share",
    "t_pointer_move_share",
    "t_click_share",
    "t_scroll_share",
    "t_visibility_share",
    "t_xy_bucket_entropy",
    "t_duration_per_hand_ms",
    "t_decision_count_ratio",
    "t_decision_mean_ms",
    "t_decision_std_ms",
    "t_decision_cv",
    "t_decision_mismatch",
]

FEATURE_NAMES = [
    *[f"agg_{suffix}_{key}" for key in HAND_KEYS for suffix in AGG_SUFFIXES],
    "n_hands",
    "cross_decision_cv_of_means",
    *TELEMETRY_KEYS,
]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    probs = np.asarray(list(counts.values()), dtype=np.float64) / total
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def _hand_heuristic_score(hand: dict[str, Any]) -> float:
    actions = hand.get("actions") or []
    counts = Counter(str(a.get("action_type") or "") for a in actions)
    meaningful = max(1, sum(counts.get(k, 0) for k in ACTION_TYPES))
    fold_ratio = counts.get("fold", 0) / meaningful
    raise_ratio = (counts.get("raise", 0) + counts.get("all_in", 0)) / meaningful
    call_ratio = counts.get("call", 0) / meaningful
    phases = {str(a.get("phase") or "") for a in actions if a.get("phase")}
    phase_depth = len(phases) / 4.0
    score = (
        0.35 * _clamp01(phase_depth)
        + 0.20 * _clamp01(call_ratio / 0.4)
        - 0.20 * _clamp01(fold_ratio / 0.55)
        - 0.15 * _clamp01(raise_ratio / 0.25)
    )
    return _clamp01(score + 0.5)


def _hand_feature_dict(hand: dict[str, Any]) -> dict[str, float]:
    actions = [a for a in (hand.get("actions") or []) if isinstance(a, dict)]
    counts = Counter(str(a.get("action_type") or "") for a in actions)
    n_actions = len(actions)
    agg = sum(counts.get(k, 0) for k in AGG_ACTIONS)
    passive = sum(counts.get(k, 0) for k in PASSIVE_ACTIONS)

    bet_pot_fracs: list[float] = []
    stack_pot_ratios: list[float] = []
    active_players: list[int] = []
    decision_times: list[float] = []
    gap_times: list[float] = []

    for action in actions:
        pot_size = float(action.get("pot_size") or 0.0)
        amount = action.get("amount")
        raise_to = action.get("raise_to")
        call_amount = action.get("call_amount")
        chip_move = 0.0
        for candidate in (raise_to, amount, call_amount):
            if candidate is not None:
                chip_move = float(candidate)
                break
        if chip_move > 0.0 and pot_size > 0.0:
            bet_pot_fracs.append(chip_move / pot_size)

        player_stack = action.get("player_stack")
        if player_stack is not None and pot_size > 0.0:
            stack_pot_ratios.append(float(player_stack) / pot_size)

        if action.get("active_players") is not None:
            active_players.append(int(action["active_players"]))

        decision_ms = action.get("decision_time_ms")
        if decision_ms is not None:
            decision_times.append(float(decision_ms))
        gap_ms = action.get("time_since_last_action_ms")
        if gap_ms is not None:
            gap_times.append(float(gap_ms))

    phases = [str(a.get("phase") or "") for a in actions]
    phase_set = set(phases)
    types = [str(a.get("action_type") or "") for a in actions]
    bigrams = list(zip(types[:-1], types[1:]))
    bigram_counts = Counter(bigrams)

    snap_uniq = 0.0
    if bet_pot_fracs:
        snap = [round(v * 20) / 20 for v in bet_pot_fracs]
        snap_uniq = float(len(set(snap)))

    decision_mean = float(np.mean(decision_times)) if decision_times else 0.0
    decision_std = float(np.std(decision_times)) if len(decision_times) > 1 else 0.0
    decision_cv = decision_std / (decision_mean + 1e-6) if decision_mean else 0.0

    gap_mean = float(np.mean(gap_times)) if gap_times else 0.0
    gap_std = float(np.std(gap_times)) if len(gap_times) > 1 else 0.0
    gap_cv = gap_std / (gap_mean + 1e-6) if gap_mean else 0.0

    return {
        "h_n_actions": float(n_actions),
        "h_max_active_players": float(max(active_players)) if active_players else 0.0,
        "h_n_phases": float(len(phase_set)),
        "h_ratio_fold": counts.get("fold", 0) / max(n_actions, 1),
        "h_ratio_check": counts.get("check", 0) / max(n_actions, 1),
        "h_ratio_call": counts.get("call", 0) / max(n_actions, 1),
        "h_ratio_bet": counts.get("bet", 0) / max(n_actions, 1),
        "h_ratio_raise": counts.get("raise", 0) / max(n_actions, 1),
        "h_ratio_all_in": counts.get("all_in", 0) / max(n_actions, 1),
        "h_agg_ratio": agg / max(agg + passive, 1),
        "h_betpot_mean": float(np.mean(bet_pot_fracs)) if bet_pot_fracs else 0.0,
        "h_betpot_std": float(np.std(bet_pot_fracs)) if len(bet_pot_fracs) > 1 else 0.0,
        "h_betpot_snap_uniq": snap_uniq,
        "h_stackpot_mean": float(np.mean(stack_pot_ratios)) if stack_pot_ratios else 0.0,
        "h_bigram_uniq": float(len(bigram_counts)),
        "h_bigram_ent": _entropy(bigram_counts),
        "h_reached_flop": 1.0 if "FLOP" in phase_set or "flop" in phase_set else 0.0,
        "h_reached_turn": 1.0 if "TURN" in phase_set or "turn" in phase_set else 0.0,
        "h_reached_river": 1.0 if "RIVER" in phase_set or "river" in phase_set else 0.0,
        "h_decision_ms_mean": decision_mean,
        "h_decision_ms_cv": decision_cv,
        "h_gap_ms_mean": gap_mean,
        "h_gap_ms_cv": gap_cv,
        "h_heuristic": _hand_heuristic_score(hand),
    }


def _telemetry_feature_dict(
    telemetry: dict[str, Any], *, n_hands: int, hand_decision_means: list[float]
) -> dict[str, float]:
    events = [e for e in (telemetry.get("events") or []) if isinstance(e, dict)]
    summary = telemetry.get("summary") or {}
    n_actions = sum(1 for _ in hand_decision_means) or 1

    event_type_counts = Counter(str(e.get("event_type") or "") for e in events)
    target_counts = Counter(str(e.get("target_category") or "null") for e in events)

    offsets = sorted(float(e.get("offset_ms") or 0.0) for e in events)
    gaps = [b - a for a, b in zip(offsets[:-1], offsets[1:])]
    gap_mean = float(np.mean(gaps)) if gaps else 0.0
    gap_std = float(np.std(gaps)) if len(gaps) > 1 else 0.0
    gap_cv = gap_std / (gap_mean + 1e-6) if gap_mean else 0.0

    xy_buckets = Counter()
    for e in events:
        value = e.get("value") or {}
        if not isinstance(value, dict):
            continue
        xb = value.get("x_bucket")
        yb = value.get("y_bucket")
        if xb is not None or yb is not None:
            xy_buckets[(xb, yb)] += 1

    n_events = max(1, len(events))
    duration_ms = float(summary.get("duration_ms") or 0.0)
    decision_count = float(summary.get("decision_count") or 0.0)
    decision_mean_ms = float(summary.get("decision_mean_ms") or 0.0)
    decision_std_ms = float(summary.get("decision_std_ms") or 0.0)
    decision_cv = decision_std_ms / (decision_mean_ms + 1e-6) if decision_mean_ms else 0.0

    # Cross-check: the platform-computed telemetry summary's decision mean
    # should roughly track the mean of hand-level decision_time_ms we derive
    # ourselves. A large mismatch is a data-quality/consistency signal.
    own_mean = float(np.mean(hand_decision_means)) if hand_decision_means else 0.0
    mismatch = 0.0
    if decision_mean_ms > 0.0 and own_mean > 0.0:
        mismatch = abs(decision_mean_ms - own_mean) / max(decision_mean_ms, own_mean)

    return {
        "t_events_per_action": len(events) / n_actions,
        "t_offset_gap_cv": gap_cv,
        "t_offset_gap_mean_ms": gap_mean,
        "t_event_type_entropy": _entropy(event_type_counts),
        "t_target_entropy": _entropy(target_counts),
        "t_poker_action_share": target_counts.get("poker_action", 0) / n_events,
        "t_pointer_move_share": event_type_counts.get("pointer_move", 0) / n_events,
        "t_click_share": event_type_counts.get("click", 0) / n_events,
        "t_scroll_share": event_type_counts.get("scroll", 0) / n_events,
        "t_visibility_share": event_type_counts.get("visibility", 0) / n_events,
        "t_xy_bucket_entropy": _entropy(xy_buckets),
        "t_duration_per_hand_ms": duration_ms / max(1, n_hands),
        "t_decision_count_ratio": decision_count / max(1, n_hands),
        "t_decision_mean_ms": decision_mean_ms,
        "t_decision_std_ms": decision_std_ms,
        "t_decision_cv": decision_cv,
        "t_decision_mismatch": mismatch,
    }


def session_features(session: dict[str, Any]) -> np.ndarray:
    """Return the feature vector for one subject-session v2 record."""
    hands = [h for h in (session.get("hands") or []) if isinstance(h, dict)]
    if not hands:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    hand_dicts = [_hand_feature_dict(hand) for hand in hands]
    matrix = np.asarray(
        [[hd.get(key, 0.0) for key in HAND_KEYS] for hd in hand_dicts],
        dtype=np.float64,
    )

    aggregated: dict[str, float] = {}
    for index, key in enumerate(HAND_KEYS):
        column = matrix[:, index]
        aggregated[f"agg_mean_{key}"] = float(np.mean(column))
        aggregated[f"agg_std_{key}"] = float(np.std(column))
        aggregated[f"agg_q25_{key}"] = float(np.quantile(column, 0.25))
        aggregated[f"agg_q75_{key}"] = float(np.quantile(column, 0.75))
        aggregated[f"agg_max_{key}"] = float(np.max(column))

    aggregated["n_hands"] = float(len(hands))

    decision_means = [hd["h_decision_ms_mean"] for hd in hand_dicts if hd["h_decision_ms_mean"] > 0]
    if len(decision_means) > 1:
        mean_of_means = float(np.mean(decision_means))
        aggregated["cross_decision_cv_of_means"] = (
            float(np.std(decision_means)) / (mean_of_means + 1e-6) if mean_of_means else 0.0
        )
    else:
        aggregated["cross_decision_cv_of_means"] = 0.0

    telemetry = session.get("telemetry") or {}
    aggregated.update(
        _telemetry_feature_dict(
            telemetry, n_hands=len(hands), hand_decision_means=decision_means
        )
    )

    return np.asarray([aggregated[name] for name in FEATURE_NAMES], dtype=np.float32)


def sessions_to_matrix(sessions: list[dict[str, Any]]) -> np.ndarray:
    if not sessions:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.vstack([session_features(s) for s in sessions])
