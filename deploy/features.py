"""Feature extraction aligned with validator miner-visible payloads."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import numpy as np

from poker44.validator.payload_view import payload_chunk_signature, prepare_hand_for_miner

HAND_FEATURE_NAMES = [
    "actions",
    "call_ratio",
    "fold_ratio",
    "raise_ratio",
    "check_ratio",
    "bet_ratio",
    "street_depth",
    "player_count",
    "hero_action_share",
    "non_hero_action_share",
    "aggression_ratio",
    "passive_ratio",
    "mean_sizing_bb",
    "max_sizing_bb",
    "sizing_std_bb",
    "mean_pot_growth_bb",
    "action_entropy",
    "preflop_share",
    "flop_share",
    "turn_share",
    "river_share",
    "heuristic_score",
]

P75_FEATURE_NAMES = [
    "heuristic_score",
    "aggression_ratio",
    "raise_ratio",
    "fold_ratio",
    "passive_ratio",
    "max_sizing_bb",
]

FEATURE_NAMES = [
    *[f"{name}_{suffix}" for name in HAND_FEATURE_NAMES for suffix in ("mean", "std")],
    *[f"{name}_p75" for name in P75_FEATURE_NAMES],
    "heuristic_max",
    "heuristic_std",
    "sig_calls",
    "sig_checks",
    "sig_raises",
    "sig_folds",
    "sig_actions",
    "sig_streets",
    "sig_players",
    "sig_amount",
    "sig_pot_after",
]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _heuristic_hand_score(hand: dict) -> float:
    actions = hand.get("actions") or []
    players = hand.get("players") or []
    streets = hand.get("streets") or []

    action_counts = Counter(action.get("action_type") for action in actions)
    meaningful_actions = max(
        1,
        sum(action_counts.get(kind, 0) for kind in ("call", "check", "bet", "raise", "fold")),
    )

    call_ratio = action_counts.get("call", 0) / meaningful_actions
    check_ratio = action_counts.get("check", 0) / meaningful_actions
    fold_ratio = action_counts.get("fold", 0) / meaningful_actions
    raise_ratio = action_counts.get("raise", 0) / meaningful_actions
    street_depth = len(streets) / 3.0

    player_count_signal = 0.0
    if players:
        player_count_signal = (6 - min(len(players), 6)) / 4.0

    score = 0.0
    score += 0.32 * street_depth
    score += 0.18 * _clamp01(call_ratio / 0.35)
    score += 0.12 * _clamp01(check_ratio / 0.30)
    score += 0.08 * _clamp01(player_count_signal)
    score -= 0.18 * _clamp01(fold_ratio / 0.55)
    score -= 0.10 * _clamp01(raise_ratio / 0.20)
    return _clamp01(score)


def canonicalize_chunk(chunk: list[dict], *, for_training: bool) -> list[dict]:
    if not for_training:
        return chunk
    return [prepare_hand_for_miner(hand) for hand in chunk]


def hand_features(hand: dict) -> list[float]:
    actions = hand.get("actions") or []
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    metadata = hand.get("metadata") or {}

    action_counts = Counter(action.get("action_type") for action in actions)
    meaningful_actions = max(
        1,
        sum(action_counts.get(kind, 0) for kind in ("call", "check", "bet", "raise", "fold")),
    )

    call_ratio = action_counts.get("call", 0) / meaningful_actions
    check_ratio = action_counts.get("check", 0) / meaningful_actions
    fold_ratio = action_counts.get("fold", 0) / meaningful_actions
    raise_ratio = action_counts.get("raise", 0) / meaningful_actions
    bet_ratio = action_counts.get("bet", 0) / meaningful_actions
    street_depth = len(streets) / 3.0
    player_count = float(len(players))

    hero_seat = int(metadata.get("hero_seat", 0) or 0)
    hero_actions = sum(
        1 for action in actions if int(action.get("actor_seat", 0) or 0) == hero_seat
    )
    hero_action_share = hero_actions / max(len(actions), 1)
    non_hero_action_share = 1.0 - hero_action_share
    aggression_ratio = (action_counts.get("bet", 0) + action_counts.get("raise", 0)) / meaningful_actions
    passive_ratio = (action_counts.get("call", 0) + action_counts.get("check", 0)) / meaningful_actions

    sizing_values = [
        float(action.get("normalized_amount_bb", 0.0) or 0.0)
        for action in actions
        if action.get("action_type") in {"bet", "raise", "call"}
    ]
    pot_growth_values = [
        float(action.get("pot_after", 0.0) or 0.0) - float(action.get("pot_before", 0.0) or 0.0)
        for action in actions
        if action.get("pot_before") is not None and action.get("pot_after") is not None
    ]

    street_counts = Counter(str(action.get("street") or "").lower() for action in actions)
    street_actions = max(1, sum(street_counts.values()))
    preflop_share = street_counts.get("preflop", 0) / street_actions
    flop_share = street_counts.get("flop", 0) / street_actions
    turn_share = street_counts.get("turn", 0) / street_actions
    river_share = street_counts.get("river", 0) / street_actions

    entropy = 0.0
    for count in action_counts.values():
        if count <= 0:
            continue
        probability = count / meaningful_actions
        entropy -= probability * math.log(probability)

    return [
        float(len(actions)),
        call_ratio,
        fold_ratio,
        raise_ratio,
        check_ratio,
        bet_ratio,
        street_depth,
        player_count,
        hero_action_share,
        non_hero_action_share,
        aggression_ratio,
        passive_ratio,
        float(np.mean(sizing_values)) if sizing_values else 0.0,
        float(np.max(sizing_values)) if sizing_values else 0.0,
        float(np.std(sizing_values)) if sizing_values else 0.0,
        float(np.mean(pot_growth_values)) if pot_growth_values else 0.0,
        entropy,
        preflop_share,
        flop_share,
        turn_share,
        river_share,
        _heuristic_hand_score(hand),
    ]


def chunk_features(chunk: list[dict], *, for_training: bool = False) -> np.ndarray:
    if not chunk:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    visible_chunk = canonicalize_chunk(chunk, for_training=for_training)
    rows = np.asarray([hand_features(hand) for hand in visible_chunk], dtype=np.float32)
    name_to_idx = {name: idx for idx, name in enumerate(HAND_FEATURE_NAMES)}
    heuristic_scores = rows[:, name_to_idx["heuristic_score"]]

    p75_values = [
        float(np.percentile(rows[:, name_to_idx[name]], 75))
        for name in P75_FEATURE_NAMES
    ]
    signature = payload_chunk_signature(visible_chunk)

    aggregated = np.concatenate(
        [
            rows.mean(axis=0),
            rows.std(axis=0),
            np.asarray(p75_values, dtype=np.float32),
            np.array(
                [
                    float(np.max(heuristic_scores)),
                    float(np.std(heuristic_scores)),
                ],
                dtype=np.float32,
            ),
            np.asarray(signature, dtype=np.float32),
        ]
    )
    return aggregated.astype(np.float32)


def chunks_to_matrix(chunks: Iterable[list[dict]], *, for_training: bool = False) -> np.ndarray:
    return np.vstack([chunk_features(chunk, for_training=for_training) for chunk in chunks])
