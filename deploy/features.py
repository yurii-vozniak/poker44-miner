"""Feature extraction for Poker44 hand chunks."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import numpy as np

STREET_ORDER = ("preflop", "flop", "turn", "river")

FEATURE_NAMES = [
    "mean_actions",
    "std_actions",
    "mean_call_ratio",
    "mean_fold_ratio",
    "mean_raise_ratio",
    "mean_check_ratio",
    "mean_bet_ratio",
    "mean_street_depth",
    "showdown_rate",
    "mean_players",
    "std_players",
    "mean_bet_amount_bb",
    "mean_raise_amount_bb",
    "mean_preflop_share",
    "mean_flop_share",
    "mean_turn_share",
    "mean_river_share",
    "mean_pot_before_bb",
    "mean_pot_growth_bb",
    "mean_action_entropy",
]


def hand_features(hand: dict) -> list[float]:
    actions = hand.get("actions") or []
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    outcome = hand.get("outcome") or {}

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
    showdown_flag = 1.0 if outcome.get("showdown") else 0.0
    player_count = float(len(players))

    bet_amounts = [
        float(action.get("normalized_amount_bb", 0.0) or 0.0)
        for action in actions
        if action.get("action_type") == "bet"
    ]
    raise_amounts = [
        float(action.get("normalized_amount_bb", 0.0) or 0.0)
        for action in actions
        if action.get("action_type") == "raise"
    ]

    street_counts = Counter(str(action.get("street") or "").lower() for action in actions)
    street_actions = max(1, sum(street_counts.values()))
    street_shares = [
        street_counts.get(street, 0) / street_actions for street in STREET_ORDER
    ]

    pot_before_values = [
        float(action.get("pot_before", 0.0) or 0.0)
        for action in actions
        if action.get("pot_before") is not None
    ]
    pot_growth_values = [
        float(action.get("pot_after", 0.0) or 0.0) - float(action.get("pot_before", 0.0) or 0.0)
        for action in actions
        if action.get("pot_before") is not None and action.get("pot_after") is not None
    ]

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
        showdown_flag,
        player_count,
        float(np.mean(bet_amounts)) if bet_amounts else 0.0,
        float(np.mean(raise_amounts)) if raise_amounts else 0.0,
        street_shares[0],
        street_shares[1],
        street_shares[2],
        street_shares[3],
        float(np.mean(pot_before_values)) if pot_before_values else 0.0,
        float(np.mean(pot_growth_values)) if pot_growth_values else 0.0,
        entropy,
    ]


def chunk_features(chunk: list[dict]) -> np.ndarray:
    if not chunk:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    rows = np.asarray([hand_features(hand) for hand in chunk], dtype=np.float32)
    per_hand_actions = rows[:, 0]

    return np.array(
        [
            per_hand_actions.mean(),
            per_hand_actions.std(ddof=0),
            rows[:, 1].mean(),
            rows[:, 2].mean(),
            rows[:, 3].mean(),
            rows[:, 4].mean(),
            rows[:, 5].mean(),
            rows[:, 6].mean(),
            rows[:, 7].mean(),
            rows[:, 8].mean(),
            rows[:, 8].std(ddof=0),
            rows[:, 9].mean(),
            rows[:, 10].mean(),
            rows[:, 11].mean(),
            rows[:, 12].mean(),
            rows[:, 13].mean(),
            rows[:, 14].mean(),
            rows[:, 15].mean(),
            rows[:, 16].mean(),
            rows[:, 17].mean(),
        ],
        dtype=np.float32,
    )


def chunks_to_matrix(chunks: Iterable[list[dict]]) -> np.ndarray:
    return np.vstack([chunk_features(chunk) for chunk in chunks])
