"""Rich feature engineering for Poker44 chunk-level bot detection.

Original implementation aligned with validator-visible payloads. Patterns inspired by
public SN126 research (cross-hand consistency, pot-fraction grids, action n-grams)
but implemented independently for this miner.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np

from poker44.validator.payload_view import payload_chunk_signature, prepare_hand_for_miner

ACTION_TYPES = ("call", "check", "bet", "raise", "fold")
STREETS = ("preflop", "flop", "turn", "river")
AGG_ACTIONS = frozenset({"bet", "raise"})
PASSIVE_ACTIONS = frozenset({"call", "check"})
VISIBLE_BB = 0.02
COMMON_POT_FRACS = (0.5, 0.66, 0.75, 1.0)

HAND_KEYS = [
    "h_n_actions",
    "h_n_players",
    "h_n_streets",
    "h_max_seats",
    "h_ratio_call",
    "h_ratio_check",
    "h_ratio_bet",
    "h_ratio_raise",
    "h_ratio_fold",
    "h_agg_ratio",
    "h_amt_max_bb",
    "h_amt_mean_nz_bb",
    "h_amt_std_nz_bb",
    "h_potfrac_mean",
    "h_potfrac_std",
    "h_potfrac_snap_uniq",
    "h_bigram_uniq",
    "h_actor_uniq",
    "h_pot_growth_bb",
    "h_hero_stack_bb",
    "h_reached_flop",
    "h_reached_turn",
    "h_reached_river",
    "h_heuristic",
]

AGG_SUFFIXES = ("mean", "std", "q25", "q75", "max")
FEATURE_NAMES = [
    *[f"agg_{suffix}_{key}" for key in HAND_KEYS for suffix in AGG_SUFFIXES],
    "c_n_hands",
    "c_decision_ent_mean",
    "c_decision_ent_std",
    "c_decision_buckets",
    "c_betbb_cv",
    "c_betbb_uniq_round",
    "c_betbb_p90",
    "c_potfrac_cv",
    "c_potfrac_snap_uniq",
    "c_potfrac_near_050",
    "c_potfrac_near_066",
    "c_potfrac_near_075",
    "c_potfrac_near_100",
    "c_bigram_ent",
    "c_bigram_uniq",
    "c_hand_heuristic_std",
    "c_hand_heuristic_max",
    "c_hand_heuristic_p75",
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


def _heuristic_score(hand: dict) -> float:
    actions = hand.get("actions") or []
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    counts = Counter(a.get("action_type") for a in actions)
    meaningful = max(
        1,
        sum(counts.get(kind, 0) for kind in ("call", "check", "bet", "raise", "fold")),
    )
    call_ratio = counts.get("call", 0) / meaningful
    check_ratio = counts.get("check", 0) / meaningful
    fold_ratio = counts.get("fold", 0) / meaningful
    raise_ratio = counts.get("raise", 0) / meaningful
    street_depth = len(streets) / 3.0
    player_signal = (6 - min(len(players), 6)) / 4.0 if players else 0.0
    score = (
        0.32 * street_depth
        + 0.18 * _clamp01(call_ratio / 0.35)
        + 0.12 * _clamp01(check_ratio / 0.30)
        + 0.08 * _clamp01(player_signal)
        - 0.18 * _clamp01(fold_ratio / 0.55)
        - 0.10 * _clamp01(raise_ratio / 0.20)
    )
    return _clamp01(score)


def _hand_feature_dict(hand: dict) -> dict[str, float]:
    actions = hand.get("actions") or []
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    metadata = hand.get("metadata") or {}

    counts = Counter(a.get("action_type") for a in actions)
    n_actions = len(actions)
    agg = sum(counts.get(a, 0) for a in AGG_ACTIONS)
    passive = sum(counts.get(a, 0) for a in PASSIVE_ACTIONS)

    amts_bb = [float(a.get("normalized_amount_bb") or 0.0) for a in actions]
    nonzero = [value for value in amts_bb if value > 0.0]

    pot_fracs: list[float] = []
    for action in actions:
        amount = float(action.get("normalized_amount_bb") or 0.0)
        pot_before = float(action.get("pot_before") or 0.0)
        pot_bb = pot_before / VISIBLE_BB if pot_before > 0 else 0.0
        if amount > 0 and pot_bb > 0:
            pot_fracs.append(amount / pot_bb)

    types = [a.get("action_type") or "" for a in actions]
    bigrams = list(zip(types[:-1], types[1:]))
    actors = {a.get("actor_seat") for a in actions if a.get("actor_seat")}

    streets_set = {s.get("street") for s in streets if isinstance(s, dict)}
    hero_seat = int(metadata.get("hero_seat") or 0)
    hero_stack_bb = 0.0
    for player in players:
        if int(player.get("seat") or 0) == hero_seat:
            hero_stack_bb = float(player.get("starting_stack") or 0.0) / VISIBLE_BB
            break

    if actions:
        first_pot = float(actions[0].get("pot_before") or 0.0) / VISIBLE_BB
        last_pot = float(actions[-1].get("pot_after") or 0.0) / VISIBLE_BB
        pot_growth_bb = last_pot - first_pot
    else:
        pot_growth_bb = 0.0

    snap_uniq = 0.0
    if pot_fracs:
        snap = [round(value * 20) / 20 for value in pot_fracs]
        snap_uniq = float(len(set(snap)))

    return {
        "h_n_actions": float(n_actions),
        "h_n_players": float(len(players)),
        "h_n_streets": float(len(streets)),
        "h_max_seats": float(metadata.get("max_seats") or 0),
        "h_ratio_call": float(counts.get("call", 0) / max(n_actions, 1)),
        "h_ratio_check": float(counts.get("check", 0) / max(n_actions, 1)),
        "h_ratio_bet": float(counts.get("bet", 0) / max(n_actions, 1)),
        "h_ratio_raise": float(counts.get("raise", 0) / max(n_actions, 1)),
        "h_ratio_fold": float(counts.get("fold", 0) / max(n_actions, 1)),
        "h_agg_ratio": float(agg / max(agg + passive, 1)),
        "h_amt_max_bb": float(max(amts_bb)) if amts_bb else 0.0,
        "h_amt_mean_nz_bb": float(np.mean(nonzero)) if nonzero else 0.0,
        "h_amt_std_nz_bb": float(np.std(nonzero)) if len(nonzero) > 1 else 0.0,
        "h_potfrac_mean": float(np.mean(pot_fracs)) if pot_fracs else 0.0,
        "h_potfrac_std": float(np.std(pot_fracs)) if len(pot_fracs) > 1 else 0.0,
        "h_potfrac_snap_uniq": snap_uniq,
        "h_bigram_uniq": float(len(set(bigrams))),
        "h_actor_uniq": float(len(actors)),
        "h_pot_growth_bb": float(pot_growth_bb),
        "h_hero_stack_bb": float(hero_stack_bb),
        "h_reached_flop": 1.0 if "flop" in streets_set else 0.0,
        "h_reached_turn": 1.0 if "turn" in streets_set else 0.0,
        "h_reached_river": 1.0 if "river" in streets_set else 0.0,
        "h_heuristic": _heuristic_score(hand),
    }


def _chunk_consistency_features(chunk: list[dict]) -> dict[str, float]:
    decisions: dict[tuple[str, int], list[str]] = {}
    pot_fracs: list[float] = []
    bet_sizes_bb: list[float] = []
    bigram_counter: Counter = Counter()
    heuristic_scores: list[float] = []

    for hand in chunk:
        heuristic_scores.append(_heuristic_score(hand))
        types = [a.get("action_type") or "" for a in hand.get("actions") or []]
        for bigram in zip(types[:-1], types[1:]):
            bigram_counter[bigram] += 1
        for action in hand.get("actions") or []:
            street = str(action.get("street") or "")
            seat = int(action.get("actor_seat") or 0)
            action_type = str(action.get("action_type") or "")
            decisions.setdefault((street, seat), []).append(action_type)
            amount = float(action.get("normalized_amount_bb") or 0.0)
            pot_before = float(action.get("pot_before") or 0.0)
            pot_bb = pot_before / VISIBLE_BB if pot_before > 0 else 0.0
            if amount > 0:
                bet_sizes_bb.append(amount)
            if amount > 0 and pot_bb > 0:
                pot_fracs.append(amount / pot_bb)

    entropies: list[float] = []
    for action_list in decisions.values():
        if len(action_list) <= 1:
            continue
        counts = Counter(action_list)
        total = sum(counts.values())
        probs = np.asarray(list(counts.values()), dtype=np.float64) / total
        entropies.append(float(-np.sum(probs * np.log(probs + 1e-12))))

    if bigram_counter:
        counts = np.asarray(list(bigram_counter.values()), dtype=np.float64)
        probs = counts / counts.sum()
        bigram_ent = float(-np.sum(probs * np.log(probs + 1e-12)))
    else:
        bigram_ent = 0.0

    feats: dict[str, float] = {
        "c_n_hands": float(len(chunk)),
        "c_decision_ent_mean": float(np.mean(entropies)) if entropies else 0.0,
        "c_decision_ent_std": float(np.std(entropies)) if len(entropies) > 1 else 0.0,
        "c_decision_buckets": float(len(decisions)),
        "c_bigram_ent": bigram_ent,
        "c_bigram_uniq": float(len(bigram_counter)),
        "c_hand_heuristic_std": float(np.std(heuristic_scores)) if heuristic_scores else 0.0,
        "c_hand_heuristic_max": float(np.max(heuristic_scores)) if heuristic_scores else 0.0,
        "c_hand_heuristic_p75": float(np.percentile(heuristic_scores, 75)) if heuristic_scores else 0.0,
    }

    if bet_sizes_bb:
        bet_arr = np.asarray(bet_sizes_bb, dtype=np.float64)
        feats["c_betbb_cv"] = float(np.std(bet_arr) / (np.mean(bet_arr) + 1e-9))
        feats["c_betbb_uniq_round"] = float(len({round(value, 1) for value in bet_arr}))
        feats["c_betbb_p90"] = float(np.quantile(bet_arr, 0.9))
    else:
        feats["c_betbb_cv"] = 0.0
        feats["c_betbb_uniq_round"] = 0.0
        feats["c_betbb_p90"] = 0.0

    if pot_fracs:
        pot_arr = np.asarray(pot_fracs, dtype=np.float64)
        feats["c_potfrac_cv"] = float(np.std(pot_arr) / (np.mean(pot_arr) + 1e-9))
        snap = [round(value * 20) / 20 for value in pot_arr]
        feats["c_potfrac_snap_uniq"] = float(len(set(snap)))
        for target in COMMON_POT_FRACS:
            key = f"c_potfrac_near_{int(target * 100):03d}"
            feats[key] = float(np.mean([abs(value - target) < 0.07 for value in pot_arr]))
    else:
        feats["c_potfrac_cv"] = 0.0
        feats["c_potfrac_snap_uniq"] = 0.0
        for target in COMMON_POT_FRACS:
            feats[f"c_potfrac_near_{int(target * 100):03d}"] = 0.0

    return feats


def canonicalize_chunk(chunk: list[dict], *, for_training: bool) -> list[dict]:
    if not for_training:
        return chunk
    return [prepare_hand_for_miner(hand) for hand in chunk]


def chunk_features(chunk: list[dict], *, for_training: bool = False) -> np.ndarray:
    if not chunk:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    visible_chunk = canonicalize_chunk(chunk, for_training=for_training)
    hand_dicts = [_hand_feature_dict(hand) for hand in visible_chunk]
    matrix = np.asarray(
        [[hand_dict.get(key, 0.0) for key in HAND_KEYS] for hand_dict in hand_dicts],
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

    aggregated.update(_chunk_consistency_features(visible_chunk))
    signature = payload_chunk_signature(visible_chunk)
    sig_keys = FEATURE_NAMES[-9:]
    for key, value in zip(sig_keys, signature):
        aggregated[key] = float(value)

    return np.asarray([aggregated[name] for name in FEATURE_NAMES], dtype=np.float32)


def chunks_to_matrix(chunks: Iterable[list[dict]], *, for_training: bool = False) -> np.ndarray:
    return np.vstack([chunk_features(chunk, for_training=for_training) for chunk in chunks])
