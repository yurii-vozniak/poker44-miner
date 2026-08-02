"""Feature engineering for Poker44 schema-v4.1 micro-sessions."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np

PHASES = ("preflop", "flop", "turn", "river")
POSITION_GROUPS = ("early", "late", "blinds")
PRESSURES = ("facing_bet", "no_call")
ACTION_TYPES = ("fold", "check", "call", "bet", "raise", "all_in")
SIZE_BUCKETS = (
    "not_applicable",
    "unknown",
    "third_pot_or_less",
    "half_pot",
    "three_quarter_pot",
    "pot",
    "overbet",
    "all_in",
)
MECHANICAL_SIZE_BUCKETS = frozenset({"half_pot", "three_quarter_pot", "pot"})


def _one_hot(value: str, categories: tuple[str, ...]) -> list[float]:
    return [1.0 if value == category else 0.0 for category in categories]


def _decision_features(decision: dict) -> dict[str, float]:
    feats: dict[str, float] = {}
    for prefix, value, categories in (
        ("phase", str(decision.get("phase") or ""), PHASES),
        ("pos", str(decision.get("position_group") or ""), POSITION_GROUPS),
        ("press", str(decision.get("pressure") or ""), PRESSURES),
        ("act", str(decision.get("action_type") or ""), ACTION_TYPES),
        ("size", str(decision.get("size_bucket") or ""), SIZE_BUCKETS),
    ):
        for index, category in enumerate(categories):
            feats[f"d_{prefix}_{index}"] = 1.0 if value == category else 0.0
    feats["d_all_in"] = 1.0 if decision.get("is_all_in") else 0.0
    feats["d_aggressive"] = 1.0 if str(decision.get("action_type") or "") in {
        "bet",
        "raise",
        "all_in",
    } else 0.0
    feats["d_passive"] = 1.0 if str(decision.get("action_type") or "") in {
        "check",
        "call",
    } else 0.0
    feats["d_fold"] = 1.0 if str(decision.get("action_type") or "") == "fold" else 0.0
    feats["d_postflop"] = 1.0 if str(decision.get("phase") or "") in {
        "flop",
        "turn",
        "river",
    } else 0.0
    feats["d_mechanical_size"] = (
        1.0 if str(decision.get("size_bucket") or "") in MECHANICAL_SIZE_BUCKETS else 0.0
    )
    return feats


def _session_aggregate_features(decisions: list[dict]) -> dict[str, float]:
    if not decisions:
        return {}

    n = len(decisions)
    action_types = [str(d.get("action_type") or "") for d in decisions]
    size_buckets = [str(d.get("size_bucket") or "") for d in decisions]
    phases = [str(d.get("phase") or "") for d in decisions]
    pressures = [str(d.get("pressure") or "") for d in decisions]

    counts = Counter(action_types)
    agg = counts["bet"] + counts["raise"] + counts["all_in"]
    passive = counts["check"] + counts["call"]
    folds = counts["fold"]

    bigrams = list(zip(action_types[:-1], action_types[1:]))
    bigram_counter = Counter(bigrams)
    if bigram_counter:
        probs = np.asarray(list(bigram_counter.values()), dtype=np.float64)
        probs = probs / probs.sum()
        bigram_ent = float(-np.sum(probs * np.log(probs + 1e-12)))
    else:
        bigram_ent = 0.0

    transition_counter: Counter = Counter()
    state_counter: Counter = Counter()
    for src, dst in bigrams:
        transition_counter[(src, dst)] += 1
        state_counter[src] += 1

    neg_log_likelihoods: list[float] = []
    if transition_counter and state_counter:
        for (src, dst), count in transition_counter.items():
            total = state_counter[src]
            prob = count / total
            neg_log_likelihoods.extend([-np.log(prob + 1e-12)] * count)
    markov_self_perplexity = float(np.mean(neg_log_likelihoods)) if neg_log_likelihoods else 0.0

    facing_bet = [d for d in decisions if str(d.get("pressure") or "") == "facing_bet"]
    facing_fold_rate = (
        sum(str(d.get("action_type") or "") == "fold" for d in facing_bet) / len(facing_bet)
        if facing_bet
        else 0.0
    )

    mechanical_frac = sum(bucket in MECHANICAL_SIZE_BUCKETS for bucket in size_buckets) / n
    overbet_frac = sum(bucket in {"overbet", "all_in"} for bucket in size_buckets) / n
    unique_actions = len(set(action_types))
    unique_sizes = len(set(size_buckets))
    phase_changes = sum(phases[i] != phases[i - 1] for i in range(1, n))

    return {
        "s_n_decisions": float(n),
        "s_agg_ratio": float(agg / max(n, 1)),
        "s_passive_ratio": float(passive / max(n, 1)),
        "s_fold_ratio": float(folds / max(n, 1)),
        "s_bigram_ent": bigram_ent,
        "s_bigram_uniq": float(len(bigram_counter)),
        "s_markov_self_perplexity": markov_self_perplexity,
        "s_facing_bet_fold_rate": float(facing_fold_rate),
        "s_mechanical_size_frac": float(mechanical_frac),
        "s_overbet_frac": float(overbet_frac),
        "s_action_uniq": float(unique_actions),
        "s_size_uniq": float(unique_sizes),
        "s_phase_changes": float(phase_changes),
        "s_facing_bet_frac": float(sum(p == "facing_bet" for p in pressures) / n),
        "s_postflop_frac": float(sum(p in {"flop", "turn", "river"} for p in phases) / n),
        "s_all_in_frac": float(sum(bool(d.get("is_all_in")) for d in decisions) / n),
    }


def _feature_names() -> list[str]:
    names: list[str] = []
    sample_decision = {
        "phase": "preflop",
        "position_group": "early",
        "pressure": "no_call",
        "action_type": "check",
        "size_bucket": "not_applicable",
        "is_all_in": False,
    }
    per_decision_keys = list(_decision_features(sample_decision).keys())
    for slot in range(4):
        names.extend(f"slot{slot}_{key}" for key in per_decision_keys)
    names.extend(_session_aggregate_features([sample_decision] * 4).keys())
    return names


FEATURE_NAMES = _feature_names()


def session_features(session: dict) -> np.ndarray:
    decisions = [
        decision for decision in (session.get("decisions") or []) if isinstance(decision, dict)
    ]
    feats: dict[str, float] = {}
    for slot in range(4):
        if slot < len(decisions):
            decision_feats = _decision_features(decisions[slot])
        else:
            decision_feats = {key: 0.0 for key in _decision_features({}).keys()}
        for key, value in decision_feats.items():
            feats[f"slot{slot}_{key}"] = value
    feats.update(_session_aggregate_features(decisions))
    return np.asarray([feats[name] for name in FEATURE_NAMES], dtype=np.float32)


def sessions_to_matrix(sessions: Iterable[dict]) -> np.ndarray:
    return np.vstack([session_features(session) for session in sessions])
