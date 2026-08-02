"""Synthetic schema-v4.1 micro-sessions for offline model bootstrapping."""

from __future__ import annotations

import random
import uuid
from typing import Iterator

PHASES = ["preflop", "flop", "turn", "river"]
POSITION_GROUPS = ["early", "late", "blinds"]
ACTIONS = ["fold", "check", "call", "bet", "raise", "all_in"]
SIZE_BUCKETS = [
    "not_applicable",
    "unknown",
    "third_pot_or_less",
    "half_pot",
    "three_quarter_pot",
    "pot",
    "overbet",
    "all_in",
]


def _make_decision(
    *,
    decision_number: int,
    phase: str,
    position_group: str,
    pressure: str,
    action_type: str,
    size_bucket: str,
    is_all_in: bool,
) -> dict:
    return {
        "decision_number": decision_number,
        "phase": phase,
        "position_group": position_group,
        "pressure": pressure,
        "action_type": action_type,
        "size_bucket": size_bucket,
        "is_all_in": is_all_in,
    }


def _human_session(rng: random.Random, window_id: str) -> dict:
    phases = ["preflop", "flop", "turn", "river"]
    rng.shuffle(phases)
    phase_seq = [phases[0], phases[1], phases[2], "flop"]
    decisions = []
    for index, phase in enumerate(phase_seq, start=1):
        pressure = rng.choice(["facing_bet", "no_call"])
        action = rng.choices(
            ACTIONS,
            weights=[0.18, 0.24, 0.24, 0.12, 0.10, 0.12],
        )[0]
        if action in {"check", "fold"}:
            size = "not_applicable"
            all_in = False
        else:
            size = rng.choices(
                SIZE_BUCKETS[2:],
                weights=[0.12, 0.18, 0.16, 0.18, 0.14, 0.22],
            )[0]
            all_in = action == "all_in" or size == "all_in"
        decisions.append(
            _make_decision(
                decision_number=index,
                phase=phase,
                position_group=rng.choice(POSITION_GROUPS),
                pressure=pressure,
                action_type=action,
                size_bucket=size,
                is_all_in=all_in,
            )
        )
    return {
        "schema_version": "4.1",
        "item_id": str(uuid.uuid4()),
        "window_id": window_id,
        "decisions": decisions,
    }


def _bot_session(rng: random.Random, window_id: str) -> dict:
    template = rng.choice(
        [
            ["check", "bet", "call", "fold"],
            ["fold", "check", "bet", "fold"],
            ["call", "raise", "fold", "check"],
            ["bet", "bet", "fold", "check"],
        ]
    )
    mechanical_size = rng.choice(["half_pot", "three_quarter_pot", "pot"])
    phases = ["preflop", "flop", "turn", "river"]
    decisions = []
    for index, (phase, action) in enumerate(zip(phases, template), start=1):
        pressure = "facing_bet" if action in {"fold", "call", "raise"} else "no_call"
        if action in {"check", "fold"}:
            size = "not_applicable"
            all_in = False
        elif action == "all_in":
            size = "all_in"
            all_in = True
        else:
            size = mechanical_size if rng.random() < 0.78 else rng.choice(SIZE_BUCKETS[2:])
            all_in = size == "all_in"
        decisions.append(
            _make_decision(
                decision_number=index,
                phase=phase,
                position_group=rng.choice(["early", "late"]),
                pressure=pressure,
                action_type=action,
                size_bucket=size,
                is_all_in=all_in,
            )
        )
    return {
        "schema_version": "4.1",
        "item_id": str(uuid.uuid4()),
        "window_id": window_id,
        "decisions": decisions,
    }


def iter_labeled_sessions(
    *,
    n_sessions: int = 4000,
    bot_rate: float = 0.5,
    seed: int = 42,
) -> Iterator[tuple[dict, int]]:
    rng = random.Random(seed)
    window_id = "synthetic-window-v1"
    for _ in range(n_sessions):
        label = 1 if rng.random() < bot_rate else 0
        session = _bot_session(rng, window_id) if label else _human_session(rng, window_id)
        yield session, label


def make_dataset(*, n_sessions: int = 4000, bot_rate: float = 0.5, seed: int = 42):
    sessions: list[dict] = []
    labels: list[int] = []
    for session, label in iter_labeled_sessions(
        n_sessions=n_sessions,
        bot_rate=bot_rate,
        seed=seed,
    ):
        sessions.append(session)
        labels.append(label)
    return sessions, labels
