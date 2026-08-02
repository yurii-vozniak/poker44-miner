"""Convert legacy labeled hand-chunk groups into schema-v4.1 micro-sessions.

There is no public labeled v4.1 corpus yet (retired chunk API, Jul 2026).
This proxy maps miner-visible hand actions to the four strategic decision
fields used in live evaluation. Labels come from chunk ``groundTruth``.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Any

PHASES = ("preflop", "flop", "turn", "river")
ACTION_TYPES = frozenset({"fold", "check", "call", "bet", "raise", "all_in"})
SIZE_THRESHOLDS = (
    (0.40, "third_pot_or_less"),
    (0.60, "half_pot"),
    (0.85, "three_quarter_pot"),
    (1.15, "pot"),
    (999.0, "overbet"),
)


def _position_group(hero_seat: int, button_seat: int, max_seats: int) -> str:
    if hero_seat <= 0:
        return "early"
    sb = (button_seat % max_seats) + 1 if button_seat else max_seats
    bb = (sb % max_seats) + 1
    if hero_seat in {sb, bb}:
        return "blinds"
    if hero_seat in {button_seat or 1, (button_seat - 1) if button_seat else max_seats}:
        return "late"
    return "early"


def _size_bucket(action: dict[str, Any], pot_before: float) -> str:
    action_type = str(action.get("action_type") or "")
    if action_type in {"check", "fold"}:
        return "not_applicable"
    amount = float(action.get("amount") or 0.0)
    if pot_before <= 0.0:
        return "unknown"
    ratio = amount / pot_before
    for threshold, bucket in SIZE_THRESHOLDS:
        if ratio <= threshold:
            return bucket
    return "overbet"


def _hero_decisions(hand: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = hand.get("metadata") if isinstance(hand.get("metadata"), dict) else {}
    hero_seat = int(metadata.get("hero_seat") or 0)
    button_seat = int(metadata.get("button_seat") or 0)
    max_seats = int(metadata.get("max_seats") or 6)
    actions = hand.get("actions") if isinstance(hand.get("actions"), list) else []

    decisions: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        if int(action.get("actor_seat") or 0) != hero_seat:
            continue

        street = str(action.get("street") or "preflop")
        if street not in PHASES:
            street = "preflop"
        prior = actions[:index]
        facing_bet = any(
            str(item.get("action_type") or "") in {"bet", "raise"}
            and str(item.get("street") or "") == street
            and int(item.get("actor_seat") or 0) != hero_seat
            for item in prior
            if isinstance(item, dict)
        )
        pot_before = float(action.get("pot_before") or 0.0)
        raw_type = str(action.get("action_type") or "")
        all_in = float(action.get("normalized_amount_bb") or 0.0) > 80.0
        action_type = raw_type
        if all_in and raw_type in {"bet", "raise", "call"}:
            action_type = "all_in"
        if action_type not in ACTION_TYPES:
            action_type = "call" if raw_type == "call" else "check"

        size = _size_bucket(action, pot_before)
        if action_type == "all_in":
            size = "all_in"

        decisions.append(
            {
                "phase": street,
                "position_group": _position_group(hero_seat, button_seat, max_seats),
                "pressure": "facing_bet" if facing_bet else "no_call",
                "action_type": action_type,
                "size_bucket": size,
                "is_all_in": action_type == "all_in" or size == "all_in",
            }
        )
    return decisions


def chunk_group_to_micro_session(
    chunk_group: list[dict[str, Any]],
    *,
    window_id: str = "legacy-chunk-proxy",
    item_id: str | None = None,
) -> dict[str, Any] | None:
    """Build one v4.1 session from a labeled legacy chunk group."""
    pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for hand in chunk_group:
        if not isinstance(hand, dict):
            continue
        for decision in _hero_decisions(hand):
            pool[str(decision["phase"])].append(decision)

    picks: list[dict[str, Any]] = []
    for phase in PHASES:
        if pool.get(phase):
            picks.append(pool[phase][0])
    if len(picks) < 4:
        remainder: list[dict[str, Any]] = []
        for phase in PHASES:
            remainder.extend(pool.get(phase, [])[1:])
        picks.extend(remainder[: 4 - len(picks)])
    if len(picks) < 4:
        return None

    decisions = [
        {"decision_number": index, **decision}
        for index, decision in enumerate(picks[:4], start=1)
    ]
    return {
        "schema_version": "4.1",
        "item_id": item_id or str(uuid.uuid4()),
        "window_id": window_id,
        "decisions": decisions,
    }
