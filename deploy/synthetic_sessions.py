"""Synthetic subject-session v2 generator for offline calibration and testing.

There is no public v3.0 benchmark endpoint yet (the platform's own evaluation
windows are validator-signed and not accessible to miners), so real labeled
session+telemetry data does not exist for us to train on before launch. This
module encodes reasoned assumptions about how scripted/automated play differs
from human play, purely to (a) sanity-check that `session_features` behaves
sensibly on both ends of the spectrum, and (b) fit a rough isotonic
calibrator so raw heuristic scores land in a plausible probability range
ahead of the first real evaluation-window feedback.

IMPORTANT: this is a *placeholder* prior, not training data. Once v3.0 goes
live and we start observing our own composite scores on the dashboard, this
calibration should be replaced/refined using that real feedback signal, the
same way max_pos_frac was re-tuned for v2 using live dashboard results.
"""

from __future__ import annotations

import numpy as np

ACTION_TYPES = ("fold", "check", "call", "bet", "raise", "all_in")
PHASES = ("PREFLOP", "FLOP", "TURN", "RIVER")


def _make_hand(rng: np.random.Generator, *, hand_number: int, bot: bool) -> dict:
    n_actions = rng.integers(2, 6)
    actions = []
    pot = float(rng.integers(50, 300))
    stack = float(rng.integers(5000, 20000))

    if bot:
        # Mechanical: near-constant decision time, canonical pot fractions,
        # tight bet-size vocabulary.
        base_decision = float(rng.choice([350.0, 500.0, 800.0]))
        decision_jitter = 15.0
        pot_fracs = rng.choice([0.5, 0.66, 1.0], size=n_actions)
    else:
        base_decision = float(rng.uniform(600.0, 3500.0))
        decision_jitter = base_decision * float(rng.uniform(0.35, 0.9))
        pot_fracs = rng.uniform(0.15, 1.4, size=n_actions)

    for i in range(n_actions):
        action_type = str(rng.choice(ACTION_TYPES, p=[0.22, 0.2, 0.28, 0.12, 0.15, 0.03]))
        phase = PHASES[min(i, len(PHASES) - 1)]
        amount = None
        raise_to = None
        call_amount = None
        if action_type in ("bet", "raise", "all_in"):
            amount = max(1.0, pot * float(pot_fracs[i % len(pot_fracs)]))
            raise_to = amount
        elif action_type == "call":
            call_amount = max(1.0, pot * 0.3)

        decision_ms = max(50.0, rng.normal(base_decision, decision_jitter))
        gap_ms = max(10.0, rng.normal(base_decision * 1.3, decision_jitter))

        actions.append(
            {
                "sequence": i,
                "event_type": f"PLAYER_{action_type.upper()}",
                "action_type": action_type,
                "phase": phase,
                "amount": amount,
                "call_amount": call_amount,
                "raise_to": raise_to,
                "is_all_in": action_type == "all_in",
                "pot_size": int(pot),
                "current_bet": int(pot * 0.1),
                "player_stack": int(stack),
                "active_players": int(rng.integers(2, 7)),
                "seat_position": int(rng.integers(0, 6)),
                "position_name": "MP",
                "community_cards": [],
                "hole_cards": None,
                "decision_time_ms": int(decision_ms),
                "time_since_last_action_ms": int(gap_ms),
                "session_offset_ms": i * 1000,
            }
        )
        pot += amount or 0.0
    return {"hand_number": hand_number, "actions": actions}


def _make_telemetry(rng: np.random.Generator, *, n_hands: int, bot: bool) -> dict:
    n_actions_est = n_hands * 4
    if bot:
        n_events = int(max(0, rng.integers(0, 3)) * n_actions_est / 4)
        gap_mean = 400.0
        gap_jitter = 20.0
        xy_pool = [(1, 1), (2, 2)]
    else:
        n_events = int(n_actions_est * rng.uniform(1.5, 5.0))
        gap_mean = float(rng.uniform(150.0, 900.0))
        gap_jitter = gap_mean * float(rng.uniform(0.4, 1.1))
        xy_pool = [(x, y) for x in range(8) for y in range(8)]

    events = []
    offset = 0.0
    event_types = ["click", "pointer_down", "pointer_move", "scroll", "focus_in", "visibility"]
    weights_bot = [0.35, 0.1, 0.35, 0.05, 0.1, 0.05]
    weights_human = [0.15, 0.1, 0.45, 0.1, 0.1, 0.1]
    for i in range(max(0, n_events)):
        offset += max(5.0, rng.normal(gap_mean, gap_jitter))
        etype = str(rng.choice(event_types, p=weights_bot if bot else weights_human))
        value = {}
        if etype in ("click", "pointer_down", "pointer_move"):
            x, y = xy_pool[rng.integers(0, len(xy_pool))]
            value = {"x_bucket": x, "y_bucket": y}
            if etype == "click":
                value["button"] = 0
        events.append(
            {
                "sequence": i,
                "offset_ms": int(offset),
                "event_type": etype,
                "target_category": str(rng.choice(["poker_action", "navigation", "control", "other"])),
                "value": value,
            }
        )

    decisions = [float(rng.normal(500 if bot else 1800, 20 if bot else 900)) for _ in range(n_hands * 3)]
    decisions = [max(50.0, d) for d in decisions]
    return {
        "events": events,
        "summary": {
            "event_count": len(events),
            "action_count": n_actions_est,
            "duration_ms": int(offset),
            "decision_count": len(decisions),
            "decision_mean_ms": float(np.mean(decisions)),
            "decision_std_ms": float(np.std(decisions)),
        },
    }


def make_session(rng: np.random.Generator, *, session_id: str, bot: bool) -> dict:
    n_hands = int(rng.integers(1, 4))
    hands = [_make_hand(rng, hand_number=i + 1, bot=bot) for i in range(n_hands)]
    telemetry = _make_telemetry(rng, n_hands=n_hands, bot=bot)
    return {
        "schema_version": "2",
        "session_id": session_id,
        "window_id": "synthetic-window",
        "hands": hands,
        "telemetry": telemetry,
    }


def make_dataset(n_per_class: int = 500, seed: int = 42) -> tuple[list[dict], np.ndarray]:
    rng = np.random.default_rng(seed)
    sessions: list[dict] = []
    labels: list[int] = []
    for i in range(n_per_class):
        sessions.append(make_session(rng, session_id=f"bot_{i}", bot=True))
        labels.append(1)
        sessions.append(make_session(rng, session_id=f"human_{i}", bot=False))
        labels.append(0)
    order = rng.permutation(len(sessions))
    sessions = [sessions[i] for i in order]
    labels_arr = np.asarray(labels, dtype=int)[order]
    return sessions, labels_arr
