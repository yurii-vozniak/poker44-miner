"""Fit a v3.0-ready session classifier on synthetic bot/human sessions.

No real subject-session v2 data is available to miners yet (see
`deploy/synthetic_sessions.py` for why). This script trains a first-pass
classifier purely on reasoned synthetic priors, wrapped with probability
calibration, as a working starting point that can be swapped/retrained the
moment real evaluation-window feedback becomes available after v3.0 ships.

Usage:
    python -m deploy.train_session_model
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split

from deploy.session_features import FEATURE_NAMES, sessions_to_matrix
from deploy.synthetic_sessions import make_dataset

MODEL_VERSION = "session-v1-synthetic"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "models" / "session_model_v1.joblib"


def main() -> None:
    sessions, labels = make_dataset(n_per_class=1500, seed=7)
    X = sessions_to_matrix(sessions)
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.25, random_state=7, stratify=labels
    )

    base = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.8,
        random_state=7,
    )
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)
    calibrated.fit(X_train, y_train)

    probs = calibrated.predict_proba(X_test)[:, 1]
    ap = average_precision_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    prevalence = float(np.mean(y_test))
    ap_skill = max(0.0, (ap - prevalence) / max(1e-12, 1.0 - prevalence))
    baseline_brier = float(np.mean(np.square(prevalence - y_test)))
    brier_skill = max(0.0, 1.0 - brier / max(1e-12, baseline_brier))

    print(f"Synthetic holdout AP={ap:.4f} (skill={ap_skill:.4f})")
    print(f"Synthetic holdout Brier={brier:.4f} (skill={brier_skill:.4f})")
    print(
        "NOTE: these numbers reflect separability of our synthetic prior, "
        "not real-world performance. Recalibrate once live v3.0 feedback exists."
    )

    final = CalibratedClassifierCV(base, method="isotonic", cv=3)
    final.fit(X, labels)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final,
            "feature_names": FEATURE_NAMES,
            "version": MODEL_VERSION,
        },
        OUTPUT_PATH,
    )
    print(f"Saved {OUTPUT_PATH}")

    manifest_path = OUTPUT_PATH.parent / "session_model_v1.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": MODEL_VERSION,
                "trained_on": "synthetic-prior-v1",
                "holdout_ap": ap,
                "holdout_ap_skill": ap_skill,
                "holdout_brier_skill": brier_skill,
                "feature_count": len(FEATURE_NAMES),
                "caveat": (
                    "No real subject-session v2 data existed at training time. "
                    "This model encodes reasoned heuristics about decision-timing "
                    "regularity, bet-sizing consistency and telemetry patterns, "
                    "not learned real-world signal. Recalibrate ASAP once v3.0 "
                    "produces real evaluation-window results."
                ),
            },
            indent=2,
        )
    )
    print(f"Saved {manifest_path}")


if __name__ == "__main__":
    main()
