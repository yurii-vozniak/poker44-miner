# Poker44 v3.0 migration plan (subject-session + telemetry)

Status as of 2026-07-28: v3.0 is published on the framework's `dev` branch
(https://github.com/Poker44/Poker44-subnet/tree/dev) but **not merged to
`main`** yet (confirmed: `main` still 404s on `poker44/protocol.py` /
`SessionDetectionSynapse`). Our production miner is unaffected for now and
keeps running the current chunk-based `DetectionSynapse` through the rest of
this epoch (R2-R5).

This repo vendors its own copy of the `poker44` framework package (it is not
a pip dependency), so nothing updates automatically when the framework repo
changes. When v3.0 merges to `main`, do the following.

## What's already prepared (this session)

- `deploy/session_features.py` — feature extraction for `subject-session.v2`
  (hand action stats + telemetry event/timing stats). Validated against the
  real `examples/subject-session.v2.json` fixture from the dev branch and
  against defensive edge cases (nulls, empty telemetry, empty hands).
- `deploy/synthetic_sessions.py` — synthetic bot/human session generator
  encoding reasoned priors (decision-timing regularity, bet-sizing
  consistency, telemetry event patterns), used only because no real
  session-level data is available to miners pre-launch.
- `deploy/train_session_model.py` — trains a calibrated
  `GradientBoostingClassifier` on the synthetic prior, saves
  `models/session_model_v1.joblib` + manifest.
- `deploy/session_model.py` — `SessionBotDetectionModel` implementing the
  `BotDetectionModel` protocol (`version`, `load()`, `predict(sessions)`)
  and a `create_model(config)` factory entrypoint compatible with
  `POKER44_MODEL_FACTORY`.

**Caveat:** the shipped model is fit on synthetic data only (perfect
synthetic AP, meaningless as a real-world estimate). It exists so we are not
scrambling from zero when v3.0 goes live, not as a finished, tuned model.
Recalibrate/retrain the moment real evaluation-window results are available,
the same way v21->v23 was iterated using live dashboard feedback for the
current chunk-based competition.

## What still needs to happen once v3.0 merges to `main`

1. **Vendor the new framework pieces** into this repo's local `poker44/`
   package (diff against `main` once merged, not `dev`, in case anything
   changes before release):
   - `poker44/protocol.py` (adds `SessionDetectionSynapse`; note
     `DetectionSynapse = SessionDetectionSynapse` alias exists for
     transition)
   - `poker44/miner/config.py`, `poker44/miner/loader.py`,
     `poker44/miner/service.py`, `poker44/miner/model.py`
   - `poker44/base/miner.py` if changed (check diff against our vendored copy)
2. **New neuron entrypoint**: create `neurons/miner_session.py` modeled on
   the reference `neurons/miner.py` (async `forward(SessionDetectionSynapse)`
   calling `MinerInferenceService.predict`), wired to our model via
   `POKER44_MODEL_FACTORY=deploy.session_model:create_model`.
3. **Update `.env`**:
   - `POKER44_MODEL_FACTORY=deploy.session_model:create_model`
   - `POKER44_MODEL_VERSION=session-v1-synthetic` (bump after retraining)
   - Optionally `POKER44_SESSION_MODEL_PATH` if not using the default
     `models/session_model_v1.joblib`
   - Keep `POKER44_MAX_SESSIONS_PER_REQUEST` / `POKER44_MAX_REQUEST_BYTES` at
     framework defaults unless we have a reason to change them
4. **Run the framework's own test suite** before deploying:
   `PYTHONPATH=. pytest -q` (per their checklist) plus our own
   `deploy/session_features.py` / `deploy/session_model.py` sanity checks.
5. **Switch pm2 process** to the new entrypoint and confirm the miner logs
   `Poker44 miner model loaded | factory=deploy.session_model:create_model`.
6. **Re-tune once real data exists**: as soon as our miner starts receiving
   real `SessionDetectionSynapse` requests and dashboard composite scores
   start appearing, treat this the same way the v21->v23 calibration bug was
   diagnosed — pull real session/label pairs if the platform ever exposes a
   v2 benchmark release, otherwise iterate purely on dashboard-observed
   score trends since the new reward
   (`0.50*AP_skill + 0.30*recall@5%FPR + 0.20*brier_skill`, see
   `poker44/validator/evaluation/reward.py` on the dev branch) has **no hard
   0.5-threshold human-safety cliff** like the current v2 reward does — so
   the batch-calibration "max_pos_frac" trick from v23 does not apply here.
   Calibration quality (Brier skill) matters directly instead.

## Key behavioral differences vs. the current (v2) reward, worth remembering

- No more forced positive-fraction gating: v3.0's reward has no hard-FPR
  cliff, so there is no need to suppress how many scores cross 0.5. Raw,
  honestly-calibrated probabilities are directly rewarded via Brier skill.
- `average_precision_skill` and `recall_at_fpr_05` are both computed
  relative to the *actual* prevalence of the batch, so the reward is
  self-normalizing across different bot/human mixes (a difference from v2,
  where the 50/50 training prevalence didn't match live batches).
- Sessions replace hands as the scoring unit: exactly one score per session,
  not one per hand.
- Telemetry (click/pointer/scroll timing + bucketed coordinates) is a first
  class signal now, in addition to poker decisions.
