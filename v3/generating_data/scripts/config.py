"""Single source of truth for pipeline parameters. Imported by every stage
(generation, injection, feature building, training, viewing) so values can
never drift between them."""

from pathlib import Path

# --- paths -------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
GENERATING_DATA_DIR = SCRIPTS_DIR.parent
V3_DIR = GENERATING_DATA_DIR.parent

RAW_GOOD_DATA_DIR = GENERATING_DATA_DIR / "raw_good_data"
ANOMALY_DATA_DIR = GENERATING_DATA_DIR / "anomaly_data"

IDENTIFYING_ANOMALIES_DIR = V3_DIR / "identifying_anomalies"
DATASET_DIR = IDENTIFYING_ANOMALIES_DIR / "dataset"
MODEL_DIR = IDENTIFYING_ANOMALIES_DIR / "model"
PLOTS_DIR = IDENTIFYING_ANOMALIES_DIR / "plots"

BASE_SIGNALS_INDEX_CSV = RAW_GOOD_DATA_DIR / "base_signals_index.csv"
LABELS_CSV = ANOMALY_DATA_DIR / "labels.csv"
FEATURES_CSV = DATASET_DIR / "features.csv"
MODEL_PATH = MODEL_DIR / "model.joblib"
BASELINE_THRESHOLD_PATH = MODEL_DIR / "baseline_threshold.json"
HGB_MODEL_PATH = MODEL_DIR / "model_hgb.joblib"
HGB_THRESHOLD_PATH = MODEL_DIR / "hgb_threshold.json"
METRICS_PATH = MODEL_DIR / "metrics.json"
EVAL_PER_FILE_CSV = MODEL_DIR / "eval_per_file.csv"
EVAL_PER_FILE_RAW_CSV = MODEL_DIR / "eval_per_file_raw.csv"
EVAL_SWEEP_BY_TYPE_CSV = MODEL_DIR / "eval_sweep_by_type.csv"
EVAL_TEST_PREDICTIONS_CSV = MODEL_DIR / "eval_test_predictions.csv"

# --- signal generation ---------------------------------------------------
FS = 1000  # Hz
DURATION_S = 120  # seconds (2 minutes)
N_SAMPLES = FS * DURATION_S

# wider grid than Initial_Run (5x3=15): 7 freqs x 4 phases = 28 base signals,
# so a GroupShuffleSplit/GroupKFold holdout has real statistical weight
BASE_FREQS = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]  # Hz -- stay >=1Hz: WINDOW_LEN
                                                     # (2s) needs >=2 cycles at the lowest freq
BASE_PHASES = [0.0, 3.14159265358979 / 4, 3.14159265358979 / 2, 3 * 3.14159265358979 / 4]  # radians

AMPLITUDE = 1.0

RANDOM_SEED = 42

# --- non-stationary clean baseline (new vs. Initial_Run's pure single tone) ---
# Real sensors have noise, and a "steady" signal is rarely a perfect single
# tone: this layers slow amplitude/frequency drift, 2nd/3rd harmonics, and
# measurement noise onto the fundamental. This deliberately breaks the
# single-tone assumption several features rely on (lsq_residual_energy,
# estimate_f0) -- the point of "harder".
NOISE_SIGMA = 0.03            # i.i.d. Gaussian noise sigma, fraction of AMPLITUDE
AMP_DRIFT_DEPTH = 0.08         # slow AM envelope depth (fraction of AMPLITUDE)
AMP_DRIFT_FREQ_HZ = 0.01       # AM period 100s
FREQ_DRIFT_FRAC = 0.02         # FM depth, fraction of f0 (peak deviation)
FREQ_DRIFT_FREQ_HZ = 0.013     # FM period ~77s -- non-commensurate with AMP_DRIFT_FREQ_HZ on purpose
HARMONIC2_AMP_RATIO = 0.06     # 2nd harmonic amplitude, fraction of A
HARMONIC3_AMP_RATIO = 0.03     # 3rd harmonic amplitude, fraction of A

# --- anomaly injection -----------------------------------------------------
# three anomaly types instead of one: a different-frequency tone (as in
# Initial_Run), an amplitude dropout (sensor attenuation), and a DC offset
# jump (sensor bias shift) -- each a distinct failure mode with a distinct
# physical signature
ANOMALY_TYPES = ["freq_swap", "amp_dropout", "dc_jump"]
INJECTIONS_PER_BASE = 6  # 2 reps x 3 types per base signal
N_CONTROL_FILES = 10  # anomaly-free copies of (randomly chosen) base signals

ANOMALY_DURATION_RANGE_S = (3.0, 20.0)  # shorter than Initial_Run's (10,30) -- harder
EDGE_MARGIN_S = 2.0

# multiplicative separation from f0, applied as f0 * U. "wide" keeps the
# anomaly freq clearly distinguishable from f0 (Initial_Run's only regime);
# "narrow" is a new, harder regime close to f0.
ANOMALY_FREQ_RATIO_REGIMES = {
    "wide": [(1.8, 4.0), (0.25, 0.55)],
    "narrow": [(1.05, 1.3), (0.75, 0.95)],
}
ANOMALY_FREQ_MIN_HZ = 0.5
ANOMALY_FREQ_MAX_HZ = 50.0

# per-type severity sweeps, each spanning hard (quiet/subtle) to easy (loud/obvious)
AMP_RATIO_CHOICES = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]  # freq_swap: anomaly-tone / A
DROPOUT_DEPTH_CHOICES = [0.3, 0.5, 0.7, 0.85, 0.95]  # amp_dropout: fractional attenuation
DC_JUMP_OFFSET_RATIO_CHOICES = [0.15, 0.3, 0.5, 0.75, 1.0]  # dc_jump: |offset| / A (sign randomized)

TUKEY_ALPHA = 0.15  # reused for freq_swap/amp_dropout envelopes and as the dc_jump ramp fraction

# --- sliding-window features / labeling -----------------------------------
WINDOW_LEN = 2000  # samples (2 s) -- >=2 cycles even at the 1 Hz base freq
WINDOW_STRIDE = 200  # samples (0.2 s) -- finer stride tightens localization
OVERLAP_THRESHOLD = 0.5  # fraction of window that must overlap true anomaly

assert ANOMALY_DURATION_RANGE_S[0] * FS >= WINDOW_LEN, (
    "shortest anomaly duration must still span at least one full sliding window"
)

# --- model -----------------------------------------------------------------
TEST_SIZE = 0.25  # fraction of base signals held out (was 0.2) -- ~7 of 28
N_CV_FOLDS = 5  # GroupKFold folds for confidence-interval metrics, on top of
                 # the single held-out split used for all downstream artifacts

RF_PARAMS = {
    "n_estimators": 300,
    "min_samples_leaf": 3,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
}
HGB_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_depth": None,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "early_stopping": True,
    "validation_fraction": 0.1,
}
CALIB_HOLDOUT_FRACTION = 0.2  # fraction of *training* groups carved out for HGB calibration

# --- reconstruction ----------------------------------------------------------
RECONSTRUCT_MEDIAN_KERNEL = 5  # odd kernel, applied to window scores before interpolation
RECONSTRUCT_HYSTERESIS = (0.6, 0.4)  # (enter_threshold, exit_threshold)
