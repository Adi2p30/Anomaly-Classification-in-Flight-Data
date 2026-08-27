"""Single source of truth for pipeline parameters. Imported by every stage
(generation, injection, feature building, training, viewing) so values can
never drift between them."""

from pathlib import Path

# --- paths -------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
GENERATING_DATA_DIR = SCRIPTS_DIR.parent
INITIAL_RUN_DIR = GENERATING_DATA_DIR.parent

RAW_GOOD_DATA_DIR = GENERATING_DATA_DIR / "raw_good_data"
ANOMALY_DATA_DIR = GENERATING_DATA_DIR / "anomaly_data"

IDENTIFYING_ANOMALIES_DIR = INITIAL_RUN_DIR / "identifying_anomalies"
DATASET_DIR = IDENTIFYING_ANOMALIES_DIR / "dataset"
MODEL_DIR = IDENTIFYING_ANOMALIES_DIR / "model"
PLOTS_DIR = IDENTIFYING_ANOMALIES_DIR / "plots"

BASE_SIGNALS_INDEX_CSV = RAW_GOOD_DATA_DIR / "base_signals_index.csv"
LABELS_CSV = ANOMALY_DATA_DIR / "labels.csv"
FEATURES_CSV = DATASET_DIR / "features.csv"
MODEL_PATH = MODEL_DIR / "model.joblib"
BASELINE_THRESHOLD_PATH = MODEL_DIR / "baseline_threshold.json"
METRICS_PATH = MODEL_DIR / "metrics.json"

# --- signal generation ---------------------------------------------------
FS = 1000  # Hz
DURATION_S = 120  # seconds (2 minutes)
N_SAMPLES = FS * DURATION_S

BASE_FREQS = [1.0, 2.0, 3.0, 5.0, 8.0]  # Hz
BASE_PHASES = [0.0, 3.14159265358979 / 3, 2 * 3.14159265358979 / 3]  # radians

AMPLITUDE = 1.0

RANDOM_SEED = 42

# --- anomaly injection -----------------------------------------------------
INJECTIONS_PER_BASE = 4
N_CONTROL_FILES = 5  # anomaly-free copies of (randomly chosen) base signals

ANOMALY_DURATION_RANGE_S = (10.0, 30.0)
EDGE_MARGIN_S = 2.0

# multiplicative separation from f0, applied as f0 * U; keeps anomaly freq
# clearly distinguishable from f0 across the whole base-frequency grid
ANOMALY_FREQ_RATIO_CHOICES = [
    (1.8, 4.0),
    (0.25, 0.55),
]
ANOMALY_FREQ_MIN_HZ = 0.5
ANOMALY_FREQ_MAX_HZ = 50.0

AMP_RATIO_CHOICES = [0.5, 0.75, 1.0, 1.5, 2.0]

TUKEY_ALPHA = 0.15

# --- sliding-window features / labeling -----------------------------------
WINDOW_LEN = 2000  # samples (2 s) -- >=2 cycles even at the 1 Hz base freq
WINDOW_STRIDE = 200  # samples (0.2 s) -- finer stride tightens localization
OVERLAP_THRESHOLD = 0.5  # fraction of window that must overlap true anomaly

# --- model -----------------------------------------------------------------
TEST_SIZE = 0.2  # fraction of base signals held out
RF_PARAMS = {
    "n_estimators": 300,
    "min_samples_leaf": 3,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
}
