"""
Configuration File for NIDS CICIDS2018 Project
All project settings and hyperparameters
"""

import os

# ============================================================
# PROJECT PATHS
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PREPROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'preprocessed')
TRAINED_MODEL_DIR = os.path.join(PROJECT_ROOT, 'trained_model')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')

# Report subdirectories
REPORTS_EXPLORATION_DIR = os.path.join(REPORTS_DIR, 'exploration')
REPORTS_PREPROCESSING_DIR = os.path.join(REPORTS_DIR, 'preprocessing')
REPORTS_TRAINING_DIR = os.path.join(REPORTS_DIR, 'training')
REPORTS_TESTING_DIR = os.path.join(REPORTS_DIR, 'testing')

# ============================================================
# DATA LOADING SETTINGS
# ============================================================
# Expected label column name variations
LABEL_COLUMN_CANDIDATES = ['Label', 'label', ' Label', 'Label ', 'class', 'Class']
# Expected protocol column name variations (for one-hot encoding)
# Note: Dst Port is NOT encoded - it's a feature, not a categorical variable
PROTOCOL_COLUMN_CANDIDATES = ['Protocol', 'protocol', ' Protocol']

# Data type optimization
OPTIMIZE_DTYPES = True  # Convert float64→float32, int64→int32

# ============================================================
# DATA EXPLORATION SETTINGS
# ============================================================
# Correlation analysis
TOP_N_FEATURES_CORRELATION = 30  # Top N features for correlation heatmap (increased for better analysis)
HIGH_CORRELATION_THRESHOLD = 0.9  # Threshold for highly correlated pairs

# Visualization settings
FIGURE_DPI = 300  # Resolution for saved figures
FIGURE_FORMAT = 'png'  # Image format

# ============================================================
# DATA PREPROCESSING SETTINGS
# ============================================================

# Label Consolidation Mapping (15 → 6 classes)
# User requested mapping:
# Benign → Benign
# DDoS variants → DDoS
# DoS variants → DoS
# Brute Force variants (SSH, FTP, Web, XSS) → Brute Force
# Botnet → Botnet
# Infilteration → Infilteration (keep original)
# SQL Injection → DROP
LABEL_MAPPING = {
    # Benign
    'Benign': 'Benign',
    'BENIGN': 'Benign',
    
    # DDoS variants → DDoS
    'DDoS attacks-LOIC-HTTP': 'DDoS',
    'DDoS attacks-LOIC-UDP': 'DDoS',
    'DDOS attack-LOIC-UDP': 'DDoS',
    'DDOS attack-HOIC': 'DDoS',
    'DDoS attacks-HOIC': 'DDoS',
    
    # DoS variants → DoS
    'DoS attacks-Hulk': 'DoS',
    'DoS attacks-SlowHTTPTest': 'DoS',
    'DoS attacks-GoldenEye': 'DoS',
    'DoS attacks-Slowloris': 'DoS',
    'DoS GoldenEye': 'DoS',
    'DoS Hulk': 'DoS',
    'DoS Slowhttptest': 'DoS',
    'DoS slowloris': 'DoS',
    
    # Brute Force variants → Brute Force (including SSH, FTP, Web, XSS)
    'FTP-BruteForce': 'Brute Force',
    'FTP-Patator': 'Brute Force',
    'SSH-Bruteforce': 'Brute Force',
    'SSH-Patator': 'Brute Force',
    'Brute Force -Web': 'Brute Force',
    'Brute Force -XSS': 'Brute Force',
    
    # Botnet
    'Bot': 'Botnet',
    'Botnet': 'Botnet',
    
    # Infilteration (keep original name as user requested)
    'Infilteration': 'Infilteration',
    'Infiltration': 'Infilteration',
    
    # SQL Injection → DROP (will be filtered out)
    'SQL Injection': '__DROP__',
    
    # Heartbleed (if present)
    'Heartbleed': '__DROP__',
}

# Train-test split
TEST_SIZE = 0.20  # 80:20 split
RANDOM_STATE = 42  # For reproducibility
STRATIFY = True  # Maintain class proportions in split

# Feature Scaling
SCALER_TYPE = 'standard'  # 'standard' or 'minmax'

# SMOTE (Synthetic Minority Over-sampling)
APPLY_SMOTE = True  # Enabled for tiered oversampling strategy
SMOTE_K_NEIGHBORS = 5
SMOTE_STRATEGY = 'tiered'  # 'uniform' or 'tiered' (tiered = different targets per class)
# Tiered targets for 6-class system after consolidation:
# Classes: Benign, Botnet, Brute Force, DDoS, DoS, Infilteration
SMOTE_TIERED_TARGETS = {
    # Format: class_index: target_percentage_of_train_set
    # Index 0: Benign (83.07%) - no SMOTE needed (majority class)
    # Index 1: Botnet (1.76%) - needs oversampling
    1: 0.015,  # Botnet → 1.5%
    # Index 2: Brute Force (2.35%) - needs oversampling
    2: 0.020,  # Brute Force → 2.0%
    # Index 3: DDoS (7.79%) - likely no SMOTE needed (reasonable size)
    # Index 4: DoS (4.03%) - likely no SMOTE needed
    # Index 5: Infilteration (1.00%) - minority, likely needs oversampling
    5: 0.015,  # Infilteration → 1.5%
}
SMOTE_TARGET_PERCENTAGE = 0.03  # Fallback for uniform strategy

# Feature Selection (RF Importance - Paper 1 Method)
ENABLE_RFE = False  # ✗ DISABLED - Using RF importance instead (faster)
ENABLE_RF_IMPORTANCE = True  # ✓ ENABLED - Fast RF Gini importance (Paper 1 method)
RF_IMPORTANCE_TREES = 100  # Trees for importance calculation (balance speed/stability)
RF_IMPORTANCE_MAX_DEPTH = 15  # Max depth for importance RF
RF_IMPORTANCE_N_JOBS = 16  # Parallel jobs (reduced from 32 - threading backend limit)
TARGET_FEATURES_MIN = 40  # Minimum features to keep
TARGET_FEATURES_MAX = 45  # Maximum features to keep
IMPORTANCE_THRESHOLD = 0.005  # Keep features above this importance

# Legacy RFE parameters (not used when ENABLE_RFE=False)
APPLY_RFE = False  # Legacy flag
RFE_MIN_FEATURES = 30
RFE_STEP = 1
RFE_CV_FOLDS = 5
RFE_SCORING = 'f1_macro'
RFE_TARGET_FEATURES_MIN = 35
RFE_TARGET_FEATURES_MAX = 45

# ============================================================
# MODEL TRAINING SETTINGS
# ============================================================

# Hyperparameter Tuning (RandomizedSearchCV)
HYPERPARAMETER_TUNING = True
N_ITER_SEARCH = 15  # Optimized for 16vCPU: 15 iterations × 3 folds = 45 total fits
CV_FOLDS = 3  # Cross-validation folds (3 for memory efficiency)
TUNING_SAMPLE_FRACTION = 0.2  # Sample 20% of training data for tuning to reduce peak RAM
TUNING_SCORING = 'f1_macro'  # Optimization metric

# Memory Management during Training
GARBAGE_COLLECTION_INTERVAL = 5  # Run gc.collect() every N iterations
ENABLE_MEMORY_OPTIMIZATION = True  # Enable periodic memory cleanup

# Random Forest Hyperparameter Search Space (FAST MODE)
PARAM_DISTRIBUTIONS = {
    'n_estimators': [100, 150],  # FAST: Only 2 options
    'max_depth': [20, 25, 30],  # FAST: Focused mid-range
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],  # Removed None
    'bootstrap': [True],  # FAST: Only True (faster subsampling)
    'class_weight': ['balanced_subsample', None]  # FAST: Removed 'balanced'
}

# Default hyperparameters (if tuning is skipped)
DEFAULT_RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 25,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'bootstrap': True,
    'class_weight': 'balanced_subsample',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,  # Use all CPU cores
    'verbose': 1
}

# ============================================================
# MODEL TESTING SETTINGS
# ============================================================

# Evaluation metrics
CLASSIFICATION_REPORT_DIGITS = 4  # Decimal places in classification report
CONFUSION_MATRIX_NORMALIZE = None  # None, 'true', 'pred', 'all'

# ROC Curve settings
ROC_MICRO_AVERAGE = True
ROC_MACRO_AVERAGE = True

# ============================================================
# SYSTEM SETTINGS
# ============================================================

# Parallel processing (auto-scale across VM specs)
# Use -1 to utilize all available cores for CV/final fits; keep RF per-fit single-threaded to avoid nested parallelism
N_JOBS = -1          # Final training uses all logical CPUs
N_JOBS_LIGHT = 1     # Per-estimator threads during tuning (prevents nested parallelism)
N_JOBS_CV = -1       # CV workers use all logical CPUs (threading backend shares memory)

# Memory settings (optimized for 200GB RAM + 16vCPU)
LOW_MEMORY = False  # 200GB is still substantial, but limited
RF_MAX_SAMPLES = 0.5  # Cap per-tree bootstrap sample to 50% to lower memory

# Logging
VERBOSE = True  # Detailed console output
LOG_TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'

# ============================================================
# EXPECTED PERFORMANCE TARGETS
# ============================================================
TARGET_MACRO_F1_SCORE = 0.96  # >96%
TARGET_ACCURACY = 0.99  # >99%
TARGET_INFILTRATION_F1 = 0.89  # >89% for hardest class
