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

# Label Consolidation Mapping (15 → 8 classes)
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
    
    # Brute Force variants → Brute Force
    'FTP-BruteForce': 'Brute Force',
    'FTP-Patator': 'Brute Force',
    'SSH-Bruteforce': 'Brute Force',
    'SSH-Patator': 'Brute Force',
    
    # Web Attack variants → Web Attack
    'SQL Injection': 'Web Attack',
    'Brute Force -Web': 'Web Attack',
    'Brute Force -XSS': 'Web Attack',
    
    # Botnet
    'Bot': 'Botnet',
    'Botnet': 'Botnet',
    
    # Infiltration (fix typo in dataset)
    'Infilteration': 'Infiltration',
    'Infiltration': 'Infiltration',
    
    # Heartbleed
    'Heartbleed': 'Heartbleed',
}

# Train-test split
TEST_SIZE = 0.20  # 80:20 split
RANDOM_STATE = 42  # For reproducibility
STRATIFY = True  # Maintain class proportions in split

# Feature Scaling
SCALER_TYPE = 'standard'  # 'standard' or 'minmax'

# SMOTE (Synthetic Minority Over-sampling)
APPLY_SMOTE = True  # Enabled for moderate oversampling
SMOTE_K_NEIGHBORS = 5
SMOTE_TARGET_PERCENTAGE = 0.03  # Moderate: Bring minorities to ~3% of dataset (not 1% which would be excessive)

# Feature Selection (RFE)
ENABLE_RFE = False  # Temporarily disabled for testing (enable after verifying other steps work)
APPLY_RFE = False  # Legacy flag, use ENABLE_RFE instead
RFE_MIN_FEATURES = 30  # Start from 30 features minimum
RFE_STEP = 1  # Features to remove per iteration
RFE_CV_FOLDS = 5
RFE_SCORING = 'f1_macro'
RFE_TARGET_FEATURES_MIN = 35  # Moderate target (not too aggressive)
RFE_TARGET_FEATURES_MAX = 45  # Moderate target (not too aggressive)

# ============================================================
# MODEL TRAINING SETTINGS
# ============================================================

# Hyperparameter Tuning (RandomizedSearchCV)
HYPERPARAMETER_TUNING = True
N_ITER_SEARCH = 50  # Number of parameter combinations to try
CV_FOLDS = 5  # Cross-validation folds
TUNING_SCORING = 'f1_macro'  # Optimization metric

# Random Forest Hyperparameter Search Space
PARAM_DISTRIBUTIONS = {
    'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500],
    'max_depth': [10, 15, 20, 25, 30, 35, 40, 45, 50, None],
    'min_samples_split': [2, 3, 4, 5, 7, 10, 15],
    'min_samples_leaf': [1, 2, 3, 4, 5, 7],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

# Default hyperparameters (if tuning is skipped)
DEFAULT_RF_PARAMS = {
    'n_estimators': 300,
    'max_depth': 30,
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

# Parallel processing
N_JOBS = -1  # Use all available CPU cores (32 vCPU)

# Memory settings (optimized for 208GB RAM)
LOW_MEMORY = False  # We have plenty of RAM

# Logging
VERBOSE = True  # Detailed console output
LOG_TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'

# ============================================================
# EXPECTED PERFORMANCE TARGETS
# ============================================================
TARGET_MACRO_F1_SCORE = 0.96  # >96%
TARGET_ACCURACY = 0.99  # >99%
TARGET_INFILTRATION_F1 = 0.89  # >89% for hardest class
