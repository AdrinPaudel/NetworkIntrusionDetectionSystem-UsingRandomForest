# Configuration & Values Used

All actual values configured in the project code

---

## Data Processing Pipeline

### Label Consolidation (15 → 6 Classes)
Original 15 classes mapped to 6 consolidated classes:
- Benign → Benign
- DDoS variants (LOIC-HTTP, LOIC-UDP, HOIC) → DDoS
- DoS variants (Hulk, SlowHTTPTest, GoldenEye, Slowloris) → DoS
- Brute Force variants (FTP, SSH, Web, XSS) → Brute Force
- Botnet → Botnet
- Infilteration → Infilteration

Why: Simplifies model to 6 clear threat categories. If changed, model must be retrained with new label mappings.

### Features: 78 → 45
Original: 78 network flow features from CICIDS2018
Selected: 45 features using Random Forest importance
Drop: SQL Injection class (designated as __DROP__ - filtered during cleaning)

How: Random Forest Gini-based importance scoring selects features. Threshold: 0.005 importance
If changed to lower threshold: More features selected (slower, higher accuracy but overfitting risk)
If changed to higher threshold: Fewer features (faster, lower accuracy)

---

## Train-Test Split

TEST_SIZE = 0.20
Meaning: 80% training data, 20% testing data

STRATIFY = True
Meaning: Split maintains class distribution - each class has same % in train and test
If removed: Train/test class distributions could differ (unrealistic evaluation)

RANDOM_STATE = 42
Meaning: Fixed seed ensures reproducibility - same split every time
If changed: Different rows selected for train/test (results vary)

---

## Data Cleaning

Removed columns (not useful for ML):
- Flow ID
- Src IP
- Dst IP
- Src Port
- Timestamp

These are metadata, not features. Removing them reduces memory and prevents model from learning on identifiers.

Removed rows:
- Rows with class label = 'Label' (header row mistakenly included as data)
- Rows with NaN values
- Rows with Inf (infinity) values
- Duplicate rows
- Outliers

If relaxed: Dirty data → worse accuracy. If stricter: Less training data → potentially worse accuracy.

---

## Feature Scaling

SCALER_TYPE = 'standard'
Meaning: StandardScaler normalizes to mean=0, std=1
Formula: (x - mean) / std_dev

Why: Random Forest doesn't need scaling, but ensures consistent feature magnitudes
If removed: Model still works (RF is scale-invariant) but preprocessing consistency lost
If changed to 'minmax': Scales 0-1, different but similar performance

---

## SMOTE Oversampling

APPLY_SMOTE = True
Enabled to balance class distribution (many benign, few attacks)

SMOTE_K_NEIGHBORS = 5
Creates synthetic samples using 5 nearest neighbors
If lower (e.g., 3): More synthetic variation, sometimes unstable
If higher (e.g., 10): Smoother synthetic samples, more interpolation

SMOTE_STRATEGY = 'tiered'
Different oversampling targets per class:
- Benign (83.07%): No oversampling (majority class)
- Botnet (1.76%): Oversampled to 1.5%
- Brute Force (2.35%): Oversampled to 2.0%
- DDoS (7.79%): No oversampling (sufficient size)
- DoS (4.03%): No oversampling (sufficient size)
- Infilteration (1.00%): Oversampled to 1.5%

Result after SMOTE: ~920k rows (from ~880k original train)
If changed to uniform: All classes get same oversampling (often worse - over-oversamples large classes)
If disabled: Imbalanced training (benign dominates, poor attack detection)

Memory impact: SMOTE creates synthetic data, peak RAM usage reaches 18.5 GB during oversampling

---

## Random Forest Model

### Hyperparameter Tuning

HYPERPARAMETER_TUNING = True
Automatic search for best parameters using RandomizedSearchCV

N_ITER_SEARCH = 15
Evaluates 15 random parameter combinations
15 iterations × 3 CV folds = 45 total model fits
If lower: Faster tuning but worse hyperparameters
If higher: Better parameters but longer tuning time (days)

CV_FOLDS = 3
Cross-validation with 3 splits for parameter evaluation
Why 3 not 5: Memory constraint (5 folds = 5 model copies, exceeds 200GB during tuning)
If changed to 5: Better validation but requires more RAM
If changed to 2: Faster but less reliable parameter assessment

TUNING_SAMPLE_FRACTION = 0.2
Uses 20% of training data (~180k rows) for hyperparameter tuning
Why: Peak RAM during tuning reaches 18 GB; using full dataset would exceed limit
If removed (use full data): Better parameter selection but crashes on smaller systems
If increased to 0.5: Better but requires 200GB system minimum

TUNING_SCORING = 'f1_macro'
Optimization metric: Macro F1-score (average across all classes)
Why: Macro F1 treats all classes equally (important for minority attacks)
If changed to 'accuracy': Optimizes for overall accuracy (favors benign detection)

### Hyperparameter Search Space (Arrays to Test)

These are the options we give to RandomizedSearchCV to test:

n_estimators: [100, 150]
Test: 100 trees OR 150 trees
If changed to [50]: Too few trees, underfit
If changed to [200, 250]: Better accuracy but training takes much longer

max_depth: [20, 25, 30]
Test: depth 20 OR depth 25 OR depth 30
If changed to [50]: Trees too deep, overfitting
If changed to [10]: Trees too shallow, underfitting

min_samples_split: [2, 5, 10]
Test: split at 2 OR 5 OR 10 samples
If changed to [100]: Only splits large nodes, underfitting

min_samples_leaf: [1, 2, 4]
Test: leaf with 1 OR 2 OR 4 minimum samples
If changed to [10]: Large leaves only, underfitting

max_features: ['sqrt', 'log2']
Test: sqrt (7 features) OR log2 (6 features)
If added 'None': Would test all 45 features (slower, overfitting risk)

bootstrap: [True]
Test: only True (sampling with replacement)
If changed to [True, False]: Tests both (False is slower)

class_weight: ['balanced_subsample', None]
Test: balanced_subsample OR no weighting
If changed to ['balanced']: Different weighting method

### Training Configuration

N_JOBS = -1
Use all available CPU cores for training
Alternative: Set to 4 (uses 4 cores), 1 (single core), etc.

N_JOBS_LIGHT = 1
Per-tree parallelism during hyperparameter tuning
Set to 1 to prevent nested parallelism issues
Alternative: Change to 2 or higher for more parallelism

RF_MAX_SAMPLES = 0.5
Each tree samples 50% of training data (bootstrap sampling)
Alternative: Set to 1.0 (100% sampling), 0.3 (30% sampling), etc.

---

## Feature Importance Selection

ENABLE_RF_IMPORTANCE = True (enabled)
ENABLE_RFE = False (disabled)
Uses fast Random Forest Gini importance (not slower RFE)

RF_IMPORTANCE_TREES = 100
Trees for importance calculation
If lower (50): Faster but noisier importance scores
If higher (200): More stable but slower

RF_IMPORTANCE_MAX_DEPTH = 15
Depth limit for importance RF (prevents overfitting during importance calc)
If higher (25): More accurate importance but slower
If lower (10): Faster but less reliable

TARGET_FEATURES_MIN = 40
TARGET_FEATURES_MAX = 45
Final feature count: between 40-45 features
If changed to 30-50: Wider range (could select too few or too many)

IMPORTANCE_THRESHOLD = 0.005
Only keep features with importance > 0.005
If lower (0.001): More features selected (45-60, more complex)
If higher (0.01): Fewer features (30-35, faster but less accurate)

---

## Alert System (Prediction Phase)

Alert levels based on model prediction confidence:

RED_THRESHOLD = 0.50 (50% confidence)
Alerts >= 50% confidence = RED (Critical attack)
If lowered to 0.30: More RED alerts (higher false positive rate)
If raised to 0.70: Fewer RED alerts (miss actual attacks)

YELLOW_THRESHOLD = 0.25 (25% confidence)
Alerts 25-50% confidence = YELLOW (Suspicious, needs investigation)
If lowered to 0.10: More YELLOW alerts (noisier)
If raised to 0.40: Fewer YELLOW alerts (miss suspicious activity)

GREEN_THRESHOLD < 0.25 (implicit)
Below 25% confidence = GREEN (Benign, safe)
No alerting for GREEN

Thresholds set to detect:
- RED: Clear attacks (50%+)
- YELLOW: Suspicious but uncertain (25-50%)
- GREEN: Safe benign traffic (<25%)

If both thresholds increased: Miss attacks but fewer false alarms
If both decreased: Catch more attacks but many false alarms

---

## System Resource Configuration

LOW_MEMORY = False
System memory setting

Configured for system with substantial RAM (200GB available)
Alternative: Set to True if running on constrained system

Memory-Related Settings:

TUNING_SAMPLE_FRACTION = 0.2
Uses 20% of training data for hyperparameter tuning
Alternative: Set to 0.1 (10%) for lower memory, or 0.5 (50%) for better tuning

CV_FOLDS = 3
Cross-validation with 3 splits
Alternative: Set to 2 (lower memory), 5 (more thorough), etc.

N_ITER_SEARCH = 15
Number of hyperparameter combinations to test
Alternative: Set to 10 (faster), 20 (more thorough), etc.

GARBAGE_COLLECTION_INTERVAL = 5
Run garbage collection every 5 iterations during training
Alternative: Set to 10 (less frequent), 1 (more aggressive)

ENABLE_MEMORY_OPTIMIZATION = True
Enable periodic memory cleanup during training

---

## Reproducibility Settings

RANDOM_STATE = 42
Fixed seed for reproducibility
Same seed ensures:
- Same train-test split
- Same SMOTE synthetic samples
- Same hyperparameter tuning results
- Same model (if using same params)

LOG_TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'
Consistent timestamp format for logging

VERBOSE = True
Detailed console output during training
If False: No progress output (faster but no feedback)
