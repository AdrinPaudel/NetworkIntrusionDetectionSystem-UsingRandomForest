# Complete System Architecture

Complete technical architecture covering training, testing, and prediction systems

---

## System Overview

The NIDS CICIDS2018 system has two main parts:

1. Training Pipeline: 5 modules for building the ML model
2. Prediction System: 4 modes for making predictions

---

## Part 1: Training & Testing Architecture

### Overview

The training pipeline consists of 5 modules:

Module 1 → Module 2 → Module 3 → Module 4 → Module 5
Data Loading | Exploration | Preprocessing | Training | Testing

---

## Module 1: Data Loading Architecture

Purpose: Load 10 raw CSV files, consolidate into single dataset, optimize for memory

Input: 10 CSV files (1.1M total flows, 10.4 GB)
- Friday-02-03-2018_TrafficForML_CICFlowMeter.csv
- Friday-16-02-2018_TrafficForML_CICFlowMeter.csv
- (8 more files)

Processing: Parallel threaded loading
- ThreadPoolExecutor with 5 worker threads
- Each thread loads one CSV file
- Chunk-based reading (50k rows per chunk)
- Data type optimization (float64→float32, int64→int32)

Output: Consolidated parquet file
- Path: data/preprocessed/combined_raw_data.parquet
- Rows: 16.2M
- Columns: 80
- Size: ~2.3 GB (compressed)

Data Flow:
CSV Files (10) → ThreadPoolExecutor → Chunk Reading → Type Optimization → Concatenation → Parquet Save

Memory Pattern:
- Peak: 12-14 GB (all files loading in parallel)
- Result: 10-11 GB (consolidated data)

---

## Module 2: Data Exploration Architecture

Purpose: Analyze dataset structure, quality, class distribution, correlations

Input: combined_raw_data.parquet (16.2M × 80)

Analysis:
1. Load data into memory
2. Compute class distribution (15 original classes)
3. Detect missing values (NaN, Inf)
4. Calculate feature statistics (mean, std, min, max)
5. Compute correlation matrix
6. Visualize distributions

Outputs:
- exploration_results.txt (statistics and findings)
- reports/exploration/*.png (visualizations)

Key Statistics Generated:

Class Distribution:
- Count per class
- Percentage of total
- Imbalance ratio

Data Quality:
- NaN cells and affected rows
- Inf cells and affected rows
- Duplicate rows
- Missing percentage per column

Feature Statistics:
- Mean, median, std dev, min, max for each numeric feature
- Quartiles and distribution shape
- Correlation strength between features

---

## Module 3: Data Preprocessing Architecture

Purpose: Clean data, consolidate labels, encode categories, scale, balance with SMOTE, select features

Step 1: Data Cleaning

Input: combined_raw_data.parquet (16.2M × 80)

Removal:
- Useless columns (Flow ID, IP addresses, Timestamp) → 75 columns remain
- Bad 'Label' class rows (header mistakenly included) → 4.3M rows removed
- Rows with NaN values → 59k rows removed
- Rows with Inf values → 36k rows removed
- Duplicate rows → 4.2M rows removed

Output: Cleaned dataset
- Rows: 11.9M
- Columns: 79
- Memory: ~7.5 GB

Step 2: Label Consolidation

Input: 15 original classes

Mapping:
- Benign → Benign
- DDoS variants → DDoS
- DoS variants → DoS
- Brute Force variants → Brute Force
- Botnet → Botnet
- Infilteration → Infilteration
- SQL Injection → DROP (removed)

Output: 6 consolidated classes
- Benign: 88.72%
- DDoS: 6.48%
- DoS: 1.64%
- Botnet: 1.21%
- Infilteration: 1.16%
- Brute Force: 0.79%

Step 3: Feature Encoding

Categorical Features:
- Protocol column → One-hot encoding
  - Creates binary columns for TCP (6), UDP (17), ICMP (0)
  - 3 new columns generated

Target (Label):
- 6 classes → Label encoding
- Benign→0, Botnet→1, Brute Force→2, DDoS→3, DoS→4, Infilteration→5

Output: Fully numeric dataset
- Columns: 81 (79 + 2 protocol columns)
- All features numeric

Step 4: Train-Test Split

Method: Stratified split
- Maintains class proportions in both sets
- Random seed: 42 (reproducibility)

Split: 80-20
- Training: 9.5M rows (80%)
- Testing: 2.4M rows (20%)

Step 5: Feature Scaling

Method: StandardScaler
- Transforms each feature: (x - mean) / std dev
- Result: mean=0, std=1

Output: Scaled training and test datasets

Step 6: SMOTE Oversampling

Applied to training data only

Targets (per class):
- Benign (88.72%): No oversampling (majority)
- DDoS (6.48%): No oversampling (sufficient)
- DoS (1.64%): No oversampling (sufficient)
- Botnet (1.21%): Oversample to 1.5%
- Brute Force (0.79%): Oversample to 2.0%
- Infilteration (1.16%): Oversample to 1.5%

Method: k-nearest neighbors (k=5)
- Creates synthetic samples by interpolating between neighbors
- Balances minority classes

Output: Balanced training dataset
- Rows: ~9.7M (before SMOTE: 9.5M)
- Synthetic samples created: ~200k

Step 7: Feature Selection

Method: Random Forest Importance
- Train lightweight RF (100 trees, depth 15)
- Calculate Gini-based importance
- Select features above threshold (0.005)

Process:
- Total features: 81
- Threshold filtering: Keep importance > 0.005
- Selected: 45 features
- Removed: 36 low-importance features

Output: Reduced feature set (45 features)

Data Flow Diagram:

Raw Data → Clean → Encode → Scale → SMOTE → Select Features → Final Dataset
16.2M×80    11.9M×79    11.9M×81    -        9.7M×81        -         9.7M×45

Memory Usage:
- Cleaning: 12 GB
- Encoding: 13 GB
- Scaling: 14 GB
- SMOTE: 18.5 GB (peak)
- Selection: 16 GB
- Final: 10 GB

---

## Module 4: Model Training Architecture

Purpose: Tune hyperparameters, train Random Forest model, analyze feature importance

Step 1: Data Preparation

Input: SMOTE-balanced training data (9.7M × 45)

Sampling for tuning: 20% subsample (1.9M rows)
- Reduces peak memory during hyperparameter tuning
- Still representative of full dataset

Step 2: Hyperparameter Tuning

Method: RandomizedSearchCV
- Evaluates 15 random parameter combinations
- 3-fold cross-validation per combination
- Total model fits: 45 (15 × 3)
- Optimization metric: Macro F1-score

Search Space:

n_estimators: [100, 150]
max_depth: [20, 25, 30]
min_samples_split: [2, 5, 10]
min_samples_leaf: [1, 2, 4]
max_features: ['sqrt', 'log2']
bootstrap: [True]
class_weight: ['balanced_subsample', None]

Best Parameters Found:
- n_estimators: 150
- max_depth: 20
- min_samples_split: 5
- min_samples_leaf: 1
- max_features: sqrt
- class_weight: None
- bootstrap: True

Best CV F1-Score: 0.8640

Step 3: Final Model Training

Input: Full training data (9.7M × 45)
Best parameters from tuning

Model:
- Random Forest Classifier
- 150 decision trees
- Max depth: 20
- Features: 45

Result:
- 150 trees trained
- 2.65M nodes total
- 1.33M leaves total
- Average tree depth: 20
- Training time: 5 minutes

Step 4: Feature Importance Analysis

Method: Gini-based importance from trained model

Top 10 Important Features:
1. Dst Port: 6.78%
2. Fwd Seg Size Min: 5.38%
3. Init Fwd Win Byts: 4.91%
4. Fwd Header Len: 4.54%
5. Bwd Seg Size Avg: 3.19%
6. Bwd Pkt Len Max: 3.15%
7. TotLen Fwd Pkts: 3.09%
8. Bwd Header Len: 3.04%
9. TotLen Bwd Pkts: 2.91%
10. Bwd Pkt Len Std: 2.89%

Top 10 = 39.89% of importance
Top 20 = 64.12% of importance

Output Artifacts:

trained_model/
- random_forest_model.joblib (trained model)
- scaler.joblib (StandardScaler used in preprocessing)
- label_encoder.joblib (class label encoder)
- feature_importances.csv (feature names and importance scores)
- randomized_search_cv.joblib (tuning results)
- training_metadata.json (training configuration)

---

## Module 5: Testing & Evaluation Architecture

Purpose: Evaluate model on held-out test data, calculate metrics, identify errors

Step 1: Load Test Data

Input: Test dataset (2.4M × 45)
- Preprocessed with same scaler and encoder as training
- Same 45 features
- 20% of original data

Step 2: Generate Predictions

Process:
- Feed each test sample through 150 trees
- Get prediction class (0-5)
- Get confidence (probability for predicted class)
- Decode to class name

Output:
- Predicted class per sample
- Confidence scores
- Actual class labels

Step 3: Calculate Metrics

Per-Class Metrics:
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1-Score: 2 × (Precision × Recall) / (Precision + Recall)
- AUC: Area under ROC curve

Overall Metrics:
- Accuracy: (TP + TN) / Total
- Macro Precision: Average across classes
- Macro Recall: Average across classes
- Macro F1-Score: Average F1 across classes
- Weighted F1-Score: Class-weighted average

Binary Metrics (Benign vs Attack):
- Sensitivity (TPR): True Positive Rate
- Specificity (TNR): True Negative Rate
- False Positive Rate: FP / (FP + TN)
- False Negative Rate: FN / (FN + TP)

Step 4: Analyze Errors

Confusion Matrix:
- Rows: Actual classes
- Columns: Predicted classes
- Shows which classes confused with each other

Error Analysis:
- Count false positives (false alarms)
- Count false negatives (missed attacks)
- Identify hardest-to-predict classes

Top Error Patterns:
1. Infilteration→Benign: 24,238 errors (86.97% of Infilteration missed)
2. Benign→Infilteration: 3,733 errors (false alarms for Infilteration)

Step 5: Generate Visualizations

Confusion Matrix Heatmap:
- Shows all class-to-class predictions
- Darker = more predictions
- Diagonal = correct predictions

ROC Curves:
- One curve per class (6 total)
- Shows sensitivity vs specificity trade-off

Classification Report:
- Per-class metrics in table format
- Overall statistics

Results Summary:

Overall Accuracy: 98.82%
Macro F1-Score: 86.57%
Binary F1-Score: 94.55%

Per-Class Performance:
- Benign: 99.34% F1 (excellent)
- DDoS: 99.94% F1 (perfect)
- DoS: 99.96% F1 (perfect)
- Botnet: 99.80% F1 (excellent)
- Brute Force: 99.75% F1 (excellent)
- Infilteration: 20.61% F1 (difficult - minority class)

Key Insight:
Model excellent for common attacks, struggles with minority Infilteration class despite SMOTE

---

## Part 2: Prediction Architecture

### Overview

The prediction system accepts input in 4 modes and generates predictions and alerts

Input → Validation → Preprocessing → Inference → Alert Classification → Report Generation

---

## Prediction System Components

Core Components:

1. PredictionPreprocessor (predict_preprocess.py)
   - Loads and validates input data
   - Maps column names to standard names
   - Fills missing values
   - Applies feature scaling
   - Returns preprocessed feature matrix

2. NetworkPredictor (predictor.py)
   - Loads trained Random Forest model
   - Executes model inference
   - Returns predictions and confidence scores
   - Handles batch and single predictions

3. ThreatActionHandler (threat_action_handler.py)
   - Determines alert level (RED/YELLOW/GREEN)
   - Evaluates confidence thresholds
   - Logs threat actions
   - Generates alert metadata

4. ReportGenerator (report_generator.py)
   - Compiles prediction results into reports
   - Writes log files for alerts
   - Generates statistics and summaries
   - Saves results to timestamped directories

5. RealtimeMonitor (realtime_monitor_clean.py)
   - Captures network packets from interface
   - Extracts flow-level features from packets
   - Triggers predictions on extracted flows
   - Accumulates statistics during monitoring

6. RealtimeSimulation (realtime_simulation.py)
   - Loads pre-recorded flows from CSV
   - Simulates traffic arrival at specified rate
   - Triggers predictions on simulated flows
   - Mimics real-time behavior

Supporting Files:
- predict_cli.py: Command-line interface for all prediction modes
- config.py: Configuration parameters (alert thresholds, etc.)

---

## Prediction Workflow

### Mode 1: Batch Prediction

Input Data (CSV File)

1. Load CSV
   - Read CSV file
   - Detect encoding
   - Parse headers

2. Validate Structure
   - Check all 45 required columns present
   - Report missing columns
   - Map alternate column names

3. Preprocess Each Row
   - StandardScaler: Normalize to mean=0, std=1
   - LabelEncoder: Encode categorical values
   - Handle missing: Fill with 0
   - Result: [N × 45] feature matrix

4. Batch Inference
   - Feed feature matrix to Random Forest
   - Generate predictions: class index per row
   - Generate confidence: probability per row
   - Decode class indices to names
   - Result: [N × 3] (class, confidence, timestamp)

5. Alert Classification
   - For each prediction:
     - If confidence >= 0.50: RED alert (critical)
     - If 0.25 <= confidence < 0.50: YELLOW alert (warning)
     - If confidence < 0.25: GREEN (benign, no alert)

6. Report Generation
   - Count RED, YELLOW, GREEN alerts
   - Count attacks by type
   - Calculate percentages
   - Write attack.log (RED/YELLOW only)
   - Write report.txt (summary)
   - Save statistics.json

Output: Results directory with reports

### Mode 2: Single Prediction

User Input (Interactive)

1. Display Required Features
   - Show all 45 feature names
   - Explain each feature type

2. Get User Input
   - Prompt for each feature value
   - Validate numeric input
   - Validate range constraints

3. Preprocess Input
   - StandardScaler: Normalize feature values
   - Encode categories (if applicable)
   - Create 1D feature vector [45]

4. Inference
   - Feed to Random Forest
   - Get prediction class
   - Get confidence score

5. Display Result
   - Show predicted class name
   - Show confidence percentage
   - Show alert level (RED/YELLOW/GREEN)
   - Offer to predict another

Output: Console display, optional save to log

### Mode 3: Real-Time Monitoring

Network Interface

1. Initialize Capture
   - Select network interface (eth0, wlan0, etc.)
   - Start packet capture
   - Set capture timeout

2. Extract Flows
   - Capture packets from interface
   - Group packets by flow (src/dst IP:port pair)
   - Aggregate into flow-level statistics

3. Calculate Features
   - From packet statistics, compute 45 features:
     - Packet counts (forward, backward)
     - Byte counts (forward, backward)
     - Timing statistics (inter-arrival times)
     - Packet size statistics
     - Header lengths
   - Create 1D feature vector [45]

4. Predict
   - Preprocess extracted features
   - Feed to Random Forest
   - Get prediction and confidence

5. Alert & Log
   - Classify alert level (RED/YELLOW/GREEN)
   - Log to file (RED/YELLOW only, not GREEN)
   - Update real-time statistics
   - Display to console

6. Continue Monitoring
   - Repeat steps 2-5 for next flow
   - Stop after duration expires
   - Generate summary report

Output: Session directory with alerts and summary

### Mode 4: Simulated Traffic

Pre-recorded Flows (CSV)

1. Load Simulation Data
   - Read flows from realtraffic_for_prediction.csv
   - Parse flow features
   - Prepare for replay

2. Simulate Arrival
   - Iterate through flows
   - Space flows over time at specified rate
   - Mimic real-time arrival (e.g., 5 flows/second)

3. Preprocess
   - StandardScaler: Normalize features
   - Encode categories
   - Create feature vector [45]

4. Predict
   - Feed to Random Forest
   - Get prediction and confidence

5. Alert & Log
   - Classify alert level
   - Log attacks (RED/YELLOW)
   - Accumulate statistics

6. Continue Simulation
   - Repeat until duration expires
   - Generate summary report

Output: Session directory mimicking real-time session

---

## Data Transformation Pipeline

Input Data Formats:

Batch (CSV):
- Rows: Network flows
- Columns: 45+ (at least 45 required features)
- Format: Numeric values, optional column order

Single (Interactive):
- Input: User enters 45 feature values
- Format: Numeric, validated per feature type

Real-Time (Network):
- Input: Live packet stream from network interface
- Format: Binary packet data
- Transformed: To flow-level features

Simulated (CSV):
- Input: Pre-recorded flow data
- Format: CSV with flow features
- Replayed: At specified rate

Feature Transformation:

Step 1: Column Mapping
- Input column names vary (Dst Port, destination_port, etc.)
- Mapped to standard names using FEATURE_MAPPING dict
- Result: 45 columns in standard order

Step 2: Type Validation
- All values should be numeric (float or int)
- Non-numeric: Attempted conversion or filled with 0

Step 3: Missing Value Handling
- NaN or missing: Filled with 0
- Inf: Filled with mean feature value or 0

Step 4: StandardScaler
- Saved scaler from training loaded
- Formula: (x - mean_training) / std_training
- Result: Feature values with mean ≈ 0, std ≈ 1

Step 5: Feature Order
- Reorder columns to match training feature order
- Critical: Must be exact order used during training
- Result: Ready for model inference

Model Inference:

Input: [1 × 45] feature vector (or [N × 45] batch)

Process:
1. Pass through Random Forest (150 trees)
2. Each tree votes on class
3. Majority vote determines prediction
4. Probability: Count votes / 150 trees

Output:
- Predicted class: 0-5 (Benign=0, Botnet=1, etc.)
- Confidence: 0.0-1.0 (higher = more confident)
- Class name: Decode index to name

Alert Determination:

Based on confidence score and class:

Benign prediction:
- Always GREEN (no threat)

Attack prediction (non-Benign):
- Confidence >= 0.50 → RED (alert immediately)
- Confidence 0.25-0.50 → YELLOW (investigate)
- Confidence < 0.25 → GREEN (uncertain, no alert)

Alert metadata:
- Timestamp: When prediction made
- Class: Predicted attack type
- Confidence: Prediction confidence
- Alert level: RED/YELLOW/GREEN
- Action: Taken based on level (log, notify, etc.)

---

## Output Architecture

### Report Structure

results/ (main results directory)
- batch_YYYYMMDD_HHMMSS/ (batch prediction session)
  - report.txt (summary)
  - attacks.log (detailed attacks)
  - statistics.json (threat stats)
- realtime_YYYYMMDD_HHMMSS/ (real-time monitoring)
  - report.txt
  - attacks.log
  - statistics.json
- simulated_YYYYMMDD_HHMMSS/ (simulated session)
  - report.txt
  - attacks.log
  - statistics.json

Report Content:

report.txt:
- Summary of session
- Total samples/flows processed
- Attack counts by alert level
- Attack breakdown by type
- Key statistics (accuracy, confidence levels)
- File paths for detailed logs

attacks.log:
- Line-by-line log of RED and YELLOW alerts
- Columns: Timestamp | Source | Dest | Alert Level | Class | Confidence
- GREEN alerts not logged (would be too verbose)
- One line per alert

statistics.json:
- Machine-readable statistics
- Total samples: count
- RED alerts: count and percentage
- YELLOW alerts: count and percentage
- GREEN: count
- Attacks by type: count per class
- Confidence statistics: mean, min, max

---

## Model Artifact Architecture

Required Artifacts:

trained_model/random_forest_model.joblib
- Serialized Random Forest model
- 150 trees, trained parameters
- Size: ~200 MB

trained_model/scaler.joblib
- Fitted StandardScaler
- Mean and std for each of 45 features
- Size: <1 KB

trained_model/label_encoder.joblib
- Fitted LabelEncoder
- Maps: Benign→0, Botnet→1, ..., Infilteration→5
- Size: <1 KB

data/preprocessed/feature_importances.csv
- Feature names and importance scores
- Column 1: Feature name
- Column 2: Importance value (0-1)
- Used for: Feature selection, documentation

config.py
- RED_THRESHOLD = 0.50
- YELLOW_THRESHOLD = 0.25
- FEATURE_MAPPING = {...} (column name mappings)

Loading Process:

On startup:
1. Load config.py parameters
2. Load random_forest_model.joblib
3. Load scaler.joblib
4. Load label_encoder.joblib
5. Load feature_importances.csv
6. Initialize prediction system

All artifacts loaded into memory for fast inference

---

## Performance Characteristics

Inference Speed:

Single sample: <1 millisecond
Batch (1,000 samples): ~2 milliseconds
Speed: ~475,000 samples/second on typical hardware

Why fast:
- Random Forest prediction: Simple voting algorithm
- No deep learning overhead
- Model compact (150 trees, depth 20)
- Minimal preprocessing per sample

Memory Usage:

Model in memory: ~300 MB (model + scaler + encoder)
Per-batch memory: Depends on batch size
- Batch 1,000: ~1 MB
- Batch 10,000: ~10 MB
- Batch 100,000: ~100 MB

For real-time: Minimal (process one flow at a time)

Accuracy:

Macro F1-Score: 86.57% (test set)
Accuracy: 98.82% (overall)

Per-class accuracy:
- DDoS: 99.94%
- DoS: 99.96%
- Brute Force: 99.75%
- Botnet: 99.80%
- Benign: 99.34%
- Infilteration: 20.61% (challenge)

---

## Deployment Architecture

Production Setup:

Web Service (Optional):
- Expose REST API for predictions
- Accept JSON with 45 features
- Return JSON with prediction and confidence

Batch Processing:
- Periodic CSV file processing
- Scheduled or on-demand
- Generates reports automatically

Real-Time Integration:
- Integrate with network TAP or mirror
- Feed packet stream to system
- Generate alerts in real-time
- Integration with SIEM/monitoring systems

Scaling Considerations:

Batch Predictions:
- Linear scaling with data size
- Memory-bounded by batch size
- Can process GB of data with small batches

Real-Time:
- Depends on packet rate
- Can handle millions of flows/day
- Bottleneck: Network packet capture, not prediction

Integration Points:

Input Integration:

CSV Files:
- Read directly from disk
- Validate column names
- Transform to features

Network Interface:
- Use scapy or similar to capture
- Extract flow-level features
- Feed to prediction

API Endpoint:
- Receive JSON with feature values
- Preprocess
- Generate prediction

Output Integration:

Alert System:
- Send RED alerts to security team
- Send YELLOW to monitoring dashboard
- Suppress GREEN (no action)

SIEM Integration:
- Write syslog format
- Send to Splunk/ELK/other SIEM
- Indexed and searched

Report Storage:
- Save to local results/ directory
- Archive to centralized location
- Email summaries to stakeholders

