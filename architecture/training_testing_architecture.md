# Training & Testing Architecture

Complete technical architecture for the training and testing pipeline

---

## Overview

The training pipeline consists of 5 modules:

Module 1 → Module 2 → Module 3 → Module 4 → Module 5
Data Loading | Exploration | Preprocessing | Training | Testing

---

## Module 1: Data Loading Architecture

### Purpose
Load 10 raw CSV files, consolidate into single dataset, optimize for memory

### Components

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

### Data Flow

CSV Files (10) → ThreadPoolExecutor → Chunk Reading → Type Optimization → Concatenation → Parquet Save

Memory Pattern:
- Peak: 12-14 GB (all files loading in parallel)
- Result: 10-11 GB (consolidated data)

---

## Module 2: Data Exploration Architecture

### Purpose
Analyze dataset structure, quality, class distribution, correlations

### Components

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

### Key Statistics Generated

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

### Purpose
Clean data, consolidate labels, encode categories, scale, balance with SMOTE, select features

### Components

#### Step 1: Data Cleaning

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

#### Step 2: Label Consolidation

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

#### Step 3: Feature Encoding

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

#### Step 4: Train-Test Split

Method: Stratified split
- Maintains class proportions in both sets
- Random seed: 42 (reproducibility)

Split: 80-20
- Training: 9.5M rows (80%)
- Testing: 2.4M rows (20%)

#### Step 5: Feature Scaling

Method: StandardScaler
- Transforms each feature: (x - mean) / std dev
- Result: mean=0, std=1

Output: Scaled training and test datasets

#### Step 6: SMOTE Oversampling

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

#### Step 7: Feature Selection

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

### Data Flow Diagram

Raw Data → Clean → Encode → Scale → SMOTE → Select Features → Final Dataset
16.2M×80    11.9M×79    11.9M×81    -        9.7M×81        -         9.7M×45

### Memory Usage

- Cleaning: 12 GB
- Encoding: 13 GB
- Scaling: 14 GB
- SMOTE: 18.5 GB (peak)
- Selection: 16 GB
- Final: 10 GB

---

## Module 4: Model Training Architecture

### Purpose
Tune hyperparameters, train Random Forest model, analyze feature importance

### Components

#### Step 1: Data Preparation

Input: SMOTE-balanced training data (9.7M × 45)

Sampling for tuning: 20% subsample (1.9M rows)
- Reduces peak memory during hyperparameter tuning
- Still representative of full dataset

#### Step 2: Hyperparameter Tuning

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

#### Step 3: Final Model Training

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

#### Step 4: Feature Importance Analysis

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

### Output Artifacts

trained_model/
- random_forest_model.joblib (trained model)
- scaler.joblib (StandardScaler used in preprocessing)
- label_encoder.joblib (class label encoder)
- feature_importances.csv (feature names and importance scores)
- randomized_search_cv.joblib (tuning results)
- training_metadata.json (training configuration)

---

## Module 5: Testing & Evaluation Architecture

### Purpose
Evaluate model on held-out test data, calculate metrics, identify errors

### Components

#### Step 1: Load Test Data

Input: Test dataset (2.4M × 45)
- Preprocessed with same scaler and encoder as training
- Same 45 features
- 20% of original data

#### Step 2: Generate Predictions

Process:
- Feed each test sample through 150 trees
- Get prediction class (0-5)
- Get confidence (probability for predicted class)
- Decode to class name

Output:
- Predicted class per sample
- Confidence scores
- Actual class labels

#### Step 3: Calculate Metrics

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

#### Step 4: Analyze Errors

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

#### Step 5: Generate Visualizations

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

### Results Summary

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

## Data Architecture Summary

### Training Data Flow

Raw CSV Files
↓ (Module 1 - Loading)
Consolidated Parquet (16.2M × 80)
↓ (Module 2 - Exploration)
Analysis Reports
↓ (Module 3a - Cleaning)
Clean Data (11.9M × 79)
↓ (Module 3b - Encoding)
Encoded Data (11.9M × 81)
↓ (Module 3c - Scaling)
Scaled Data (11.9M × 81)
↓ (Module 3d - Train-Test Split)
Train (80%), Test (20%)
↓ (Module 3e - SMOTE)
Balanced Train (9.7M × 81)
↓ (Module 3f - Feature Selection)
Final Train (9.7M × 45), Final Test (2.4M × 45)
↓ (Module 4 - Training)
Trained Model + Artifacts
↓ (Module 5 - Testing)
Evaluation Metrics + Reports

### Feature Architecture

Original Features: 78 (from CICIDS2018)
Added (One-hot encoding): 3 (Protocol variants)
Total: 81 features

Removed (Low importance): 36 features
Selected: 45 features for model

Feature Categories:

Network Flow Features (duration, packet counts, byte counts)
Packet Size Features (length statistics: mean, std, min, max)
Timing Features (inter-arrival times, rates)
Header Features (header lengths and statistics)

### Class Architecture

Original Classes: 15
- Dominant: Benign (83.07%)
- Balanced: Attack types 3-7 (0.79% to 2.85%)
- Minority: Rare attacks (<1%)

Consolidated Classes: 6
- Benign: 88.72%
- DDoS: 6.48%
- DoS: 1.64%
- Botnet: 1.21%
- Infilteration: 1.16%
- Brute Force: 0.79%

Imbalance Handling:
- SMOTE: Oversample minority classes
- Class weights: Balanced_subsample during training
- Stratified split: Maintain proportions in train/test

