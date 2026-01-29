# Prediction Module - Setup, Workflow & Configuration

Complete guide to setting up and using the prediction system

---

## Setup & Installation

### Prerequisites

- Trained Random Forest model (from Module 4)
- Preprocessing artifacts (scaler, label encoder, feature mapping)
- Python 3.8+ with dependencies installed
- Network interface available (for real-time monitoring mode)

### Required Files

Verify these files exist in trained_model/:
- random_forest_model.joblib (trained model)
- scaler.joblib (StandardScaler for feature normalization)
- label_encoder.joblib (class label encoder)

Verify these files exist in data/preprocessed/:
- feature_importances.csv (feature names and importance)

### Directory Structure

prediction/
├── __init__.py
├── predict_preprocess.py (data preprocessing for predictions)
├── predictor.py (model inference)
├── realtime_monitor_clean.py (real-time network monitoring)
├── realtime_simulation.py (simulated traffic testing)
├── report_generator.py (output report generation)
└── threat_action_handler.py (alert handling)

### Installation Steps

1. Ensure model is trained:
python main.py --full

2. Verify prediction dependencies:
pip list | grep -E "scikit-learn|pandas|numpy|joblib|imbalanced"

3. Test prediction CLI:
python predict_cli.py --features

If successful, shows 45 required features and system ready

---

## Prediction Workflow

### Overall Data Flow

Input Data → Validation → Preprocessing → Model Inference → Alert Classification → Report Generation

### Step-by-Step Process

#### Step 1: Load Configuration
- Load trained model from trained_model/random_forest_model.joblib
- Load scaler from trained_model/scaler.joblib
- Load label encoder from trained_model/label_encoder.joblib
- Load feature list from data/preprocessed/feature_importances.csv

#### Step 2: Receive Input
Depending on prediction mode:
- Batch: Load CSV file with network flows
- Single: Get feature values from user input
- Real-Time: Capture packets from network interface
- Simulated: Load pre-recorded flows from CICIDS2018 data

#### Step 3: Data Validation
- Check all 45 required features present
- Identify missing columns
- Map alternate column names to standard names
- Report any missing features

#### Step 4: Data Preprocessing
- Fill missing values with 0
- Encode categorical features using saved label encoder
- Scale numeric features using saved StandardScaler
- Ensure feature order matches training

#### Step 5: Feature Transformation
- Apply saved StandardScaler normalization
- Result: Features with mean=0, std=1 (matching training)

#### Step 6: Model Prediction
- Feed preprocessed features to Random Forest model
- Get prediction class (0-5)
- Get prediction probability/confidence (0.0-1.0)
- Decode class index to class name (Benign, DDoS, DoS, etc.)

#### Step 7: Alert Classification
- Evaluate prediction confidence
- Assign alert level:
  - RED: >= 50% confidence (critical attack)
  - YELLOW: 25-50% confidence (suspicious)
  - GREEN: < 25% confidence (benign, no alert)

#### Step 8: Report Generation
- Log results to file
- Generate summary statistics
- Create threat summary report
- Save to results/ folder with timestamp

---

## Configuration Parameters

### Model Configuration

Model file location: trained_model/random_forest_model.joblib
Scaler file location: trained_model/scaler.joblib
Label encoder file location: trained_model/label_encoder.joblib

Model specifications (from training):
- Algorithm: Random Forest
- Trees: 150
- Depth: 20
- Features: 45 selected features
- Classes: 6 (Benign, Botnet, Brute Force, DDoS, DoS, Infilteration)

### Feature Configuration

Required features: 45
Total available features: 78 (45 selected, 33 dropped)

Dropped features (not used):
- Src Port (too variable, not discriminative)
- Flow ID (metadata)
- Timestamp (metadata)
- Source/Destination IPs (metadata)

Selected 45 features based on importance threshold: 0.005

### Alert Thresholds

RED_THRESHOLD = 0.50
Confidence >= 50% = RED alert (critical attack detected)
Alternative: Adjust to 0.30 (more alerts), 0.70 (fewer alerts)

YELLOW_THRESHOLD = 0.25
Confidence 25-50% = YELLOW alert (suspicious)
Alternative: Adjust to 0.10 (more warnings), 0.40 (fewer warnings)

GREEN_THRESHOLD < 0.25 (implicit)
Confidence < 25% = GREEN (benign, no alert)

Alert distribution depends on model confidence output:
- If most predictions are RED: Model is confident
- If many YELLOW: Model is uncertain
- If many GREEN: Model sees benign traffic

### Prediction Preprocessing Configuration

Missing value handling: Fill with 0
Categorical encoding: Label encoding (from training)
Feature scaling: StandardScaler (mean=0, std=1)
Feature order: Must match training feature order

### Batch Processing Configuration

BATCH_SIZE = 1000
Process predictions in batches of 1000 rows to manage memory
Alternative: Adjust to 500 (more memory-efficient), 2000 (faster)

### Real-Time Monitoring Configuration

PACKET_TIMEOUT = 30
Wait maximum 30 seconds per packet capture
Alternative: Adjust based on network speed

FLOW_TIMEOUT = 60
Consider flow complete after 60 seconds of inactivity
Alternative: Adjust based on monitoring requirements

### Output Configuration

Results directory: results/
Reports saved with timestamp: results/batch_YYYYMMDD_HHMMSS/
Real-time sessions: results/realtime_YYYYMMDD_HHMMSS/
Simulated sessions: results/simulated_YYYYMMDD_HHMMSS/

Report files generated:
- report.txt (summary)
- attacks.log (detailed attacks - RED and YELLOW only)
- statistics.json (threat statistics)

---

## Prediction Modes

### Mode 1: Batch Prediction

Command:
python predict_cli.py --batch data/sample/prediction_sample.csv

Input: CSV file with network flows
Output: Predictions, confidence scores, alert levels, reports

Process:
1. Load CSV file
2. Validate all 45 features present
3. Preprocess all rows
4. Generate predictions for all rows
5. Classify alerts (RED/YELLOW/GREEN)
6. Generate summary report
7. Save results to results/ folder

Requirements:
- CSV must have all 45 required features
- Column names auto-mapped from variants
- Missing values filled with 0
- Extra columns ignored

### Mode 2: Single Prediction

Command:
python predict_cli.py --single

Input: User-provided feature values (interactive)
Output: Single prediction with confidence and alert level

Process:
1. Display all 45 required features
2. Prompt user to enter feature values
3. Validate and preprocess input
4. Generate prediction
5. Show result and alert level
6. Offer option to predict another flow

Use case: Test predictions on custom data or single network flows

### Mode 3: Real-Time Monitoring

Command:
python predict_cli.py --monitor [--interface eth0] [--duration 300]

Input: Live network packet capture from selected interface
Output: Real-time threat detection with alerts and reports

Process:
1. Select network interface (auto-detect or specify)
2. Start packet capture
3. Extract network flows from packets
4. Calculate flow features from packet data
5. Generate predictions on live flows
6. Alert on detected threats (RED/YELLOW)
7. Log alerts to file
8. Stop after specified duration (default: 5 minutes)
9. Generate summary report

Configuration:
- Default duration: 300 seconds (5 minutes)
- Default rate: All packets (real-time)
- Alternate: --duration 600 (10 minutes), --interface wlan0 (WiFi)

### Mode 4: Simulated Traffic

Command:
python predict_cli.py --simulate [--duration 180] [--rate 5]

Input: Pre-recorded flows from CICIDS2018 dataset
Output: Simulated threat detection as if live

Process:
1. Load sample flows from realtraffic_for_prediction.csv
2. Simulate traffic arrival at specified rate
3. For each flow: preprocess, predict, classify alert
4. Log alerts as if real-time
5. Accumulate statistics
6. Stop after specified duration (default: 3 minutes)
7. Generate summary report

Configuration:
- Default duration: 180 seconds (3 minutes)
- Default rate: 5 flows/second
- Alternate: --duration 600 (10 minutes), --rate 10 (10 flows/sec)

Use case: Test prediction system without requiring live network traffic

### Mode 5: Show Required Features

Command:
python predict_cli.py --features

Output: List of 45 required features for batch/single prediction

Use case: Determine what columns needed in CSV file or what values to enter

---

## Input Requirements

### Required Features (45 total)

All predictions require these 45 features. CSV column names can vary; system auto-maps:

Network Flow Features:
- Dst Port (destination port)
- Protocol (transport protocol: TCP, UDP, etc.)
- Tot Fwd Pkts (total forward packets)
- Tot Bwd Pkts (total backward packets)
- TotLen Fwd Pkts (total forward bytes)
- TotLen Bwd Pkts (total backward bytes)
- (40 more flow-level statistics)

Timing Features:
- Flow Duration (milliseconds)
- Fwd IAT Tot (inter-arrival time total)
- Flow IAT Max, Mean, Min (arrival time statistics)
- (more timing features)

Packet Size Features:
- Pkt Len Max, Min, Mean, Std (packet length statistics)
- Fwd Pkt Len (forward packet lengths)
- Bwd Pkt Len (backward packet lengths)
- (more packet size features)

Header Features:
- Fwd Header Len (forward header length)
- Bwd Header Len (backward header length)
- (more header statistics)

All features are numeric (float or int)

### Column Name Mapping

Standard CICIDS2018 names are used, but system accepts variations:

Standard Name               Accepted Variations
Dst Port                   destination_port, dest_port, dstport
Protocol                   protocol, proto
Tot Fwd Pkts              total_fwd_packets, packets_forward
Tot Bwd Pkts              total_bwd_packets, packets_backward
TotLen Fwd Pkts           total_fwd_bytes, bytes_forward
(and so on for all 45 features)

### Missing Value Handling

If values missing: Filled with 0
If entire column missing: Error reported, prediction aborts
If extra columns present: Ignored (only 45 required columns used)

---

## Output & Reports

### Batch Prediction Output

Summary printed to console:
- Total samples processed
- Attacks detected (count and percentage)
- Benign traffic (count and percentage)
- Attack breakdown by type

Files saved to results/batch_YYYYMMDD_HHMMSS/:
- report.txt: Summary report
- attacks.log: Detailed log of RED and YELLOW alerts
- statistics.json: Threat statistics in JSON format

### Real-Time Monitoring Output

Live console output:
- Flow count and alert status
- Attack type detected
- Alert level (RED/YELLOW)
- Confidence score

Session directory: results/realtime_YYYYMMDD_HHMMSS/

Files saved:
- report.txt: Session summary
- attacks.log: All RED and YELLOW alerts during monitoring
- statistics.json: Threat statistics

### Simulated Traffic Output

Summary printed to console:
- Total flows simulated
- Attacks detected
- Benign traffic
- Attack breakdown by type

Files saved to results/simulated_YYYYMMDD_HHMMSS/:
- report.txt: Simulation summary
- attacks.log: Detected attacks log
- statistics.json: Threat statistics

### Alert Log Format

attacks.log contains:
Timestamp | Source | Dest | Protocol | Alert Level | Class | Confidence

Example:
2026-01-29 12:34:56 | 192.168.1.100 | 10.0.0.5 | TCP | RED | DDoS | 0.9847

---

## Troubleshooting

### Missing Required Features

Error: "Column 'Dst Port' not found"

Solution:
1. Check CSV has all 45 required columns
2. Run: python predict_cli.py --features (shows required columns)
3. Map alternate column names to standard names
4. Ensure no typos in column headers

### Model Not Found

Error: "File not found: trained_model/random_forest_model.joblib"

Solution:
1. Train model: python main.py --full
2. Verify trained_model/ folder exists
3. Check model files present (random_forest_model.joblib, scaler.joblib)

### Preprocessing Mismatch

Error: "Feature scaling failed - scaler not compatible"

Solution:
1. Verify scaler.joblib matches current model
2. Delete and retrain: python main.py --full
3. Ensure same preprocessing pipeline used for training and prediction

### Memory Error During Batch Prediction

Error: "MemoryError - insufficient RAM"

Solution:
1. Reduce batch size (BATCH_SIZE parameter)
2. Process smaller CSV files
3. Use simulated mode instead (lower memory)
4. Upgrade system RAM if needed

### No Alerts Detected

Issue: All predictions GREEN (benign), no RED/YELLOW alerts

Possible causes:
1. Input data is mostly benign traffic
2. Alert thresholds too high (adjust RED_THRESHOLD lower)
3. Model confidence low for certain traffic patterns

Solution:
1. Test with known attack data: python predict_cli.py --simulate
2. Lower alert thresholds temporarily for testing
3. Verify model was trained on similar network patterns
