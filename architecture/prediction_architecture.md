# Prediction Architecture

Complete technical architecture for the prediction system

---

## Overview

The prediction system accepts input in 4 modes and generates predictions and alerts

Input → Validation → Preprocessing → Inference → Alert Classification → Report Generation

---

## System Components

### Core Components

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

### Supporting Files

- predict_cli.py: Command-line interface for all prediction modes
- config.py: Configuration parameters (alert thresholds, etc.)

---

## Prediction Workflow

### Mode 1: Batch Prediction

Input Data (CSV File)
↓
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
↓
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
↓
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
↓
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

### Input Data Formats

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

### Feature Transformation

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

### Model Inference

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

### Alert Determination

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
├── batch_YYYYMMDD_HHMMSS/ (batch prediction session)
│   ├── report.txt (summary)
│   ├── attacks.log (detailed attacks)
│   └── statistics.json (threat stats)
├── realtime_YYYYMMDD_HHMMSS/ (real-time monitoring)
│   ├── report.txt
│   ├── attacks.log
│   └── statistics.json
└── simulated_YYYYMMDD_HHMMSS/ (simulated session)
    ├── report.txt
    ├── attacks.log
    └── statistics.json

### Report Content

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

### Output Example

Console Output (Batch):
Total samples: 100,000
Attacks detected: 2,451 (2.45%)
Benign traffic: 97,549 (97.55%)

Attack breakdown:
  DDoS: 1,200 (49.0%)
  DoS: 800 (32.7%)
  Brute Force: 451 (18.4%)

Files saved:
  report.txt: results/batch_20260129_120000/report.txt
  attacks.log: results/batch_20260129_120000/attacks.log
  statistics.json: results/batch_20260129_120000/statistics.json

---

## Model Artifact Architecture

### Required Artifacts

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

### Loading Process

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

### Inference Speed

Single sample: <1 millisecond
Batch (1,000 samples): ~2 milliseconds
Speed: ~475,000 samples/second on typical hardware

Why fast:
- Random Forest prediction: Simple voting algorithm
- No deep learning overhead
- Model compact (150 trees, depth 20)
- Minimal preprocessing per sample

### Memory Usage

Model in memory: ~300 MB (model + scaler + encoder)
Per-batch memory: Depends on batch size
- Batch 1,000: ~1 MB
- Batch 10,000: ~10 MB
- Batch 100,000: ~100 MB

For real-time: Minimal (process one flow at a time)

### Accuracy

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

### Production Setup

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

### Scaling Considerations

Batch Predictions:
- Linear scaling with data size
- Memory-bounded by batch size
- Can process GB of data with small batches

Real-Time:
- Depends on packet rate
- Can handle millions of flows/day
- Bottleneck: Network packet capture, not prediction

---

## Integration Points

### Input Integration

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

### Output Integration

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

