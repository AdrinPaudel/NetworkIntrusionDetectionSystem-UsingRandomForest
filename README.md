# NIDS CICIDS2018 - Network Intrusion Detection System

Production-grade Network Intrusion Detection System using the CICIDS2018 dataset with machine learning-based attack classification and real-time threat detection.

---

## Quick Start (5 Minutes)

Prerequisites: Python 3.8+, 200GB disk space, 16GB+ RAM

Install dependencies:
pip install -r requirements.txt

Try a prediction:
python predict_cli.py --simulate --duration 60

---

## What This Project Does

Detects and classifies network attacks using Random Forest machine learning:
- 6-Class Detection: Benign, DDoS, DoS, Brute Force, Botnet, Infilteration
- 98.82% Accuracy: On test set evaluation
- 4 Prediction Modes: Batch, Single, Real-Time, Simulated
- Real-Time Alerts: RED (critical), YELLOW (warning), GREEN (safe)

---

## Key Features

Complete ML Pipeline: Data loading → exploration → preprocessing → training → testing
Hyperparameter Tuning: Automated parameter search (15 combinations tested)
Class Balancing: SMOTE for handling imbalanced attack types
Feature Selection: 45 most important features from 78 original
Production Ready: Fast inference (475k samples/sec), low false positives (0.18%)

---

## Performance Metrics

Overall Accuracy: 98.82%
Macro F1-Score: 86.57%
Binary Classification F1: 94.55%

Per-attack type:
- DDoS: 99.94% F1-Score
- DoS: 99.96% F1-Score
- Brute Force: 99.75% F1-Score
- Botnet: 99.80% F1-Score
- Benign: 99.34% F1-Score

---

## Training Pipeline (5 Modules)

Module 1: Data Loading (15-30 min)
- Load 10 raw CSV files (16.2M rows)
- Parallel processing with 5 worker threads
- Output: Consolidated parquet file (11.66 GB)

Module 2: Data Exploration (5-10 min)
- Analyze dataset structure and quality
- Check class distribution
- Output: Exploration reports and visualizations

Module 3: Data Preprocessing (30-45 min)
- Clean data, consolidate labels (15 → 6)
- Train-test split (80-20)
- SMOTE class balancing
- Feature selection (45 from 78)
- Output: Preprocessed datasets

Module 4: Model Training (30-45 min)
- Hyperparameter tuning (15 combinations, 3-fold CV)
- Train Random Forest (150 trees, depth 20)
- Analyze feature importance
- Output: Trained model

Module 5: Model Testing (5-10 min)
- Evaluate on test set
- Calculate accuracy, precision, recall, F1-score
- Generate confusion matrix and ROC curves
- Output: Test results

---

## Prediction System (4 Modes)

Batch Prediction - Process CSV file
python predict_cli.py --batch data/sample/prediction_sample.csv

Single Prediction - Interactive mode
python predict_cli.py --single

Real-Time Monitoring - Live network packet capture
python predict_cli.py --monitor --duration 300 --interface eth0

Simulated Traffic - Test with CICIDS2018 data
python predict_cli.py --simulate --duration 180 --rate 5

---

## Alert System

Three alert levels based on model confidence:

RED (≥ 50% confidence) - Critical Attack
Immediate action required, full logging and notification

YELLOW (25-50% confidence) - Suspicious Activity
Requires investigation, logged for review

GREEN (< 25% confidence) - Safe/Benign
No action taken, normal network traffic

---

## Project Structure

z:\Nids\
├── main.py                          Main entry point for training
├── predict_cli.py                   Prediction command-line interface
├── config.py                        Configuration parameters
├── requirements.txt                 Python dependencies
│
├── documentation/                   Complete documentation
│   ├── INDEX.md                    Documentation guide
│   ├── 01_setup.md                 Setup and installation
│   ├── 02_workflow.md              Training workflow details
│   ├── 03_config.md                Configuration parameters
│   ├── 04_prediction.md            Complete prediction guide
│   ├── 05_results.md               Pipeline results and performance
│   └── 06_architecture.md          Complete system architecture
│
├── src/                             Training pipeline modules
│   ├── data_loader.py              Module 1 - Data loading
│   ├── explorer.py                 Module 2 - Exploration
│   ├── preprocessor.py             Module 3 - Preprocessing
│   ├── trainer.py                  Module 4 - Training
│   ├── tester.py                   Module 5 - Testing
│   └── utils.py                    Utility functions
│
├── prediction/                      Prediction system
│   ├── predict_preprocess.py       Input preprocessing
│   ├── predictor.py                Model inference
│   ├── realtime_monitor_clean.py   Real-time monitoring
│   ├── realtime_simulation.py      Traffic simulation
│   ├── report_generator.py         Report generation
│   └── threat_action_handler.py    Alert handling
│
├── data/                            Data directories
│   ├── raw/                        Raw CSV files (10 CICIDS2018)
│   ├── preprocessed/               Cleaned and processed datasets
│   ├── sample/                     Sample data for testing
│   └── realtrafficsimul/          Pre-recorded flows
│
├── trained_model/                  Trained model artifacts
│   ├── random_forest_model.joblib  Trained Random Forest
│   ├── scaler.joblib               StandardScaler object
│   ├── label_encoder.joblib        Class label encoder
│   └── feature_importances.csv     Feature importance scores
│
├── reports/                        Training reports
│   ├── exploration/               Module 2 reports
│   ├── preprocessing/             Module 3 reports
│   ├── training/                  Module 4 reports
│   ├── testing/                   Module 5 reports
│   └── realtime/                  Real-time reports
│
└── results/                        Prediction results
    ├── batch_YYYYMMDD_HHMMSS/     Batch sessions
    ├── realtime_YYYYMMDD_HHMMSS/  Real-time sessions
    └── simulated_YYYYMMDD_HHMMSS/ Simulated sessions

---

## Key Statistics

Dataset (CICIDS2018):
- Total samples: 16.2M network flows
- Raw data: 10.4 GB (10 CSV files)
- Original features: 78
- Original classes: 15
- Time period: February 2018

Training Results:
- Final dataset: 11.9M flows after cleaning
- Features selected: 45 (from 78)
- Classes consolidated: 6 (from 15)
- Training time: ~1 hour (all modules)
- Model: 150 decision trees, depth 20

Performance:
- Test accuracy: 98.82%
- Macro F1-score: 86.57%
- Binary F1-score: 94.55%
- False positive rate: 0.18%
- Inference speed: 475k samples/sec

---

## Quick Reference Commands

Training:

python main.py --full                      Run full pipeline
python main.py --module 3                  Run specific module
python main.py --module 1 --module 2 --module 3    Run multiple

Prediction:

python predict_cli.py --batch data/sample/prediction_sample.csv    Batch prediction
python predict_cli.py --single                                      Single interactive
python predict_cli.py --monitor --duration 600 --interface eth0     Real-time
python predict_cli.py --simulate --duration 300 --rate 10           Simulated
python predict_cli.py --features                                    Show required features

---

## Installation & Setup

Step 1: Clone Repository
git clone <repository>
cd z:\Nids

Step 2: Create Virtual Environment
python -m venv venv
venv\Scripts\activate

Step 3: Install Dependencies
pip install -r requirements.txt

Step 4: Download Data
See documentation/01_setup.md for CICIDS2018 dataset download instructions

Step 5: Verify Installation
python predict_cli.py --features

Step 6: Train Model (Optional)
python main.py --full
(Pre-trained model included in trained_model/)

---

## Documentation

All documentation in documentation/ folder:

- documentation/INDEX.md - Navigation guide for all documents
- documentation/01_setup.md - Setup and installation
- documentation/02_workflow.md - Training workflow details
- documentation/03_config.md - Configuration parameters
- documentation/04_prediction.md - Complete prediction guide
- documentation/05_results.md - Pipeline results and analysis
- documentation/06_architecture.md - Complete system architecture

---

## Hardware Requirements

Minimum:
- CPU: 4 cores, 2.0+ GHz
- RAM: 32 GB
- Storage: 50 GB SSD
- Time: 2-3 hours for full training

Recommended (Production):
- CPU: 16+ cores
- RAM: 200 GB
- Storage: 1 TB SSD
- Time: 1 hour for full training

For Large-Scale Prediction:
- CPU: 16+ cores
- RAM: 64+ GB
- Network: 1 Gbps+ for real-time monitoring

---

## Usage Examples

Example 1: Make a Prediction
python predict_cli.py --batch data/sample/prediction_sample.csv

Example 2: Monitor Network
python predict_cli.py --monitor --duration 300

Example 3: Test Predictions
python predict_cli.py --simulate --duration 120

Example 4: Retrain Model
python main.py --full

---

## Output & Results

Each prediction session generates:
- report.txt: Summary of threats detected
- attacks.log: Detailed log of RED and YELLOW alerts
- statistics.json: Machine-readable threat statistics

Example output:
PREDICTION SUMMARY
Total samples: 10,000
Attacks detected: 245 (2.45%)
Benign traffic: 9,755 (97.55%)

Attack breakdown:
  DDoS: 120 (49.0%)
  DoS: 82 (33.5%)
  Brute Force: 43 (17.6%)

Results saved to: results/batch_YYYYMMDD_HHMMSS/

---

## Troubleshooting

Model Not Found
Error: "File not found: trained_model/random_forest_model.joblib"
Solution: Train model with python main.py --full

Missing Features
Error: "Column 'Dst Port' not found"
Solution: Verify CSV has all 45 required features. Run python predict_cli.py --features

Memory Error
Error: "MemoryError - insufficient RAM"
Solution: Reduce batch size, process smaller CSV files, or upgrade RAM

Packet Capture Failed
Error: "Permission denied - cannot access network interface"
Solution: Run with administrator/sudo privileges

---

## More Information

For detailed information, see documentation/ folder:
- See documentation/01_setup.md for complete setup instructions
- See documentation/02_workflow.md for process workflow details
- See documentation/03_config.md for configuration parameters
- See documentation/04_prediction.md for prediction system details
- See documentation/05_results.md for pipeline results
- See documentation/06_architecture.md for system architecture
- See documentation/INDEX.md for navigation guide

---

Last Updated: January 29, 2026
Version: 1.0
Status: Production Ready
