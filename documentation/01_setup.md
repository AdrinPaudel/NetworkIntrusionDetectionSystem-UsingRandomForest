# Setup & Data Download Guide

For: Someone who just cloned from GitHub and needs to get the system running

---

## 🚀 Step 1: Initial Setup (5 minutes)

### A. Create Virtual Environment

Windows:
Navigate to your project directory (can be D:\Nids, E:\MyProject, C:\Users\Name\Nids, etc.)

cd your_project_directory
python -m venv venv
.\venv\Scripts\activate

Linux/Mac:
Navigate to your project directory

cd /path/to/your/project
python3 -m venv venv
source venv/bin/activate

You should see (venv) at the start of your terminal prompt.

### B. Install Dependencies

Both Windows and Linux/Mac:

pip install --upgrade pip
pip install -r requirements.txt
pip list

Expected packages:
pandas, numpy, scikit-learn, imbalanced-learn
matplotlib, seaborn
scapy (for network monitoring)
joblib (for model serialization)

---

## 📊 Step 2: Get the Data

### A. Where to Download

Source: CICIDS2018 Dataset (publicly available)

Option 1: Direct Download (Recommended)
Website: https://www.unb.ca/cic/datasets/ids-2018.html
File: CIC-IDS2018-Aggregated-Flows.zip
Size: ~10 GB
Contains: 10 CSV files with network flow data

Option 2: Alternative Source
Kaggle: https://www.kaggle.com/datasets/solarmora/cicids2018
Community: Various research repositories maintain copies

### B. Which CSV Files You Need

After downloading and extracting, you'll have files like:
Friday-02-03-2018_TrafficForML_CICFlowMeter.csv (Large ~7M rows)
Friday-16-02-2018_TrafficForML_CICFlowMeter.csv
Friday-23-02-2018_TrafficForML_CICFlowMeter.csv
... Thursday, Wednesday files ...

Files: 10 CSV files total
One of them has ~7 million rows
Others: One has 600k, another 300k, remaining 7 have 1M rows each

### C. Where to Place Them

Your project directory structure:
your_project_directory/
  └─ data/
      └─ raw/                    (PUT ALL 10 CSV FILES HERE)
          ├─ Friday-02-03-2018_TrafficForML_CICFlowMeter.csv
          ├─ Friday-16-02-2018_TrafficForML_CICFlowMeter.csv
          ├─ ... (8 more files)
          └─ Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv

---

## 🔧 Step 3: Handle Extra Columns (CRITICAL!)

### A. The Problem

The 10 CSV files may have different columns. Some files have extra columns that don't exist in others.

What to do:
Check which columns are in ALL 10 files
Remove any columns that only exist in one or a few files
Keep only the common columns that appear in ALL files

### B. Solution: Auto-Clean Script

A script is provided in the setup/ folder: standardize_csv_columns.py

Run this BEFORE running Module 1:

python setup/standardize_csv_columns.py

This script will:
Identify common columns across all CSV files
Remove any columns that aren't in all files
Standardize all CSV files to have identical columns
Create backup files before modifying

### C. Manual Check (Optional)

Check that all CSV files have the same number of columns:

python setup/check_csv_columns.py

---

## 📁 Verify Your Setup

After setup, your directory should look like:

Windows:

your_project_directory/
├─ venv/                         (Virtual environment)
├─ setup/                        (Setup scripts)
├─ data/
│   ├─ raw/                      (10 CSV files - standardized)
│   ├─ preprocessed/             (Created by Module 3)
│   └─ realtrafficsimul/
├─ src/
├─ prediction/
├─ trained_model/               (Created by Module 4)
├─ config.py
├─ main.py
├─ predict_cli.py
└─ requirements.txt

Linux/Mac:

/path/to/your/project/
├─ venv/                        (Virtual environment)
├─ setup/                       (Setup scripts)
├─ data/
│   ├─ raw/                     (10 CSV files - standardized)
│   ├─ preprocessed/            (Created by Module 3)
│   └─ realtrafficsimul/
├─ src/
├─ prediction/
├─ trained_model/              (Created by Module 4)
├─ config.py
├─ main.py
├─ predict_cli.py
└─ requirements.txt

---

## ⚡ Quick Startup Checklist

- [ ] Virtual environment created and activated
- [ ] Dependencies installed: pip install -r requirements.txt
- [ ] 10 CSV files in data/raw/
- [ ] CSV columns standardized: python setup/standardize_csv_columns.py
- [ ] All CSVs verified: python setup/check_csv_columns.py
- [ ] Ready to run: python main.py --module 1

---

## 🚨 Troubleshooting Setup

Issue: Python not found

Windows: Add Python to PATH or use full path: C:\Python310\python.exe
Linux/Mac: Install Python: sudo apt install python3 or brew install python3

Issue: venv command not recognized

Use: python -m venv venv (not just venv)

Issue: pip install fails

1. pip install --upgrade pip
2. pip install -r requirements.txt --no-cache-dir
3. Check internet connection

Issue: Module not found after pip install

1. Verify venv is activated (check (venv) in terminal)
2. Run: pip list (should show installed packages)
3. Reinstall: pip install -r requirements.txt

Issue: Out of memory during Module 3

See 3_CONFIGURATION_AND_PERFORMANCE.md - Reduce SMOTE targets and other settings

Issue: CSV files not found

Windows: Check path is your_project_directory/data/raw/
Linux/Mac: Check path is /path/to/project/data/raw/
Verify: python -c "import os; print(os.listdir('data/raw'))"

Issue: Permission denied on scapy (real-time monitoring)

Windows: Run terminal as Administrator
Linux/Mac: Use sudo or configure capabilities

---

## 🎯 Next Steps

Once setup is complete:

1. Read: 2_PROCESS_WORKFLOW_DETAILED.md
   Understand what each module does step-by-step

2. Configure: 3_CONFIGURATION_AND_PERFORMANCE.md
   Adjust for your system's memory/CPU

3. Run: python main.py --module 1
   Start with Module 1 (data loading)

4. Monitor: Check reports/ directory for generated reports

---

## 📋 Data Summary

File | Size | Rows | Purpose
---|---|---|---
Friday-02-03 | 3.5 GB | 7.0M | Largest dataset (benign + attacks)
Friday-16-02 | 1.2 GB | 2.3M | DDoS attacks
Friday-23-02 | 800 MB | 1.5M | DoS attacks
Thursday-01-03 | 500 MB | 900k | Brute force attacks
Thursday-15-02 | 600 MB | 1.1M | Mixed attacks
Thursday-22-02 | 700 MB | 1.3M | Botnet traffic
Wednesday-14-02 | 450 MB | 800k | Web attacks
Wednesday-21-02 | 550 MB | 1.0M | Infrastructure attacks
Wednesday-28-02 | 400 MB | 750k | Infiltration attempts
Total | ~10 GB | ~18M | Complete CICIDS2018

---

## ✅ You're Ready!

Once this checklist is complete, you have:
Virtual environment with all dependencies
Clean, standardized CSV data
Correct directory structure
Ready to run training pipeline

Next: Run Module 1 to start training!

python main.py --module 1
