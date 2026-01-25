# NIDS CICIDS2018 Project

Network Intrusion Detection System using Random Forest on CICIDS2018 Dataset

## Project Status

**Phase 1 (Completed):**
- ✅ Module 1: Data Loading
- ✅ Module 2: Data Exploration

**Phase 2 (To be implemented):**
- ⏳ Module 3: Data Preprocessing
- ⏳ Module 4: Model Training
- ⏳ Module 5: Model Testing

## Requirements

- Python 3.8+
- 32 vCPU, 208GB RAM
- ~50GB free disk space

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Dataset Setup

Place your 10 CICIDS2018 CSV files in the `data/raw/` directory:
```
data/raw/
├── 02-14-2018.csv
├── 02-15-2018.csv
├── 02-16-2018.csv
├── 02-20-2018.csv
├── 02-21-2018.csv
├── 02-22-2018.csv
├── 02-23-2018.csv
├── 02-28-2018.csv
├── 03-01-2018.csv
└── 03-02-2018.csv
```

## Usage

### Run Full Pipeline (Phase 1)
```bash
python main.py --full
```

### Run Specific Modules
```bash
# Run Module 1 only (Data Loading)
python main.py --module 1

# Run Module 2 only (Data Exploration)
python main.py --module 2

# Run Modules 1 and 2
python main.py --module 1 --module 2
```

## Output Structure

```
nids_cicids2018_project/
├── data/
│   ├── raw/              # Your 10 CSV files
│   └── preprocessed/     # Processed data (Phase 2)
├── reports/
│   ├── exploration/      # Phase 1 reports ✅
│   │   ├── exploration_results.txt
│   │   ├── class_distribution.png
│   │   ├── class_imbalance_log_scale.png
│   │   ├── correlation_heatmap.png
│   │   ├── missing_data_summary.png
│   │   └── data_types_memory.png
│   ├── preprocessing/    # Phase 2 reports
│   ├── training/         # Phase 2 reports
│   └── testing/          # Phase 2 reports
└── trained_model/        # Trained model (Phase 2)
```

## Expected Performance

- **Target Macro F1-Score:** >96%
- **Target Accuracy:** >99%
- **Processing Time (Phase 1):** ~5-10 minutes
- **Memory Usage:** ~10-15 GB during Phase 1

## Project Configuration

All settings can be modified in `config.py`:
- Data paths
- Hyperparameters
- Feature selection settings
- SMOTE configuration
- And more...

## Detailed Logging

All operations are logged to:
1. **Console:** Real-time colored output with timestamps
2. **Report files:** Comprehensive text reports with all statistics

## Phase 1 Outputs

After running Phase 1, you will have:

### Console Output:
- Detailed loading statistics
- Class distribution analysis
- Data quality assessment
- Correlation analysis
- Memory usage breakdown

### Files Generated:
- `reports/exploration/exploration_results.txt` - Complete analysis report
- `reports/exploration/*.png` - 5 visualization charts

## Next Steps

Phase 2 will implement:
1. **Module 3:** Data preprocessing (cleaning, SMOTE, RFE)
2. **Module 4:** Model training with hyperparameter tuning
3. **Module 5:** Model evaluation and testing

## Support

For issues or questions, refer to the detailed specification in `DetailedStepsReport.md`
