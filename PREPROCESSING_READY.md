# Preprocessing Module Ready - Configuration Summary

**Date:** January 24, 2026  
**Status:** ✅ Ready to run Module 3 (Data Preprocessing)

---

## **1. Resource Requirements (CPU vs RAM)**

### **SMOTE (Class Imbalance Handling)**
- **Primary Resource:** RAM-intensive
- **RAM Usage:** ~2-3 GB additional (generates synthetic samples in memory)
- **CPU Usage:** Moderate (k-neighbors search)
- **Processing Time:** 12-20 minutes (estimated)
- **Your System:** 208GB RAM ✅ More than enough

### **RFE (Feature Selection)**
- **Primary Resource:** CPU-intensive
- **CPU Usage:** High (trains 80-400 Random Forest models with cross-validation)
- **RAM Usage:** Moderate (~4-5 GB during training)
- **Processing Time:** 20-30 minutes (estimated)
- **Your System:** 32 vCPU ✅ Excellent parallelization

### **Other Cleaning Operations**
- **Resource:** Minimal
- **Operations:** NaN/Inf removal, duplicates, label consolidation, encoding, scaling
- **Processing Time:** ~3-5 minutes total
- **Your System:** Trivial with your specs

### **Total Estimated Time:** ~40-55 minutes for full preprocessing pipeline

---

## **2. Configuration Settings (config.py)**

### **RFE Settings - MODERATE (Focused on F1)**
```python
ENABLE_RFE = True                    # ✅ Enabled
RFE_MIN_FEATURES = 30                # Start from 30 features
RFE_TARGET_FEATURES_MIN = 35         # Moderate target (not aggressive)
RFE_TARGET_FEATURES_MAX = 45         # Moderate target (not aggressive)
RFE_CV_FOLDS = 5                     # 5-fold cross-validation
RFE_SCORING = 'f1_macro'             # Optimized for balanced F1
```

**Strategy:** Not too aggressive (35-45 features vs all 80), focused on maximizing F1-macro score

### **SMOTE Settings - MODERATE**
```python
APPLY_SMOTE = True                   # ✅ Enabled
SMOTE_TARGET_PERCENTAGE = 0.03       # Bring minorities to 3% of dataset
SMOTE_K_NEIGHBORS = 5                # Standard k-neighbors
```

**Strategy:** Moderate oversampling (3% instead of aggressive 1%), balances minority classes without extreme synthetic samples

### **Scaling Settings**
```python
SCALER_TYPE = 'standard'             # StandardScaler (mean=0, std=1)
```

**Why StandardScaler?** Better for Random Forest and handles outliers better than MinMaxScaler

### **Train-Test Split**
```python
TEST_SIZE = 0.20                     # 80:20 split
STRATIFY = True                      # Maintain class proportions
RANDOM_STATE = 42                    # Reproducibility
```

---

## **3. Protocol Column Encoding**

**Found Columns:**
- `Protocol`: object type → **One-hot encode** (TCP/UDP/ICMP → binary columns)
- `Dst Port`: object type → **NOT encoded** (it's a feature, not categorical)

**Configuration:**
```python
PROTOCOL_COLUMN_CANDIDATES = ['Protocol', 'protocol', ' Protocol']
```

**Implementation:** Only `Protocol` column will be one-hot encoded, creating binary columns like:
- `Protocol_TCP`
- `Protocol_UDP`
- `Protocol_ICMP`

---

## **4. Label Consolidation Mapping (15 → 8 classes)**

**Consolidation Strategy:**

### **DDoS Variants → DDoS**
- DDoS-LOIC-HTTP
- DDoS-LOIC-UDP
- DDoS-HOIC
- All variants combined into single `DDoS` class

### **DoS Variants → DoS**
- DoS-Hulk
- DoS-GoldenEye
- DoS-Slowloris
- DoS-SlowHTTPTest
- All variants combined into single `DoS` class

### **Brute Force Variants → Brute Force**
- FTP-BruteForce
- SSH-Bruteforce
- FTP-Patator
- SSH-Patator
- All variants combined into single `Brute Force` class

### **Web Attack Variants → Web Attack**
- SQL Injection
- Brute Force -Web
- Brute Force -XSS
- All variants combined into single `Web Attack` class

### **Other Mappings**
- `Bot` → `Botnet`
- `Infilteration` (typo) → `Infiltration`
- `Benign` → `Benign` (unchanged)
- `Heartbleed` → `Heartbleed` (unchanged)

**Final 8 Classes:**
1. Benign
2. DDoS
3. DoS
4. Botnet
5. Brute Force
6. Web Attack
7. Infiltration
8. Heartbleed

---

## **5. Bad "Label" Class Handling**

**Issue:** 59 rows have label = "Label" (misplaced CSV header)

**Solution:** ✅ Automatically removed in `clean_data()` function:
```python
df = df[df['Label'] != 'Label']  # Remove bad rows
```

**Validation:** No "Label" values will remain in Label column

---

## **6. Checkpoint Files**

**4 Checkpoints Saved Throughout Pipeline:**

1. **After Cleaning:** `cleaned_data.parquet`
   - NaN, Inf, duplicates removed
   - Useless columns dropped
   
2. **After Encoding:** `train_encoded.parquet`, `test_encoded.parquet`
   - Labels consolidated (15→8)
   - Protocol one-hot encoded
   - Train-test split complete
   
3. **After SMOTE:** `train_scaled_smoted.parquet`, `test_scaled.parquet`
   - Features scaled (StandardScaler)
   - SMOTE applied to training set
   
4. **After RFE (Final):** `train_final.parquet`, `test_final.parquet`
   - Optimal features selected
   - Ready for model training

**Additional Artifacts:**
- `scaler.joblib` - StandardScaler object (reusable)
- `label_encoder.joblib` - LabelEncoder object (reusable)
- `rfe_model.joblib` - RFECV object (if RFE enabled)

---

## **7. Detailed Reports Generated**

### **Report 1: preprocessing_results.txt**
**Content:**
- Complete preprocessing summary
- All 7 steps documented:
  1. Data Cleaning (what was cleaned, before/after)
  2. Label Consolidation (15→8 mapping)
  3. Categorical Encoding (Protocol one-hot, Label encoding)
  4. Train-Test Split (stratification verification)
  5. Feature Scaling (StandardScaler, no data leakage)
  6. SMOTE (class distribution before/after, synthetic samples)
  7. RFE (optimal features, F1 score improvement)
- Final dataset summary
- Quality assessment
- Performance expectations

### **Report 2: preprocessing_steps.txt**
**Content:**
- Detailed step-by-step chronological log
- Each substep explained:
  - Before state
  - Action taken
  - After state
  - Validation checks
- Terminal-like output format
- Shows exact commands/operations
- Verifies each step completed

**Example Detail Level:**
```
[SUBSTEP 5.2] Fit Scaler on Training Data ONLY
  • Action: scaler.fit(X_train)
  • Training samples: 13,000,000
  • Features: 80
  • CRITICAL: Scaler learns statistics from TRAINING data ONLY
  • Purpose: Prevent data leakage (test data must not influence scaler)
```

---

## **8. Data Exploration - Already Complete**

**Status:** ✅ Exploration reports already exist

**Implementation:**
- `main.py` updated to check for existing exploration reports
- If found: Skip Module 2 (don't re-run exploration)
- Benefits:
  - Saves ~10 minutes
  - Avoids regenerating same reports
  - Uses existing reports as baseline

**Reports Available:**
- `reports/exploration/exploration_results.txt`
- `reports/exploration/exploration_steps.txt`
- 6 visualization files

---

## **9. How to Run**

### **Option 1: Full Pipeline (Recommended)**
```bash
cd /home/paudeladrin/Nids
python main.py --full
```

**What happens:**
1. ✅ Module 1: Load data (10 CSV files)
2. ⏭️ Module 2: Skip (already done)
3. 🔄 Module 3: Preprocessing (NEW - runs full pipeline)
   - Cleaning → Consolidation → Encoding → Split → Scaling → SMOTE → RFE
4. 📄 Generate detailed reports

### **Option 2: Module 3 Only**
```bash
cd /home/paudeladrin/Nids
python main.py --module 3
```

**What happens:**
1. Load data (Module 1 auto-runs)
2. Run preprocessing (Module 3)
3. Skip exploration (not needed)

---

## **10. Expected Output**

### **Console Output:**
```
[2026-01-24 XX:XX:XX] ========================================
[2026-01-24 XX:XX:XX]   MODULE 3: DATA PREPROCESSING
[2026-01-24 XX:XX:XX] ========================================

[2026-01-24 XX:XX:XX] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[2026-01-24 XX:XX:XX] STEP 1/7: DATA CLEANING
[2026-01-24 XX:XX:XX] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[SUBSTEP 1.1] Removing NaN values...
✓ Removed 8,123 rows
...

[2026-01-24 XX:XX:XX] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[2026-01-24 XX:XX:XX] STEP 2/7: LABEL CONSOLIDATION
...
```

### **Files Generated:**
```
data/preprocessed/
├── cleaned_data.parquet
├── train_encoded.parquet
├── test_encoded.parquet
├── train_scaled_smoted.parquet
├── test_scaled.parquet
├── train_final.parquet          ← MAIN OUTPUT
├── test_final.parquet            ← MAIN OUTPUT
├── scaler.joblib
├── label_encoder.joblib
└── rfe_model.joblib

reports/preprocessing/
├── preprocessing_results.txt     ← DETAILED REPORT
└── preprocessing_steps.txt       ← STEP-BY-STEP LOG
```

---

## **11. Next Steps After Preprocessing**

### **Module 4: Model Training**
- Load `train_final.parquet`
- Hyperparameter tuning (RandomizedSearchCV)
- Train final Random Forest model
- Save trained model

### **Module 5: Model Testing**
- Load `test_final.parquet` and trained model
- Generate predictions
- Evaluation metrics (confusion matrix, F1, accuracy, ROC)
- Final performance report

---

## **12. Summary**

✅ **RFE:** Enabled with moderate settings (35-45 features)  
✅ **SMOTE:** Enabled with moderate oversampling (3%)  
✅ **Protocol:** One-hot encoding enabled  
✅ **Label Consolidation:** 15→8 classes (all DDoS/DoS variants merged)  
✅ **Bad Labels:** Auto-removed  
✅ **Checkpoints:** 4 stages saved  
✅ **Reports:** 2 detailed reports (results + steps)  
✅ **Exploration:** Skipped (already complete)  
✅ **Ready:** Run `python main.py --full`

**Estimated Runtime:** 40-55 minutes  
**RAM Usage:** ~15-20 GB peak (you have 208 GB ✅)  
**CPU Usage:** All 32 cores utilized during RFE ✅  

---

## **13. No More Questions - Ready to Execute!**

All configurations confirmed. Pipeline ready to run.

**Command to start:**
```bash
cd /home/paudeladrin/Nids
python main.py --full
```

Good luck! 🚀
