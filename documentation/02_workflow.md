# Detailed Process Workflow & System Internals

For: Understanding HOW everything works internally (process flows without detailed results)

---

## Module 1: Data Loading

### What Happens When You Run: `python main.py --module 1`

Step 1: Initialization
- Read configuration settings
- Create required directories
- Log session start

Step 2: File Discovery
- Scan data/raw/ directory
- Find all CSV files
- Validate files are readable

Step 3: Memory Planning
- Calculate total data size
- Check available system RAM
- Plan loading strategy with thread workers

Step 4: Parallel Loading
- Use multiple worker threads to load CSV files in parallel
- Load files in chunks to manage memory
- Optimize data types for memory efficiency

Step 5: Data Combination
- Merge all loaded data into single DataFrame
- Save to parquet format for fast loading next time

Step 6: Validation & Logging
- Verify data loaded successfully
- Log completion to reports/preprocessing/preprocessing_steps.txt
- Record number of rows, columns, and file size

---

## Module 2: Data Exploration

### What Happens When You Run: `python main.py --module 2`

Step 1: Check What Data Exists
- Look for preprocessed data from Module 1
- Check if exploration already completed
- Skip if already done, run if missing

Step 2: Load Data into Memory
- Read parquet file from Module 1
- Convert columns to correct data types
- Load dataset into memory for analysis

Step 3: Analyze Class Distribution
- Count unique classes in Label column
- Calculate distribution percentages
- Identify class imbalance
- Generate visualization and report

Step 4: Check for Missing Data
- Find NaN and Inf values
- Count missing data per column
- Assess data quality
- Generate report

Step 5: Calculate Feature Statistics
- For each numeric column: calculate mean, median, std dev, min, max
- Generate statistical summary table
- Report overall data statistics

Step 6: Analyze Correlations
- Calculate correlation matrix between features
- Identify highly correlated feature pairs
- Generate correlation heatmap visualization
- Report correlation findings

Step 7: Generate Reports
- Create exploration_results.txt with summary
- Save visualizations to reports/exploration/
- Log completion status

---

## Module 3: Data Preprocessing

### What Happens When You Run: `python main.py --module 3`

Step 1: Data Cleaning
- Remove rows with NaN values
- Remove Inf values
- Remove duplicate rows
- Remove outliers
- Report cleaned data statistics

Step 2: Label Consolidation
- Consolidate original attack classes into simplified class set
- Map original labels to new simplified labels
- Verify consolidation complete
- Report class mapping

Step 3: Feature Encoding
- Encode categorical features
- Handle string values
- Convert to numeric representation
- Generate feature name mappings

Step 4: Train-Test Split
- Divide data into training and testing sets
- Use 80-20 split ratio
- Ensure balanced distribution in both sets
- Report split statistics

Step 5: Feature Scaling
- Normalize numeric features using StandardScaler
- Ensure features have mean=0, std=1
- Save scaler for later prediction use
- Report scaling completion

Step 6: SMOTE Oversampling
- Apply SMOTE to training data
- Balance minority classes by generating synthetic samples
- Create balanced training dataset
- Report oversampling statistics

Step 7: Feature Selection
- Calculate feature importance using Random Forest
- Identify top features by importance
- Select features above importance threshold
- Report selected feature count

Step 8: Create Preprocessed Datasets
- Save processed training data
- Save processed testing data
- Save label encoder for later use
- Save scaler for predictions
- Log completion

---

## Module 4: Model Training

### What Happens When You Run: `python main.py --module 4`

Step 1: Hyperparameter Tuning
- Load preprocessed training data
- Define hyperparameter search space
- Run RandomizedSearchCV for optimization
- Evaluate different parameter combinations
- Report best parameters found

Step 2: Train Random Forest Model
- Use best hyperparameters from tuning
- Train Random Forest classifier
- Report training completion

Step 3: Feature Importance Analysis
- Extract feature importance scores from trained model
- Rank features by importance
- Generate importance visualization
- Save importance scores to CSV

Step 4: Model Validation
- Validate model on training data
- Calculate training accuracy and F1-score
- Report validation metrics

Step 5: Save Model Artifacts
- Save trained Random Forest model
- Save hyperparameter tuning results
- Save training metadata and configuration
- Log completion

---

## Module 5: Model Testing

### What Happens When You Run: `python main.py --module 5`

Step 1: Load Test Data & Model
- Load test dataset from Module 3
- Load trained model from Module 4
- Load preprocessing artifacts (scaler, encoder)
- Verify all components loaded successfully

Step 2: Generate Predictions
- Run model predictions on entire test set
- Get prediction probabilities for each class
- Report prediction generation completion

Step 3: Calculate Classification Metrics
- Calculate accuracy on test data
- Calculate precision, recall, F1-score for each class
- Calculate macro and weighted averages
- Calculate macro F1-score across all classes

Step 4: Generate Confusion Matrix
- Create confusion matrix showing prediction vs actual
- Identify misclassified samples
- Visualize confusion matrix as heatmap
- Save visualization to reports/training/

Step 5: Generate ROC Curves
- Calculate ROC curves for each class
- Calculate Area Under Curve (AUC) metrics
- Visualize ROC curves
- Save visualizations to reports/training/

Step 6: Generate Classification Reports
- Create detailed classification report with all metrics
- Include per-class precision, recall, F1-score
- Report overall model performance
- Save to reports/training/testing_results.txt

Step 7: Compare with Training Results
- Compare test performance to training performance
- Assess for overfitting
- Report comparison analysis

Step 8: Finalize Results
- Log all test results and metrics
- Save summary to reports/training/testing_results.txt
- Mark pipeline completion
- Report overall model performance and readiness for prediction

---

## Prediction Module Workflow

### Batch Prediction: `python predict_cli.py --batch data.csv`

Step 1: Load Model & Configuration
- Load trained Random Forest model
- Load preprocessing artifacts (scaler, encoder, feature mapping)
- Load required feature list

Step 2: Load Input CSV
- Read CSV file
- Parse column headers
- Validate file structure

Step 3: Feature Validation & Mapping
- Check CSV has required columns
- Map input columns to feature names
- Handle alternate column names
- Report any missing columns

Step 4: Preprocessing
- Fill missing values
- Encode categorical features using saved encoder
- Scale features using saved scaler
- Ensure feature order matches training

Step 5: Generate Predictions
- Run model inference on preprocessed data
- Get prediction class for each row
- Get confidence probability for each prediction
- Report prediction generation

Step 6: Generate Report
- Create prediction results with class and confidence
- Calculate alert counts per threat level
- Generate summary statistics
- Save report to results/ folder with timestamp

---

### Single Prediction: `python predict_cli.py --single`

Step 1: Initialize Interactive Mode
- Load trained model and preprocessing artifacts
- Display required features list
- Prompt for input method

Step 2: Get User Input
- Accept feature values from user
- Request all required features
- Validate input values (numeric/valid range)

Step 3: Preprocess Input
- Encode categorical values
- Scale numeric features
- Create feature vector

Step 4: Generate Prediction
- Run model on single flow
- Get predicted class
- Get confidence probability

Step 5: Display Result
- Show predicted threat classification
- Show confidence level
- Show alert level (RED/YELLOW/GREEN)
- Offer option to predict another flow

---

### Real-Time Monitoring: `python predict_cli.py --monitor`

Step 1: Initialize Network Capture
- Select network interface
- Set capture parameters
- Begin packet capture

Step 2: Extract Network Flows
- Capture packets from network interface
- Group packets into network flows
- Extract flow-level features from packets

Step 3: Feature Extraction
- Calculate flow statistics from packets
- Build feature vector from flow data
- Map to required features

Step 4: Generate Predictions
- Run model on extracted features
- Get prediction and confidence
- Classify as benign or threat

Step 5: Alert & Report
- Check alert threshold
- Generate alert for threats (RED/YELLOW)
- Log prediction and confidence
- Update real-time statistics

Step 6: Generate Reports
- Accumulate predictions over monitoring duration
- Calculate summary statistics
- Generate report with threat summary
- Save to results/ folder

Step 7: Complete Monitoring
- Stop packet capture
- Finalize report
- Display summary to user

---

### Simulated Traffic: `python predict_cli.py --simulate`

Step 1: Load Simulation Data
- Load sample data from CICIDS2018 dataset
- Select flows for simulation

Step 2: Simulate Traffic Playback
- Read flows one at a time
- Simulate flow arrival at specified rate
- Space flows over simulation duration

Step 3: Process Each Flow
- Extract features from flow data
- Preprocess using saved artifacts
- Generate prediction

Step 4: Classify & Alert
- Get prediction and confidence
- Assign alert level
- Log prediction and threat status

Step 5: Generate Statistics
- Count benign and threat flows
- Calculate threat percentage
- Track alert levels (RED/YELLOW/GREEN)

Step 6: Generate Report
- Compile summary of simulation
- Include flow counts and classifications
- Include statistics by threat type
- Save to results/ folder with timestamp

Step 7: Display Summary
- Show simulation results
- Show threat summary
- Show flow classification breakdown
