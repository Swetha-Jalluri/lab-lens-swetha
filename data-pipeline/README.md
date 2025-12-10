# Lab Lens Data Pipeline
### MLOps Pipeline for Medical Text Processing with Bias Detection

> **Production-grade data pipeline processing MIMIC-III discharge summaries with automated bias detection and mitigation**

---

## 📊 Pipeline Overview

**5-Stage Automated Pipeline:**
```
Raw Data (9,520 records) 
    ↓ 
[1. Preprocessing] → 7,069 records (duplicates removed)
    ↓
[2. Validation] → 82% quality score
    ↓
[3. Feature Engineering] → 90 features created
    ↓
[4. Bias Detection] → 18.63% age bias detected
    ↓
[5. Bias Mitigation] → 0% bias (99.99% reduction)
```

**Performance:**
- ⏱️ Total Duration: 55.34 seconds
- 📈 Validation Score: 82% (PASS)
- 🎯 Ready for ML: YES

---

## 📁 Directory Structure

```
data-pipeline/
│
├── configs/
│   └── pipeline_config.json           # Pipeline configuration
│
├── data/
│   ├── raw/
│   │   └── mimic_discharge_labs.csv   # Raw MIMIC-III data (90.19 MB)
│   └── processed/
│       ├── processed_discharge_summaries.csv  # Cleaned data (7,069 rows)
│       ├── mimic_features.csv                 # Engineered features (90 columns)
│       └── mimic_features_mitigated.csv       # Bias-corrected dataset ⭐
│
├── scripts/                           # Processing scripts
│   ├── preprocessing.py               # Stage 1: Data cleaning (9.13s)
│   ├── validation.py                  # Stage 2: Quality checks (0.72s)
│   ├── feature_engineering.py         # Stage 3: Feature creation (42.62s)
│   ├── bias_detection.py              # Stage 4: Bias analysis (1.88s)
│   ├── automated_bias_handler.py      # Stage 5: Bias mitigation (0.99s)
│   └── main_pipeline.py               # Orchestrator (runs all stages)
│
├── tests/                             # Unit tests (58 total)
│   ├── test_preprocessing.py          # 18 tests ✓
│   ├── test_validation.py             # 13 tests ✓
│   └── test_feature_engineering.py    # 27 tests ✓
│
├── logs/                              # Output reports
│   ├── validation_report.json
│   ├── validation_summary.csv
│   ├── bias_report.json
│   ├── bias_summary.csv
│   ├── bias_mitigation_report.json
│   ├── pipeline_results_latest.json
│   ├── feature_engineering.log
│   └── bias_plots/                    # Visualizations
│
├── notebooks/
│   └── data_acquisition.ipynb         # BigQuery data extraction
│
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9.6
- 90 MB disk space
- Virtual environment

### Run Complete Pipeline (3 Commands)

**macOS:**
```bash
cd /Users/Admin/Desktop/lab-lens
source venv/bin/activate
python data-pipeline/scripts/main_pipeline.py
```

**Windows:**
```powershell
cd C:\Users\YourUsername\Desktop\lab-lens
venv\Scripts\Activate
python data-pipeline\scripts\main_pipeline.py
```

**Expected Output:**
```
============================================================
LAB LENS PIPELINE EXECUTION SUMMARY
============================================================
Status: SUCCESS
Duration: 0.92 minutes
Steps Completed: 5/5

Quality Metrics:
  Validation Score: 82.00% - PASS
  Bias Score (Before): 18.63
  Bias Score (After): 0.00

Overall Status:
  Ready for Modeling: YES
============================================================
```

---

## 🔧 Setup Instructions

### Step 1: Environment Setup

**macOS:**
```bash
# Navigate to project
cd /Users/Admin/Desktop/lab-lens

# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r data-pipeline/requirements.txt
```

**Windows PowerShell:**
```powershell
# Navigate to project
cd C:\Users\YourUsername\Desktop\lab-lens

# Create virtual environment
python -m venv venv

# Activate environment
venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt
pip install -r data-pipeline\requirements.txt
```

**Verify Installation:**
```bash
python --version     # Should show 3.9.6
pytest --version     # Should show 8.4.2+
pip list | grep pandas  # Should show pandas 2.0.0+
```

---

### Step 2: Data Acquisition

**Option A: Use Existing Data**
```bash
# Verify data file exists
ls -lh data-pipeline/data/raw/mimic_discharge_labs.csv
# Should show ~90 MB file
```

**Option B: Extract Fresh Data from BigQuery**
```bash
# Install Jupyter
pip install jupyter ipykernel notebook

# Launch notebook
jupyter notebook data-pipeline/notebooks/data_acquisition.ipynb

# In notebook: Execute all cells
# This will:
# 1. Authenticate with Google Cloud
# 2. Query MIMIC-III database
# 3. Extract 9,520 discharge summaries
# 4. Save to data-pipeline/data/raw/mimic_discharge_labs.csv
```

**Required for BigQuery:**
```bash
pip install google-cloud-bigquery google-auth-oauthlib db-dtypes
```

---

## 🎯 Running the Pipeline

### Method 1: Python Orchestrator (Recommended for Testing)

**Run Complete Pipeline:**
```bash
# macOS
cd /Users/Admin/Desktop/lab-lens
source venv/bin/activate
python data-pipeline/scripts/main_pipeline.py

# Windows
cd C:\Users\YourUsername\Desktop\lab-lens
venv\Scripts\Activate
python data-pipeline\scripts\main_pipeline.py
```

**Run Individual Stages:**
```bash
# Stage 1: Preprocessing only
python data-pipeline/scripts/preprocessing.py

# Stage 2: Validation only
python data-pipeline/scripts/validation.py

# Stage 3: Feature engineering only
python data-pipeline/scripts/feature_engineering.py

# Stage 4: Bias detection only
python data-pipeline/scripts/bias_detection.py

# Stage 5: Bias mitigation only
python data-pipeline/scripts/automated_bias_handler.py
```

**Note:** Individual stages read inputs from `data/processed/` and `logs/`, so run in sequence.

---

### Method 2: Airflow Orchestration (Recommended for Production)

![alt text](lab-lens-dags.png)

**Start Airflow:**

**macOS:**
```bash
cd /Users/Admin/Desktop/lab-lens
docker compose up -d
sleep 60
docker compose ps
open http://localhost:8080
```

**Windows:**
```powershell
cd C:\Users\YourUsername\Desktop\lab-lens
docker compose up -d
Start-Sleep -Seconds 60
docker compose ps
Start-Process "http://localhost:8080"
```

**In Airflow UI:**
1. Open: http://localhost:8080
2. Login: Username `admin`, Password `admin`
3. Find: `lab_lens_mimic_pipeline`
4. Toggle: Switch to ON (blue)
5. Trigger: Click ▶ Play button → "Trigger DAG"

**Monitor Execution:**
- Click DAG name → See execution graph
- Click task box → View logs
- Click "Gantt" tab → See performance
- Click "XCom" tab → See metrics

**Stop Airflow:**
```bash
docker compose down
```

**View Logs:**
```bash
docker compose logs -f airflow-scheduler
```

---

### Method 3: DVC Pipeline (Recommended for Reproducibility)

**Check Pipeline Status:**
```bash
# macOS
cd /Users/Admin/Desktop/lab-lens
source venv/bin/activate
dvc status

# Windows
cd C:\Users\YourUsername\Desktop\lab-lens
venv\Scripts\Activate
dvc status
```

**Run Complete Pipeline:**
```bash
dvc repro
```

**What DVC Does:**
1. Checks which stages have changed
2. Runs only modified stages
3. Caches outputs
4. Updates `dvc.lock` file
5. Tracks data versions

**View Pipeline DAG:**
```bash
dvc dag
```

**Output:**
```
+------------------------------------------+
| mimic_discharge_labs.csv.dvc             |
+------------------------------------------+
                 *
        +-----------------+
        | preprocessing   |
        +-----------------+
                 *
        +-----------------+
        | validation      |
        +-----------------+
                 *
  +----------------------+
  | feature_engineering  |
  +----------------------+
                 *
      +------------------+
      | bias_detection   |
      +------------------+
                 *
      +------------------+
      | bias_mitigation  |
      +------------------+
```

**Force Rerun All Stages:**
```bash
dvc repro --force
```

---

## 📋 Pipeline Stages Detailed

### Stage 1: Preprocessing

**Input:** `data/raw/mimic_discharge_labs.csv` (9,520 records, 18 columns)  
**Output:** `data/processed/processed_discharge_summaries.csv` (7,069 records, 46 columns)  
**Duration:** 9.13 seconds

**Operations:**
1. **Load Data** - Read CSV file
2. **Remove Duplicates** - 4,573 duplicates removed → 4,947 unique records
3. **Demographic Features:**
   - Ethnicity: WHITE (3,334), OTHER (660), BLACK (440), HISPANIC (157), ASIAN (124)
   - Age Groups: 65+ (2,159), 50-65 (1,250), 35-50 (620), <18 (385), 18-35 (301)
4. **Extract Sections** - 9 clinical sections using regex
5. **Expand Abbreviations** - 50+ medical terms normalized
6. **Clean Text** - Remove extra spaces, special characters
7. **Calculate Features** - 28 new feature columns
8. **Handle Missing** - Fill empty values

**Run:**
```bash
python data-pipeline/scripts/preprocessing.py
```

**Logs:**
```
06:20:41 - INFO - Loaded 9520 records
06:20:41 - INFO - Removed 4573 duplicate records
06:20:41 - INFO - Ethnicity distribution: {'WHITE': 3334, 'OTHER': 660, 'BLACK': 440}
06:20:41 - INFO - Age group distribution: {'65+': 2159, '50-65': 1250, '35-50': 620}
06:20:50 - INFO - Saved processed data
```

---

### Stage 2: Validation

**Input:** `data/processed/processed_discharge_summaries.csv` (7,069 records)  
**Output:** `logs/validation_report.json`, `logs/validation_summary.csv`  
**Duration:** 0.72 seconds

**Validation Checks:**
- ✓ Schema: 46 columns, correct data types
- ⚠️ Completeness: 2,354 records without text (33.3%)
- ⚠️ Duplicates: 119 duplicate rows
- ✓ Age Range: All ages 0-120 (valid)
- ⚠️ Cross-Field: 977 logical inconsistencies

**Validation Score:** 82% (PASS - threshold: 80%)

**Run:**
```bash
python data-pipeline/scripts/validation.py
```

**View Results:**
```bash
# Summary table
cat data-pipeline/logs/validation_summary.csv

# Detailed report
cat data-pipeline/logs/validation_report.json | python -m json.tool
```

---

### Stage 3: Feature Engineering

**Input:** `data/processed/processed_discharge_summaries.csv` (7,069 rows, 46 columns)  
**Output:** `data/processed/mimic_features.csv` (7,069 rows, 90 columns)  
**Duration:** 42.62 seconds (bottleneck)

**Features Created (90 total):**

| Category | Features | Count |
|----------|----------|-------|
| Text Metrics | text_chars, text_tokens, sentence_count, readability | 15 |
| Clinical Risk | high_risk_score, comorbidity_score, acute_chronic_ratio | 18 |
| Treatment | polypharmacy_flag, treatment_intensity, medication_count | 12 |
| Lab Testing | total_labs, abnormal_count, critical_labs | 15 |
| Documentation | completeness_score, section_presence, medical_density | 15 |
| Demographics | One-hot encoded gender, ethnicity, age_group | 15 |

**Run:**
```bash
python data-pipeline/scripts/feature_engineering.py
```

**Logs:**
```
06:20:52 - INFO - Loading preprocessed data...
06:20:52 - INFO - Loaded 7069 rows, 46 columns
06:20:52 - INFO - Computing sentence counts...
06:20:52 - INFO - Calculating readability scores...
06:20:56 - INFO - Counting medical terms...
06:21:35 - INFO - Feature engineering complete: 90 features created
06:21:35 - INFO - Output saved: mimic_features.csv
06:21:35 - INFO - Final shape: 7069 rows, 90 columns
```

---

### Stage 4: Bias Detection

**Input:** `data/processed/mimic_features.csv` (7,069 rows, 90 columns)  
**Output:** `logs/bias_report.json`, `logs/bias_summary.csv`, `logs/bias_plots/`  
**Duration:** 1.88 seconds

**3-Stage Analysis:**

**Stage 1 - Raw Bias:**
```
Age CV: 18.63% ⚠️ (threshold: 8%)
Gender CV: 0.0% ✓
Ethnicity CV: 0.0% ✓
```

**Stage 2 - Adjusted Analysis:**
```
R² Score: 0.525 (52.5% explained by clinical complexity)
Residual CV: 44.62% ⚠️ (unexplained variation)
Interpretation: POTENTIAL SYSTEMATIC BIAS
```

**Stage 3 - Quality Parity:**
```
Documentation Completeness CV: 3.29% ✓
Has Diagnosis CV: 3.38% ✓
Has Medications CV: 3.70% ✓
Assessment: PARITY MAINTAINED
```

**Run:**
```bash
python data-pipeline/scripts/bias_detection.py
```

**View Results:**
```bash
# Summary table
cat data-pipeline/logs/bias_summary.csv

# Detailed analysis
cat data-pipeline/logs/bias_report.json | python -m json.tool

# Visualizations
open data-pipeline/logs/bias_plots/text_length_by_age.png        # macOS
start data-pipeline\logs\bias_plots\text_length_by_age.png       # Windows
```

---

### Stage 5: Bias Mitigation

**Input:** `data/processed/mimic_features.csv` + `logs/bias_report.json`  
**Output:** `data/processed/mimic_features_mitigated.csv`, `logs/bias_mitigation_report.json`  
**Duration:** 0.99 seconds

**Strategy Applied:** SEVERE (feature normalization)

**Results:**
```
Before Mitigation:
  Age CV: 18.63%
  Overall Bias Score: 18.63%

After Mitigation:
  Age CV: 0.00%
  Overall Bias Score: 0.00%

Improvement: 99.99% ✓
```

**Run:**
```bash
python data-pipeline/scripts/automated_bias_handler.py
```

**View Results:**
```bash
cat data-pipeline/logs/bias_mitigation_report.json | python -m json.tool
```

---

## 🧪 Testing

### Run All Tests

**macOS:**
```bash
cd /Users/Admin/Desktop/lab-lens
source venv/bin/activate
pytest data-pipeline/tests/ -v
```

**Windows:**
```powershell
cd C:\Users\YourUsername\Desktop\lab-lens
venv\Scripts\Activate
pytest data-pipeline\tests\ -v
```

**Expected Output:**
```
data-pipeline/tests/test_preprocessing.py::test_remove_duplicates PASSED
data-pipeline/tests/test_preprocessing.py::test_standardize_ethnicity PASSED
...
data-pipeline/tests/test_validation.py::test_validate_schema PASSED
...
data-pipeline/tests/test_feature_engineering.py::test_readability_scores PASSED
...

============================================
test_preprocessing.py:        18 passed in 0.24s
test_validation.py:           13 passed in 0.22s
test_feature_engineering.py:  27 passed in 1.66s
============================================
Total:                        58 passed in 2.12s
============================================
```

### Run Individual Test Files

```bash
# Preprocessing tests (18 tests)
pytest data-pipeline/tests/test_preprocessing.py -v

# Validation tests (13 tests)
pytest data-pipeline/tests/test_validation.py -v

# Feature engineering tests (27 tests)
pytest data-pipeline/tests/test_feature_engineering.py -v
```

### Run with Coverage

```bash
pytest data-pipeline/tests/ --cov=data-pipeline --cov-report=html
open htmlcov/index.html        # macOS
start htmlcov\index.html       # Windows
```

---

## 📦 Dependencies

### Python Packages

**Install All:**
```bash
pip install -r data-pipeline/requirements.txt
```

**Core Libraries:**
```
pandas>=2.0.0           # Data manipulation
numpy>=1.24.0           # Numerical operations
scipy>=1.10.0           # Statistical analysis
scikit-learn>=1.3.0     # Machine learning
```

**Visualization:**
```
matplotlib>=3.7.0       # Plotting
seaborn>=0.12.0         # Statistical visualization
```

**Google Cloud (for data acquisition):**
```
google-cloud-bigquery>=3.11.0
google-auth-oauthlib>=1.0.0
db-dtypes>=1.1.0
```

**Testing:**
```
pytest>=8.4.2
pytest-cov>=7.0.0
```

**Python 3.9.6 Compatibility:**
```
importlib-metadata>=4.13.0,<5.0.0
setuptools>=65.0.0
```

---

## 🔄 Running with Airflow

### Setup Airflow (Docker)

**macOS:**
```bash
cd /Users/Admin/Desktop/lab-lens
docker compose up -d
sleep 60
docker compose ps
```

**Windows:**
```powershell
cd C:\Users\YourUsername\Desktop\lab-lens
docker compose up -d
Start-Sleep -Seconds 60
docker compose ps
```

**Expected Status:**
```
NAME                           STATUS
lab-lens-postgres-1            Up (healthy)
lab-lens-airflow-webserver-1   Up (healthy)
lab-lens-airflow-scheduler-1   Up (healthy)
```

### Access Airflow UI

**URL:** http://localhost:8080  
**Username:** admin  
**Password:** admin

### Run Pipeline in Airflow

1. **Enable DAG:** Find `lab_lens_mimic_pipeline` → Toggle ON
2. **Trigger Run:** Click ▶ Play button → "Trigger DAG"
3. **Monitor:** Click DAG name → View Graph

**DAG Tasks (7 total):**
```
check_data          → Verify raw data exists
    ↓
preprocess_data     → Clean and transform
    ↓
validate_data       → Quality checks
    ↓
engineer_features   → Create 90 features
    ↓
detect_bias         → 3-stage bias analysis
    ↓
mitigate_bias       → Reduce bias to 0%
    ↓
generate_summary    → Create final report
```

### View Airflow Results

**Task Logs:**
1. Click any task box (green = success)
2. Click "Log" button
3. View execution details

**XCom Metrics:**
1. Click task box
2. Click "XCom" tab
3. View:
   - `records_processed`: 7069
   - `validation_score`: 82.0
   - `bias_score_raw`: 18.63
   - `bias_score_after`: 0.00

**Gantt Chart:**
1. Click DAG name
2. Click "Gantt" tab
3. See performance breakdown

**Stop Airflow:**
```bash
docker compose down
```

---

## 🔄 Running with DVC

### DVC Pipeline Workflow

**Check Status:**
```bash
# macOS
cd /Users/Admin/Desktop/lab-lens
source venv/bin/activate
dvc status

# Windows
cd C:\Users\YourUsername\Desktop\lab-lens
venv\Scripts\Activate
dvc status
```

**Run Pipeline:**
```bash
dvc repro
```

**What Happens:**
```
Verifying data sources...
Running stage 'preprocessing': ✓ (9.13s)
Running stage 'validation': ✓ (0.72s)
Running stage 'feature_engineering': ✓ (42.62s)
Running stage 'bias_detection': ✓ (1.88s)
Running stage 'bias_mitigation': ✓ (0.99s)
Updating lock file 'dvc.lock'
```

**Track Data with DVC:**
```bash
# Add raw data to DVC
dvc add data-pipeline/data/raw/mimic_discharge_labs.csv

# Commit to Git
git add data-pipeline/data/raw/mimic_discharge_labs.csv.dvc .gitignore
git commit -m "Add raw data to DVC"
```

**View Pipeline:**
```bash
dvc dag
```

### DVC Remote Storage (Optional)

**Setup Remote:**
```bash
# AWS S3
dvc remote add -d storage s3://my-bucket/lab-lens-data

# Google Cloud Storage
dvc remote add -d storage gs://my-bucket/lab-lens-data

# Azure Blob
dvc remote add -d storage azure://container/lab-lens-data
```

**Push Data:**
```bash
dvc push
```

**Pull Data:**
```bash
dvc pull
```

---

## 📊 Viewing Results

### Pipeline Summary

**macOS:**
```bash
cat data-pipeline/logs/pipeline_results_latest.json | python -m json.tool
```

**Windows:**
```powershell
type data-pipeline\logs\pipeline_results_latest.json | python -m json.tool
```

**Output:**
```json
{
  "pipeline_execution": {
    "success": true,
    "total_duration_minutes": 0.92,
    "steps_completed": ["preprocessing", "validation", "feature_engineering", "bias_detection", "bias_mitigation"]
  },
  "quality_metrics": {
    "validation_score": 82.0,
    "bias_score_before": 18.63,
    "bias_score_after": 0.0,
    "bias_improvement": 18.63
  },
  "overall_status": {
    "data_quality": "good",
    "bias_status": "acceptable",
    "ready_for_modeling": true
  }
}
```

### Validation Results

**Summary Table:**
```bash
cat data-pipeline/logs/validation_summary.csv
```

**Output:**
```csv
Metric,Value,Status
Total Records,7069,INFO
Total Columns,46,INFO
Schema Valid,True,PASS
Records Without Text,2354,WARNING
Duplicate Records,119,WARNING
Invalid Ages,0,PASS
Cross-Field Issues,977,WARNING
Validation Score,82.00%,PASS
```

### Bias Analysis

**Summary Table:**
```bash
cat data-pipeline/logs/bias_summary.csv
```

**Output:**
```csv
Analysis_Stage,Metric,Value,Alert,Interpretation
Raw Bias,age_cv,18.63%,YES,Significant age-based variation
Adjusted Bias,model_r2_score,0.525,INFO,52.5% explained by clinical complexity
Adjusted Bias,age_group_residual_cv,44.62%,YES,Unexplained bias beyond clinical factors
Quality Parity,age_group_quality_cv,3.29%,NO,Consistent quality across all groups
```

**Detailed Report:**
```bash
cat data-pipeline/logs/bias_report.json | python -m json.tool | less
```

### Mitigation Results

```bash
cat data-pipeline/logs/bias_mitigation_report.json | python -m json.tool
```

**Key Metrics:**
```json
{
  "mitigation_applied": true,
  "strategy": {
    "action": "severe",
    "methods": ["feature_normalization"],
    "requires_review": true
  },
  "before_mitigation": {
    "age_cv": 18.63,
    "overall_bias_score": 18.63
  },
  "after_mitigation": {
    "age_cv": 0.0,
    "overall_bias_score": 0.0
  },
  "improvement": {
    "improvement_percentage": 99.99,
    "absolute_reduction": 18.63
  }
}
```

### Visualizations

**View Bias Plots:**

**macOS:**
```bash
open data-pipeline/logs/bias_plots/text_length_by_age.png
open data-pipeline/logs/bias_plots/text_length_by_gender.png
open data-pipeline/logs/bias_plots/text_length_by_ethnicity.png
open data-pipeline/logs/bias_plots/abnormal_labs_by_age.png
open data-pipeline/logs/bias_plots/treatment_intensity_by_age.png
```

**Windows:**
```powershell
start data-pipeline\logs\bias_plots\text_length_by_age.png
start data-pipeline\logs\bias_plots\text_length_by_gender.png
start data-pipeline\logs\bias_plots\text_length_by_ethnicity.png
start data-pipeline\logs\bias_plots\abnormal_labs_by_age.png
start data-pipeline\logs\bias_plots\treatment_intensity_by_age.png
```

---

## 📈 Configuration

### Pipeline Configuration

**File:** `data-pipeline/configs/pipeline_config.json`

**Key Settings:**
```json
{
  "pipeline_config": {
    "input_path": "data-pipeline/data/raw",
    "output_path": "data-pipeline/data/processed",
    "logs_path": "data-pipeline/logs",
    "enable_preprocessing": true,
    "enable_validation": true,
    "enable_bias_detection": true,
    "auto_mitigation": true
  },
  "bias_detection_config": {
    "alert_thresholds": {
      "gender_cv_max": 5.0,
      "ethnicity_cv_max": 10.0,
      "age_cv_max": 8.0,
      "residual_cv_threshold": 5.0
    }
  },
  "validation_config": {
    "text_length_min": 100,
    "text_length_max": 100000,
    "age_min": 0,
    "age_max": 120,
    "validation_score_threshold": 80
  }
}
```

**Modify Settings:**
```bash
# Edit configuration
nano data-pipeline/configs/pipeline_config.json  # macOS/Linux
notepad data-pipeline\configs\pipeline_config.json  # Windows

# Then rerun pipeline
python data-pipeline/scripts/main_pipeline.py
```

---

## 🔄 Complete Workflow Examples

### Scenario 1: First Time Setup

**macOS:**
```bash
# 1. Navigate to project
cd /Users/Admin/Desktop/lab-lens

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r data-pipeline/requirements.txt

# 4. Verify data exists
ls -lh data-pipeline/data/raw/mimic_discharge_labs.csv

# 5. Run tests
pytest data-pipeline/tests/ -v

# 6. Run pipeline
python data-pipeline/scripts/main_pipeline.py

# 7. View results
cat data-pipeline/logs/pipeline_results_latest.json | python -m json.tool
```

**Windows:**
```powershell
# 1. Navigate to project
cd C:\Users\YourUsername\Desktop\lab-lens

# 2. Create virtual environment
python -m venv venv
venv\Scripts\Activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r data-pipeline\requirements.txt

# 4. Verify data exists
dir data-pipeline\data\raw\mimic_discharge_labs.csv

# 5. Run tests
pytest data-pipeline\tests\ -v

# 6. Run pipeline
python data-pipeline\scripts\main_pipeline.py

# 7. View results
type data-pipeline\logs\pipeline_results_latest.json | python -m json.tool
```

---

### Scenario 2: Daily Development

**macOS:**
```bash
cd /Users/Admin/Desktop/lab-lens
source venv/bin/activate
dvc status
dvc repro
git add dvc.lock
git commit -m "Update pipeline"
```

**Windows:**
```powershell
cd C:\Users\YourUsername\Desktop\lab-lens
venv\Scripts\Activate
dvc status
dvc repro
git add dvc.lock
git commit -m "Update pipeline"
```

---

### Scenario 3: Production Deployment

**macOS:**
```bash
cd /Users/Admin/Desktop/lab-lens

# 1. Start Airflow
docker compose up -d
sleep 60

# 2. Access UI
open http://localhost:8080

# 3. Enable & trigger DAG in UI

# 4. Monitor execution
docker compose logs -f airflow-scheduler

# 5. When done, stop
docker compose down
```

**Windows:**
```powershell
cd C:\Users\YourUsername\Desktop\lab-lens

# 1. Start Airflow
docker compose up -d
Start-Sleep -Seconds 60

# 2. Access UI
Start-Process "http://localhost:8080"

# 3. Enable & trigger DAG in UI

# 4. Monitor execution
docker compose logs -f airflow-scheduler

# 5. When done, stop
docker compose down
```

---

## 🛠️ Troubleshooting

### Issue: Import errors when running scripts

**Solution:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS
venv\Scripts\Activate     # Windows

# Verify Python packages installed
pip list | grep pandas
pip list | grep scipy
```

---

### Issue: "FileNotFoundError: mimic_discharge_labs.csv"

**Solution:**
```bash
# Check if file exists
ls data-pipeline/data/raw/mimic_discharge_labs.csv  # macOS
dir data-pipeline\data\raw\mimic_discharge_labs.csv  # Windows

# If missing, run data acquisition notebook
jupyter notebook data-pipeline/notebooks/data_acquisition.ipynb
```

---

### Issue: Tests failing

**Solution:**
```bash
# Run tests with verbose output to see which test fails
pytest data-pipeline/tests/ -v -s

# Run specific failing test
pytest data-pipeline/tests/test_preprocessing.py::test_remove_duplicates -v
```

---

### Issue: DVC pipeline failing

**Solution:**
```bash
# Force rerun all stages
dvc repro --force

# Clean cache and retry
dvc gc
dvc repro
```

---

### Issue: Airflow DAG not appearing

**Solution:**
```bash
# Check if DAG file exists
ls airflow/dags/pipeline_dag.py

# Check for Python syntax errors
docker compose exec airflow-scheduler python /opt/airflow/dags/pipeline_dag.py

# Restart scheduler
docker compose restart airflow-scheduler

# Wait 30 seconds and refresh browser
```

---

### Issue: Docker services unhealthy

**Solution:**
```bash
# View logs
docker compose logs

# Restart services
docker compose down
docker compose up -d

# Complete reset
docker compose down -v
docker compose up -d
```

---

## 📝 Output Files Reference

### Data Files

| File | Size | Rows | Columns | Purpose |
|------|------|------|---------|---------|
| `raw/mimic_discharge_labs.csv` | 90 MB | 9,520 | 18 | Raw MIMIC-III data |
| `processed/processed_discharge_summaries.csv` | ~75 MB | 7,069 | 46 | Cleaned data |
| `processed/mimic_features.csv` | ~85 MB | 7,069 | 90 | Engineered features |
| `processed/mimic_features_mitigated.csv` | ~85 MB | 7,069 | 90 | Bias-corrected ⭐ |

### Report Files

| File | Format | Content |
|------|--------|---------|
| `validation_report.json` | JSON | Detailed quality metrics |
| `validation_summary.csv` | CSV | Quality score table |
| `bias_report.json` | JSON | 3-stage bias analysis |
| `bias_summary.csv` | CSV | Bias metrics table |
| `bias_mitigation_report.json` | JSON | Mitigation results |
| `pipeline_results_latest.json` | JSON | Overall summary |
| `feature_engineering.log` | LOG | Feature engineering trace |

### Visualization Files

| File | Type | Content |
|------|------|---------|
| `bias_plots/text_length_by_age.png` | PNG | Age gradient visualization |
| `bias_plots/text_length_by_gender.png` | PNG | Gender parity |
| `bias_plots/text_length_by_ethnicity.png` | PNG | Ethnicity parity |
| `bias_plots/abnormal_labs_by_age.png` | PNG | Clinical complexity |
| `bias_plots/treatment_intensity_by_age.png` | PNG | Treatment need |

---

## 🎓 Assignment Requirements Checklist


- ✅ **1. Data Acquisition** - BigQuery integration, 9,520 records
- ✅ **2. Data Preprocessing** - Modular, reusable, 28 features
- ✅ **3. Test Modules** - 58 tests, pytest framework, edge cases
- ✅ **4. Airflow Orchestration** - Docker-based DAG, 7 tasks
- ✅ **5. DVC Versioning** - dvc.yaml, dvc.lock, data tracking
- ✅ **6. Logging & Tracking** - Comprehensive logs, metrics
- ✅ **7. Schema & Statistics** - Automated validation reports
- ✅ **8. Anomaly Detection** - Multi-level checks, alerts
- ✅ **9. Pipeline Optimization** - Performance profiling, 55s total
- ✅ **10. Bias Detection** - 3-stage analysis, 99.99% mitigation
- ✅ **11. Reproducibility** - DVC, Docker, clear instructions
- ✅ **12. Error Handling** - Try-except, centralized framework


---

## 🚀 Quick Command Summary

### macOS
```bash
# Complete workflow
cd /Users/Admin/Desktop/lab-lens
source venv/bin/activate
pytest data-pipeline/tests/ -v
python data-pipeline/scripts/main_pipeline.py
```

### Windows
```powershell
# Complete workflow
cd C:\Users\YourUsername\Desktop\lab-lens
venv\Scripts\Activate
pytest data-pipeline\tests\ -v
python data-pipeline\scripts\main_pipeline.py
```

### Airflow (Both Platforms)
```bash
docker compose up -d
# Wait 60 seconds → Open http://localhost:8080
# Login: admin/admin → Enable DAG → Trigger
docker compose down
```

### DVC (Both Platforms)
```bash
dvc status
dvc repro
git add dvc.lock
git commit -m "Update pipeline"
```
