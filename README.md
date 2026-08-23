# SmartCare Hospital AI
### CCS3440 – Artificial Intelligence Coursework | Option C: Disease Risk Classification

An end-to-end ML pipeline that classifies hospital patients into disease risk categories
(**Low / Medium / High**) from demographic, clinical, and hospital-operations data, complete
with explainability analysis, fairness checks, and a Streamlit prototype for interactive
predictions.

## 📌 Module Information
- **Module Code:** CCS3440
- **Module Name:** Artificial Intelligence
- **Institution:** SLTC Research University
- **Lecturer in Charge:** Dr. Chameera De Silva
- **Teaching Assistants:** Mr. Chamod Hewage, Mr. Pamod Dilshan

## 👥 Team Members
| Name | Student ID | Task |
|------|-----------|------|
| Volga Indeewari | CIT-23-02-0159 | Task 01 & 02 – Problem Definition & Dataset Understanding |
| G. Ishini Sivod | CIT-23-02-0044 | Task 03 & 04 – Preprocessing & EDA |
| S.A. Thilani Dilmani | CIT-23-02-0173 | Task 05 & 06 – Model Development & Evaluation |
| R.M. Nuwani Umanda | CIT-23-02-0153 | Task 07 & 08 – Explainable AI & Prototype |

## 🎯 Selected Prediction Task
**Option C – Disease Risk Classification**: a multi-class model (Low / Medium / High) that
combines patient demographics, clinical readings (blood pressure, blood sugar, cholesterol,
BMI), and hospital-operations data (admissions, length of stay, treatment/lab counts) to
flag disease risk.

## 📂 Project Structure
```
SmartCare-Hospital-AI/
│
├── data/                                  # Raw, cleaned, and train/test-split datasets
│   ├── smartcare_ai_dataset_1000.csv          # Raw 1,000-record dataset
│   ├── smartcare_ai_dataset_data_dictionary.csv   # Column-by-column definitions
│   ├── smartcare_cleaned (1).csv              # Cleaned/engineered dataset
│   ├── X_train (1).csv / X_test (1).csv       # Unscaled train/test features
│   └── X_train_scaled (1).csv / X_test_scaled (1).csv  # Scaled variants
│
├── notebooks/                             # One notebook per task pair, per member
│   ├── Task_02_CIT-23-02-0159.ipynb           # Dataset understanding
│   ├── task_03_and_04_CIT_23_02_0044.ipynb    # Preprocessing & EDA
│   ├── Task_05_&_06_CIT_23_02_0173.ipynb      # Model development & evaluation
│   └── Task 07 & 08_CIT-23-02-0153.ipynb      # Explainable AI & prototype
│
├── models/                                # Trained models + preprocessing artifacts
│   ├── logistic_regression (1).joblib
│   ├── decision_tree (1).joblib
│   ├── xgboost (1).joblib
│   ├── voting_ensemble (1).joblib
│   ├── stacking_ensemble (1).joblib
│   ├── disease_risk_model.pkl                 # Deployed model used by the prototype
│   ├── encoder_unscaled.pkl                   # Fitted ColumnTransformer
│   ├── target_encoding.pkl                    # Label ↔ risk-class mapping
│   ├── feature_columns.pkl / raw_input_columns.pkl
│
├── results/                               # Evaluation outputs, metrics & fairness checks
│   ├── task06_evaluation_results.csv          # Held-out test-set metrics (all 6 models)
│   ├── model_comparison_cv (1).csv            # 5-fold CV macro-F1 comparison
│   ├── model_metadata (1).json                # Best hyperparameters per model
│   ├── all_confusion_matrices.png
│   ├── shap_summary.png
│   ├── gender_fairness.csv / agegroup_fairness.csv
│   └── fairness_results.png / agegroup_fairness.png
│
├── prototype/                             # Streamlit prediction app
│   ├── app.py
│   └── input_page.png.png / prediction_result.png.png   # UI screenshots
│
├── literature_review/                     # Background research for Task 01
│   ├── SmartCare_reasearch.xlsx
│   ├── references.bib
│   └── README.md
│
├── docs/                                  # Supporting diagrams & summary
│   ├── architecture_diagram.png
│   ├── workflow_diagram.png
│   └── project_summary.md
│
├── report/                                # Technical report (PDF) — added at submission
├── Presentation                           # Notes on slides/demo video/screenshots for submission
├── requirements.txt
└── README.md
```

## 🛠️ Technologies Used
- **Language:** Python
- **Environment:** Jupyter Notebook / Google Colab
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine Learning:** Scikit-Learn, XGBoost
- **Explainable AI:** SHAP, LIME
- **Prototype:** Streamlit, Altair

## 📊 Dataset
`data/smartcare_ai_dataset_1000.csv` contains 1,000 synthetic hospital records covering:
- **Patient info:** age, gender, blood group
- **Clinical readings:** blood pressure, blood sugar, cholesterol, BMI
- **Hospital operations:** department, appointment history, admissions, length of stay, room type, treatment/lab counts
- **Financial data:** consultation, lab, room, and medicine charges (LKR)

**Target:** `disease_risk_level` (Low / Medium / High) — moderately imbalanced across classes.
See `data/smartcare_ai_dataset_data_dictionary.csv` for the full column reference (the dataset
also carries two other potential targets, `no_show` and `readmitted_30_days`, not used here).

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/ThilaniDilmani/SmartCare-Hospital-AI.git
cd SmartCare-Hospital-AI
```

### 2. Set up the environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 3. Run the notebooks
Open any notebook in `notebooks/` in Jupyter — they're ordered by task and should be run
Task 02 → 04 → 06 → 08, since later tasks depend on the cleaned data / trained models the
earlier ones produce.

### 4. Run the prototype
```bash
streamlit run prototype/app.py
```
The app loads `models/disease_risk_model.pkl` and its accompanying encoder/column artifacts
from `models/`, so keep the folder structure intact when running locally.

## 📈 Models Trained
All six models were tuned via 5-fold stratified cross-validation using **macro-F1** as the
scoring metric (chosen over accuracy so the minority `Low`-risk class isn't drowned out by
`Medium`/`High`). Held-out test-set results:

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|-------|----------|--------------------|-----------------|------------|
| **Logistic Regression (best)** | **0.970** | **0.970** | **0.958** | **0.964** |
| Stacking Ensemble (bonus) | 0.945 | 0.949 | 0.931 | 0.940 |
| Voting Ensemble (bonus) | 0.870 | 0.878 | 0.830 | 0.849 |
| XGBoost | 0.845 | 0.858 | 0.793 | 0.817 |
| Random Forest | 0.790 | 0.830 | 0.714 | 0.746 |
| Decision Tree | 0.675 | 0.665 | 0.674 | 0.668 |

*(Source: `results/task06_evaluation_results.csv`. Cross-validation macro-F1 scores used for
model selection are in `results/model_comparison_cv (1).csv`, and tuned hyperparameters for
each model are in `results/model_metadata (1).json`.)*

**Best model:** Logistic Regression, wrapped in a scikit-learn `Pipeline` with `StandardScaler`,
selected on macro-F1. It edges out the tree-based models and ensembles because the engineered
categorical features (e.g. `age_group`, `bmi_category`) already capture the sharp, threshold-like
separation between risk classes — leaving little extra signal for tree-based flexibility to
exploit, especially with only 800 training rows.

## 🔍 Explainable AI
Predictions from the best model (Logistic Regression) are interpreted with **SHAP** (primary)
and **LIME** (cross-check):
- **Global explanations:** SHAP beeswarm/bar plots (`results/shap_summary.png`) rank
  `blood_sugar_mg_dl`, `cholesterol_mg_dl`, `bmi`, `age`, and `previous_admissions` as the
  top drivers of High-risk predictions.
- **Local explanations:** Waterfall plots explain individual correct and misclassified cases.
- **LIME cross-check:** Reproduces the same top-5 features and ranking as SHAP for a
  misclassified case.
- **Fairness:** Performance is checked across gender and age bands
  (`results/gender_fairness.csv`, `results/agegroup_fairness.csv`) to confirm the model
  doesn't systematically underperform for any subgroup.
- **Ethical framing:** Since missing a High-risk patient is costlier than a false alarm, the
  prototype surfaces full class probabilities rather than just the predicted label, so the
  model supports — rather than replaces — clinical judgement.

## 🖥️ Prototype
The Streamlit app (`prototype/app.py`) takes raw patient inputs, applies the same
preprocessing pipeline used in training, and returns a predicted risk level with class
probabilities. See `prototype/input_page.png.png` and `prototype/prediction_result.png.png`
for screenshots of the input form and results view.

## 📋 Coursework Tasks Checklist
- [x] Task 01 – Problem Definition & Literature Review
- [x] Task 02 – Dataset Understanding
- [x] Task 03 – Data Preprocessing & Feature Engineering
- [x] Task 04 – Exploratory Data Analysis
- [x] Task 05 – Machine Learning Model Development
- [x] Task 06 – Model Evaluation
- [x] Task 07 – Explainable AI Analysis
- [x] Task 08 – AI Prototype Development
- [x] Task 09 – Technical Report 

## 📄 License
Submitted as academic coursework for SLTC Research University (CCS3440 – Artificial
Intelligence). Not licensed for external reuse.

## 🙏 Acknowledgements
Dataset provided by SLTC Research University for CCS3440 coursework purposes.
  


