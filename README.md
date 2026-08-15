# CCS3440 – Artificial Intelligence Coursework
### SmartCare Hospital AI Dataset — Option C: Disease Risk Classification

## 📌 Module Information
- **Module Code:** CCS3440
- **Module Name:** Artificial Intelligence
- **Institution:** SLTC Research University
- **Lecturer in Charge:** Dr. Chameera De Silva
- **Teaching Assistants:** Mr. Chamod Hewage, Mr. Pamod Dilshan

## 👥 Team Members
| Name | Student ID | Role/Contribution |
|------|-----------|-------------------|
| S.A.Thilani Dilmani | CIT-23-02-0173 | Task 05 & 06 – Model Development & Evaluation |
| R.M.Nuwani Umanda | CIT-23-02-0153 | Task 07 & 08 – Explainable AI & Prototype |
| G.Ishini Sivod | CIT-23-02-0044 | Task 03 & 04 – Preprocessing & EDA |
| Volga Indeewari | CIT-23-02-0159 | Task 01 & 02 – Problem Definition & Dataset Understanding |

## 🎯 Selected Prediction Task
- [x] **Option C – Disease Risk Classification**
  Classifying patients into disease risk categories (Multi-Class: Low / Medium / High) using demographic, clinical, and hospital operations data.

## 📂 Project Structure
```
CCS3440-SmartCare-AI-Coursework/
│
├── data/                          # Dataset files
│   ├── smartcare_ai_dataset_1000.csv
│   └── smartcare_ai_dataset_data_dictionary.csv
│
├── notebooks/                     # Jupyter notebooks
│   └── CCS4340_Artificial_Intelligence_Coursework.ipynb   # Tasks 02-08, single notebook
│
├── models/                        # Trained model files (.joblib) + metadata + CV comparison
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   ├── logistic_regression.joblib
│   ├── xgboost.joblib
│   ├── voting_ensemble.joblib
│   ├── stacking_ensemble.joblib
│   ├── model_metadata.json
│   └── model_comparison_cv.csv
│
├── prototype/                     # Streamlit prototype app
│   └── app.py
│
├── report/                        # Technical report (PDF)
│   └── CCS3440_Technical_Report.pdf
│
├── .gitignore
└── README.md
```

## 🛠️ Technologies Used
- **Language:** Python
- **Environment:** Google Colab / Jupyter Notebook
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-Learn, XGBoost
- **Explainable AI:** SHAP, LIME
- **Prototype:** Streamlit

## 📊 Dataset
The dataset (`smartcare_ai_dataset_1000.csv`) contains 1000 hospital records covering:
- **Patient Info:** Patient ID, Age, Gender, Blood Group
- **Clinical Info:** Diagnosis, Blood Pressure, Blood Sugar, Cholesterol, BMI
- **Hospital Operations:** Department, Appointment History, Previous Admissions, Length of Stay, Room Type, Treatment Count, Lab Test Count
- **Financial Data:** Consultation, Lab, Room, Medicine Charges, Total Bill Amount

**Target variable:** `disease_risk_level` (Low / Medium / High) — moderately imbalanced: Medium 46.9%, High 40.0%, Low 13.1%.

See `data/smartcare_ai_dataset_data_dictionary.csv` for full attribute definitions.

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/ThilaniDilmani/CCS3440-SmartCare-AI-Coursework.git
cd CCS3440-SmartCare-AI-Coursework
```

### 2. Set up environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook notebooks/CCS4340_Artificial_Intelligence_Coursework.ipynb
```

### 4. Run the prototype
```bash
streamlit run prototype/app.py
```

## 📈 Models Trained
All models were tuned via 5-fold stratified cross-validation using **macro-F1** as the scoring metric (chosen over accuracy because it weighs the minority `Low` class equally with `Medium`/`High`). Test-set results (200 held-out records):

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC (macro, OvR) |
|-------|----------|--------------------|-----------------|------------|------------------------|
| **Logistic Regression (best)** | **0.935** | **0.951** | **0.924** | **0.936** | **0.992** |
| Stacking Ensemble (bonus) | 0.905 | 0.930 | 0.874 | 0.897 | 0.980 |
| Voting Ensemble (bonus) | 0.865 | 0.875 | 0.816 | 0.839 | 0.964 |
| XGBoost | 0.860 | 0.859 | 0.803 | 0.825 | 0.963 |
| Random Forest | 0.790 | 0.817 | 0.721 | 0.751 | 0.914 |
| Decision Tree | 0.700 | 0.698 | 0.701 | 0.699 | 0.783 |

**Best model:** Logistic Regression (wrapped in a scikit-learn `Pipeline` with `StandardScaler`), selected on macro-F1. It outperforms the tree-based models and ensembles because the engineered categorical bands (`age_group`, `bmi_category`, etc.) already capture the sharp, threshold-like separation between risk classes that Task 04's EDA revealed — leaving little room for tree-based flexibility to add value, especially with only 800 training rows. Every model, including the best one, struggles most on the minority `Low` class, which has direct implications for clinical safety and is carried through into the Task 07 explainability discussion.

## 🔍 Explainable AI
Model predictions are interpreted using **SHAP** (primary) and **LIME** (cross-validation) on the best-performing model, Logistic Regression:
- **Global explanations:** SHAP beeswarm and bar plots show `blood_sugar_mg_dl`, `cholesterol_mg_dl`, `bmi`, `age`, and `previous_admissions` as the top drivers of High-risk predictions, consistent across all three classes.
- **Local explanations:** Waterfall plots explain individual correct and misclassified predictions.
- **Dependence plots:** Confirm the linear, additive relationships expected from Logistic Regression.
- **LIME cross-check:** Produces the same top-5 features and ranking as SHAP for a misclassified case, reinforcing confidence in the explanation.
- **Ethical framing:** Since a missed High-risk patient is more costly than a false alarm, the prototype surfaces full class probabilities rather than just the predicted label, so predictions support — rather than replace — clinical judgement.

## 📋 Coursework Tasks Checklist
- [ ] Task 01 – Problem Definition & Literature Review
- [x] Task 02 – Dataset Understanding
- [x] Task 03 – Data Preprocessing & Feature Engineering
- [x] Task 04 – Exploratory Data Analysis
- [x] Task 05 – Machine Learning Model Development
- [x] Task 06 – Model Evaluation
- [x] Task 07 – Explainable AI Analysis
- [x] Task 08 – AI Prototype Development
- [ ] Task 09 – Technical Report

## 📄 License
This project is submitted as academic coursework for SLTC Research University (CCS3440 – Artificial Intelligence).

## 🙏 Acknowledgements
Dataset provided by SLTC Research University for CCS3440 coursework purposes.
