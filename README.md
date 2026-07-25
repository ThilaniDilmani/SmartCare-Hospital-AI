# CCS3440 – Artificial Intelligence Coursework
### SmartCare Hospital AI Dataset

## 📌 Module Information
- **Module Code:** CCS3440
- **Module Name:** Artificial Intelligence
- **Institution:** SLTC Research University
- **Lecturer in Charge:** Dr. Chameera De Silva
- **Teaching Assistants:** Mr. Chamod Hewage, Mr. Pamod Dilshan

## 👥 Team Members
| Name | Student ID | Role/Contribution |
|------|-----------|-------------------|
| [S.A.Thilani Dilmani] | [CIT-23-02-0173] | [task 5 and 6] |
| [R.M.Nuwani Umanda] | [CIT-23-02-0153] | [task 7 and 8] |
| [G.Ishini Sivod] | [CIT-23-02-0044] | [task 3 and 4] |
| [Volga Indeewari] | [CIT-23-02-0159] | [task 1 and 2] |

## 🎯 Selected Prediction Task
> **[Choose one and delete the others]**

- [ ] **Option A – Appointment No-Show Prediction**
  Predicting whether a patient will miss a scheduled appointment (Binary Classification: No Show / Attended)
- [ ] **Option B – Patient Readmission Prediction**
  Predicting whether a patient will be readmitted within 30 days (Binary Classification: Readmitted / Not Readmitted)
- [ ] **Option C – Disease Risk Classification**
  Classifying patients into disease risk categories (Multi-Class: Low / Medium / High)

## 📂 Project Structure

CCS3440-SmartCare-AI-Coursework/
│
├── data/ # Dataset files
│ ├── smartcare_ai_dataset_1000.csv
│ └── smartcare_ai_dataset_data_dictionary.csv
│
├── notebooks/ # Jupyter notebooks
│ └── smartcare_analysis.ipynb
│
├── src/ # Python source code (.py scripts)
│ ├── preprocessing.py
│ ├── feature_engineering.py
│ ├── train_models.py
│ └── evaluate.py
│
├── models/ # Trained model files (.pkl / .joblib)
│
├── prototype/ # Streamlit/Flask prototype app
│ └── app.py
│
├── report/ # Technical report (PDF)
│ └── CCS3440_Technical_Report.pdf
│
├── .gitignore
└── README.md

## 🛠️ Technologies Used
- **Language:** Python
- **Environment:** Jupyter Notebook
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-Learn
- **Explainable AI:** SHAP / LIME
- **Prototype:** Streamlit / Flask

## 📊 Dataset
The dataset (`smartcare_ai_dataset_1000.csv`) contains 1000 hospital records covering:
- **Patient Info:** Patient ID, Age, Gender, Blood Group
- **Clinical Info:** Diagnosis, Blood Pressure, Blood Sugar, Cholesterol, BMI
- **Hospital Operations:** Department, Appointment History, Previous Admissions, Length of Stay, Room Type, Treatment Count, Lab Test Count
- **Financial Data:** Consultation, Lab, Room, Medicine Charges, Total Bill Amount

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
jupyter notebook notebooks/smartcare_analysis.ipynb
```

### 4. Run the prototype
```bash
streamlit run prototype/app.py
```
*(or `python prototype/app.py` if using Flask)*

## 📈 Models Trained
| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | - | - | - | - |
| Random Forest | - | - | - | - |
| [Model 3] | - | - | - | - |

*(Table to be filled in after Task 05/06 – Model Development & Evaluation)*

## 🔍 Explainable AI
Model predictions are interpreted using **[SHAP / LIME]** to identify key features driving predictions, ensuring transparency in clinical decision support.

## 📋 Coursework Tasks Checklist
- [ ] Task 01 – Problem Definition & Literature Review
- [ ] Task 02 – Dataset Understanding
- [ ] Task 03 – Data Preprocessing & Feature Engineering
- [ ] Task 04 – Exploratory Data Analysis
- [ ] Task 05 – Machine Learning Model Development
- [ ] Task 06 – Model Evaluation
- [ ] Task 07 – Explainable AI Analysis
- [ ] Task 08 – AI Prototype Development
- [ ] Task 09 – Technical Report

## 📄 License
This project is submitted as academic coursework for SLTC Research University (CCS3440 – Artificial Intelligence).

## 🙏 Acknowledgements
Dataset provided by SLTC Research University for CCS3440 coursework purposes.
