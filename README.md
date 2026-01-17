# 🩺 Diabetes Risk Prediction System

An end-to-end machine learning application that predicts the likelihood of diabetes in patients based on medical history and demographic details. Built with Python, Scikit-Learn, and Streamlit.

# Project Overview
In medical diagnostics, the cost of a False Negative (telling a sick patient they are healthy) is much higher than a False Positive.
This project focuses on optimizing the Recall of the model to ensure potential diabetes cases are not missed, while maintaining a reasonable accuracy.
# 🔑 Key Features
Data Analysis: Handled missing values (imputation), detected outliers, and analyzed feature correlations.
Model Comparison: Evaluated performance across multiple families of algorithms:
Linear: Logistic Regression
Probabilistic: Naive Bayes
Distance-based: K-Nearest Neighbors (KNN)
Ensemble: Random Forest
Imbalance Handling: Utilized class_weight='balanced' in Random Forest to handle the dataset's class imbalance.
Interactive Interface: A user-friendly web app built with Streamlit for real-time inference.
# 🛠️ Technologies Used
Python 3.9+
Scikit-Learn (Model Training & Evaluation)
Pandas & NumPy (Data Manipulation)
Streamlit (Web Interface)
Joblib (Model Persistence)

# 📊 Model Performance
After extensive testing, the Balanced Random Forest was selected as the final model due to its superior safety profile (High Recall).
Model	Accuracy	Recall (Sensitivity)	Verdict
Logistic Regression	71%	71%	Baseline
Naive Bayes	75%	67%	High Precision, Low Recall
K-Nearest Neighbors	75%	75%	Good
Random Forest (Balanced)	76%	82%	Selected (Best Safety)
The final model identified 82% of all positive diabetes cases in the test set, minimizing the risk of missed diagnoses.

# 📂 File Structure
app.py: The main source code for the Streamlit web application.
diabetes_prediction_final.ipynb: Jupyter Notebook containing data cleaning, EDA, and model training steps.
diabetes_model_rf.pkl: The trained and serialized Random Forest model.
diabetes.csv: The dataset used for training.

# requirements.txt: List of Python dependencies.

💻 How to Run Locally

Clone the repository:
code
Bash
git clone https://github.com/lokesh172757/Diabetes-Risk-Prediction-System.git

Navigate to the directory:
code
Bash
cd diabetes

Install dependencies:
code
Bash
pip install -r requirements.txt

Run the App:
code
Bash
streamlit run app.py
