# 🤖 AI & ML Intelligent System and Model Tuning — Python

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Google%20Colab-Notebook-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Google Colab">
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge&logo=python&logoColor=white" alt="Matplotlib">
  <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
</p>

<p align="center">
  <strong>DevSphere Internship — Week 5</strong><br>
  AI-Based Intelligent Classification System and Machine Learning Model Evaluation & Tuning
</p>

---

## 📌 Project Overview

This repository contains two practical tasks completed for **Week 5 of the DevSphere Internship**, implemented in one Google Colab notebook:

- 🤖 **Task 1 — AI:** Intelligent classification system using a Random Forest classifier.
- 📊 **Task 2 — ML:** Model comparison and hyperparameter tuning using Logistic Regression, Decision Tree, and Random Forest.

Both tasks use the **Breast Cancer Wisconsin dataset** provided by Scikit-learn. The notebook demonstrates data inspection, preprocessing, model training, predictions, evaluation metrics, and visualizations.

## 📂 Repository Structure

```text
ai_ml_intelligent_model_tuning_python/
├── AI_ML_Week5_Intelligent_System.ipynb
├── README.md
└── DevSphere_Week5_Report.docx
```

## 🧰 Technologies & Tools

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Scikit-learn | Dataset loading, preprocessing, model training, tuning, and evaluation |
| Pandas | Dataset inspection |
| NumPy | Numerical operations |
| Matplotlib | Charts and visualizations |
| Google Colab | Notebook development and execution |
| GitHub | Repository hosting |

---

# 🤖 Task 1 — AI-Based Intelligent Classification System

## 🎯 Objective

Develop a working classification system that trains a model, processes data, predicts classes, evaluates accuracy, accepts user input, and visualizes results.

## 📚 Dataset Details

**Dataset:** Breast Cancer Wisconsin Dataset

| Property | Value |
|---|---:|
| Total samples | 569 |
| Features | 30 |
| Classes | 2 |
| Class labels | Malignant and Benign |
| Training samples | 455 |
| Testing samples | 114 |
| Model | Random Forest Classifier |

The dataset was loaded directly from Scikit-learn. A Pandas DataFrame was created from the feature matrix and feature names to display the first five records.

## ⚙️ Workflow

```text
Breast Cancer Dataset
        ↓
Dataset Inspection
        ↓
Train/Test Split
        ↓
Data Preprocessing
        ↓
Random Forest Classifier
        ↓
Predictions and Evaluation
        ↓
Confusion Matrix
        ↓
Feature Importance
        ↓
Select a Test Sample
        ↓
Predicted Class and Confidence
```

## 📈 Actual Results

The provided notebook output shows the following classification results:

| Metric | Result |
|---|---:|
| Accuracy | **94.74%** |
| Malignant Precision | 0.9286 |
| Malignant Recall | 0.9286 |
| Malignant F1 Score | 0.9286 |
| Benign Precision | 0.9583 |
| Benign Recall | 0.9583 |
| Benign F1 Score | 0.9583 |

### Confusion Matrix

| Actual / Predicted | Malignant | Benign |
|---|---:|---:|
| Malignant | 39 | 3 |
| Benign | 3 | 69 |

### Feature Importance

The notebook visualizes the top 10 features identified by the trained Random Forest model. In the supplied output, **worst perimeter**, **worst area**, and **worst concave points** are among the most important features.

### Sample Predictions

```text
1. Actual: malignant | Predicted: malignant
2. Actual: benign | Predicted: benign
3. Actual: malignant | Predicted: malignant
4. Actual: benign | Predicted: malignant
5. Actual: malignant | Predicted: malignant
```

### User Input Demonstration

The supplied execution selected test sample **14**.

```text
Predicted Class: malignant
Prediction Confidence: 100.00%
```

The interactive section selects a complete sample from the test dataset, rather than asking the user to provide only a few of the dataset's 30 features. This is a demonstration for learning purposes and is **not a medical diagnostic tool**.

## 📸 Task 1 Screenshots

The `100` series corresponds to Task 1. Upload these image files to the repository root to display them below.

### 1. Dataset Information and Sample Records

![Task 1 — Dataset Information](101.PNG)

### 2. Class Distribution and Model Evaluation

![Task 1 — Class Distribution and Results](102.PNG)

### 3. Classification Report and Confusion Matrix

![Task 1 — Confusion Matrix](103.PNG)

### 4. Feature Importance and User Input Prediction

![Task 1 — Feature Importance and Prediction](104.PNG)

### 5. Dataset Table Preview

![Task 1 — Dataset Preview](105.PNG)

---

# 📊 Task 2 — Machine Learning Model Evaluation & Tuning

## 🎯 Objective

Train multiple machine learning models, compare their performance, tune a model using cross-validation, evaluate it with multiple metrics, and visualize the results.

## 📚 Dataset Details

**Dataset:** Breast Cancer Wisconsin Dataset

| Property | Value |
|---|---:|
| Total samples | 569 |
| Features | 30 |
| Training samples | 455 |
| Testing samples | 114 |
| Models compared | Logistic Regression, Decision Tree, Random Forest |
| Tuning method | GridSearchCV with stratified 5-fold cross-validation |

## 🤖 Model Comparison

The provided output reports the following results before tuning:

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9954 |
| Decision Tree | 0.9123 | 0.9559 | 0.9028 | 0.9286 | 0.9157 |
| Random Forest | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9937 |

## 🎛️ Hyperparameter Tuning

Logistic Regression was tuned using `GridSearchCV`, with stratified five-fold cross-validation. The provided output shows:

| Tuning Result | Recorded Value |
|---|---|
| Best `C` parameter | 0.1 |
| Best solver | `liblinear` |
| Best cross-validation accuracy | 98.24% |
| Tuned model test accuracy | 98.25% |
| Precision | 0.9861 |
| Recall | 0.9861 |
| F1 Score | 0.9861 |
| ROC AUC | 0.9960 |

## 📉 Confusion Matrix

| Actual / Predicted | Malignant | Benign |
|---|---:|---:|
| Malignant | 41 | 1 |
| Benign | 1 | 71 |

## 📈 Interpretation

- Logistic Regression achieved the highest accuracy among the three original models in the supplied results.
- Hyperparameter tuning selected `C = 0.1` and the `liblinear` solver.
- The tuned Logistic Regression model achieved **98.25% test accuracy**.
- The model comparison, confusion matrix, ROC curve, and before/after accuracy chart are generated in the notebook.

## 🔎 Sample Predictions After Tuning

```text
1. Actual: malignant | Predicted: malignant
2. Actual: benign | Predicted: benign
3. Actual: malignant | Predicted: malignant
4. Actual: benign | Predicted: benign
5. Actual: malignant | Predicted: malignant
```

## 📸 Task 2 Screenshots

The `200` series corresponds to Task 2. Upload these image files to the repository root to display them below.

### 1. Model Comparison and Tuning Metrics

![Task 2 — Model Comparison](201.PNG)

### 2. Confusion Matrix and ROC Curve

![Task 2 — Confusion Matrix and ROC Curve](202.PNG)

### 3. Accuracy Comparison and Sample Predictions

![Task 2 — Accuracy Comparison](203.PNG)

---

# 📊 Week 5 Results Summary

| Task | Model | Result |
|---|---|---:|
| 🤖 AI — Intelligent Classification | Random Forest Classifier | **94.74% Accuracy** |
| 📊 ML — Model Evaluation & Tuning | Tuned Logistic Regression | **98.25% Accuracy** |

## 🎓 Learning Outcomes

- Inspecting and preparing a dataset
- Creating a DataFrame for readable dataset inspection
- Splitting data into training and testing sets
- Building Scikit-learn pipelines
- Training a Random Forest classifier
- Generating predictions and handling interactive input
- Comparing multiple classification models
- Tuning hyperparameters using GridSearchCV
- Evaluating accuracy, precision, recall, F1 score, and ROC AUC
- Interpreting confusion matrices
- Visualizing feature importance and model comparisons

## 🚀 How to Run

1. Open `AI_ML_Week5_Intelligent_System.ipynb` in Google Colab.
2. Run the first code block for Task 1.
3. Enter a test sample number when prompted.
4. Run the second code block for Task 2.
5. Review the printed metrics, confusion matrices, and charts.

Both tasks load the dataset directly from Scikit-learn; no separate dataset file is needed.

---

## 👨‍💻 Author

**Abdul Samad**  
GitHub: [abdulsamad010](https://github.com/abdulsamad010)

**Internship:** DevSphere — Week 5
