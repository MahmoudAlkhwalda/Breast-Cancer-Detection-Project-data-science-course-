# Breast Cancer Detection Project

## 📌 Overview
This Data Science project aims to predict the diagnosis of breast cancer (Malignant vs. Benign) using machine learning techniques. It utilizes the **Wisconsin Breast Cancer Diagnostic Dataset** to train and evaluate a Support Vector Machine (SVM) model. The workflow includes extensive exploratory data analysis (EDA), dimensionality reduction with PCA, and hyperparameter tuning to ensure robust performance.

## 🚀 Key Features
*   **Data Preprocessing**: Handling of missing values (if any), outlier detection, and feature scaling.
*   **Exploratory Data Analysis (EDA)**: Visualizations using `Seaborn` and `Matplotlib` to understand feature distributions and correlations.
*   **Dimensionality Reduction**: Application of **Principal Component Analysis (PCA)** to visualize high-dimensional data in 2D space.
*   **Machine Learning Model**: Implementation of **Support Vector Machines (SVM)** with a linear kernel for classification.
*   **Model Optimization**: Usage of `GridSearchCV` to find the optimal hyperparameters (e.g., regularization parameter `C`).
*   **Evaluation**: Comprehensive model evaluation using Cross-Validation (10-fold) and metrics like Accuracy, Precision, Recall, and F1-score.

## 🛠️ Technologies Used
*   **Python**
*   **Jupyter Notebook**
*   **Pandas** & **NumPy** (Data Manipulation)
*   **Matplotlib** & **Seaborn** (Visualization)
*   **Scikit-Learn** (Machine Learning: SVM, PCA, GridSearchCV, Cross-Validation)

## 📂 Dataset
The project uses the **Wisconsin Breast Cancer Diagnostic Dataset** (`data.csv`).
*   **Target Variable**: `diagnosis` (M = Malignant, B = Benign)
*   **Features**: Ten real-valued features are computed for each cell nucleus (e.g., Radius, Texture, Perimeter, Area, Smoothness, etc.).

## 📊 Model Performance
The SVM model, after hyperparameter tuning and cross-validation, achieved high performance metrics:
*   **Accuracy**: ~98%
*   **Precision**: ~98%
*   **Recall**: ~95%
*   **F1-Score**: ~97%

## 🔧 How to Run
1.  Ensure you have Python installed along with the required libraries:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
2.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook best_cancer_model.ipynb
    ```
3.  Run the cells sequentially to reproduce the analysis and results.

## 📈 Visualizations
The notebook includes various plots such as:
*   **Countplots** for target class balance.
*   **PCA Scatter Plots** showing the separation of classes in reduced dimensions.
*   **Bar Charts** comparing model performance metrics across cross-validation folds.
