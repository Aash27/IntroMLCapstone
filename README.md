House Price Prediction Using Classical & Literature-Based Machine Learning Models
====================================================================================

Predicting housing prices using Kaggle dataset with 5 ML models + research-based pipelines

Project Overview
-------------------

This project explores multiple machine learning models to predict house prices using the **Ames Housing Dataset**.The goal is to understand how different types of models perform on structured tabular data and how classical approaches compare with pipelines inspired by published research.

The project implements **5 models**:

### Classical ML Models

*   **Linear Regression**
    
*   **Polynomial Regression (Degree 2)**
    
*   **MLP Regressor**
    

### Literature-Based Models

*   **XGBoost (from Xu & Nguyen, 2022)**
    
*   **CatBoost (from IEEE Paper on House Price Prediction)**
    

Each model is trained, evaluated, and compared using both **regression metrics** and **classification-style metrics** (Low / Mid / High price categories).SHAP values are used for feature interpretability in the XGBoost model.

Dataset Source:[https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview)

Repository Structure
------------------------
````markdown
HousePricePrediction/
├── data/
│   ├── train.csv                 # Kaggle dataset
│   └── test.csv (optional)
│
├── preprocessing.py              # Cleaning, encoding, scaling, splitting
│
├── model_linear_regression.py    # Classical Model 1
├── model_polynomial.py           # Classical Model 2
├── model_mlp.py                  # Classical Model 3
│
├── paper_model_xgboost.py        # Literature Model 1 (Xu, 2022)
├── paper_catboost.py             # Literature Model 2 (IEEE, 2021)
│
├── main.py                       # Runs all models + compares metrics
│
├── requirements.txt              # Python dependencies
│
└── README.md                     

````
```text
```


Getting Started
------------------

### **1\. Install Requirements**

`   pip install -r requirements.txt   `

### Required Libraries

*   pandas
    
*   numpy
    
*   scikit-learn
    
*   matplotlib
    
*   seaborn
    
*   xgboost
    
*   catboost
    
*   shap
    

Preprocessing Pipeline (preprocessing.py)
============================================

The preprocessing.py file prepares the **Ames Housing dataset** for all five machine-learning models used in the project.Ensures that the data is cleaned, transformed, and split in a consistent way.

*   Initial dataset loading & inspection
    
*   Target variable exploration
    
*   Distribution analysis (raw + log-transformed)
    
*   Missing value analysis with visualizations
    
*   Feature engineering
    
*   Outlier detection and removal
    
*   Correlation analysis
    
*   Preparation of a clean, model-ready dataset
    

**1\. Target Variable Exploration**
--------------------------------------

### **Original SalePrice Distribution**

*   The distribution of SalePrice is **heavily right-skewed**.
    
*   Most home prices fall around **150k–200k**, with fewer high-value homes.
    
*   High-end properties create a **long right tail**, indicating extreme values.
    

### **Log-Transformed SalePrice Distribution**

*   Using log1p() produces a **more symmetric, bell-shaped distribution**.
    
*   Reduces the influence of extremely expensive homes.
    
*   Makes patterns easier for models to learn.

**2\. Missing Value Analysis**
---------------------------------

A missing-values bar plot shows how many values are missing in each feature.

Key observations:

*   Features like **PoolQC, Alley, Fence, MiscFeature** have extremely high missing counts.
    
*   Many of these indicate **absence of the feature** (e.g., no pool).
    
*   Helps decide:
    
    *   Which columns require imputation
        
    *   Which features may be dropped
        
    *   Which contain meaningful structural zeros
        
**3\. Feature Engineering**
------------------------------

The notebook creates additional structural features:

*   **TotalSF** — combines all major square-footage areas
    
*   **HouseAge** — age of the property at sale
    
*   **RemodelAge** — years since last remodel

**4\. Outlier Detection & Removal**
--------------------------------------

### **Boxplot of SalePrice (Before Outlier Removal)**

*   Many extreme points, especially on the high-end.
    
*   Very long whiskers show large deviations from the median.
    

### **Histogram of SalePrice (Before Outlier Removal)**

*   Confirms **right-skewed distribution**.
    
*   Shows a heavy tail on the right side.
    
**5\. Correlation Analysis**
-------------------------------

### **Top 20 Most Correlated Features with SalePrice**

A bar plot displays the 20 features most correlated with SalePrice.

Insights:

*   Strongest positive correlations include:
    
    *   **OverallQual**
        
    *   **GrLivArea**
        
    *   **GarageCars**
        
    *   **TotalSF**
        
*   Helps identify the most predictive features.
    

### **Correlation Heatmap of Numeric Features**

Visualizes pairwise correlations across all numeric features.

**What this plot reveals:**

*   **Red squares** → strong positive correlation
    
*   **Blue squares** → strong negative correlation
    
*   **White/light areas** → weak or no correlation
    

**Insights:**

*   Structural features group together (basement areas, floor area, garage variables).
    
*   Shows multicollinearity among related features.
    
*   Highlights which variables influence SalePrice the most.    



Methodology
==============

A clear explanation of what each file does.

### **Models Implemented**

#### Classical:

*   Linear Regression
    
*   Polynomial Regression (Degree 2)
    
*   MLP Neural Network
    

#### Literature:

*   **XGBoost** hyperparameter grid from research paper
    
*   **CatBoost** tuned as described in IEEE study
    

### **Evaluation**

Metrics:

*   MSE
    
*   RMSE
    
*   MAE
    
*   R²
    

Classification Metrics (via price binning):

*   Accuracy
    
*   Precision
    
*   Recall
    
*   F1-Score
    

Visualizations:

*   Actual vs Predicted
    
*   Residual plots
    
*   Error distribution
    
*   Confusion Matrix
    
*   SHAP Summary Plot



**Classical Models**
--------------------

### **1\. Linear Regression (model\_linear\_regression.py)**
    
*   Trains a baseline Linear Regression model.
    
*   Computes regression metrics (MSE, RMSE, MAE, R²).
    
*   Converts predictions into price categories (Low/Mid/High) for extra evaluation.
    
*   Generates:
    
    *   Actual vs Predicted scatter
        
    *   Residual plot
        
    *   Prediction error histogram
        
*   Serves as the baseline benchmark for all other models.
    

### **2\. Polynomial Regression (model\_polynomial.py)**
    
*   Adds PolynomialFeatures(degree=2) to model non-linear relationships.
    
*   Trains Linear Regression on the expanded feature set.
    
*   Evaluates regression + classification-style metrics.
    
*   Produces:
    
    *   Actual vs Predicted
        
    *   Residual diagnostics
        
    *   Error distribution plots
        
*   Used to test whether quadratic feature interactions improve accuracy.
    

### **3\. MLP Regressor (model\_mlp.py)**
    
*   Builds a neural network:
    
    *   Two hidden layers
        
    *   ReLU activation
        
    *   Adam optimizer
        
*   Learns non-linear relationships beyond polynomial transformations.
    
*   Outputs:
    
    *   Regression metrics
        
    *   Confusion matrix for price categories
        
    *   Actual vs Predicted visualization
        
    *   Residual and error plots
        
*   Demonstrates how deep learning performs on tabular housing data.
    

**Literature-Based Models**
---------------------------

### **4\. XGBoost (Paper-Inspired) — (paper\_model\_xgboost.py)**

*   Implements the pipeline used in the **Chicago housing price prediction research paper**.
    
*   Performs manual one-hot encoding + parameter tuning (via GridSearchCV).
    
*   Trains a strong gradient-boosted tree regressor optimized for tabular data.
    
*   Produces:
    
    *   Regression metrics
        
    *   SHAP feature importance
        
*   One of the best-performing models in the project.
    

### **5\. CatBoost (IEEE-Inspired) — (paper\_catboost.py)**

*   Implements CatBoostRegressor following methodology from IEEE work.
    
*   Handles categorical features natively (no manual encoding needed).
    
*   Trains with tuned depth, learning rate, and iterations.
    
*   Outputs:
    
    *   MAE, MSE, RMSE, R²
        
    *   Prediction analysis
        
*   Performs extremely well with minimal preprocessing.
    

**Integrated Comparison (main.py)**
-----------------------------------

*   Loads preprocessed data.
    
*   Trains **all 5 models** in one pipeline.
    
*   Collects regression metrics in a unified table.
    
*   Creates:
    
    *   Barplot of model performance
        
    *   Comparison charts (MAE, RMSE, R²)
        
*   Helps determine which model is best suited for this prediction task.




## Model Performance Comparison (Regression Metrics)

| Model                          | Train MAE     | Train MSE     | Train RMSE    | Train R² | Test MAE     | Test MSE     | Test RMSE    | Test R² |
|--------------------------------|----------------|----------------|----------------|----------|---------------|----------------|----------------|----------|
| **Linear Regression**          | 12676.18       | 3.81e+08       | 19529.05       | 0.9365   | 18285.20      | 8.68e+08       | 29475.25       | 0.8867   |
| **Polynomial Regression (Deg 2)** | 6840.76     | 1.35e+02       | 11.61          | 1.0000   | 21595.39      | 1.02e+09       | 31991.19       | 0.8666   |
| **MLP Regressor**              | 15948.50       | 8.23e+08       | 28699.04       | 0.8619   | 18744.39      | 1.05e+09       | 32888.57       | 0.8632   |
| **XGBoost (Paper-inspired)**   | 3821.60        | 2.77e+07       | 5256.46        | 0.9954   | 15623.08      | 6.52e+08       | 25543.27       | 0.9149   |
| **CatBoost (Paper-inspired)**  | 8617.81        | 1.47e+08       | 12121.66       | 0.9754   | 16296.41      | 7.23e+08       | 26896.21       | 0.9057   |





### **Classification Metrics (Price Categories)**

MLP performs best in **precision/recall/F1** for classifying homes into Low / Mid / High categories.XGBoost performs best in **numerical accuracy**, making it the strongest regressor.


  
Key Findings
---------------

*   **XGBoost** achieves the best regression accuracy and lowest error.
    
*   **CatBoost** performs strongly and handles categorical features efficiently.
    
*   **MLP** performs best for classification-style metrics.
    
*   **Polynomial Regression** captures some non-linear patterns but increases variance.
    
*   **Linear Regression** serves as a strong baseline.
    
*   SHAP values show:
    
    *   **OverallQual**, **GrLivArea**, **GarageCars**, and **TotalBsmtSF**are the most influential predictors of sale price.
        

Acknowledgements
-------------------

Dataset:
Ames Housing Dataset — Kaggle Competition
[https://www.kaggle.com/c/house-prices-advanced-regression-techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

Research Papers:

*   Predicting housing prices and analyzing real estate markets in the Chicago suburbs using machine learning.**Kevin Xu & Hieu Nguyen (2022)** — XGBoost model
    
*   Prediction of House Price Using Machine Learning Algorithms.**1G Kiran Kumar, D Malathi Rani, Neeraja Koppula, Syed Ashraf (2021)** — CatBoost model


    

----------

**Aashlesha Rajput**
