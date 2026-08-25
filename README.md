# Bosch Manufacturing Defect Prediction

## Project Overview

This project develops machine learning models to predict manufacturing
defects using the **Bosch Production Line Performance** dataset from
Kaggle.

The objective is to identify defective products before they leave the
production line by analyzing large-scale manufacturing sensor data.
Multiple machine learning approaches were evaluated to understand how
feature engineering and model complexity affect predictive performance.

------------------------------------------------------------------------

# Business Problem

In modern manufacturing, even a small number of defective products can
lead to:

-   Increased production costs
-   Product recalls
-   Customer dissatisfaction
-   Warranty claims
-   Production downtime

Early defect detection allows manufacturers to inspect high-risk
products before shipment, improving product quality while reducing
operational costs.

------------------------------------------------------------------------

# Dataset

**Source**

Bosch Production Line Performance (Kaggle)

### Files Used

-   `train_numeric.csv`
-   `train_date.csv`

### Target Variable

-   **Response**
    -   0 = Non-defective product
    -   1 = Defective product

### Dataset Characteristics

-   Over **1 million manufacturing records**
-   Nearly **1,000 numeric manufacturing measurements**
-   Extremely imbalanced target variable
-   Large number of missing values
-   Anonymous production station features

------------------------------------------------------------------------

# Project Workflow

``` text
Raw Manufacturing Data
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Feature Engineering
        │
        ▼
Model Development
        │
        ▼
Model Evaluation
        │
        ▼
Business Interpretation
```

------------------------------------------------------------------------

# Project Structure

``` text
bosch-defect-prediction/

├── 1. Data/
├── 2. Notebook/
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Modeling.ipynb
│   └── 04_Evaluation.ipynb
├── 3. Models/
├── README.md
└── .gitignore
```

------------------------------------------------------------------------

# Modeling Strategy

Four progressively more advanced modeling approaches were evaluated.

  Version     Description
  ----------- ---------------------------------------------------------
  Version 1   Numeric baseline using engineered numeric features
  Version 2   Numeric features + missing-value indicators
  Version 3   Numeric + missing indicators + production date features
  Version 4   Deep Learning (MLP)

------------------------------------------------------------------------

# Model Performance

  -----------------------------------------------------------------------------
  Model             Accuracy     Precision      Recall     F1 Score     ROC-AUC
  ------------ ------------- ------------- ----------- ------------ -----------
  V1 Numeric           0.875         0.019       0.406        0.036       0.695
  Baseline                                                          

  V2 Numeric +         0.856         0.016       0.409        0.032       0.708
  Missing                                                           
  Indicators                                                        

  **V3             **0.866**     **0.019**   **0.425**    **0.036**   **0.714**
  Numeric +                                                         
  Missing +                                                         
  Date                                                              
  Features**                                                        

  V4 Deep              0.698         0.011   **0.552**        0.021       0.674
  Learning                                                          
  (MLP)                                                             
  -----------------------------------------------------------------------------

------------------------------------------------------------------------

# Best Model

**Version 3** achieved the best overall performance.

**Performance Highlights**

-   ROC-AUC: **0.714**
-   Recall: **0.425**
-   Best overall balance between discrimination and defect detection

The model combines:

-   Numeric manufacturing measurements
-   Missing-value indicators
-   Production timing features

------------------------------------------------------------------------

# Key Findings

-   Feature engineering consistently improved model performance.
-   Missing-value patterns contain useful manufacturing information.
-   Production timing features further improved defect prediction.
-   Deep learning achieved the highest recall but lower overall ROC-AUC
    due to increased false positives.
-   For structured manufacturing data, domain-specific feature
    engineering provided greater gains than increasing model complexity.

------------------------------------------------------------------------

# Technologies Used

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   XGBoost
-   TensorFlow / Keras
-   SHAP
-   Matplotlib

------------------------------------------------------------------------

# Future Improvements

-   Incorporate categorical manufacturing features
-   Hyperparameter optimization
-   Compare LightGBM and CatBoost
-   Ensemble learning
-   Cost-sensitive optimization
-   Real-time deployment pipeline

------------------------------------------------------------------------

# Conclusion

This project demonstrates an end-to-end machine learning workflow for
manufacturing defect prediction. Among four modeling approaches,
combining numeric measurements, missing-value indicators, and production
timing features produced the strongest overall performance. The results
highlight the value of feature engineering and manufacturing domain
knowledge when building predictive quality control systems.
