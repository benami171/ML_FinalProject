# Text Classification on the 20 Newsgroups Dataset

This project investigates the performance of various classifiers on the [20 Newsgroups](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html) dataset using different vectorization techniques, dimensionality reduction, and data preprocessing steps. The main goals are to explore classifier performance differences between TF‑IDF and CountVectorizer representations, evaluate the impact of dimensionality reduction techniques (TruncatedSVD and Feature Agglomeration), and perform hyperparameter optimization using GridSearchCV.

## Overview

In this project we:
- **Load and preprocess** the 20 Newsgroups dataset while removing headers, footers, and quoted texts to reduce noise.
- **Visualize the data distribution** across classes and perform hierarchical clustering of class centroids to understand class similarities.
- **Vectorize the text data** using two approaches:
  - TF‑IDF (using a custom analyzer with cleaning, tokenization, stopword removal, and stemming)
  - Count Vectorization
- **Evaluate a suite of classifiers** including:
  - Logistic Regression
  - Ridge Classifier
  - Decision Tree
  - Random Forest
  - LinearSVC
  - K-Nearest Neighbors
  - Complement Naive Bayes  
- **Perform oversampling** to mitigate imbalanced data issues for minority classes.
- **Apply dimensionality reduction** using TruncatedSVD and Feature Agglomeration.
- **Optimize hyperparameters** of each classifier using GridSearchCV.
- **Visualize results** with confusion matrices and performance tables.

## Project Structure

- **Data Loading and Cleaning:**  
  The raw text data is loaded from the 20 Newsgroups dataset with extraneous elements (headers, footers, quotes) removed.
  
- **Preprocessing & Vectorization:**  
  A custom analyzer function is defined. The text data is vectorized using both TF‑IDF and CountVectorizer. Comparisons are made between the two vectorizations.
  
- **Data Distribution & Exploration:**  
  Distribution plots (bar charts) show the number of documents per class. Hierarchical clustering on class centroids is used to reveal class similarities.
  
- **Classifier Evaluation:**  
  A set of classifiers are wrapped into evaluation functions that compute metrics such as accuracy, precision, recall, F1-score, and training time.  
  Predictions, confusion matrices, and misclassification analysis are generated.
  
- **Dimensionality Reduction:**  
  TruncatedSVD is used to reduce the high-dimensional vectorized data while preserving a target threshold (e.g., 70% explained variance). Feature Agglomeration is also tested after converting the data to dense format.
  
- **Hyperparameter Tuning:**  
  GridSearchCV is used for each classifier (with the appropriate parameter grids) to identify optimal settings.

## Requirements

- Python ≥ 3.8  
- Jupyter Notebook or JupyterLab  
- Libraries:  
  - scikit-learn  
  - imbalanced-learn  
  - pandas  
  - numpy  
  - matplotlib  
  - seaborn  
  - nltk  
  - scipy

Install the required packages via pip:

```bash
pip install scikit-learn imbalanced-learn pandas numpy matplotlib seaborn nltk scipy
```

## Running the Project

1. **Data Preprocessing & Vectorization:**  
   Run the notebook cells that load, clean, and preprocess the raw text data. Then apply TF‑IDF and Count vectorizers with your custom analyzer.
   
2. **Exploratory Data Analysis:**  
   Visualize the data distribution and hierarchical clustering of class centroids.
   
3. **Classifier Evaluation:**  
   Evaluate the baseline performance of various classifiers on both vectorized representations.  
   
4. **Handling Imbalanced Data:**  
   Apply RandomOverSampler on the TF‑IDF data and rerun classifier evaluations.
   
5. **Dimensionality Reduction:**  
   Use TruncatedSVD (and Feature Agglomeration) to reduce the feature space and evaluate the impact on classification performance.
   
6. **Hyperparameter Optimization:**  
   Run GridSearchCV to tune each classifier’s hyperparameters and then evaluate the optimized models on the test set.
   
7. **Results & Analysis:**  
   Review the generated performance tables, confusion matrices, and misclassification insights to compare the approaches.

## Observations & Considerations

- **Vectorization:**  
  TF‑IDF data produced better results in this project compared to Count Vectorizer.
  
- **Dimensionality:**  
  While reducing dimensions (via SVD) can help reduce noise and overfitting, it may also degrade performance if key discriminative information is lost or if models that handle sparse data efficiently are slowed by dense data.
  
- **Model Behavior:**  
  Some classifiers (e.g., Complement Naive Bayes and LinearSVC) are highly effective on high-dimensional, sparse data, and show sensitivity to specific hyperparameter settings.


## Kaggle link

You can access the project in kaggle through [this link](https://www.kaggle.com/code/galbenami/text-classification-20-newsgroups-dataset)