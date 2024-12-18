# Titanic Survival Prediction

This repository contains a data science project aimed at predicting survival outcomes for passengers aboard the Titanic. The project is based on the Kaggle **Titanic: Machine Learning from Disaster** competition, where the goal is to use machine learning algorithms to predict whether a passenger survived or not.

## Project Overview

The Titanic dataset contains various features about passengers, such as age, class, sex, and ticket fare, which are used to predict the likelihood of survival. The notebook explores the dataset, performs data preprocessing, applies machine learning models, and evaluates the results.

### **Key Steps in the Notebook:**
1. **Data Exploration and Preprocessing:**
   - Analyzing the dataset to understand the distribution of features.
   - Handling missing values and encoding categorical variables.
   
2. **Feature Engineering:**
   - Creating new features based on existing data (e.g., family size, titles).
   
3. **Modeling:**
   - Implementing machine learning algorithms like Logistic Regression, Random Forest, and Support Vector Machines (SVM) to build predictive models.
   
4. **Model Evaluation:**
   - Evaluating model performance using metrics like accuracy, precision, recall, and F1-score.
   
5. **Submission:**
   - Preparing the results in the format required for Kaggle competition submission.

## Dataset

The dataset used in this project is from Kaggle’s **Titanic: Machine Learning from Disaster** competition. It consists of two main CSV files:
- **train.csv:** Contains data on passengers, including survival status and various features like age, sex, and class.
- **test.csv:** Contains data on passengers for whom we need to predict survival.

Dataset source: [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

## Requirements

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

You can install the required packages by running the following:
   ```bash 
   pip install pandas numpy matplotlib seaborn scikit-learn
