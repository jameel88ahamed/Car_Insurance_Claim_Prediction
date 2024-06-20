# Car_Insurance_Claim_Prediction

## Introduction
The project aims to develop a predictive model that assesses the likelihood of policyholders filing a claim and analyzes the predicted claim amount. As automobiles become a necessity for every family, car insurance prospers, requiring companies to sell tailored plans to diverse customers. Traditional statistics cannot accurately judge customer diversity and correlations, but machine learning enables deeper data analysis and more accurate predictions. Based on research on car insurance, this project constructs machine learning models to classify customers by characteristics and predict claim amounts. By analyzing historical claims data, this study explores machine learning's role in predicting car insurance claims, highlighting its efficiency and accuracy over traditional methods. Accurate predictions help insurers allocate resources, streamline processes, and enhance policyholder experience.

## Problem Statement
The objective of this project is to develop a machine learning model and deploy it as a user-friendly web application that predicts the car insurance claim amount. The main goal is to accurately forecast the amount of money a user could claim for insurance based on various characteristics. This involves researching the subject of car insurance, constructing machine learning models to classify customers by their attributes, and predicting the claim amounts, thereby aiding both insurers and customers in making informed decisions.

## Required Libraries:
- Streamlit : To Create Graphical user Interface and build web application.
- Pandas : To Clean and maipulate the data.
- NumPy: A library for numerical computations in Python.
- Matplotlib: A plotting library for creating visualizations.
- Seaborn: A data visualization library built on top of Matplotlib.
- scipy.stats-skew: To check and correct skewness.
- Pickle: Library used for saving the model and use whenever required.
- Scikit-learn: A machine learning library that provides various regression and classification algorithms & Evaluation Metrices.

## Workflow:
1. Loading Data & Preprocessing:
   - By importing the required libraries, loaded the dataset to examine its structure, and processed the headers for clarity and understanding the basic description, I explored the summary, shape, and data types of the dataset.
   - Found out the missing values and treated them by imputing KNN imputer, median imputaion and mode imputation.
2. Exploratory Data Analysis:
   - Started the EDA with basic statistical analysis by separating the numeric and categorical variables.
   - Checking the distributions of datapoints and getting the insights.
   - Analyzing the distribution of the target columns with other features.
   - Checking the correlation by plotting heatmap.
3. Outlier Analysis: In order take care of the outliers that might affect the working of the model, with the help of boxplot analysed which columns have outliers and treated them by writing a function.
4. Model Selection and Training:
   - We check the evaluation score by writing a single function that can split the dataset, build the model, find RMSE value, and the best parameters to get the best score, I concluded that Random Forest Regressor performs better with lower RMSE value and good r2-score.
   - After selecting the model as Random Forest Regressor, I decided to optimize the model by hyperparameter tuning with gridsearchCV & saving the best estimator model.
5. Model Evaluation: Evaluating the model's predictive performance using regression metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), or Root Mean Squared Error (RMSE) and R2 Score on test dataset.
6. Streamlit Web Application: Developing a user-friendly web application using Streamlit that allows users to input details of a kidsdriv, homekids, car details, age, income, etc., and predicting the Claim Amount.
7. Deployment on Render: Deploying the Streamlit application on the Render platform to make it accessible to users over the internet.
8. Testing and Validation: Thoroughly testing the deployed application to ensure it functions correctly and provides accurate predictions.

## Learning Outcomes:
- Got more experience on Python and its data analysis libraries, including Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, and Streamlit.
- Acquiring expertise in data preprocessing techniques, encompassing handling missing values, imputing missing values, outlier detection, skewness handing and converting data into correct data types format to optimize data for machine learning modeling.
- Understanding and visualizing the data distribution using EDA techniques such as boxplots, distribution plot, density plot, & histograms.
- Learned advanced machine learning techniques including regression and random forest for predictive modeling, and tuning hyperparametres with gridsearchCV alongside optimizing models through evaluation metrics like MSE, MAE, RMSE & R2-score.
- Learned about creating a web application using the Streamlit module to showcase the machine learning models and make predictions on new data.
- Learned to deploy the model on the Render platform to help buyers and sellers accurately predict the resale price based on their requirements.

## Result & Conclusion: 
Achieved an R² score of 0.997761 and an MSE of 0.002168, indicating a highly accurate model. We can conclude that being able to predict whether a customer will claim insurance on their vehicle is crucial for the business of insurance providers.

## Application Link:
[Link](https://car-insurance-claim-prediction.onrender.com/)
