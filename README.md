Case Study : Big Mart Sales Prediction


Objective
To predict the sales of products(item_outlet_sales) at various outlets of Big Mart using the provided training sales data and generate the submission file for evaluation containing item_identifier, outlet_identifier, and the predicted sales.

Rationale

My approach is structured into the following ways:

Understanding the Problem Statement
The goal is to estimate the product sales after looking at historical data and help business to improve their sales strategies

Understanding the data
Explored the data and identified the relationship among various independent and dependent variables
Identify the missing values, inconsistencies and outliers 
Check whether the numerical variables need normalization or not.
Decide the mechanism to encode the categorical variables

Exploratory Data Analysis (EDA)

Univariate Analysis
Observed that the data is asymmetrical having positive skewness 
Plotted distribution of categorical variables

Bivariate Analysis
Impact of outlet_type and item_type on sales

Outlier detection and removal
Detected outliers in item_outlet_sales
Capped extreme values in target variable i.e. item_outlet_sales using IQR method. 
After removal of outliers, the model wasn’t giving good results so I decided later not to remove them as they have impact on the model predictions


Data Preprocessing

Normalize the relevant numerical variables including item_mrp, item_visibility and item_weight
Encode the categorical variable
Perform ‘Label Encoding’ on outlet_identifier
Apply ‘One-hot-encoding’ on other categorical features
Feature Engineering: Normalize the item_fat_content column


Model Prediction & Feature Importance
Baseline Model: Linear Regression 
Poor performance due to non-linear relationships

Other model tested:
Random Forest Regressor
	i) R^2 squared without hyper-parameter tuning : 0.573
           ii) R^2 squared after hyper-parameter tuning: 0.607

XGBoost Regressor
		i) R^2 squared without hyper-parameter tuning : 0.538
           ii) R^2 squared after hyper-parameter tuning:  0.501


Hyperparameter tuning
I applied GridSearchCV for Random Forest Regressor Random Search CV for XGBoost Regressor and achieved better results with Random forest. Hence, I am using a Random Forest trained model for predicting sales on test data.

Feature Importance: Outlet_type and item_mrp are the features that have a significant impact on the model predictions.


Submission Generation

Used the trained model on provided test data to predict the sales of the products after applying all the data transformations and normalizations as per training data

Generated a submission.csv with Item_Identifier, Outlet_Identifier and Item_Outlet_Sales.


Final Outcome
Cleaned and transformed data effectively
Build a robust model using Random Forest Regressor
Achieved a reliable prediction for submission

