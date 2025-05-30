#!/usr/bin/env python
# coding: utf-8

# # Install the required libraries

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# In[2]:


pd.options.mode.chained_assignment = None  # default='warn'


# In[3]:


file_name = "train_mart_sales_data.csv"

train_df = pd.read_csv(file_name)
train_df.head(2)


# In[4]:


# Checking the count in all the columns
train_df.count()


# In[6]:


print(train_df['Item_Outlet_Sales'].min())
print(train_df['Item_Outlet_Sales'].max())


# In[5]:


train_df.info()


# In[6]:


# Now check the summary of the numerical columns
train_df.describe()


# # Data Transformation

# In[7]:


# Convert the columns into lower case
train_df = train_df.rename(columns = str.lower)


# 
# 
# Replace NaN values in both numerical and categorical columns
# 
# 

# In[8]:


# Replace NaN with 0 in numeric columns
cols_to_replace = ['item_weight', 'item_visibility', 'item_mrp', 'outlet_establishment_year', 'item_outlet_sales']
train_df[cols_to_replace] = train_df[cols_to_replace].replace(np.nan, 0)

train_df = train_df.drop('outlet_establishment_year', axis = 1)


# In[9]:


# Replace NaN with None in categorical columns
categorical_cols = train_df.select_dtypes(include='object').columns

for col in categorical_cols:
    train_df[col] = train_df[col].fillna('none')


# Standardize the numerical columns by apply Min-max scaling

# In[10]:


scaler = MinMaxScaler()
scaler.fit(train_df[['item_weight', 'item_visibility', 'item_mrp']])
train_df[['item_weight', 'item_visibility', 'item_mrp']]= pd.DataFrame(scaler.transform(train_df[['item_weight', 'item_visibility', 'item_mrp']]))


# Compute the missing values
# 
# 1) item_weight: Fill by median per item_identifier
# 2) item_visibility: Replace 0s by mean

# In[11]:


train_df['item_weight'] = train_df.groupby('item_identifier')['item_weight'].transform(lambda x:x.fillna(x.median))

visibility_mean = train_df['item_visibility'].mean()
train_df['item_visibility'] = train_df['item_visibility'].replace(0, visibility_mean)


# # Exploratory Data Analysis (EDA)

# # Univariate & Bivariate Analysis

# In[12]:


plt.figure(figsize = (10,5))
sns.histplot(train_df['item_outlet_sales'], bins = 30, kde = True)
plt.title('Distribution of Item Outlet Sales')
plt.show()


# In[13]:


# Categorical feature counts
categorical_features = ['item_fat_content', 'item_type', 'outlet_identifier', 'outlet_size', 'outlet_location_type', 'outlet_type']

for col in categorical_features:
    plt.figure(figsize = (10,4))
    sns.countplot(data = train_df, x = col)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation = 45)
    plt.show()


# In[14]:


# Sales by Outlet Type
plt.figure(figsize = (8,5))
sns.boxplot(data = train_df, x = 'outlet_type', y = 'item_outlet_sales')
plt.title('Sales by Outlet Type')
plt.xticks(rotation = 45)
plt.show()


# In[15]:


# Sales by item_type
plt.figure(figsize = (8,5))
sns.boxplot(data = train_df, x = 'item_type', y = 'item_outlet_sales')
plt.title('Sales by Item Type')
plt.xticks(rotation = 90)
plt.show()


# In[16]:


# Co-relation heatmap
plt.figure(figsize = (8,6))
sns.heatmap(train_df.corr(numeric_only=True), annot = True, cmap = 'coolwarm')
plt.title('Co-relation matrix')
plt.show()


# # Insights after Univariate & Bivariate analysis
# 1) Asymmetry: Right-skewed which clearly shows that most of the data is concerntrated on the left hand side.
# 2) Mean vs median: Mean > median
# 3) Baking Goods is the type of item which sells th most.
# 4) Supermarket Type 1 generate the highest sales. 
# 5) Presence of missing and inconsistent data in outlet_size.

# # Outlier analysis with removal

# In[17]:


# Univariate Distribution - Item Outlet sales
plt.figure(figsize = (8,5))
sns.histplot(train_df['item_outlet_sales'], kde = 30, bins = True)
plt.title('Distribution of item outlet sales')
plt.show()


# # Key Observations and insights
# 1) It is positively skewed which indicates that the majority of item iutlet sales are concerntrates on lower values.
# 2) There may be few high-selling items that drive the majority of sales.

# In[18]:


# Boxplot for outlier detection
plt.figure(figsize = (8,5))
sns.boxplot(x = train_df['item_outlet_sales'])
plt.title('Boxplot for item output sales')
plt.show()


# # Observations 
# 
# 1) Presence of outliers indicates that there may be some items driving high sales
# 2) The majority of the data points fall between 0 and 6000.

# In[19]:


# No of outliers detected

Q1 = train_df['item_outlet_sales'].quantile(0.25)
Q3 = train_df['item_outlet_sales'].quantile(0.75)

IQR = Q3-Q1

lower_bound = Q1 - 1.5* IQR
upper_bound = Q3 + 1.5* IQR

outliers = train_df[(train_df['item_outlet_sales']< lower_bound) | (train_df['item_outlet_sales'] > upper_bound)]
print(f'Outliers detected: {outliers.shape[0]}')


# In[24]:


# Removal of outliers
train_df_filtered = train_df[(train_df['item_outlet_sales']>= lower_bound) & (train_df['item_outlet_sales'] <= upper_bound)]

train_df_filtered.count()


# In[25]:


# Compare original and filtered distribution

plt.figure(figsize = (12,5))
plt.subplot(1,2,1)
sns.histplot(train_df['item_outlet_sales'], kde = True, color = 'red')
plt.title('Original Sales Distribution')


plt.subplot(1,2,2)
sns.histplot(train_df_filtered['item_outlet_sales'], kde = True, color = 'green')
plt.title('Filtered Sales Distribution with no outliers')
plt.show()


# # Outlier analysis insights
# 1) Outliers can skew the sales preduction and and lead to biased co-efficients
# 2) IQR filtering helps in cleaning extreme cases
# 3) After filtering out outliers, the sales looks less skewed.
Outlier removal had a noticeable impact on the model's R^2 score. Therefore, I chose to retain the outliers in the dataset, as items priced above 6,000 appear to significantly contribute to higher sales and are valuable for accurate prediction.
# # Feature Engineering

# In[20]:


# Normalize item_fat_content column

fat_map = {'LF': 'Low Fat', 'low fat' : 'Low Fat', 'reg' : 'Regular'}
train_df['item_fat_content'] = train_df['item_fat_content'].replace(fat_map)


# In[21]:


# Encode categorical feature

# Label encoding on outlet identifier

label_encode = LabelEncoder()
train_df['outlet'] = label_encode.fit_transform(train_df['outlet_identifier'])

# One-hot encoding
train_df = pd.get_dummies(train_df, columns = ['item_fat_content', 'outlet_size', 'outlet_location_type', 'outlet_type', 'outlet', 'item_type'])


# # Feature Selection & Model Training

# In[22]:


X = train_df.drop(columns = ['item_identifier', 'item_outlet_sales','outlet_identifier'])
y = train_df['item_outlet_sales']


# In[23]:


# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.2, random_state= 42)


# # Linear Regression

# In[26]:


lr = LinearRegression()
lr.fit(X_train, y_train)


# In[27]:


y_pred = lr.predict(X_val)
r2 = r2_score(y_val, y_pred)

print('R^2:', r2)


# # RandomForest

# In[28]:


# Model training
rf_regressor = RandomForestRegressor(random_state = 42)
rf_regressor.fit(X_train, y_train)


# In[29]:


# Prediction and evaluation

y_pred = rf_regressor.predict(X_val)
r2 = r2_score(y_val, y_pred)

print('R^2:', r2)


# In[30]:


rf_imp = rf_regressor.feature_importances_
features= pd.Series(rf_imp, index = X.columns).sort_values(ascending = False)
plt.figure(figsize = (10,6))
features[:10].plot(kind = 'barh')
plt.title('Top 10 Imp Features')

plt.show()


# # XGBoost Regressor

# In[31]:


xgb_regressor = XGBRegressor(random_state = 42)
xgb_regressor.fit(X_train, y_train)


# In[32]:


# Prediction and evaluation

y_pred = xgb_regressor.predict(X_val)
r2 = r2_score(y_val, y_pred)

print('R^2:', r2)


# In[33]:


xgb_imp = xgb_regressor.feature_importances_
features= pd.Series(xgb_imp, index = X.columns).sort_values(ascending = False)
plt.figure(figsize = (10,6))
features[:10].plot(kind = 'barh')
plt.title('Top 10Imp Features')
plt.show()


# Item_mrp and outlet_type features are considered as important features for predicting the sales of the products.

# # Hyperparameters tuning 

# # Grid Search in Random Forest Regressor

# In[34]:


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(rf_regressor, param_grid=param_grid, cv=5, verbose =2, n_jobs = -1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Estimator:", grid_search.best_estimator_)

best_rf_model = grid_search.best_estimator_

### PREDICT after applying the best parameters ### 
# In[35]:


y_pred_rf = best_rf_model.predict(X_val)

r2_rf = r2_score(y_val, y_pred_rf)
print('R^2:', r2_rf)


# # Random Search in XGBoost Regresssor

# In[36]:


param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

xgb_random_search = RandomizedSearchCV(estimator = xgb_regressor, param_distributions=param_dist, n_iter = 10,cv=5, verbose =2, random_state = 42)
xgb_random_search.fit(X_train, y_train)

print("Best Parameters:", xgb_random_search.best_params_)
print("Best Estimator:", xgb_random_search.best_estimator_)

best_xgb_model_rs = xgb_random_search.best_estimator_


# In[37]:


y_pred_xgb_rs = best_xgb_model_rs.predict(X_val)

r2_xgb_rs = r2_score(y_val, y_pred_xgb_rs)
print('R^2:', r2_xgb_rs)


# I applied GridSearchCV on Random Forest and RandomSearchCV on XGBoost and achieved better results with Random forest. Hence, I am using Random Forest trained model for predicting sales on test data.

# # Prediction on test data

# In[38]:


test_file_name = "test_mart_sales_data.csv"

test_df = pd.read_csv(test_file_name)
test_df.head(2)


# In[39]:


# Checking the count in all the columns
test_df.count()


# In[40]:


# Now check the summary of the numerical columns
test_df.describe()


# # Data Transformation

# In[41]:


# Convert the columns into lower case
test_df = test_df.rename(columns = str.lower)


# Replace NaN values in both numerical and categorical columns
# 

# In[42]:


# Replace NaN with 0 in numeric columns
cols_to_replace = ['item_weight', 'item_visibility', 'item_mrp', 'outlet_establishment_year']
test_df[cols_to_replace] = test_df[cols_to_replace].replace(np.nan, 0)

test_df = test_df.drop('outlet_establishment_year', axis = 1)


# In[43]:


# Replace NaN with None in categorical columns
categorical_cols = test_df.select_dtypes(include='object').columns

for col in categorical_cols:
    test_df[col] = test_df[col].fillna('none')


# Standardize the numerical columns by apply Min-max scaling

# In[44]:


scaler = MinMaxScaler()
scaler.fit(test_df[['item_weight', 'item_visibility', 'item_mrp']])
test_df[['item_weight', 'item_visibility', 'item_mrp']]= pd.DataFrame(scaler.transform(test_df[['item_weight', 'item_visibility', 'item_mrp']]))


# Compute the missing values
# 
# 1) item_weight: Fill by median per item_identifier
# 2) item_visibility: Replace 0s by mean

# In[45]:


test_df['item_weight'] = test_df.groupby('item_identifier')['item_weight'].transform(lambda x:x.fillna(x.median))

visibility_mean = test_df['item_visibility'].mean()
test_df['item_visibility'] = test_df['item_visibility'].replace(0, visibility_mean)


# # Feature Engineering

# In[46]:


# Normalize item_fat_content column
fat_map = {'LF': 'Low Fat', 'low fat' : 'Low Fat', 'reg' : 'Regular'}
test_df['item_fat_content'] = test_df['item_fat_content'].replace(fat_map)


# In[47]:


# Encode categorical feature

# Label encoding on outlet identifier
label_encode = LabelEncoder()
test_df['outlet'] = label_encode.fit_transform(test_df['outlet_identifier'])

# One-hot encoding
test_df = pd.get_dummies(test_df, columns = ['item_fat_content', 'outlet_size', 'outlet_location_type', 'outlet_type', 'outlet', 'item_type'])


# # Model Prediction

# In[48]:


# Copy test dataset
X_test_df = test_df.copy()


# In[49]:


# Drop columns that were not used during training
X_test_df = X_test_df.drop(['item_identifier', 'outlet_identifier'], axis = 1)


# In[50]:


# column order matches as per training set
X_test_df = X_test_df[X_train.columns]


# # Use random forest trained model

# In[51]:


predictions = best_rf_model.predict(X_test_df)


# In[52]:


submission = pd.DataFrame({'Item_Identifier' : test_df['item_identifier'], 'Outlet_Identifier': test_df['outlet_identifier'],
                          'Item_Outlet_Sales': predictions})


# In[53]:


submission.to_csv('submission.csv', index = False)

