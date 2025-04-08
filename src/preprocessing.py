from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np 
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile, chi2

"""
I used a builtin wine dataset for lab2, however I will still go through
the steps of dvc updating, and rename my new processed files so that
I understand the pipeline process.
"""

col_names = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price', 'x', 'y', 'z']

# Load data
diamonds_df = pd.read_csv('data/diamonds.csv')
train_data, test_data = train_test_split(diamonds_df, test_size=0.2, random_state=42)
train_y = train_data["price"]
test_y = test_data["price"]

# Drop target variable 
train_data = train_data.drop(columns="price")
test_data = test_data.drop(columns="price")

# Create pipeline for imputing and scaling numeric variables
# one-hot encoding categorical variables, and select features based on chi-squared value
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("selector", SelectPercentile(chi2, percentile=50)),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, make_column_selector(dtype_include = ['int', 'float'])),
        ("cat", categorical_transformer, make_column_selector(dtype_exclude = ['int', 'float'])),
    ]
)

clf = Pipeline(
    steps=[("preprocessor", preprocessor)]
)

# Create new train and test data using the pipeline
clf.fit(train_data, train_y)
train_new = clf.transform(train_data)
test_new = clf.transform(test_data)

# Transform to dataframe and save as a csv
train_new = pd.DataFrame(clf.transform(train_data))
test_new = pd.DataFrame(clf.transform(test_data))
train_new['price'] = train_y
test_new['price'] = test_y

train_new.to_csv('data/processed_diamond_train_data.csv')
test_new.to_csv('data/processed_diamond_test_data.csv')

# Save pipeline
with open('data/diamond_pipeline.pkl','wb') as f:
    pickle.dump(clf,f)

