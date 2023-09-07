import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import pickle

df = pd.read_csv('D:\\NMIMS-Hiten\\Sem 3\\new_cleaned.csv')
# print(df.head())

df = df.drop(columns=['society','sector'])
df['floorNum'] = df['floorNum'].astype(float)
def categorize_floor(floor):
    if 0 <= floor <= 2:
        return "Low Floor"
    elif 3 <= floor <= 10:
        return "Mid Floor"
    elif 11 <= floor <= 51:
        return "High Floor"
    else:
        return None

df['floor_category'] = df['floorNum'].apply(categorize_floor)

df = df.drop(columns=['floorNum'])

df = df.rename(columns={'study room':'study_room',
                        'servant room':'servant_room',
                        'store room':'store_room',
                        'pooja room':'pooja_room'
                        })

data_label_encoded = df.copy()

categorical_cols = df.select_dtypes(include=['object']).columns

# Apply label encoding to categorical columns
for col in categorical_cols:
    oe = OrdinalEncoder()
    data_label_encoded[col] = oe.fit_transform(data_label_encoded[[col]])
    print(oe.categories_)

X = data_label_encoded.drop('price', axis=1)
y = data_label_encoded['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train,y_train)
y_pred3 = rf.predict(X_test)
print(f"R2 RandomForest: {r2_score(y_test,y_pred3)}")
print(f"MSE RandomForest: {mean_squared_error(y_test,y_pred3)}")
print(df.columns)
# 0.7276769808245319
# 0.9090659582849572

pickle_out = open("rf.pkl", "wb")
pickle.dump(rf, pickle_out)
pickle_out.close()