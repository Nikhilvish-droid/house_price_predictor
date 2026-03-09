# 1.importing libraries

import numpy as np
import pandas as pd
import os

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import joblib

# 2.laoding dataset using pandas
data = pd.read_csv("housing.csv")

# 3.creating a strata column to divide test and train data
data["income_cat"] = pd.cut(data["median_income"],
                            bins = [0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                            labels = [1, 2, 3, 4, 5]
                            )

# 4.using stratifiedshufflesplit form sklearn
spl = StratifiedShuffleSplit(n_splits= 1, test_size= 0.2, random_state = 42)

for train_index, test_index in spl.split(data, data["income_cat"]):
    train_data = data.loc[train_index]
    test_data = data.loc[test_index]

# 5. now deleting strata column as we dont need further
data.drop("income_cat", axis = 1, inplace = True)
train_data.drop("income_cat", axis = 1, inplace = True)
test_data.drop("income_cat", axis = 1, inplace = True)

# 6.now working with copy of train data 
housing_data = train_data.copy()

# 7.dividing data into feature and lables
feature = housing_data.iloc[:, :-1]  
labels  = housing_data["price"]

# 8.dividing feature into numerical data(list) and categorical data
num_col = list(feature.columns[:-1])  
cat_col = ["ocean_proximity"]

# 9.creating a numerical pipeline
num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler() ),
])

cat_pipeline = Pipeline([
    ("onehotencoding", OneHotEncoder(handle_unknown="ignore"))
])

# 10.creating a full pipeline
full_pipeline = ColumnTransformer([
    ("num_pipe", num_pipeline, num_col),
    ("cat_pipe", cat_pipeline, cat_col),
])

# 11.now fit_transform
feature_transformed = full_pipeline.fit_transform(feature)

# 12.using randomforest regressor as model
model = RandomForestRegressor(random_state=42)
model.fit(feature_transformed, labels)
print("model successfully trained")

# dumping model into file 
joblib.dump(model, "model.pkl")
print("model is saved as model.pkl")
joblib.dump(full_pipeline, "pipeline.pkl")
print("pipeline is saved in pipeline.pkl")

# 13.prediction using test
predictions = model.predict(feature_transformed)

housing_data["prediction"] = predictions
housing_data.to_csv("output.csv")

rmse = root_mean_squared_error(labels, predictions)

print(rmse)
