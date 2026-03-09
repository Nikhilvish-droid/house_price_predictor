# 🏡 California House Price Prediction

## 📌 Overview
This project is an end-to-end Machine Learning pipeline designed to predict housing prices in California based on various features such as location, median income, and total rooms. The project emphasizes robust data preprocessing, exploratory data analysis (EDA), and scalable model architecture using Scikit-Learn pipelines.

## 📊 Dataset
The dataset used is the classic **California Housing Dataset**, sourced from Kaggle. It contains geographical and demographic data for various housing districts.

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Manipulation:** NumPy, Pandas
* **Data Visualization:** Matplotlib
* **Machine Learning:** Scikit-Learn (`RandomForestRegressor`, `Pipeline`, `ColumnTransformer`)
* **Model Export:** Joblib

## 🚀 Workflow & Methodology

### 1. Exploratory Data Analysis (EDA)
* Visualized data distributions across all features using histograms to identify skewness and outliers.
* Created geographical scatter plots mapping longitude and latitude, using color maps (`cmap="jet"`) to identify high-density, high-price areas (like coastal regions).

### 2. Data Splitting
* Created an `income_cat` attribute to categorize median income.
* Used `StratifiedShuffleSplit` to divide the data into training and testing sets, ensuring that both sets maintained the same income distribution as the overall dataset.

### 3. Data Preprocessing (Scikit-Learn Pipelines)
To prevent data leakage and ensure clean code, a full preprocessing pipeline was built using `ColumnTransformer`:
* **Numerical Pipeline:** Used `SimpleImputer(strategy="median")` to handle missing values and `StandardScaler()` to standardize features with high variance.
* **Categorical Pipeline:** Used `OneHotEncoder(handle_unknown="ignore")` to convert the `ocean_proximity` text feature into binary vectors.

### 4. Model Training & Evaluation
* Fitted the preprocessed data into a `RandomForestRegressor`.
* Evaluated the model's performance on the test set using the **Root Mean Squared Error (RMSE)** metric.

### 5. Exporting Results
* Saved the trained model as `model.pkl` and the preprocessing pipeline as `pipeline.pkl` using `joblib` to allow for easy deployment in a web app or API.
* Exported the final predictions alongside the original prices to `output.csv` for side-by-side comparison.

## 📂 Project Structure
* `main.py`: The core script containing the EDA, pipeline construction, and model training.
* `housing.csv`: The raw dataset.
* `output.csv`: The final dataset appended with the model's predictions.
* `model.pkl` & `pipeline.pkl`: The exported, ready-to-deploy machine learning files.

## 🔮 Future Scope
* Experiment with advanced regression models like XGBoost or LightGBM.
* Perform hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV`.
* Deploy the model and pipeline using a Flask or FastAPI backend with a React frontend.
