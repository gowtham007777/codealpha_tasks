# ==============================================================
# CAR PRICE PREDICTION USING MACHINE LEARNING
# Dataset : car_data.csv
# ==============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

plt.style.use("ggplot")

print("="*80)
print("CAR PRICE PREDICTION PROJECT")
print("="*80)

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv("car data.csv")

# Remove spaces from column names
df.columns = df.columns.str.strip()

print("\nDataset Loaded Successfully!")

# ==========================================================
# BASIC INFORMATION
# ==========================================================

print("\nFirst Five Rows\n")
print(df.head())

print("\nDataset Shape :", df.shape)

print("\nColumns\n")
print(df.columns.tolist())

print("\nInformation\n")
print(df.info())

print("\nMissing Values\n")
print(df.isnull().sum())

print("\nDuplicate Rows :", df.duplicated().sum())

df.drop_duplicates(inplace=True)

df.ffill(inplace=True)

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

CURRENT_YEAR = 2025

df["Car_Age"] = CURRENT_YEAR - df["Year"]

# Remove Car Name

df.drop("Car_Name", axis=1, inplace=True)

print("\nNew Dataset Shape :", df.shape)

# ==========================================================
# DESCRIPTIVE STATISTICS
# ==========================================================

print("\nStatistical Summary\n")

print(df.describe())

# ==========================================================
# DATA VISUALIZATION
# ==========================================================

# Selling Price Distribution

plt.figure(figsize=(8,5))

sns.histplot(df["Selling_Price"], bins=25, kde=True)

plt.title("Selling Price Distribution")

plt.show()

# Present Price Distribution

plt.figure(figsize=(8,5))

sns.histplot(df["Present_Price"], bins=25, kde=True)

plt.title("Present Price Distribution")

plt.show()

# Fuel Type

plt.figure(figsize=(6,5))

sns.countplot(data=df, x="Fuel_Type")

plt.title("Fuel Type Count")

plt.show()

# Transmission

plt.figure(figsize=(6,5))

sns.countplot(data=df, x="Transmission")

plt.title("Transmission")

plt.show()

# Seller Type

plt.figure(figsize=(6,5))

sns.countplot(data=df, x="Selling_type")

plt.title("Seller Type")

plt.show()

# Scatter Plot

plt.figure(figsize=(8,6))

sns.scatterplot(

    data=df,

    x="Present_Price",

    y="Selling_Price"

)

plt.title("Present Price vs Selling Price")

plt.show()

# Car Age

plt.figure(figsize=(8,6))

sns.scatterplot(

    data=df,

    x="Car_Age",

    y="Selling_Price"

)

plt.title("Car Age vs Selling Price")

plt.show()

# ==========================================================
# CORRELATION
# ==========================================================

corr = df.copy()

corr["Fuel_Type"] = corr["Fuel_Type"].astype("category").cat.codes

corr["Selling_type"] = corr["Selling_type"].astype("category").cat.codes

corr["Transmission"] = corr["Transmission"].astype("category").cat.codes

plt.figure(figsize=(10,8))

sns.heatmap(

    corr.corr(),

    annot=True,

    cmap="coolwarm"

)

plt.title("Correlation Matrix")

plt.show()

print("\nPART 1 COMPLETED")

# ==========================================================
# PART 2 : PREPROCESSING & TRAIN-TEST SPLIT
# ==========================================================

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# ==========================================================
# FEATURE MATRIX & TARGET
# ==========================================================

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

print("\nFeature Matrix Shape :", X.shape)
print("Target Shape :", y.shape)

# ==========================================================
# IDENTIFY FEATURE TYPES
# ==========================================================

categorical_features = [
    "Fuel_Type",
    "Selling_type",
    "Transmission"
]

numerical_features = [
    "Year",
    "Present_Price",
    "Driven_kms",
    "Owner",
    "Car_Age"
]

print("\nCategorical Features")
print(categorical_features)

print("\nNumerical Features")
print(numerical_features)

# ==========================================================
# PREPROCESSOR
# ==========================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",

            OneHotEncoder(

                handle_unknown="ignore",

                sparse_output=False

            ),

            categorical_features

        ),

        (
            "numerical",

            "passthrough",

            numerical_features

        )

    ]

)

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42

)

print("\n")
print("="*70)

print("TRAIN TEST SPLIT")

print("="*70)

print("Training Samples :", len(X_train))

print("Testing Samples :", len(X_test))

# ==========================================================
# VERIFY PREPROCESSOR
# ==========================================================

preprocessor.fit(X_train)

X_train_processed = preprocessor.transform(X_train)

X_test_processed = preprocessor.transform(X_test)

print("\nProcessed Training Shape :", X_train_processed.shape)

print("Processed Testing Shape :", X_test_processed.shape)

# ==========================================================
# FEATURE NAMES AFTER ENCODING
# ==========================================================

feature_names = preprocessor.get_feature_names_out()

print("\nNumber of Final Features :", len(feature_names))

print("\nFirst 15 Features")

for feature in feature_names[:15]:
    print(feature)

print("\nPART 2 COMPLETED SUCCESSFULLY")

# ==========================================================
# PART 3 : MODEL TRAINING
# ==========================================================

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np

# ==========================================================
# CREATE MODELS
# ==========================================================

models = {

    "Random Forest":
        RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1
        ),

    "Extra Trees":
        ExtraTreesRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42
        )

}

# ==========================================================
# TRAIN ALL MODELS
# ==============================================================

results = []

best_model = None
best_name = ""
best_r2 = -999

print("\n")
print("="*80)
print("TRAINING MODELS")
print("="*80)

for name, model in models.items():

    # Create Pipeline
    pipeline = Pipeline([

        ("preprocessor", preprocessor),

        ("model", model)

    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    prediction = pipeline.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, prediction)

    mse = mean_squared_error(y_test, prediction)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, prediction)

    results.append([
        name,
        mae,
        rmse,
        r2
    ])

    print(f"\n{name}")

    print("-"*40)

    print(f"MAE  : {mae:.4f}")

    print(f"RMSE : {rmse:.4f}")

    print(f"R2   : {r2:.4f}")

    if r2 > best_r2:

        best_r2 = r2

        best_model = pipeline

        best_name = name

# ==========================================================
# RESULTS TABLE
# ==============================================================

results_df = pd.DataFrame(

    results,

    columns=[

        "Model",

        "MAE",

        "RMSE",

        "R2 Score"

    ]

)

print("\n")
print("="*80)
print("MODEL COMPARISON")
print("="*80)

print(results_df)

# ==========================================================
# BEST MODEL
# ==============================================================

print("\n")
print("="*80)
print("BEST MODEL")
print("="*80)

print(f"Best Model : {best_name}")

print(f"Best R2 Score : {best_r2:.4f}")

# Save best model

trained_model = best_model

# ==========================================================
# FINAL PREDICTION
# ==============================================================

y_pred = trained_model.predict(X_test)

# Store metrics for Part 4

final_mae = mean_absolute_error(y_test, y_pred)

final_mse = mean_squared_error(y_test, y_pred)

final_rmse = np.sqrt(final_mse)

final_r2 = r2_score(y_test, y_pred)

print("\n")
print("="*80)
print("FINAL MODEL PERFORMANCE")
print("="*80)

print(f"Model : {best_name}")

print(f"MAE   : {final_mae:.4f}")

print(f"RMSE  : {final_rmse:.4f}")

print(f"R2    : {final_r2:.4f}")

print("\nPART 3 COMPLETED SUCCESSFULLY")
