import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import os


# --- 1. Data Loading ---
def load_data(data_path="data"):
    """Loads the census dataset from a specified path."""
    COLUMNS = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]
    train_path = os.path.join(data_path, "adult.data")
    test_path = os.path.join(data_path, "adult.test")

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print(f"Error: Dataset files not found in the '{data_path}' directory.")
        print(
            "Please place 'adult.data' and 'adult.test' into the 'data' subdirectory."
        )
        exit()

    df_train = pd.read_csv(
        train_path, names=COLUMNS, sep=r",\s*", engine="python", na_values="?"
    )
    df_test = pd.read_csv(
        test_path,
        names=COLUMNS,
        sep=r",\s*",
        engine="python",
        na_values="?",
        skiprows=1,
    )

    return df_train, df_test


# --- 2. Data Preprocessing ---
def preprocess_data(df_train, df_test):
    """Cleans and preprocesses the data for modeling."""
    train_len = len(df_train)
    df_combined = pd.concat([df_train, df_test], ignore_index=True)

    print(f"Data loaded. Total combined rows: {len(df_combined)}")
    print("Overview of missing values before handling:")
    print(df_combined.isnull().sum())

    # Impute missing values using the mode
    for col in ["workclass", "occupation", "native-country"]:
        df_combined[col]=df_combined[col].fillna(df_combined[col].mode()[0])

    print("\nMissing values handled.")

    # Clean and encode the target variable
    df_combined["income"] = df_combined["income"].str.replace(r"\.", "", regex=True)
    df_combined["income"] = df_combined["income"].apply(
        lambda x: 1 if x == ">50K" else 0
    )

    # Separate features and target
    X = df_combined.drop("income", axis=1)
    y = df_combined["income"]

    # Define feature types and create a preprocessor
    categorical_features = X.select_dtypes(include=["object"]).columns
    numerical_features = X.select_dtypes(include=np.number).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)

    # Split back into training and testing sets
    X_train = X_processed[:train_len]
    X_test = X_processed[train_len:]
    y_train = y[:train_len]
    y_test = y[train_len:]

    return X_train, X_test, y_train, y_test


# --- 3. Main Execution Flow ---
if __name__ == "__main__":
    # Load and preprocess data
    df_train, df_test = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df_train, df_test)

    print("\nData preprocessing complete.")

    results = {}

    # --- Logistic Regression (Baseline Model) ---
    print("\nTraining Logistic Regression model...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    start_time = time.time()
    lr_model.fit(X_train, y_train)
    lr_train_time = time.time() - start_time

    y_pred_lr = lr_model.predict(X_test)
    results["Logistic Regression (Default)"] = {
        "Accuracy": accuracy_score(y_test, y_pred_lr),
        "Precision": precision_score(y_test, y_pred_lr),
        "Recall": recall_score(y_test, y_pred_lr),
        "F1-Score": f1_score(y_test, y_pred_lr),
        "Training Time (s)": lr_train_time,
    }
    print("Logistic Regression training complete.")

    # --- Random Forest (Default Parameters) ---
    print("\nTraining Random Forest model with default parameters...")
    rf_default_model = RandomForestClassifier(random_state=42)
    start_time = time.time()
    rf_default_model.fit(X_train, y_train)
    rf_default_train_time = time.time() - start_time

    y_pred_rf_default = rf_default_model.predict(X_test)
    results["Random Forest (Default)"] = {
        "Accuracy": accuracy_score(y_test, y_pred_rf_default),
        "Precision": precision_score(y_test, y_pred_rf_default),
        "Recall": recall_score(y_test, y_pred_rf_default),
        "F1-Score": f1_score(y_test, y_pred_rf_default),
        "Training Time (s)": rf_default_train_time,
    }
    print("Default Random Forest training complete.")

    # --- Random Forest (with Hyperparameter Tuning) ---
    print(
        "\nStarting hyperparameter tuning for Random Forest (this may take a few minutes)..."
    )
    rf = RandomForestClassifier(random_state=42)

    # Define the parameter grid to search
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_leaf": [1, 2],
    }

    # Set up GridSearchCV with F1-score as the evaluation metric
    grid_search = GridSearchCV(
        estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring="f1"
    )

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    rf_tuned_train_time = time.time() - start_time

    print("\nHyperparameter tuning complete.")
    print(f"Best parameters found: {grid_search.best_params_}")

    # Predict using the best model
    best_rf_model = grid_search.best_estimator_
    y_pred_rf_tuned = best_rf_model.predict(X_test)

    results["Random Forest (Tuned)"] = {
        "Accuracy": accuracy_score(y_test, y_pred_rf_tuned),
        "Precision": precision_score(y_test, y_pred_rf_tuned),
        "Recall": recall_score(y_test, y_pred_rf_tuned),
        "F1-Score": f1_score(y_test, y_pred_rf_tuned),
        "Training Time (s)": rf_tuned_train_time,
    }

    # --- 4. Display Results ---
    results_df = pd.DataFrame(results).T
    print("\n--- Final Model Evaluation Results ---")
    print(results_df.round(4))
