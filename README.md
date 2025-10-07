## Adult Income Classification (Logistic Regression & Random Forest)

This repository contains a minimal machine learning pipeline to predict whether a person earns more than 50K using the UCI Adult (Census Income) dataset. It provides:

-   Data loading and preprocessing
-   A baseline Logistic Regression model
-   A Random Forest model (default and tuned via GridSearchCV)
-   A compact, reproducible script in `1.py`

### Dataset

The project uses the UCI Adult (Census Income) dataset. Files expected under the `data/` directory:

-   `adult.data` (training)
-   `adult.test` (testing; first line is a header comment in the original dataset)
-   `adult.names` (metadata/feature descriptions)

If any of the required files are missing, the script will exit and print an error explaining how to place them.

### Requirements

Python 3.9+ is recommended. Install dependencies:

```bash
pip install pandas numpy scikit-learn
```

### Project Structure

```
code/
  1.py                 # Main script (data prep, training, evaluation)
  data/
    adult.data         # Train split
    adult.test         # Test split
    adult.names        # Dataset description
    Index, old.adult.names (optional/original extras)
```

### How to Run

From the `code/` directory:

```bash
python 1.py
```

You should see logs for:

-   Data loading and preprocessing
-   Training Logistic Regression
-   Training Random Forest (default)
-   Hyperparameter tuning for Random Forest
-   Final evaluation table printed to stdout

### Output & Metrics

At the end, the script prints a summary table with:

-   Accuracy
-   Precision
-   Recall
-   F1-Score
-   Training Time (s)

Keys are in English to facilitate downstream processing/plotting.

### Notes

-   Preprocessing includes mode imputation for selected categorical features and one-hot encoding, with standardization applied to numeric features.
-   The `adult.test` file includes a trailing period in labels; the script strips it during target cleaning.
-   Random Forest hyperparameter tuning uses a small grid for speed; expand it if you need better performance.
