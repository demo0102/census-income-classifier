## IN6227-2023-Assignment-1: Adult Income Classification

Minimal pipeline to predict whether income >50K using the UCI Adult dataset.

### Requirements

Python 3.9+ is recommended.

```bash
pip install pandas numpy scikit-learn
```

### Data

Place the following files under `data/`:

-   `adult.data` (train)
-   `adult.test` (test)
-   `adult.names` (description)

If any file is missing, the script will exit with an error message.

### How to Run

From the `code/` directory:

```bash
python 1.py
```

The script will preprocess data, train Logistic Regression and Random Forest (plus a tuned variant), and print a final evaluation table (Accuracy, Precision, Recall, F1-Score, Training Time).
