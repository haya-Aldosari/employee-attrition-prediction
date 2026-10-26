# Employee Attrition Prediction

A machine learning classification project that explores employee attrition, compares classification algorithms, and provides an interactive **Streamlit** application for predicting **Stay** or **Leave** with a transparent, feature-level explanation.

> **Educational demonstration only.** The IBM HR dataset is fictional/synthetic. Predictions are not validated estimates of real employee resignation risk and should **not** be used for hiring, retention interventions, or other employment decisions.

## Project overview

- **Task:** Binary classification of employee attrition (`No` = Stay / `0`; `Yes` = Leave / `1`).
- **Dataset:** IBM HR Analytics Employee Attrition & Performance — 1,470 records, 35 original columns.
- **Challenges:** Imbalanced target (~84% Stay / ~16% Leave) and a mixture of numeric, categorical, and ordinal-coded attributes.
- **Final approach:** Logistic Regression with `class_weight` and tuned regularization, wrapped in a scikit-learn Pipeline.
- **Interface:** Streamlit form, model score, and per-feature local explanations.

## Workflow

1. **Data exploration** — examine column types, class balance, relationships with attrition, and potential data quality issues.
2. **Train/test split** — stratified 80/20 partition (`random_state=42`).
3. **Preprocessing** — scale numerical features with `StandardScaler`; one-hot encode categorical fields and ordinal-coded ratings/levels using `OneHotEncoder`. Fit these transformations **inside** the Pipeline to avoid leakage during cross-validation.
4. **Model comparison** — evaluate Logistic Regression, Decision Tree, Random Forest, KNN, SVM, and XGBoost with five-fold stratified cross-validation.
5. **Model tuning** — compare class weights and use `GridSearchCV` to select Logistic Regression's `C` and `class_weight`, optimizing Leave-class F1.
6. **Final evaluation** — train on the training set and evaluate predictions on the held-out test partition.
7. **Demo** — save the fitted preprocessing-and-model Pipeline with `joblib` and use it in Streamlit.

## Evaluation

The following results are from a local run of the final training notebook (294 test records):

| Metric | Value |
| --- | ---: |
| Accuracy | 0.84 |
| Leave precision | 0.50 |
| Leave recall | 0.49 |
| Leave F1-score | 0.49 |
| ROC-AUC | 0.835 |
| Average precision | 0.586 |

Leave-class confusion matrix: **23 true positives, 24 false negatives, 23 false positives**, with 224 true negatives.

These results illustrate the distinction between high overall accuracy and much weaker detection of the minority Leave class. The test partition was also inspected during earlier experiments, so these results should be treated as **exploratory**, not an unbiased external validation.

## Explainability

The Streamlit application explains a single prediction relative to a **reference employee**:

- Numerical reference values come from fitted training means.
- Categorical reference values are predefined example categories.
- Both examples pass through the **same fitted preprocessing**.
- For the fitted Logistic Regression model, the change in transformed feature values is multiplied by the learned coefficients; encoded categories are aggregated back into their original fields.

The resulting feature contributions add up to the **change in model log-odds** relative to the reference. They are **not SHAP values**, percentages, or causal explanations.

## Repository files

```text
data_exploration.ipynb          # Exploratory data analysis
model_training.ipynb           # Preprocessing, training, evaluation, and saving
app.py                         # Interactive Streamlit application
employee_attrition_model.joblib # Saved trained Pipeline for the app
requirements.txt               # Python dependencies
README.md                      # Project documentation
.gitignore                     # Local files not tracked by Git
```

## Run locally

**1. Install dependencies** (from the project folder):

```powershell
py -m pip install -r requirements.txt
```

**2. Launch the app:**

```powershell
py -m streamlit run app.py
```

The app loads `employee_attrition_model.joblib` from the same directory. Only load model files you trust; `joblib` deserialization can execute code. If the saved model is missing or incompatible with your installed scikit-learn version, retrain and save it from `model_training.ipynb`.

**3. Reproduce training (optional):** Download the dataset from [Kaggle: IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset). Rename the CSV to `HR-Analytics-Employee.csv` and place it next to `model_training.ipynb`, then run the notebook from top to bottom. The raw CSV is not required to launch the app if the saved fitted model is present.

## Limitations

- The dataset is **synthetic**, not real HR records.
- Model scores are **not calibrated probabilities** of real resignation.
- Explanations describe the model's computation, **not causes of attrition**.
- Demographic and other employment-related fields require particular care in any real application. This demo is not suitable for consequential personnel decisions.

## Dataset credit

[IBM HR Analytics Employee Attrition & Performance on Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) — fictional sample data attributed to IBM data scientists on the dataset page.
