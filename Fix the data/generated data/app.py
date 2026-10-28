"""Employee Attrition Streamlit demo: focused inputs and Pearson heatmap.

Run: py -m streamlit run app.py
Place the trained .pkl/.joblib model and improved balanced CSV next to this file.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


st.set_page_config(page_title="Employee Attrition Prediction", page_icon="📊", layout="wide")
st.title("Employee Attrition Prediction")
st.write("Enter employee details to explore the model's Leave / Stay prediction.")

BASE_DIR = Path(__file__).resolve().parent
# Support both names used during our notebook experiments.
MODEL_PATH = next(
    (p for p in [BASE_DIR / "employee_attrition_model.pkl",
                 BASE_DIR / "employee_attrition_model.joblib"] if p.exists()),
    None,
)
DATA_PATH = BASE_DIR / "HR_Employee_Attrition_Balanced_10000_Improved.csv"


@st.cache_resource
def load_model():
    # Load ONLY the trusted Pipeline saved by your training notebook.
    return joblib.load(MODEL_PATH)


if MODEL_PATH is None:
    st.error("Model not found: employee_attrition_model.pkl / .joblib")
    st.info("Save best_pipeline and place the model file next to app.py.")
    st.stop()

try:
    model = load_model()
    processor = model.named_steps["preprocessor"]
    feature_order = list(processor.feature_names_in_)

    # Read the fitted ColumnTransformer so the app matches the model exactly.
    numerical_features = list(next(cols for name, _, cols in processor.transformers_ if name == "num"))
    categorical_features = list(next(cols for name, _, cols in processor.transformers_ if name == "cat"))
    encoder = processor.named_transformers_["cat"]
    category_options = dict(zip(categorical_features, encoder.categories_))
except Exception as exc:
    st.error(f"Could not load the trained Pipeline: {exc}")
    st.stop()


def fitted_feature_importance(pipe, raw_features):
    """Sum Gradient Boosting feature_importances_ by original field.

    This is model-specific importance, not Pearson correlation, causality,
    or a statement that a weak field has no possible predictive signal.
    """
    estimator = pipe.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return None
    processor = pipe.named_steps["preprocessor"]
    names = processor.get_feature_names_out()
    importance = np.asarray(estimator.feature_importances_)
    if len(names) != len(importance):
        return None
    categories = sorted(categorical_features, key=len, reverse=True)
    result = {name: 0.0 for name in raw_features}
    for transformed_name, value in zip(names, importance):
        if transformed_name.startswith("num__"):
            original = transformed_name[len("num__"):]
        elif transformed_name.startswith("cat__"):
            suffix = transformed_name[len("cat__"):]
            original = next((col for col in categories if suffix.startswith(col + "_")), None)
        else:
            original = None
        if original not in result:
            return None
        result[original] += float(value)
    return pd.Series(result, name="Model importance").sort_values(ascending=False)


def load_attrition_correlations(path, modified_time):
    """Correlation for exploration only: numeric ratings as numbers, text one-hot."""
    data = pd.read_csv(path)
    required = "Attrition"
    if required not in data.columns:
        raise ValueError("The CSV is missing the Attrition column.")
    labels = data["Attrition"].map({"No": 0, "Yes": 1})
    if labels.isna().any():
        raise ValueError("Attrition must contain Yes and No values.")
    fields = [name for name in feature_order if name in data.columns]
    encoded = pd.get_dummies(data[fields], drop_first=True, dtype=int)
    encoded["Attrition"] = labels
    corr = encoded.corr().loc[:, "Attrition"].drop("Attrition").dropna()
    all_corr = encoded.corr()
    return corr, all_corr


# A model can use a feature with low Pearson correlation (nonlinear/interactions).
# For this reason use model importance, not the heatmap, to simplify inputs.
importance = fitted_feature_importance(model, feature_order)

# Default to the 0.5% feature-importance cutoff; no slider in the UI.
MIN_IMPORTANCE = 0.005
if importance is None:
    st.warning("This saved model has no supported feature importance; all inputs will stay visible.")

prediction_tab, analysis_tab = st.tabs(["Prediction", "Heatmap & Feature Importance"])

# Default values are only example inputs, not real employee records.
NUMERIC_DEFAULTS = {
    "Age": 32,
    "DailyRate": 800,
    "DistanceFromHome": 10,
    "HourlyRate": 65,
    "MonthlyIncome": 5000,
    "MonthlyRate": 15000,
    "NumCompaniesWorked": 2,
    "PercentSalaryHike": 14,
    "StockOptionLevel": 1,
    "TotalWorkingYears": 10,
    "TrainingTimesLastYear": 3,
    "YearsAtCompany": 5,
    "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 2,
    "YearsWithCurrManager": 3,
}

CATEGORY_DEFAULTS = {
    "BusinessTravel": "Travel_Rarely",
    "Department": "Research & Development",
    "EducationField": "Life Sciences",
    "Gender": "Female",
    "JobRole": "Research Scientist",
    "MaritalStatus": "Married",
    "OverTime": "No",
    "Education": 3,
    "EnvironmentSatisfaction": 3,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobSatisfaction": 3,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 3,
    "WorkLifeBalance": 3,
}

# Display-friendly labels for ordinal values. The returned widget value remains
# the original number, matching how the fitted model was trained.
VALUE_LABELS = {
    "Education": {
        1: "1 — Below College", 2: "2 — College", 3: "3 — Bachelor's",
        4: "4 — Master's", 5: "5 — Doctorate"
    },
    "EnvironmentSatisfaction": {
        1: "1 — Low", 2: "2 — Medium", 3: "3 — High", 4: "4 — Very High"
    },
    "JobInvolvement": {
        1: "1 — Low", 2: "2 — Medium", 3: "3 — High", 4: "4 — Very High"
    },
    "JobLevel": {
        1: "1 — Entry Level", 2: "2 — Junior Level", 3: "3 — Mid Level",
        4: "4 — Senior Level", 5: "5 — Executive Level"
    },
    "JobSatisfaction": {
        1: "1 — Low", 2: "2 — Medium", 3: "3 — High", 4: "4 — Very High"
    },
    "PerformanceRating": {
        1: "1 — Low", 2: "2 — Good", 3: "3 — Excellent", 4: "4 — Outstanding"
    },
    "RelationshipSatisfaction": {
        1: "1 — Low", 2: "2 — Medium", 3: "3 — High", 4: "4 — Very High"
    },
    "WorkLifeBalance": {
        1: "1 — Bad", 2: "2 — Good", 3: "3 — Better", 4: "4 — Best"
    },
    "StockOptionLevel": {
        0: "0 — No stock options", 1: "1 — Level 1", 2: "2 — Level 2", 3: "3 — Level 3"
    },
    "OverTime": {"Yes": "Yes — Works overtime", "No": "No — No overtime"},
    "BusinessTravel": {
        "Non-Travel": "No work travel", "Travel_Rarely": "Rarely travels",
        "Travel_Frequently": "Frequently travels"
    },
}

# Explanations clarify similarly named data fields without changing the model.
FIELD_HELP = {
    "Education": "Employee's highest educational level, coded from 1 to 5.",
    "JobLevel": "Employee's level within the organization, coded from 1 to 5.",
    "JobSatisfaction": "Employee's rating of overall job satisfaction.",
    "EnvironmentSatisfaction": "Satisfaction with the working environment.",
    "RelationshipSatisfaction": "Satisfaction with workplace relationships.",
    "JobInvolvement": "Level of engagement and involvement with work; not a performance score.",
    "PerformanceRating": "Job performance rating; use a rating represented in the training data.",
    "WorkLifeBalance": "Employee-reported work-life balance rating.",
    "StockOptionLevel": "Level of employee stock option benefits (0 to 3).",
    "MonthlyIncome": "Monthly employee income in the sample dataset.",
    "DailyRate": "Dataset's daily rate variable; it is not necessarily a daily salary.",
    "HourlyRate": "Dataset's hourly rate variable; it is not necessarily an hourly salary.",
    "MonthlyRate": "Dataset's monthly rate variable; it is not necessarily monthly income.",
    "PercentSalaryHike": "Percentage increase in the employee's salary.",
    "YearsSinceLastPromotion": "Number of years since the last promotion.",
    "YearsWithCurrManager": "Years working with the current manager.",
    "YearsInCurrentRole": "Years spent in the current job role.",
    "NumCompaniesWorked": "Number of companies the employee has worked for.",
    "OverTime": "Whether the employee works overtime.",
    "DistanceFromHome": "Distance from home as recorded in the dataset.",
}


def readable_value(column, value):
    """Human-readable values for the interface; preserve underlying values."""
    label = VALUE_LABELS.get(column, {}).get(value)
    if label is not None:
        return label
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return str(int(number)) if number.is_integer() else f"{number:.2f}"
    return str(value).replace("_", " ")


# Group the inputs for a cleaner presentation; the fitted Pipeline still gets every feature.
GROUPS = {
    "Personal details": [
        "Age", "Gender", "MaritalStatus", "Education", "EducationField", "DistanceFromHome"
    ],
    "Job and experience": [
        "Department", "JobRole", "BusinessTravel", "OverTime", "JobLevel",
        "JobInvolvement", "TotalWorkingYears", "YearsAtCompany",
        "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
        "NumCompaniesWorked"
    ],
    "Pay and satisfaction": [
        "MonthlyIncome", "PercentSalaryHike", "StockOptionLevel",
        "JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction",
        "WorkLifeBalance", "PerformanceRating", "TrainingTimesLastYear"
    ],
}

# Model also uses DailyRate, HourlyRate and MonthlyRate: keep these editable.
grouped = {c for cols in GROUPS.values() for c in cols}
other_features = [c for c in feature_order if c not in grouped]


def field_label(column):
    """Make technical column names readable without changing feature keys."""
    import re
    return re.sub(r"(?<!^)(?=[A-Z])", " ", column).replace("  ", " ")


def input_for(column):
    """Create readable inputs while retaining the training feature names and types."""
    # Text categories come from the fitted OneHotEncoder.
    if column in category_options:
        options = list(category_options[column])
        preferred = CATEGORY_DEFAULTS.get(column, options[0])
        initial = next((i for i, value in enumerate(options) if value == preferred), 0)
        return st.selectbox(
            field_label(column),
            options=options,
            index=initial,
            format_func=lambda value: readable_value(column, value),
            help=FIELD_HELP.get(column),
            key=column,
        )

    # The NEW Pipeline treats ordinal rating fields as numeric, not one-hot.
    # Show valid labeled choices while still passing the original integer values.
    if column in numerical_features and column in VALUE_LABELS:
        labels = VALUE_LABELS[column]
        if all(isinstance(value, int) for value in labels):
            options = list(labels.keys())
            if column == "PerformanceRating":
                options = [3, 4]  # Original IBM HR categories; adjust if training data differs.
            preferred = CATEGORY_DEFAULTS.get(column, NUMERIC_DEFAULTS.get(column, options[0]))
            initial = options.index(preferred) if preferred in options else 0
            return st.selectbox(
                field_label(column),
                options=options,
                index=initial,
                format_func=lambda value: readable_value(column, value),
                help=FIELD_HELP.get(column),
                key=column,
            )

    if column in numerical_features:
        default = int(NUMERIC_DEFAULTS.get(column, 0))
        return st.number_input(
            field_label(column),
            min_value=18 if column == "Age" else 0,
            value=max(18 if column == "Age" else 0, default),
            step=1,
            help=FIELD_HELP.get(column),
            key=column,
        )

    raise ValueError(f"Unexpected feature in model: {column}")


def default_for(column):
    """Hidden features remain valid model inputs (a reference employee)."""
    if column in category_options:
        options = category_options[column]
        preferred = CATEGORY_DEFAULTS.get(column, options[0])
        return preferred if preferred in options else options[0]
    if column in numerical_features:
        return CATEGORY_DEFAULTS.get(column, NUMERIC_DEFAULTS.get(column, 0))
    raise ValueError(f"No valid default available for {column}")


if importance is None:
    visible_fields = set(feature_order)
else:
    visible_fields = set(importance[importance >= MIN_IMPORTANCE].index)

hidden_fields = [name for name in feature_order if name not in visible_fields]
with prediction_tab:
    show_all = st.checkbox("Show all model inputs", value=False)
    if show_all:
        visible_fields = set(feature_order)
        hidden_fields = []
    st.caption(
        f"Showing {len(visible_fields)} of {len(feature_order)} inputs. "
        f"Hidden inputs receive fixed example values; they are NOT removed from the model. "
        f"Ratings send their original numeric codes."
    )
    if hidden_fields:
        with st.expander(f"View {len(hidden_fields)} hidden inputs and their fixed values"):
            st.dataframe(pd.DataFrame([
                {"Feature": field_label(name), "Default value": readable_value(name, default_for(name))}
                for name in hidden_fields
            ]), hide_index=True, use_container_width=True)
    with st.form("employee_prediction_form"):
        values = {name: default_for(name) for name in feature_order}
        for section_title, columns in GROUPS.items():
            present = [column for column in columns if column in visible_fields]
            if not present:
                continue
            with st.expander(section_title, expanded=True):
                left, right = st.columns(2)
                for idx, column in enumerate(present):
                    with (left if idx % 2 == 0 else right):
                        values[column] = input_for(column)

        remaining = [c for c in other_features if c in visible_fields]
        if remaining:
            with st.expander("Additional model inputs", expanded=False):
                left, right = st.columns(2)
                for idx, column in enumerate(remaining):
                    with (left if idx % 2 == 0 else right):
                        values[column] = input_for(column)

        submitted = st.form_submit_button("Predict Attrition", type="primary")

with analysis_tab:
    st.subheader("Feature Importance — trained model")
    st.caption(
        "Gradient Boosting importance shows which variables the fitted model uses most. "
        "For categorical variables, the one-hot values are combined back into one field. "
        "Values are relative and are not causal effects."
    )
    if importance is not None:
        ranked = (importance * 100).rename("Importance (%)")
        st.bar_chart(ranked.head(20), horizontal=True)
        table = ranked.to_frame()
        table["Shown by default"] = [name in visible_fields for name in table.index]
        st.dataframe(table.round(3), use_container_width=True)
        st.caption(f"Fields under {MIN_IMPORTANCE:.1%} model importance are hidden by default. Use Show all model inputs to reveal them.")

    st.subheader("Pearson Correlation Heatmap — dataset")
    st.caption(
        "Pearson correlation is a linear association, not the same as model importance. "
        "Text categories are one-hot encoded; ordinal ratings remain numeric."
    )
    if DATA_PATH.exists():
        try:
            corr_with_target, full_corr = load_attrition_correlations(
                DATA_PATH, DATA_PATH.stat().st_mtime_ns
            )
            strongest = corr_with_target.abs().nlargest(16).index.tolist()
            plotted = strongest + ["Attrition"]
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(
                full_corr.loc[plotted, plotted], annot=True, fmt=".2f",
                cmap="coolwarm", center=0, vmin=-1, vmax=1,
                annot_kws={"size": 7}, ax=ax
            )
            ax.set_title("Top 16 Pearson Correlations with Attrition")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Correlations here describe the synthetic balanced dataset, not real employee outcomes.")
        except Exception as exc:
            st.warning(f"Could not draw the heatmap: {exc}")
    else:
        st.info(
            "To display the heatmap, place "
            "HR_Employee_Attrition_Balanced_10000_Improved.csv next to app.py. "
            "The prediction form will still work without the CSV."
        )


def feature_contributions(model, employee):
    """Local one-feature-at-a-time sensitivity for any binary predict_proba model.

    For each feature, replace only its value with a reference value while
    keeping all other inputs unchanged. The difference in predicted Leave
    score is expressed in percentage points. These effects are NOT additive
    and are NOT causal explanations or SHAP values.
    """
    processor = model.named_steps["preprocessor"]
    scaler = processor.named_transformers_["num"]
    encoder = processor.named_transformers_["cat"]
    numeric_columns = list(next(cols for name, _, cols in processor.transformers_ if name == "num"))
    categorical_columns = list(next(cols for name, _, cols in processor.transformers_ if name == "cat"))
    leave_index = list(model.classes_).index(1)

    # Fitted means for continuous features; valid defaults for ordinal ratings.
    baselines = {}
    for column, mean in zip(numeric_columns, scaler.mean_):
        preferred = CATEGORY_DEFAULTS.get(column, NUMERIC_DEFAULTS.get(column))
        if column in VALUE_LABELS and preferred is not None:
            baselines[column] = int(preferred)
        else:
            baselines[column] = float(mean)

    # Fitted categories ensure we never send an unknown text label.
    for column, categories in zip(categorical_columns, encoder.categories_):
        preferred = CATEGORY_DEFAULTS.get(column)
        baselines[column] = preferred if preferred in categories else categories[0]

    actual_score = float(model.predict_proba(employee)[0, leave_index])

    # Evaluate all one-feature substitutions in one batch.
    alternatives = pd.concat([employee.copy() for _ in employee.columns], ignore_index=True)
    # Numeric fitted means may be fractional, so allow float replacements safely.
    alternatives[numeric_columns] = alternatives[numeric_columns].astype(float)
    for i, column in enumerate(employee.columns):
        alternatives.at[i, column] = baselines[column]
    alternate_scores = model.predict_proba(alternatives)[:, leave_index]

    return pd.DataFrame([
        {
            "Feature": field_label(column),
            "Your value": readable_value(column, employee.iloc[0][column]),
            "Reference value": readable_value(column, baselines[column]),
            "Impact": round((actual_score - float(alternate_scores[i])) * 100, 2),
        }
        for i, column in enumerate(employee.columns)
    ])


with prediction_tab:
    if submitted:
        # Use the original feature names and order; one row represents one employee.
        employee = pd.DataFrame([values], columns=feature_order)

        try:
            prediction = int(model.predict(employee)[0])
            classes = list(model.classes_)
            leave_index = classes.index(1)
            leave_score = float(model.predict_proba(employee)[0][leave_index])
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
        else:
            st.divider()
            st.subheader("Prediction result")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted class", "Leave" if prediction == 1 else "Stay")
            with col2:
                st.metric("Model score for Leave", f"{leave_score:.1%}")
            st.progress(min(max(leave_score, 0.0), 1.0))
            st.caption(
                "This is a model-generated score, not a verified real-world chance of resignation. "
                "The model was trained on an artificially balanced synthetic dataset, "
                "so this percentage is NOT a validated real-world resignation probability. "
                "For learning/demo purposes only."
            )

            st.subheader("Why did the model predict this?")
            st.write(
                "This is a local sensitivity check: for each feature, we replace only that "
                "value with a reference value and measure the change in the model's "
                "Leave score. Positive bars mean the current value raises the score "
                "relative to that replacement; negative bars mean it lowers it."
            )
            try:
                impacts = feature_contributions(model, employee)
                changed = impacts.loc[impacts["Impact"].abs() > 0.00001].copy()
                if changed.empty:
                    st.info("Changing individual fields to reference values did not affect this prediction.")
                else:
                    top = changed.reindex(
                        changed["Impact"].abs().sort_values(ascending=False).index
                    ).head(10)
                    chart = top.sort_values("Impact")

                    # Percentage-point change in Leave score when using the current value.
                    st.bar_chart(
                        chart, x="Feature", y="Impact", horizontal=True,
                        sort=False,
                    )

                    toward_leave = top[top["Impact"] > 0]
                    toward_stay = top[top["Impact"] < 0]
                    left, right = st.columns(2)
                    with left:
                        st.markdown("**Raises Leave score vs. reference**")
                        if toward_leave.empty:
                            st.write("No large positive contributions among top factors.")
                        else:
                            for _, item in toward_leave.head(4).iterrows():
                                st.write(f"• {item['Feature']}: {item['Your value']}")
                    with right:
                        st.markdown("**Lowers Leave score vs. reference**")
                        if toward_stay.empty:
                            st.write("No large negative contributions among top factors.")
                        else:
                            for _, item in toward_stay.head(4).iterrows():
                                st.write(f"• {item['Feature']}: {item['Your value']}")

                with st.expander("See how this explanation was calculated"):
                    st.write(
                        "Works with Gradient Boosting and other classifiers that have "
                        "predict_proba(). For each input, the app predicts again after "
                        "replacing only that field with a reference value, keeping every "
                        "other input unchanged. The Impact column is the difference in "
                        "Leave score, expressed in percentage points. Continuous "
                        "references use fitted training averages; ratings and categories "
                        "use valid example values."
                    )
                    st.dataframe(
                        impacts.sort_values("Impact", ascending=False),
                        use_container_width=True, hide_index=True,
                    )
                    st.caption(
                        "One-at-a-time effects are model sensitivity comparisons, not "
                        "additive feature contributions; they may interact and cannot "
                        "be summed. They are not SHAP values or proof of causation. "
                        "Changing a single feature can produce unrealistic combinations. "
                        "Do not use synthetic-model outputs for real employment decisions."
                    )
            except Exception as exc:
                st.warning(f"The prediction succeeded, but the explanation failed: {exc}")

# Run from the directory containing app.py:
# py -m streamlit run app.py
