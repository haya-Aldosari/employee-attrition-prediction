"""Streamlit demo for the trained IBM employee attrition Pipeline.

Run: python -m streamlit run app.py
Place employee_attrition_model.joblib next to this file.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Employee Attrition Prediction", page_icon="📊", layout="wide")
st.title("Employee Attrition Prediction")
st.write("Enter employee details to explore the model's Leave / Stay prediction.")

MODEL_PATH = Path(__file__).resolve().parent / "employee_attrition_model.joblib"


@st.cache_resource
def load_model():
    # Load ONLY the trusted Pipeline saved by your training notebook.
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("Model file not found: employee_attrition_model.joblib")
    st.info("Run the last cell in model_training.ipynb, then put the saved .joblib file next to app.py.")
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
    "PerformanceRating": "Job performance rating; the IBM sample contains values 3 and 4.",
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
    """Create readable inputs while retaining exact model training values."""
    # StockOptionLevel is numerical in this pipeline. Still show a dropdown,
    # but send the selected integer to the model's StandardScaler.
    if column == "StockOptionLevel" and column in numerical_features:
        levels = [0, 1, 2, 3]
        return st.selectbox(
            field_label(column),
            options=levels,
            index=levels.index(int(NUMERIC_DEFAULTS["StockOptionLevel"])),
            format_func=lambda value: readable_value(column, value),
            help=FIELD_HELP.get(column),
            key=column,
        )

    # For ordinal columns one-hot encoded as categories, show words but return
    # the same integer categories as the fitted OneHotEncoder expects.
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

    if column in numerical_features:
        default = int(NUMERIC_DEFAULTS.get(column, 0))
        return st.number_input(
            field_label(column),
            min_value=0,
            value=max(0, default),
            step=1,
            help=FIELD_HELP.get(column),
            key=column,
        )

    raise ValueError(f"Unexpected feature in model: {column}")


st.caption("Example values are pre-filled. Ratings are shown as words, but the original numeric codes are sent to the model.")
with st.form("employee_prediction_form"):
    values = {}

    for section_title, columns in GROUPS.items():
        present = [column for column in columns if column in feature_order]
        if not present:
            continue
        with st.expander(section_title, expanded=(section_title == "Personal details")):
            left, right = st.columns(2)
            for idx, column in enumerate(present):
                with (left if idx % 2 == 0 else right):
                    values[column] = input_for(column)

    if other_features:
        with st.expander("Additional model inputs", expanded=False):
            left, right = st.columns(2)
            for idx, column in enumerate(other_features):
                with (left if idx % 2 == 0 else right):
                    values[column] = input_for(column)

    submitted = st.form_submit_button("Predict Attrition", type="primary")


def feature_contributions(model, employee):
    """Explain one prediction relative to a readable reference employee.

    For logistic regression, the change in log-odds is exactly additive:
    model_score(employee) - model_score(reference) = sum(feature contributions).
    This explains the MODEL's computation, not real-world causes.
    """
    processor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]

    if not hasattr(classifier, "coef_") or len(classifier.classes_) != 2:
        raise ValueError("Feature explanations require a binary linear model.")
    if list(classifier.classes_) != [0, 1]:
        raise ValueError("Expected Stay=0 and Leave=1 in the saved model.")

    # Create a reference using training means for numerical features and
    # valid example categories for categorical features.
    reference = employee.reset_index(drop=True).copy()
    scaler = processor.named_transformers_["num"]
    encoder = processor.named_transformers_["cat"]

    numeric_columns = list(next(cols for name, _, cols in processor.transformers_ if name == "num"))
    categorical_columns = list(next(cols for name, _, cols in processor.transformers_ if name == "cat"))

    # Means may be fractional even when a form field uses integer input.
    reference[numeric_columns] = reference[numeric_columns].astype(float)
    for column, average in zip(numeric_columns, scaler.mean_):
        reference.at[0, column] = float(average)

    for column, categories in zip(categorical_columns, encoder.categories_):
        preferred = CATEGORY_DEFAULTS.get(column)
        if preferred not in categories:
            preferred = categories[0]
        reference.at[0, column] = preferred

    # The pipeline applies the SAME fitted encoders and scaler to both rows.
    actual_encoded = processor.transform(employee)
    reference_encoded = processor.transform(reference)
    if hasattr(actual_encoded, "toarray"):
        actual_encoded = actual_encoded.toarray()
    if hasattr(reference_encoded, "toarray"):
        reference_encoded = reference_encoded.toarray()

    # Positive contribution increases Leave log-odds relative to the reference.
    differences = np.asarray(actual_encoded - reference_encoded).ravel()
    impact_by_encoded_feature = differences * np.asarray(classifier.coef_[0])
    encoded_names = processor.get_feature_names_out()

    # Merge one-hot encoded columns back into their original form fields.
    cat_names_sorted = sorted(categorical_columns, key=len, reverse=True)
    impacts = {column: 0.0 for column in employee.columns}
    for name, impact in zip(encoded_names, impact_by_encoded_feature):
        if name.startswith("num__"):
            original = name[len("num__"):]
        elif name.startswith("cat__"):
            expanded = name[len("cat__"):]
            original = next(
                (column for column in cat_names_sorted if expanded.startswith(column + "_")),
                None,
            )
        else:
            original = None
        if original not in impacts:
            raise ValueError(f"Unknown transformed feature: {name}")
        impacts[original] += float(impact)

    # Verify contributions reconstruct the prediction's log-odds.
    reference_logit = float(model.decision_function(reference)[0])
    actual_logit = float(model.decision_function(employee)[0])
    if not np.isclose(reference_logit + sum(impacts.values()), actual_logit, atol=1e-6):
        raise ValueError("Feature explanation did not match the saved model.")

    return pd.DataFrame([
        {
            "Feature": field_label(column),
            "Your value": readable_value(column, employee.iloc[0][column]),
            "Reference value": readable_value(column, reference.iloc[0][column]),
            "Impact": round(impact, 4),
        }
        for column, impact in impacts.items()
    ])


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
            "The model was trained on synthetic IBM HR data and uses class weighting; "
            "probabilities have not been calibrated. For learning/demo purposes only."
        )

        st.subheader("Why did the model predict this?")
        st.write(
            "The chart shows which input features push this model's score "
            "toward **Leave** or **Stay**, relative to a reference employee."
        )
        try:
            impacts = feature_contributions(model, employee)
            changed = impacts.loc[impacts["Impact"].abs() > 0.00001].copy()
            if changed.empty:
                st.info("The inputs match the reference values, so no individual differences are shown.")
            else:
                top = changed.reindex(
                    changed["Impact"].abs().sort_values(ascending=False).index
                ).head(10)
                chart = top.sort_values("Impact")

                # Bars right of zero increase Leave; bars left increase Stay.
                st.bar_chart(
                    chart, x="Feature", y="Impact", horizontal=True,
                    sort=False,
                )

                toward_leave = top[top["Impact"] > 0]
                toward_stay = top[top["Impact"] < 0]
                left, right = st.columns(2)
                with left:
                    st.markdown("**Pushes toward Leave**")
                    if toward_leave.empty:
                        st.write("No large positive contributions among top factors.")
                    else:
                        for _, item in toward_leave.head(4).iterrows():
                            st.write(f"• {item['Feature']}: {item['Your value']}")
                with right:
                    st.markdown("**Pushes toward Stay**")
                    if toward_stay.empty:
                        st.write("No large negative contributions among top factors.")
                    else:
                        for _, item in toward_stay.head(4).iterrows():
                            st.write(f"• {item['Feature']}: {item['Your value']}")

            with st.expander("See how this explanation was calculated"):
                st.write(
                    "This uses the fitted Logistic Regression coefficients. "
                    "For each feature, it computes the change in the model's "
                    "log-odds score compared with a reference input. "
                    "Numerical reference values are training averages; "
                    "categorical references are preselected example categories. "
                    "One-hot encoded parts are combined under the original feature."
                )
                st.dataframe(
                    impacts.sort_values("Impact", ascending=False),
                    use_container_width=True, hide_index=True,
                )
                st.caption(
                    "Contributions are changes in log-odds, NOT percentage points, "
                    "and are relative to the stated reference. They are not SHAP "
                    "values, causal effects, or proof of why someone would leave. "
                    "This synthetic-data demonstration must not be used for "
                    "real employment decisions."
                )
        except Exception as exc:
            st.warning(f"The prediction succeeded, but the explanation failed: {exc}")

# py -m streamlit run app.py
