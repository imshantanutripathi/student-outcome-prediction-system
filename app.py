
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

# 1. Application configuration

st.set_page_config(
    page_title="Student Outcome Prediction",
    page_icon="🎓",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "saved_model" / "student_outcome_final_model.joblib"
)

FEATURES_PATH = (
    BASE_DIR / "saved_model" / "selected_features.json"
)

DATA_PATH = BASE_DIR / "student_outcome_dataset.csv"


# 2. Check required files

required_paths = [
    MODEL_PATH,
    FEATURES_PATH,
    DATA_PATH
]

missing_paths = [
    str(path) for path in required_paths
    if not path.exists()
]

if missing_paths:
    st.error("Some required project files are missing.")
    st.write(missing_paths)
    st.info(
        "Keep app.py beside student_outcome_dataset.csv "
        "and the saved_model folder."
    )
    st.stop()

# 3. Load model, features and dataset


@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)


model = load_model(str(MODEL_PATH))
student_data = load_dataset(str(DATA_PATH))

with open(FEATURES_PATH, "r", encoding="utf-8") as file:
    selected_features = json.load(file)


# Validate that the feature list matches the dataset
missing_columns = [
    column for column in selected_features
    if column not in student_data.columns
]

if missing_columns:
    st.error(
        f"Required dataset columns are missing: {missing_columns}"
    )
    st.stop()

# 4. Helper functions for input ranges

def integer_bounds(column):
    values = pd.to_numeric(
        student_data[column],
        errors="coerce"
    ).dropna()

    return (
        int(values.min()),
        int(values.max()),
        int(round(values.median()))
    )


# 5. Application heading


st.title("🎓 Student Outcome Prediction System")

st.write(
    "Enter student information to predict a likely academic "
    "outcome using the trained machine learning model."
)

st.caption(
    "This model uses the selected demographic and first-semester "
    "academic features. Predictions are estimates, not guarantees."
)

st.divider()


# 6. Student input form

qualification_options = sorted(
    student_data["Previous qualification"]
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)

if not qualification_options:
    st.error("No previous qualification categories were found.")
    st.stop()


age_min, age_max, age_default = integer_bounds(
    "Age at enrollment"
)

enrolled_min, enrolled_max, enrolled_default = integer_bounds(
    "Curricular units 1st sem (enrolled)"
)

approved_min, approved_max, approved_default = integer_bounds(
    "Curricular units 1st sem (approved)"
)

evaluations_min, evaluations_max, evaluations_default = integer_bounds(
    "Curricular units 1st sem (evaluations)"
)

grade_min = float(
    student_data["Curricular units 1st sem (grade)"].min()
)

grade_max = float(
    student_data["Curricular units 1st sem (grade)"].max()
)

grade_default = float(
    student_data["Curricular units 1st sem (grade)"].median()
)


with st.form("student_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Student Information")

        previous_qualification = st.selectbox(
            "Previous qualification (dataset code)",
            options=qualification_options
        )

        age = st.number_input(
            "Age at enrollment",
            min_value=age_min,
            max_value=age_max,
            value=age_default,
            step=1
        )

        tuition_status = st.selectbox(
            "Tuition fees up to date",
            options=[1, 0],
            format_func=lambda value: (
                "Yes" if value == 1 else "No"
            )
        )

        debtor_status = st.selectbox(
            "Debtor",
            options=[0, 1],
            format_func=lambda value: (
                "No" if value == 0 else "Yes"
            )
        )

        scholarship_status = st.selectbox(
            "Scholarship holder",
            options=[0, 1],
            format_func=lambda value: (
                "No" if value == 0 else "Yes"
            )
        )

    with col2:
        st.subheader("First-Semester Performance")

        enrolled_units = st.number_input(
            "Curricular units enrolled",
            min_value=enrolled_min,
            max_value=enrolled_max,
            value=enrolled_default,
            step=1
        )

        approved_units = st.number_input(
            "Curricular units approved",
            min_value=approved_min,
            max_value=approved_max,
            value=approved_default,
            step=1
        )

        evaluations = st.number_input(
            "Curricular units evaluations",
            min_value=evaluations_min,
            max_value=evaluations_max,
            value=evaluations_default,
            step=1
        )

        first_semester_grade = st.number_input(
            "First-semester grade",
            min_value=grade_min,
            max_value=grade_max,
            value=grade_default,
            step=0.1,
            format="%.2f"
        )

    submitted = st.form_submit_button(
        "Predict Student Outcome",
        use_container_width=True
    )


# 7. Run prediction

if submitted:

    if approved_units > enrolled_units:
        st.warning(
            "Approved units cannot exceed enrolled units. "
            "Please check these inputs."
        )

    else:
        input_values = {
            "Previous qualification": previous_qualification,
            "Age at enrollment": age,
            "Tuition fees up to date": tuition_status,
            "Debtor": debtor_status,
            "Scholarship holder": scholarship_status,
            "Curricular units 1st sem (enrolled)": enrolled_units,
            "Curricular units 1st sem (approved)": approved_units,
            "Curricular units 1st sem (evaluations)": evaluations,
            "Curricular units 1st sem (grade)": first_semester_grade
        }

        input_df = pd.DataFrame(
            [input_values],
            columns=selected_features
        )

        try:
            prediction = model.predict(input_df)[0]

            st.divider()
            st.subheader("Prediction Result")

            st.metric(
                label="Predicted Student Outcome",
                value=str(prediction)
            )

            st.caption(
                "The result is a model prediction, not a confirmed "
                "future outcome or a causal assessment."
            )

            # Display estimated class probabilities
            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(input_df)[0]

                classifier = model.named_steps["classifier"]
                classes = classifier.classes_

                probability_df = pd.DataFrame({
                    "Outcome": classes,
                    "Estimated probability (%)": probabilities * 100
                })

                probability_df = probability_df.sort_values(
                    "Estimated probability (%)",
                    ascending=False
                )

                st.subheader("Estimated Class Probabilities")

                display_df = probability_df.copy()
                display_df["Estimated probability (%)"] = (
                    display_df["Estimated probability (%)"].round(2)
                )

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.bar_chart(
                    probability_df.set_index("Outcome")[
                        "Estimated probability (%)"
                    ]
                )

        except Exception as error:
            st.error(f"Prediction failed: {error}")
