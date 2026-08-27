import streamlit as st
import pandas as pd

from model import predict_disease


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MedBuddy AI",
    page_icon="🩺",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .subtitle {
            font-size: 18px;
            color: #666;
            margin-bottom: 25px;
        }

        .warning-box {
            padding: 15px;
            border-radius: 10px;
            background-color: #fff3cd;
            border: 1px solid #ffe69c;
            margin-top: 20px;
        }

        .result-box {
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #ddd;
            margin-top: 20px;
        }

        .confidence {
            font-size: 26px;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🩺 MedBuddy AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "AI-assisted symptom analysis and health information"
    "</div>",
    unsafe_allow_html=True,
)


st.info(
    "MedBuddy AI is an educational decision-support prototype. "
    "It does not replace a qualified healthcare professional."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Patient Information")

age = st.sidebar.number_input(
    "Age",
    min_value=0,
    max_value=120,
    value=25,
    step=1,
)

gender = st.sidebar.selectbox(
    "Gender",
    [
        "Male",
        "Female",
        "Other",
        "Prefer not to say",
    ],
)


# ============================================================
# SYMPTOM LIST
# ============================================================

SYMPTOMS = [
    "itching",
    "skin_rash",
    "nodal_skin_eruptions",
    "continuous_sneezing",
    "shivering",
    "chills",
    "joint_pain",
    "stomach_pain",
    "acidity",
    "ulcers_on_tongue",
    "muscle_wasting",
    "vomiting",
    "burning_micturition",
    "spotting_urination",
    "fatigue",
    "weight_gain",
    "anxiety",
    "cold_hands_and_feets",
    "mood_swings",
    "weight_loss",
    "restlessness",
    "lethargy",
    "patches_in_throat",
    "irregular_sugar_level",
    "cough",
    "high_fever",
    "sunken_eyes",
    "breathlessness",
    "sweating",
    "dehydration",
    "indigestion",
    "headache",
    "yellowish_skin",
    "dark_urine",
    "nausea",
    "loss_of_appetite",
    "pain_behind_the_eyes",
    "back_pain",
    "constipation",
    "abdominal_pain",
    "diarrhoea",
    "mild_fever",
    "yellow_urine",
    "yellowing_of_eyes",
    "acute_liver_failure",
    "fluid_overload",
    "swelling_of_stomach",
    "swelled_lymph_nodes",
    "malaise",
    "blurred_and_distorted_vision",
    "phlegm",
    "throat_irritation",
    "redness_of_eyes",
    "sinus_pressure",
    "runny_nose",
    "congestion",
    "chest_pain",
    "weakness_in_limbs",
    "fast_heart_rate",
    "pain_during_bowel_movements",
    "pain_in_anal_region",
    "bloody_stool",
    "irritation_in_anus",
    "neck_pain",
    "dizziness",
    "cramps",
    "bruising",
    "obesity",
    "swollen_legs",
    "swollen_blood_vessels",
    "puffy_face_and_eyes",
    "enlarged_thyroid",
    "brittle_nails",
    "swollen_extremities",
    "excessive_hunger",
    "extra_marital_contacts",
    "drying_and_tingling_lips",
    "slurred_speech",
    "knee_pain",
    "hip_joint_pain",
    "muscle_weakness",
    "stiff_neck",
    "swelling_joints",
    "movement_stiffness",
    "spinning_movements",
    "loss_of_balance",
    "unsteadiness",
    "weakness_of_one_body_side",
    "loss_of_smell",
    "bladder_discomfort",
    "foul_smell_of_urine",
    "continuous_feel_of_urine",
    "passage_of_gases",
    "internal_itching",
    "toxic_look_typhos",
    "depression",
    "irritability",
    "muscle_pain",
    "altered_sensorium",
    "red_spots_over_body",
    "belly_pain",
    "abnormal_menstruation",
    "dischromic_patches",
    "watering_from_eyes",
    "increased_appetite",
    "polyuria",
    "family_history",
    "mucoid_sputum",
    "rusty_sputum",
    "lack_of_concentration",
    "visual_disturbances",
    "receiving_blood_transfusion",
    "receiving_unsterile_injections",
    "coma",
    "stomach_bleeding",
    "distention_of_abdomen",
    "history_of_alcohol_consumption",
    "blood_in_sputum",
    "prominent_veins_on_calf",
    "palpitations",
    "painful_walking",
    "pus_filled_pimples",
    "blackheads",
    "scurring",
    "skin_peeling",
    "silver_like_dusting",
    "small_dents_in_nails",
    "inflammatory_nails",
    "blister",
    "red_sore_around_nose",
    "yellow_crust_ooze",
]


# ============================================================
# HUMAN-READABLE SYMPTOM NAMES
# ============================================================

def display_symptom(symptom):
    """
    Convert dataset column names into readable labels.
    """

    return (
        symptom
        .replace("_", " ")
        .replace("paroymsal", "paroxysmal")
        .title()
    )


# ============================================================
# SYMPTOM SELECTION
# ============================================================

st.header("📝 Select Symptoms")

st.write(
    "Select the symptoms that apply to the patient."
)

# Search box
search = st.text_input(
    "🔎 Search symptoms",
    placeholder="Example: headache, cough, fever...",
)

if search:
    filtered_symptoms = [
        symptom
        for symptom in SYMPTOMS
        if search.lower() in display_symptom(symptom).lower()
    ]
else:
    filtered_symptoms = SYMPTOMS


# Split symptoms into columns
columns = st.columns(3)

selected_symptoms = []

for index, symptom in enumerate(filtered_symptoms):

    with columns[index % 3]:

        selected = st.checkbox(
            display_symptom(symptom),
            key=f"symptom_{symptom}",
        )

        if selected:
            selected_symptoms.append(symptom)


# ============================================================
# SELECTED SYMPTOMS
# ============================================================

if selected_symptoms:

    st.subheader("Selected Symptoms")

    selected_display = [
        display_symptom(symptom)
        for symptom in selected_symptoms
    ]

    st.write(
        ", ".join(selected_display)
    )

else:

    st.warning(
        "Please select at least one symptom."
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.divider()

analyze = st.button(
    "🔍 Analyze Symptoms",
    type="primary",
    use_container_width=True,
)


# ============================================================
# PREDICTION
# ============================================================

if analyze:

    if not selected_symptoms:

        st.error(
            "Please select at least one symptom "
            "before running the analysis."
        )

        st.stop()

    # --------------------------------------------------------
    # Build patient input
    # --------------------------------------------------------

    patient_data = {}

    # Demographic information
    patient_data["age"] = age
    patient_data["gender"] = gender

    # All symptoms default to 0
    for symptom in SYMPTOMS:
        patient_data[symptom] = 0

    # Selected symptoms become 1
    for symptom in selected_symptoms:
        patient_data[symptom] = 1

    # --------------------------------------------------------
    # Run model
    # --------------------------------------------------------

    try:

        with st.spinner(
            "Analyzing symptoms..."
        ):

            result = predict_disease(
                patient_data,
                model_path="medbuddy_model.joblib",
                top_k=5,
            )

    except FileNotFoundError:

        st.error(
            "❌ Trained model not found.\n\n"
            "Run `python model.py` first to train "
            "and save the model."
        )

        st.stop()

    except Exception as error:

        st.error(
            f"Model error: {error}"
        )

        st.stop()


    # ========================================================
    # RESULTS
    # ========================================================

    st.header("🧠 Analysis Result")

    prediction = result["prediction"]
    confidence = result["confidence"]
    confidence_level = result["confidence_level"]


    # --------------------------------------------------------
    # Low confidence
    # --------------------------------------------------------

    if not result["safe_to_predict"]:

        st.warning(
            "⚠️ The model does not have enough evidence "
            "to make a confident prediction."
        )

        st.markdown(
            f"""
            <div class="result-box">
                <h2>Insufficient Evidence</h2>
                <p>
                    Model confidence:
                    <strong>{confidence:.1%}</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.success(
            f"Most likely model prediction: {prediction}"
        )

        st.markdown(
            f"""
            <div class="result-box">
                <h2>{prediction}</h2>
                <p class="confidence">
                    Confidence: {confidence:.1%}
                </p>
                <p>
                    Confidence level:
                    <strong>{confidence_level.title()}</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


    # ========================================================
    # TOP PREDICTIONS
    # ========================================================

    st.subheader(
        "📊 Other Possible Conditions"
    )

    predictions = result[
        "top_predictions"
    ]

    prediction_df = pd.DataFrame(
        predictions
    )

    if not prediction_df.empty:

        prediction_df[
            "confidence"
        ] = prediction_df[
            "confidence"
        ].apply(
            lambda x: f"{x:.1%}"
        )

        prediction_df.columns = [
            "Condition",
            "Model Confidence",
        ]

        st.table(
            prediction_df
        )


    # ========================================================
    # SAFETY MESSAGE
    # ========================================================

    st.markdown(
        f"""
        <div class="warning-box">

        <strong>⚠️ Important medical safety notice</strong>

        <p>
        {result["warning"]}
        </p>

        <p>
        Do not use this prediction to start, stop, or change
        medication. If symptoms are severe, rapidly worsening,
        or concerning, seek appropriate medical care.
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MedBuddy AI • Educational/Research Prototype • "
    "Not a substitute for professional medical advice"
)
