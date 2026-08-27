"""
MedBuddy AI - Improved Medical Prediction Model

Educational / research prototype.
NOT a clinically validated diagnostic system.
"""

from __future__ import annotations

import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/dataset.csv"
MODEL_PATH = "medbuddy_model.joblib"

TARGET_COLUMN = "disease"

TEST_SIZE = 0.20
RANDOM_STATE = 42

# Predictions below this threshold are reported as
# "Insufficient evidence" rather than forcing a diagnosis.
CONFIDENCE_THRESHOLD = 0.55


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(
    data_path: str = DATA_PATH,
    target_column: str = TARGET_COLUMN,
):
    """Load and clean the dataset."""

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    df = pd.read_csv(data_path)

    if df.empty:
        raise ValueError("Dataset is empty.")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found.\n"
            f"Available columns:\n{list(df.columns)}"
        )

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    # Remove exact duplicate rows
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        df = df.drop_duplicates()

    # Remove rows with missing target
    df = df.dropna(
        subset=[target_column]
    )

    # Normalize disease labels
    df[target_column] = (
        df[target_column]
        .astype(str)
        .str.strip()
    )

    # Remove empty disease labels
    df = df[
        df[target_column] != ""
    ]

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    if y.nunique() < 2:
        raise ValueError(
            "Dataset must contain at least "
            "two disease classes."
        )

    print("\n" + "=" * 60)
    print("MEDBUDDY AI DATASET")
    print("=" * 60)

    print(f"Samples:       {len(df):,}")
    print(f"Features:      {X.shape[1]:,}")
    print(f"Diseases:      {y.nunique():,}")
    print(
        f"Duplicates removed: {duplicate_count}"
    )

    print("\nDisease distribution:")
    print(
        y.value_counts().to_string()
    )

    return X, y


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor(
    X: pd.DataFrame,
):
    """
    Automatically handles numerical and categorical
    features.

    Numerical:
        Median imputation
        Standard scaling

    Categorical:
        Most-frequent imputation
        One-hot encoding
    """

    numerical_features = (
        X.select_dtypes(
            include=[
                "int64",
                "int32",
                "float64",
                "float32",
            ]
        )
        .columns
        .tolist()
    )

    categorical_features = (
        X.select_dtypes(
            include=[
                "object",
                "category",
                "bool",
            ]
        )
        .columns
        .tolist()
    )

    print("\n" + "=" * 60)
    print("FEATURE INFORMATION")
    print("=" * 60)

    print(
        f"Numerical features:   "
        f"{len(numerical_features)}"
    )

    print(
        f"Categorical features: "
        f"{len(categorical_features)}"
    )

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    transformers = []

    if numerical_features:

        transformers.append(
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            )
        )

    if categorical_features:

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    if not transformers:

        raise ValueError(
            "No usable features found."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# ============================================================
# BUILD MODEL
# ============================================================

def build_model(
    X: pd.DataFrame,
):
    """
    Build preprocessing + classification pipeline.

    Logistic Regression provides a strong baseline for
    multi-class symptom classification and supports
    probability estimates.
    """

    preprocessor = build_preprocessor(X)

    classifier = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )

    return model


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    data_path: str = DATA_PATH,
    target_column: str = TARGET_COLUMN,
    model_path: str = MODEL_PATH,
):
    """Train, evaluate and save the model."""

    X, y = load_dataset(
        data_path,
        target_column,
    )

    # --------------------------------------------------------
    # Stratified train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    print("\n" + "=" * 60)
    print("TRAIN / TEST SPLIT")
    print("=" * 60)

    print(
        f"Training samples: {len(X_train):,}"
    )

    print(
        f"Testing samples:  {len(X_test):,}"
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_model(
        X_train
    )

    print("\nTraining model...")

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print(
        f"Accuracy:        {accuracy:.4f}"
    )

    print(
        f"Macro Precision: {precision:.4f}"
    )

    print(
        f"Macro Recall:    {recall:.4f}"
    )

    print(
        f"Macro F1:        {macro_f1:.4f}"
    )

    print(
        f"Weighted F1:     {weighted_f1:.4f}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    print(
        confusion_matrix(
            y_test,
            y_pred,
        )
    )

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    try:

        probabilities = (
            model.predict_proba(
                X_test
            )
        )

        if len(model.classes_) == 2:

            auc = roc_auc_score(
                y_test,
                probabilities[:, 1],
            )

        else:

            auc = roc_auc_score(
                y_test,
                probabilities,
                multi_class="ovr",
                average="macro",
            )

        print(
            f"\nMacro ROC-AUC: {auc:.4f}"
        )

    except Exception as error:

        print(
            f"\nROC-AUC unavailable: "
            f"{error}"
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    artifact = {
        "model": model,
        "target_column": target_column,
        "classes": list(
            model.classes_
        ),
        "feature_columns": list(
            X.columns
        ),
        "confidence_threshold":
            CONFIDENCE_THRESHOLD,
    }

    joblib.dump(
        artifact,
        model_path,
    )

    print("\n" + "=" * 60)
    print("MODEL SAVED")
    print("=" * 60)

    print(
        f"Location: {model_path}"
    )

    return artifact


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    model_path: str = MODEL_PATH,
):
    """Load trained model."""

    if not os.path.exists(
        model_path
    ):
        raise FileNotFoundError(
            f"Model not found: "
            f"{model_path}\n\n"
            "Run:\n"
            "python model.py"
        )

    artifact = joblib.load(
        model_path
    )

    if not isinstance(
        artifact,
        dict,
    ):
        raise ValueError(
            "Invalid model file."
        )

    return artifact


# ============================================================
# PREDICT
# ============================================================

def predict_disease(
    patient_data: dict,
    model_path: str = MODEL_PATH,
    top_k: int = 5,
):
    """
    Predict the most likely conditions.

    Returns:
        prediction
        confidence
        confidence_level
        safe_to_predict
        top_predictions
        warning
    """

    artifact = load_model(
        model_path
    )

    model = artifact["model"]

    expected_features = (
        artifact["feature_columns"]
    )

    threshold = artifact.get(
        "confidence_threshold",
        CONFIDENCE_THRESHOLD,
    )

    # --------------------------------------------------------
    # Convert input to DataFrame
    # --------------------------------------------------------

    patient_df = pd.DataFrame(
        [patient_data]
    )

    # --------------------------------------------------------
    # Add missing features
    # --------------------------------------------------------

    for feature in expected_features:

        if feature not in patient_df.columns:

            patient_df[
                feature
            ] = np.nan

    # --------------------------------------------------------
    # Remove unexpected features
    # --------------------------------------------------------

    patient_df = patient_df[
        expected_features
    ]

    # --------------------------------------------------------
    # Probability prediction
    # --------------------------------------------------------

    probabilities = (
        model.predict_proba(
            patient_df
        )[0]
    )

    classes = np.asarray(
        model.classes_
    )

    # Sort highest → lowest
    sorted_indices = np.argsort(
        probabilities
    )[::-1]

    top_k = min(
        top_k,
        len(classes),
    )

    predictions = []

    for index in sorted_indices[
        :top_k
    ]:

        predictions.append(
            {
                "condition": str(
                    classes[index]
                ),
                "confidence": round(
                    float(
                        probabilities[index]
                    ),
                    4,
                ),
            }
        )

    # --------------------------------------------------------
    # Best prediction
    # --------------------------------------------------------

    best_index = sorted_indices[0]

    best_prediction = str(
        classes[best_index]
    )

    confidence = float(
        probabilities[best_index]
    )

    # --------------------------------------------------------
    # Confidence level
    # --------------------------------------------------------

    if confidence >= 0.75:

        confidence_level = "high"

    elif confidence >= threshold:

        confidence_level = "moderate"

    else:

        confidence_level = "low"

    # --------------------------------------------------------
    # Don't force low-confidence predictions
    # --------------------------------------------------------

    safe_to_predict = (
        confidence >= threshold
    )

    if not safe_to_predict:

        final_prediction = (
            "Insufficient evidence"
        )

    else:

        final_prediction = (
            best_prediction
        )

    return {
        "prediction":
            final_prediction,

        "confidence":
            round(
                confidence,
                4,
            ),

        "confidence_level":
            confidence_level,

        "safe_to_predict":
            safe_to_predict,

        "top_predictions":
            predictions,

        "warning":
            (
                "This is an AI model prediction "
                "and is not a medical diagnosis. "
                "Consult a qualified healthcare "
                "professional for medical advice."
            ),
    }


# ============================================================
# FEATURE INFORMATION
# ============================================================

def get_feature_information(
    model_path: str = MODEL_PATH,
    top_n: int = 20,
):
    """
    Show the most influential encoded features.

    Useful for understanding the model.
    """

    artifact = load_model(
        model_path
    )

    model = artifact["model"]

    preprocessor = (
        model.named_steps[
            "preprocessor"
        ]
    )

    classifier = (
        model.named_steps[
            "classifier"
        ]
    )

    try:

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        coefficients = (
            classifier.coef_
        )

        importance = np.mean(
            np.abs(coefficients),
            axis=0,
        )

        result = pd.DataFrame(
            {
                "feature":
                    feature_names,

                "importance":
                    importance,
            }
        )

        return (
            result
            .sort_values(
                "importance",
                ascending=False,
            )
            .head(top_n)
        )

    except Exception as error:

        print(
            f"Unable to calculate "
            f"feature importance: {error}"
        )

        return pd.DataFrame()


# ============================================================
# DATASET QUALITY REPORT
# ============================================================

def dataset_quality_report(
    data_path: str = DATA_PATH,
    target_column: str = TARGET_COLUMN,
):
    """Print dataset quality information."""

    if not os.path.exists(
        data_path
    ):
        raise FileNotFoundError(
            data_path
        )

    df = pd.read_csv(
        data_path
    )

    print("\n" + "=" * 60)
    print("DATASET QUALITY REPORT")
    print("=" * 60)

    print(
        f"\nDataset shape: {df.shape}"
    )

    # Missing values
    print("\nMissing values:")
    print("-" * 60)

    missing = (
        df.isnull()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing = missing[
        missing > 0
    ]

    if len(missing):

        print(
            missing.to_string()
        )

    else:

        print(
            "No missing values."
        )

    # Duplicates
    print("\nDuplicate rows:")
    print("-" * 60)

    print(
        df.duplicated().sum()
    )

    # Target distribution
    if target_column in df.columns:

        print(
            "\nDisease distribution:"
        )

        print("-" * 60)

        print(
            df[target_column]
            .value_counts()
            .to_string()
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\nStarting MedBuddy AI model training..."
    )

    train_model()

    print(
        "\nTraining finished successfully."
    )
