import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder


TEXT_COLS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

CAT_COLS = [
    "location",
    "department",
    "salary_range",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function"
]

BINARY_COLS = [
    "telecommuting",
    "has_company_logo",
    "has_questions"
]

FEATURE_COLS = TEXT_COLS + CAT_COLS + BINARY_COLS


def prepare_features(data):
    """
    Prepare raw job records using fixed cleaning rules.

    Also supports incomplete inputs from the future app.
    No statistics are learnt from the data.
    """

    result = data.copy()

    for column in TEXT_COLS:
        if column not in result.columns:
            result[column] = ""

        result[column] = (
            result[column]
            .fillna("")
            .astype(str)
        )

    for column in CAT_COLS:
        if column not in result.columns:
            result[column] = "Missing"

        result[column] = (
            result[column]
            .fillna("Missing")
            .astype(str)
            .str.strip()
            .replace("", "Missing")
        )

    for column in BINARY_COLS:
        if column not in result.columns:
            result[column] = 0

        result[column] = (
            np.asarray(
                result[column]
                .pipe(
                    lambda values: __import__("pandas")
                    .to_numeric(values, errors="coerce")
                )
                .fillna(0),
                dtype=np.float32
            )
        )

        if not result[column].isin([0, 1]).all():
            raise ValueError(
                f"Invalid binary values in {column}"
            )

    return result[FEATURE_COLS]


def build_preprocessor():
    """
    Build a combined text and structured-data
    feature extraction pipeline.
    """

    text_limits = {
        "title": 400,
        "company_profile": 600,
        "description": 1400,
        "requirements": 800,
        "benefits": 400
    }

    transformers = []

    for column, max_features in text_limits.items():

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=3,
            max_df=0.98,
            ngram_range=(1, 2),
            sublinear_tf=True,
            dtype=np.float32
        )

        transformers.append(
            (
                f"tfidf_{column}",
                vectorizer,
                column
            )
        )

    categorical_encoder = OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=20,
        dtype=np.float32
    )

    transformers.append(
        (
            "categorical",
            categorical_encoder,
            CAT_COLS
        )
    )

    transformers.append(
        (
            "binary",
            "passthrough",
            BINARY_COLS
        )
    )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0
    )

    return preprocessor
