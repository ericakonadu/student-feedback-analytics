# ============================================================
# GROUP 12 - STUDENT FEEDBACK ANALYTICS
# STREAMLIT APPLICATION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk

from collections import Counter

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

from gensim.models import Word2Vec


# ============================================================
# 1. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Student Feedback Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. PROFESSIONAL STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .stApp {
        background: linear-gradient(
            180deg,
            #F8FAFC 0%,
            #F3F6F9 100%
        );
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #102A43 0%,
            #163D59 100%
        );
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #F5F7FA;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: white;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.15);
    }


    /* Metric cards */
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #E1E7EE;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 3px 10px rgba(15,23,42,0.04);
    }

    div[data-testid="stMetric"]:hover {
        border-color: #B8CAD8;
        box-shadow: 0 6px 16px rgba(15,23,42,0.07);
        transition: 0.2s ease;
    }


    /* Bordered containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px;
        border-color: #E1E7EE;
        background: rgba(255,255,255,0.85);
        box-shadow: 0 2px 8px rgba(15,23,42,0.03);
    }


    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        border-radius: 9px;
        font-weight: 600;
        min-height: 42px;
        transition: all 0.2s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 5px 12px rgba(15,23,42,0.12);
    }


    /* Inputs */
    .stTextInput input,
    .stTextArea textarea {
        border-radius: 9px;
    }

    div[data-baseweb="select"] > div {
        border-radius: 9px;
    }


    /* File uploader */
    section[data-testid="stFileUploaderDropzone"] {
        border-radius: 12px;
        border: 1.5px dashed #9FB3C8;
        background-color: #F8FAFC;
        padding: 18px;
    }


    /* Dataframes */
    div[data-testid="stDataFrame"] {
        border: 1px solid #E1E7EE;
        border-radius: 12px;
        overflow: hidden;
        background: white;
    }


    /* Alerts */
    div[data-testid="stAlert"] {
        border-radius: 10px;
    }


    /* Expanders */
    details {
        background: white;
        border-radius: 10px;
        border: 1px solid #E1E7EE;
        padding: 3px 8px;
    }


    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #E1E7EE;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }


    /* Mobile */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1.25rem;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. HEADER
# ============================================================

st.title("Student Feedback Analytics")

st.caption(
    "Sentiment classification using TF-IDF + Support Vector Machine "
    "and Word2Vec + Random Forest"
)

header_col1, header_col2, header_col3 = st.columns([1, 1, 2])

with header_col1:
    st.caption("MSBA 610")

with header_col2:
    st.caption("Group 12")

with header_col3:
    st.caption("Natural Language Processing & Machine Learning")

st.divider()


# ============================================================
# 4. NLTK SETUP
# ============================================================

@st.cache_resource
def setup_nltk():

    resources = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "omw-1.4"
    ]

    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass


setup_nltk()


# ============================================================
# 5. TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# 6. NLP OBJECTS
# ============================================================

stop_words = set(stopwords.words("english"))

negation_words = {
    "no",
    "nor",
    "not"
}

stop_words = stop_words - negation_words

lemmatizer = WordNetLemmatizer()


# ============================================================
# 7. TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):

    cleaned = clean_text(text)

    tokens = word_tokenize(cleaned)

    filtered_tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    lemmatized_tokens = [
        lemmatizer.lemmatize(word)
        for word in filtered_tokens
    ]

    processed_text = " ".join(lemmatized_tokens)

    return (
        cleaned,
        tokens,
        filtered_tokens,
        lemmatized_tokens,
        processed_text
    )


@st.cache_data
def preprocess_dataframe(data):

    data = data.copy()

    results = (
        data["feedback_text"]
        .astype(str)
        .apply(preprocess_text)
    )

    data["cleaned_text"] = results.apply(
        lambda x: x[0]
    )

    data["tokens"] = results.apply(
        lambda x: x[1]
    )

    data["filtered_tokens"] = results.apply(
        lambda x: x[2]
    )

    data["lemmatized_tokens"] = results.apply(
        lambda x: x[3]
    )

    data["processed_text"] = results.apply(
        lambda x: x[4]
    )

    data["word_count"] = (
        data["feedback_text"]
        .astype(str)
        .apply(
            lambda x: len(x.split())
        )
    )

    return data


# ============================================================
# 8. LOAD DEFAULT PROJECT DATA
# ============================================================

@st.cache_data
def load_project_data():

    return pd.read_csv(
        "student_feedback_dataset.csv"
    )


try:

    default_df = load_project_data()

except FileNotFoundError:

    st.error(
        "student_feedback_dataset.csv was not found. "
        "Place it in the same folder as app.py."
    )

    st.stop()


# ============================================================
# 9. SIDEBAR - DATA SOURCE
# ============================================================

st.sidebar.title(
    "Student Feedback"
)

st.sidebar.caption(
    "Sentiment Analytics Platform"
)

st.sidebar.divider()

st.sidebar.subheader(
    "Data Source"
)


data_source = st.sidebar.radio(
    "Choose the dataset used for analysis and model training",
    [
        "Use Project Dataset",
        "Upload Labelled Dataset"
    ]
)


# ============================================================
# 10. SELECT ACTIVE DATA
# ============================================================

if data_source == "Use Project Dataset":

    df = default_df.copy()

    dataset_name = "Project Dataset"


else:

    uploaded_training_file = (
        st.sidebar.file_uploader(
            "Upload labelled CSV",
            type=["csv"],
            key="training_upload"
        )
    )


    st.sidebar.caption(
        "Required columns: feedback_text and sentiment_label"
    )


    if uploaded_training_file is None:

        st.info(
            "Upload a labelled CSV file from the sidebar to continue."
        )

        st.stop()


    try:

        df = pd.read_csv(
            uploaded_training_file
        )

    except Exception as error:

        st.error(
            f"The CSV could not be read: {error}"
        )

        st.stop()


    dataset_name = uploaded_training_file.name


# ============================================================
# 11. DATASET VALIDATION
# ============================================================

required_training_columns = [
    "feedback_text",
    "sentiment_label"
]


missing_columns = [
    column
    for column in required_training_columns
    if column not in df.columns
]


if missing_columns:

    st.error(
        "The selected dataset is missing required columns: "
        + ", ".join(missing_columns)
    )

    st.stop()


df = df.dropna(
    subset=[
        "feedback_text",
        "sentiment_label"
    ]
).copy()


df["feedback_text"] = (
    df["feedback_text"]
    .astype(str)
    .str.strip()
)


df["sentiment_label"] = (
    df["sentiment_label"]
    .astype(str)
    .str.strip()
    .str.lower()
)


df = df[
    df["feedback_text"] != ""
].copy()


df = df[
    df["sentiment_label"] != ""
].copy()


if df["sentiment_label"].nunique() < 2:

    st.error(
        "The labelled dataset must contain at least two sentiment classes."
    )

    st.stop()


# ============================================================
# 12. PREPROCESS ACTIVE DATASET
# ============================================================

df = preprocess_dataframe(df)


# ============================================================
# 13. MODEL TRAINING
# ============================================================

@st.cache_resource
def train_models(data):

    X = data[
        "processed_text"
    ]

    y = data[
        "sentiment_label"
    ]


    class_counts = y.value_counts()

    can_stratify = (
        class_counts.min() >= 2
    )


    if can_stratify:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y
            )
        )

    else:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42
            )
        )


    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )


    X_train_tfidf = (
        tfidf.fit_transform(
            X_train
        )
    )


    X_test_tfidf = (
        tfidf.transform(
            X_test
        )
    )


    # --------------------------------------------------------
    # SVM
    # --------------------------------------------------------

    svm_model = SVC(
        kernel="linear",
        C=10,
        probability=True,
        random_state=42
    )


    svm_model.fit(
        X_train_tfidf,
        y_train
    )


    svm_predictions = (
        svm_model.predict(
            X_test_tfidf
        )
    )


    svm_probabilities = (
        svm_model.predict_proba(
            X_test_tfidf
        )
    )


    # --------------------------------------------------------
    # WORD2VEC
    # --------------------------------------------------------

    train_tokens = (
        data.loc[
            X_train.index,
            "lemmatized_tokens"
        ]
        .tolist()
    )


    test_tokens = (
        data.loc[
            X_test.index,
            "lemmatized_tokens"
        ]
        .tolist()
    )


    w2v_model = Word2Vec(
        sentences=train_tokens,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        sg=1,
        seed=42
    )


    def document_vector(tokens):

        vectors = [
            w2v_model.wv[word]
            for word in tokens
            if word in w2v_model.wv
        ]


        if len(vectors) == 0:

            return np.zeros(
                w2v_model.vector_size
            )


        return np.mean(
            vectors,
            axis=0
        )


    X_train_w2v = np.array([
        document_vector(tokens)
        for tokens in train_tokens
    ])


    X_test_w2v = np.array([
        document_vector(tokens)
        for tokens in test_tokens
    ])


    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )


    rf_model.fit(
        X_train_w2v,
        y_train
    )


    rf_predictions = (
        rf_model.predict(
            X_test_w2v
        )
    )


    rf_probabilities = (
        rf_model.predict_proba(
            X_test_w2v
        )
    )


    return {

        "X_train":
        X_train,

        "X_test":
        X_test,

        "y_train":
        y_train,

        "y_test":
        y_test,

        "tfidf":
        tfidf,

        "svm_model":
        svm_model,

        "svm_predictions":
        svm_predictions,

        "svm_probabilities":
        svm_probabilities,

        "w2v_model":
        w2v_model,

        "rf_model":
        rf_model,

        "rf_predictions":
        rf_predictions,

        "rf_probabilities":
        rf_probabilities
    }


with st.spinner(
    "Preparing text and models..."
):

    try:

        model_results = (
            train_models(
                df
            )
        )

    except Exception as error:

        st.error(
            "The models could not be trained on this dataset."
        )

        st.exception(
            error
        )

        st.stop()


# ============================================================
# 14. METRICS
# ============================================================

def calculate_metrics(
    y_true,
    predictions,
    probabilities,
    classes
):

    accuracy = accuracy_score(
        y_true,
        predictions
    )


    precision = precision_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0
    )


    recall = recall_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0
    )


    f1 = f1_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0
    )


    try:

        if len(classes) == 2:

            positive_class = classes[1]

            binary_true = (
                y_true
                == positive_class
            ).astype(int)

            roc_auc = roc_auc_score(
                binary_true,
                probabilities[:, 1]
            )

        else:

            roc_auc = roc_auc_score(
                y_true,
                probabilities,
                multi_class="ovr",
                average="weighted",
                labels=classes
            )

    except Exception:

        roc_auc = np.nan


    return {

        "Accuracy":
        accuracy,

        "Precision":
        precision,

        "Recall":
        recall,

        "Weighted F1":
        f1,

        "ROC-AUC":
        roc_auc
    }


svm_metrics = calculate_metrics(
    model_results["y_test"],
    model_results["svm_predictions"],
    model_results["svm_probabilities"],
    model_results["svm_model"].classes_
)


rf_metrics = calculate_metrics(
    model_results["y_test"],
    model_results["rf_predictions"],
    model_results["rf_probabilities"],
    model_results["rf_model"].classes_
)


# ============================================================
# 15. MODEL COMPARISON
# ============================================================

model_comparison = pd.DataFrame({

    "Model": [
        "TF-IDF + SVM",
        "Word2Vec + Random Forest"
    ],

    "Accuracy": [
        svm_metrics["Accuracy"],
        rf_metrics["Accuracy"]
    ],

    "Precision": [
        svm_metrics["Precision"],
        rf_metrics["Precision"]
    ],

    "Recall": [
        svm_metrics["Recall"],
        rf_metrics["Recall"]
    ],

    "Weighted F1": [
        svm_metrics["Weighted F1"],
        rf_metrics["Weighted F1"]
    ],

    "ROC-AUC": [
        svm_metrics["ROC-AUC"],
        rf_metrics["ROC-AUC"]
    ]
})


selected_model = (
    model_comparison.loc[
        model_comparison[
            "Weighted F1"
        ].idxmax(),
        "Model"
    ]
)


# ============================================================
# 16. LIVE PREDICTION MODEL
# ============================================================

@st.cache_resource
def train_live_model(data):

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )


    X_full = (
        vectorizer.fit_transform(
            data[
                "processed_text"
            ]
        )
    )


    model = SVC(
        kernel="linear",
        C=10,
        probability=True,
        random_state=42
    )


    model.fit(
        X_full,
        data[
            "sentiment_label"
        ]
    )


    return (
        vectorizer,
        model
    )


live_vectorizer, live_model = (
    train_live_model(
        df
    )
)


# ============================================================
# 17. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.divider()


page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Data Explorer",
        "Text Preprocessing",
        "Exploratory Text Analytics",
        "Model Performance",
        "Live Prediction",
        "Batch Prediction"
    ]
)


st.sidebar.divider()

st.sidebar.caption(
    f"Active data: {dataset_name}"
)

st.sidebar.caption(
    "Group 12 | MSBA 610"
)


# ============================================================
# PAGE 1 - OVERVIEW
# ============================================================

if page == "Overview":

    st.header(
        "Dataset Overview"
    )

    st.caption(
        "A consolidated view of the active dataset and its sentiment structure."
    )


    st.info(
        f"Current data source: {dataset_name}"
    )


    col1, col2, col3, col4 = (
        st.columns(4)
    )


    col1.metric(
        "Feedback Records",
        f"{len(df):,}"
    )


    if "department" in df.columns:

        col2.metric(
            "Departments",
            df[
                "department"
            ].nunique()
        )

    else:

        col2.metric(
            "Departments",
            "N/A"
        )


    if "subject_name" in df.columns:

        col3.metric(
            "Subjects",
            df[
                "subject_name"
            ].nunique()
        )

    else:

        col3.metric(
            "Subjects",
            "N/A"
        )


    col4.metric(
        "Sentiment Classes",
        df[
            "sentiment_label"
        ].nunique()
    )


    st.divider()


    left, right = (
        st.columns(
            [1.4, 1]
        )
    )


    with left:

        with st.container(
            border=True
        ):

            st.subheader(
                "Sentiment Distribution"
            )


            fig, ax = plt.subplots(
                figsize=(8, 4)
            )


            sns.countplot(
                data=df,
                x="sentiment_label",
                ax=ax
            )


            ax.set_xlabel(
                "Sentiment"
            )


            ax.set_ylabel(
                "Number of Records"
            )


            sns.despine()


            st.pyplot(
                fig,
                use_container_width=True
            )


    with right:

        with st.container(
            border=True
        ):

            st.subheader(
                "Dataset Profile"
            )


            class_counts = (
                df[
                    "sentiment_label"
                ]
                .value_counts()
            )


            for class_name, count in (
                class_counts.items()
            ):

                st.metric(
                    class_name.title(),
                    int(count)
                )


    st.divider()


    col_left, col_right = (
        st.columns(2)
    )


    with col_left:

        if "emotion_tag" in df.columns:

            with st.container(
                border=True
            ):

                st.subheader(
                    "Emotion Distribution"
                )


                st.bar_chart(
                    df[
                        "emotion_tag"
                    ].value_counts()
                )


    with col_right:

        if "feedback_type" in df.columns:

            with st.container(
                border=True
            ):

                st.subheader(
                    "Feedback Type"
                )


                st.bar_chart(
                    df[
                        "feedback_type"
                    ].value_counts()
                )


# ============================================================
# PAGE 2 - DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    st.header(
        "Data Explorer"
    )


    st.caption(
        "Filter and inspect individual records from the active dataset."
    )


    filtered_df = (
        df.copy()
    )


    available_filters = []


    if "department" in df.columns:

        available_filters.append(
            "department"
        )


    if "sentiment_label" in df.columns:

        available_filters.append(
            "sentiment_label"
        )


    if "feedback_type" in df.columns:

        available_filters.append(
            "feedback_type"
        )


    if available_filters:

        filter_columns = st.columns(
            len(
                available_filters
            )
        )


        for index, column in enumerate(
            available_filters
        ):

            options = [
                "All"
            ] + sorted(
                df[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )


            with filter_columns[index]:

                selection = (
                    st.selectbox(
                        column.replace(
                            "_",
                            " "
                        ).title(),
                        options,
                        key=f"filter_{column}"
                    )
                )


            if selection != "All":

                filtered_df = (
                    filtered_df[
                        filtered_df[
                            column
                        ].astype(str)
                        == selection
                    ]
                )


    st.metric(
        "Matching Records",
        len(
            filtered_df
        )
    )


    display_columns = [
        column
        for column in [
            "student_id",
            "department",
            "subject_name",
            "feedback_text",
            "sentiment_label",
            "emotion_tag",
            "feedback_type",
            "sarcasm_flag"
        ]
        if column
        in filtered_df.columns
    ]


    with st.container(
        border=True
    ):

        st.dataframe(
            filtered_df[
                display_columns
            ],
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PAGE 3 - TEXT PREPROCESSING
# ============================================================

elif page == "Text Preprocessing":

    st.header(
        "Text Preprocessing"
    )


    st.caption(
        "Transforming raw student feedback into consistent model-ready text."
    )


    step1, step2, step3 = (
        st.columns(3)
    )


    with step1:

        with st.container(
            border=True
        ):

            st.subheader(
                "Cleaning"
            )

            st.write(
                "Lowercase conversion and removal of HTML, URLs, "
                "emails, digits, punctuation and excess spaces."
            )


    with step2:

        with st.container(
            border=True
        ):

            st.subheader(
                "Tokenisation"
            )

            st.write(
                "Feedback text is split into individual word tokens using NLTK."
            )


    with step3:

        with st.container(
            border=True
        ):

            st.subheader(
                "Stopword Removal"
            )

            st.write(
                "Common English words are removed while no, nor and not are retained."
            )


    step4, step5 = (
        st.columns(2)
    )


    with step4:

        with st.container(
            border=True
        ):

            st.subheader(
                "Lemmatisation"
            )

            st.write(
                "Words are reduced towards their base or dictionary form."
            )


    with step5:

        with st.container(
            border=True
        ):

            st.subheader(
                "Final Text"
            )

            st.write(
                "Processed tokens are reconstructed into the final model-ready text."
            )


    st.divider()


    st.subheader(
        "Preprocessing Examples"
    )


    sample_size = st.slider(
        "Number of records",
        1,
        15,
        5
    )


    with st.container(
        border=True
    ):

        st.dataframe(
            df[
                [
                    "feedback_text",
                    "cleaned_text",
                    "processed_text"
                ]
            ].head(
                sample_size
            ),
            use_container_width=True,
            hide_index=True
        )


    st.divider()


    st.subheader(
        "Try the Preprocessing Pipeline"
    )


    with st.container(
        border=True
    ):

        example_text = (
            st.text_input(
                "Enter sample feedback",
                "The Lecturer was VERY helpful!!!"
            )
        )


        if example_text:

            (
                cleaned,
                tokens,
                filtered,
                lemmatized,
                processed
            ) = preprocess_text(
                example_text
            )


            col1, col2 = (
                st.columns(2)
            )


            with col1:

                st.write(
                    "**Original Text**"
                )

                st.write(
                    example_text
                )


                st.write(
                    "**Cleaned Text**"
                )

                st.write(
                    cleaned
                )


            with col2:

                st.write(
                    "**Final Processed Text**"
                )

                st.write(
                    processed
                )


            with st.expander(
                "View intermediate stages"
            ):

                st.write(
                    "**Tokens**"
                )

                st.write(
                    tokens
                )


                st.write(
                    "**After Stopword Removal**"
                )

                st.write(
                    filtered
                )


                st.write(
                    "**After Lemmatisation**"
                )

                st.write(
                    lemmatized
                )


# ============================================================
# PAGE 4 - EDA
# ============================================================

elif page == "Exploratory Text Analytics":

    st.header(
        "Exploratory Text Analysis"
    )


    st.caption(
        "Exploring vocabulary, feedback length and relationships within the active dataset."
    )


    all_words = " ".join(
        df[
            "processed_text"
        ]
    ).split()


    word_counts = Counter(
        all_words
    )


    top_words = pd.DataFrame(
        word_counts.most_common(
            20
        ),
        columns=[
            "Word",
            "Frequency"
        ]
    )


    col1, col2 = (
        st.columns(2)
    )


    with col1:

        with st.container(
            border=True
        ):

            st.subheader(
                "Most Frequent Words"
            )


            fig, ax = plt.subplots(
                figsize=(7, 5)
            )


            sns.barplot(
                data=top_words,
                x="Frequency",
                y="Word",
                ax=ax
            )


            sns.despine()


            st.pyplot(
                fig,
                use_container_width=True
            )


    with col2:

        with st.container(
            border=True
        ):

            st.subheader(
                "Feedback Length"
            )


            fig, ax = plt.subplots(
                figsize=(7, 5)
            )


            sns.histplot(
                data=df,
                x="word_count",
                bins=15,
                ax=ax
            )


            sns.despine()


            st.pyplot(
                fig,
                use_container_width=True
            )


    st.divider()


    if "emotion_tag" in df.columns:

        with st.container(
            border=True
        ):

            st.subheader(
                "Emotion vs Sentiment"
            )


            emotion_sentiment = (
                pd.crosstab(
                    df[
                        "emotion_tag"
                    ],
                    df[
                        "sentiment_label"
                    ]
                )
            )


            fig, ax = plt.subplots(
                figsize=(7, 5)
            )


            sns.heatmap(
                emotion_sentiment,
                annot=True,
                fmt="d",
                ax=ax
            )


            st.pyplot(
                fig,
                use_container_width=True
            )


    if "department" in df.columns:

        with st.container(
            border=True
        ):

            st.subheader(
                "Sentiment by Department"
            )


            st.bar_chart(
                pd.crosstab(
                    df[
                        "department"
                    ],
                    df[
                        "sentiment_label"
                    ]
                )
            )


    if "feedback_type" in df.columns:

        with st.container(
            border=True
        ):

            st.subheader(
                "Sentiment by Feedback Type"
            )


            st.bar_chart(
                pd.crosstab(
                    df[
                        "feedback_type"
                    ],
                    df[
                        "sentiment_label"
                    ]
                )
            )


# ============================================================
# PAGE 5 - MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header(
        "Model Performance"
    )


    st.caption(
        "Comparing TF-IDF + SVM with Word2Vec + Random Forest."
    )


    with st.container(
        border=True
    ):

        st.subheader(
            "Performance Summary"
        )


        st.dataframe(
            model_comparison
            .set_index(
                "Model"
            )
            .round(4),
            use_container_width=True
        )


    st.divider()


    with st.container(
        border=True
    ):

        st.subheader(
            "Selected Model"
        )

        st.metric(
            "Best Model by Weighted F1",
            selected_model
        )

        st.write(
            "Weighted F1 is used as the primary model-selection criterion."
        )


    st.divider()


    with st.container(
        border=True
    ):

        st.subheader(
            "Metric Comparison"
        )


        comparison_long = (
            model_comparison
            .melt(
                id_vars="Model",
                var_name="Metric",
                value_name="Score"
            )
        )


        fig, ax = plt.subplots(
            figsize=(9, 5)
        )


        sns.barplot(
            data=comparison_long,
            x="Metric",
            y="Score",
            hue="Model",
            ax=ax
        )


        ax.set_ylim(
            0,
            max(
                1,
                comparison_long[
                    "Score"
                ].max()
                + 0.1
            )
        )


        sns.despine()


        st.pyplot(
            fig,
            use_container_width=True
        )


    st.divider()


    st.subheader(
        "Confusion Matrices"
    )


    col1, col2 = (
        st.columns(2)
    )


    with col1:

        with st.container(
            border=True
        ):

            st.write(
                "**TF-IDF + SVM**"
            )


            cm_svm = (
                confusion_matrix(
                    model_results[
                        "y_test"
                    ],
                    model_results[
                        "svm_predictions"
                    ],
                    labels=model_results[
                        "svm_model"
                    ].classes_
                )
            )


            fig, ax = plt.subplots(
                figsize=(6, 5)
            )


            sns.heatmap(
                cm_svm,
                annot=True,
                fmt="d",
                xticklabels=model_results[
                    "svm_model"
                ].classes_,
                yticklabels=model_results[
                    "svm_model"
                ].classes_,
                ax=ax
            )


            ax.set_xlabel(
                "Predicted"
            )


            ax.set_ylabel(
                "Actual"
            )


            st.pyplot(
                fig,
                use_container_width=True
            )


    with col2:

        with st.container(
            border=True
        ):

            st.write(
                "**Word2Vec + Random Forest**"
            )


            cm_rf = (
                confusion_matrix(
                    model_results[
                        "y_test"
                    ],
                    model_results[
                        "rf_predictions"
                    ],
                    labels=model_results[
                        "rf_model"
                    ].classes_
                )
            )


            fig, ax = plt.subplots(
                figsize=(6, 5)
            )


            sns.heatmap(
                cm_rf,
                annot=True,
                fmt="d",
                xticklabels=model_results[
                    "rf_model"
                ].classes_,
                yticklabels=model_results[
                    "rf_model"
                ].classes_,
                ax=ax
            )


            ax.set_xlabel(
                "Predicted"
            )


            ax.set_ylabel(
                "Actual"
            )


            st.pyplot(
                fig,
                use_container_width=True
            )


# ============================================================
# PAGE 6 - LIVE PREDICTION
# ============================================================

elif page == "Live Prediction":

    st.header(
        "Sentiment Prediction"
    )


    st.caption(
        "Enter one student feedback comment and classify its predicted sentiment."
    )


    with st.container(
        border=True
    ):

        new_feedback = (
            st.text_area(
                "Student Feedback",
                height=150,
                placeholder=(
                    "Example: The lecturer explains the topics clearly "
                    "and the course is very helpful."
                )
            )
        )


        predict_button = (
            st.button(
                "Predict Sentiment",
                type="primary",
                use_container_width=True
            )
        )


    if predict_button:

        if not new_feedback.strip():

            st.warning(
                "Please enter some feedback first."
            )


        else:

            (
                cleaned,
                tokens,
                filtered,
                lemmatized,
                processed
            ) = preprocess_text(
                new_feedback
            )


            transformed_text = (
                live_vectorizer
                .transform(
                    [processed]
                )
            )


            prediction = (
                live_model
                .predict(
                    transformed_text
                )[0]
            )


            probabilities = (
                live_model
                .predict_proba(
                    transformed_text
                )[0]
            )


            st.divider()


            with st.container(
                border=True
            ):

                st.subheader(
                    "Prediction Result"
                )


                st.metric(
                    "Predicted Sentiment",
                    prediction.title()
                )


                probability_df = (
                    pd.DataFrame(
                        {
                            "Sentiment":
                            live_model.classes_,

                            "Probability":
                            probabilities
                        }
                    )
                )


                probability_df[
                    "Probability"
                ] = (
                    probability_df[
                        "Probability"
                    ]
                    * 100
                ).round(2)


                col1, col2 = (
                    st.columns(2)
                )


                with col1:

                    st.dataframe(
                        probability_df,
                        use_container_width=True,
                        hide_index=True
                    )


                with col2:

                    st.bar_chart(
                        probability_df
                        .set_index(
                            "Sentiment"
                        )
                    )


# ============================================================
# PAGE 7 - BATCH PREDICTION
# ============================================================

elif page == "Batch Prediction":

    st.header(
        "Batch Feedback Prediction"
    )


    st.caption(
        "Upload a CSV containing feedback comments and predict sentiment for every row."
    )


    with st.container(
        border=True
    ):

        st.subheader(
            "Prediction File"
        )

        st.write(
            "Upload a CSV containing a column named `feedback_text`. "
            "The trained SVM model will predict the sentiment of each row."
        )


        prediction_file = (
            st.file_uploader(
                "Upload CSV for prediction",
                type=["csv"],
                key="batch_prediction_upload"
            )
        )


    if prediction_file is not None:

        try:

            prediction_df = (
                pd.read_csv(
                    prediction_file
                )
            )

        except Exception as error:

            st.error(
                f"The CSV could not be read: {error}"
            )

            st.stop()


        if (
            "feedback_text"
            not in prediction_df.columns
        ):

            st.error(
                "The uploaded prediction file must contain "
                "a column named 'feedback_text'."
            )

            st.stop()


        st.divider()


        with st.container(
            border=True
        ):

            st.subheader(
                "Uploaded Feedback"
            )


            st.dataframe(
                prediction_df.head(
                    20
                ),
                use_container_width=True,
                hide_index=True
            )


            st.metric(
                "Rows Ready for Prediction",
                len(
                    prediction_df
                )
            )


            run_batch = (
                st.button(
                    "Run Batch Prediction",
                    type="primary",
                    use_container_width=True
                )
            )


        if run_batch:

            with st.spinner(
                "Predicting sentiment..."
            ):

                batch_results = (
                    prediction_df.copy()
                )


                processed_batch = (
                    batch_results[
                        "feedback_text"
                    ]
                    .astype(str)
                    .apply(
                        lambda text:
                        preprocess_text(
                            text
                        )[4]
                    )
                )


                batch_vectors = (
                    live_vectorizer
                    .transform(
                        processed_batch
                    )
                )


                batch_predictions = (
                    live_model
                    .predict(
                        batch_vectors
                    )
                )


                batch_probabilities = (
                    live_model
                    .predict_proba(
                        batch_vectors
                    )
                )


                batch_results[
                    "predicted_sentiment"
                ] = (
                    batch_predictions
                )


                batch_results[
                    "prediction_confidence"
                ] = (
                    batch_probabilities.max(
                        axis=1
                    )
                    * 100
                ).round(2)


                for index, class_name in enumerate(
                    live_model.classes_
                ):

                    batch_results[
                        f"probability_{class_name}"
                    ] = (
                        batch_probabilities[
                            :,
                            index
                        ]
                        * 100
                    ).round(2)


            st.success(
                "Batch prediction completed."
            )


            with st.container(
                border=True
            ):

                st.subheader(
                    "Prediction Results"
                )


                st.dataframe(
                    batch_results,
                    use_container_width=True,
                    hide_index=True
                )


            col1, col2 = (
                st.columns(
                    [1.2, 0.8]
                )
            )


            with col1:

                with st.container(
                    border=True
                ):

                    st.subheader(
                        "Predicted Sentiment Distribution"
                    )


                    st.bar_chart(
                        batch_results[
                            "predicted_sentiment"
                        ]
                        .value_counts()
                    )


            with col2:

                with st.container(
                    border=True
                ):

                    st.subheader(
                        "Export Results"
                    )


                    csv_output = (
                        batch_results
                        .to_csv(
                            index=False
                        )
                        .encode(
                            "utf-8"
                        )
                    )


                    st.download_button(
                        label=(
                            "Download Prediction Results"
                        ),
                        data=csv_output,
                        file_name=(
                            "student_feedback_predictions.csv"
                        ),
                        mime="text/csv",
                        use_container_width=True
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "Group 12 | MSBA 610 Advanced Text Analytics | "
    "Student Feedback Sentiment Analytics"
)
