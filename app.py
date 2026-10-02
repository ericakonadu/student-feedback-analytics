# ============================================================
# GROUP 12 - STUDENT FEEDBACK ANALYTICS
# Streamlit Application
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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize

from gensim.models import Word2Vec


# ============================================================
# 1. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Feedback Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("Student Feedback Analytics")

st.caption(
    "Sentiment Classification using TF-IDF + Support Vector Machine "
    "and Word2Vec + Random Forest"
)


# ============================================================
# 2. NLTK RESOURCE CHECK
# ============================================================

def ensure_nltk_resource(resource_name, resource_path):
    """
    Check whether an NLTK resource exists.
    If not, attempt to download it.
    """
    try:
        nltk.data.find(resource_path)
    except LookupError:
        try:
            nltk.download(resource_name, quiet=True)
        except Exception:
            pass


ensure_nltk_resource("punkt", "tokenizers/punkt")
ensure_nltk_resource("punkt_tab", "tokenizers/punkt_tab")
ensure_nltk_resource("stopwords", "corpora/stopwords")
ensure_nltk_resource("wordnet", "corpora/wordnet")
ensure_nltk_resource("omw-1.4", "corpora/omw-1.4")


# ============================================================
# 3. LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("student_feedback_dataset.csv")


try:
    df = load_data()

except FileNotFoundError:

    st.error(
        "student_feedback_dataset.csv was not found. "
        "Place the CSV file in the same folder as app.py."
    )

    st.stop()


if df.empty:
    st.error("The dataset is empty.")
    st.stop()


# ============================================================
# 4. TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean raw feedback using regular expressions.
    """

    text = str(text).lower()

    # Remove HTML
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Keep alphabetic characters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# 5. NLP OBJECTS
# ============================================================

try:
    stop_words = set(stopwords.words("english"))

except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))


# Preserve negation words because they are important in sentiment
negation_words = {"no", "nor", "not"}

stop_words = stop_words - negation_words

lemmatizer = WordNetLemmatizer()


# ============================================================
# 6. TEXT PREPROCESSING FUNCTION
# ============================================================

def preprocess_text(text):

    cleaned = clean_text(text)

    try:
        tokens = word_tokenize(cleaned)

    except LookupError:
        nltk.download("punkt")
        nltk.download("punkt_tab")
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


# ============================================================
# 7. PREPROCESS COMPLETE DATAFRAME
# ============================================================

@st.cache_data
def preprocess_dataframe(data):

    data = data.copy()

    results = data["feedback_text"].apply(preprocess_text)

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
        .apply(lambda x: len(x.split()))
    )

    return data


df = preprocess_dataframe(df)


# ============================================================
# 8. MODEL TRAINING AND EVALUATION
# ============================================================

@st.cache_resource
def train_evaluation_models(data):

    # --------------------------------------------------------
    # Train/Test Split
    # --------------------------------------------------------

    X = data["processed_text"]
    y = data["sentiment_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
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

    X_train_tfidf = tfidf.fit_transform(X_train)

    X_test_tfidf = tfidf.transform(X_test)


    # --------------------------------------------------------
    # Tuned SVM
    # Best parameter from notebook: C = 10
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

    svm_predictions = svm_model.predict(
        X_test_tfidf
    )

    svm_probabilities = svm_model.predict_proba(
        X_test_tfidf
    )


    # --------------------------------------------------------
    # Word2Vec
    # Train ONLY using training documents
    # --------------------------------------------------------

    train_tokens = data.loc[
        X_train.index,
        "lemmatized_tokens"
    ].tolist()

    test_tokens = data.loc[
        X_test.index,
        "lemmatized_tokens"
    ].tolist()


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
    # Tuned Random Forest
    # Best parameters obtained from notebook
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

    rf_predictions = rf_model.predict(
        X_test_w2v
    )

    rf_probabilities = rf_model.predict_proba(
        X_test_w2v
    )


    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,

        "tfidf": tfidf,
        "svm_model": svm_model,
        "svm_predictions": svm_predictions,
        "svm_probabilities": svm_probabilities,

        "w2v_model": w2v_model,
        "rf_model": rf_model,
        "rf_predictions": rf_predictions,
        "rf_probabilities": rf_probabilities
    }


model_results = train_evaluation_models(df)


# ============================================================
# 9. MODEL METRIC FUNCTION
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

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
        multi_class="ovr",
        average="weighted",
        labels=classes
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Weighted F1": f1,
        "ROC-AUC": roc_auc
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
# 10. MODEL COMPARISON TABLE
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
        model_comparison["Weighted F1"].idxmax(),
        "Model"
    ]
)


# ============================================================
# 11. FINAL SVM FOR LIVE PREDICTION
# Train selected model using ALL labelled data
# ============================================================

@st.cache_resource
def train_live_prediction_model(data):

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    X_full = vectorizer.fit_transform(
        data["processed_text"]
    )

    model = SVC(
        kernel="linear",
        C=10,
        probability=True,
        random_state=42
    )

    model.fit(
        X_full,
        data["sentiment_label"]
    )

    return vectorizer, model


live_vectorizer, live_model = (
    train_live_prediction_model(df)
)


# ============================================================
# 12. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Overview",
        "Data Explorer",
        "Text Preprocessing",
        "Exploratory Text Analytics",
        "Model Performance",
        "Live Prediction"
    ]
)


# ============================================================
# PAGE 1 - OVERVIEW
# ============================================================

if page == "Overview":

    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Feedback Records",
        len(df)
    )

    col2.metric(
        "Departments",
        df["department"].nunique()
    )

    col3.metric(
        "Subjects",
        df["subject_name"].nunique()
    )

    col4.metric(
        "Sentiment Classes",
        df["sentiment_label"].nunique()
    )


    st.subheader("Sentiment Distribution")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="sentiment_label",
        order=df["sentiment_label"].value_counts().index,
        ax=ax
    )

    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Records")

    st.pyplot(fig)


    col1, col2 = st.columns(2)


    with col1:

        st.subheader("Emotion Distribution")

        emotion_counts = (
            df["emotion_tag"]
            .value_counts()
        )

        st.bar_chart(emotion_counts)


    with col2:

        st.subheader("Feedback Type")

        feedback_counts = (
            df["feedback_type"]
            .value_counts()
        )

        st.bar_chart(feedback_counts)


    st.subheader("Key Dataset Findings")

    st.write(
        """
        - The dataset contains 1,223 student feedback records.
        - Positive feedback forms the largest sentiment class.
        - The three sentiment classes are reasonably balanced.
        - Feedback comments are generally short.
        - Exploratory analysis shows considerable vocabulary overlap
          between positive, neutral and negative feedback.
        """
    )


# ============================================================
# PAGE 2 - DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    st.header("Data Explorer")

    col1, col2, col3 = st.columns(3)

    with col1:

        selected_department = st.selectbox(
            "Department",
            ["All"] + sorted(
                df["department"].unique().tolist()
            )
        )


    with col2:

        selected_sentiment = st.selectbox(
            "Sentiment",
            ["All"] + sorted(
                df["sentiment_label"].unique().tolist()
            )
        )


    with col3:

        selected_type = st.selectbox(
            "Feedback Type",
            ["All"] + sorted(
                df["feedback_type"].unique().tolist()
            )
        )


    filtered_df = df.copy()


    if selected_department != "All":

        filtered_df = filtered_df[
            filtered_df["department"]
            == selected_department
        ]


    if selected_sentiment != "All":

        filtered_df = filtered_df[
            filtered_df["sentiment_label"]
            == selected_sentiment
        ]


    if selected_type != "All":

        filtered_df = filtered_df[
            filtered_df["feedback_type"]
            == selected_type
        ]


    st.write(
        "Records displayed:",
        len(filtered_df)
    )


    st.dataframe(
        filtered_df[
            [
                "student_id",
                "department",
                "subject_name",
                "feedback_text",
                "sentiment_label",
                "emotion_tag",
                "feedback_type",
                "sarcasm_flag"
            ]
        ],
        use_container_width=True
    )


# ============================================================
# PAGE 3 - TEXT PREPROCESSING
# ============================================================

elif page == "Text Preprocessing":

    st.header("Text Preprocessing Pipeline")

    st.write(
        """
        Raw student feedback was processed using the following stages:

        1. Lowercase conversion
        2. Regular-expression cleaning
        3. Tokenisation
        4. Stopword removal
        5. Negation preservation
        6. Lemmatisation
        7. Reconstruction into final processed text
        """
    )


    sample_size = st.slider(
        "Number of examples",
        min_value=1,
        max_value=20,
        value=5
    )


    st.dataframe(
        df[
            [
                "feedback_text",
                "cleaned_text",
                "tokens",
                "filtered_tokens",
                "lemmatized_tokens",
                "processed_text"
            ]
        ].head(sample_size),
        use_container_width=True
    )


    st.subheader("Preprocessing Example")

    example_text = st.text_input(
        "Enter text to preprocess",
        "The Lecturer was VERY helpful!!!"
    )


    if example_text:

        (
            cleaned,
            tokens,
            filtered,
            lemmatized,
            processed
        ) = preprocess_text(example_text)


        st.write("**Original:**", example_text)
        st.write("**Cleaned:**", cleaned)
        st.write("**Tokens:**", tokens)
        st.write("**After Stopword Removal:**", filtered)
        st.write("**Lemmatized:**", lemmatized)
        st.write("**Final Processed Text:**", processed)


# ============================================================
# PAGE 4 - EXPLORATORY TEXT ANALYTICS
# ============================================================

elif page == "Exploratory Text Analytics":

    st.header("Exploratory Text Analytics")


    # --------------------------------------------------------
    # TOP WORDS
    # --------------------------------------------------------

    all_words = " ".join(
        df["processed_text"]
    ).split()

    word_counts = Counter(all_words)


    top_words = pd.DataFrame(
        word_counts.most_common(20),
        columns=[
            "Word",
            "Frequency"
        ]
    )


    st.subheader("Top 20 Most Frequent Words")

    fig, ax = plt.subplots(figsize=(9, 6))

    sns.barplot(
        data=top_words,
        x="Frequency",
        y="Word",
        ax=ax
    )

    st.pyplot(fig)


    # --------------------------------------------------------
    # WORDS BY SENTIMENT
    # --------------------------------------------------------

    st.subheader("Most Frequent Words by Sentiment")


    selected_sentiment_words = st.selectbox(
        "Select Sentiment",
        sorted(
            df["sentiment_label"]
            .unique()
            .tolist()
        )
    )


    sentiment_text = " ".join(
        df.loc[
            df["sentiment_label"]
            == selected_sentiment_words,
            "processed_text"
        ]
    )


    sentiment_counts = Counter(
        sentiment_text.split()
    )


    sentiment_words = pd.DataFrame(
        sentiment_counts.most_common(15),
        columns=[
            "Word",
            "Frequency"
        ]
    )


    st.dataframe(
        sentiment_words,
        use_container_width=True
    )


    # --------------------------------------------------------
    # EMOTION VS SENTIMENT
    # --------------------------------------------------------

    st.subheader("Emotion Tag by Sentiment")


    emotion_sentiment = pd.crosstab(
        df["emotion_tag"],
        df["sentiment_label"]
    )


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    sns.heatmap(
        emotion_sentiment,
        annot=True,
        fmt="d",
        ax=ax
    )

    st.pyplot(fig)


    # --------------------------------------------------------
    # DEPARTMENT VS SENTIMENT
    # --------------------------------------------------------

    st.subheader("Sentiment Distribution by Department")


    department_sentiment = pd.crosstab(
        df["department"],
        df["sentiment_label"]
    )


    st.bar_chart(
        department_sentiment
    )


    # --------------------------------------------------------
    # FEEDBACK TYPE
    # --------------------------------------------------------

    st.subheader("Sentiment by Feedback Type")


    feedback_sentiment = pd.crosstab(
        df["feedback_type"],
        df["sentiment_label"]
    )


    st.bar_chart(
        feedback_sentiment
    )


# ============================================================
# PAGE 5 - MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header("Model Performance")


    st.subheader("Model Comparison")

    comparison_display = (
        model_comparison
        .set_index("Model")
        .round(4)
    )

    st.dataframe(
        comparison_display,
        use_container_width=True
    )


    st.success(
        f"Selected Model: {selected_model}"
    )


    st.write(
        """
        Weighted F1 was defined as the primary model-selection criterion.
        The TF-IDF + SVM model achieved the higher weighted F1 score,
        although Random Forest achieved slightly higher accuracy and
        ROC-AUC.
        """
    )


    # --------------------------------------------------------
    # PERFORMANCE BAR CHART
    # --------------------------------------------------------

    comparison_long = (
        model_comparison
        .melt(
            id_vars="Model",
            var_name="Metric",
            value_name="Score"
        )
    )


    fig, ax = plt.subplots(
        figsize=(10, 6)
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
        0.6
    )


    ax.set_title(
        "Performance Comparison"
    )


    st.pyplot(fig)


    # --------------------------------------------------------
    # CONFUSION MATRICES
    # --------------------------------------------------------

    st.subheader("Confusion Matrices")


    col1, col2 = st.columns(2)


    with col1:

        st.write("### TF-IDF + SVM")

        cm_svm = confusion_matrix(
            model_results["y_test"],
            model_results["svm_predictions"],
            labels=model_results[
                "svm_model"
            ].classes_
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


        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")


        st.pyplot(fig)


    with col2:

        st.write("### Word2Vec + Random Forest")

        cm_rf = confusion_matrix(
            model_results["y_test"],
            model_results["rf_predictions"],
            labels=model_results[
                "rf_model"
            ].classes_
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


        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")


        st.pyplot(fig)


    # --------------------------------------------------------
    # ROC CURVE - SVM
    # --------------------------------------------------------

    st.subheader(
        "One-vs-Rest ROC Curves - TF-IDF + SVM"
    )


    svm_classes = model_results[
        "svm_model"
    ].classes_


    y_binary = label_binarize(
        model_results["y_test"],
        classes=svm_classes
    )


    fig, ax = plt.subplots(
        figsize=(8, 6)
    )


    for i, class_name in enumerate(
        svm_classes
    ):

        fpr, tpr, _ = roc_curve(
            y_binary[:, i],
            model_results[
                "svm_probabilities"
            ][:, i]
        )


        class_auc = auc(
            fpr,
            tpr
        )


        ax.plot(
            fpr,
            tpr,
            label=(
                f"{class_name} "
                f"(AUC={class_auc:.3f})"
            )
        )


    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier"
    )


    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.legend()


    st.pyplot(fig)


    st.subheader("Interpretation")

    st.write(
        """
        Neither model demonstrates strong sentiment discrimination.

        The exploratory analysis showed considerable vocabulary overlap
        between positive, neutral and negative feedback. Manual inspection
        also identified several feedback texts whose wording has weak or
        unclear semantic correspondence with the supplied sentiment label.

        Consequently, limited predictive performance appears to be
        influenced by the quality of the textual signal available in the
        supplied dataset rather than by class imbalance alone.
        """
    )


# ============================================================
# PAGE 6 - LIVE PREDICTION
# ============================================================

elif page == "Live Prediction":

    st.header("Live Student Feedback Prediction")

    st.write(
        """
        Enter a new student-feedback comment below.

        The selected TF-IDF + SVM model will classify the feedback
        as positive, neutral or negative.
        """
    )


    new_feedback = st.text_area(
        "Student Feedback",
        height=150,
        placeholder=(
            "Example: The lecturer explains the topics clearly "
            "and the course is very helpful."
        )
    )


    if st.button(
        "Predict Sentiment",
        type="primary"
    ):

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


            st.success(
                f"Predicted Sentiment: "
                f"{prediction.upper()}"
            )


            probability_df = pd.DataFrame({
                "Sentiment":
                    live_model.classes_,

                "Probability":
                    probabilities
            })


            probability_df[
                "Probability"
            ] = (
                probability_df[
                    "Probability"
                ] * 100
            )


            probability_df[
                "Probability"
            ] = (
                probability_df[
                    "Probability"
                ].round(2)
            )


            st.subheader(
                "Prediction Probabilities"
            )


            st.dataframe(
                probability_df,
                use_container_width=True
            )


            st.bar_chart(
                probability_df.set_index(
                    "Sentiment"
                )
            )


            st.subheader(
                "Processed Input"
            )

            st.write(
                processed
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Group 12 | Student Feedback Analytics | "
    "Text Analytics Project"
)