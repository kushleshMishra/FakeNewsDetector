"""
Fake News Detection Streamlit Application

Project: Fake News Detection Using Machine Learning
"""

import os
import joblib
import pandas as pd
import streamlit as st

import plotly.express as px

from utils import clean_text


# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# Load Model
# =====================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/fake_news_model.pkl"
    )

    vectorizer = joblib.load(
        "models/tfidf_vectorizer.pkl"
    )

    return model, vectorizer


model, vectorizer = load_model()


# =====================================================
# Load Dataset
# =====================================================

@st.cache_data
def load_dataset():

    fake = pd.read_csv(
        "dataset/Fake.csv"
    )

    true = pd.read_csv(
        "dataset/True.csv"
    )

    fake["label"] = "Fake"

    true["label"] = "Real"

    df = pd.concat(
        [fake, true],
        ignore_index=True
    )

    return df


df = load_dataset()


# =====================================================
# Custom CSS
# =====================================================

def local_css():

    st.markdown(
        """
        <style>

        .main-title {

            font-size:45px;
            font-weight:700;
            text-align:center;
            margin-bottom:20px;

        }


        .subtitle {

            font-size:20px;
            text-align:center;
            color:gray;

        }


        .card {

            padding:20px;
            border-radius:15px;
            background:#f7f7f7;
            margin:10px;

        }


        .fake {

            color:red;
            font-size:30px;
            font-weight:bold;

        }


        .real {

            color:green;
            font-size:30px;
            font-weight:bold;

        }


        </style>
        """,
        unsafe_allow_html=True
    )


local_css()


# =====================================================
# Sidebar
# =====================================================


with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/2491/2491928.png",
        width=120
    )


    st.title(
        "📰 Fake News AI"
    )


    page = st.radio(

        "Navigation",

        [
            "🏠 Home",
            "📊 Dataset Analysis",
            "🤖 Model Information",
            "🔍 Predict News"
        ]

    )


    st.divider()


    st.info(
        """
        This application uses
        Machine Learning and NLP
        to classify news articles
        as Fake or Real.
        """
    )


# =====================================================
# Header
# =====================================================


st.markdown(
    """
    <div class="main-title">
    📰 Fake News Detection System
    </div>

    <div class="subtitle">
    Machine Learning + NLP Powered Application
    </div>
    """,

    unsafe_allow_html=True
)
# =====================================================
# HOME PAGE
# =====================================================

if page == "🏠 Home":

    st.subheader(
        "Welcome to Fake News Detection System 🚀"
    )

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Total Articles",
            len(df)
        )


    with col2:

        st.metric(
            "Fake News",
            len(df[df["label"] == "Fake"])
        )


    with col3:

        st.metric(
            "Real News",
            len(df[df["label"] == "Real"])
        )


    st.divider()


    st.markdown(
        """
        ### About Project

        This project uses Natural Language Processing
        and Machine Learning algorithms to identify
        whether a news article is Fake or Real.

        ### Technologies Used

        - Python
        - Pandas
        - NLTK
        - TF-IDF
        - Machine Learning
        - Streamlit

        ### Workflow

        News Text
        →
        Text Cleaning
        →
        TF-IDF Feature Extraction
        →
        Machine Learning Model
        →
        Prediction

        """
    )


    st.success(
        "Application is ready for prediction!"
    )


# =====================================================
# DATASET ANALYSIS PAGE
# =====================================================


elif page == "📊 Dataset Analysis":


    st.subheader(
        "Dataset Analysis 📊"
    )


    st.write(
        "Dataset Preview"
    )


    st.dataframe(
        df.head(10),
        use_container_width=True
    )


    st.divider()


    col1, col2 = st.columns(2)


    with col1:

        label_count = (
            df["label"]
            .value_counts()
            .reset_index()
        )

        label_count.columns = [
            "Category",
            "Count"
        ]


        fig = px.pie(

            label_count,

            names="Category",

            values="Count",

            title="Fake vs Real News Distribution"

        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:


        subject_count = (

            df["subject"]
            .value_counts()
            .reset_index()
            .head(10)

        )


        subject_count.columns = [

            "Subject",
            "Count"

        ]


        fig2 = px.bar(

            subject_count,

            x="Subject",

            y="Count",

            title="Top News Categories"

        )


        st.plotly_chart(

            fig2,

            use_container_width=True

        )



    st.divider()


    st.subheader(
        "Dataset Information"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.write(
            "Number of Columns:",
            df.shape[1]
        )


        st.write(
            "Number of Rows:",
            df.shape[0]
        )


    with col2:

        st.write(
            "Missing Values"
        )

        st.write(
            df.isnull().sum()
        )



# =====================================================
# MODEL INFORMATION PAGE
# =====================================================


elif page == "🤖 Model Information":


    st.subheader(
        "Machine Learning Model Information 🤖"
    )


    st.markdown(

        """

        ## Algorithms Used

        ### 1. Logistic Regression

        Used for binary classification problems.

        ---

        ### 2. Multinomial Naive Bayes

        Effective algorithm for text classification.

        ---

        ### 3. Passive Aggressive Classifier

        Fast online learning algorithm.

        ---

        ### 4. Random Forest

        Ensemble learning method using multiple trees.

        ---

        ### 5. Decision Tree

        Tree-based classification algorithm.

        ---

        ### 6. Linear SVM

        Finds the best decision boundary between classes.

        """

    )


    st.divider()


    st.info(

        """
        The best performing model was automatically
        selected during training and saved using Joblib.
        """

    )
    # =====================================================
# PREDICT NEWS PAGE
# =====================================================


elif page == "🔍 Predict News":


    st.subheader(
        "🔍 Fake News Prediction"
    )


    st.write(
        """
        Enter a news article below and our
        Machine Learning model will predict
        whether it is Fake or Real.
        """
    )


    news_text = st.text_area(

        "Enter News Content",

        height=250,

        placeholder=
        "Paste your news article here..."

    )


    if st.button(
        "🚀 Predict News",
        use_container_width=True
    ):


        if news_text.strip() == "":


            st.warning(
                "Please enter news content first."
            )


        else:


            with st.spinner(
                "Analyzing news..."
            ):


                cleaned_news = clean_text(
                    news_text
                )


                vectorized_news = vectorizer.transform(
                    [cleaned_news]
                )


                prediction = model.predict(
                    vectorized_news
                )[0]


                confidence = None


                if hasattr(
                    model,
                    "predict_proba"
                ):

                    confidence = (
                        model
                        .predict_proba(
                            vectorized_news
                        )
                        .max()
                    )



            st.divider()


            if prediction == 1:


                st.markdown(

                    """
                    <div class="card">

                    <p class="real">
                    ✅ Real News
                    </p>

                    The model predicts that
                    this news article is genuine.

                    </div>

                    """,

                    unsafe_allow_html=True

                )


            else:


                st.markdown(

                    """
                    <div class="card">

                    <p class="fake">
                    ❌ Fake News
                    </p>

                    The model predicts that
                    this news article may be misleading.

                    </div>

                    """,

                    unsafe_allow_html=True

                )



            if confidence:


                st.subheader(
                    "Prediction Confidence"
                )


                confidence_percentage = (
                    confidence * 100
                )


                st.progress(
                    float(confidence)
                )


                st.write(

                    f"Confidence Score: {confidence_percentage:.2f}%"

                )



            # Download Report

            report = f"""
Fake News Detection Report
==========================

News:
{news_text}


Prediction:
{"Real News" if prediction==1 else "Fake News"}


Confidence:
{confidence_percentage:.2f}%

"""


            st.download_button(

                label="📥 Download Report",

                data=report,

                file_name="news_prediction_report.txt",

                mime="text/plain"

            )



# =====================================================
# FOOTER
# =====================================================


st.divider()


st.markdown(

    """

    <center>

    <h4>
    📰 Fake News Detection System
    </h4>

    <p>
    Built using Python | Machine Learning | NLP | Streamlit
    </p>

    </center>

    """,

    unsafe_allow_html=True

)