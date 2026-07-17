import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """
    Clean and preprocess news text.
    """

    if not isinstance(text, str):
        text = str(text)

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+", "", text)

    text = text.translate(str.maketrans("", "", string.punctuation))

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


def preprocess_dataframe(df):
    """
    Clean title and text columns and create
    a combined content column.
    """

    df = df.copy()

    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")

    df["content"] = df["title"] + " " + df["text"]

    df["content"] = df["content"].apply(clean_text)

    return df


def predict_news(news, vectorizer, model):
    """
    Predict whether a news article is Fake or Real.
    """

    cleaned_news = clean_text(news)

    transformed_news = vectorizer.transform([cleaned_news])

    prediction = model.predict(transformed_news)[0]

    probability = None

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(transformed_news).max()

    return prediction, probability