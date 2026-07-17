import joblib
from utils import predict_news

model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


def predict(news_text):
    prediction, probability = predict_news(
        news_text,
        vectorizer,
        model
    )

    if prediction == 1:
        label = "Real News"
    else:
        label = "Fake News"

    return {
        "prediction": label,
        "confidence": probability
    }


if __name__ == "__main__":

    news = input("Enter News:\n")

    result = predict(news)

    print(result)