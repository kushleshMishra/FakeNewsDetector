"""
Fake News Detection Model Training

Author: Kush
Project: Fake News Detection Using Machine Learning
"""

import os
import warnings
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from utils import preprocess_dataframe

warnings.filterwarnings("ignore")


# ======================================================
# Load Dataset
# ======================================================

def load_dataset():
    """
    Load Fake and True news datasets.
    """

    fake_path = os.path.join("dataset", "Fake.csv")
    true_path = os.path.join("dataset", "True.csv")

    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)

    fake["label"] = 0
    true["label"] = 1

    df = pd.concat([fake, true], ignore_index=True)

    df = df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    print("=" * 60)
    print("Dataset Loaded Successfully")
    print("=" * 60)

    print("Shape :", df.shape)

    return df


# ======================================================
# Data Preprocessing
# ======================================================

def prepare_data(df):

    df = preprocess_dataframe(df)

    X = df["content"]

    y = df["label"]

    return X, y


# ======================================================
# Train Test Split
# ======================================================

def split_dataset(X, y):

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


# ======================================================
# TF-IDF
# ======================================================

def vectorize_data(
    X_train,
    X_test
):

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    X_train_vector = vectorizer.fit_transform(X_train)

    X_test_vector = vectorizer.transform(X_test)

    return (
        vectorizer,
        X_train_vector,
        X_test_vector
    )


# ======================================================
# Models
# ======================================================

def get_models():

    return {

        "Logistic Regression":
        LogisticRegression(max_iter=1000),

        "Multinomial Naive Bayes":
        MultinomialNB(),

        "Passive Aggressive":
        PassiveAggressiveClassifier(max_iter=1000),

        "Decision Tree":
        DecisionTreeClassifier(random_state=42),

        "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

        "Linear SVM":
        LinearSVC()
    }


# ======================================================
# Evaluation Function
# ======================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions
    )

    return {

        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": report

    }
# ======================================================
# Train Models
# ======================================================

def train_models(models, X_train_vector, y_train, X_test_vector, y_test):
    """
    Train all models and evaluate them.
    """

    results = []
    trained_models = {}

    print("\n" + "=" * 70)
    print("Training Machine Learning Models")
    print("=" * 70)

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train_vector, y_train)

        metrics = evaluate_model(
            model,
            X_test_vector,
            y_test
        )

        trained_models[name] = model

        results.append({
            "Model": name,
            "Accuracy": round(metrics["accuracy"], 4),
            "Precision": round(metrics["precision"], 4),
            "Recall": round(metrics["recall"], 4),
            "F1 Score": round(metrics["f1"], 4)
        })

        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1 Score : {metrics['f1']:.4f}")

    results_df = pd.DataFrame(results)

    return trained_models, results_df


# ======================================================
# Select Best Model
# ======================================================

def select_best_model(results_df, trained_models):
    """
    Select the model with the highest F1 Score.
    """

    best_row = results_df.loc[
        results_df["F1 Score"].idxmax()
    ]

    best_name = best_row["Model"]

    best_model = trained_models[best_name]

    print("\n" + "=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(f"Model    : {best_name}")
    print(f"Accuracy : {best_row['Accuracy']}")
    print(f"Precision: {best_row['Precision']}")
    print(f"Recall   : {best_row['Recall']}")
    print(f"F1 Score : {best_row['F1 Score']}")

    return best_name, best_model


# ======================================================
# Display Comparison
# ======================================================

def display_results(results_df):
    """
    Display model comparison.
    """

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.sort_values(
            by="F1 Score",
            ascending=False
        ).to_string(index=False)
    )
    # ======================================================
# Save Model
# ======================================================

def save_model(model, vectorizer):

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        model,
        "models/fake_news_model.pkl"
    )

    joblib.dump(
        vectorizer,
        "models/tfidf_vectorizer.pkl"
    )

    print("\nModels saved successfully.")


# ======================================================
# Main Function
# ======================================================

def main():

    print("=" * 70)
    print("FAKE NEWS DETECTION MODEL TRAINING")
    print("=" * 70)

    # Load data
    df = load_dataset()

    # Preprocess
    X, y = prepare_data(df)

    # Split
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Vectorize
    vectorizer, X_train_vector, X_test_vector = vectorize_data(
        X_train,
        X_test
    )

    # Models
    models = get_models()

    # Train
    trained_models, results_df = train_models(
        models,
        X_train_vector,
        y_train,
        X_test_vector,
        y_test
    )

    # Results
    display_results(results_df)

    # Best Model
    best_name, best_model = select_best_model(
        results_df,
        trained_models
    )

    # Save
    save_model(best_model, vectorizer)

    # Final Report
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    predictions = best_model.predict(X_test_vector)

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    print("\nTraining completed successfully.")
    print(f"Best Model: {best_name}")

    # Example Prediction
    sample_news = """
    Scientists announce a breakthrough in renewable
    energy technology after years of research.
    """

    sample_vector = vectorizer.transform([sample_news])

    prediction = best_model.predict(sample_vector)[0]

    print("\nSample Prediction")

    if prediction == 1:
        print("Real News")
    else:
        print("Fake News")


if __name__ == "__main__":
    main()