import re
from nltk.corpus import stopwords
from cleaning import cleaned_text
import joblib

# Load saved model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')


def predict_sentiment(review):
    
    # Clean the review
    cleaned_review  = cleaned_text(review)
    
    # Convert review to numeric vector using the loaded vectorizer
    X_new = vectorizer.transform([cleaned_review])  # Note: transform, not fit_transform
    
    # Make prediction
    prediction = model.predict(X_new)
    
    # Return readable output
    return "Positive" if prediction[0] == 1 else "Negative"


if __name__ == "__main__":
    

    reviews = [
        "This movie was absolutely fantastic, I loved the acting and the story!",
        "Terrible movie. Waste of time. The acting was awful.",
        "It was okay, some parts were good but overall not that interesting."
    ]

    for r in reviews:
        print(f"Review: {r}")
        print("Predicted Sentiment:", predict_sentiment(r))
        print("-" * 50)
