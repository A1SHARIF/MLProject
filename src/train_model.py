from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression        
from sklearn.metrics import accuracy_score, classification_report  
import joblib
import pandas as pd
from cleaning import cleaned_text, label



df = pd.read_csv("IMDB Dataset.csv")
df['clean_review'] = df['review'].apply(cleaned_text)
df['sentiment_label'] = df['sentiment'].apply(label)
X = df['clean_review']
y = df['sentiment_label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    shuffle=True,
    stratify=y
)

vectorizer = TfidfVectorizer(max_features=5000)  # Keep top 5000 words
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)



model = LogisticRegression(max_iter=1000)  
model.fit(X_train_vectors, y_train)
y_pred = model.predict(X_test_vectors)
# Overall accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Detailed metrics
print("Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, 'sentiment_model.pkl')

# Save vectorizer
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
