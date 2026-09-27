import json
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Load dataset
with open("dataset.json", "r", encoding="utf-8") as file:
    data = json.load(file)


patterns = []
tags = []


# Prepare training data
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])


# Convert text into numerical features
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X = vectorizer.fit_transform(patterns)


# Train Logistic Regression model
model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

model.fit(X, tags)


# Save vectorizer
with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)


# Save model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)


print("======================================")
print("AI model trained successfully!")
print("======================================")
print("Model saved as: model.pkl")
print("Vectorizer saved as: vectorizer.pkl")
print("Training examples:", len(patterns))
print("Intents:", len(set(tags)))
print("======================================")