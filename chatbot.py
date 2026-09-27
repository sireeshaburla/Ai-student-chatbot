import json
import pickle
import random
import os

MODEL_FILE = "model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"
DATASET_FILE = "dataset.json"


def load_files():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            "model.pkl not found. Run train_model.py first."
        )

    if not os.path.exists(VECTORIZER_FILE):
        raise FileNotFoundError(
            "vectorizer.pkl not found. Run train_model.py first."
        )

    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(
            "dataset.json not found."
        )

    with open(MODEL_FILE, "rb") as file:
        model = pickle.load(file)

    with open(VECTORIZER_FILE, "rb") as file:
        vectorizer = pickle.load(file)

    with open(DATASET_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    return model, vectorizer, data


model, vectorizer, data = load_files()


def get_response(user_message):
    user_message = user_message.strip()

    if not user_message:
        return "Please type a question."

    # Convert the user's question into TF-IDF features
    user_vector = vectorizer.transform([user_message])

    # Predict the intent
    predicted_tag = model.predict(user_vector)[0]

    # Find the response for that intent
    for intent in data["intents"]:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])

    return (
        "Sorry, I don't understand that question yet. "
        "Please try asking in another way."
    )


if __name__ == "__main__":
    print("🎓 Student AI Chatbot")
    print("Type 'quit' to exit.")

    while True:
        user_message = input("You: ")

        if user_message.lower() == "quit":
            print("Bot: Goodbye! 👋")
            break

        print("Bot:", get_response(user_message))