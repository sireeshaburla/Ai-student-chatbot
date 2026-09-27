from flask import Flask, request, render_template, jsonify
from chatbot import get_response

app = Flask(__name__)


@app.route("/", methods=["GET", "HEAD"])
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET", "HEAD"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True) or {}
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({
                "response": "Please type a question."
            }), 400

        response = get_response(user_message)

        return jsonify({
            "response": response
        }), 200

    except Exception as e:
        print("Chat error:", e)

        return jsonify({
            "response": "Sorry, I could not process your question. Please try again."
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )