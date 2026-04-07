from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)

# Enable CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# Load model and vectorizer
model = pickle.load(open("../model/model.pkl", "rb"))
vectorizer = pickle.load(open("../model/vectorizer.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Get input text
        text = data.get("text", "")

        if text.strip() == "":
            return jsonify({"error": "Empty input"}), 400

        # Convert text to vector
        vector = vectorizer.transform([text])

        # Get prediction probabilities
        proba = model.predict_proba(vector)[0]
        confidence = max(proba)

        # Apply neutral logic
        if confidence < 0.6:
            result = "neutral"
        else:
            result = model.predict(vector)[0]

        return jsonify({
            "sentiment": result,
            "confidence": float(confidence)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)