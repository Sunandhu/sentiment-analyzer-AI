from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import os

app = Flask(__name__)

# Enable CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# ✅ Load model (correct path for deployment)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model", "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "model", "vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

# ✅ Homepage route (serves UI)
@app.route("/")
def home():
    return render_template("index.html")


# ✅ Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        text = data.get("text", "")

        if text.strip() == "":
            return jsonify({"error": "Empty input"}), 400

        vector = vectorizer.transform([text])

        # Confidence logic
        proba = model.predict_proba(vector)[0]
        confidence = max(proba)

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


# ✅ Render deployment config
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)