from flask import Flask, render_template, request
import pickle
from preprocessing import clean_text

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("Frontend.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get user input
    review = request.form["review"]

    # Clean text
    cleaned_review = clean_text(review)

    # Convert to TF-IDF (IMPORTANT: use same vectorizer from training)
    vectorized_review = vectorizer.transform([cleaned_review])

    # Prediction
    prediction = model.predict(vectorized_review)[0]

    # Probabilities
    probabilities = model.predict_proba(vectorized_review)[0]

    negative_prob = round(probabilities[0] * 100, 2)
    positive_prob = round(probabilities[1] * 100, 2)

    # Label mapping
    result = "Positive 😊" if prediction == 1 else "Negative 😞"

    return render_template(
        "Frontend.html",
        prediction_text=result,
        negative=negative_prob,
        positive=positive_prob
    )


if __name__ == "__main__":
    app.run(debug=True)