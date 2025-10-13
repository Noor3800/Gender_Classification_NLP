from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load model and TF-IDF vectorizer
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

def predict_gender(text):
    data = tfidf.transform([text])
    prediction = model.predict(data)[0]
    return "👨 Male" if prediction.lower() == "male" else "👩 Female"

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form["text"]
        result = predict_gender(text)
        return render_template("index.html", prediction=result, text=text)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
