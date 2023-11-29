from flask import Flask, render_template, request
#from tokenize import tokenizer
import pickle
from tokenize import tokenize

cv = pickle.load(open("models/cv.pkl", "rb"))
model = pickle.load(open("models/clf.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home(): # Retrieve the entered text
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        email = request.form.get("email-body") # Retrieve the entered text
    toknized_email = cv.transform([email])
    predictions = model.predict(toknized_email)
    if predictions == 1:
        predictions = 1
    else:
        predictions = -1
    return render_template("home.html", predictions=predictions, email=email)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)