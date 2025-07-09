from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("titanic_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    pclass = int(request.form["Pclass"])
    sex = int(request.form["Sex"])
    age = float(request.form["Age"])
    sibsp = int(request.form["SibSp"])
    parch = int(request.form["Parch"])
    fare = float(request.form["Fare"])
    embarked = int(request.form["Embarked"])

    features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    prediction = model.predict(features)[0]
    result = "Survived" if prediction == 1 else "Did Not Survive"

    return render_template("index.html", prediction_text=f"Prediction: {result}")

if __name__ == "__main__":
    app.run(debug=True)
