from flask import Flask, render_template, request, url_for
import pandas as pd
import joblib
import os
from datetime import datetime

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.lb")

# Home page - form input
@app.route('/')
def home():
    return render_template("index.html")

# About page
@app.route('/about')
def about():
    return render_template("about.html")

# Contact page
@app.route('/contact', methods=["GET", "POST"])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        print(f"[📩 Contact] {name} <{email}>: {message}")
        return render_template("contact.html", success=True)
    return render_template("contact.html")

# Predict route
@app.route('/project', methods=["POST"])
def predict():
    try:
        # Collect user inputs from form
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Prepare input for model
        input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                  columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

        # Predict
        prediction = model.predict(input_data)[0]

        # Save to history
        input_data['prediction'] = prediction
        input_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if not os.path.exists("history.csv"):
            input_data.to_csv("history.csv", index=False)
        else:
            input_data.to_csv("history.csv", mode='a', header=False, index=False)

        return render_template("project.html", prediction=prediction)

    except Exception as e:
        print("❌ Prediction failed:", e)
        return render_template("project.html", prediction=None)

# Show history
@app.route('/history')
def history():
    try:
        df = pd.read_csv("history.csv")
        return render_template("history.html", data=df.to_dict(orient="records"))
    except:
        return render_template("history.html", data=[])

# Run the server
if __name__ == '__main__':
    app.run(debug=True)
