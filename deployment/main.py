from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("dib_model.pkl")
@app.route("/")
def home():
    return render_template('home.html')

@app.route("/submit", methods=['post'])
def submit():
    inp_val = [float(value) for value in request.form.values()]
    res = model.predict([inp_val])
    if res[0] == 1:
        return "Oops! Lot of sugar intake! you are at risk of diabetes!"

    return "Great Sugar control! you are not diabetic!"




app.run(debug = True, host='0.0.0.0',port=7080)