#TODO: Import all the libraries and modules needed.
import os
import flask
import numpy as np
import pickle as pck
from flask import Flask, render_template, request

#TODO: Create instance of the class.
app = Flask(__name__)
#loaded_model = pck.load(open("predicton_model.pkl", "rb"))

#TODO: Tell flask the url that will trigger the function index()
@app.route("/")
@app.route("/income_prediction")
def index():
    return flask.render_template("income_prediction.html")

#TODO: The prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 12)
    loaded_model = pck.load(open("prediction_model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route("/prediction_result", methods=["POST"])
def result():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)

        if int(result) == 1:
            prediction = "Income more than 50K"
        else:
            prediction = "Income less that 50K"

        return render_template("prediction_result.html", prediction=prediction)

if __name__ =="__main__":
    app.run(debug = True)