from flask import Flask, request , jsonify
from model import xgbc_train
import json
import joblib as jbl 
import numpy as np 
import os 

app = Flask(__name__)

@app.route("/mainpage", methods=["GET"])
def main_page():
    return "<H3> Pankaj Ghared VNIT MT23AAI037s"

@app.route("/predict", methods=["POST"])
def predictme():
    col = ["Iris Setosa", "Iris Versicolor", "Iris Virginica"]
    payload = request.get_json()
    data_user = [payload["Id"], payload["SepalLengthCm"], payload["SepalWidthCm"], payload["PetalLengthCm"], payload["PetalWidthCm"]]
    data_user_array = np.array(data_user).reshape(1,-1)
    # print(data_user_array)
    model = jbl.load("./model/xgbmodel.pkl")
    pred = model.predict(data_user_array)
    # print(pred)
    return jsonify({"answer" : col[pred[0]]})

@app.route("/train_model", methods=["GET"])
def trainme():
    if os.path.exists("./data_update/Iris_new.csv") :
        out = xgbc_train()
        return "<H3> ########### new model traind ##########"
    return "<H3> ########### NO NEED of new model traind ##########"

if __name__ == "__main__":
    app.run()
