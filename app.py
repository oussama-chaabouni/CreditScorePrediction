import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import requests
import json
app = Flask(__name__)
model = pickle.load(open('finalized_model_score.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/data',methods=['GET'])
def data():

    r = requests.get('http://localhost:8082/CROTUN/loan/score?age=40&sex=male&housing=OWN&savingAccount=3000&capital=4000&mois=30&purpose=Car')
    
    
    return jsonify(r.json())

@app.route('/predict/<id>',methods=['GET'])
def predict(id):

    r = requests.get('http://localhost:8082/CROTUN/loan/predict-score/'+id)
    
    
    r2=r.json()
 
    
    
    #r2=r2[0]['customerLoans']
    df3=pd.DataFrame()
    df3["Age"]=[r2['age']]
    df3["Sex"]=[r2['sex']]
    df3["Job"]=[r2['age']]
    df3["Housing"]=[r2['housing']]
    df3["Saving accounts"]=[r2['savingAccount']]
    df3["Credit amount"]=[r2['amount']]
    df3["Duration"]=[r2['mois']]
    df3["Purpose"]=[r2['purpose']]
    
    
    prediction =model.predict(df3)
    output=round(prediction[0][0],2)
   # print(output)
    
    return jsonify(output)


    #return data

if __name__ == "__main__":
    app.run(debug=True)