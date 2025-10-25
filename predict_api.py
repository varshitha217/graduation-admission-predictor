from flask import Flask,request,jsonify
import joblib
import numpy as np

app=Flask(__name__)
model=joblib.load('model.pkl')

@app.route('/predict',methods=['POST'])
def predict():
    d=request.json
    arr=[d['gpa'],d['entrance_score'],d['projects'],d['internships'],1 if d.get('gender','M')=='M' else 0,d.get('age',21)]
    pred=int(model.predict([arr])[0])
    prob=float(model.predict_proba([arr])[0][pred])
    return jsonify({'admitted':pred,'confidence':prob})

if __name__=='__main__':
    app.run(port=5000)
