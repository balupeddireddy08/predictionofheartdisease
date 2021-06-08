from flask import Flask, request, jsonify, render_template
import pickle
import numpy as numpy
app = Flask(__name__,template_folder='templates')
model_gb = pickle.load(open('heart_mod_algo_ada.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    Name=str(request.form.get('Name'))
    Email=str(request.form.get('Email'))
    Phone=str(request.form.get('Phone'))
    gender=str(request.form.get('M/F')).lower()
    age=float(request.form.get('Age'))
    cp=float(request.form.get('cp'))
    trestbps=float(request.form.get('trestbps'))
    chol=float(request.form.get('chol'))
    fbs=float(request.form.get('fbs'))
    restecg=float(request.form.get('restecg'))
    thalach=float(request.form.get('thalach'))
    exang = float(request.form.get('exang'))
    oldpeak = float(request.form.get('oldpeak'))
    slope = float(request.form.get('slope'))
    ca = float(request.form.get('ca'))
    thal = float(request.form.get('thal'))
    m_f=-1
    if gender=='m' or gender=='male':
        m_f=1
    else:
        m_f=0
    inputs = [[	age	,m_f,	cp,	trestbps,	chol,	fbs,	restecg,	thalach,	exang,	oldpeak,	slope,	ca,	thal]]
    output_gb=model_gb.predict(inputs)[0]
    
    if output_gb==1:
        result='Heart Disease Predicted'
    else:
        result='Heart Disease Not Predicted'

    return render_template('index.html', prediction_text='Your Result: '+result)
if __name__ == '__main__':
    app.run(port = 5000, debug=False)
