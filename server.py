from flask import Flask, request, render_template
from Phq9Model import Phq9Model
from sentimentModel import SentimentModel 
import os 

app = Flask(__name__)
model = SentimentModel()

@app.route('/')
def root():
    return render_template('index.html')

@app.route("/sentiment")
def sentiment():
    return render_template("sentiment.html")


@app.route('/predictSentiment', methods=['POST'])
def predictSentiment():
    message = request.form['form10']
    result, message = model.predict(message)
    return render_template('sentimentResult.html', prediction=result, message=message)


@app.route('/predict', methods=["POST"])
def predict():
    q1 = int(request.form['a1'])
    q2 = int(request.form['a2'])
    q3 = int(request.form['a3'])
    q4 = int(request.form['a4'])
    q5 = int(request.form['a5'])
    q6 = int(request.form['a6'])
    q7 = int(request.form['a7'])
    q8 = int(request.form['a8'])
    q9 = int(request.form['a9'])

    values = [q1, q2, q3, q4, q5, q6, q7, q8, q9] 
    model = Phq9Model() 
    classifier = model.naiveBayes_classifier() 
    prediction = classifier.predict([values]) 
    if prediction[0] == 0:
            result = 'Your Mental Health test result :\n No Risk'
    if prediction[0] == 1:
            result = 'Your Mental Heath test result : \n Mild Risk, Seek Help!'
    if prediction[0] == 2:
            result = 'Your Mental Health test result :\n Moderate Risk, Seek Help!'
    if prediction[0] == 3:
            result = 'Your Mental Health test result :\n Moderately severe Risk, Seek Help!'
    if prediction[0] == 4:
            result = 'Your Mental Health test result :\n Severe Risk, Seek Help!'
    return render_template("result.html", result=result) 

app.secret_key = os.urandom(12) 
app.run(debug=True)