from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle

# load the model from disk
loaded_model=pickle.load(open('sms_spam_classifier.pkl', 'rb'))
cv=pickle.load(open('td-idf.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message =request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction =loaded_model.predict(vect)
    return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug=True)