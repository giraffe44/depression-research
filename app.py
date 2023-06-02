import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request

le = pickle.load(open('le', 'rb'))
cv = pickle.load(open('cv', 'rb'))
nb_model = pickle.load(open('nb_model', 'rb'))
lr_model = pickle.load(open('lr_model', 'rb'))

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('home.html')
    elif request.method == 'POST':
        user_text = request.form['user_text'].lower()
        encoded_input = cv.transform([user_text]).toarray()
        nb_prediction_probabilities = nb_model.predict_proba(encoded_input)
        nb_prediction_label = le.inverse_transform(nb_model.predict(encoded_input))
        lr_prediction_probabilities = lr_model.predict_proba(encoded_input)
        lr_prediction_label = le.inverse_transform(lr_model.predict(encoded_input))
        ensemble_prediction_probabilities = (nb_prediction_probabilities + lr_prediction_probabilities) / 2
        ensemble_prediction_label = le.inverse_transform([np.argmax(ensemble_prediction_probabilities)])
        response_page = ''
        if ensemble_prediction_label == 'not depressed': 
            response_page = 'notdepressed.html'
        elif ensemble_prediction_label == 'depressed':
            response_page = 'depressed.html'
        return render_template(response_page, user_text=user_text, 
                               label=ensemble_prediction_label[0].upper(), probability=ensemble_prediction_probabilities[0, np.argmax(ensemble_prediction_probabilities)], 
                               nb_label=nb_prediction_label[0].upper(), nb_probability=nb_prediction_probabilities[0, np.argmax(nb_prediction_probabilities)], 
                               lr_label=lr_prediction_label[0].upper(), lr_probability=lr_prediction_probabilities[0, np.argmax(lr_prediction_probabilities)])


app.run(host='0.0.0.0', port=80)

# deploy site



