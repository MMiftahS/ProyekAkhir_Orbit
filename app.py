
from flask import Flask , render_template, request
import joblib
import numpy as np
import pickle
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from unittest import result

app = Flask(__name__)

pipeline = joblib.load("model/model_sentiment_naive.joblib")

tfidf = TfidfVectorizer

loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("model/selected_feature_tf-idf (2).pkl", "rb"))))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/sub", methods=['POST'])
def predict():
    if request.method=="POST":
        komen=request.form["komen"]
    hasil = pipeline.predict(loaded_vec.fit_transform([komen]))
    kode = {
        'happy': '😊Senang😊',
        'anger': '😡Marah😡',
        'sadness': '😥Sedih😥',
        'love': '😍LOVE😍',
        'fear': '😨Takut😨',
    }
    hasil = kode[str(hasil[0])]
    result = hasil
    #.py -> HTML
    return render_template("sub.html", n = result)


if __name__=="__main__":
    app.run(debug=True)