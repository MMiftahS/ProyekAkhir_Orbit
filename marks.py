import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
#TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('Emotion_Proses.csv')
#
import re #panggil regex

def casefolding(text):
  text = text.lower()                               
  text = re.sub(r'https?://\S+|www\.\S+', '', text) 
  text = re.sub(r'[-+]?[0-9]+', '', text)           
  text = re.sub(r'[^\w\s]','', text)                
  text = text.strip()
  return text

raw_sample = data['tweet'].iloc[5]
case_folding = casefolding(raw_sample)

print('Raw data\t: ', raw_sample)
print('Case folding\t: ', case_folding)

#Text_normalize
key_norm = pd.read_csv('https://raw.githubusercontent.com/Andre480/Emotion-Text-Classification/main/kamus_singkatan.csv')
def text_normalize(teks):
  teks = ' '.join([key_norm[key_norm['singkat'] == word]['hasil'].values[0] if (key_norm['singkat'] == word).any() else word for word in teks.split()])
  teks = str.lower(teks)
  return teks

#Filtering
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

stopwords_ind = stopwords.words('indonesian')
more_stopword = ['lu','username','url','orang','ya']                    # Tambahkan kata dalam daftar stopword
stopwords_ind = stopwords_ind + more_stopword

def remove_stop_words(teks):
  clean_words = []
  teks = teks.split()
  for word in teks:
      if word not in stopwords_ind:
          clean_words.append(word)
  return " ".join(clean_words)

raw_sample = data['tweet'].iloc[5]
case_folding = casefolding(raw_sample)
stopword_removal = remove_stop_words(case_folding)

print('Raw data\t\t: ', raw_sample)
print('Case folding\t\t: ', case_folding)
print('Stopword removal\t: ', stopword_removal)

#Stemming
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Buat fungsi untuk langkah stemming bahasa Indonesia
def stemming(teks):
  teks = stemmer.stem(teks)
  return teks

# Buat fungsi untuk menggabungkan seluruh langkah text preprocessing
def text_preprocessing_process(teks):
  teks = casefolding(teks)
  teks = text_normalize(teks)
  teks = remove_stop_words(teks)
  teks = stemming(teks)
  return teks

data = pd.read_csv('Emotion_Proses.csv')

from joblib import load
#Hasil
pipeline = load("model_sentiment_naive.joblib")

data_input = input("Masukkan komentar:\n")
data_input = text_preprocessing_process(data_input)

#Load
tfidf = TfidfVectorizer


#loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(new_selected_features))
loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("selected_feature_tf-idf.pkl", "rb"))))

hasil = pipeline.predict(loaded_vec.fit_transform([data_input]))

#print("Hasil Preprocessing:\n", proses)

if(hasil=='happy'):
    s ="Senang"
elif (hasil=='anger'):
    s ="Marah"
elif (hasil=='sadness'):
    s ="sedih"
elif (hasil=='love'):
    s ="Love"
elif (hasil=='fear'):
    s ="takut"      
else:
    s ="Psikopat"
    
print("Hasil prediksi:\n", s) 
