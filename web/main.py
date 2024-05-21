from flask import Flask
from flask import render_template, request
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import joblib
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import load_model

app = Flask(__name__)

kmer_to_index={(1, 1, 3): 0,(1, 3, 2): 1,(3, 2, 2): 2,(2, 2, 2): 3,(2, 2, 1): 4,(2, 1, 1): 5,(1, 1, 1): 6,(1, 1, 2): 7,(1, 2, 3): 8,(2, 3, 2): 9,(3, 2, 4): 10,(2, 4, 2): 11,(4, 2, 4): 12,(2, 4, 4): 13,
 (4, 4, 3): 14,(4, 3, 2): 15,(4, 2, 3): 16,(2, 3, 1): 17,(3, 1, 3): 18,(3, 2, 3): 19,(2, 3, 4): 20,(3, 4, 4): 21,(2, 4, 3): 22,(4, 3, 1): 23,(3, 1, 2): 24,
 (1, 2, 4): 25,(2, 1, 4): 26,(1, 4, 2): 27,(1, 3, 4): 28,(3, 4, 3): 29,(3, 1, 4): 30,(4, 2, 1): 31,(2, 1, 2): 32,(1, 2, 1): 33,(1, 2, 2): 34,(3, 2, 1): 35,(2, 1, 3): 36,
 (3, 4, 2): 37,(4, 2, 2): 38,(2, 2, 4): 39,(2, 4, 1): 40,(4, 1, 3): 41,(1, 3, 1): 42,(1, 4, 4): 43,(4, 4, 1): 44,(3, 4, 1): 45,(4, 1, 4): 46,(2, 2, 3): 47,
 (4, 4, 2): 48,(2, 3, 3): 49,(3, 3, 4): 50,(1, 4, 3): 51,(4, 3, 3): 52,(4, 1, 2): 53,(4, 4, 4): 54,(1, 3, 3): 55,(4, 1, 1): 56,(1, 1, 4): 57,(1, 4, 1): 58,
 (3, 3, 2): 59,(3, 3, 3): 60,(3, 1, 1): 61,(3, 3, 1): 62,(4, 3, 4): 63,(2, 2, 0): 64,(2, 0, 4): 65,(0, 4, 2): 66,(2, 0, 1): 67,(0, 1, 1): 68,(1, 2, 0): 69,
 (2, 0, 2): 70,(0, 2, 2): 71,(4, 2, 0): 72,(2, 0, 3): 73,(0, 3, 2): 74,(3, 1, 0): 75,(1, 0, 2): 76,(0, 2, 3): 77,(2, 1, 0): 78,(1, 0, 3): 79, (0, 1, 2): 80,
 (4, 1, 0): 81,(0, 2, 1): 82,(0, 1, 3): 83,(1, 1, 0): 84,(0, 3, 1): 85,(1, 0, 4): 86,(0, 4, 1): 87,(1, 4, 0): 88,(4, 0, 4): 89,(0, 4, 3): 90,(4, 4, 0): 91,
 (4, 0, 2): 92,(0, 2, 4): 93,(4, 3, 0): 94,(3, 0, 1): 95,(2, 3, 0): 96,(3, 0, 4): 97,(1, 0, 1): 98,(0, 1, 4): 99,(1, 3, 0): 100,(3, 0, 2): 101,(3, 2, 0): 102,
 (0, 4, 4): 103,(3, 0, 3): 104,(2, 4, 0): 105,(4, 0, 3): 106,(0, 3, 3): 107,(3, 4, 0): 108,(3, 3, 0): 109,(4, 0, 1): 110,(0, 3, 0): 111,(3, 0, 0): 112,
 (0, 0, 2): 113,(1, 0, 0): 114,(0, 0, 4): 115,(0, 4, 0): 116,(0, 3, 4): 117,(0, 2, 0): 118,(4, 0, 0): 119,(0, 0, 0): 120,(2, 0, 0): 121,(0, 0, 1): 122,
 (0, 0, 3): 123,(0, 1, 0): 124}

def sequence_to_vector(sequence_kmer_counts):
    vector = np.zeros(125)
    for kmer, count in sequence_kmer_counts.items():
        kmer_index = kmer_to_index.get(kmer, None)
        if kmer_index is not None:
            vector[kmer_index] = count
    return vector

def count_kmers(sequence, k):
    kmer_counts = {}
    for i in range(len(sequence) - k + 1):
        kmer = tuple(sequence[i:i+k]) 
        kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1
    return kmer_counts

def integer_encoding(seq):
    base_dict = {'A': 1, 'T': 2, 'C': 3, 'G': 4, 'N': 0}
    return [base_dict.get(base, 0) for base in seq]

def get_disease(y):
    c = np.argmax(y)
    dict = {0: 'SARS-CoV-2',1: 'SARS-CoV-1', 2: 'Ebola', 3: 'Dengue', 4: 'Influenza', 5: 'MERS-CoV'}
    return dict[c]

@app.route('/')
def fun1():
    return render_template('/index.html')

@app.route('/', methods = ['POST'])
def fun2():
    seq = request.form['txtarea1']
    seq = integer_encoding(seq)
    sequence_kmer_counts = count_kmers(seq,3)
    final_vector = sequence_to_vector(sequence_kmer_counts)
    final_vector = final_vector.T
    model = load_model('trained_model_new.h5')
    scaler = joblib.load('scaler_new.joblib')
    final_vector = final_vector.reshape(1, -1)
    scaled_final_vector = scaler.transform(final_vector)
    y = model.predict(scaled_final_vector)

    disease = get_disease(y)

    return render_template('index.html', disease=disease)

if __name__=='__main__': 
   app.run() 