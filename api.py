import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import contractions
import argparse
import os.path
import flask

from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from scipy.spatial import distance
from gensim.models import Word2Vec
from kmeans import KMeans_Summ
from pathlib import Path
from tfidf import TF_IDF
from utils import clean_text
from flask import Flask, request, jsonify, render_template

ALLOWED_EXTENSIONS = set(['txt'])
UPLOAD_FOLDER = '.'

def is_valid(parser, arg):
    #Checks if the filepath is valid, and if not returns an error.
    if not os.path.exists(arg):
        parser.error("The file %s does not exist, please input a valid filepath." % arg)
    else:
        return open(arg, 'r')

def check_txt(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = flask.Flask(__name__)
app.config["DEBUG"] = True
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route('/')
def form():
    return render_template('index.html')

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        tfidf = request.form.get('tfidf')
        file = request.files['file']
        if file.filename == '':
            resp = jsonify({'message': 'No file uploaded.'})
            resp.status_code = 400
            return resp
        if file and check_txt(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'temp.txt'))

            #Access file and perform KMeans
            with open('temp.txt', 'r') as f:
                text = f.read()

            print(tfidf)

            if len(text) >= 1000:
                try:
                    if tfidf:
                        corp = pd.read_csv("data/cleaned_data.csv")["Text"][:10000]
                        corp = corp.apply(clean_text).to_list()
                        tf = TF_IDF(corp)
                        res_keyword = tf.extract_keywords(text, 10)
                        summ = KMeans_Summ(400, 5)
                        vec = summ.embed_article(1, 300, text)
                        result = summ.summarize_article(3, vec)
                        return render_template('index.html', processed_text=result, keywords=res_keyword)
                    
                    else:
                        summ = KMeans_Summ(400, 5)
                        vec = summ.embed_article(1, 300, text)
                        result = summ.summarize_article(3, vec)
                        return render_template('index.html', processed_text=result)
                                
                except:
                    print("Cannot process this text file.")
                    resp = jsonify({'message': 'Cannot process this text file.'})
                    resp.status_code=400
                    return resp

            #If file is too small, return error
            elif len(text) <= 1000 and len(text) > 0:
                print("File is too small for summarization.")
                resp = jsonify({'message': 'File too small for summarization.'})
                resp.status_code=400
                return resp

            #If file is empty, return error
            elif len(text) == 0:
                print("File is empty.")
                return render_template('index.html', processed_text="")
            

        else:
            resp = jsonify({'message': 'File is not an allowed file type -- please upload a .txt file.'})
            resp.status_code = 400
            return resp

app.run()
