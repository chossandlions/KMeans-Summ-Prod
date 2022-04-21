import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import contractions
import argparse
import os.path

from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from scipy.spatial import distance
from gensim.models import Word2Vec
from kmeans import KMeans_Summ
from pathlib import Path
from tfidf import TF_IDF
from utils import clean_text

def is_valid(parser, arg):
    #Checks if the filepath is valid, and if not returns an error.
    if not os.path.exists(arg):
        parser.error("The file %s does not exist, please input a valid filepath." % arg)
    else:
        return open(arg, 'r')

#Load the parser arguments from console
parser = argparse.ArgumentParser()
parser.add_argument('--file', dest="filepath", required=True, help="Insert filepath to .txt file", type=lambda x: is_valid(parser, x))
parser.add_argument('--tf', dest='tfidf', default=False, action="store_true")
args = parser.parse_args()

#Load text file into text
with open(args.filepath.name, 'r') as f:
    text = f.read()

#Logic for executing KMeans
if len(text) >= 1000:
    try:
        summ = KMeans_Summ(400, 5)
        vec = summ.embed_article(1, 300, text)
        result = summ.summarize_article(3, vec)
        print(result)
                    
    except:
        print("Cannot process this text file.")

#If file is too small, return error
elif len(text) <= 1000 and len(text) > 0:
    print("File is too small for summarization.")

#If file is empty, return error
elif len(text) == 0:
    print("File is empty.")

if args.tfidf:
    #print("Summary keywords: ")
    corp = pd.read_csv("data/cleaned_data.csv")["Text"][:10000]
    corp = corp.apply(clean_text).to_list()
    tf = TF_IDF(corp)
    res_keyword = tf.extract_keywords(result, 10)
    #print("\n")
    #print(res_keyword)

