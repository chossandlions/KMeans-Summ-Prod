import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import contractions
import argparse

from utils import clean_text
from tqdm import tqdm
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from scipy.spatial import distance
from gensim.models import Word2Vec

class KMeans_Summ():
    def __init__(self, max_iters=300, n_init=10):
        self.sentence = None
        self.max_iters = max_iters
        self.n_init = n_init
    
    def embed_article(self, min_count, vector_size, article):
        clean=[]
        
        sentence = sent_tokenize(article)
        self.sentence = sentence
        
        for sen in sentence:
            clean.append(clean_text(sen))
        all_words = [i.split() for i in clean]
        model = Word2Vec(all_words, min_count=min_count, vector_size=vector_size)
        
        sent_vector=[]
        for i in clean:
            plus=0
            for j in i.split():
                plus+=model.wv[j]
            if len(i.split()) != 0:
                plus = plus/len(i.split())
            sent_vector.append(plus)
        return sent_vector
    
    def summarize_article(self, n_clusters, vector):
        kmeans = KMeans(n_clusters, init='k-means++', random_state=42, max_iter = self.max_iters, n_init = self.n_init) 
        y_kmeans = kmeans.fit_predict(vector)

        my_list=[]
        for i in range(n_clusters):
            my_dict={}

            for j in range(len(y_kmeans)):
                if y_kmeans[j]==i:
                    my_dict[j] = distance.euclidean(kmeans.cluster_centers_[i],vector[j])
            min_distance = min(my_dict.values())
            my_list.append(min(my_dict, key=my_dict.get))

        result = ""
        for i in sorted(my_list):
            result += self.sentence[i] + " "
        return result 