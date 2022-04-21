# KMeans Text Summarization

Extractive summarization tool using KMeans clustering on Word2Vec embeddings of the input text. Each sentence vector is clustered using cosine similarity between other sentence vectors, and then the centroid of each cluster is chosen for the summary. An option for keyword extraction of the summary text as well. 

## Requirements

* Numpy
* Pandas
* Nltk
* Scikit-learn
* Gensim
* Scipy

## How to Use

To run, first create a text file with the text you wish to summarize, and place it in the same directory as run.py. run.py takes in two arguments:

**Parameter --file**

This parameter is for the filepath of the text file you wish to summarize. This argument is required.

**Parameter -tf**

Flag for returning keywords along with the summary using TF-IDF.
