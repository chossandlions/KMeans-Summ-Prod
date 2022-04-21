# KMeans Text Summarization

Extractive summarization tool using KMeans clustering on Word2Vec embeddings of the input text. Each sentence vector is clustered using cosine similarity between other sentence vectors, and then the centroid of each cluster is chosen for the summary. An option for keyword extraction of the summary text as well. 

## Requirements

* Numpy
* Pandas
* Nltk
* Scikit-learn
* Gensim
* Scipy
* Matplotlib
* Contractions
* Flask (if running api.py)

After installing these dependencies, you must run two commands within the Python interpreter in your console:

```
>>> import nltk
>>> nltk.download('stopwords')
>>> nltk.download('punkt')
```

This should fully set up your environment!

## How to Use

### Locally

To run the program locally, you will need to use ```run.py```. First create a text file with the text you wish to summarize, and place it in the same directory as ```run.py```. ```run.py``` takes in two arguments:

**Parameter --file**

This parameter is for the filepath of the text file you wish to summarize. This argument is required.

**Parameter -tf**

Flag for returning keywords along with the summary using TF-IDF.

### API

To launch the api, you want to run ```api.py```. This deploys a Flask server that runs on localhost:5000 and render ```index.html``` in the templates folder. To change this to run on a server, you would need to change the host and port in ```app.run()``` on the last line of ```api.py``` as well as change the redirect in ```index.html```. 

If launched properly, you should see a bare-minimum static html page as so:
<img width="521" alt="Screen Shot 2022-04-21 at 4 26 23 PM" src="https://user-images.githubusercontent.com/103800402/164567044-b9dec2d6-3f90-418c-9ba0-ac09455f2220.png">

To use, simply upload a .txt file of suitable length and click submit. If you wish to extract keywords from the document as well, click the checkbox.
