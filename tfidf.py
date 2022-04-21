from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords

class TF_IDF():
    def __init__(self, corpus):
        self.text = corpus
        self.stopwords = set(stopwords.words("english"))
        self.cv = CountVectorizer(max_df=0.85, stop_words=self.stopwords)
        self.wordcount = self.cv.fit_transform(corpus)
    
        self.transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        self.transformer.fit(self.wordcount)
    
    def sort_vals(self, matrix):
        tuples = zip(matrix.col, matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
    
    def extract_top_k(self, feature_names, items, k=10):
        items = items[:k]

        scores = []
        features = []

        for idx, score in items:
            scores.append(round(score, 3))
            features.append(feature_names[idx])
        
        results = {}
        for idx in range(len(features)):
            results[features[idx]] = scores[idx]
        
        return results

    def extract_keywords(self, doc, k=10):
        feature_names = self.cv.get_feature_names_out()
        tf_idf_vector = self.transformer.transform(self.cv.transform([doc]))

        sort_items = self.sort_vals(tf_idf_vector.tocoo())
        keywords = self.extract_top_k(feature_names, sort_items, k)

        #print("\nDocument")
        #print(doc)
        print("\nKeywords:")
        for k in keywords:
            print(k, keywords[k])
        return keywords


