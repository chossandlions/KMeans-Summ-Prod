import re
import contractions
from nltk.corpus import stopwords

stop = set(stopwords.words('english'))  

def clean_text(text):
    ret = text.lower()
    ret = ret.replace('\\', '').replace('/', '').replace('.,', '.').replace('.;,', '.') 
    ret = contractions.fix(text)
    ret = re.sub(r'\([^)]*\)', '', ret)
    ret = re.sub('"','', ret)
    ret = re.sub(r"'s\b","", ret)
    ret = re.sub("[^a-zA-Z]", " ", ret)
    
    #Remove any words shorter than 2 letters
    tokens = [w for w in ret.split() if not w in stop]
    long_words=[]
    for i in tokens:
        if len(i)>=3:                 
            long_words.append(i)   
    return (" ".join(long_words)).strip() 