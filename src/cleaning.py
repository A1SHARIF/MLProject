import re
import pandas as pd
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def cleaned_text(text):
    cleaner_review = re.sub( r'[^a-z\s]' , '', text.lower())
    words = cleaner_review.split()
    words = [w for w in words if w not in stop_words]
    cleaned_text = " ".join(words)
    return cleaned_text

def label(sentiment):
    if sentiment == 'positive' :
        return 1
    else:
        return 0
