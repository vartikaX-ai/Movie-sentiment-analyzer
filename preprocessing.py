import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

negation_words = {"not", "no", "never"}
stop_words = set(stopwords.words('english')) - negation_words

#Cleans the text by removing extrawhitespace,html tags,punctuation
def clean_text(text):
    if pd.isna(text):
        return ''
    text = text.lower()
    text = re.sub(r'<.*?>','',text)  #Remove the html tags
    text = re.sub(r'\s+',' ',text)   #Remove the extra whitespace
    text = text.translate(text.maketrans('','',string.punctuation))  #Remove the punctuation
    words = text.split()
    words = [word for word in words if word not in stop_words ] #Remove the stopwords
    words = ' '.join(words)
    return words

#Converting text into numbers using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer( max_features=50000,ngram_range=(1,2),min_df=5,max_df=0.9,sublinear_tf=True)
def preprocess_data(train_df,test_df):
    train_df['reviews'] = train_df['reviews'].apply(clean_text)
    test_df['reviews'] = test_df['reviews'].apply(clean_text)
    X_train = tf.fit_transform(train_df['reviews'])
    X_test = tf.transform(test_df['reviews'])
    return tf,X_train,X_test