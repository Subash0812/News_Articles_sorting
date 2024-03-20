import streamlit as st
import pandas as pd
import pickle as pkl
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
# nltk.download('punkt')
# nltk.download('stopwords')

vectorizer = TfidfVectorizer()
df = pd.read_csv('tfidf.csv')
vectorizer.fit(df['processed_txt'])

with open('model.pkl', 'rb') as f:
    model = pkl.load(f)


stopwords = nltk.corpus.stopwords.words('english')
PS = PorterStemmer()


def detect_label(num):
  topic_names = {0:'Business', 1:'Entertainment', 2:'Politics', 3:'Sport', 4:'Tech'}
  return topic_names[num]


def txt_preprocess(article):
  word_tokens = word_tokenize(article)
  sw_removal = []
  filtered_article = []
  for word in word_tokens:
    if word not in stopwords:
       sw_removal.append(word)
  for i in sw_removal:
    filtered_article.append(PS.stem(i))
  return ' '.join(filtered_article)



st.title('News Articles Classification')
txt = st.text_area('Enter the article')
processed = txt_preprocess(txt)
s = vectorizer.transform([processed])
if st.button("Classify Article"):
        predicted_type = int(model.predict(s))
        label = detect_label(predicted_type)
        st.write(f"Predicted Article Type: {label}")
