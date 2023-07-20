import streamlit as st
import string
import pickle
st.title("Email Spam Predictor")
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

with open('finalmodel' , 'rb') as f:
    model = pickle.load(f)
with open('vector','rb') as g:
    tfidf = pickle.load(g)
nav=st.sidebar.radio("Home",['Prediction','About the App'])


if nav =='Prediction':
    raw_text = st.text_area('Enter the Text')
    if st.button('Predict'):
        preprocessed = transform_text(raw_text)
        #st.header(preprocessed)
        vector = tfidf.transform([preprocessed])
        #st.header(vector)
        result = model.predict(vector)[0]
        if (result==0):
           st.header('Not a Spam')
        if (result==1):
            st.header('Spam')
if nav=='About the App':
    st.write('A spam email classifier is a machine learning model designed to distinguish between spam (unsolicited, often malicious emails) and legitimate (non-spam) emails. It involves data preprocessing, feature selection, and model training using different machine learning  algorithms and deep learning . The model is evaluated on a testing/validation set to measure its performance. Once trained, the classifier can automatically filter out spam emails from incoming messages.This classifier is not that accurate but is reliable.'
             'This Model might not be able to classify all of the spam emails. ')
