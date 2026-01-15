# import necessary libraries

import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# intialization 

port_stem = PorterStemmer()
vect = TfidfVectorizer()

# Load vector and model

vector_form = pickle.load(open('Fake_News_Detection_Using_NLP/vector.pkl', 'rb'))
load_svm = pickle.load(open('Fake_News_Detection_Using_NLP/model.pkl','rb'))

# paste function here
stop_words = set(stopwords.words('english'))
def stemming(content):
    con = re.sub('[^a-zA-Z]', ' ', content)
    con1 = con.lower()
    con2 = con1.split()
    con3 = [port_stem.stem(word) for word in con2 if not word in stop_words]
    con4 = ' '.join(con3)
    return con4
    

def news_detection(news):
    news1 = stemming(news)
    input_data = [news1]
    vectorize = vector_form.transform(input_data)
    prediction = load_svm.predict(vectorize)
    return prediction



#if __name__ == '__main__':
    #st.title('Fake News Classification app ')
    #st.subheader("Input the News content below")
    #sentence = st.text_area("Enter your news content here", "",height=200)
    #predict_btt = st.button("predict")
    #if predict_btt:
        #prediction_class=news_detection(sentence)
        #print(prediction_class)
        #if prediction_class == [0]:
            #st.success('Reliable')
        #if prediction_class == [1]:
            #st.warning('Unreliable')

st.sidebar.header("Input Options")
input_option = st.sidebar.radio(
    "Select Input Options:",
    ('Enter Text', 'Upload Document')
)

st.title('📰 Fake News Detection App')

if input_option == 'Enter Text':
    st.subheader("Input the News content below")
    content = st.text_area('Enter your news content here', height=200)
    predict_btn = st.button('Predict', help="Click to Predict News", type='primary')
    if predict_btn:
        prediction = news_detection(content)
        print(prediction)
        if prediction == [0]:
            st.warning('Fake News')
        if prediction == [1]:
            st.success('True News')

elif input_option == 'Upload Document':
    st.subheader("Upload News Document")

    uploaded_file = st.file_uploader(
        "Upload a .txt file",
        type=["txt"]
    )

    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")

        if st.button("Predict", help="Click to Predict News", type='primary'):
            prediction = news_detection(text)
            if prediction == [0]:
                st.warning('Fake News')
            if prediction == [1]:
                st.success('True News')

sentiment_mapping = ["one", "two", "three", "four", "five"]
selected = st.feedback("stars")
if selected is not None:
    st.markdown(f"You selected {sentiment_mapping[selected]} star(s).")

st.write(' ')
st.markdown('This tool provides AI-based predictions and should not be considered as absolute truth.', text_alignment = 'center' )