import numpy as np
import pandas as pd
import joblib
from io import StringIO
import streamlit as st

pickle_in = open("grid2.pkl", "rb")
classifier = joblib.load(pickle_in)
cv_model = open('vectorizer.pkl', 'rb')
cv = joblib.load(cv_model)


def predict(data):
    vect = cv.transform(data)
    predictions = classifier.predict(vect)[0]
    return predictions


def predict1(data1):
    vect = cv.transform(data1['v2'])
    vect = vect.toarray()
    predictions = classifier.predict(vect)
    return predictions


def main():
    from PIL import Image
    image = Image.open('images.jpg')
    image_spam = Image.open('index.jpg')

    st.image(image, use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?", ("Online", "Batch"))

    st.sidebar.info('This app checks if an SMS is a spam or not')
    st.sidebar.image(image_spam)

    st.title("Spam SMS Detection")

    if add_selectbox == 'Online':
        url = st.text_input('Input the SMS')
        output = ""
        data = [url]

        if st.button("Predict"):
            output = predict(data)
        if output == 1:
            output = 'This SMS is not a spam'
        if output == 0:
            output = 'This SMS is a spam'

        st.success(output)

    if add_selectbox == 'Batch':
        st.title('Make sure the CSV file is in the same format as spam.csv before uploading to avoid errors')
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", encoding="utf-8")

        if uploaded_file is not None:
            data1 = pd.read_csv(uploaded_file, encoding='latin-1')
            predictions = predict1(data1)
            st.write(predictions)


if __name__ == '__main__':
    main()

