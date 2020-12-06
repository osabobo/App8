import numpy as np
import pandas as pd
import joblib
import streamlit as st
pickle_in = open("logit.pkl","rb")
classifier=joblib.load(pickle_in)#step 2
cv_model = open('vectorizer.pkl', 'rb')
cv = joblib.load(cv_model)
#logit=joblib.load('./logit1.pkl')
def predict(data):
    #data=[int_features]
    vect =cv.transform(data)

    predictions = classifier.predict(vect)[0]
    return predictions
def main ():
    from PIL import Image
    image = Image.open('images.jpg')
    image_spam = Image.open('index.jpg')

    st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to predict website spam')


    st.sidebar.image(image_spam)





    st.title("Website Prediction spam App")

    if add_selectbox == 'Online':

        url = st.text_input('URL(Please input a url  E.g linkdin.com)')

    output=""
    data= [url]
    if st.button("Predict"):
        output = predict(data)


    st.success('The output is {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict(data)
            st.write(predictions)
if __name__ == '__main__':
    main()
