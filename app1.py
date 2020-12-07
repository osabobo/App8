import numpy as np
import pandas as pd
import joblib
import streamlit as st
pickle_in = open("grid2.pkl","rb")
classifier=joblib.load(pickle_in)#step 2
cv_model = open('vectorizer.pkl', 'rb')
cv = joblib.load(cv_model)
#logit=joblib.load('./logit1.pkl')
def predict(data):
    #data=[int_features]
    vect =cv.transform(data)
    #vect=vect.toarray()

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

    st.sidebar.info('This app is created to predict spam')


    st.sidebar.image(image_spam)





    st.title("Prediction of spam App")

    if add_selectbox == 'Online':

        url = st.text_input('Words(Please input  words)')

    output=""
    data= [url]
    if st.button("Predict"):
        output = predict(data)
    if output==1:
        output='ham'
    else:
        output='spam'

    st.success(' output is {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict(data)
            st.write(predictions)
if __name__ == '__main__':
    main()
