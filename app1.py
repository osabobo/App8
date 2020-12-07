import numpy as np
import pandas as pd
import joblib
import streamlit as st
pickle_in = open("grid2.pkl","rb")
classifier=joblib.load(pickle_in)#step 2
cv_model = open('vectorizer.pkl', 'rb')
cv = joblib.load(cv_model)

def predict(data):

    vect =cv.transform(data)


    predictions = classifier.predict(vect)[0]
    return predictions
def predict1(data1):

    vect =cv.transform(data1['v2'])
    vect=vect.toarray()

    predictions = classifier.predict(vect)
    return predictions
def main ():
    from PIL import Image
    image = Image.open('images.jpg')
    image_spam = Image.open('index.jpg')

    st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to predict spam and nonspam')


    st.sidebar.image(image_spam)





    st.title("Spam SMS Detection")

    if add_selectbox == 'Online':
        url = st.text_input('Input the SMS')

        output=""
        data=[url]


        if st.button("Predict"):
            output = predict(data)
        if output==1:
            output='Nonspam'
        if output==0:
            output='spam'





        st.success(' output is {}'.format(output))

    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        st.title('Make sure the csv File is in the same format  as spam.csv before uploading to avoid Error')
        if file_upload is not None:
            data1 = pd.read_csv(file_upload,  encoding = 'latin-1')


            predictions = predict1(data1)

            st.write(predictions)
if __name__ == '__main__':
    main()
