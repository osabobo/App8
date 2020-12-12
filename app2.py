import numpy as np
import pandas as pd
from keras.models import load_model
import joblib
from keras import backend as K
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
#model = load_model('model2.h5')
model=tf.keras.models.load_model('model2.h5')
cv_model = open('token.pkl', 'rb')
cv = joblib.load(cv_model)

embedding_dim = 100
max_length = 150
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
def predict(data):

    #vect =cv.transform(data)
    # padding and converting to numeric sequence
    cv1 = cv.texts_to_sequences(data)
    #cv1 = pad_sequences(cv1, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    predictions = model.predict(cv1)[0]
    return predictions
#@st.cache(allow_output_mutation=True)
def predict1(data1):

    vect =cv.texts_to_sequences(data1['v2'])
    vect = pad_sequences(vect, maxlen=max_length, padding=padding_type, truncating=trunc_type)




    predictions=model.predict(vect)

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
            output=np.around(predict(data)[0])

        if output==1.0:
            output='Nonspam'
        elif output==0.0:
            output='spam'





        st.success(' output is {}'.format(output))

    if add_selectbox == 'Batch':


        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"],encoding =None, key = 'a')




        st.title('Make sure the csv File is in the same format  as spam.csv before uploading to avoid Error')

        if file_upload is not None:
            data1 = pd.read_csv(file_upload,encoding = 'latin-1')



            predictions =np.asarray(predict1(data1))
            

            st.write(predictions)



if __name__ == '__main__':
    main()
