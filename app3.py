import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
def predict(message):
    model=load_model('model2.h5')
    with open('token.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    x_1 = tokenizer.texts_to_sequences([message])
    x_1 = pad_sequences(x_1, maxlen=500)
    predictions = model.predict(x_1)[0][0]
    return predictions
from PIL import Image
image = Image.open('images.jpg')
st.image(image,use_column_width=False)
st.title('Spam Review  Analyzer')
message = st.text_area('Enter Review','Type Here ..')
if st.button('Analyze'):
     with st.spinner('Analyzing the text …'):
         prediction=predict(message)
     if prediction > 0.6:
         st.success('Positive review with {:.2f} confidence'.format(prediction))
         st.balloons()
     elif prediction <0.4:
         st.error('Negative review with {:.2f} confidence'.format(1-prediction))
     else:
         st.warning('Not sure! Try to add some more words')
