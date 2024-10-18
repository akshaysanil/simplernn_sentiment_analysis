from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st


# load imdb dataset,word index
word_index = imdb.get_word_index()

# reverse word index for access
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# load model
model = load_model('simple_rnn_imdb.h5')

# helper functions
# decode function
def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in encoded_review])

def pre_process_text(text):
    words = text.lower().split()
    encoded_review = [reverse_word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500,padding='pre')
    return padded_review

def predict_sentiment(review):
    pre_processed_review = pre_process_text(review)
    prediction = model.predict(pre_processed_review)
    sentiment = "Positive review" if prediction[0][0] > 0.5 else "Negative review"
    return sentiment, prediction[0][0]

# streamlit app
# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Find Sentiment'):
    sentiment, score = predict_sentiment(user_input)
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {score}')
else:
    st.write('Please enter a movie review.')