import tensorflow as tf
import pickle
import os

BASE_DIR = os.path.dirname(__file__)

# Kiểm tra xem TensorFlow có nhận GPU không
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

MODEL_PATH = os.path.join(BASE_DIR, "my_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
with open(TOKENIZER_PATH, "rb") as handle:
    tokenizer = pickle.load(handle)

def predict_sentiment(text, tokenizer, max_length=50):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length)
    
    prediction = model.predict(padded_sequences)
    return "Positive" if prediction[0] > 0.5 else "Negative"
