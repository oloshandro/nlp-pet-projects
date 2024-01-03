import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing import preprocessing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Bidirectional, Dense, Embedding, Dropout
from tensorflow.keras.models import Sequential


def preprocess_data(file_path):
    # read CSV into pandas dataframe
    reviews = pd.read_csv(file_path, encoding='utf-8')

    # preprocess text with methods defined in preprocessing.py
    data = [preprocessing(custom_comment) for custom_comment in reviews['custom_comment'].to_list()]
    labels = reviews['sentiment'].values.tolist()

    labels = [0 if label == -1 else label for label in labels]
    labels = np.array(labels)

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def prepare_data(X_train, X_test):
    # Tokenize the text and convert to sequences (raw string input into integer input suitable for a Keras Embedding layer)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)    
    vocab_size = len(tokenizer.word_index) + 1    

    train_seq = tokenizer.texts_to_sequences(X_train)    #convert each text to a sequence of integers based on the word index created during fitting. The result is a list of sequences, where each sequence corresponds to a sentence or document
    test_seq = tokenizer.texts_to_sequences(X_test)

    # padding to ensure that all sequences have the same length
    train_pad = pad_sequences(train_seq, padding='post')
    test_pad = pad_sequences(test_seq, padding='post')

    return train_pad, test_pad, vocab_size


def build_lstm_model(vocab_size, embedding_dim = 50, max_length = 400):
    lstm_model = Sequential()
    lstm_model.add(Embedding(input_dim = vocab_size, output_dim = embedding_dim))
    lstm_model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
    lstm_model.add(Bidirectional(LSTM(units=32))) 
    lstm_model.add(Dense(units=32, activation="relu"))
    lstm_model.add(Dropout(rate=0.25))
    lstm_model.add(Dense(units=1, activation="sigmoid"))

    lstm_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), #1e-3 = 0.001
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"]) 

    lstm_model.summary()
    return lstm_model


def train_lstm_model(lstm_model, train_pad, y_train, batch_size=64, epochs=5, verbose=1):
    history = lstm_model.fit(train_pad, y_train, 
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose)
    return history

if __name__ == "__main__":

    # define the path to the data
    file_path = '../datasets/annotated_data_sentiment.csv'

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(file_path)

    # Prepare data
    train_pad, test_pad, vocab_size = prepare_data(X_train, X_test)

    # Build LSTM model
    lstm_model = build_lstm_model(vocab_size)

    # Train LSTM model
    history = train_lstm_model(lstm_model, train_pad, y_train)