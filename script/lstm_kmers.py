import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional
from keras.models import Sequential
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE

EPCOHS = 100 
BATCH_SIZE = 500 
INPUT_DIM = 66
OUTPUT_DIM = 50
HIDDEN_DIM = 60

def oversample_data(X_train, y_train, sampling_proportion=0.7):
    over = SMOTE(random_state = 42,sampling_strategy=sampling_proportion)
    X_train, y_train = over.fit_resample(X_train, y_train)
    return np.array(X_train), np.array(y_train)
    

def create_model(input_length, hidden_dim = HIDDEN_DIM, output_dim = OUTPUT_DIM, input_dim = INPUT_DIM):
    model = Sequential()
    model.add(Embedding(input_dim = input_dim, output_dim = output_dim, input_length = input_length))
    model.add(Bidirectional(LSTM(hidden_dim, return_sequences=True)))
    model.add(Dropout(0.4))
    model.add(Bidirectional(LSTM(hidden_dim)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy', 'AUC'])
    return model


def train_lstm_model(sequence_kmers, label):
    tf.keras.backend.clear_session()
    tf.random.set_seed(4262)
    X = sequence_kmers
    y = np.array(label)

    model = create_model(len(X[0])) 

    print ('Fitting lstm model...')
    class_weight_dict = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y = y)
    class_weight_dict =  {i : class_weight_dict[i] for i in range(2)}

    model.fit(X, y, batch_size=BATCH_SIZE, class_weight=class_weight_dict, epochs=EPCOHS, validation_split = 0.1, verbose = 1)
    
    print('Saving LSTM model')
    model.save('../model/lstm_kmers_sequence_model')
    return model