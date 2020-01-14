'''
https://medium.com/@shivambansal36/language-modelling-text-generation-using-lstms-deep-learning-for-nlp-ed36b224b275
full code at
https://github.com/shivam5992/language-modelling/blob/master/model.py
'''

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku
import numpy as np

data = """The cat and her kittens
They put on their mittens,
To eat a Christmas pie.
The poor little kittens
They lost their mittens,
And then they began to cry.
O mother dear, we sadly fear
We cannot go to-day,
For we have lost our mittens."
"If it be so, ye shall not go,
For ye are naughty kittens."""


tokenizer = Tokenizer()
def dataset_preparation(data):
    corpus = data.lower().split("\n")
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences,maxlen=max_sequence_len, padding='pre'))

    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)

def create_model(predictors, label, max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    model.add(Embedding(total_words, 10, input_length=input_len))
    model.add(LSTM(150))
    model.add(Dropout(0.1))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(predictors, label, epochs=100, verbose=1)



def generate_text():
    pass
