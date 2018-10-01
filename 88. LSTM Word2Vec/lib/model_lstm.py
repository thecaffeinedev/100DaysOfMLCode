from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers.embeddings import Embedding
from keras.layers import Convolution1D, MaxPooling1D
import numpy as np


def classif_imdb(X_train, y_train, X_test, y_test, word2vec_model = None, embedding_weights = None,
            dense_dim=100, nb_epoch=3, include_cnn=False,
            nb_filter=64, filter_length=5, pool_length=4, batch_size=64,
            default_vocab_dim = 300 ):

    # if word2vec model is not parsed in,
    # We will estimate the max number of words in the dictionary
    # And then we will assign an arbitrary number of features.
    # 1. if word2vec model does not exist, but we have embedding_weights:
    #    we can derive input dim and vocab dim from embedding_weights
    #
    # 2. if word2vec model exists and embedding weights exist:
    #    we can derive input dime and vocab dim from embedding_weights
    #
    # 3. if word2vec model exist, but embedding_weights does not exist,
    #    we will prompt for embedding to be created and terminate the run
    #
    # 4. neither exists, we will create an empty embedding from scratch.

    if embedding_weights == None and word2vec_model != None:
        print("You should create embedding weights first!\n")
        return
        # if you really want to derive an empty embedding from word2vec model:
        # len(model.vocab) is number of words
        # n_symbols = len(word2vec_model.vocab)
        # vocab_dim is the feature vector length. 300 by default for each word.
        # vocab_dim = len(word2vec_model.vocab.itervalues().next())

    elif embedding_weights == None and word2vec_model == None:
        n_symbols = np.max(X_train)+1
        vocab_dim = default_vocab_dim
        embedding_weights = np.zeros((n_symbols, vocab_dim))
    else:
        n_symbols = len(embedding_weights)
        vocab_dim = len(embedding_weights[0])


    # assemble the model
    model = Sequential()  # or Graph or whatever
    # vocab_dim is basically the dimension of each embedding vector; 300 by default.
    # input_dim is number of words in vocabulary +1
    # embedding weights is the dictionary
    # change mask_zero to False
    model.add(Embedding(output_dim=vocab_dim, input_dim=n_symbols, \
                    mask_zero=False, weights=[embedding_weights], \
                    dropout=0.2))  # note you have to put embedding weights in a list by convention

    if include_cnn:
        model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))

        model.add(MaxPooling1D(pool_length=pool_length))

    # Now add LSTM layer
    model.add(LSTM(dense_dim))
    #, return_sequences=False))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=nb_epoch, batch_size=batch_size)


