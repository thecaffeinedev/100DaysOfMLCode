from lib import data_split, features_word2vec, model_lstm, model_randomforest
import pandas as pd
import os

if __name__ == '__main__':

    # Read data
    # Use the kaggle Bag of words vs Bag of popcorn data:
    # https://www.kaggle.com/c/word2vec-nlp-tutorial/data
    data = pd.read_csv("./data/labeledTrainData.tsv", header=0,
                       delimiter="\t", quoting=3, encoding="utf-8")

    data2 = pd.read_csv("./data/unlabeledTrainData.tsv", header=0,
                              delimiter="\t", quoting=3, encoding="utf-8")

    # data2 and data are combined to train word2vec model
    data2 = data.append(data2)

    # Construct word2vec model
    model_path = "./model/300features_40minwords_10context"

    if not os.path.isfile(model_path):
        model = features_word2vec.get_word2vec_model(data2, "review", num_features=300, downsampling=1e-3, model_name=model_path)
    else:
        # After model is created, we can load it as an existing file
        model = features_word2vec.load_word2vec_model(model_name=model_path)

    # Create word embeddings, which is essentially a dictionary
    # that maps indices to features
    embedding_weights = features_word2vec.create_embedding_weights(model)

    # Map words to indices
    features = features_word2vec.get_indices_word2vec(data, "review", model, maxLength=500,
                             writeIndexFileName="./model/imdb_indices.pickle", padLeft=True )
    y = data["sentiment"]
    X_train, y_train, X_test, y_test = data_split.train_test_split_shuffle(y, features, test_size=0.1)

    # Vanilla LSTM
    model_lstm.classif_imdb( X_train, y_train, X_test, y_test, embedding_weights = embedding_weights, dense_dim = 256, nb_epoch = 3 )

    # Compare with LSTM + CNN
    #model_lstm.classif_imdb(X_train, y_train, X_test, y_test, embedding_weights=embedding_weights, dense_dim=256,
    #                        nb_epoch=3, include_cnn = True)

    # Compare with LSTM without embedding
    #model_lstm.classif_imdb(X_train, y_train, X_test, y_test, embedding_weights=None, dense_dim=256,
    #                        nb_epoch=3, include_cnn=True)

    # Compare with RF
    # features_avg_word2vec = features_word2vec.get_avgfeatures_word2vec(data, "review", model)
    # X_train, y_train, X_test, y_test = data_split.train_test_split_shuffle(y, features_avg_word2vec, test_size=0.1)
    # model_randomforest.classif(X_train, y_train, X_test, y_test)
