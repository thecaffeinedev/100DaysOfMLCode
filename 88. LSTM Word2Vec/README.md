# Use word2vec + LSTM for the "Bag of Words meets Bag of Popcorn" challenge

I have yet to find a nice tutorial on LSTM + word2vec embedding using keras. You would assume there are tons of them out there, given how popular the combination is for sentiment analysis. [The kera blog](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html) came close for this purpose, but uses GLOVE embedding instead. 

In this repo, check out lstm_word2vec.ipynb, where I show how to do the following: 
1. Create word2vec embedding and add it as the underlying weights to LSTM training 
2. On top of 1, add a convolution layer. 
3. Back to the basics: How does the performance of training the weights from scratch compare to using embeddings? 
4. Yet another step back to the basics: How does RF compare to with LSTM, when both of them adapt word2vec features? 

I have created a simple library to support the notebook. (Some of the text processing code was lifted from the [Kaggle competition tutorial](https://www.kaggle.com/c/word2vec-nlp-tutorial).) The one you may find the most useful is lib/model_lstm.py, where details are taken care of for creating embeddings(*create_embedding_weights*), and for parsing it into the keras LSTM implementation(I created a simple wrapper called *classif_lstm*). If you don't care much for ipython notebook, you can also run the entire process directly through __main__.py.  

Things that I didn't cover in this tutorial:
1. lemmatization and stemming when processing the raw data.
2. play with the embedding and add arbitrary features. 
3. parameter tuning of LSTM/CNN.
4. parameter tuning of RF. 

Feel free to go beyond this tutorial and experiment with different settings, and see how you can make it better... :) 

Anyways, hope you have fun with BOW/BOP and beyond! Please leave me a comment if you find this tutorial useful, or if there's something you'd like to see improve on. 

Note: The working version of this code employs gensim version 0.12.4. Some tweaking may be needed if you are using a later version. 

Credit: https://github.com/rantsandruse/lstm_word2vec