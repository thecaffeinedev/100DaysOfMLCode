import os
import datetime
from keras.utils import to_categorical
import tensorflow as tf

from models.utils import load_dataset, generate_model_file_names
from models.preprocessing import DocIdTransformer
from models.models import BiLSTM
from models.training import TrainModel

class SentimentAnalysisModel(object):
	""" 
	Wrapper for sentiment dataset and model

	## TODO: Add documentation

	Example:
	--------
	## Train and save
	from models.wrapper import SentimentAnalysisModel	
	sentiment_model = SentimentAnalysisModel(save_model=True)
	sentiment_model.fit()

	## Load
	from models.wrapper import SentimentAnalysisModel
	weights_file = "trained_models/2018_06_15_10_weights.h5"
	params_file = "trained_models/2018_06_15_10_params.json"
	preprocessor_file = "trained_models/2018_06_15_10_preprocessor.pkl" 
	sentiment_model = SentimentAnalysisModel.load(weights_file,params_file,preprocessor_file)

	## Predict 
	sentiment_model.predict(["this is really bad documentation"])
	"""
	def __init__(self, dataset_path="data/training.txt", model_name='bilstm', word_embedding_dim=100, word_lstm_size=100,
				 fc_dim=100, fc_activation='tanh', fc_n_layers=2, dropout=0.5, embeddings=None, 
				 loss = 'binary_crossentropy', optimizer="adam", shuffle=True, batch_size=64, epochs=4,
				 verbose=1, callbacks = None, save_model=False, trained_model_dir = "trained_models/"):
		self._dataset_path = dataset_path
		self._model_name = model_name
		self._word_embedding_dim = word_embedding_dim
		self._word_lstm_size = word_lstm_size
		self._fc_dim = fc_dim
		self._fc_activation = fc_activation
		self._fc_n_layers = fc_n_layers
		self._dropout = dropout
		self._embeddings = embeddings
		self._loss = loss
		self._optimizer = optimizer
		self._shuffle = shuffle
		self._epochs = epochs
		self._batch_size = batch_size
		self._verbose = verbose
		self._callbacks = callbacks
		self._save_model = save_model
		self._trained_model_dir = trained_model_dir 
		self._doc_id_transformer = DocIdTransformer()
		self.validate_parameters()

	def validate_parameters(self):
		""" Check that parameters are valid """
		self._valid_models = ['bilstm']
		assert self._model_name.lower() in self._valid_models, "InvalidModelName: %s. Model can be %s"%(self._model_name, self._valid_models)

	def get_dataset(self):
		""" Load dataset internally for now """
		return load_dataset(self._dataset_path)

	def initialize_model(self):
		""" Initialize BiLSTM model """
		model = BiLSTM(n_labels = 2, ## TODO: Hard code for now, config file later...
						word_vocab_size = len(self._doc_id_transformer._vocab_builder.vocabulary),
						max_word_seq_len = self._doc_id_transformer._vocab_builder._max_word_sequence_len,
						word_embedding_dim = self._word_embedding_dim, 
						word_lstm_size = self._word_lstm_size,
			 			fc_dim = self._fc_dim, 
			 			fc_activation = self._fc_activation, 
			 			fc_n_layers = self._fc_n_layers, 
			 			dropout = self._dropout, 
			 			embeddings = self._embeddings, 
				 		loss = self._loss)
		model.construct()
		return model

	def fit(self):
		""" Fit model to internal training data """
		X, Y = self.get_dataset()

		## Prepare X and Y
		self._doc_id_transformer.fit(X)
		Y = to_categorical(Y)

		## Get model
		bilstm_model = self.initialize_model()

		## Train model
		tm = TrainModel(model=bilstm_model, 
						preprocessor=self._doc_id_transformer, 
						optimizer=self._optimizer)

		tm.train(x_train = X, 
				y_train = Y, 
				epochs=self._epochs, 
				batch_size=self._batch_size, 
				verbose=self._verbose,
				callbacks=self._callbacks, 
				shuffle=self._shuffle)

		self.model = tm._model

		if self._save_model:
			weights_file, params_file, preprocessor_file = generate_model_file_names(self._trained_model_dir, self._model_name)
			self.save(weights_file, params_file, preprocessor_file)
			print("weights, params and preprocessor saved")

	def predict(self, sentence):
		""" Predict sentiment class given list of sentences """
		## TODO: validate input is text
		X_features, _ = self._doc_id_transformer.transform(sentence)

		with self._graph.as_default():
			y = self.model.model.predict(X_features)

		## Take max confidence to be class label
		## TODO: consider low confidence scores as Neutral
		labels = y.argmax(axis=-1)
		return ["Positive" if l==1 else "Negative" for l in labels]

	def score(self, X_test, Y_test):
		pass

	def save(self, weights_file, params_file, preprocessor_file):
		""" Save model weight and params as well as processor """
		self._doc_id_transformer.save(preprocessor_file)
		self.model.save(weights_file, params_file)

	@classmethod
	def load(cls, weights_file, params_file, preprocessor_file):
		""" Load files to reinstantiate trained SentimentAnalysisModel """
		self = cls()
		self._doc_id_transformer = DocIdTransformer.load(preprocessor_file)
		self.model = BiLSTM.load(weights_file, params_file)
		self._graph = tf.get_default_graph()
		return self



























































