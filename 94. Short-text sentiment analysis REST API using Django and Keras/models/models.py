import keras.backend as K 
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout, Lambda, Activation, Reshape, Flatten, TimeDistributed
from keras.layers.merge import Concatenate
from keras.models import Model

from models.base_model import BaseModel



class BiLSTM(BaseModel):
	""" 
	Word embeddings + Bidirectional LSTM for sentiment prediction 
	
	Example:
	--------
	from models.models import BiLSTM
	
	## TODO: Add documentation

	"""
	def __init__(self, n_labels, word_vocab_size, max_word_seq_len, word_embedding_dim=100, word_lstm_size=100, 
			fc_dim=100, fc_activation='tanh', fc_n_layers=2, dropout=0.5, embeddings=None, 
			loss = 'binary_crossentropy'):
		super(BiLSTM).__init__()
		self._n_labels = n_labels
		self._word_vocab_size = word_vocab_size
		self._max_word_seq_len = max_word_seq_len
		self._word_embedding_dim = word_embedding_dim
		self._word_lstm_size = word_lstm_size
		self._fc_dim = fc_dim
		self._fc_activation = fc_activation
		self._fc_n_layers = fc_n_layers
		self._dropout = dropout
		self._embeddings = embeddings
		self._loss = loss
		self.validate_parameters()

	def validate_parameters(self):
		""" TODO: Check parameters are valid """
		pass

	def construct(self):
		""" Build Keras Computational Graph """
		word_ids = Input(shape=(self._max_word_seq_len,), dtype='int32')
		
		## Create embedding layer if not provided
		if self._embeddings is None:
			word_embeddings = Embedding(input_dim=self._word_vocab_size,
									output_dim=self._word_embedding_dim,
									mask_zero=False, ## TODO: Flatten does not support masking?
									input_length=self._max_word_seq_len)(word_ids)

		## Load pre-trained embeddings 
		else:
			word_embeddings = Embedding(input_dim=self._embeddings.shape[0],
									output_dim=self._embeddings.shape[1],
									mask_zero=False, ## TODO: Flatten does not support masking?
									weights=[self._embeddings])(word_ids)

		## Build Bidirectional LSTM layer with Fully Connected layers on top
		word_embeddings = Dropout(self._dropout)(word_embeddings)
		h = Bidirectional(LSTM(units=self._word_lstm_size, return_sequences=True))(word_embeddings)
		h = Dropout(self._dropout)(h)
		h = Flatten()(h)
		for fc_layers in range(self._fc_n_layers):
			h = Dense(self._fc_dim, activation=self._fc_activation)(h)
		y_pred = Dense(self._n_labels, activation="softmax")(h)
		self.model = Model(inputs=[word_ids], outputs=y_pred)

	def __repr__(self):
		""" Ugly printing of class variables """
		return "BiLSTM: %s"%self.__dict__




































