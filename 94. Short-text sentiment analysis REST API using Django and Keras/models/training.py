import numpy as np



class TrainModel(object):
	"""
	Train model using Keras fit_generator method

	Example:
	--------
	from models.models import BiLSTM
	from models.training import TrainModel
	
	X = ['This is a document', 'This is another document!', "And this a third..."]*100
	Y = [1,0,1]*100

	## TODO: Add documentation

	"""
	def __init__(self, model, preprocessor, optimizer="adam"):
		self._model = model
		self._loss = model._loss
		self._preprocessor = preprocessor
		self._optimizer = optimizer

	def data_generator(self, X, Y, batch_size, shuffle, batches_per_epoch):
		""" Return batch generator of X and y """ 
		data_size = len(X)
		while True:
			indices = np.arange(data_size)

			## Randomly permute X indices
			if shuffle:
				indices = np.random.permutation(indices)

			## Loop through batch, index X and Y and preprocess
			for batch_num in range(batches_per_epoch):
				start_index = batch_num * batch_size
				end_index = min((batch_num + 1) * batch_size, data_size)
				x_batch = [X[j] for j in indices[start_index:end_index]]
				y_batch = [Y[j].astype('int32') for j in indices[start_index:end_index]]
				yield self._preprocessor.transform(x_batch, y_batch)

	def batch_iterator(self, X, Y, batch_size, shuffle):
		""" Return batch iterator """
		batches_per_epoch = int(int(len(X) - 1)/ batch_size) + 1
		print("Batches per epoch: %s"%batches_per_epoch)
		X_generator = self.data_generator(X, Y, batch_size, shuffle, batches_per_epoch)
		return batches_per_epoch, X_generator

	def train(self, x_train, y_train, epochs=5, batch_size=64, verbose=1,
				callbacks=None, shuffle=True):
		""" Create batch generator and train the model """
		## Get training generator
		training_data_steps, training_data_generator = self.batch_iterator(x_train, y_train, batch_size, shuffle)

		## Compile the model
		self._model.model.compile(loss=self._loss, optimizer=self._optimizer, metrics=["accuracy"])

		## Train the model
		self._model.model.fit_generator(
						generator=training_data_generator,
						steps_per_epoch=training_data_steps,
						epochs=epochs,
						callbacks=callbacks,
						verbose=verbose)












































