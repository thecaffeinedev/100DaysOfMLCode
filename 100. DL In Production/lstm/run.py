import lstm
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.losses import mean_absolute_percentage_error,mean_absolute_error,mean_squared_error,mean_squared_logarithmic_error,mean_squared_logarithmic_error
from lstm import unnormalise_windows
from io import StringIO
import io

def plot_results(predicted_data, true_data):
	fig = plt.figure(facecolor='white')
	ax = fig.add_subplot(111)
	ax.plot(true_data, label='True Data')
	plt.plot(predicted_data, label='Prediction')
	#plt.legend()
	#plt.show()
	img = io.BytesIO()
	plt.savefig(img, format='png')
	img.seek(0)
	return img

def plot_results_multiple(predicted_data, true_data, prediction_len):
	fig = plt.figure(facecolor='white')
	ax = fig.add_subplot(111)
	ax.plot(true_data, label='True Data')
	#Pad the list of predictions to shift it in the graph to it's correct start
	for i, data in enumerate(predicted_data):
		padding = [None for p in range(i * prediction_len)]
		plt.plot(padding + data, label='Prediction')
	#plt.legend()
	#plt.show()
	img = io.BytesIO()
	plt.savefig(img, format='png')
	img.seek(0)
	return img

def predict():
	global_start_time = time.time()
	epochs = 10
	seq_len = 10
	num_predict = 5

	print('> Loading data... ')

	# X_train, y_train, X_test, Y_test = lstm.load_data('sp500_2.csv', seq_len, True)
	# X_train_, y_train_, X_test_, Y_test_ = lstm.load_data('sp500_2.csv', seq_len, False)
	X_train, y_train, X_test, Y_test = lstm.load_data('ibermansa.csv', seq_len, True)
	X_train_, y_train_, X_test_, Y_test_ = lstm.load_data('ibermansa.csv', seq_len, False)

	print('> Data Loaded. Compiling...')

	model = lstm.build_model([1, seq_len, 100, 1])

	model.fit(
		X_train,
		y_train,
		batch_size=100,
		nb_epoch=epochs,
		validation_split=0.40)

	predictions2, full_predicted = lstm.predict_sequences_multiple(model, X_test, seq_len, num_predict)
	# predictions = lstm.predict_sequence_full(model, X_test, seq_len)
	predictions = lstm.predict_point_by_point(model, X_test, Y_test, batch_size=100)

	# sequence_length = seq_len + 1
	# result = []
	# for index in range(len(predictions) - sequence_length):
	#	result.append(predictions[index: index + sequence_length])
	# result = lstm.unnormalise_windows(result)
	# predictions = np.array(result)

	# result = []
	# for index in range(len(Y_test) - sequence_length):
	#	result.append(Y_test[index: index + sequence_length])
	# result = lstm.unnormalise_windows(result)
	# Y_test = np.array(result)


	# Y_test = Y_test+Y_test_.astype(np.float)
	# Y_test = Y_test.astype(np.float)[:296]
	# aux = predictions[:]+Y_test_
	# print(aux)

	# mape = mean_absolute_percentage_error(Y_test[-42:-1], np.array(predictions2)[:,0])
	# mse = mean_squared_error(Y_test[-42:-1],np.array(predictions2)[:,0])
	# mae = mean_absolute_percentage_error(Y_test[-42:-1],np.array(predictions2)[:,0])

	mape = mean_absolute_percentage_error(Y_test[-2050:-1], full_predicted[0:-1])
	mse = mean_squared_error(Y_test[-2050:-1], full_predicted[0:-1])
	mae = mean_absolute_percentage_error(Y_test[-2050:-1], full_predicted[0:-1])

	# msle = mean_squared_logarithmic_error(Y_test, predictions)

	# print(mape)

	init_op = tf.initialize_all_variables()
	# def weighted_mape_tf(Y_test,predictions):
	#tot = tf.reduce_sum(Y_test)
	#tot = tf.clip_by_value(tot, clip_value_min=1,clip_value_max=1000)
	#wmape = tf.realdiv(tf.reduce_sum(tf.abs(tf.subtract(Y_test,predictions))),tot)*100#/tot
	#return(wmape)

	# mape = weighted_mape_tf(Y_test,predictions)

	# run the graph
	with tf.Session() as sess:
		sess.run(init_op)
		print('mape -> {} '.format(sess.run(mape)))
		print('mse -> {}'.format(sess.run(mse)))
		print('mae -> {} '.format(sess.run(mae)))
	# print ('msle -> {} %'.format(sess.run(msle)))

	print('Training duration (s) : ', time.time() - global_start_time)
	print(predictions)
	im1 = plot_results(predictions, Y_test)
	im2 = plot_results(np.array(Y_test_) + np.array(predictions), Y_test_)
	im3 = plot_results_multiple(predictions2, Y_test, num_predict)
	im4 = plot_results(np.array(Y_test_)[-118:-1] + np.array(full_predicted)[-118:-1], Y_test_)
	return im1,im2,im3,im4


#Main Run Thread
if __name__=='__main__':

   #im1,im2,im3,im4 = predict()
   print('Service running...')