import numpy as np
import yaml
import datetime
import os

def load_config(config_path='config.yml'):
	""" Load config file with model details """
	try:
		with open(config_path, 'r') as config_file:
			config = yaml.load(config_file)
		return config
	except yaml.YAMLError as e:
		print(e)
		return None

def generate_model_file_names(trained_model_dir, model_name):
	""" Crude model versioning """
	date_time = datetime.datetime.today().strftime('%Y_%m_%d_%H')
	weights_file =  os.path.join(trained_model_dir,date_time+"_weights.h5")
	params_file = os.path.join(trained_model_dir,date_time+"_params.json")
	preprocessor_file = os.path.join(trained_model_dir,date_time+"_preprocessor.pkl")
	return weights_file, params_file, preprocessor_file

def load_dataset(file_path):
	""" 
	Load training dataset 

	Example:
	--------
	from models.utils import load_dataset

	labels, sentences = load_dataset("data/training.txt")
	"""
	sentences, labels = [], [] 
	words, tags = [], []
	with open(file_path) as f:
		for i,line in enumerate(f):
			## Remove any trailing characters 
			line = line.rstrip()

			## Split on tab
			label, sentence = line.split("\t")
			
			## Store and return
			sentences.append(sentence)
			labels.append(label)
	return (sentences, labels)			


## TODO: Consider class to wrap embeddings and handle logic
def filter_embeddings(embeddings, vocab, embedding_dim):
	"""
	Filter word embeddings by vocab   

	Example:
	--------
	import numpy as np
	from models.utils import filter_embeddings
	
	embeddings = {"this": np.array([0]*100), "vocab": np.array([0]*100)} 
	vocab = {"this":0, "is":1, "vocab":2}

	word_embeddings = filter_embeddings(embeddings, vocab, embedding_dim=100)
	"""
	## TODO: Add useful error messages
	if not isinstance(embeddings, dict):
		return
	
	## TODO: Check embedding_dim == embeddings.shape[0]

	## Loop through vocab and obtain embeddings
	_embeddings = np.zeros([len(vocab), embedding_dim])
	for word in vocab:
		if word in embeddings:
			word_idx = vocab[word]
			_embeddings[word_idx] = embeddings[word]
	return _embeddings


## TODO: Add function to load Google vectors and FasText embeddings
def load_glove(file_path):
	""" Loads Glove vectors into np.array """
	word_vect_dict = {}
	with open(file_path) as f:
		for line in f:
			line = line.split(' ')
			word = line[0]
			gl_vector = np.array([float(val) for val in line[1:]])
			word_vect_dict[word] = gl_vector
	return word_vect_dict


def normalize_number(text):
	""" Convert numbers to 0 """
	return re.sub(r'[0-9０１２３４５６７８９]', r'0', text)


def pad_nested_sequence(sequences, dtype='int32'):
	""" 
	Pad nested sequences to the same length 
	
	Example:
	--------
	from models.utils import pad_nested_sequence

	sequences = [[[1,2,3,4], [1,2]], [[1,2,3,4,5,6,6], [1,2,3,4],[1,2]]]
	pad_nested_sequence(seqs)
	"""
	max_sent_len = 0
	max_word_len = 0
	for sentence in sequences:
		max_sent_len = max(len(sentence), max_sent_len)
		for word in sentence:
			max_word_len = max(len(word), max_word_len)
	x = np.zeros((len(sequences), max_sent_len, max_word_len)).astype(dtype)
	for i, sentence in enumerate(sequences):
		for j, word in enumerate(sentence):
			x[i, j, :len(word)] = word
	return x

