import re
import numpy as np 
import nltk
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

from models import utils 


## TODO: Use gensim instead for vocabulary creation
## TODO: Context-sensitive spelling correction
## TODO: More comprehensive punctuation remove

class VocabularyBuilder(object):
	""" 
	Converts vocabulary to word and char ids
	
	Example:
	--------
	from models.preprocessing import VocabularyBuilder
	
	documents = ["This is a sentence. Additional one.", "This is another sentence!"]
	dw = VocabularyBuilder()
	dw.build(documents)
	"""
	def __init__(self, lowercase=True, unk_token=True, specials=('<pad>',),
					 training=True):
		self._lowercase = lowercase
		self._unk_token = unk_token
		self._token2id = {token: i for i, token in enumerate(specials)}
		self._id2token = list(specials)
		self._training = training
		self._word_count = Counter()
		self._char_count = Counter()
		self._max_word_sequence_len = 0

	def process_document(self, document):
		""" Sentence tokenization of document """
		return nltk.sent_tokenize(document)

	def process_sentence(self, sentence):
		""" Process sentence string and store word ids"""
		if self._lowercase:
			words = nltk.word_tokenize(sentence.lower())
		else:
			words = nltk.word_tokenize(sentence)

		## Increment max sequence length
		if len(words) > self._max_word_sequence_len:
			self._max_word_sequence_len = len(words)

		## Update word counter if training 
		if self._training:
			self._word_count.update(words)
		return words

	def process_word(self, word): 
		""" Process words and store character ids """
		## Update character counter if training
		if self._training:
			chars = [l for l in word]
			self._char_count.update(chars)

	def construct_vocab_and_reverse_vocab(self):
		""" Construct token-to-id dict and id-to-token list """
		idx = len(self._token2id)
		for word in self._word_count.keys():
			self._token2id[word] = idx
			self._id2token.append(word)
			idx += 1

		## Add characters to token2id
		for char in self._char_count.keys():
			if char not in self._token2id:
				self._token2id[char] = idx
				self._id2token.append(char)
				idx += 1

		## Add unknown token to end of vocabulary
		if self._unk_token:
			unk = '<unk>'
			self._token2id[unk] = idx
			self._id2token.append(unk)
			idx += 1

	def build(self, X):
		""" Obtain word and char ids """
		for document in X:
			words = self.process_sentence(document)
			for word in words:
				self.process_word(word)
		self.construct_vocab_and_reverse_vocab()

		## Indicate training is complete
		self._training = False
	
	def token_to_id(self, token):
		""" Return index of token """
		return self._token2id.get(token, len(self._token2id)-1)

	def id_to_token(self, idx):
		""" Return token given inedx """
		return self._id2token[idx]

	def document_to_word_ids(self, document):
		""" Convert single unseen document to word ids """
		words = self.process_sentence(document)
		## If None then return unk_token
		doc_ids = [self.token_to_id(word) for word in words]
		return doc_ids

	def document_to_char_ids(self, document):
		""" Convert single unseen document to character ids """
		doc_ids = [self.token_to_id(l) for l in document]
		return doc_ids

	@property
	def vocabulary(self):
		""" Access vocabulary """
		return self._token2id

	@property
	def reverse_vocabulary(self):
		""" Access reverse vocabulary """
		return self._id2token



class DocIdTransformer(BaseEstimator, TransformerMixin):
	""" 
	Convert documents to document-id matrix 
	
	Example:
	--------
	from models.preprocessing import DocIdTransformer
	
	X = ['This is a document', 'This is another document!', "And this a third..."]
	doc_id_transformer = DocIdTransformer()
	features, Y = doc_id_transformer.fit_transform(X) 
	"""
	def __init__(self, lower=True, normalize_num=True, user_char=True, initial_vocab=None):
		self._normalize_num = normalize_num
		self._use_char = user_char
		self._vocab_builder = VocabularyBuilder(lowercase=lower)

	def fit(self, X):
		""" Get vocabulary from documents """
		## TODO: Validate input type and shape of X
		self._vocab_builder.build(X)
		return self

	def transform(self, X, Y=None):
		""" Convert documents to document ids """
		w_ids = [self._vocab_builder.document_to_word_ids(doc) for doc in X]
		features = pad_sequences(w_ids, maxlen=self._vocab_builder._max_word_sequence_len, padding='post')
		return features, [Y]

	def fit_transform(self, X):
		""" Return document-id matrix given documents """
		return self.fit(X).transform(X)

	def save(self, file_path):
		""" Save object to file_path """
		joblib.dump(self, file_path)

	@classmethod
	def load(cls, file_path):
		""" Load saved DocIdTransformer from file_path """
		doc_id_transformer = joblib.load(file_path)
		return doc_id_transformer 





























































