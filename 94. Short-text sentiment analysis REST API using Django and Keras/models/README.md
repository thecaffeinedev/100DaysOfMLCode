# sentiment_api/models

base_model.py: 
- BaseModel: parent class with save and load methods

models.py: 
- BiLSTM: buidls BiLSTM + MLP computational graph. Contains the construct method. 

preprocessor.py: 
- VocabularyBuilder: takes in list of sentences, cleans, tokenises, collects word and character frequency counts and coverts documents to sequences of word ids. Used in the DocIdTransformer class. 
- DocIdTransformer: takes in list of sentences, converts documents to sequences of word ids using VocabularyBuilder and pads sequences to the max word sequence length determined by VocabularyBuilder. Contains fit, transform, fit_transform, save and load methods. 

training.py: 
- TrainModel: batches dataset using python generator, compiles model and iterates over generator to train the model. Contains train, batch_iterator and data_iterator methods.
 
wrapper.py:
- SentimentAnalysisModel: Loads dataset internally, builds the model graph, preprocesses the dataset, trains the model and saves the weights, parameters and processor (if save_model=True). Prediction involves preprocessing a list of sentences, defining the previous computational graph and returning labels corresponding to the maximum probabilities assigned to the two sentiment classes. Contains initialize_model, fit, predict, score, save and load methods. 

utils.py: 
- load_config: loads YML file
- load_dataset: reads in tab separated .txt file
- filter_embeddings: extracts embeddings if word exists in embedding vocab
- load_glove: loads Glove embeddings from file
- pad_nested_sequences: pads character sequences within word sequences
