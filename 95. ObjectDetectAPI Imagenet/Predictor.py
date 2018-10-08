import keras
import numpy as np
from keras.applications import inception_v3
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions

def predictor(filename):
    # Load the Model
    """
    Cant load model outside as flask generates error. Let it be here!!!
    """
    inception_model = inception_v3.InceptionV3(weights='imagenet')
    # The initial size of image is 224, 224. Later, I have to make it into a batch
    original = load_img(filename, target_size=(224, 224))
    # converting the image loaded to an array to feed to the system
    numpy_image = img_to_array(original)
    # Creating a batch i.e., an array of size [1, 224, 224]
    image_batch = np.expand_dims(numpy_image, axis=0)
    # Inception has it's own method of processing on the image
    processed_image = inception_v3.preprocess_input(image_batch.copy())
    # Predict the objects in the image
    predictions = inception_model.predict(processed_image)
    # get the labels
    label = decode_predictions(predictions)
    # Make labels into the format I want
    predictions = ""
    for i in label[0]:
        predictions += "name: "+i[1]+" "+"- confidence: "+str((i[2]*100))+"% <br>"
    return predictions

    
