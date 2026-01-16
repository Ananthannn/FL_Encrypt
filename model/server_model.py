from re import VERBOSE
import os
import tensorflow as tf
from model_maker import ModelMaker
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

class handlerModel:
    def __init__(self, model, weights):
        self.model = model
        self.model.set_weights(weights)
        self.weights = weights

    def get_weights(self):
        return self.model.get_weights()
        
    def predict_one(self, image):
        image = tf.expand_dims(image, axis=0)
        preds = self.model.predict(image, verbose=0)
        return np.argmax(preds, axis=1)[0]   

    def predict_batch(self, image):
        return self.model.predict(image , verbose=0)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y, verbose=1)

    def summary(self):
        self.model.summary()

