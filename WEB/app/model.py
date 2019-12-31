from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np

class ModelGenerator():
    def __init__(self):
        self.session = tf.Session()
        self.graph = tf.get_default_graph()

        # Inicializacion de las variables del modelo
        self.model = None
        self.noise = None

        # Sesion para evitar problemas de multithreading
        with self.graph.as_default():
            with self.session.as_default():

                # Cargar modelo desde archivo
                self.model = load_model('./models/Edificios.1501.h5')
                
    
    def get_pred(self, noise):
        with self.graph.as_default():
            with self.session.as_default():
                return self.model.predict(noise)

    def get_noise(self):
        # De 0 - 1 -> un vector de 256 elementos
        self.noise = np.random.normal(size = (1, 256))

    def get_prediction(self):
        self.get_noise()
        gen_imgs = self.get_pred(self.noise)
        # Normalizacion del resultado
        gen_imgs = (gen_imgs + 1.) / 2.
        return gen_imgs