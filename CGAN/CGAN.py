import math
import os
from datetime import datetime

import keras as keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import (BatchNormalization, Conv2D, Conv2DTranspose, Dense,
                          Dropout, Flatten, LeakyReLU, Reshape)
from keras.models import Sequential
from keras.optimizers import Adam
from numpy import ones, zeros
from numpy.random import randint, randn
from tqdm import tqdm

from dataset import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

class CGAN:
    def __init__(self, epochs, batch_size, save_interval, log_dir):

        self.run_time = datetime.now().strftime("%Y%m%d %H%M%S")
        self.result_path = f"./results/{self.run_time}"
        os.makedirs(self.result_path)

        self.epochs = epochs
        self.batch_size = batch_size
        self.save_interval = save_interval
        self.log_dir = log_dir

        # Configuracion Progresiva
        self.resolutions = [4, 8, 16, 32, 64]
        self.resolution = 64
        self.channels = 3

        self.shape = (self.resolution, self.resolution, self.channels)

        # Configurar data loader
        self.dataset_path = "./data"
        self.dataset_name = "DCars_8000"

        self.data = Dataset(self.dataset_path, self.dataset_name, self.resolutions, self.batch_size)

        # Configuracion de los modelos
        self.z = 64
        self.beta_l = 0.5
        self.learning_rate = 0.0001
        self.leaky_alpha = 0.2

        # Caracteristicas de los Plots
        self.r = 5
        self.c = 5

        # Ruido fijo para ver la evolucion de una figura
        self.noise_results = self.generate_latent_points(self.r * self.c)

        # Definicion del discriminador
        self.discriminator = self.build_discriminator()
        
        # Definicion del generador
        self.generator = self.build_generator()
        
        # Definicion de la red compuesta GAN
        self.GAN = self.build_gan()
        
        self.tensorboard = self.build_tensorboard()

    def build_gan(self):
        # Los pesos del discriminador no se entrenan cuando lo hace el generador
        self.discriminator.trainable = False

        model = Sequential()
        # Anadir el generador
        model.add(self.generator)
        # Anadir el discriminador
        model.add(self.discriminator)
        # Compilar modelo
        opt = Adam(lr=self.learning_rate, beta_1=self.beta_l)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

    def build_discriminator(self):
        # Generacion automatica de capas en funcion de la resolucion
        def conv_block(model):
            model.add(Conv2D(128, (4, 4), strides=(1, 1), padding='same'))
            model.add(Conv2D(256, (4, 4), strides=(2, 2), padding='same'))
            model.add(BatchNormalization())
            model.add(LeakyReLU(alpha=self.leaky_alpha))
            return model

        model = Sequential()
        # normal - 4
        model.add(Conv2D(128, (4, 4), padding='same', input_shape=self.shape))
        model.add(Conv2D(256, (4, 4), strides=(1, 1), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=self.leaky_alpha))

        iter = int(math.log(self.resolution, 2) - 2)

        for i in range(0, iter):
            model = conv_block(model)

        # Clasificador monoclase
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        # Compilar modelo
        opt = Adam(lr=self.learning_rate, beta_1=self.beta_l)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    def build_generator(self):
        # Generacion automatica de capas en funcion de la resolucion
        def deconv_block(model):
            model.add(Conv2DTranspose(128, (4, 4), strides=(1, 1), padding='same'))
            model.add(Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same'))
            # model.add(BatchNormalization())
            model.add(LeakyReLU(alpha=self.leaky_alpha))
            return model

        model = Sequential()
        # Imagen inicial de 4x4
        n_nodes = 1024 * 4 * 4
        model.add(Dense(n_nodes, input_dim=self.z))
        model.add(LeakyReLU(alpha=self.leaky_alpha))
        model.add(Reshape((4, 4, 1024)))

        iter = int(math.log(self.resolution, 2) - 2)
        for i in range(0, iter):
            model = deconv_block(model)

        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=self.leaky_alpha))

        # Capa de salida
        model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))
        return model

    def build_tensorboard(self):
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=0,
            batch_size=self.batch_size,
            write_graph=True,
            write_grads=True
        )
        tensorboard.set_model(self.GAN)
        return tensorboard

    # Muestras reales del dataset cargado
    def generate_real_samples(self, dataset, n_samples):
        ix = randint(0, dataset.shape[0], n_samples)
        X = dataset[ix]
        y = ones((n_samples, 1))
        return X, y

    # Muestras del espacio latente
    def generate_latent_points(self, n_samples):
        x_input = randn(self.z * n_samples)
        x_input = x_input.reshape(n_samples, self.z)
        return x_input
    
    # Uso del generador para crear imagenes falsas        
    def generate_fake_samples(self, n_samples):
        x_input = self.generate_latent_points(n_samples)
        x = self.generator.predict(x_input)
        y = zeros((n_samples, 1))
        return x, y

    # Generador de imagenes falsas
    def generate_fake(self, noise):
        x = self.generator.predict(noise)
        y = zeros((noise.shape[0], 1))
        return x, y

    def train(self):
        generator_shape = self.generator.output_shape
        print(generator_shape)
        # Carga del set de datos correspondiente
        self.data.load_data(generator_shape[1])
        print(f'Dataset info: {self.data.data_shape}')
        dataset = self.data.get_data()

        # Definicion de los diferentes batches utilzados
        bat_per_epo = int(self.data.data_shape[0] / self.batch_size)
        half_batch = int(self.batch_size / 2)

        # Inicio del proceso de entrenamiento
        bar = tqdm(range(self.epochs), desc="Epoca")
        for i in bar:
            for j in range(bat_per_epo):
                bar.set_description(f"Epoch: {i} - Batch: {j}/{bat_per_epo}")
                x_real, y_real = self.generate_real_samples(dataset, half_batch)
                # Actualizacion de los pesos del discriminador
                d_loss1, _ = self.discriminator.train_on_batch(x_real, y_real)

                # Generar ejemplos falsos
                x_fake, y_fake = self.generate_fake_samples(half_batch)
                # Actualizar los pesos del discriminador con modelos falsos
                d_loss2, _ = self.discriminator.train_on_batch(x_fake, y_fake)

                # Generar puntos del espacio latente
                x_gan = self.generate_latent_points(self.batch_size)
                # Asumir que las imagenes falsas son reales
                y_gan = ones((self.batch_size, 1))-0.1
                # Actualizar el peso del generador mediante el error del discriminador
                g_loss = self.GAN.train_on_batch(x_gan, y_gan)

                # Salida de las diferentes perdidas
                bar.write('>%d, %d/%d, d1=%.3f, d2=%.3f g1=%.3f' %
                      (i + 1, j + 1, bat_per_epo, d_loss1, d_loss2, g_loss))

            # Obtener plots
            if (i + 1) % self.save_interval == 0:
                self.show_performance(i)

        self.tensorboard.on_train_end(None)

    def show_performance(self, epoch):
        shape = self.generator.output_shape
        name = f"{shape[1]}x{shape[2]}-{epoch}"

        # Se utiliza el mismo ruido para ver la progresion de las imagenes
        x, _ = self.generate_fake(self.noise_results)

        # Normalizar pixeles
        x = (x - x.min()) / (x.max() - x.min())

        # Caso de que solo se quiera 1 imagen de salida
        if self.r == 1:
            fig = plt.figure()
            plt.imshow(x[0])
            plt.axis('off')
        else:
            # Multiples imagenes en el plot
            fig, axs = plt.subplots(self.r, self.c)
            cnt = 0
            for i in range(self.r):
                for j in range(self.c):
                    axs[i, j].imshow(x[cnt])
                    axs[i, j].axis('off')
                    cnt += 1
        filename1 = f"plot_{name}.png"
        fig.savefig(f"{self.result_path}/{filename1}")
        plt.close()

        if (epoch + 1) % 10 == 0:
            # Guardar modelos
            filename2 = f"model_{name}.h5"
            self.save_model(self.generator, f"{self.result_path}/{filename2}")

    def save_model(self, model, out_model_path):
        model.save(out_model_path)

if __name__ == '__main__':
    cgan = CGAN(epochs=2000, batch_size=32, save_interval=1, log_dir="./tmp/logs/")
    cgan.train()
