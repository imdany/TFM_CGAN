from glob import glob
from numpy import load

class Dataset():
    def __init__(self, dataset_path, dataset_name, resolutions, batch_size):

        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.resolutions = resolutions
        self.batch_size = batch_size

        self.datasets = self.get_list_datasets()

        self.position = 0

        self.data = None
        self.data_shape = None
        self.batches = None

    # Obtener archivos disponibles
    def get_list_datasets(self):
        path = glob(f"{self.dataset_path}/{self.dataset_name}*.npz")
        path.sort()
        zipbObj = zip(self.resolutions, path)
        model_dict = dict(zipbObj)
        return model_dict

    # Devolver el dataset completo
    def get_data(self):
        return self.data

   # Cargar la resolucion especifica del dataset
    def load_data(self, resolution):
        filename = self.datasets[resolution]
        
        data = load(filename)
        X = data['arr_0']
        # Convertir a float32
        X = X.astype('float32')
        # Escalar imagenes [-1, 1]
        X = (X - 127.5) / 127.5
        
        self.data = X
        self.data_shape = X.shape
        self.batches = int(self.data_shape[0] / self.batch_size)

    # Gestion de batches
    def get_next_batch(self):

        if self.position == self.data_shape[0]:
            self.position = 0

        ini = self.position
        fin = ini + self.batch_size

        if fin >= self.data_shape[0]:
            fin = self.data_shape[0]

        x = self.data[ini:fin]
        self.position = fin

        return x