from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2

class DataGenerator():
    def __init__(self):
        self.dataset_name = "Cars"

        self.preview_elements = 25

        self.data_path = "./data"
        self.sourcedata_path = f"./datasets/{self.dataset_name}/"
        self.out_resolutions = [4, 8, 16, 32, 64]

        self.dataset = self.read_data()

        pbar = tqdm(self.out_resolutions, desc= "Resoluciones")
        for i in pbar:
            self.process_data(i)

    # Carga de imagen individual
    def read_img(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    # Modificacion de la resolucion
    def resize_img(self, img, resolution):
        img2 = cv2.resize(img, (resolution, resolution))
        return img2

    # Carga de todas las imagenes
    def read_data(self):
        path = glob(f"{self.sourcedata_path}*.png")
        imgs = []

        for img_path in path:
            img = self.read_img(img_path)
            imgs.append(img)
        return imgs

    # Guardar archivo resultante en npz
    def save_data(self, imgs, out_resolution):
        path = f"{self.data_path}/{self.dataset_name}_{out_resolution}.npz"
        np.savez_compressed(path, imgs)

    # Generar vista previa del resultado
    def generate_preview(self, imgs, out_resolution):
        row_elements = int(np.sqrt(self.preview_elements))

        elements = np.random.randint(imgs.shape[0], size=self.preview_elements)

        fig, axs = plt.subplots(row_elements, row_elements)
        cnt = 0
        for i in range(row_elements):
            for j in range(row_elements):
                axs[i, j].imshow(imgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1

        path = f"{self.data_path}/{self.dataset_name}_{out_resolution}.png"
        fig.savefig(path)
        plt.close()

    # Orquestador
    def process_data(self, out_resolution):
        imgs_down = []
        for img in self.dataset:
            img_r = self.resize_img(img, out_resolution)
            imgs_down.append(img_r)

        imd = np.array(imgs_down)
        self.generate_preview(imd, out_resolution)
        self.save_data(imd, out_resolution)