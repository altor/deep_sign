import os
import urllib.request as urllib

from sklearn import datasets
from scipy import ndimage
import cv2
import numpy as np

import helper


class Dataset:
    
    def __init__(self):
        self.dataset_path='../GTSRB/Final_Training/Images'
        self.dataset_url='http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip'
        self.dataset_zip_file_name = '../GTSRB_Final_Training_Images.zip'
        self.data = []
        self.label = []

        # Téléchargement de l'ensemble s'il n'existe pas
        if(not os.access(self.dataset_path, os.R_OK)):
            self.download()

        # chargement des données en mémoire
        self.load()

        # conversion en ndArray
        self.data = np.asarray(self.data)
        self.label = np.asarray(self.label)

    def download():
        u = urllib.urlopen(dataset_url)
        f = open(dataset_zip_file_name, 'wb')
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            f.write(buffer)
        f.close()

        zip_ref = zipfile.ZipFile(dataset_zip_file_name, 'r')
        zip_ref.extractall("./")
        zip_ref.close()

    def load(self, resize=False, size_x=None, size_y=None):

        # parcour l'arboressence du dataset
        for directory in os.listdir(self.dataset_path):
            class_id = int(directory)
            for img_file in os.listdir(self.dataset_path +
                                       "/" + directory):

                # On vérifie que l'image n'est pas le csv de description de la classe
                if img_file.split('.')[1] == 'csv':
                    continue

                # extraction des données de l'image
                img = cv2.imread(self.dataset_path + "/" + directory
                                 + "/" + img_file, cv2.IMREAD_COLOR)

                # On convertie l'image pour n'avoir qu'une valeur par pixle au lieu d'un triplet
                img = helper.img_to_rgb(img)

                # Ajout de l'image au dataset
                self.data.append(img)

                # Ajout de la classe de l'image au dataset
                self.label.append(class_id)
