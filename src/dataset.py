import os
import urllib2
import h5py
# from sklearn import datasets
from scipy import ndimage
import cv2
import numpy as np
import zipfile
import helper




class Dataset:
    
    def __init__(self, dataset_path, dataset_url, dataset_zip_file_name, dataset_hdf_file_name):
        self.dataset_path=dataset_path
        self.dataset_url=dataset_url 
        self.dataset_zip_file_name = dataset_zip_file_name
        self.dataset_hdf_file_name = dataset_hdf_file_name
        self.data = []
        self.label = []



        # chargement des donnees en memoire
        if(not os.access(self.dataset_hdf_file_name, os.R_OK)):
            # Telechargement de l'ensemble s'il n'existe pas
            if(not os.access(self.dataset_path, os.R_OK)):
                self.download()
            self.extract()
            self.data = np.asarray(self.data)
            self.label = np.asarray(self.label)
            self.save()

        else:
            self.load()

        # conversion en ndArray



    def download(self):
        print("[INFO] Download dataset")
        u = urllib2.urlopen(self.dataset_url)
        f = open(self.dataset_zip_file_name, 'wb')
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            f.write(buffer)
        f.close()

        zip_ref = zipfile.ZipFile(self.dataset_zip_file_name, 'r')
        zip_ref.extractall("./")
        zip_ref.close()

    def save(self):
        f = h5py.File(self.dataset_hdf_file_name, "w")
        data_set = f.create_dataset("data", data=self.data)
        label_set = f.create_dataset("label", data=self.label)
        f.close()


    def load(self):
        print("[INFO] Load saved dataset")
        f = h5py.File(self.dataset_hdf_file_name,'r')
        self.data = f['data'][()]
        self.label = f['label'][()]
        f.close()
        
        
    def extract(self, resize=False, size_x=None, size_y=None):
        print("[INFO] Extract dataset")
        # parcour l'arboressence du dataset
        for directory in os.listdir(self.dataset_path):
            print("[INFO] Extract directory : " + directory)
            class_id = int(directory)
            for img_file in os.listdir(self.dataset_path +
                                       "/" + directory):

                # On verifie que l'image n'est pas le csv de description de la classe
                if img_file.split('.')[1] == 'csv':
                    continue

                # extraction des donnees de l'image
                img = cv2.imread(self.dataset_path + "/" + directory
                                 + "/" + img_file, cv2.IMREAD_COLOR)

                # On convertie l'image pour n'avoir qu'une valeur par pixle au lieu d'un triplet
                img = helper.img_to_rgb(img)

                # Ajout de l'image au dataset
                self.data.append(img)

                # Ajout de la classe de l'image au dataset
                self.label.append(class_id)


def get_gtsrb():
    return Dataset(
        'GTSRB/Final_Training/Images',
        'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip',
        'GTSRB_Final_Training_Images.zip',
        'GTSRB.hdf5'
    )

def get_gtsrb_min():
    return Dataset(
        'GTSRB_min_28/Final_Training/Images',
        'http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip',
        'GTSRB_Final_Training_Images.zip',
        'GTSRB_min.hdf5'
    )                
