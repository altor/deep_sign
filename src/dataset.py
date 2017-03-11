import os
import urllib.request as urllib

from sklearn import datasets
from scipy import ndimage
import cv2


class Dataset:
    
    def __init__(self):
        self.dataset_path='../GTSRB_min_1/Final_Training/Images'
        self.dataset_url='http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip'
        self.dataset_zip_file_name = '../GTSRB_Final_Training_Images.zip'
        self.data = []
        self.label = []

        # Téléchargement de l'ensemble s'il n'existe pas
        if(not os.access(self.dataset_path, os.R_OK)):
            self.download()

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

    def extract(self, resize=False, size_x=None, size_y=None):
        for directory in os.listdir(self.dataset_path):
            class_id = str(directory)
            for img_file in os.listdir(self.dataset_path +
                                       "/" + directory):

                if img_file.split('.')[1] == 'csv':
                    continue
                img = cv2.imread(self.dataset_path + "/" + directory
                                 + "/" + img_file)
                self.data.append(img)
                self.label.append(class_id)
