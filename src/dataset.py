from sklearn import datasets
from scipy import ndimage
from io import BytesIO
import urllib.request as urllib


dataset_url='http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip'
dataset_zip_file_name = 'GTSRB_Final_Training_Images.zip'

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

def extract(resize=False, size_x=None, size_y=None):
    data = datasets.load_files("./GTSRB_min/Final_Training/Images")
    new_data = []

    for raw_img in data.data:
        img = ndimage.imread(BytesIO(raw_img))
        # if resize:
        #     img.reshape
        new_data.append(img)
        

    data.data = new_data
    return data
