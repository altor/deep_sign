import numpy as np


def __rgb_to_int(rgb):
    return rgb[0] * 65536 + rgb[1] * 256 + rgb[2]


def __int_to_rgb(n):
    rgb = [1,2,3]
    rgb[2] = n % 256
    rgb[1] = ((n - rgb[2]) % 65536) // 256
    rgb[0] = (n - rgb[2] - rgb[1]) // 65536
    return rgb

def rgb_to_img(img):
    new_image = []
    for line in img:
        new_line = []
        for n in line:
            new_line.append(__int_to_rgb(n))
        new_image.append(new_line)
    return np.asarray(new_image)

def img_to_rgb(img):
    new_image = []
    for line in img:
        new_line = []
        for rgb in line:
            new_line.append(__rgb_to_int(rgb))
        new_image.append(new_line)
    return np.asarray(new_image)
        
