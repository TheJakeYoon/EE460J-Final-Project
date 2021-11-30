from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import seaborn as sns

import cv2
import numpy as np
from PIL import Image


mypath = './unsplash'
img_names = [f for f in listdir(mypath) if isfile(join(mypath, f))]

img_dict = {ix:cv2.cvtColor(cv2.imread(f'{mypath}/{img_name}'), cv2.COLOR_RGB2BGR) for ix,img_name in enumerate(img_names)}
fig, axs = plt.subplots(1,len(img_dict.keys()), figsize=(18,10))
plt.suptitle('Unsplash dataset', fontsize=18)
plots = [axs[ix].imshow(img_dict[ix]) for ix in range(len(img_dict.keys()))]
fig.tight_layout()
fig.show()

sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})
fig, axs = plt.subplots(3,len(img_dict.keys()), figsize=(28,18))
figh, axsh = plt.subplots(3,len(img_dict.keys()), figsize=(28,18))
for img in list(img_dict.keys()):
    image = img_dict[img].copy()
    red_img, green_img, blue_img = image.copy(), image.copy(), image.copy()
    red_img[:,:,1] = 0 # set green and blue channel to 0
    red_img[:,:,2] = 0 # set green and blue channel to 0
    green_img[:,:,0] = 0
    green_img[:,:,2] = 0
    blue_img[:,:,0] = 0 # set green and red channel to 0
    blue_img[:,:,1] = 0
    for ix, el in enumerate([red_img, green_img, blue_img]):
        axs[ix,img].imshow(el)
        axsh[ix,img].hist(el.ravel(), bins=25, range=[0,256])