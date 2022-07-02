from skimage import io, filters, feature
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import cv2
import numpy as np

#mount the drive
from google.colab import drive
drive.mount('/content/drive/')

#Apply CLAHE and Dilation methods

#Apply CLAHE to L channel
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
clahe_img = clahe.apply(l)
#plt.hist(clahe_img.flat, bins=100, range=(0,255))

#Combine the CLAHE enhanced L-channel back with A and B channels
updated_lab_img2 = cv2.merge((clahe_img,a,b))

#Convert LAB image back to color (RGB)
CLAHE_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)

cv2_imshow(img)
cv2_imshow(CLAHE_img)

#Apply Dilation Method
img = cv2.imread('image',0)
_, mask = cv2.threshold(img, 255, 255, cv2.THRESH_BINARY_INV)

kernal = np.ones((10,10), np.uint8)

dilation = cv2.dilate(mask, kernal, iterations=2)

titles = ['dilation']
images = [dilation]
plt.show()

#Apply Data Augmentation Techniques
train_datagen = ImageDataGenerator( rotation_range=10,
                                    shear_range=0.2,
                                    zoom_range=0.1,
                                    horizontal_flip=True,
                                    #vertical_flip=True,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1)

test_datagen = ImageDataGenerator()

