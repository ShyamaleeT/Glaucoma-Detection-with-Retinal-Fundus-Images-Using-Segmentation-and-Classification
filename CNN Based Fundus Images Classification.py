#scikit-image (formerly scikits. image) is an open-source image processing library for the Python programming language.
!pip install scikit-image

#mount google drive
from google.colab import drive
drive.mount('/content/drive/')

from keras import optimizers
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler, ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Concatenate
import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

#set the train and test paths
train_path = '/content/drive/MyDrive/RIM-ONE/train'
test_path = '/content/drive/MyDrive/RIM-ONE/test' 

train_classes = os.listdir(train_path)
train_batch_size = 8
test_batch_size = 8
train_n = 4233 # number of training images
test_n = 932 # number of testing images
train_steps = train_n//train_batch_size
test_steps = test_n//test_batch_size
input_shape = (299, 299, 3) # input image sizes for Inceptionv3
num_classes = len(train_classes) # number of classes
epochs = 150


