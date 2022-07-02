#scikit-image (formerly scikits. image) is an open-source image processing library for the Python programming language.
!pip install scikit-image

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
from skimage import io, filters, feature
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import cv2

#mount google drive
from google.colab import drive
drive.mount('/content/drive/')

#set the train and test paths
train_path = '/content/drive/MyDrive/RIM-ONE/train'
test_path = '/content/drive/MyDrive/RIM-ONE/test' 
validation_path = '/content/drive/MyDrive/RIM-ONE/validation'

#Train,test and Validation Spliit
X_train, X_test, y_train, y_test 
    = train_test_split(X, y, test_size=0.3, random_state=1)

 X_train, X_val, y_train, y_val 
    = train_test_split(X_train, y_train, test_size=0.42, random_state=1) # 0.42 x 0.7 = 0.3


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

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(input_shape[0],input_shape[1]),
        batch_size=train_batch_size,
        classes=train_classes,
        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
        validation_path,
        target_size=(input_shape[0],input_shape[1]),
        batch_size=test_batch_size,
        classes=train_classes,
        class_mode='categorical')


#Apply CLAHE and Dilation methods
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

validation_datagen= ImageDataGenerator()

train_generator.class_indices
validation_generator.class_indices

#Inceptionv3 model
InceptionV3_model = InceptionV3(input_shape=(299,299,3),weights='imagenet', include_top=False)
for layer in InceptionV3_model.layers[:249]:
   layer.trainable = False
for layer in InceptionV3_model.layers[249:]:
   layer.trainable = True
   
InceptionV3_last_output = InceptionV3_model.output
InceptionV3_maxpooled_output = Flatten()(InceptionV3_last_output)
InceptionV3_x = Dense(512, activation='relu')(InceptionV3_maxpooled_output)
InceptionV3_x = Dropout(0.7)(InceptionV3_x)
InceptionV3_x = Dense(2, activation='softmax')(InceptionV3_x)
InceptionV3_x_final_model = Model(inputs=InceptionV3_model.input,outputs=InceptionV3_x)
InceptionV3_x_final_model.summary()

#Apply early stopping technique to prevent from overfitting
es_callback = EarlyStopping(
        monitor='val_loss',
        verbose=1,
        mode='max',
        restore_best_weights =True,
        patience=10)

#Reduce learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=1/3, min_lr=1e-5)

callbacks_list = []
checkpoint = ModelCheckpoint("./checkpoints/InceptionV3.hdf5",
                                                monitor="val_acc",
                                                verbose = 1,
                                                save_best_only = True,
                                                save_weights_only = False,
                                                mode= "max")
callbacks_list.append(checkpoint)

#Apply optimizer
adam = optimizers.Adam(learning_rate=0.0001, name='Adam')

InceptionV3_x_final_model.compile(optimizer=adam, loss='binary_crossentropy',metrics=['accuracy'])

model_history = InceptionV3_x_final_model.fit_generator(train_generator, 
                                    validation_data=validation_generator, 
                                    validation_steps=test_steps, 
                                    steps_per_epoch=train_steps, 
                                    epochs=epochs,
                                    callbacks=[callbacks_list])


#VGG19 model
vgg19_model = VGG19(pooling='avg', weights='imagenet', include_top=False, input_shape=(224,224,3))
for layers in vgg19_model.layers:
    layers.trainable=False

last_output = vgg19_model.layers[-1].output
vgg_x = Flatten()(last_output)
vgg_x = Dense(256, activation = 'relu')(vgg_x)
vgg_x = Dropout(0.5)(vgg_x)
vgg_x = Dense(2, activation = 'softmax')(vgg_x)
vgg19_final_model = Model(vgg19_model.input, vgg_x)
vgg19_final_model.summary()

#Apply early stopping technique to prevent from overfitting
es_callback = EarlyStopping(
        monitor='val_loss',
        verbose=1,
        mode='max',
        restore_best_weights =True,
        patience=10)

#Reduce learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=1/3, min_lr=1e-4)

callbacks_list = []
checkpoint = ModelCheckpoint("./checkpoints/vgg19.hdf5",
                                                monitor="val_acc",
                                                verbose = 1,
                                                save_best_only = True,
                                                save_weights_only = False,
                                                mode= "max")
callbacks_list.append(checkpoint)

sgd = optimizers.SGD(learning_rate=0.001, momentum=0.9)

vgg19_final_model.compile(optimizer=sgd, loss='binary_crossentropy',metrics=['accuracy'])

model_history = vgg19_final_model.fit_generator(train_generator, 
                                    validation_data=validation_generator, 
                                    validation_steps=test_steps, 
                                    steps_per_epoch=train_steps, 
                                    epochs=epochs,
                                    callbacks=[callbacks_list])


#ResNet50 model
ResNet50_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
for layers in ResNet50_model.layers:
    layers.trainable=True

resnet50_x = Flatten()(ResNet50_model.output)
resnet50_x = Dense(256,activation='relu')(resnet50_x)
resnet50_x = Dense(2,activation='softmax')(resnet50_x)
resnet50_x_final_model = Model(inputs=ResNet50_model.input, outputs=resnet50_x)
resnet50_x_final_model.summary()

#Apply early stopping technique to prevent from overfitting
es_callback = EarlyStopping(
        monitor='val_loss',
        verbose=1,
        mode='max',
        restore_best_weights =True,
        patience=10)

#Reduce learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=1/3, min_lr=1e-4)

callbacks_list = []
checkpoint = ModelCheckpoint("./checkpoints/resnet50.hdf5",
                                                monitor="val_acc",
                                                verbose = 1,
                                                save_best_only = True,
                                                save_weights_only = False,
                                                mode= "max")
callbacks_list.append(checkpoint)

sgd = optimizers.SGD(learning_rate=0.001, momentum=0.9)

resnet50_x_final_model.compile(optimizer=sgd, loss='binary_crossentropy',metrics=['accuracy'])

model_history = resnet50_x_final_model.fit_generator(train_generator, 
                                    validation_data=validation_generator, 
                                    validation_steps=test_steps, 
                                    steps_per_epoch=train_steps, 
                                    epochs=epochs,
                                    callbacks=[callbacks_list])


#Train/Validation accuracy,loss graphs Inceptionv3
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = '12'
plt.plot(epochs,a,color = 'Maroon',linestyle='-',linewidth = '3' ) #accuracy
plt.plot(epochs,b,color = 'red',linewidth = '3') # val_accuracy
plt.plot(epochs,c,color = 'Purple',linewidth = '3',linestyle = ':') #accuracy
plt.plot(epochs,d,color = 'RoyalBlue',linewidth = '3', linestyle = ':') # val_loss

plt.rcParams["figure.figsize"] = (8,7)

plt.ylabel("Accuracy/Loss")
plt.xlabel("Epochs")
plt.legend(["Train_Acc", "Val_Acc", "Loss", "Val_Loss"], loc="best")
plt.title("RIM-ONE_Inceptionv3")
plt.grid()
plt.show()

#VGG19
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = '12'
plt.plot(epochs,a,color = 'Maroon',linestyle='-',linewidth = '3' ) #accuracy
plt.plot(epochs,b,color = 'red',linewidth = '3') # val_accuracy
plt.plot(epochs,c,color = 'Purple',linewidth = '3',linestyle = ':') #accuracy
plt.plot(epochs,d,color = 'RoyalBlue',linewidth = '3', linestyle = ':') # val_loss

plt.rcParams["figure.figsize"] = (8,7)

plt.ylabel("Accuracy/Loss")
plt.xlabel("Epochs")
plt.legend(["Train_Acc", "Val_Acc", "Loss", "Val_Loss"], loc="best")
plt.title("RIM-ONE_VGG19")
plt.grid()
plt.show()

#ResNet50
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = '12'
plt.plot(epochs,a,color = 'Maroon',linestyle='-',linewidth = '3' ) #accuracy
plt.plot(epochs,b,color = 'red',linewidth = '3') # val_accuracy

plt.plot(epochs,c,color = 'Purple',linewidth = '3',linestyle = ':') #accuracy
plt.plot(epochs,d,color = 'RoyalBlue',linewidth = '3', linestyle = ':') # val_loss

plt.rcParams["figure.figsize"] = (8,7)

plt.ylabel("Accuracy/Loss")
plt.xlabel("Epochs")
plt.legend(["Train_Acc", "Val_Acc", "Loss", "Val_Loss"], loc="best")
plt.title("RIM-ONE_ResNet50")
plt.grid()
plt.show()

#Confusion_Matrix
validation_generator = validation_datagen.flow_from_directory(
        test_path,
        target_size=(299, 299),
        shuffle = False,
        batch_size=8,
        class_mode='categorical')

filenames = validation_generator.filenames
Y_test = validation_generator.classes
nb_samples = len(filenames)

#inceptionv3
preds = InceptionV3_x_final_model.predict_generator(validation_generator, test_n)

#VGG19
preds = vgg19_final_model.predict_generator(validation_generator, test_n)

#ResNet50
preds = resnet50_x_final_model.predict_generator(validation_generator, test_n)

Y_pred = np.argmax(preds, axis = 1)

ans = 0
for i in range(Y_test.shape[0]):
    if Y_test[i] == Y_pred[i]:
        #print Y_test[i], Y_pred[i]
        ans = ans + 1    

print("Test Accuracy is " + str((float(ans/Y_test.shape[0]))*100))

#Confusion Matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
                                                    
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        
    #fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

state = {0:'Glaucoma', 1:'Normal'}
Y_state = np.array([state[a] for a in Y_test])
Y_pred_state = np.array([state[a] for a in Y_pred])

plt.rcParams.update({'font.size': 15})
class_names =[]
for k in range(2):
    class_names.append(state[k])
cnf_matrix = confusion_matrix(Y_state, Y_pred_state)
np.set_printoptions(precision=2)

#Plot non-normalized confusion matrix
#plt.figure()
plt.figure(figsize=(6, 6))
plot_confusion_matrix(cnf_matrix,classes=class_names,title='Inception V3 Confusion matrix') #can use same code for VGG19 and ResNet50

#Output is an array -----> A
confusion_matrix(Y_state, Y_pred_state)

#Classification Report
from sklearn.metrics import classification_report
cr = classification_report(validation_generator.labels,Y_pred)

#Instead of classification Report can use below code using A output in above.
total1=sum(sum(cnf_matrix))

accuracy1=(cnf_matrix[0,0]+cnf_matrix[1,1])/total1
print ('Accuracy : ', accuracy1)

Precision1 = cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[1,0])
print('precision : ', Precision1 )

recall1 = cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[0,1])
print('Recall : ', recall1 )

F1_score=(Precision1 * recall1 / (Precision1 + recall1 ))   * 2 
print('F1_score : ', F1_score )

sensitivity1 = cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])
print('Specificity : ', specificity1)
print(cr)

#ROC Curve
from sklearn import metrics
# calculate the fpr and tpr for all thresholds of the classification
preds1 = preds[:,1]
fpr, tpr, threshold = metrics.roc_curve(Y_test, preds1)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
plt.title('InceptionV3 ROC Curve')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("/content/drive/MyDrive/RIM-ONE/diagrams/" + "Inceptionv3.svg")
plt.show()
