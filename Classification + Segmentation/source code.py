#The overall process
#Datasets --> Apply Preprocessing techniques to the dataset (CLAHE, Median Filtering) --> Apply data augmentation techniques --> Spliit the dataset into 70:15:15 (train, test and validation) ratio
#--> fed to the data into attention U-Net architecture (as the backbne of the U-Net used pre-trained Inception-v3, VGG19 and ResNet50 architectures)

import tensorflow as tf
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import keras 
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19,InceptionV3,ResNet50 
from imutils import contours
from skimage import measure
import argparse
import imutils
from google.colab.patches import cv2_imshow

#mount drive
from google.colab import drive
drive.mount('/content/drive')

#set the train, test and validation paths
train_path = '/content/drive/MyDrive/RIM-ONE/train'
test_path = '/content/drive/MyDrive/RIM-ONE/test' 
validation_path = '/content/drive/MyDrive/RIM-ONE/validation'

#Apply CLAHE and Median Filtering methods
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
clahe_img = clahe.apply(l)
#plt.hist(clahe_img.flat, bins=100, range=(0,255))

#Combine the CLAHE enhanced L-channel back with A and B channels
updated_lab_img2 = cv2.merge((clahe_img,a,b))

#Convert LAB image back to color (RGB)
CLAHE_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)

cv2_imshow(img)
cv2_imshow(CLAHE_img)

#Apply Median Filtering Method
image = cv2.imread('img_path')
median=cv2.medianBlur(blur,5)
cv2_imshow(median)

#Apply Data Augmentation Techniques
train_datagen = ImageDataGenerator( rotation_range=10,
                                    shear_range=0.2,
                                    zoom_range=0.1,
                                    horizontal_flip=True,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    brightness = 1,
                                    contrast = 2)

validation_datagen= ImageDataGenerator()
  
#Spliit the dataset (70:15:15) --> Train,test and Validation 
X_train, X_test, y_train, y_test 
    = train_test_split(X, y, test_size=0.3, random_state=1)

 X_train, X_val, y_train, y_val 
    = train_test_split(X_train, y_train, test_size=0.42, random_state=1) # 0.42 x 0.7 = 0.3

X_train_x = input(X_train)
X_test_x = input(X_test)

#Attention U-Net - For segmentation process
def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
 
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
 
    return x

#Implement the Decoder Block
#2×2 Transpose Convolution layer <-- concatenation layer( skip connection) <-- conv_block

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

# define pre-trained Inceptionv3 Encoder for U-Net
#for the segmentation, used keras pre-trained inceptionv3, VGG19 and ResNet50 for backbone of the U-Net.
InceptionV3_model = InceptionV3(input_shape=(299,299,3),weights='imagenet', include_top=False)
InceptionV3_model.summary()

#Replace the UNET encoder with the VGG19 implementation 
p1 = InceptionV3_model.get_layer("block1_conv2").output       
p2 = InceptionV3_model.get_layer("block2_conv2").output         
p3 = InceptionV3_model.get_layer("block3_conv3").output        
p4 = InceptionV3_model.get_layer("block4_conv3").output    
   
#Attention U-Net bridge
p5 = InceptionV3_model.get_layer("block5_conv3").output 

#decoder path
q1 = decoder_block(p5, p4, 128)                  
q2 = decoder_block(q1, p3, 64)                    
q3 = decoder_block(q2, p2, 32)       
q4 = decoder_block(q3, p1, 16) 
 
Inceptionv3_output = Conv2D(1, 1, padding="same", activation="softmax")(q4)

adam = optimizers.Adam(learning_rate=0.0001, name='Adam')
Inceptionv3_output.compile(optimizer=adam, loss='binary_crossentropy',metrics=['accuracy'])

model_history=Inceptionv3_output.fit(X_train_x, 
                                y_train,
                                batch_size=8, 
                                epochs=150,
                                verbose=1,
                                validation_data=(X_test_x))

Inceptionv3_output.save('inceptionv3.hdf5')

# define  pre-trained VGG19 Encoder for U-Net
VGG19_model = VGG19(pooling='avg', weights='imagenet', include_top=False, input_shape=(224,224,3))

VGG19_model.summary()

#Replace the UNET encoder with the VGG19 implementation 
x1 = VGG19_model.get_layer("block1_conv2").output       
x2 = VGG19_model.get_layer("block2_conv2").output         
x3 = VGG19_model.get_layer("block3_conv3").output        
x4 = VGG19_model.get_layer("block4_conv3").output    

#Attention U-Net bridge
x5 = VGG19_model.get_layer("block5_conv3").output 

#decoder path
y1 = decoder_block(x5, x4, 128)                  
y2 = decoder_block(y1, x3, 64)                    
y3 = decoder_block(y2, x2, 32)       
y4 = decoder_block(y3, x1, 16) 

VGG_19_output = Conv2D(1, 1, padding="same", activation="softmax")(y4)

sgd = optimizers.SGD(learning_rate=0.001, momentum=0.9)
VGG_19_output.compile(optimizer=sgd, loss='binary_crossentropy',metrics=['accuracy'])

model_history=VGG_19_output.fit(X_train_x, 
                                y_train,
                                batch_size=8, 
                                epochs=150,
                                verbose=1,
                                validation_data=(X_test_x))

VGG_19_output.save('VGG19.hdf5')

# define pre-trained ResNet50 Encoder for U-Net
ResNet50_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
ResNet50_model.summary()

#Replace the UNET encoder with the VGG19 implementation 
u1 = ResNet50_model.get_layer("block1_conv2").output       
u2 = ResNet50_model.get_layer("block2_conv2").output         
u3 = ResNet50_model.get_layer("block3_conv3").output        
u4 = ResNet50_model.get_layer("block4_conv3").output    

#Attention U-Net bridge
u5 = ResNet50_model.get_layer("block5_conv3").output 

#decoder path
v1 = decoder_block(x5, u4, 128)                  
v2 = decoder_block(v1, u3, 64)                    
v3 = decoder_block(v2, u2, 32)       
v4 = decoder_block(v3, u1, 16) 

ResNet50_output = Conv2D(1, 1, padding="same", activation="softmax")(y4)

sgd = optimizers.SGD(learning_rate=0.001, momentum=0.9)
ResNet50_output.compile(optimizer=sgd, loss='binary_crossentropy',metrics=['accuracy'])

model_history=ResNet50_output.fit(X_train_x, 
                                y_train,
                                batch_size=8, 
                                epochs=150,
                                verbose=1,
                                validation_data=(X_test_x))

ResNet50_output.save('ResNet50.hdf5')

#for the classification process - Spliit the segmented images into 70:15:15 ratio --> fed to the modified CNN architecture (Inceptionv3, VGG19 and ResNet50) 

# define model-Inceptionv3 Encoder for U-Net
# The global average pooling layer was placed after the Inception-v3 model to reduce the parameters,followed by the dense layer (512 units), and lastly, added theSoftmax layer.
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



# define modified VGG19 Encoder for U-Net
#The global average pooling layer is pursued by the last three newly added layers namely, dropout layer with 0.5 rates, dense layer (256 units) with ReLU activation function, and finally, Softmax layer with two outputs.
VGG_19 = VGG19(pooling='avg', weights='imagenet', include_top=False, input_shape=(224,224,3))
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


# define modified ResNet50 Encoder for U-Net
#The fully connected layer is substituted with another fully connected dense layer with 256 units
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
plt.title("AttentionU-Net_Inceptionv3")
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
plt.title("AttentionU-Net_VGG19")
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
plt.title("AttentionU-Net_ResNet50")
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
preds = Inceptionv3_output.predict_generator(validation_generator, test_n)

#VGG19
preds =VGG19_output.predict_generator(validation_generator, test_n)

#ResNet50
preds = ResNet50_output.predict_generator(validation_generator, test_n)

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

#Classification Report
from sklearn.metrics import classification_report
cr = classification_report(validation_generator.labels,Y_pred)

#Instead of classification Report can use below code using A output in above.
total1=sum(sum(cnf_matrix))

accuracy=(cnf_matrix[0,0]+cnf_matrix[1,1])/total1
print ('Accuracy : ', accuracy)

Precision = cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[1,0])
print('precision : ', Precision)

recall = cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[0,1])
print('Recall : ', recall)

F1_score=(Precision * recall1 / (Precision1 + recall1 ))   * 2 
print('F1_score : ', F1_score )

sensitivity = cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[0,1])
print('Sensitivity : ', sensitivity)

specificity = cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])
print('Specificity : ', specificity)

Dice_Coefficient = 2 * cnf_matrix[0,0]/(2 * cnf_matrix[0,0]+cnf_matrix[1,1]+cnf_matrix[0,1])
print('Dice Coefficient : ', Dice_Coefficient)

Jaccard_Coefficient = cnf_matrix[0,0]/cnf_matrix[0,0]+cnf_matrix[1,1]+cnf_matrix[0,1]
print('Jaccard Coefficient : ', Jaccard_Coefficient)

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
#plt.savefig("/content/drive/MyDrive/RIM-ONE/diagrams/" + "Inceptionv3.svg")
plt.show()
