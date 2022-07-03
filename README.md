# Glaucoma-Detection-with-Retinal-Fundus-Images-Using-Segmentation-and-Classification

We've implemented two methods. The first one is CNN Based Fundus Images Classification For Glaucoma Identification  and the second one is Attention U-Net for 
Glaucoma Identification Using Fundus Image Segmentation

1. CNN Based Fundus Images Classification For Glaucoma Identification - 
This study uses three different Convolutional Neural Networks (CNNs) architectures, namely Inception-v3, Visual Geometry Group 19 (VGG19), Residual Neural Network 50 (ResNet50), to classify glaucoma subjects using eye fundus images. In addition, several data pre-processing and augmentation techniques were used to
avoid overfitting and achieve high accuracy. The aim of thispaper is to comparative analysis of the performance obtainedfrom different configurations with CNN architectures and hyperparameter tuning.

Link to the paper:- https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9754171

2. Attention U-Net for Glaucoma Identification Using Fundus Image Segmentation - 
This study proposes attention UNet models with three Convolutional Neural Networks (CNNs) architectures, namely Inception-v3, Visual Geometry Group 19 (VGG19), Residual Neural Network 50 (ResNet50) to segment fundus images.

Link to the paper:- https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9765303

3. Glaucoma Detection with Retinal Fundus Images Using Segmentation and Classification - 
This study proposes a computational model to segment and classify retinal fundus images for glaucoma detection. Different data augmentation techniques were applied to refrain overfitting while employing several data pre-processing approaches to improve the image quality and achieve high accuracy. The segmentation models are based on an attention U-Net with separate three Convolutional Neural Networks (CNNs) backbones: Inception-v3, Visual Geometry Group19 (VGG19), and Residual Neural Network 50 (ResNet50). The classification models are also employing a modified version of the above three CNN architectures.

Datasets

We've used two dataset and spliit the datasets into 70:15:15 (train, test and validation) ratio.

After spliit the dataset

--> RIM-ONE dataset

      Train set - 4512 images (Glaucoma:2232 images, Normal: 2280 images)
      
      Test set - 976 images (Glaucoma:480 images, Normal:496 images)
      
      Validation set - 970 images (Glaucoma:480 images, Normal:490 images)
      
Link to the RIM-ONE dataset - https://drive.google.com/drive/folders/18kHb_hPrX_dxbZTyfbvH9fQmgz3n1I5I?usp=sharing

--> ACRIMA dataset

      Train set - 3193 images (Glaucoma:1590 images, Normal: 1603 images)
      
      Test set - 724 images (Glaucoma:372 images, Normal:352 images)
      
      Validation set - 718 images (Glaucoma:366 images, Normal:352 images)
      
Link to the ACRIMA dataset - https://drive.google.com/drive/folders/1uiXUZL5EZ2-F0tE8qWJVdYVRb5oyDm3g?usp=sharing

For the Segmentation + Classification process, used segmented images generate from segmentati0n process.

--> Segmented dataset

      Train set - 2722 images
      
      Test set - 583 images
      
      Validation set - 583 images
      
Link to the Segmented image dataset - https://drive.google.com/drive/folders/1cQV3WnFSMLsMj-kqOC-wub54DSBe1wGE?usp=sharing
      
   
      
      

      
