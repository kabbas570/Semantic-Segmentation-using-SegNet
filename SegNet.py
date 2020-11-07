import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "1";
import tensorflow as tf
import tensorflow.keras as    keras
import matplotlib.pyplot as plt
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)
from tensorflow.keras import optimizers

import tensorflow as tf
import cv2
import glob
import numpy as np
import tensorflow.keras.backend as K
from sklearn.metrics import confusion_matrix

## train data
mask_id = []
for infile in sorted(glob.glob('/home/user01/data_ssd/Abbas/XEEEE/unet-master/data/membrane/train/label/*.png')):
    mask_id.append(infile)
image_ = []
for infile in sorted(glob.glob('/home/user01/data_ssd/Abbas/XEEEE/unet-master/data/membrane/train/image/*.jpg')):
    image_.append(infile)
    
mask_id_test = []
for infile in sorted(glob.glob('/home/user01/data_ssd/Abbas/XEEEE/unet-master/data/membrane/test/label/*.png')):
    mask_id_test.append(infile)
image_tets = []
for infile in sorted(glob.glob('/home/user01/data_ssd/Abbas/XEEEE/unet-master/data/membrane/test/img/*.jpg')):
    image_tets.append(infile)    
    
height=256
width=256

image = cv2.imread(image_tets[0])
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (height,width), interpolation = cv2.INTER_AREA)  

flipVertical = cv2.flip(image,0)      
plt.figure()
plt.imshow(flipVertical)
        
def DataGen():  
    img_ = []
    mask_  = []
    for i in range(len(image_)):
        target=np.zeros([256,256,8])
        image = cv2.imread(image_[i])
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=image/255
        image = cv2.resize(image, (height,width), interpolation = cv2.INTER_AREA) 
        image_f = cv2.flip(image,0) 
        mask = cv2.imread(mask_id[i],0)
        mask = cv2.resize(mask, (height,width), interpolation = cv2.INTER_AREA)  
        target[:,:,0][np.where(mask==89)]=1
        target[:,:,1][np.where(mask==38)]=1
        target[:,:,2][np.where(mask==14)]=1
        target[:,:,3][np.where(mask==113)]=1
        target[:,:,4][np.where(mask==75)]=1
        target[:,:,5][np.where(mask==128)]=1
        target[:,:,6][np.where(mask==52)]=1
        target[:,:,7][np.where(mask==0)]=1
        #mask = np.expand_dims(mask, axis=-1)
        target1 = cv2.flip(target,0) 
        img_.append(image)
        img_.append(image_f)
        mask_.append(target)
        mask_.append(target1)      
    img_ = np.array(img_)
    mask_  = np.array(mask_)
    return img_,mask_

images,labels = DataGen()




def DataGen_test():  
    img_ = []
    mask_  = []    
    for i in range(len(image_tets)):
        target=np.zeros([256,256,8])
        image = cv2.imread(image_tets[i])
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=image/255
        image = cv2.resize(image, (height,width), interpolation = cv2.INTER_AREA)  
        mask = cv2.imread(mask_id_test[i],0)
        mask = cv2.resize(mask, (height,width), interpolation = cv2.INTER_AREA)  
        target[:,:,0][np.where(mask==89)]=1
        target[:,:,1][np.where(mask==38)]=1
        target[:,:,2][np.where(mask==14)]=1
        target[:,:,3][np.where(mask==113)]=1
        target[:,:,4][np.where(mask==75)]=1
        target[:,:,5][np.where(mask==128)]=1
        target[:,:,6][np.where(mask==52)]=1
        target[:,:,7][np.where(mask==0)]=1
        #mask = np.expand_dims(mask, axis=-1)
        img_.append(image)
        mask_.append(target)
       
    img_ = np.array(img_)
    mask_  = np.array(mask_)
    return img_,mask_
test_images,test_labels = DataGen_test()


def segnet(
        input_size = (256,256,3)):
    # Block 1
    inputs = keras.layers.Input(input_size)
    x = keras.layers.Conv2D(64, (3, 3),activation='relu',padding='same',name='block1_conv1',kernel_initializer = 'he_normal')(inputs)
    x = keras.layers.Conv2D(64, (3, 3),activation='relu',padding='same',name='block1_conv2',kernel_initializer = 'he_normal')(x)
    x=keras.layers.BatchNormalization()(x)
    pool_1 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    # Block 2
    x = keras.layers.Conv2D(128, (3, 3), activation='relu',padding='same',name='block2_conv1',kernel_initializer = 'he_normal')(pool_1)
    x = keras.layers.Conv2D(128, (3, 3),activation='relu',padding='same',name='block2_conv2',kernel_initializer = 'he_normal')(x)
    x=keras.layers.BatchNormalization()(x)
    pool_2 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
     # Block 3
    x = keras.layers.Conv2D(256, (3, 3),activation='relu',padding='same',name='block3_conv1',kernel_initializer = 'he_normal')(pool_2)
    x = keras.layers.Conv2D(256, (3, 3),activation='relu',padding='same',name='block3_conv2',kernel_initializer = 'he_normal')(x)
    x = keras.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block3_conv3',kernel_initializer = 'he_normal')(x)
    x=keras.layers.BatchNormalization()(x)
    pool_3 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    # Block 4
    x = keras.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block4_conv1',kernel_initializer = 'he_normal')(pool_3)
    x = keras.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block4_conv2',kernel_initializer = 'he_normal')(x)
    x = keras.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block4_conv3',kernel_initializer = 'he_normal')(x)
    x=keras.layers.BatchNormalization()(x)
    pool_4 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    # Block 5
    x = keras.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block5_conv1',kernel_initializer = 'he_normal')(pool_4)
    x = keras.layers.Conv2D(512, (3, 3),activation='relu',padding='same',name='block5_conv2',kernel_initializer = 'he_normal')(x)
    x = keras.layers.Conv2D(512, (3, 3),activation='relu', padding='same',name='block5_conv3',kernel_initializer = 'he_normal')(x)
    x=keras.layers.BatchNormalization()(x)
    pool_5 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    #DECONV_BLOCK
    #Block_1
    unpool_1=keras.layers.UpSampling2D(size = (2,2),interpolation="nearest")(pool_5)
    conv_14= keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(unpool_1)
    conv_15 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(conv_14)
    conv_16 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(conv_15)
    conv_16= keras.layers.BatchNormalization()(conv_16)
    #Block_2
    unpool_2 = keras.layers.UpSampling2D(size = (2,2),interpolation="nearest")(conv_16)  
    conv_17= keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(unpool_2)
    conv_18 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(conv_17)
    conv_19 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(conv_18)
    conv_19= keras.layers.BatchNormalization()(conv_19)
    #Block_3
    unpool_3 =  keras.layers.UpSampling2D(size = (2,2),interpolation="nearest")(conv_19)   
    conv_20= keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(unpool_3)
    conv_21 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(conv_20)
    conv_22 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(conv_21)
    conv_22= keras.layers.BatchNormalization()(conv_22)
    #Block_4
    unpool_4 = keras.layers.UpSampling2D(size = (2,2),interpolation="nearest")(conv_22)  
    conv_23= keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(unpool_4)
    conv_24 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(conv_23)
    conv_24 = keras.layers.BatchNormalization()(conv_24) 
    #BLock_5
    unpool_5 =keras.layers.UpSampling2D(size = (2,2),interpolation="nearest")(conv_24) 
    conv_25= keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(unpool_5)
    conv_26 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same',kernel_initializer = 'he_normal')(conv_25)
    conv_26 = keras.layers.BatchNormalization()(conv_26)
    out=keras.layers.Conv2D(8,1, activation = 'sigmoid', padding = 'same',kernel_initializer = 'he_normal')(conv_26)
    print("Build decoder done..")
    model = keras.models.Model(inputs=inputs, outputs=out, name="SegNet")
    return model
epochs=300
Adam = optimizers.Adam(lr=0.001,  beta_1=0.9, beta_2=0.9)
def dice_coef(y_true, y_pred, smooth=2):
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice
def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
model=segnet()
model.summary()
model.compile(optimizer=Adam, loss=dice_coef_loss, metrics=[dice_coef])
model.fit(images,labels,validation_data=(test_images,test_labels),batch_size=2, 
                    epochs=epochs)
model.save_weights("SEG_11_nearest_3.h5")
model.load_weights("SEG_11_nearest_3.h5")
result = model.predict(test_images,batch_size=1)
def mean_iou(result,Target):
    P=np.zeros([result.shape[0],256,256,8])
    P[:,:,:,:][np.where(result[:,:,:,:]>0.5)]=1
    predicted=np.reshape(P,(result.shape[0]*8*256*256))
    predicted=predicted.astype(int)
    Target=np.reshape(Target,(result.shape[0]*8*256*256))
    target=Target.astype(int)
    tn, fp, fn, tp=confusion_matrix(target, predicted).ravel()
    print(tp)
    iou=tp/(tp+fn+fp)
    precision=tp/(tp+fp)
    recal=tp/(tp+fn)
    F1=(2*precision*recal)/(precision+recal)
    print("F1_Score is:  ",F1)
    return iou
IOU_result=mean_iou(result,test_labels)

P=np.zeros([12,256,256,8])
P[:,:,:,:][np.where(result[:,:,:,:]>0.5)]=1
import os
P3='/home/user01/data_ssd/Abbas/XEEEE/unet-master/data/membrane/results/'
for i in range(len(result)):
    img=np.zeros([256,256,3]) 
    l1=P[i,:,:,:]
    k1=np.argmax(l1, axis=-1)
    img[np.where(k1==0)]=[128,128,0]
    img[np.where(k1==1)]=[0,0,128]
    img[np.where(k1==2)]=[128,0,0]
    img[np.where(k1==3)]=[0,128,0]
    img[np.where(k1==4)]=[0,128,0]
    img[np.where(k1==5)]=[128,128,128]
    img[np.where(k1==6)]=[128,0,128]
    img[np.where(k1==7)]=[0,0,0]
    cv2.imwrite(os.path.join(P3 , str(i)+".png"),img) 