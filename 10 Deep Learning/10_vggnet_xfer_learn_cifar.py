from keras import backend as K
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from scipy.misc import imresize
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import (ZeroPadding2D,
                          Convolution2D,
                          MaxPooling2D, Dense,
                          Flatten, Dropout)
from keras.optimizers import SGD
import wget
from sklearn.metrics import log_loss, accuracy_score


config = {
    'nbr_train_samples': 3000,
    'nbr_valid_samples': 300,
    'nbr_classes': 10,
    'img_rows': 224,
    'img_cols': 224,
    'color_channels': 3,
    'batch_size': 32
    'epochs': 20
    }


#-------------------------------------------------------------------------------------------------#
#-----------------------Load and Prepare the CIFAR-10 Data----------------------------------------#
#-------------------------------------------------------------------------------------------------#

K.set_image_dim_ordering('tf')

(x_train, y_train), (x_valid, y_valid) = cifar10.load_data()
print(x_train.shape, x_valid.shape)
print(K.image_dim_ordering())
# Ensure that dim ordering matches image shape
# (nbr_samples, nbr_channels, rows, cols) is 'th' and (nbr_samples, rows, cols, nbr_channels) is 'tf'

plt.imshow(x_train[0])
plt.show()


def preprocess_for_vgg(input_array):
    '''
    This function takes a numpy array of pixels (h, w, channel) and 
    outputs a numpy array that is standardized to mean 0 and is in BGR channel
    order.  This channel order matches the requirements for VGG16.
    '''
    rgb_means = np.array([123.68, 116.779, 103.939], 
                         dtype=np.float32).reshape((3,1,1))
    # Assuming the ordering is for tensorflow, reshape the image to have desired input dimensions
    input_array = imresize(input_array, (config['img_rows'], config['img_cols']))
    # Transpose dimension ordering from tf format (h,w,c) to th format (c,h,w), which is required for vgg
    input_array_t = input_array.transpose(2,0,1)
    # Standardize to mean 0
    output_array = input_array_t - rgb_means
    # Reverse color channels from RGB to BGR
    output_array = output_array[:, ::-1]
    return output_array


x_train = np.array([preprocess_for_vgg(im) for im in x_train[:config['nbr_train_samples']:,:,:]])
x_valid = np.array([preprocess_for_vgg(im) for im in x_valid[:config['nbr_valid_samples']:,:,:]])
print(x_train.shape, x_valid.shape)


K.set_image_dim_ordering('th')

# Convert the classes to categorical format (the format Keras expects)
y_train = np_utils.to_categorical(y_train[:config['nbr_train_samples']], config['nbr_classes'])
y_valid = np_utils.to_categorical(y_valid[:config['nbr_valid_samples']], config['nbr_classes'])


# Make sure the input shape is correct
if K.image_dim_ordering()=='th':
    inshape = (config['color_channels'], config['img_rows'], config['img_cols'])
else:
    inshape = (config['img_rows'], config['img_cols'], config['color_channels'])


#-------------------------------------------------------------------------------------------------#
#----------------------------------------Build VGG16----------------------------------------------#
#-------------------------------------------------------------------------------------------------#

model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=inshape))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='softmax'))

print(model.summary())

# Download VGG16 pre-trained weights
#url = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5'
#wget.download(url, 'vgg16_weights_th_dim_ordering_th_kernels.h5')

# Load the pre-trained ImageNet weights
model.load_weights('vgg16_weights_th_dim_ordering_th_kernels.h5')


#-------------------------------------------------------------------------------------------------#
#---------------------------------------Fine Tune VGG16-------------------------------------------#
#-------------------------------------------------------------------------------------------------#

def replace_last_layer_lock_the_rest(m):
    '''
    This function takes a model as input and outputs the model with a new
    final layer.  It replaces the final layer with a fully connected 
    layer for classification.  All layers, except the new final layer, are 
    locked so their weights cannot change.  
    '''
    m.pop()
    for layer in m.layers: 
        layer.trainable=False
    m.add(Dense(config['nbr_classes'], activation='softmax'))
    return m

model = replace_last_layer_lock_the_rest(model)
print(model.summary())


# Optimize with a smaller learning rate than what vgg16 was originally trained with
# Good heuristic is to use 1-2 orders of magnitude smaller than original learning rate
opt_alg = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=opt_alg, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=config['batch_size'], epochs=config['epochs'],
          shuffle=True, verbose=2, validation_data=(x_valid, y_valid))

val_preds_prob = model.predict(x_valid, batch_size=config['batch_size'], verbose=1)
val_preds = np.argmax(val_preds_prob, axis=1)

final_validation_score = log_loss(y_valid, val_preds_prob)
final_validation_acc = accuracy_score(np.argmax(y_valid, axis=1), val_preds)
print("Cross entropy validation loss:", final_validation_score)
print("Validation accuracy:", final_validation_acc)
