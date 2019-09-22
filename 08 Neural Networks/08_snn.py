"""
Chapter 8: Neural Networks
Example of Spiking Neural Network (SNN) and MNIST dataset
"""

from keras.datasets import mnist
import numpy as np
from keras.models import Sequential
from keras.layers import (Dense, BatchNormalization,
                          Flatten, Dropout, GaussianNoise)
from keras.layers.convolutional import (Convolution2D,
                                        MaxPooling2D)
from keras.initializers import Constant
from keras.utils import np_utils
from keras.optimizers import Nadam
import matplotlib.pyplot as plt
from time import time
from keras.callbacks import TensorBoard
from sklearn.metrics import (roc_auc_score, log_loss,
                             confusion_matrix)
import keras.backend as K

import nengo
from nengo_extras.keras import (load_model_pair, 
                                save_model_pair,
                                SequentialNetwork,
                                SoftLIF)
from nengo_extras.gui import image_display_function

seed = 14
np.random.seed(seed)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize input values from 0-255 to 0-1
x_train = x_train/np.max([255, np.max(x_train)])
x_test = x_test/np.max([255, np.max(x_test)])
print("Sense check:\n",
      np.min(x_train), np.max(x_train), "\n",
      np.min(x_test), np.max(x_test))

# One hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
nbr_classes = y_test.shape[1]
'''
# Inspect a few images
plt.subplot(221)
plt.imshow(x_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(x_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(x_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(x_train[3], cmap=plt.get_cmap('gray'))
plt.show()
'''
#-------------------------------------------------------------------------------------------------#
#-----------------------------------Spiking Neural Network (SNN)----------------------------------#
#-------------------------------------------------------------------------------------------------#

# Reshape input data for SNN
vector_len = x_train.shape[1]*x_train.shape[2]
full_batch = x_train.shape[0]
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
'''
snn_params = {'use_ocl': True,
              'presentation_time': 0.15,
              'n_presentations': 100,
              'l1_activation': 'relu',
              'l3_activation': 'relu',
              'l5_activation': 'relu',
              'l7_activation': 'softmax',
              'optimizer': 'nadam',
              'epochs': 5,
              'batch_size': 100
              }
def snn_one():
    model = Sequential()
    model.add(Convolution2D(32, kernel_size=(5, 5),
                            strides=(1, 1),
                            padding='valid',
                            activation=snn_params['l1_activation'],
                            bias_initializer='zeros',
                            kernel_initializer='glorot_uniform',
                            input_shape=(1, 28, 28),
                            data_format='channels_first'))
    model.add(SoftLIF(sigma=0.01, amplitude=0.063,
                      tau_rc=0.022, tau_ref=0.002))
    model.add(Convolution2D(16, kernel_size=(2, 2),
                            strides=(1, 1),
                            padding='valid',
                            activation=snn_params['l3_activation'],
                            bias_initializer='zeros',
                            kernel_initializer='glorot_uniform'))
    model.add(SoftLIF(sigma=0.01, amplitude=0.063,
                      tau_rc=0.022, tau_ref=0.002))
    model.add(Flatten())
    model.add(Dense(128,
                    activation=snn_params['l5_activation']))
    model.add(SoftLIF(sigma=0.01, amplitude=0.063,
                      tau_rc=0.022, tau_ref=0.002))
    model.add(Dropout(0.5))
    model.add(Dense(nbr_classes,
                    bias_initializer='zeros',
                    kernel_initializer='normal',
                    activation=snn_params['l7_activation']))
    model.compile(loss='categorical_crossentropy',
                  optimizer=snn_params['optimizer'],
                  metrics=['accuracy'])
    return model

"""
Note: The line below starts TensorBoard from the command prompt.
Run it after training the MLP.
tensorboard --logdir="dsilt-ml-code/08 Neural Networks/logs"
"""
callbacks = [
    TensorBoard(
        log_dir='dsilt-ml-code/08 Neural Networks/logs/snn1_{}'.format(time()),
        histogram_freq=2,
        batch_size=snn_params['batch_size'],
        write_graph=True,
        write_grads=True,
        write_images=True
        )
    ]
snn_model = snn_one()
print(snn_model.summary())
print("Initial loss for softmax layer should be approximately:", \
      -np.log(1/nbr_classes))
snn_model_hist = snn_model.fit(x_train,
                               y_train,
                               validation_data=\
                               (x_test,
                                y_test),
                               epochs=\
                               snn_params['epochs'],
                               batch_size=\
                               snn_params['batch_size'],
                               verbose=2,
                               callbacks=callbacks)
snn_scores = snn_model.evaluate(x_test,
                                y_test,
                                verbose=0)
print("Baseline classification error: {}"\
      .format(round(100-snn_scores[1]*100, 4)))
# Evaluation
snn_preds_prob = snn_model.predict(x_test)
snn_preds = [np.argmax(x) for x in snn_preds_prob]
true_labels = [np.argmax(x) for x in y_test]
print("SNN Log Loss", log_loss(true_labels, snn_preds_prob))
print("SNNP Confusion Matrix\n", confusion_matrix(true_labels, snn_preds))
'''



'''

snn_params = {'use_ocl': True,
              'presentation_time': 0.15,
              'n_presentations': 100,
              'l1_activation': 'relu',
              'l3_activation': 'relu',
              'l5_activation': 'relu',
              'l7_activation': 'softmax',
              'optimizer': 'nadam',
              'epochs': 5,
              'batch_size': 100
              }
def snn_two():
    model = Sequential()
    model.add(Convolution2D(32, kernel_size=(5, 5),
                            strides=(1, 1),
                            padding='valid',
                            activation=snn_params['l1_activation'],
                            bias_initializer='zeros',
                            kernel_initializer='glorot_uniform',
                            input_shape=(1, 28, 28),
                            data_format='channels_first'))
    model.add(SoftLIF(sigma=0.01, amplitude=0.063,
                      tau_rc=0.022, tau_ref=0.002))
    model.add(Flatten())
    model.add(Dense(nbr_classes,
                    bias_initializer='zeros',
                    kernel_initializer='normal',
                    activation=snn_params['l7_activation']))
    model.compile(loss='categorical_crossentropy',
                  optimizer=snn_params['optimizer'],
                  metrics=['accuracy'])
    return model

"""
Note: The line below starts TensorBoard from the command prompt.
Run it after training the MLP.
tensorboard --logdir="dsilt-ml-code/08 Neural Networks/logs"
"""
callbacks = [
    TensorBoard(
        log_dir='dsilt-ml-code/08 Neural Networks/logs/snn2_{}'.format(time()),
        histogram_freq=2,
        batch_size=snn_params['batch_size'],
        write_graph=True,
        write_grads=True,
        write_images=True
        )
    ]
snn_model = snn_two()
print(snn_model.summary())
print("Initial loss for softmax layer should be approximately:", \
      -np.log(1/nbr_classes))
snn_model_hist = snn_model.fit(x_train,
                               y_train,
                               validation_data=\
                               (x_test,
                                y_test),
                               epochs=\
                               snn_params['epochs'],
                               batch_size=\
                               snn_params['batch_size'],
                               verbose=2,
                               callbacks=callbacks)
snn_scores = snn_model.evaluate(x_test,
                                y_test,
                                verbose=0)
'''







snn_params = {'use_ocl': True,
              'presentation_time': 0.15,
              'n_presentations': 100,
              'l1_activation': 'relu',
              'l3_activation': 'relu',
              'l5_activation': 'relu',
              'l7_activation': 'softmax',
              'optimizer': 'adam',
              'epochs': 5,
              'batch_size': 100
              }
def snn_two():
    model = Sequential()
    #model.add(GaussianNoise(0.1, input_shape=(1, 28, 28)))
    model.add(Convolution2D(32, kernel_size=(5, 5),
                            strides=(1, 1),
                            padding='valid',
                            activation=snn_params['l1_activation'],
                            bias_initializer='zeros',
                            kernel_initializer='glorot_uniform',
                            input_shape=(1, 28, 28),
                            data_format='channels_first'))
    model.add(SoftLIF(sigma=0.1, amplitude=0.001,#sigma=0.01, amplitude=0.063,
                      tau_rc=0.02, tau_ref=0.002))
    model.add(Flatten())
    model.add(Dense(nbr_classes,
                    bias_initializer='zeros',
                    kernel_initializer='normal',
                    activation=snn_params['l7_activation']))
    model.compile(loss='categorical_crossentropy',
                  optimizer=snn_params['optimizer'],
                  metrics=['accuracy'])
    return model

"""
Note: The line below starts TensorBoard from the command prompt.
Run it after training the MLP.
tensorboard --logdir="dsilt-ml-code/08 Neural Networks/logs"
"""
callbacks = [
    TensorBoard(
        log_dir='dsilt-ml-code/08 Neural Networks/logs/snn2_{}'.format(time()),
        histogram_freq=2,
        batch_size=snn_params['batch_size'],
        write_graph=True,
        write_grads=True,
        write_images=True
        )
    ]
snn_model = snn_two()
print(snn_model.summary())
print("Initial loss for softmax layer should be approximately:", \
      -np.log(1/nbr_classes))
snn_model_hist = snn_model.fit(x_train,
                               y_train,
                               validation_data=\
                               (x_test,
                                y_test),
                               epochs=\
                               snn_params['epochs'],
                               batch_size=\
                               snn_params['batch_size'],
                               verbose=2,
                               callbacks=callbacks)
snn_scores = snn_model.evaluate(x_test,
                                y_test,
                                verbose=0)
