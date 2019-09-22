"""
Chapter 8: Neural Networks
Example of Convolutional Neural Network (CNN) with Keras and MNIST dataset
"""

from keras.datasets import mnist
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Flatten, Dropout, LSTM
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.initializers import Constant
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
from time import time
from keras.callbacks import TensorBoard
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix
import keras.backend as K

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

#-------------------------------------------------------------------------------------------------#
#-----------------------------------Multilayer Perceptron (MLP)-----------------------------------#
#-------------------------------------------------------------------------------------------------#

# Reshape input data to 1 dimension for MLP
vector_len = x_train.shape[1]*x_train.shape[2]
full_batch = x_train.shape[0]
x_train_flat = x_train.reshape(x_train.shape[0],
                               vector_len).astype('float32')
x_test_flat = x_test.reshape(x_test.shape[0],
                             vector_len).astype('float32')

"""
First set up a baseline model with linear activations in the first layer,
no batches (train on entire dataset), and only 10 epochs.
"""
baseline_params = {'l1_activation': 'linear',
                   'l2_activation': 'softmax',
                   'optimizer': 'sgd',
                   'epochs': 20,
                   'batch_size': full_batch #Full size/no batch
                   }

def mlp_one():
    model = Sequential()
    model.add(Dense(vector_len, input_dim=vector_len,
                    bias_initializer='zeros',
                    kernel_initializer='normal',
                    activation=baseline_params['l1_activation']))
    model.add(Dense(nbr_classes,
                    bias_initializer='zeros',
                    kernel_initializer='normal',
                    activation=baseline_params['l2_activation']))
    model.compile(loss='categorical_crossentropy',
                  optimizer=baseline_params['optimizer'],
                  metrics=['accuracy'])
    return model

"""
Note: The line below starts TensorBoard from the command prompt.
Run it after training the MLP.
tensorboard --logdir="dsilt-ml-code/08 Neural Networks/logs"
"""
callbacks = [
    TensorBoard(
        log_dir='dsilt-ml-code/08 Neural Networks/logs/mlp1_{}'.format(time()),
        histogram_freq=2,
        batch_size=baseline_params['batch_size'],
        write_graph=True,
        write_grads=True,
        write_images=True,
        #embeddings_freq=1,
        #embeddings_data=x_train_flat[:1000], # Limit to 1k samples
        )
    ]

baseline_mlp_model = mlp_one()
print(baseline_mlp_model.summary())
print("Initial loss for softmax layer should be approximately:", \
      -np.log(1/nbr_classes))
baseline_mlp_model_hist = baseline_mlp_model.fit(x_train_flat,
                                                 y_train,
                                                 validation_data=\
                                                 (x_test_flat,
                                                  y_test),
                                                 epochs=\
                                                 baseline_params['epochs'],
                                                 batch_size=\
                                                 baseline_params['batch_size'],
                                                 verbose=2,
                                                 callbacks=callbacks)
baseline_mlp_scores = baseline_mlp_model.evaluate(x_test_flat,
                                                  y_test,
                                                  verbose=0)
print("Baseline classification error: {}"\
      .format(round(100-baseline_mlp_scores[1]*100, 4)))
plt.plot(baseline_mlp_model_hist.history['acc'])
plt.plot(baseline_mlp_model_hist.history['val_acc'])
plt.title('Baseline MLP Model Training Curve')
plt.ylabel('Classification Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.plot(baseline_mlp_model_hist.history['loss'])
plt.plot(baseline_mlp_model_hist.history['val_loss'])
plt.title('Baseline MLP Model Training Curve')
plt.ylabel('Categorical Cross-Entropy Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# Evaluation
baseline_mlp_preds_prob = baseline_mlp_model.predict(x_test_flat)
baseline_mlp_preds = [np.argmax(x) for x in baseline_mlp_preds_prob]
true_labels = [np.argmax(x) for x in y_test]
print("Baseline MLP Log Loss", log_loss(true_labels, baseline_mlp_preds_prob))
print("Baseline MLP Confusion Matrix\n", confusion_matrix(true_labels, baseline_mlp_preds))

"""
Continue modeling the data, making adjustments to improve the model
and looking at Tensorboard to see if the adjustments helped
"""
# Use relu activation in l1
K.clear_session()
mlp_params = {'l1_activation': 'relu',
              'l2_activation': 'softmax',
              'optimizer': 'sgd',
              'epochs': 20,
              'batch_size': full_batch #Full size/no batch
              }
def mlp_two():
    model = Sequential()
    model.add(Dense(vector_len, input_dim=vector_len,
                    bias_initializer='zeros',
                    kernel_initializer='normal',
                    activation=mlp_params['l1_activation']))
    model.add(Dense(nbr_classes,
                    bias_initializer='zeros',
                    kernel_initializer='normal',
                    activation=mlp_params['l2_activation']))
    model.compile(loss='categorical_crossentropy',
                  optimizer=mlp_params['optimizer'],
                  metrics=['accuracy'])
    return model
callbacks = [
    TensorBoard(
        log_dir='dsilt-ml-code/08 Neural Networks/logs/mlp2_{}'.format(time()),
        histogram_freq=2,
        batch_size=mlp_params['batch_size'],
        write_graph=True,
        write_grads=True,
        write_images=True,
        )
    ]
mlp_model = mlp_two()
print(mlp_model.summary())
print("Initial loss for softmax layer should be approximately:", \
      -np.log(1/nbr_classes))
mlp_model_hist = mlp_model.fit(x_train_flat,
                               y_train,
                               validation_data=\
                               (x_test_flat,
                                y_test),
                               epochs=mlp_params['epochs'],
                               batch_size=\
                               mlp_params['batch_size'],
                               verbose=2,
                               callbacks=callbacks)
mlp_scores = mlp_model.evaluate(x_test_flat,
                                y_test,
                                verbose=0)


# Use a smaller learning rate
K.clear_session()
sgd = SGD(lr=0.001, momentum=0.0, decay=0.0)
mlp_params = {'l1_activation': 'relu',
              'l2_activation': 'softmax',
              'optimizer': sgd,
              'epochs': 20,
              'batch_size': full_batch #Full size/no batch
              }
def mlp_three():
    model = Sequential()
    model.add(Dense(vector_len, input_dim=vector_len,
                    bias_initializer='zeros',
                    kernel_initializer='normal',
                    activation=mlp_params['l1_activation']))
    model.add(Dense(nbr_classes,
                    bias_initializer='zeros',
                    kernel_initializer='normal',
                    activation=mlp_params['l2_activation']))
    model.compile(loss='categorical_crossentropy',
                  optimizer=mlp_params['optimizer'],
                  metrics=['accuracy'])
    return model
callbacks = [
    TensorBoard(
        log_dir='dsilt-ml-code/08 Neural Networks/logs/mlp3_{}'.format(time()),
        histogram_freq=2,
        batch_size=mlp_params['batch_size'],
        write_graph=True,
        write_grads=True,
        write_images=True,
        )
    ]
mlp_model = mlp_three()
print(mlp_model.summary())
print("Initial loss for softmax layer should be approximately:", \
      -np.log(1/nbr_classes))
mlp_model_hist = mlp_model.fit(x_train_flat,
                               y_train,
                               validation_data=\
                               (x_test_flat,
                                y_test),
                               epochs=mlp_params['epochs'],
                               batch_size=\
                               mlp_params['batch_size'],
                               verbose=2,
                               callbacks=callbacks)
mlp_scores = mlp_model.evaluate(x_test_flat,
                                y_test,
                                verbose=0)

# Go back to original learning rate but add batch normalization
#https://www.dlology.com/blog/one-simple-trick-to-train-keras-model-faster-with-batch-normalization/
#https://stackoverflow.com/questions/46256747/can-not-use-both-bias-and-batch-normalization-in-convolution-layers
K.clear_session()
mlp_params = {'l1_activation': 'relu',
              'l2_activation': 'softmax',
              'optimizer': 'sgd',
              'epochs': 20,
              'batch_size': full_batch #Full size/no batch
              }
def mlp_four():
    model = Sequential()
    model.add(Dense(vector_len, input_dim=vector_len,
                    use_bias=False,  #Turn bias off bc batch norm includes a bias
                    #bias_initializer='zeros',
                    kernel_initializer='normal',
                    activation=mlp_params['l1_activation']))
    model.add(BatchNormalization())
    model.add(Dense(nbr_classes,
                    bias_initializer='zeros',
                    kernel_initializer='normal',
                    activation=mlp_params['l2_activation']))
    model.compile(loss='categorical_crossentropy',
                  optimizer=mlp_params['optimizer'],
                  metrics=['accuracy'])
    return model
callbacks = [
    TensorBoard(
        log_dir='dsilt-ml-code/08 Neural Networks/logs/mlp4_{}'.format(time()),
        histogram_freq=2,
        batch_size=mlp_params['batch_size'],
        write_graph=True,
        write_grads=True,
        write_images=True,
        )
    ]
mlp_model = mlp_four()
print(mlp_model.summary())
print("Initial loss for softmax layer should be approximately:", \
      -np.log(1/nbr_classes))
mlp_model_hist = mlp_model.fit(x_train_flat,
                               y_train,
                               validation_data=\
                               (x_test_flat,
                                y_test),
                               epochs=mlp_params['epochs'],
                               batch_size=\
                               mlp_params['batch_size'],
                               verbose=2,
                               callbacks=callbacks)
mlp_scores = mlp_model.evaluate(x_test_flat,
                                y_test,
                                verbose=0)


# Use adam optimizer with a small learning rate
K.clear_session()
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,
            epsilon=None, decay=0.0)
mlp_params = {'l1_activation': 'relu',
              'l2_activation': 'softmax',
              'optimizer': adam,  #To use default adam, replace with 'adam'
              'epochs': 20,
              'batch_size': full_batch #Full size/no batch
              }
def mlp_five():
    model = Sequential()
    model.add(Dense(vector_len, input_dim=vector_len,
                    bias_initializer='zeros',
                    kernel_initializer='normal',
                    activation=mlp_params['l1_activation']))
    model.add(Dense(nbr_classes,
                    bias_initializer='zeros',
                    kernel_initializer='normal',
                    activation=mlp_params['l2_activation']))
    model.compile(loss='categorical_crossentropy',
                  optimizer=mlp_params['optimizer'],
                  metrics=['accuracy'])
    return model
callbacks = [
    TensorBoard(
        log_dir='dsilt-ml-code/08 Neural Networks/logs/mlp5_{}'.format(time()),
        histogram_freq=2,
        batch_size=mlp_params['batch_size'],
        write_graph=True,
        write_grads=True,
        write_images=True,
        )
    ]
mlp_model = mlp_five()
print(mlp_model.summary())
print("Initial loss for softmax layer should be approximately:", \
      -np.log(1/nbr_classes))
mlp_model_hist = mlp_model.fit(x_train_flat,
                               y_train,
                               validation_data=\
                               (x_test_flat,
                                y_test),
                               epochs=mlp_params['epochs'],
                               batch_size=\
                               mlp_params['batch_size'],
                               verbose=2,
                               callbacks=callbacks)
mlp_scores = mlp_model.evaluate(x_test_flat,
                                y_test,
                                verbose=0)


# Increase learning rate, add a hidden layer l2, and change l2 to l3
K.clear_session()
mlp_params = {'l1_activation': 'relu',
              'l2_activation': 'relu',
              'l3_activation': 'softmax',
              'optimizer': 'adam',
              'epochs': 20,
              'batch_size': full_batch #Full size/no batch
              }
def mlp_six():
    model = Sequential()
    model.add(Dense(vector_len, input_dim=vector_len,
                    bias_initializer='zeros',
                    kernel_initializer='normal',
                    activation=mlp_params['l1_activation']))
    model.add(Dense(units=50,
                    bias_initializer='zeros',
                    kernel_initializer='normal',
                    activation=mlp_params['l2_activation']))
    model.add(Dense(nbr_classes,
                    bias_initializer='zeros',
                    kernel_initializer='normal',
                    activation=mlp_params['l3_activation']))
    model.compile(loss='categorical_crossentropy',
                  optimizer=mlp_params['optimizer'],
                  metrics=['accuracy'])
    return model
callbacks = [
    TensorBoard(
        log_dir='dsilt-ml-code/08 Neural Networks/logs/mlp6_{}'.format(time()),
        histogram_freq=2,
        batch_size=mlp_params['batch_size'],
        write_graph=True,
        write_grads=True,
        write_images=True,
        )
    ]
mlp_model = mlp_six()
print(mlp_model.summary())
print("Initial loss for softmax layer should be approximately:", \
      -np.log(1/nbr_classes))
mlp_model_hist = mlp_model.fit(x_train_flat,
                               y_train,
                               validation_data=\
                               (x_test_flat,
                                y_test),
                               epochs=mlp_params['epochs'],
                               batch_size=\
                               mlp_params['batch_size'],
                               verbose=2,
                               callbacks=callbacks)
mlp_scores = mlp_model.evaluate(x_test_flat,
                                y_test,
                                verbose=0)


# Widen the network by giving the hidden layer (l2) more units
K.clear_session()
mlp_params = {'l1_activation': 'relu',
              'l2_activation': 'relu',
              'l3_activation': 'softmax',
              'optimizer': 'adam',
              'epochs': 20,
              'batch_size': full_batch #Full size/no batch
              }
def mlp_seven():
    model = Sequential()
    model.add(Dense(vector_len, input_dim=vector_len,
                    bias_initializer='zeros',
                    kernel_initializer='normal',
                    activation=mlp_params['l1_activation']))
    model.add(Dense(units=150,
                    bias_initializer='zeros',
                    kernel_initializer='normal',
                    activation=mlp_params['l2_activation']))
    model.add(Dense(nbr_classes,
                    bias_initializer='zeros',
                    kernel_initializer='normal',
                    activation=mlp_params['l3_activation']))
    model.compile(loss='categorical_crossentropy',
                  optimizer=mlp_params['optimizer'],
                  metrics=['accuracy'])
    return model
callbacks = [
    TensorBoard(
        log_dir='dsilt-ml-code/08 Neural Networks/logs/mlp7_{}'.format(time()),
        histogram_freq=2,
        batch_size=mlp_params['batch_size'],
        write_graph=True,
        write_grads=True,
        write_images=True,
        )
    ]
mlp_model = mlp_seven()
print(mlp_model.summary())
print("Initial loss for softmax layer should be approximately:", \
      -np.log(1/nbr_classes))
mlp_model_hist = mlp_model.fit(x_train_flat,
                               y_train,
                               validation_data=\
                               (x_test_flat,
                                y_test),
                               epochs=mlp_params['epochs'],
                               batch_size=\
                               mlp_params['batch_size'],
                               verbose=2,
                               callbacks=callbacks)
mlp_scores = mlp_model.evaluate(x_test_flat,
                                y_test,
                                verbose=0)


# Add more layers
K.clear_session()
mlp_params = {'l1_activation': 'relu',
              'l2_activation': 'relu',
              'l3_activation': 'relu',
              'l4_activation': 'relu',
              'l5_activation': 'relu',
              'l6_activation': 'softmax',
              'optimizer': 'adam',
              'epochs': 20,
              'batch_size': full_batch #Full size/no batch
              }
def mlp_eight():
    model = Sequential()
    model.add(Dense(vector_len, input_dim=vector_len,
                    bias_initializer='zeros',
                    kernel_initializer='normal',
                    activation=mlp_params['l1_activation']))
    model.add(Dense(units=150,
                    bias_initializer='zeros',
                    kernel_initializer='normal',
                    activation=mlp_params['l2_activation']))
    model.add(Dense(units=50,
                    bias_initializer='zeros',
                    kernel_initializer='normal',
                    activation=mlp_params['l3_activation']))
    model.add(Dense(units=15,
                    bias_initializer='zeros',
                    kernel_initializer='normal',
                    activation=mlp_params['l4_activation']))
    model.add(Dense(units=100,
                    bias_initializer='zeros',
                    kernel_initializer='normal',
                    activation=mlp_params['l5_activation']))
    model.add(Dense(nbr_classes,
                    bias_initializer='zeros',
                    kernel_initializer='normal',
                    activation=mlp_params['l6_activation']))
    model.compile(loss='categorical_crossentropy',
                  optimizer=mlp_params['optimizer'],
                  metrics=['accuracy'])
    return model
callbacks = [
    TensorBoard(
        log_dir='dsilt-ml-code/08 Neural Networks/logs/mlp8_{}'.format(time()),
        histogram_freq=2,
        batch_size=mlp_params['batch_size'],
        write_graph=True,
        write_grads=True,
        write_images=True,
        )
    ]
mlp_model = mlp_eight()
print(mlp_model.summary())
print("Initial loss for softmax layer should be approximately:", \
      -np.log(1/nbr_classes))
mlp_model_hist = mlp_model.fit(x_train_flat,
                               y_train,
                               validation_data=\
                               (x_test_flat,
                                y_test),
                               epochs=mlp_params['epochs'],
                               batch_size=\
                               mlp_params['batch_size'],
                               verbose=2,
                               callbacks=callbacks)
mlp_scores = mlp_model.evaluate(x_test_flat,
                                y_test,
                                verbose=0)


#-------------------------------------------------------------------------------------------------#
#-------------------------------Convolutional Neural Network (CNN)--------------------------------#
#-------------------------------------------------------------------------------------------------#

# Reshape input data for CNN
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')
full_batch = x_train.shape[0]
print("Input image shape:", x_train[0].shape)

# Baseline CNN
K.clear_session()
baseline_params = {'l1_activation': 'relu',
                   'l2_activation': 'softmax',
                   'optimizer': 'adam',
                   'epochs': 5,      #Adjust to what works for the computer
                   'batch_size': 200 #Adjust to what works for the computer
                   }

def cnn_one():
    model = Sequential()
    model.add(Convolution2D(32, kernel_size=(5, 5),
                            strides=(1, 1),
                            padding='valid',
                            activation=baseline_params['l1_activation'],
                            bias_initializer='zeros',
                            kernel_initializer='glorot_uniform',
                            input_shape=(1, 28, 28),
                            data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(nbr_classes,
                    bias_initializer='zeros',
                    kernel_initializer='normal',
                    activation=baseline_params['l2_activation']))
    model.compile(loss='categorical_crossentropy',
                  optimizer=baseline_params['optimizer'],
                  metrics=['accuracy'])
    return model

"""
Note: The line below starts TensorBoard from the command prompt.
Run it after training the model.
tensorboard --logdir="dsilt-ml-code/08 Neural Networks/logs"
"""
callbacks = [
    TensorBoard(
        log_dir='dsilt-ml-code/08 Neural Networks/logs/cnn1_{}'.format(time()),
        histogram_freq=2,
        batch_size=baseline_params['batch_size'],
        write_graph=True,
        write_grads=True,
        write_images=True,
        )
    ]

baseline_cnn_model = cnn_one()
print(baseline_cnn_model.summary())
print("Initial loss for softmax layer should be approximately:", \
      -np.log(1/nbr_classes))
baseline_cnn_model_hist = baseline_cnn_model.fit(x_train,
                                                 y_train,
                                                 validation_data=\
                                                 (x_test,
                                                  y_test),
                                                 epochs=\
                                                 baseline_params['epochs'],
                                                 batch_size=\
                                                 baseline_params['batch_size'],
                                                 verbose=2,
                                                 callbacks=callbacks)
baseline_cnn_scores = baseline_cnn_model.evaluate(x_test,
                                                  y_test,
                                                  verbose=0)

print("Baseline classification error: {}"\
      .format(round(100-baseline_cnn_scores[1]*100, 4)))
plt.plot(baseline_cnn_model_hist.history['acc'])
plt.plot(baseline_cnn_model_hist.history['val_acc'])
plt.title('Baseline CNN Model Training Curve')
plt.ylabel('Classification Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.plot(baseline_cnn_model_hist.history['loss'])
plt.plot(baseline_cnn_model_hist.history['val_loss'])
plt.title('Baseline CNN Model Training Curve')
plt.ylabel('Categorical Cross-Entropy Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# Evaluation
baseline_cnn_preds_prob = baseline_cnn_model.predict(x_test)
baseline_cnn_preds = [np.argmax(x) for x in baseline_cnn_preds_prob]
true_labels = [np.argmax(x) for x in y_test]
print("Baseline CNN Log Loss", log_loss(true_labels, baseline_cnn_preds_prob))
print("Baseline CNN Confusion Matrix\n", confusion_matrix(true_labels, baseline_cnn_preds))


# Add a dense layer after the first conv layer
K.clear_session()
cnn_params = {'l1_activation': 'relu',
              'l2_activation': 'relu',
              'l3_activation': 'softmax',
              'optimizer': 'adam',
              'epochs': 10,     #Adjust to what works for the computer
              'batch_size': 200 #Adjust to what works for the computer
              }
def cnn_two():
    model = Sequential()
    model.add(Convolution2D(32, kernel_size=(5, 5),
                            strides=(1, 1),
                            padding='valid',
                            activation=cnn_params['l1_activation'],
                            bias_initializer='zeros',
                            kernel_initializer='glorot_uniform',
                            input_shape=(1, 28, 28),
                            data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation=cnn_params['l2_activation']))
    model.add(Dense(nbr_classes,
                    bias_initializer='zeros',
                    kernel_initializer='normal',
                    activation=cnn_params['l3_activation']))
    model.compile(loss='categorical_crossentropy',
                  optimizer=cnn_params['optimizer'],
                  metrics=['accuracy'])
    return model
callbacks = [
    TensorBoard(
        log_dir='dsilt-ml-code/08 Neural Networks/logs/cnn2_{}'.format(time()),
        histogram_freq=2,
        batch_size=cnn_params['batch_size'],
        write_graph=True,
        write_grads=True,
        write_images=True,
        )
    ]
cnn_model = cnn_two()
print(cnn_model.summary())
print("Initial loss for softmax layer should be approximately:", \
      -np.log(1/nbr_classes))
cnn_model_hist = cnn_model.fit(x_train,
                               y_train,
                               validation_data=\
                               (x_test,
                                y_test),
                               epochs=\
                               cnn_params['epochs'],
                               batch_size=\
                               cnn_params['batch_size'],
                               verbose=2,
                               callbacks=callbacks)
cnn_scores = cnn_model.evaluate(x_test,
                                y_test,
                                verbose=0)


# Add dropout to regularize
K.clear_session()
cnn_params = {'l1_activation': 'relu',
              'l2_activation': 'relu',
              'l3_activation': 'softmax',
              'optimizer': 'adam',
              'epochs': 10,     #Adjust to what works for the computer
              'batch_size': 200 #Adjust to what works for the computer
              }
def cnn_three():
    model = Sequential()
    model.add(Convolution2D(32, kernel_size=(5, 5),
                            strides=(1, 1),
                            padding='valid',
                            activation=cnn_params['l1_activation'],
                            bias_initializer='zeros',
                            kernel_initializer='glorot_uniform',
                            input_shape=(1, 28, 28),
                            data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation=cnn_params['l2_activation']))
    model.add(Dense(nbr_classes,
                    bias_initializer='zeros',
                    kernel_initializer='normal',
                    activation=cnn_params['l3_activation']))
    model.compile(loss='categorical_crossentropy',
                  optimizer=cnn_params['optimizer'],
                  metrics=['accuracy'])
    return model
callbacks = [
    TensorBoard(
        log_dir='dsilt-ml-code/08 Neural Networks/logs/cnn3_{}'.format(time()),
        histogram_freq=2,
        batch_size=cnn_params['batch_size'],
        write_graph=True,
        write_grads=True,
        write_images=True,
        )
    ]
cnn_model = cnn_three()
print(cnn_model.summary())
print("Initial loss for softmax layer should be approximately:", \
      -np.log(1/nbr_classes))
cnn_model_hist = cnn_model.fit(x_train,
                               y_train,
                               validation_data=\
                               (x_test,
                                y_test),
                               epochs=\
                               cnn_params['epochs'],
                               batch_size=\
                               cnn_params['batch_size'],
                               verbose=2,
                               callbacks=callbacks)
cnn_scores = cnn_model.evaluate(x_test,
                                y_test,
                                verbose=0)


#-------------------------------------------------------------------------------------------------#
#-------------------------------Long Short Term Memory Network (LSTM)-----------------------------#
#-------------------------------------------------------------------------------------------------#

# Check input data shape for LSTM
print("Input shape:", x_train.shape)
print("Input image shape:", x_train[0].shape)


# Create a basic LSTM with 2 recurrent layers
K.clear_session()
lstm_params = {'l1_activation': 'relu',
               'l1_rec_activation': 'hard_sigmoid',
               'l2_activation': 'relu',
               'l2_rec_activation': 'hard_sigmoid',
               'l3_activation': 'relu',
               'l4_activation': 'softmax',
               'optimizer': 'adam',
               'epochs': 5,      #Adjust to what works for the computer
               'batch_size': 200 #Adjust to what works for the computer
               }
def lstm_one():
    model = Sequential()
    model.add(LSTM(128,
                   activation=lstm_params['l1_activation'],
                   recurrent_activation=lstm_params['l1_rec_activation'],
                   bias_initializer='zeros',
                   kernel_initializer='glorot_uniform',
                   recurrent_initializer='orthogonal',
                   input_shape=(28, 28),
                   return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(64,
                   activation=lstm_params['l1_activation'],
                   recurrent_activation=lstm_params['l1_rec_activation'],
                   bias_initializer='zeros',
                   kernel_initializer='glorot_uniform',
                   recurrent_initializer='orthogonal',
                   return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation=lstm_params['l3_activation']))
    model.add(Flatten())
    model.add(Dense(nbr_classes,
                    bias_initializer='zeros',
                    kernel_initializer='normal',
                    activation=lstm_params['l4_activation']))
    model.compile(loss='categorical_crossentropy',
                  optimizer=lstm_params['optimizer'],
                  metrics=['accuracy'])
    return model
callbacks = [
    TensorBoard(
        log_dir='dsilt-ml-code/08 Neural Networks/logs/lstm1_{}'.format(time()),
        histogram_freq=2,
        batch_size=lstm_params['batch_size'],
        write_graph=True,
        write_grads=True,
        write_images=True,
        )
    ]
lstm_model = lstm_one()
print(lstm_model.summary())
print("Initial loss for softmax layer should be approximately:", \
      -np.log(1/nbr_classes))
cnn_model_hist = lstm_model.fit(x_train,
                                y_train,
                                validation_data=\
                                (x_test,
                                 y_test),
                                epochs=\
                                lstm_params['epochs'],
                                batch_size=\
                                lstm_params['batch_size'],
                                verbose=2,
                                callbacks=callbacks)
lstm_scores = lstm_model.evaluate(x_test,
                                  y_test,
                                  verbose=0)
