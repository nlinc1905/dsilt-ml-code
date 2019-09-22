import numpy as np
from keras.models import Model
from keras.layers import (Input, Conv2D, ZeroPadding2D,
                          MaxPooling2D, UpSampling2D,
                          Cropping2D, Flatten)
from keras.optimizers import Adadelta
from keras.callbacks import Callback, TensorBoard
from time import time


# Create a custom callback to stop training the autoencoder when a val_loss of 0.1 is reached
class EarlyStoppingByLossVal(Callback):
    
    def __init__(self, monitor='val_loss', value=0.1, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


class Autoencoder():

    def __init__(self, params_dict):
        self.params = params_dict
        self.adadelta = Adadelta(lr=self.params['learning_rate'],
                                 rho=0.95,
                                 epsilon=1e-08,
                                 decay=self.params['learning_rate']/self.params['epochs'])
        self.callbacks = [
            EarlyStoppingByLossVal(monitor='val_loss', value=0.1, verbose=1),
            TensorBoard(
                log_dir=self.params['log_save_path']+'ae_{}'.format(time()),
                histogram_freq=2,
                batch_size=self.params['batch_size'],
                write_graph=True,
                write_grads=True,
                write_images=True
            )
        ]

    def construct(self):
        encoder_input = Input(shape=self.params['input_shape'])
        encoder_input_padded = ZeroPadding2D(padding=(2, 2))(encoder_input) # Evens out odd inputs

        encoder_l1 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1),
                            padding='same', activation='relu',
                            kernel_initializer='glorot_uniform',
                            bias_initializer='zeros')(encoder_input_padded)
        encoder_l1 = MaxPooling2D(pool_size=(2, 2), strides=None,
                                  padding='same')(encoder_l1)
        encoder_l2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                            padding='same', activation='relu',
                            kernel_initializer='glorot_uniform',
                            bias_initializer='zeros')(encoder_l1)
        encoder_l2 = MaxPooling2D(pool_size=(2, 2), strides=None,
                                  padding='same')(encoder_l2)
        encoder_flat_l = Flatten()(encoder_l2) # Only used for clustering later on
        encoder = Model(inputs=encoder_input, outputs=encoder_l2, name='encoder')
        encoder_flat = Model(inputs=encoder_input, outputs=encoder_flat_l, name='encoder')

        decoder_l1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
                            padding='same', activation='relu',
                            kernel_initializer='glorot_uniform',
                            bias_initializer='zeros')(encoder_l2)
        decoder_l1 = UpSampling2D(size=(2, 2))(decoder_l1)
        decoder_l2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1),
                            padding='same', activation='relu',
                            kernel_initializer='glorot_uniform',
                            bias_initializer='zeros')(decoder_l1)
        decoder_l2 = UpSampling2D(size=(2, 2))(decoder_l2)
        decoder_l3 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1),
                            padding='same', activation='sigmoid',
                            kernel_initializer='glorot_uniform',
                            bias_initializer='zeros')(decoder_l2)
        decoder_l3 = Cropping2D(((4, 3), (4, 3)), data_format=None)(decoder_l3) # Returns to original size

        autoencoder = Model(encoder_input, decoder_l3, name='autoencoder')
        autoencoder.compile(optimizer=self.adadelta, loss='binary_crossentropy')
        # Update self with encoder and full autoencoder models
        self.encoder = encoder
        self.encoder_flat = encoder_flat
        self.model = autoencoder

    def fit_model(self, x_train, y_train, x_valid, y_valid):
        self.model.fit(x=x_train, y=y_train,
                       batch_size=self.params['batch_size'],
                       epochs=self.params['epochs'],
                       verbose=2,
                       validation_data=(x_valid, y_valid),
                       callbacks = self.callbacks)
