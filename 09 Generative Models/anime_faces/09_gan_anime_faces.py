import os
import numpy as np
from tensorflow import set_random_seed
from sklearn.preprocessing import MinMaxScaler
from keras import Input
from keras.layers import (Dense, LeakyReLU, Reshape,
                          Conv2D, Conv2DTranspose,
                          AveragePooling2D, BatchNormalization,
                          Flatten, Dropout)
from keras.models import Model
from keras.optimizers import RMSprop, Adam
from skimage.io import imread, imsave
from skimage.transform import resize, rescale
from sklearn.metrics import (roc_auc_score, log_loss,
                             confusion_matrix)
import matplotlib.pyplot as plt
from time import time
from keras.callbacks import TensorBoard
import keras.backend as K


config_params = {
        'data_dir' : 'dsilt-ml-code/09 Generative Models/anime_faces_data/',
        'save_dir' :  'dsilt-ml-code/09 Generative Models/anime_faces_generated/',
        'random_seed': 14,
        'latent_dim' :  50,
        'image_pixel_height' :  64,
        'image_pixel_width' :  64,
        'image_color_channels' :  3,
        'prob_of_flipped_label': 0.1,
        'nbr_images_limit': 200,
        'batch_size' :  20,  #20
        'nbr_epochs' :  10,  #60
        'epochs_per_network_b4_switching': 2,
        'discriminator_learning_rate': 0.0002, #0.0002
        'discriminator_momentum': 0.5,
        'generator_learning_rate': 0.004, #0.0004
        'generator_momentum': 0.5
}


np.random.seed(config_params['random_seed'])
set_random_seed(config_params['random_seed'])


callbacks = [
    TensorBoard(
        log_dir='dsilt-ml-code/09 Generative Models/logs/gan_{}'.format(time()),
        histogram_freq=2,
        batch_size=config_params['batch_size'],
        write_graph=True,
        write_grads=True,
        write_images=True
        )
    ]


# Build the generator
# Generator will take random input
gen_input = Input(shape=(config_params['latent_dim'],))
gen_l1 = Dense(1 * 32 * 32)(gen_input) # orig 128 * 32 * 32
gen_l1 = BatchNormalization()(gen_l1)
gen_l1 = LeakyReLU()(gen_l1)
gen_l1 = Reshape((32, 32, 1))(gen_l1)
gen_l2 = Conv2D(filters=256, kernel_size=(5, 5),
                strides=(1, 1),
                padding='same',
                activation=None,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros')(gen_l1)
gen_l2 = BatchNormalization()(gen_l2)
gen_l2 = LeakyReLU()(gen_l2)
gen_l3 = Conv2DTranspose(filters=256, kernel_size=(4, 4),
                         strides=(2, 2), padding='same',
                         activation=None,
                         kernel_initializer='glorot_uniform',
                         bias_initializer='zeros')(gen_l2)
gen_l3 = BatchNormalization()(gen_l3)
gen_l3 = LeakyReLU()(gen_l3)
gen_l4 = Conv2D(filters=config_params['image_color_channels'],
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                activation='tanh',
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros')(gen_l3)
generator = Model(inputs=gen_input,
                  outputs=gen_l4,
                  name='generator')
print(generator.summary())
# Do not compile generator with an optimizer or loss


# Build the discriminator
# Discriminator will take real and fake image input
dis_input = Input(shape=(config_params['image_pixel_height'],
                         config_params['image_pixel_width'],
                         config_params['image_color_channels']))
dis_l1 = Conv2D(filters=128, kernel_size=(5, 5),
                strides=(1, 1),
                padding='valid',
                activation=None,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros')(dis_input)
dis_l1 = BatchNormalization()(dis_l1)
dis_l1 = LeakyReLU()(dis_l1)
dis_l1 = AveragePooling2D(pool_size=(2, 2),
                          strides=None,
                          padding='valid')(dis_l1)
dis_l1 = Flatten()(dis_l1)
dis_l1 = Dropout(0.4)(dis_l1)
dis_l2 = Dense(1, activation='sigmoid',
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros')(dis_l1)
discriminator = Model(inputs=dis_input,
                      outputs=dis_l2,
                      name='discriminator')
print(discriminator.summary())
discriminator_optimizer = Adam(lr=config_params['discriminator_learning_rate'],
                               beta_1=config_params['discriminator_momentum'],
                               epsilon=None,
                               decay=0.0)
discriminator.compile(optimizer=discriminator_optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])


# Bring everything together to create the GAN
gan_input = Input(shape=(config_params['latent_dim'],))
gan_output = discriminator(generator(gan_input))
gan = Model(inputs=gan_input,
            outputs=gan_output,
            name='gan')
gan_optimizer = Adam(lr=config_params['generator_learning_rate'],
                     beta_1=config_params['generator_momentum'],
                     epsilon=None,
                     decay=0.0)
gan.compile(optimizer=gan_optimizer,
            loss='binary_crossentropy')
print(gan.summary())


# Prepare the real data
file_list = os.listdir(config_params['data_dir'])[:config_params['nbr_images_limit']]
print("Number of total samples:", len(file_list))
real_images = np.array([resize(imread(os.path.join(config_params['data_dir'], file_name)), (64, 64)) for file_name in file_list])
real_images /= np.max(real_images) # input images are already scaled, but never hurts to be sure
plt.imshow(real_images[0])
plt.show()


def smooth_labels(y, min_label=0, max_label=1):
        """
        Converts data labels from hard to soft by
        adding random noise to labels while keeping them
        in the range of min_label-max_label

        Arguments:
        y: numpy array of labels
        min_label: the minimum label value, usually 0
        max_label: the maximum label value, usually 1

        Returns:
        numpy array of labels with random noise
        """
        noise = np.random.uniform(low=0.1, high=0.2, size=y.shape[0]).reshape(y.shape[0], 1)
        if np.min(y) == 0:
                return np.round(np.add(y, noise))
        else:
                return np.round(np.subtract(y, noise))


def prepare_real_data(real_input_data, nbr_samples, prob_of_flipped_label, apply_label_smoothing=False):
        """
        Samples training data to create a mini-batch with labels

        Arguments:
        real_input_data: a numpy array of training data
        nbr_samples: number of training samples in 1 batch
        prob_of_flipped_label: percentage of labels to set incorrectly
                               (this helps the generator learn better)
        apply_label_smoothing: if True, add random noise to label
                               (label smoothing should always be done after flipping)

        Returns:
        tuple: (numpy array of training data, numpy array of labels)
        """
        sample_indices = np.random.randint(real_input_data.shape[0], size=nbr_samples)
        real_samples = real_input_data[sample_indices,:]
        real_labels = np.ones((nbr_samples, 1))
        flipped_labels = np.array([1 if np.random.uniform() <= prob_of_flipped_label else 0 for x in range(nbr_samples)]).reshape(nbr_samples, 1)
        real_labels = np.subtract(real_labels, flipped_labels)
        if smooth_labels:
                real_labels = smooth_labels(real_labels)
        return real_samples, real_labels


def scale_generated_fakes(x):
        """
        Scales input to the range (0, 1)

        Arguments:
        x: a numpy array

        Returns:
        numpy array scaled to (0, 1)
        """
        if np.min(x) < 0:
                return (x+np.min(x)*-1)/(np.max(x)+np.min(x)*-1)
        else:
                return (x+np.min(x))/(np.max(x)+np.min(x))


def generate_fake_data(gen_model, nbr_samples, latent_dim, prob_of_flipped_label, apply_label_smoothing=False, view_sample=False):
        """
        Generates fake images by creating a random vector and passing
        it through the generator.  Fake images are assigned labels
        of 0.

        Arguments:
        gen_model: the generator network
        nbr_samples: number of training samples in 1 batch
        latent_dim: length of the 1D random input vector
        prob_of_flipped_label: percentage of labels to set incorrectly
                                           (this helps the generator learn better)
        apply_label_smoothing: if True, add random noise to label
                               (label smoothing should always be done after flipping)
        view_sample: if True, plot a fake image sample as a sense check

        Returns:
        tuple: (numpy array of fake images, numpy array of labels)
        """
        random_vectors = np.random.uniform(size=(nbr_samples, latent_dim))
        fake_samples = gen_model.predict(random_vectors)
        fake_samples = scale_generated_fakes(fake_samples)
        fake_labels = np.zeros((nbr_samples, 1))
        flipped_labels = np.array([1 if np.random.uniform() <= prob_of_flipped_label else 0 for x in range(nbr_samples)]).reshape(nbr_samples, 1)
        fake_labels = np.add(fake_labels, flipped_labels)
        if smooth_labels:
                fake_labels = smooth_labels(fake_labels)
        if view_sample:
                plt.imshow(fake_samples[0])
                plt.show()
        return fake_samples, fake_labels


'''
# Overfit the discriminator

x_real, y_real = prepare_real_data(real_input_data=real_images,
                                   nbr_samples=int(config_params['batch_size']/2),
                                   prob_of_flipped_label=config_params['prob_of_flipped_label'],
                                   apply_label_smoothing=True)
x_fake, y_fake = generate_fake_data(gen_model=generator,
                                    nbr_samples=int(config_params['batch_size']/2),
                                    latent_dim=config_params['latent_dim'],
                                    prob_of_flipped_label=config_params['prob_of_flipped_label'],
                                    apply_label_smoothing=True)
x_train = np.concatenate([x_fake, x_real])
y_train = np.concatenate([y_fake, y_real])

x_real, y_real = prepare_real_data(real_input_data=real_images,
                                   nbr_samples=int(config_params['batch_size']/2),
                                   prob_of_flipped_label=0)
x_fake, y_fake = generate_fake_data(gen_model=generator,
                                    nbr_samples=int(config_params['batch_size']/2),
                                    latent_dim=config_params['latent_dim'],
                                    prob_of_flipped_label=0)
x_valid = np.concatenate([x_fake, x_real])
y_valid = np.concatenate([y_fake, y_real])

dis_train_hist = discriminator.fit(x_train,
                                   y_train,
                                   validation_data=(x_valid, y_valid),
                                   batch_size=None,
                                   epochs=config_params['nbr_epochs'],
                                   verbose=2,
                                   shuffle=True,
                                   callbacks=callbacks)


def evaluate_discriminator(model, batch_size, latent_dim):
    half_batch = int(batch_size/2) # assumes batch_size is even number
    x_real, y_real = prepare_real_data(real_input_data=real_images,
                                       nbr_samples=half_batch,
                                       prob_of_flipped_label=0)
    x_fake, y_fake = generate_fake_data(gen_model=generator,
                                        nbr_samples=half_batch,
                                        latent_dim=latent_dim,
                                        prob_of_flipped_label=0)
    x_test = np.concatenate([x_fake, x_real])
    y_test = np.concatenate([y_fake, y_real])
    test_scores = discriminator.evaluate(x_test,
                                         y_test,
                                         verbose=0)
    test_preds_prob = model.predict(x_test)
    test_preds = [1 if x > 0.5 else 0 for x in test_preds_prob.flatten().tolist()]
    print("Discriminator classification error on 1 test batch: {}"\
          .format(round(100-test_scores[1]*100, 4)))
    print("Discriminator log loss:",
          log_loss(y_test, test_preds_prob))
    print("Discriminator confusion matrix\n",
          confusion_matrix(y_test, test_preds))

evaluate_discriminator(discriminator, config_params['batch_size'], config_params['latent_dim'])
'''


def train_gan(gan, generator, discriminator, raw_data,
              batch_size, epochs, epochs_per_network,
              latent_dim, view_generated_image=False):

        # Initialize
        dis_train_hist = []
        gan_train_hist = []
        epoch_chunks = int(epochs/epochs_per_network)  # Train discriminator for 5 epochs, then switch
        nbr_batches = int(np.ceil(raw_data.shape[0]/batch_size))
        
        for epoch_chunk in range(epoch_chunks):
                if epoch_chunk == 0:#% 2 == 0:
                        for epoch in range(epochs_per_network):
                                for batch in range(nbr_batches):
                                        x_real, y_real = prepare_real_data(real_input_data=raw_data,
                                                                           nbr_samples=batch_size,
                                                                           prob_of_flipped_label=config_params['prob_of_flipped_label'],
                                                                           apply_label_smoothing=True)
                                        discriminator.train_on_batch(x_real, y_real)
                                        x_fake, y_fake = generate_fake_data(gen_model=generator,
                                                                            nbr_samples=batch_size,
                                                                            latent_dim=latent_dim,
                                                                            prob_of_flipped_label=config_params['prob_of_flipped_label'],
                                                                            apply_label_smoothing=True)
                                        dis_train_hist.append(discriminator.train_on_batch(x_fake, y_fake))
                                # Evaluate discriminator after every training epoch
                                print("Epoch discriminator training log loss and accuracy:\n", dis_train_hist[-nbr_batches:])
                                """
                                x_validation = np.concatenate([x_fake, x_real])
                                y_validation = np.concatenate([y_fake, y_real])
                                validation_scores = discriminator.evaluate(x_validation, y_validation, verbose=0)
                                validation_preds_prob = discriminator.predict(x_validation)
                                validation_preds = [1 if x > 0.5 else 0 for x in validation_preds_prob.flatten().tolist()]
                                print("Discriminator classification error: {}"\
                                          .format(round(100-validation_scores[1]*100, 4)))
                                print("Discriminator log loss:",
                                          log_loss(y_validation, validation_preds_prob))
                                print("Discriminator confusion matrix\n",
                                          confusion_matrix(y_validation, validation_preds))
                                """
                else:
                        for epoch in range(epochs_per_network):
                                for batch in range(nbr_batches):
                                        x_fake = random_vectors = np.random.uniform(size=(batch_size, latent_dim))
                                        y_fake = np.ones((batch_size, 1)) # These labels are 1, not 0
                                        # train generator on batch of fake data
                                        gan_train_hist.append(gan.train_on_batch(x_fake, y_fake))
                                # Evaluate entire GAN after every training epoch
                                print("Epoch generator training accuracy:\n", gan_train_hist[-nbr_batches:])
                        # Print a generated image after training the generator
                        x_fake, y_fake = generate_fake_data(gen_model=generator,
                                                            nbr_samples=1,
                                                            latent_dim=latent_dim,
                                                            prob_of_flipped_label=config_params['prob_of_flipped_label'])
                        imsave(config_params['save_dir']+'generated_anime_face_{}_epochs.png'.format((epoch+1)*(epoch_chunk+1)), x_fake[0])
                        if view_generated_image:
                                plt.imshow(x_fake[0])
                                plt.show()


train_gan(gan, generator, discriminator,
          real_images,
          config_params['batch_size'],
          config_params['nbr_epochs'], 
          config_params['epochs_per_network_b4_switching'], 
          config_params['latent_dim'])

