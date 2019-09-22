import numpy as np
from tensorflow.compat.v1 import set_random_seed
import os
from mp3_to_wav import MP3Processor
from wav_to_spectrogram import WavProcessor
from autoencoder_network import Autoencoder
from cluster_latent_features import HClust
import matplotlib.pyplot as plt


config = {
    "mp3_file_dir": "dsilt-ml-code/09 Generative Models/song_similarity/audio_mp3s/",
    "wav_file_dir": "dsilt-ml-code/09 Generative Models/song_similarity/audio_wavs/",
    "metadata_dir": "dsilt-ml-code/09 Generative Models/song_similarity/audio_metadata/",
    "ffmpeg_dir": "C:/Program Files (x86)/FFmpeg/bin/ffmpeg.exe"
    }

autoencoder_config = {
    "random_seed": 14,
    "validation_split_perc": 0.1,
    "learning_rate": 1.0,
    "epochs": 10,
    "batch_size": 6,
    "log_save_path": "dsilt-ml-code/09 Generative Models/song_similarity/logs/"
    }

cluster_config = {
    'linkage_method': 'ward',
    'distance_metric': 'euclidean'
    }


np.random.seed(autoencoder_config['random_seed'])
set_random_seed(autoencoder_config['random_seed'])


def plot_spectrogram(spec, times, freqs, scaling_factor=10):
    if len(spec.shape) > 2:
        spec = spec[0] # Can only plot 1 channel
    plt.pcolormesh(times, freqs, scaling_factor*spec)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Log Scaled Frequency Spectrogram')
    plt.show()

mp3Processor = MP3Processor(config['mp3_file_dir'], config['wav_file_dir'],
                           config['metadata_dir'], config['ffmpeg_dir'])
tags = mp3Processor.convertFiles()

wavProcessor = WavProcessor(config['wav_file_dir'])
labels, specs, times, freqs = wavProcessor.create_spectrograms()
plot_spectrogram(specs[0].reshape(specs[0].shape[0], specs[0].shape[1]), times[0], freqs[0])

# Split training/validation sets, using last val_size observations for validation
val_size = int(round(autoencoder_config['validation_split_perc']*len(specs), 0))
x_train = specs[:specs.shape[0]-val_size, :]
x_valid = specs[specs.shape[0]-val_size:, :]
plot_spectrogram(x_valid[0].reshape(x_valid[0].shape[0], x_valid[0].shape[1]), times[0], freqs[0])

# Build and train model, using x_train as both the input x and target y
autoencoder_config['input_shape'] = specs[0].shape
autoencoder = Autoencoder(autoencoder_config)
autoencoder.construct()
print(autoencoder.model.summary())
autoencoder.fit_model(x_train, x_train,
                      x_valid, x_valid)

# Get model outputs
encoded_valid = autoencoder.encoder.predict(x_valid)
reconstructed_valid = autoencoder.model.predict(x_valid)

# Examine the encoding and reconstruction of 1 image
plt.imshow(encoded_valid[0].reshape(encoded_valid[0].shape[0]*encoded_valid[0].shape[2],
                                    encoded_valid[0].shape[1]).T)
plt.title('Encoded Latent Features of Spectrogram')
plt.show()
plot_spectrogram(reconstructed_valid[0].reshape(reconstructed_valid[0].shape[0],
                                                reconstructed_valid[0].shape[1]),
                 times[0], freqs[0])

# Cluster data to view similar songs based on latent acoustic features
encoded_train_flat = autoencoder.encoder_flat.predict(x_train)
encoded_valid_flat = autoencoder.encoder_flat.predict(x_valid)
cluster_model = HClust(encoded_train_flat,
                       cluster_config['linkage_method'],
                       cluster_config['distance_metric'],
                       labels[:len(labels)-val_size])
cluster_model.cluster()
cluster_model.plot_dendrogram()
