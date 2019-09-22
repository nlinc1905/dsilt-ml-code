from scipy.io import wavfile
from scipy.signal import spectrogram
import numpy as np
from pathlib import Path


class WavProcessor():


    def __init__(self, wav_file_dir):
        self.wav_file_dir = wav_file_dir


    def readFile(self, file_path_string):
        return wavfile.read(file_path_string)


    def log_spectrogram(self, samples_per_sec, samples):
        if len(samples.shape) != 1 and samples.shape[1] == 2:
            left_channel = samples[:,0]
            right_channel = samples[:,1]
            frequencies, times, l_spectrogram = spectrogram(left_channel,
                                                            fs=samples_per_sec,
                                                            nperseg=int(256))
            r_spectrogram = spectrogram(right_channel,
                                        fs=samples_per_sec,
                                        nperseg=int(256))[2]
            l_spectrogram = np.log10(l_spectrogram)
            r_spectrogram = np.log10(r_spectrogram)
            return frequencies, times, np.array([l_spectrogram, r_spectrogram])
        else:
            frequencies, times, spec = spectrogram(samples,
                                                   fs=samples_per_sec,
                                                   nperseg=256)
            spec = np.log10(spec)
            return frequencies, times, spec


    def rescale_spectrogram(self, spec):
        smallest_noninf_value = np.min(np.where(spec==-np.inf, np.max(spec), spec))
        new_smallest_value = smallest_noninf_value*0.01
        spec = np.where(spec==-np.inf, new_smallest_value, spec)
        # Scale to (0, 1)
        spec = spec+abs(np.min(spec))
        spec /= np.max(spec)
        return spec


    def create_spectrograms(self):
        audio_file_names = []  # Will be the labels
        spectrograms = []      # Spectrogram arrays
        time_list = []         # Used for plotting
        freq_list = []         # Used for plotting
        file_paths = Path(self.wav_file_dir).glob("*.wav")
        for path in file_paths:
            file = str(path)
            samples_per_sec, samples = self.readFile(file)
            # Slice samples to only create a spectrogram of the left channel: [:,0]
            frequencies, times, spec = self.log_spectrogram(samples_per_sec,
                                                            samples[:,0])
            spec = self.rescale_spectrogram(spec)
            spec = spec.reshape(spec.shape[0], spec.shape[1], 1) # Reshape for Keras
            audio_file_names.append(path.stem)
            spectrograms.append(spec)
            time_list.append(times)
            freq_list.append(frequencies)
        try:
            return audio_file_names, np.array(spectrograms), time_list, freq_list
        except:
            return print("Error: not all spectrograms have the same number of samples.")
