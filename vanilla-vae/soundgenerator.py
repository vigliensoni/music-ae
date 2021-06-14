import librosa

from preprocess import MinMaxNormalizer


class SoundGenerator:
    """SoundGenerator is responsible for generating audio from spectrograms."""

    def __init__(self, vae, hop_length):
        self.vae = vae
        self.hop_length = hop_length
        self._min_max_normalizer = MinMaxNormalizer(0, 1)

    def generate(self, spectrograms, min_max_values):
        generated_spectrograms, latent_representations = \
            self.vae.reconstruct(spectrograms)
        signals = self.convert_spectrograms_to_audio(generated_spectrograms, min_max_values)
        return signals, latent_representations
    
    def convert_spectrograms_to_audio(self, spectrograms, min_max_values):
        signals = []
        for spectrogram, min_max_values in zip(spectrograms, min_max_values):
            # reshape the log spectrogram
            log_spectrogram = spectrogram[:, :, 0] # 0 drops the last dimentions
            # applying denormalization
            denorm_log_spec = self._min_max_normalizer.denormalize(log_spectrogram, min_max_values["min"], min_max_values["max"])
            # log spectrogram -> spectrogram
            spec = librosa.db_to_amplitude(denorm_log_spec)
            # apply Griffim-Lim algorithm
            signal = librosa.istft(spec, hop_length = self.hop_length)
            # append signal to "signals"
            signals.append(signal)
        return signals