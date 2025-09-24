
from rsc_toolkit.augmentations import Augmentation
from rsc_toolkit.processors.base import BaseAudioProcessor
from rsc_toolkit.processors.util import generate_fbank


class MelBankAudioProcessor(BaseAudioProcessor):

    n_mels: int
    sample_rate: int
    
    def make_features(self, audio):
        fbank = generate_fbank(audio, sample_rate=self.sample_rate, n_mels=self.n_mels)
        return fbank
