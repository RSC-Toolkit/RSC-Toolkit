from abc import ABC, abstractmethod
import torch


class Augmentation(ABC):

    @abstractmethod
    def __call__(self, audio: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        pass


class ComposeAug(Augmentation):
    def __init__(self, augs):
        self.augs = augs

    def __call__(self, audio, sampling_rate):
        for aug in self.augs:
            audio = aug(audio, sampling_rate)
        return audio


class RandomNoiseAug(Augmentation):
    def __init__(self, noise_level=0.005):
        self.noise_level = noise_level

    def __call__(self, audio, sampling_rate):
        noise = torch.randn_like(audio) * self.noise_level
        return audio + noise


# class CutPadAug(Augmentation):
#     def __init__(self, desired_length):
#         self.desired_length = desired_length

#     def __call__(self, audio, sampling_rate):
#         return cut_pad_sample_torchaudio(audio, self.desired_length)