import torchaudio


def generate_fbank(audio, sample_rate, n_mels=128):
    assert sample_rate == 16000, 'input audio sampling rate must be 16kHz'
    fbank = torchaudio.compliance.kaldi.fbank(audio, htk_compat=True, sample_frequency=sample_rate, use_energy=False, window_type='hanning', num_mel_bins=n_mels, dither=0.0, frame_shift=10)
    mean, std = -4.2677393, 4.5689974
    fbank = (fbank - mean) / (std * 2)
    fbank = fbank.unsqueeze(-1).numpy()
    return fbank
