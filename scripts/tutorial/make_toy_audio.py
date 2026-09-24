"""Generate the synthetic audio of the bundled toy dataset (sample_data/toy_dataset/files/*.wav).

The toy dataset keeps the ICBHI file layout (file names, cycle annotation .txt files, split file) so the whole pipeline
can be exercised without downloading data, but its audio is synthetic: band-limited noise shaped by the annotated
breathing cycles, with short clicks added to cycles annotated as crackle and a gliding tone added to cycles annotated
as wheeze. The signal is deterministic (seeded by file name) and has no clinical meaning.

Usage: python scripts/tutorial/make_toy_audio.py [--files sample_data/toy_dataset/files]
"""
import argparse
import hashlib
import pathlib

import numpy as np
import soundfile as sf

SR = 44100
DURATION_S = 20.0


def _rng(name: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))


def _band_noise(rng: np.random.Generator, n: int, lo: float, hi: float) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    spec[(freqs < lo) | (freqs > hi)] = 0.0
    x = np.fft.irfft(spec, n)
    return x / (np.abs(x).max() + 1e-12)


def synth(annotation: pathlib.Path) -> np.ndarray:
    n = int(SR * DURATION_S)
    rng = _rng(annotation.stem)
    t = np.arange(n) / SR
    x = 0.05 * _band_noise(rng, n, 100.0, 1500.0)
    rows = [line.split() for line in annotation.read_text().splitlines() if line.strip()]
    for start, end, crackle, wheeze in ((float(a), float(b), int(c), int(d)) for a, b, c, d in rows):
        i0, i1 = int(start * SR), min(int(end * SR), n)
        if i1 <= i0:
            continue
        seg = np.arange(i1 - i0) / SR
        env = np.sin(np.pi * seg / max(end - start, 1e-3)) ** 2
        x[i0:i1] += 0.25 * env * _band_noise(rng, i1 - i0, 150.0, 1000.0)
        if crackle:
            for c in rng.integers(i0, i1, size=12):
                k = np.arange(min(220, n - c))
                x[c:c + k.size] += 0.4 * np.exp(-k / 30.0) * np.sin(2 * np.pi * 900.0 * k / SR)
        if wheeze:
            f = 400.0 + 200.0 * seg / max(end - start, 1e-3)
            x[i0:i1] += 0.15 * env * np.sin(2 * np.pi * np.cumsum(f) / SR)
    x += 0.01 * np.sin(2 * np.pi * 50.0 * t)
    return (0.8 * x / (np.abs(x).max() + 1e-12)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--files", default="sample_data/toy_dataset/files")
    args = ap.parse_args()
    files = pathlib.Path(args.files)
    for ann in sorted(files.glob("*.txt")):
        wav = ann.with_suffix(".wav")
        subtype = sf.info(str(wav)).subtype if wav.exists() else "PCM_24"
        sf.write(str(wav), synth(ann), SR, subtype=subtype)
        print(f"wrote {wav} ({subtype})")


if __name__ == "__main__":
    main()
