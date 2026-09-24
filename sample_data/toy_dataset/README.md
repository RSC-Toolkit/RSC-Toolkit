# Toy Dataset

A minimal dataset for sanity checks, the tutorial and CI.

It keeps the ICBHI 2017 file layout (file names, cycle annotation `.txt` files and a split file) so the full pipeline runs
without downloading data, but the audio is **synthetic**: band-limited noise shaped by the annotated breathing cycles, with
clicks on crackle cycles and a gliding tone on wheeze cycles. It contains no real recordings and scores on it are
meaningless. Regenerate the audio with `python scripts/tutorial/make_toy_audio.py`.
