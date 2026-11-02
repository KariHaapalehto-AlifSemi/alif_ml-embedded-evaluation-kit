"""
Dataset handling: your recordings in data/my_voice plus Google Speech Commands v2.

Every example goes through the same chain as on the board: 1 s of 16 kHz audio,
device auto gain control emulation, then the on-device MFCC.
"""
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from kws_common import (CLIP_SAMPLES, KEYWORDS, LABELS, MY_VOICE_DIR, SILENCE_IDX,
                        SPEECH_COMMANDS_DIR, UNKNOWN_IDX, device_agc, fit_length, load_wav,
                        mfcc)

MY_VOICE_VAL_EVERY = 5


@dataclass
class Example:
    """
    One labelled source clip.
    """
    path: Path
    label: int
    mine: bool


def _is_val(path: Path) -> bool:
    """
    Deterministically assign one in MY_VOICE_VAL_EVERY of your clips to validation.

    :param path:    Clip path.
    :returns:       True for validation clips.
    """
    return zlib.crc32(path.name.encode()) % MY_VOICE_VAL_EVERY == 0


def my_voice(split: str) -> list[Example]:
    """
    List your own recordings. Folders of words that are not labels (e.g. replaced
    keywords) count as _unknown_.

    :param split:   "train" or "val".
    :returns:       Examples.
    """
    out = []
    for folder in sorted(p for p in MY_VOICE_DIR.glob("*") if p.is_dir()):
        idx = LABELS.index(folder.name) if folder.name in LABELS else UNKNOWN_IDX
        for path in sorted(folder.glob("*.wav")):
            if _is_val(path) == (split == "val"):
                out.append(Example(path, idx, True))
    return out


def _speech_commands_lists() -> tuple[set[str], set[str]]:
    """
    Read the official validation and test file lists.

    :returns:   (validation set, test set) of relative paths.
    """
    val = set((SPEECH_COMMANDS_DIR / "validation_list.txt").read_text().split())
    test = set((SPEECH_COMMANDS_DIR / "testing_list.txt").read_text().split())
    return val, test


def speech_commands(split: str) -> tuple[dict[int, list[Path]], list[Path]]:
    """
    List Speech Commands clips for one split.

    :param split:   "train", "val" or "test".
    :returns:       (keyword index -> clips, unknown-word clips).
    """
    val, test = _speech_commands_lists()
    keywords = {i: [] for i in range(len(KEYWORDS))}
    unknown = []
    for folder in sorted(SPEECH_COMMANDS_DIR.iterdir()):
        if not folder.is_dir() or folder.name.startswith("_"):
            continue
        for path in sorted(folder.glob("*.wav")):
            rel = f"{folder.name}/{path.name}"
            in_split = {"val": rel in val, "test": rel in test}.get(
                split, rel not in val and rel not in test)
            if not in_split:
                continue
            if folder.name in KEYWORDS:
                keywords[KEYWORDS.index(folder.name)].append(path)
            else:
                unknown.append(path)
    return keywords, unknown


def background_noise() -> list[np.ndarray]:
    """
    Load the Speech Commands background noise recordings.

    :returns:   List of float32 sample arrays.
    """
    noises = []
    for path in sorted((SPEECH_COMMANDS_DIR / "_background_noise_").glob("*.wav")):
        _, data = wavfile.read(path)
        noises.append(data.astype(np.float32))
    return noises


class Augmenter:
    """
    Turns source clips into model input features, optionally with augmentation.
    """

    def __init__(self, noises: list[np.ndarray], seed: int):
        """
        :param noises:  Background noise recordings.
        :param seed:    Random seed.
        """
        self.noises = noises
        self.rng = np.random.default_rng(seed)
        self._cache: dict[Path, np.ndarray] = {}

    def _load(self, path: Path) -> np.ndarray:
        """
        Load a clip as float32, cached.

        :param path:    Clip path.
        :returns:       float32 samples.
        """
        if path not in self._cache:
            self._cache[path] = load_wav(path).astype(np.float32)
        return self._cache[path]

    def _noise(self, length: int) -> np.ndarray:
        """
        Pick a random background noise segment.

        :param length:  Number of samples.
        :returns:       float32 samples.
        """
        noise = self.noises[self.rng.integers(len(self.noises))]
        start = self.rng.integers(0, len(noise) - length)
        return noise[start:start + length]

    def silence(self, augment: bool = True) -> np.ndarray:
        """
        Make one _silence_ example from background noise.

        :param augment:     Use random volume.
        :returns:           MFCC features.
        """
        volume = self.rng.uniform(0.0, 0.3) if augment else 0.1
        audio = self._noise(CLIP_SAMPLES) * volume
        audio += self.rng.normal(0.0, 2.0, CLIP_SAMPLES)
        return mfcc(device_agc(audio.astype(np.float32), self.rng))

    def features(self, path: Path, augment: bool) -> np.ndarray:
        """
        Make one example from a clip.

        :param path:        Clip path.
        :param augment:     Apply shift, speed, noise augmentation.
        :returns:           MFCC features.
        """
        audio = self._load(path)
        if augment:
            speed = self.rng.uniform(0.9, 1.1)
            src = np.arange(0, len(audio), speed)
            audio = fit_length(np.interp(src, np.arange(len(audio)), audio).astype(np.float32))
            shift = int(self.rng.integers(-1600, 1601))
            audio = np.roll(audio, shift)
            if shift > 0:
                audio[:shift] = 0
            elif shift < 0:
                audio[shift:] = 0
            if self.rng.random() < 0.8:
                speech_rms = float(np.sqrt(np.mean(audio ** 2))) + 1e-3
                noise = self._noise(CLIP_SAMPLES)
                noise_rms = float(np.sqrt(np.mean(noise ** 2))) + 1e-3
                snr_db = self.rng.uniform(5.0, 30.0)
                audio = audio + noise * (speech_rms / noise_rms) / (10 ** (snr_db / 20))
        audio = audio + self.rng.normal(0.0, 2.0, len(audio))
        return mfcc(device_agc(audio.astype(np.float32), self.rng))


def build_set(aug: Augmenter, examples: list[Example], sc_keywords: dict[int, list[Path]],
              sc_unknown: list[Path], sc_per_word: int, my_repeat: int, num_silence: int,
              augment: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build one batch of features.

    :param aug:             Augmenter.
    :param examples:        Your clips.
    :param sc_keywords:     Speech Commands keyword clips by label index.
    :param sc_unknown:      Speech Commands non-keyword clips.
    :param sc_per_word:     Speech Commands clips sampled per keyword.
    :param my_repeat:       Times each of your clips is used.
    :param num_silence:     Number of generated _silence_ examples.
    :param augment:         Apply augmentation.
    :returns:               (features (N, 49, 10, 1), labels (N,), is-mine flags (N,)).
    """
    rng = aug.rng
    items: list[tuple[Path | None, int, bool]] = []
    for label, paths in sc_keywords.items():
        take = min(sc_per_word, len(paths))
        for i in rng.choice(len(paths), take, replace=False):
            items.append((paths[i], label, False))
    take = min(int(sc_per_word * 1.5), len(sc_unknown))
    for i in rng.choice(len(sc_unknown), take, replace=False):
        items.append((sc_unknown[i], UNKNOWN_IDX, False))
    items += [(None, SILENCE_IDX, False)] * num_silence
    for ex in examples:
        items += [(ex.path, ex.label, True)] * my_repeat

    order = rng.permutation(len(items))
    feats = np.empty((len(items), 49, 10, 1), np.float32)
    labels = np.empty(len(items), np.int32)
    mine = np.empty(len(items), bool)
    for out_i, item_i in enumerate(order):
        path, label, is_mine = items[item_i]
        if path is None:
            feats[out_i, :, :, 0] = aug.silence(augment)
        else:
            feats[out_i, :, :, 0] = aug.features(path, augment)
        labels[out_i] = label
        mine[out_i] = is_mine
    return feats, labels, mine
