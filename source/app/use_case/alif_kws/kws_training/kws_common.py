"""
Shared constants and audio front end for fine-tuning the alif_kws MicroNet-M model.

The MFCC implementation is a NumPy port of the on-device code in
alif_ml-embedded-evaluation-kit/source/lib/mlek/common/Mfcc.cc with the
MicroNetKwsMFCC parameters, so features seen in training match the board.
"""
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MY_VOICE_DIR = DATA_DIR / "my_voice"
SPEECH_COMMANDS_DIR = DATA_DIR / "speech_commands"
OUTPUT_DIR = ROOT / "output"

MLEK_ROOT = Path.home() / "alif" / "alif_ml-embedded-evaluation-kit"
ORIGINAL_TFLITE = MLEK_ROOT / "resources_downloaded" / "kws" / "kws_micronet_m.tflite"
VELA_INI = MLEK_ROOT / "scripts" / "vela" / "ensemble_vela.ini"

# Output order of the model: resources/kws/labels/micronet_kws_labels.txt with "left" and
# "right" replaced by the Finnish "vasen" and "oikea", which are trained from your
# recordings only. convert.py writes this list as the firmware labels file.
LABELS = ["down", "go", "vasen", "no", "off", "on", "oikea", "stop", "up", "yes",
          "_silence_", "_unknown_"]
KEYWORDS = LABELS[:10]
SILENCE_IDX = LABELS.index("_silence_")
UNKNOWN_IDX = LABELS.index("_unknown_")

SAMPLE_RATE = 16000
CLIP_SAMPLES = 16000
FRAME_LEN = 640
FRAME_STRIDE = 320
NUM_FRAMES = 49
NUM_MFCC = 10
NUM_FBANK = 40
MEL_LO_HZ = 20.0
MEL_HI_HZ = 4000.0
FFT_LEN = 1024
FLT_MIN = np.float32(1.17549435e-38)

# Device auto gain control, see audio_alif.c.
AGC_STRIDE = 8000
AGC_MAX_GAIN = 10000.0
AGC_MAX_GAIN_INC = 1.12201845


def _mel(freq: np.ndarray) -> np.ndarray:
    """
    HTK mel scale.

    :param freq:    Frequency in Hz.
    :returns:       Mel value.
    """
    return np.float32(1127.0) * np.log1p(np.asarray(freq, np.float32) / np.float32(700.0))


def _build_mel_matrix() -> np.ndarray:
    """
    Build the mel filter bank exactly as MFCC::CreateMelFilterBank does.

    :returns:   Matrix of shape (FFT_LEN // 2, NUM_FBANK).
    """
    num_bins = FFT_LEN // 2
    bin_width = np.float32(SAMPLE_RATE) / np.float32(FFT_LEN)
    mel_lo = _mel(MEL_LO_HZ)
    mel_hi = _mel(MEL_HI_HZ)
    delta = (mel_hi - mel_lo) / np.float32(NUM_FBANK + 1)
    mels = _mel(bin_width * np.arange(num_bins, dtype=np.float32))
    matrix = np.zeros((num_bins, NUM_FBANK), np.float32)
    for b in range(NUM_FBANK):
        left = mel_lo + b * delta
        center = mel_lo + (b + 1) * delta
        right = mel_lo + (b + 2) * delta
        inside = (mels > left) & (mels < right)
        rising = (mels - left) / (center - left)
        falling = (right - mels) / (right - center)
        matrix[:, b] = np.where(inside, np.where(mels <= center, rising, falling), 0.0)
    return matrix


def _build_dct_matrix() -> np.ndarray:
    """
    Build the DCT matrix exactly as MFCC::CreateDCTMatrix does.

    :returns:   Matrix of shape (NUM_FBANK, NUM_MFCC).
    """
    n = np.arange(NUM_FBANK, dtype=np.float32)
    k = np.arange(NUM_MFCC, dtype=np.float32)
    angle = np.outer(n + 0.5, k) * np.float32(np.pi / NUM_FBANK)
    return (np.sqrt(np.float32(2.0 / NUM_FBANK)) * np.cos(angle)).astype(np.float32)


_WINDOW = (0.5 - 0.5 * np.cos(np.arange(FRAME_LEN) * (2 * np.pi / FRAME_LEN))).astype(np.float32)
_MEL_MATRIX = _build_mel_matrix()
_DCT_MATRIX = _build_dct_matrix()
_FRAME_IDX = np.arange(NUM_FRAMES)[:, None] * FRAME_STRIDE + np.arange(FRAME_LEN)[None, :]


def mfcc(audio: np.ndarray) -> np.ndarray:
    """
    Compute the model input features for one 1 s clip.

    :param audio:   Samples in int16 range (as on the board), length CLIP_SAMPLES.
    :returns:       float32 MFCC array of shape (NUM_FRAMES, NUM_MFCC).
    """
    frames = audio.astype(np.float32)[_FRAME_IDX] * np.float32(1.0 / 32768.0) * _WINDOW
    magnitude = np.abs(np.fft.rfft(frames, n=FFT_LEN, axis=1))[:, :FFT_LEN // 2]
    energies = np.log(magnitude.astype(np.float32) @ _MEL_MATRIX + FLT_MIN)
    return (energies @ _DCT_MATRIX).astype(np.float32)


def load_wav(path: Path) -> np.ndarray:
    """
    Load a WAV file as 16 kHz mono, padded or trimmed to CLIP_SAMPLES.

    Samples stay float so quiet raw board recordings (float WAV) keep their
    resolution; the device auto gain emulation brings them to full scale later.

    :param path:    WAV file path.
    :returns:       float32 samples in int16 range.
    """
    rate, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        data = data.astype(np.float32)
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 65536.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) * 256.0
    else:
        data = data.astype(np.float32) * 32768.0
    if rate != SAMPLE_RATE:
        data = resample_poly(data, SAMPLE_RATE, rate)
    return fit_length(np.clip(data, -32768, 32767).astype(np.float32))


def fit_length(audio: np.ndarray, length: int = CLIP_SAMPLES) -> np.ndarray:
    """
    Zero pad or centre-trim audio to a fixed length.

    :param audio:   Samples.
    :param length:  Wanted length.
    :returns:       Samples of the given length.
    """
    if len(audio) >= length:
        start = (len(audio) - length) // 2
        return audio[start:start + length]
    pad = length - len(audio)
    return np.pad(audio, (pad // 2, pad - pad // 2))


def device_agc(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Emulate the board's auto gain control on a 1 s clip.

    Gain is set per 0.5 s stride: it drops at once to keep the peak at full scale
    but rises by at most 1 dB per stride, so the first stride starts from a random
    gain history.

    :param audio:   float32 samples scaled to int16 range.
    :param rng:     Random generator.
    :returns:       int16 samples.
    """
    out = np.empty(len(audio), np.float32)
    gain = None
    for start in range(0, len(audio), AGC_STRIDE):
        chunk = audio[start:start + AGC_STRIDE]
        peak = max(float(np.max(np.abs(chunk))) / 32768.0, 1e-6)
        target = min(1.0 / peak, AGC_MAX_GAIN)
        if gain is None:
            gain = target * float(rng.uniform(0.25, 1.0))
        else:
            gain = min(target, gain * AGC_MAX_GAIN_INC)
        out[start:start + AGC_STRIDE] = chunk * gain
    return np.clip(out, -32768, 32767).astype(np.int16)
