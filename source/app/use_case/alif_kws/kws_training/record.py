"""
Record your own keyword clips, by default with the AppKit-E8 microphone.

Board recording needs alif_kws built with -Dalif_kws_RECORD_MODE=ON and flashed.
On each take this script sends 'R' over the board's console UART; the board records
1.5 s from the same microphone path as KWS inference (before auto gain) and sends
it back. The loudest 1 s window is saved as float WAV in data/my_voice/<label>/.

Examples:
    python record.py --port /dev/ttyACM0                # 30 takes of every keyword
    python record.py --port /dev/ttyACM0 --words yes no --takes 10
    python record.py --port /dev/ttyACM0 --words _silence_   # say nothing
    python record.py --port /dev/ttyACM0 --words _unknown_   # say other words
    python record.py --source pc                        # PC microphone via arecord
    python record.py --list-devices
"""
import argparse
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np
import serial
from serial.tools import list_ports
from scipy.io import wavfile

from kws_common import CLIP_SAMPLES, KEYWORDS, LABELS, MY_VOICE_DIR, SAMPLE_RATE, load_wav

RECORD_SECONDS = 1.5
MIN_SNR_DB = 10.0
CLIP_PEAK = 32000.0
BOARD_TIMEOUT_S = 15.0
BOARD_RETRIES = 3


class BoardRecorder:
    """
    Records through alif_kws in record mode over the console UART.
    """

    def __init__(self, port: str, baud: int):
        """
        :param port:    Serial port of the board console UART.
        :param baud:    Baud rate.
        """
        self.serial = serial.Serial(port, baud, timeout=1.0)

    def record(self) -> np.ndarray:
        """
        Record one take. A transfer that is short or fails the CRC is resent by the
        board from its buffer, so the take does not have to be repeated.

        :returns:   float32 samples in int16 range.
        """
        command = b"R"
        for attempt in range(BOARD_RETRIES + 1):
            self.serial.reset_input_buffer()
            self.serial.write(command)
            try:
                return self._receive()
            except (RuntimeError, TimeoutError) as err:
                if attempt == BOARD_RETRIES:
                    raise
                print(f"  transfer error ({err}), resending")
                time.sleep(0.2)
                command = b"S"
        raise AssertionError("unreachable")

    def _receive(self) -> np.ndarray:
        """
        Receive one clip sent by the board.

        :returns:   float32 samples in int16 range.
        """
        deadline = time.time() + BOARD_TIMEOUT_S
        while True:
            if time.time() > deadline:
                raise TimeoutError("no KWSREC header from the board; is record mode flashed?")
            line = self.serial.readline().strip()
            if line.startswith(b"KWSREC"):
                break
        _, rate, samples, crc = line.split()
        if int(rate) != SAMPLE_RATE:
            raise RuntimeError(f"board records at {int(rate)} Hz, expected {SAMPLE_RATE}")
        size = int(samples) * 2
        data = bytearray()
        deadline = time.time() + BOARD_TIMEOUT_S
        while len(data) < size and time.time() < deadline:
            chunk = self.serial.read(size - len(data))
            if not chunk:
                break
            data += chunk
        data = bytes(data)
        if len(data) != size:
            raise RuntimeError(f"got {len(data)} of {size} bytes")
        if zlib.crc32(data) != int(crc, 16):
            raise RuntimeError("CRC mismatch, UART data corrupted")
        self.serial.read_until(b"KWSEND\n")
        return np.frombuffer(data, "<f2").astype(np.float32) * 32768.0


class PcRecorder:
    """
    Records with the PC microphone via arecord.
    """

    def __init__(self, device: str | None):
        """
        :param device:  ALSA device name or None for the default.
        """
        self.device = device

    def record(self) -> np.ndarray:
        """
        Record one take.

        :returns:   float32 samples in int16 range.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "take.wav"
            cmd = ["arecord", "-q", "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1",
                   "-d", "2", str(path)]
            if self.device:
                cmd[1:1] = ["-D", self.device]
            subprocess.run(cmd, check=True)
            rate, data = wavfile.read(path)
            if rate != SAMPLE_RATE:
                data = load_wav(path)
        return data[:int(RECORD_SECONDS * SAMPLE_RATE)].astype(np.float32)


def loudest_window(audio: np.ndarray) -> np.ndarray:
    """
    Pick the 1 s window with the highest energy.

    :param audio:   Samples, ideally longer than CLIP_SAMPLES.
    :returns:       Samples of length CLIP_SAMPLES.
    """
    if len(audio) <= CLIP_SAMPLES:
        return np.pad(audio, (0, CLIP_SAMPLES - len(audio)))
    energy = np.cumsum(np.concatenate([[0.0], audio.astype(np.float64) ** 2]))
    sums = energy[CLIP_SAMPLES:] - energy[:-CLIP_SAMPLES]
    start = int(np.argmax(sums[::160]) * 160)
    return audio[start:start + CLIP_SAMPLES]


def snr_db(audio: np.ndarray) -> float:
    """
    Rough speech-to-background ratio: loudest vs quietest 100 ms block.

    :param audio:   Samples.
    :returns:       Ratio in dB.
    """
    blocks = audio[:len(audio) // 1600 * 1600].reshape(-1, 1600).astype(np.float64)
    rms = np.sqrt(np.mean(blocks ** 2, axis=1)) + 1e-9
    return float(20 * np.log10(rms.max() / rms.min()))


def next_index(folder: Path, prefix: str) -> int:
    """
    Find the next free take number in a folder.

    :param folder:  Label folder.
    :param prefix:  File name prefix.
    :returns:       Next index.
    """
    existing = [int(p.stem.rsplit("_", 1)[-1]) for p in folder.glob(f"{prefix}_*.wav")
                if p.stem.rsplit("_", 1)[-1].isdigit()]
    return max(existing, default=-1) + 1


def record_label(recorder: BoardRecorder | PcRecorder, prefix: str, label: str, takes: int):
    """
    Interactively record a number of takes for one label.

    :param recorder:    Board or PC recorder.
    :param prefix:      File name prefix, "board" or "pc".
    :param label:       Label name.
    :param takes:       Number of takes to save.
    """
    folder = MY_VOICE_DIR / label
    folder.mkdir(parents=True, exist_ok=True)
    index = next_index(folder, prefix)
    if label == "_silence_":
        prompt = "stay quiet (room background)"
    elif label == "_unknown_":
        prompt = "say any word that is NOT a keyword"
    else:
        prompt = f'say "{label.upper()}"'
    print(f"\n=== {label}: {takes} takes, {prompt}. Ctrl+C to stop. ===")
    saved = 0
    while saved < takes:
        print(f"[{saved + 1}/{takes}] get ready...", end="", flush=True)
        time.sleep(0.6)
        print(" NOW", flush=True)
        clip = loudest_window(recorder.record())
        peak = float(np.max(np.abs(clip)))
        snr = snr_db(clip)
        if label != "_silence_" and snr < MIN_SNR_DB:
            print(f"  too quiet ({snr:.0f} dB above background), again")
            continue
        if peak >= CLIP_PEAK:
            print("  clipped, move back a bit, again")
            continue
        path = folder / f"{prefix}_{index:04d}.wav"
        wavfile.write(path, SAMPLE_RATE, (clip / 32768.0).astype(np.float32))
        print(f"  saved {path.name} (peak {20 * np.log10(peak / 32768.0 + 1e-9):.0f} dBFS, "
              f"{snr:.0f} dB above background)")
        index += 1
        saved += 1


def main():
    """
    Parse arguments and record.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["board", "pc"], default="board")
    parser.add_argument("--port", default="/dev/ttyACM0", help="board console UART")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--device", help="ALSA device for --source pc, e.g. plughw:1,0")
    parser.add_argument("--words", nargs="+", default=KEYWORDS, choices=LABELS)
    parser.add_argument("--takes", type=int, default=30)
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        print("Serial ports (--port):")
        for port in list_ports.comports():
            print(f"  {port.device}: {port.description}")
        subprocess.run(["arecord", "-l"], check=False)
        return
    if args.source == "board":
        recorder = BoardRecorder(args.port, args.baud)
    else:
        recorder = PcRecorder(args.device)
    try:
        for word in args.words:
            record_label(recorder, args.source, word, args.takes)
    except KeyboardInterrupt:
        print("\nstopped")
        sys.exit(0)
    for label in LABELS:
        count = len(list((MY_VOICE_DIR / label).glob("*.wav")))
        if count:
            print(f"{label:>10}: {count} clips")


if __name__ == "__main__":
    main()
