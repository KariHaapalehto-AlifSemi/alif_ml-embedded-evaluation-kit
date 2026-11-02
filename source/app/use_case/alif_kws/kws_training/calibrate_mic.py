"""
Calibrate the AppKit-E8 PDM microphones for keyword spotting and write the best
settings into mic_listener.c.

Needs alif_kws built with -Dalif_kws_RECORD_MODE=ON and flashed: the recorder firmware
also takes the calibration commands. The board plays a 1 kHz tone on its headphone
jack (WM8904 codec). Put the earpiece right on top of the microphone port and keep the
room quiet while this runs (about 3 minutes).

Steps:
  1. Tone level: sets the tone so that the microphone sees about -20 dBFS.
  2. Phase: sweeps the PDM sampling phase 0..63 and picks the middle of the range
     where both channels give their best signal to noise and distortion (SINAD).
  3. Decimation: compares the PDM clocks 512 kHz and 768 kHz (T5838 low-power mode)
     by tone to noise floor ratio. 1.024 MHz is measured but not chosen, as it is
     outside every T5838 mode.
  4. Gain: with the tone off, finds the lowest gain whose noise floor is within 1 dB
     of the best one (lower gains are limited by the 16-bit output). The gain is
     capped so that your loudest recorded take keeps 6 dB of headroom.
  5. Mix: compares left, right and the L+R average used by audio_alif.c.

Examples:
    python calibrate_mic.py                  # calibrate and update mic_listener.c
    python calibrate_mic.py --dry-run        # report only
"""
import argparse
import json
import re
import time
import zlib
from pathlib import Path

import numpy as np
import serial
from scipy.signal.windows import blackmanharris

from kws_common import MLEK_ROOT, MY_VOICE_DIR, OUTPUT_DIR, SAMPLE_RATE, load_wav

MIC_LISTENER = (MLEK_ROOT / "source" / "hal" / "source" / "components" / "audio_alif" /
                "source" / "alif" / "mic_listener.c")
REPORT = OUTPUT_DIR / "mic_calibration.json"

TONE_HZ = 1000
BAND_HZ = (100.0, 7000.0)
TONE_FRAMES = 2048
NOISE_FRAMES = 4096
FULL_SCALE_POWER = 32768.0 ** 2 / 2
CLIP_LEVEL = 32000

TARGET_TONE_DBFS = -20.0
LEVEL_TOLERANCE_DB = 2.0
MIN_TONE_SINAD_DB = 10.0
MAX_AMPLITUDE = 16384
MIN_AMPLITUDE = 16
HP_0DB = 57
HP_MAX = 63
START_LEVEL_DB = 20 * np.log10(1024) - 12

MAX_PHASE = 63
PHASE_WINDOW = 2
PHASE_TIE_DB = 0.5
COARSE_STEP = 4
DECIMATIONS = (32, 48, 64)
IN_SPEC_DECIMATIONS = (32, 48)
MODE_PREFER_DB = 1.0

GAINS = (64, 96, 128, 160, 192, 256, 320, 384, 448, 512, 640, 768, 1024, 1536, 2048, 3072,
         4095)
NOISE_KNEE_DB = 1.0
HEADROOM_DB = 6.0
MIX_NOISE_DB = 3.0
MIN_MIX_SNR_DB = 20.0

BOARD_TIMEOUT_S = 10.0
BOARD_RETRIES = 3


class Board:
    """
    Calibration commands of alif_kws record mode over the console UART.
    """

    def __init__(self, port: str, baud: int):
        """
        :param port:    Serial port of the board console UART.
        :param baud:    Baud rate.
        """
        self.serial = serial.Serial(port, baud, timeout=1.0)

    def command(self, text: str):
        """
        Send a setting command and wait for the board to accept it.

        :param text:    Command without the leading 'C', e.g. "T 1000 4096".
        :raises RuntimeError:   The board rejected the command.
        :raises TimeoutError:   The board did not reply.
        """
        self.serial.reset_input_buffer()
        self.serial.write(f"C{text}\n".encode())
        deadline = time.time() + BOARD_TIMEOUT_S
        while time.time() < deadline:
            line = self.serial.readline().strip()
            if line == b"OK":
                return
            if line == b"ERR":
                raise RuntimeError(f"board rejected 'C{text}'")
        raise TimeoutError(f"no reply to 'C{text}'; is the record mode firmware flashed?")

    def capture(self, frames: int) -> np.ndarray:
        """
        Capture raw stereo microphone samples. A transfer that is short or fails the CRC
        is resent by the board from its buffer.

        :param frames:  Number of stereo frames.
        :returns:       Array of shape (frames, 2) in int16 units.
        """
        command = f"CW {frames}\n".encode()
        for attempt in range(BOARD_RETRIES + 1):
            self.serial.reset_input_buffer()
            self.serial.write(command)
            try:
                return self._receive(frames)
            except (RuntimeError, TimeoutError) as err:
                if attempt == BOARD_RETRIES:
                    raise
                print(f"  transfer error ({err}), resending")
                time.sleep(0.2)
                command = b"S"
        raise AssertionError("unreachable")

    def _receive(self, frames: int) -> np.ndarray:
        """
        Receive one raw stereo block.

        :param frames:  Expected number of stereo frames.
        :returns:       Array of shape (frames, 2) in int16 units.
        :raises ValueError: The board could not capture.
        """
        deadline = time.time() + BOARD_TIMEOUT_S
        while True:
            if time.time() > deadline:
                raise TimeoutError("no KWSRAW header from the board")
            line = self.serial.readline().strip()
            if line == b"ERR":
                raise ValueError("board could not capture")
            if line.startswith(b"KWSRAW"):
                break
        _, rate, count, channels, crc = line.split()
        if int(rate) != SAMPLE_RATE or int(count) != frames or int(channels) != 2:
            raise ValueError(f"unexpected block header {line!r}")
        size = frames * 4
        data = bytearray()
        while len(data) < size and time.time() < deadline:
            chunk = self.serial.read(size - len(data))
            if not chunk:
                break
            data += chunk
        if len(data) != size:
            raise RuntimeError(f"got {len(data)} of {size} bytes")
        if zlib.crc32(bytes(data)) != int(crc, 16):
            raise RuntimeError("CRC mismatch, UART data corrupted")
        self.serial.read_until(b"KWSEND\n")
        return np.frombuffer(bytes(data), "<i2").reshape(-1, 2).astype(np.float64)


def band_power(audio: np.ndarray, tone_hz: float | None) -> tuple[float, float]:
    """
    Split the 100..7000 Hz power of one channel into tone and the rest.

    :param audio:   Samples of one channel.
    :param tone_hz: Tone frequency, or None for no tone.
    :returns:       Tone power and remaining (noise and distortion) power, as mean squares.
    """
    audio = audio - np.mean(audio)
    window = blackmanharris(len(audio), sym=False)
    power = np.abs(np.fft.rfft(audio * window)) ** 2 * (2.0 / (len(audio) * np.sum(window ** 2)))
    freq = np.fft.rfftfreq(len(audio), 1.0 / SAMPLE_RATE)
    band = (freq >= BAND_HZ[0]) & (freq <= BAND_HZ[1])
    if tone_hz is None:
        return 0.0, float(power[band].sum())
    tone = np.abs(freq - tone_hz) <= 4.0 * SAMPLE_RATE / len(audio)
    return float(power[tone].sum()), float(power[band & ~tone].sum())


def dbfs(power: float) -> float:
    """
    :param power:   Mean square in int16 units.
    :returns:       Level relative to a full scale sine.
    """
    return float(10 * np.log10(max(power, 1e-12) / FULL_SCALE_POWER))


def channels(audio: np.ndarray) -> list[np.ndarray]:
    """
    :param audio:   Stereo capture of shape (frames, 2).
    :returns:       Left, right and the L+R average that audio_alif.c uses.
    """
    return [audio[:, 0], audio[:, 1], (audio[:, 0] + audio[:, 1]) / 2]


def measure_tone(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    :param audio:   Stereo capture with the tone playing.
    :returns:       Tone level (dBFS) and SINAD (dB) for left, right and mix, and whether
                    the capture clipped.
    """
    levels, sinads = [], []
    for chan in channels(audio):
        tone, rest = band_power(chan, TONE_HZ)
        levels.append(dbfs(tone))
        sinads.append(10 * np.log10(max(tone, 1e-12) / max(rest, 1e-12)))
    return np.array(levels), np.array(sinads), bool(np.max(np.abs(audio)) >= CLIP_LEVEL)


def measure_noise(audio: np.ndarray) -> np.ndarray:
    """
    :param audio:   Stereo capture in silence.
    :returns:       100..7000 Hz noise level (dBFS) for left, right and mix.
    """
    return np.array([dbfs(band_power(chan, None)[1]) for chan in channels(audio)])


def read_defaults(path: Path) -> tuple[int, int, int]:
    """
    Read the current PDM settings from mic_listener.c.

    :param path:    mic_listener.c.
    :returns:       Phase, gain and the 16 kHz decimation.
    """
    text = path.read_text()
    phase = int(re.search(r"#define PDM_CH_PHASE\s+(\S+)", text)[1], 0)
    gain = int(re.search(r"#define PDM_CH_GAIN\s+(\S+)", text)[1], 0)
    decimation = int(re.search(r"case 16000:.*?return ARM_PDM_MODE_AUDIOFREQ_16K_DECM_(\d+);",
                               text, re.S)[1])
    return phase, gain, decimation


def write_settings(path: Path, phase: int, gain: int, decimation: int):
    """
    Write the PDM settings into mic_listener.c.

    :param path:        mic_listener.c.
    :param phase:       Channel phase.
    :param gain:        Channel gain.
    :param decimation:  Decimation at 16 kHz.
    """
    text = path.read_text()
    text = re.sub(r"(#define PDM_CH_PHASE\s+)\S+", rf"\g<1>0x{phase:08X}", text, count=1)
    text = re.sub(r"(#define PDM_CH_GAIN\s+)\S+", rf"\g<1>0x{gain:08X}", text, count=1)
    text = re.sub(r"(case 16000:.*?return ARM_PDM_MODE_AUDIOFREQ_16K_DECM_)\d+",
                  rf"\g<1>{decimation}", text, count=1, flags=re.S)
    path.write_text(text)


def set_tone(board: Board, level_db: float) -> tuple[int, int]:
    """
    Play the tone at a digital level, using the highest amplitude for the best DAC SNR.

    :param board:       Board.
    :param level_db:    Wanted 20*log10(amplitude) + headphone volume in dB.
    :returns:           Amplitude and headphone volume used.
    """
    volume = int(np.clip(HP_0DB + np.ceil(level_db - 20 * np.log10(MAX_AMPLITUDE)), 0, HP_MAX))
    amplitude = int(np.clip(10 ** ((level_db - (volume - HP_0DB)) / 20),
                            MIN_AMPLITUDE, MAX_AMPLITUDE))
    board.command(f"V {volume}")
    board.command(f"T {TONE_HZ} {amplitude}")
    return amplitude, volume


def level_tone(board: Board) -> float:
    """
    Adjust the tone so that the louder channel sees TARGET_TONE_DBFS.

    :param board:   Board, with the default PDM settings.
    :returns:       Digital tone level (see set_tone()).
    :raises RuntimeError:   The microphone does not pick up the tone.
    """
    level_db = START_LEVEL_DB
    max_level = 20 * np.log10(MAX_AMPLITUDE) + HP_MAX - HP_0DB
    for _ in range(10):
        amplitude, volume = set_tone(board, level_db)
        levels, sinads, _ = measure_tone(board.capture(TONE_FRAMES))
        print(f"  amplitude {amplitude:5d}, volume {volume - HP_0DB:+3d} dB: "
              f"{levels[0]:6.1f} / {levels[1]:6.1f} dBFS")
        if max(sinads[:2]) < MIN_TONE_SINAD_DB:
            if level_db >= max_level:
                raise RuntimeError("the microphones do not pick up the tone: check that the "
                                   "earpiece is on the microphone port")
            level_db = min(level_db + 12, max_level)
            continue
        error = TARGET_TONE_DBFS - max(levels[:2])
        if abs(error) <= LEVEL_TOLERANCE_DB:
            return level_db
        level_db = min(level_db + error, max_level)
    return level_db


def sweep_phase(board: Board, phases: list[int], gain: int) -> np.ndarray:
    """
    :param board:   Board, with the tone playing.
    :param phases:  Phases to try.
    :param gain:    Channel gain.
    :returns:       SINAD (dB) of shape (len(phases), 2) for left and right.
    """
    result = []
    for phase in phases:
        board.command(f"P 2 {phase} {gain}")
        _, sinads, _ = measure_tone(board.capture(TONE_FRAMES))
        result.append(sinads[:2])
        print(f"  phase {phase:2d}: SINAD {sinads[0]:5.1f} / {sinads[1]:5.1f} dB")
    return np.array(result).reshape(-1, 2)


def pick_phase(phases: list[int], sinads: np.ndarray, window: int) -> int:
    """
    Pick the phase in the middle of the widest range where both channels are near their
    best SINAD, so that the choice tolerates drift.

    :param phases:  Phases tried, ascending.
    :param sinads:  SINAD of shape (len(phases), 2).
    :param window:  Neighbours on each side that must be good too.
    :returns:       Phase.
    """
    worst = (sinads - sinads.max(axis=0)).min(axis=1)
    score = np.array([worst[max(0, i - window):i + window + 1].min()
                      for i in range(len(phases))])
    good = score >= score.max() - PHASE_TIE_DB
    best_start, best_len, start = 0, 0, None
    for i, flag in enumerate(list(good) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start > best_len:
                best_start, best_len = start, i - start
            start = None
    return phases[best_start + (best_len - 1) // 2]


def measure_mode(board: Board, phase: int, gain: int, level_db: float) -> dict:
    """
    Measure tone SINAD and tone to noise floor ratio at one setting.

    :param board:       Board.
    :param phase:       Channel phase.
    :param gain:        Channel gain.
    :param level_db:    Digital tone level.
    :returns:           Levels, SINAD, noise and SNR for left, right and mix.
    """
    board.command(f"P 2 {phase} {gain}")
    set_tone(board, level_db)
    levels, sinads, clipped = measure_tone(board.capture(TONE_FRAMES))
    board.command("T 0 0")
    noise = measure_noise(board.capture(NOISE_FRAMES))
    return {"level": levels.tolist(), "sinad": sinads.tolist(), "noise": noise.tolist(),
            "snr": (levels - noise).tolist(), "clipped": clipped}


def calibrate_mode(board: Board, decimation: int, gain: int, level_db: float,
                   full_sweep: bool) -> dict:
    """
    Find the best phase at one decimation and measure it.

    :param board:       Board.
    :param decimation:  PDM decimation at 16 kHz.
    :param gain:        Channel gain.
    :param level_db:    Digital tone level.
    :param full_sweep:  Sweep every phase instead of coarse and fine steps.
    :returns:           Phase sweep, chosen phase and measure_mode() results.
    """
    board.command(f"M {decimation}")
    set_tone(board, level_db)
    if full_sweep:
        phases = list(range(MAX_PHASE + 1))
        sinads = sweep_phase(board, phases, gain)
        phase = pick_phase(phases, sinads, PHASE_WINDOW)
    else:
        coarse = list(range(0, MAX_PHASE + 1, COARSE_STEP))
        coarse_sinads = sweep_phase(board, coarse, gain)
        centre = pick_phase(coarse, coarse_sinads, 0)
        fine = [p for p in range(centre - COARSE_STEP + 1, centre + COARSE_STEP)
                if 0 <= p <= MAX_PHASE and p not in coarse]
        phases = sorted(coarse + fine)
        sinads = np.concatenate([coarse_sinads, sweep_phase(board, fine, gain)])
        sinads = sinads[np.argsort(coarse + fine)]
        phase = pick_phase(phases, sinads, 1)
    result = measure_mode(board, phase, gain, level_db)
    result.update({"decimation": decimation, "phase": phase, "phases": phases,
                   "phase_sinad": sinads.tolist()})
    print(f"  decimation {decimation} (PDM clock {SAMPLE_RATE * decimation / 1e3:.0f} kHz): "
          f"phase {phase}, SINAD {result['sinad'][0]:.1f} / {result['sinad'][1]:.1f} dB, "
          f"tone to noise {result['snr'][0]:.1f} / {result['snr'][1]:.1f} dB")
    return result


def loudest_take_dbfs() -> float | None:
    """
    :returns:   Peak level of the loudest recorded board take, or None if there are none.
    """
    peaks = [np.max(np.abs(load_wav(path))) for path in MY_VOICE_DIR.glob("*/board_*.wav")]
    if not peaks:
        return None
    return float(20 * np.log10(max(peaks) / 32768.0 + 1e-12))


def calibrate_gain(board: Board, phase: int, default_gain: int) -> dict:
    """
    Choose the gain from the noise floor in silence and the headroom of recorded takes.

    :param board:           Board, at the chosen decimation.
    :param phase:           Channel phase.
    :param default_gain:    Gain the recorded takes were made with.
    :returns:               Gain sweep and the chosen gain.
    """
    board.command("T 0 0")
    noise = []
    for gain in GAINS:
        board.command(f"P 2 {phase} {gain}")
        level = measure_noise(board.capture(NOISE_FRAMES))[:2]
        noise.append(level)
        print(f"  gain {gain:4d}: noise {level[0]:6.1f} / {level[1]:6.1f} dBFS")
    noise = np.array(noise)
    input_noise = noise - 20 * np.log10(np.array(GAINS))[:, None]
    near_best = input_noise <= input_noise.min(axis=0) + NOISE_KNEE_DB
    knee = max(GAINS[int(np.argmax(near_best[:, ch]))] for ch in range(2))
    # Below the knee a lower gain only makes speech quieter than in the training takes
    gain, limit = max(knee, default_gain), None
    peak = loudest_take_dbfs()
    if peak is not None:
        limit = int(default_gain * 10 ** ((-HEADROOM_DB - peak) / 20))
        if limit < gain:
            print(f"  gain {gain} wanted, but your loudest take ({peak:.1f} dBFS at "
                  f"gain {default_gain}) needs <= {limit} for {HEADROOM_DB:.0f} dB headroom")
            gain = max(limit, GAINS[0])
    return {"gains": list(GAINS), "noise": noise.tolist(), "knee": knee,
            "headroom_limit": limit, "loudest_take_dbfs": peak, "gain": int(min(gain, 0xFFF))}


def choose_mix(result: dict) -> str:
    """
    Choose USE_MIC. With the earpiece on one port the tone favours that microphone, so the
    choice is from the noise floors: the average gains 3 dB for a distant talker when the
    microphones are alike.

    :param result:  measure_mode() results at the final setting.
    :returns:       "LR", "L" or "R".
    """
    noise, snr = result["noise"], result["snr"]
    working = [snr[ch] >= MIN_MIX_SNR_DB for ch in range(2)]
    if all(working) and abs(noise[0] - noise[1]) <= MIX_NOISE_DB:
        return "LR"
    if working[0] and (not working[1] or noise[0] < noise[1]):
        return "L"
    return "R"


def main():
    """
    Parse arguments and calibrate.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="/dev/ttyACM0", help="board console UART")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--mic-listener", type=Path, default=MIC_LISTENER)
    parser.add_argument("--dry-run", action="store_true", help="do not write mic_listener.c")
    args = parser.parse_args()

    default_phase, default_gain, default_decimation = read_defaults(args.mic_listener)
    print(f"mic_listener.c: phase {default_phase}, gain {default_gain}, "
          f"decimation {default_decimation}")
    board = Board(args.port, args.baud)
    report = {"defaults": {"phase": default_phase, "gain": default_gain,
                           "decimation": default_decimation}}
    try:
        board.command(f"M {default_decimation}")
        board.command(f"P 2 {default_phase} {default_gain}")
        print("\n1. Tone level")
        level_db = level_tone(board)

        print("\n2./3. Phase and decimation")
        modes = {default_decimation: calibrate_mode(board, default_decimation, default_gain,
                                                    level_db, True)}
        for decimation in DECIMATIONS:
            if decimation not in modes:
                modes[decimation] = calibrate_mode(board, decimation, default_gain, level_db,
                                                   False)
        report["modes"] = {str(k): v for k, v in modes.items()}

        def mode_snr(decimation: int) -> float:
            mode = modes[decimation]
            return -np.inf if mode["clipped"] else float(np.mean(mode["snr"][:2]))

        decimation = default_decimation
        for candidate in IN_SPEC_DECIMATIONS:
            if mode_snr(candidate) > mode_snr(decimation) + MODE_PREFER_DB:
                decimation = candidate
        if decimation != default_decimation:
            print(f"\n  decimation {decimation} is better, full phase sweep there")
            modes[decimation] = calibrate_mode(board, decimation, default_gain, level_db, True)
            report["modes"][str(decimation)] = modes[decimation]
        phase = modes[decimation]["phase"]
        board.command(f"M {decimation}")

        print("\n4. Gain (tone off)")
        report["gain"] = calibrate_gain(board, phase, default_gain)
        gain = report["gain"]["gain"]

        print("\n5. Final check and microphone mix")
        final = measure_mode(board, phase, gain, level_db)
        report["final"] = final
        mix = choose_mix(final)
    finally:
        try:
            board.command("T 0 0")
        except (RuntimeError, TimeoutError):
            pass

    report["result"] = {"phase": phase, "gain": gain, "decimation": decimation, "use_mic": mix}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1))

    names = ["left", "right", "L+R"]
    for i, name in enumerate(names):
        print(f"  {name:>5}: SINAD {final['sinad'][i]:5.1f} dB, noise {final['noise'][i]:6.1f} "
              f"dBFS, tone to noise {final['snr'][i]:5.1f} dB")
    print(f"\nResult: phase {phase} (was {default_phase}), gain {gain} (was {default_gain}), "
          f"decimation {decimation} (was {default_decimation}), USE_MIC={mix}")
    print(f"Report: {REPORT}")
    if args.dry_run:
        print("--dry-run: mic_listener.c not changed")
    else:
        write_settings(args.mic_listener, phase, gain, decimation)
        print(f"Updated {args.mic_listener}")
    if mix != "LR":
        print(f"Microphone mix: reconfigure the builds with -DUSE_MIC={mix}")
    print("Rebuild and flash the firmware to use the new settings.")


if __name__ == "__main__":
    main()
