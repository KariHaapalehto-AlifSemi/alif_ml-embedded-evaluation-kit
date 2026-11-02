"""
Quantize the fine-tuned model to int8, compare it with the original and compile it
with Vela for the Ethos-U55-128 on the RTSS-HE core.

Outputs in output/:
    kws_micronet_m_personal.tflite              int8 model (CPU / reference)
    kws_micronet_m_personal_vela_H128.tflite    Vela model for alif_kws on RTSS-HE
    kws_micronet_m_personal_labels.txt          labels file for alif_kws

Example:
    python convert.py
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from data import Augmenter, background_noise, build_set, my_voice, speech_commands
from kws_common import LABELS, NUM_FRAMES, NUM_MFCC, ORIGINAL_TFLITE, OUTPUT_DIR, VELA_INI
from model import build_model
from train import EVAL_CACHE, WEIGHTS

INT8_TFLITE = OUTPUT_DIR / "kws_micronet_m_personal.tflite"
VELA_TFLITE = OUTPUT_DIR / "kws_micronet_m_personal_vela_H128.tflite"
LABELS_TXT = OUTPUT_DIR / "kws_micronet_m_personal_labels.txt"


def quantize(model: tf.keras.Model, num_samples: int) -> bytes:
    """
    Post-training int8 quantization with representative data from both datasets.

    :param model:           Float model.
    :param num_samples:     Number of representative examples.
    :returns:               Serialized int8 .tflite.
    """
    aug = Augmenter(background_noise(), seed=7)
    sc_kw, sc_unk = speech_commands("train")
    rep_x, _, _ = build_set(aug, my_voice("train"), sc_kw, sc_unk, sc_per_word=30,
                            my_repeat=1, num_silence=30, augment=True)
    rep_x = rep_x[:num_samples]

    def representative():
        for sample in rep_x:
            yield [sample[None]]

    saved_model = OUTPUT_DIR / "saved_model"
    shutil.rmtree(saved_model, ignore_errors=True)
    export = tf.keras.export.ExportArchive()
    export.track(model)
    export.add_endpoint("serve", lambda x: model(x, training=False),
                        input_signature=[tf.TensorSpec([1, NUM_FRAMES, NUM_MFCC, 1], tf.float32)])
    export.write_out(str(saved_model), verbose=False)
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def tflite_accuracy(path: Path, x: np.ndarray, y: np.ndarray) -> float:
    """
    Accuracy of an int8 .tflite on float features, quantizing them like the board does.

    :param path:    .tflite path.
    :param x:       Float features (N, 49, 10, 1).
    :param y:       Labels.
    :returns:       Accuracy.
    """
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    scale, zero_point = inp["quantization"]
    q = np.clip(np.round(x / scale + zero_point), -128, 127).astype(np.int8)
    correct = 0
    for sample, label in zip(q, y):
        interpreter.set_tensor(inp["index"], sample[None])
        interpreter.invoke()
        correct += int(np.argmax(interpreter.get_tensor(out["index"])[0]) == label)
    return correct / len(y)


def run_vela(model_path: Path):
    """
    Compile with the same Vela settings the eval kit uses for H128 on RTSS-HE.

    :param model_path:  int8 .tflite path.
    """
    vela = Path(sys.executable).parent / "vela"
    work = OUTPUT_DIR / "vela"
    shutil.rmtree(work, ignore_errors=True)
    subprocess.run([str(vela), str(model_path),
                    "--accelerator-config=ethos-u55-128",
                    "--optimise", "Performance",
                    "--config", str(VELA_INI),
                    "--memory-mode=Shared_Sram",
                    "--system-config=RTSS_HE_SRAM_MRAM",
                    f"--output-dir={work}"], check=True)
    shutil.copy(work / f"{model_path.stem}_vela.tflite", VELA_TFLITE)


def main():
    """
    Quantize, evaluate and compile.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rep-samples", type=int, default=500)
    args = parser.parse_args()

    model = build_model()
    model.load_weights(WEIGHTS)
    INT8_TFLITE.write_bytes(quantize(model, args.rep_samples))
    print(f"int8 model: {INT8_TFLITE}")

    cache = np.load(EVAL_CACHE)
    names = sorted({k[:-2] for k in cache.files})
    print("\nint8 accuracy          original   personal")
    for name in names:
        x, y = cache[f"{name}_x"], cache[f"{name}_y"]
        before = tflite_accuracy(ORIGINAL_TFLITE, x, y)
        after = tflite_accuracy(INT8_TFLITE, x, y)
        print(f"  {name:>20}:  {before * 100:6.1f} %   {after * 100:6.1f} %")

    run_vela(INT8_TFLITE)
    LABELS_TXT.write_text("\n".join(LABELS) + "\n")
    print(f"\nVela model: {VELA_TFLITE}\nLabels:     {LABELS_TXT}")
    print("Build alif_kws with them by adding to your cmake command:\n"
          f"  -Dalif_kws_MODEL_PATH={VELA_TFLITE}\n"
          f"  -Dalif_kws_LABELS_TXT_FILE={LABELS_TXT}")


if __name__ == "__main__":
    main()
