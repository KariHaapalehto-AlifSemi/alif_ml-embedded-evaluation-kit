"""
Fine-tune MicroNet-M on your voice, starting from the original ML-Zoo weights.

Each epoch draws a fresh, augmented mix of your clips and Speech Commands so the
model learns your voice without forgetting other speakers. The checkpoint with the
best combined validation score is kept in output/micronet_m_personal.weights.h5.

Example:
    python train.py --epochs 20
"""
import argparse
import time

import numpy as np
import tensorflow as tf

from data import Augmenter, background_noise, build_set, my_voice, speech_commands
from kws_common import LABELS, ORIGINAL_TFLITE, OUTPUT_DIR
from model import build_model, load_tflite_weights

WEIGHTS = OUTPUT_DIR / "micronet_m_personal.weights.h5"
EVAL_CACHE = OUTPUT_DIR / "eval_sets.npz"


def make_eval_sets(aug: Augmenter) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Build the fixed, non-augmented evaluation sets and cache them.

    :param aug:     Augmenter with a fixed seed.
    :returns:       Name -> (features, labels).
    """
    sc_kw, sc_unk = speech_commands("test")
    sc_x, sc_y, _ = build_set(aug, [], sc_kw, sc_unk, sc_per_word=200, my_repeat=0,
                              num_silence=200, augment=False)
    sets = {"speech_commands_test": (sc_x, sc_y)}
    mine = my_voice("val")
    if mine:
        my_x, my_y, _ = build_set(aug, mine, {}, [], sc_per_word=0, my_repeat=1,
                                  num_silence=0, augment=False)
        sets["my_voice_val"] = (my_x, my_y)
    np.savez_compressed(EVAL_CACHE, **{f"{k}_x": v[0] for k, v in sets.items()},
                        **{f"{k}_y": v[1] for k, v in sets.items()})
    return sets


def report(model: tf.keras.Model, sets: dict[str, tuple[np.ndarray, np.ndarray]],
           title: str, per_label: bool = False) -> dict[str, float]:
    """
    Print accuracy on each evaluation set.

    :param model:       Model to evaluate.
    :param sets:        Evaluation sets.
    :param title:       Heading to print.
    :param per_label:   Also print per-label accuracy for your voice.
    :returns:           Set name -> accuracy.
    """
    accs = {}
    print(f"--- {title}")
    for name, (x, y) in sets.items():
        pred = np.argmax(model.predict(x, batch_size=256, verbose=0), axis=1)
        accs[name] = float(np.mean(pred == y))
        print(f"  {name:>22}: {accs[name] * 100:5.1f} % ({len(y)} clips)")
        if per_label and name == "my_voice_val":
            for idx, label in enumerate(LABELS):
                mask = y == idx
                if mask.any():
                    print(f"      {label:>10}: {np.mean(pred[mask] == idx) * 100:5.1f} % "
                          f"({mask.sum()})")
    return accs


def main():
    """
    Run the fine-tuning.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--sc-per-word", type=int, default=400,
                        help="Speech Commands clips per keyword in each epoch")
    parser.add_argument("--my-repeat", type=int, default=8,
                        help="augmented copies of each of your clips per epoch")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    tf.keras.utils.set_random_seed(args.seed)
    mine_train = my_voice("train")
    if not mine_train:
        raise SystemExit("No recordings in data/my_voice. Run record.py first.")
    print(f"Your clips: {len(mine_train)} train, {len(my_voice('val'))} val")

    noises = background_noise()
    sets = make_eval_sets(Augmenter(noises, seed=1234))
    sc_kw, sc_unk = speech_commands("train")
    missing = [LABELS[idx] for idx, paths in sc_kw.items()
               if not paths and not any(ex.label == idx for ex in mine_train)]
    if missing:
        raise SystemExit(f"No training clips for {', '.join(missing)}: record them with "
                         "record.py first.")

    model = build_model(dropout=0.1)
    load_tflite_weights(model, ORIGINAL_TFLITE)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    base = report(model, sets, "original model (float)", per_label=True)
    best = sum(base.values()) / len(base)
    model.save_weights(WEIGHTS)

    aug = Augmenter(noises, seed=args.seed)
    schedule = tf.keras.optimizers.schedules.CosineDecay(args.lr, args.epochs)
    for epoch in range(args.epochs):
        model.optimizer.learning_rate.assign(float(schedule(epoch)))
        start = time.time()
        x, y, is_mine = build_set(aug, mine_train, sc_kw, sc_unk, args.sc_per_word,
                                  args.my_repeat, num_silence=args.sc_per_word, augment=True)
        hist = model.fit(x, y, batch_size=args.batch, epochs=1, verbose=0)
        print(f"\nepoch {epoch + 1}/{args.epochs}: {len(y)} examples ({is_mine.sum()} yours), "
              f"loss {hist.history['loss'][0]:.3f}, {time.time() - start:.0f} s")
        accs = report(model, sets, "validation")
        score = sum(accs.values()) / len(accs)
        if score > best:
            best = score
            model.save_weights(WEIGHTS)
            print(f"  saved best ({score * 100:.1f} %)")

    model.load_weights(WEIGHTS)
    report(model, sets, "best fine-tuned model (float)", per_label=True)
    print(f"\nWeights: {WEIGHTS}\nNext: python convert.py")


if __name__ == "__main__":
    main()
