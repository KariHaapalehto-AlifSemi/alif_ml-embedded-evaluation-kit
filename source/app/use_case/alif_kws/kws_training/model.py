"""
MicroNet-M (Arm ML-Zoo kws_micronet_m) rebuilt in Keras.

The layer structure was read from the int8 .tflite. Batch norm is already folded
into the convolutions there, so the float model is plain conv + bias + ReLU6 and
the original weights can be loaded by dequantizing them.
"""
from pathlib import Path

import numpy as np
import tensorflow as tf

from kws_common import LABELS, NUM_FRAMES, NUM_MFCC

# (depthwise stride, pointwise output channels) for the five separable blocks.
_BLOCKS = [(2, 140), (1, 140), (1, 140), (1, 112), (1, 196)]


def build_model(dropout: float = 0.0) -> tf.keras.Model:
    """
    Build the float MicroNet-M model that outputs logits.

    :param dropout:     Dropout rate before the classifier, 0 for none.
    :returns:           Keras model with input (49, 10, 1) and output (12,).
    """
    inputs = tf.keras.Input(shape=(NUM_FRAMES, NUM_MFCC, 1), name="input")
    x = tf.keras.layers.Conv2D(140, (10, 4), padding="same", activation=tf.nn.relu6,
                               name="conv0")(inputs)
    for i, (stride, channels) in enumerate(_BLOCKS):
        x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=stride, padding="same",
                                            activation=tf.nn.relu6, name=f"dw{i}")(x)
        x = tf.keras.layers.Conv2D(channels, 1, activation=tf.nn.relu6, name=f"pw{i}")(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(25, 5), name="pool")(x)
    if dropout:
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv2D(len(LABELS), 1, name="classifier")(x)
    outputs = tf.keras.layers.Reshape((len(LABELS),), name="logits")(x)
    return tf.keras.Model(inputs, outputs, name="micronet_m")


def _dequantize(interpreter: tf.lite.Interpreter, detail: dict) -> np.ndarray:
    """
    Read a constant tensor and convert it to float.

    :param interpreter:     TFLite interpreter holding the model.
    :param detail:          Tensor detail dict.
    :returns:               Float32 array.
    """
    data = interpreter.get_tensor(detail["index"]).astype(np.float32)
    quant = detail["quantization_parameters"]
    scales = quant["scales"]
    if scales.size == 0:
        return data
    zero_points = quant["zero_points"].astype(np.float32)
    shape = [1] * data.ndim
    if scales.size > 1:
        shape[quant["quantized_dimension"]] = -1
    return (data - zero_points.reshape(shape)) * scales.reshape(shape)


def load_tflite_weights(model: tf.keras.Model, tflite_path: Path):
    """
    Copy dequantized weights from the int8 MicroNet-M .tflite into the Keras model.

    :param model:           Model from build_model().
    :param tflite_path:     Path to kws_micronet_m.tflite.
    """
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    details = {d["index"]: d for d in interpreter.get_tensor_details()}
    ops = [op for op in interpreter._get_ops_details()  # pylint: disable=protected-access
           if op["op_name"] in ("CONV_2D", "DEPTHWISE_CONV_2D")]
    layers = [layer for layer in model.layers
              if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D))]
    if len(ops) != len(layers):
        raise RuntimeError(f"{len(ops)} conv ops in tflite but {len(layers)} conv layers")

    for op, layer in zip(ops, layers):
        kernel = _dequantize(interpreter, details[op["inputs"][1]])
        bias = _dequantize(interpreter, details[op["inputs"][2]])
        if op["op_name"] == "CONV_2D":
            kernel = kernel.transpose(1, 2, 3, 0)          # OHWI -> HWIO
        else:
            kernel = kernel[0][:, :, :, None]              # 1HWC -> HWC1
        layer.set_weights([kernel, bias])
