"""Gradio demo: pneumonia probability + Grad-CAM heatmap for a chest X-ray.

Run:  pip install -r requirements.txt && python app.py
Research and learning project only. Not a medical device.
"""
import json
from pathlib import Path

import gradio as gr
import keras
import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
from PIL import Image

HERE = Path(__file__).parent
MODEL_PATH = HERE / "outputs" / "pneumonia_densenet121.keras"
METRICS_PATH = HERE / "outputs" / "metrics.json"
IMG = 224

model = keras.models.load_model(MODEL_PATH)
THRESHOLD = json.load(open(METRICS_PATH))["threshold"] if METRICS_PATH.exists() else 0.5
backbone = model.get_layer("densenet121")   # outputs the last conv feature map
dense = model.layers[-1]                    # Dense(1, sigmoid)


def predict(image):
    """Return class probabilities, the decision, and a Grad-CAM overlay for the predicted class."""
    if image is None:
        return None, None
    gray = np.asarray(Image.fromarray(image).convert("L").resize((IMG, IMG), Image.BILINEAR), dtype=np.float32)
    x = keras.applications.densenet.preprocess_input(np.repeat(gray[None, ..., None], 3, -1))
    conv = tf.convert_to_tensor(backbone(x, training=False))
    with tf.GradientTape() as tape:
        tape.watch(conv)
        logit = tf.matmul(tf.reduce_mean(conv, axis=(1, 2)), dense.kernel) + dense.bias
        p = float(tf.sigmoid(logit).numpy().ravel()[0])
        target = logit if p >= THRESHOLD else -logit      # explain the class that was predicted
    g = tape.gradient(target, conv)
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * tf.reduce_mean(g[0], axis=(0, 1)), -1)).numpy()
    cam = np.asarray(Image.fromarray(cam / (cam.max() + 1e-8)).resize((IMG, IMG), Image.BILINEAR))
    overlay = 0.6 * np.stack([gray] * 3, -1) / 255.0 + 0.4 * cm.jet(cam)[..., :3]
    decision = "PNEUMONIA" if p >= THRESHOLD else "NORMAL"
    summary = f"{decision}  (p(pneumonia) = {p:.2f}, threshold {THRESHOLD:.2f})"
    return summary, (overlay * 255).astype(np.uint8)


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Chest X-ray (frontal view)"),
    outputs=[gr.Textbox(label="Decision"),
             gr.Image(label="Grad-CAM: regions that drove the decision")],
    title="Paediatric pneumonia detection from chest X-rays (DenseNet121)",
    description=("Trained on the Kermany et al. paediatric dataset (ages 1-5, one hospital in Guangzhou). "
                 "Test ROC-AUC 0.970, sensitivity 0.995, specificity 0.692. "
                 "Research demo only, not a medical device."),
)

if __name__ == "__main__":
    demo.launch()
