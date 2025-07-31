import gradio as gr
import cv2
from inference_sdk import InferenceHTTPClient 

# Load model
model = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="NW0OR7CAuzrGbEzhXpzF"
)
MODEL_ID = "pressed_key_check/4"
CONFIDENCE_THRESHOLD = 0.4

# Inference function
def detect_from_webcam(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = model.infer(rgb, model_id=MODEL_ID)
    predictions = result.get("predictions", [])

    for pred in predictions:
        if pred["confidence"] < CONFIDENCE_THRESHOLD:
            continue
        x = int(pred["x"] - pred["width"] / 2)
        y = int(pred["y"] - pred["height"] / 2)
        w = int(pred["width"])
        h = int(pred["height"])
        label = pred["class"]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame

# Build the interface (no deprecated params)
demo = gr.Interface(
    fn=detect_from_webcam,
    inputs=gr.Image(label="Live Webcam", tool=None, type="numpy"),
    outputs=gr.Image(label="Detected Frame"),
    live=True
)

# Launch
demo.launch()
