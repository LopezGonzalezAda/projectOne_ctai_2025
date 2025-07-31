import cv2
from inference_sdk import InferenceHTTPClient

CONFIDENCE_THRESHOLD = 0.4
MODEL_ID = "pressed_key_check/4"
WEBCAM_INDEX = 0

# Load Roboflow model
model = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="NW0OR7CAuzrGbEzhXpzF"
)

# Open webcam
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

print("🎥 Starting live detection. Press 'q' to quit.")
frame_count = 0
last_predictions = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    resized = cv2.resize(frame, (416, 416))
    input_data = resized

    frame_count += 1

    # Run inference every 3rd frame
    if frame_count % 3 == 0:
        try:
            result = model.infer(input_data, model_id=MODEL_ID)
            last_predictions = result.get("predictions", [])
        except Exception as e:
            print("❌ Inference failed:", e)
            last_predictions = []

    # Draw predictions (even on skipped frames)
    for pred in last_predictions:
        try:
            conf = pred["confidence"]
            if conf < CONFIDENCE_THRESHOLD:
                continue

            x = int(pred["x"] - pred["width"] / 2)
            y = int(pred["y"] - pred["height"] / 2)
            w = int(pred["width"])
            h = int(pred["height"])
            label = pred["class"]

            cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(resized, f"{label} ({conf:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        except Exception as e:
            print("⚠️ Drawing error:", e)
            print("Prediction row:", pred)

    cv2.imshow("Live Piano Detection", resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting live detection.")
        break

cap.release()
cv2.destroyAllWindows()
