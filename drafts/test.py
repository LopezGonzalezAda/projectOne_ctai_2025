from roboflow import Roboflow

# Replace with your actual API key
API_KEY = "NW0OR7CAuzrGbEzhXpzF"
PROJECT_NAME = "pressed_key_check"        # e.g., "pressed_key_check"
MODEL_VERSION = 4                       # e.g., 4

# Initialize Roboflow
rf = Roboflow(api_key=API_KEY)
project = rf.workspace().project(PROJECT_NAME)
model = project.version(MODEL_VERSION).model

# Run prediction on a sample image
prediction = model.predict("drafts/test_image.jpg").json()

# Print result
print("Prediction Result:", prediction)
