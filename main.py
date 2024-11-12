import cv2
from tkinter import Tk, filedialog
from PIL import Image
import os

# Hide the main Tkinter window
Tk().withdraw()

# Open a file dialog to select an image file
file_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
)

if file_path:
    image = cv2.imread(file_path)
    if image is None:
        print("Failed to load image. Please try again.")
        exit()
    image = cv2.resize(image, (720, 640))
else:
    print("No image file selected.")
    exit()

# Define model file paths
models_dir = "models"
face_pbtxt = os.path.join(models_dir, "opencv_face_detector.pbtxt")
face_pb = os.path.join(models_dir, "opencv_face_detector_uint8.pb")
age_prototxt = os.path.join(models_dir, "age_deploy.prototxt")
age_model = os.path.join(models_dir, "age_net.caffemodel")
gender_prototxt = os.path.join(models_dir, "gender_deploy.prototxt")
gender_model = os.path.join(models_dir, "gender_net.caffemodel")
MODEL_MEAN_VALUES = [104, 117, 123]

# Load models
face_net = cv2.dnn.readNet(face_pb, face_pbtxt)
age_net = cv2.dnn.readNet(age_model, age_prototxt)
gender_net = cv2.dnn.readNet(gender_model, gender_prototxt)

age_classifications = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_classifications = ['Male', 'Female']

# Copy image for processing
img_cp = image.copy()
img_h, img_w = img_cp.shape[:2]

# Prepare blob and detect faces
blob = cv2.dnn.blobFromImage(img_cp, 1.0, (300, 300), MODEL_MEAN_VALUES, True, False)
face_net.setInput(blob)
detected_faces = face_net.forward()

face_bounds = []

for i in range(detected_faces.shape[2]):
    confidence = detected_faces[0, 0, i, 2]
    if confidence > 0.99:
        x1 = int(detected_faces[0, 0, i, 3] * img_w)
        y1 = int(detected_faces[0, 0, i, 4] * img_h)
        x2 = int(detected_faces[0, 0, i, 5] * img_w)
        y2 = int(detected_faces[0, 0, i, 6] * img_h)
        cv2.rectangle(img_cp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        face_bounds.append([x1, y1, x2, y2])

if not face_bounds:
    print("No faces were detected.")
    exit()

# Display age and gender text in a fixed, clear area
try:
    for face_bound in face_bounds:
        face_img = img_cp[max(0, face_bound[1]-15): min(face_bound[3]+15, img_cp.shape[0]-1),
                          max(0, face_bound[0]-15): min(face_bound[2]+15, img_cp.shape[1]-1)]
        
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, True)

        # Predict gender
        gender_net.setInput(blob)
        gender_prediction = gender_net.forward()
        gender = gender_classifications[gender_prediction[0].argmax()]

        # Predict age
        age_net.setInput(blob)
        age_prediction = age_net.forward()
        age = age_classifications[age_prediction[0].argmax()]

        # Define the text and position
        text = f'{gender}, {age}'
        text_position = (10, 30)  # Top-left corner with margin
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]

        # Add background rectangle for visibility
        cv2.rectangle(img_cp, (text_position[0] - 5, text_position[1] - text_size[1] - 10),
                      (text_position[0] + text_size[0] + 5, text_position[1] + 5), (0, 0, 0), -1)

        # Put the text on top of the background
        cv2.putText(img_cp, text, text_position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)

except Exception as e:
    print("Error processing face:", e)

# Show the result
cv2.imshow('Result', img_cp)
cv2.waitKey(0)
cv2.destroyAllWindows()
