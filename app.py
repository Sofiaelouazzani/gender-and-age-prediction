from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import cv2
import os
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Load models
face_pbtxt = "models/opencv_face_detector.pbtxt"
face_pb = "models/opencv_face_detector_uint8.pb"
age_prototxt = "models/age_deploy.prototxt"
age_model = "models/age_net.caffemodel"
gender_prototxt = "models/gender_deploy.prototxt"
gender_model = "models/gender_net.caffemodel"
MODEL_MEAN_VALUES = [104, 117, 123]

# Load networks
face_net = cv2.dnn.readNet(face_pb, face_pbtxt)
age_net = cv2.dnn.readNet(age_model, age_prototxt)
gender_net = cv2.dnn.readNet(gender_model, gender_prototxt)

age_classifications = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_classifications = ['Male', 'Female']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    if file:
        # Save and read the image
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        image = cv2.imread(file_path)
        os.remove(file_path)  # Remove the image after use

        # Resize the image for processing
        image = cv2.resize(image, (720, 640))

        # Get image dimensions
        img_h, img_w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), MODEL_MEAN_VALUES, True, False)
        
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

                # Add larger padding to the bounding box
                padding = 40  # Increase padding significantly to avoid cropping
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(img_w, x2 + padding)
                y2 = min(img_h, y2 + padding)
                face_bounds.append([x1, y1, x2, y2])

        if not face_bounds:
            return render_template('result.html', result={'error': 'No faces detected'})

        for face_bound in face_bounds:
            face_img = image[max(0, face_bound[1]): min(face_bound[3], img_h),
                             max(0, face_bound[0]): min(face_bound[2], img_w)]
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, True)

            # Gender Prediction
            gender_net.setInput(blob)
            gender_prediction = gender_net.forward()
            gender = gender_classifications[gender_prediction[0].argmax()]

            # Age Prediction
            age_net.setInput(blob)
            age_prediction = age_net.forward()
            age = age_classifications[age_prediction[0].argmax()]

            # Draw the bounding box and label on the image with larger padding
            label = f"{gender}, {age}"
            color = (0, 255, 0)  # Green color for the bounding box and label
            cv2.rectangle(image, (face_bound[0], face_bound[1]), (face_bound[2], face_bound[3]), color, 2)

            # Draw the label below the bounding box
            label_y_position = min(face_bound[3] + 30, img_h - 10)  # Adjust to be below the bounding box
            cv2.putText(image, label, (face_bound[0], label_y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Prepare result for rendering
            result = {'gender': gender, 'age': age}

        # Save the annotated image and pass the path to the result template
        result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + file.filename)
        cv2.imwrite(result_image_path, image)

        return render_template('result.html', result=result, image_path=result_image_path)

    return redirect(url_for('index'))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if __name__ == '__main__':
    app.run(debug=True)
