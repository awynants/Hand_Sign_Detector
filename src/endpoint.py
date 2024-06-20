from flask import Flask, request, jsonify
import cv2
import numpy as np
from detector import HandSignDetector  # Make sure the path is correct

app = Flask(__name__)
detector = HandSignDetector()  # Initialize the HandSignDetector

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure an image is provided in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    # Read the image file from the request
    image_file = request.files['image']
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Process the image to detect hand sign
    annotated_image = detector.detect_hand_sign(image)

    # Return the detected sign as JSON response
    return jsonify({'detected_sign': detector.get_detected_sign()}), 200

if __name__ == '__main__':
    app.run(debug=True)

if __name__ == '__main__':
    app.run(debug=True)
