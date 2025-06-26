from flask import Flask, request, render_template, jsonify
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define categories and image size
categ = ["Bishop", "King", "Knight", "Pawn", "Queen", "Rook"]
img_size = 256

# Define the directory to save uploaded files temporarily
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # Ensure the directory exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
# Ensure model22.h5 is in the same directory or provide the full path
model = None
try:
    model_path = 'model22.h5' # Path to your trained model file
    if os.path.exists(model_path):
        model = load_model(model_path)
        print("Model loaded successfully.")
    else:
        print(f"Error: Model file not found at {model_path}")
        # You might want to raise an exception or exit if the model is essential
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

# Define preprocessing functions - These should match your notebook's final pipeline
def blur(image):
    # Median blur requires an odd kernel size
    return cv2.medianBlur(image, ksize=3)

def equalization(image):
    # Apply histogram equalization to grayscale image
    return cv2.equalizeHist(image)

def adaptive_thr(image):
    # Adaptive Gaussian thresholding
    # blockSize must be odd and greater than 1
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=11, C=2)

def sobl(image):
    # Apply Sobel edge detection
    # Sobel typically operates on grayscale or floating-point images
    # If the input 'image' here is the binary output of adaptive_thr,
    # this will detect edges based on where the binary value changes.
    # Ensure input type is suitable for Sobel (e.g., CV_64F)
    sobel_x = cv2.Sobel(image.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    # Magnitude is float64, needs conversion or scaling for display/normalization
    # For normalization by 255, it's likely expected to be in 0-255 range or similar
    # Let's keep it float and normalize later as done in the notebook's training
    return sobel_combined

def preprocess_image(image_path):
    """
    Reads, resizes, converts to grayscale, and applies the specific
    preprocessing pipeline used during training.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read the image from {image_path}")

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to the training size
    img = cv2.resize(img, (img_size, img_size))

    # Apply preprocessing steps in the SAME order as in the notebook's training
    # Notebook sequence: Blur -> Equalize -> Adaptive_thr -> Sobel
    img_processed = blur(img) # blurred grayscale
    img_processed = equalization(img_processed) # equalized blurred grayscale
    img_processed = adaptive_thr(img_processed) # binary thresholded image
    img_processed = sobl(img_processed) # Sobel applied to the binary image

    # Normalize and reshape for model input
    # The notebook normalizes by 255.0 after the Sobel step.
    # Ensure the data type is float before normalization and reshape.
    img_processed = img_processed.astype(np.float32) # Convert to float32
    img_processed = img_processed / 255.0 # Normalize

    # Flatten the image into a single row vector
    # Model expects input shape (batch_size, num_features) -> (1, img_size*img_size)
    img_processed = img_processed.reshape(1, img_size * img_size)

    return img_processed

@app.route('/', methods=['GET'])
def index_get():
    """Render the main page template."""
    return render_template('index.html')

@app.route('/', methods=['POST'])
def index_post():
    """Handle image upload, preprocessing, prediction, and return JSON result."""
    if model is None:
        print("Model not loaded, cannot process prediction.")
        return jsonify({'error': 'AI model is not available.'}), 500

    # Check if the POST request has the file part
    if 'file' not in request.files:
        print("No file part in request.")
        return jsonify({'error': 'No image file uploaded.'}), 400

    file = request.files['file']

    # If the user does not select a file, the browser submits an empty file without a filename.
    if file.filename == '':
        print("Empty filename received.")
        return jsonify({'error': 'No selected file.'}), 400

    if file:
        # Secure the filename
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        temp_upload_path = None # To keep track of the path if saved

        try:
            # Save the file temporarily
            file.save(upload_path)
            temp_upload_path = upload_path # File was saved successfully

            # Preprocess the image
            processed_image = preprocess_image(temp_upload_path)

            # Make prediction
            pred = model.predict(processed_image)

            # Get the predicted class index (index with the highest probability)
            pred_class_index = np.argmax(pred, axis=1)[0]

            # Map the index to the category name
            if 0 <= pred_class_index < len(categ):
                 prediction_name = categ[pred_class_index]
                 print(f"Prediction successful: {prediction_name} (Index: {pred_class_index})")
            else:
                 # This case should ideally not happen if the model has 6 output neurons
                 print(f"Warning: Model predicted invalid index {pred_class_index}. Categories length: {len(categ)}")
                 prediction_name = "Unknown Piece"


            # Return the prediction name as JSON
            return jsonify({'prediction': prediction_name})

        except ValueError as e:
            # Handle specific errors during image processing (like unable to read)
            print(f"Image processing error: {str(e)}")
            return jsonify({'error': f"Image processing error: {str(e)}"}), 400
        except Exception as e:
            # Catch any other unexpected exceptions during the process
            print(f"An unexpected error occurred during prediction: {str(e)}")
            return jsonify({'error': f"An unexpected server error occurred: {str(e)}"}), 500
        finally:
            # Clean up the uploaded file regardless of success or failure
            if temp_upload_path and os.path.exists(temp_upload_path):
                try:
                    os.remove(temp_upload_path)
                    print(f"Cleaned up temporary file: {temp_upload_path}")
                except OSError as e_remove:
                    print(f"Error removing temporary file {temp_upload_path}: {e_remove}")

    # Should not reach here if file exists check works, but as a fallback:
    return jsonify({'error': 'An unknown error occurred during file upload.'}), 500


if __name__ == '__main__':
    # In production, use a WSGI server like Gunicorn or uWSGI
    # e.g., gunicorn --bind 0.0.0.0:5000 app:app
    app.run(debug=True) # debug=True for development