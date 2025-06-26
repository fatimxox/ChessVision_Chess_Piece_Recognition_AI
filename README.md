
## Prerequisites

To run this project locally, you need to have Python installed on your system.

*   **Python:** Version 3.6 or higher is recommended.
*   **Python Libraries:** Install the required libraries using pip. It's highly recommended to use a virtual environment.
    ```bash
    pip install Flask tensorflow tensorflow-keras numpy opencv-python werkzeug Pillow matplotlib seaborn scikit-learn
    ```
    *   `Flask`: To build the web application.
    *   `tensorflow`, `tensorflow-keras`: For building and loading the deep learning model.
    *   `numpy`: For numerical operations.
    *   `opencv-python` (cv2): For image processing.
    *   `werkzeug`: Used by Flask for file uploads (`secure_filename`).
    *   `Pillow`: For image handling (used by Matplotlib and others).
    *   `matplotlib`, `seaborn`, `scikit-learn`: For data exploration and evaluation in the notebook.
*   **Dataset:** The `Chessman-image-dataset` organized into a specific directory structure with subfolders for each piece type is required. This dataset is commonly available on platforms like Kaggle. The notebook expects the path `/kaggle/input/chessman-image-dataset/Chessman-image-dataset/Chess`. You should place the `Chessman-image-dataset` folder in the project's root directory for local use and update the path in the notebook and `app.py` if necessary.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ChessVision_Chess_Piece_Recognition_AI.git
    cd ChessVision_Chess_Piece_Recognition_AI
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:** With your virtual environment activated, install the required libraries:
    ```bash
    pip install -r requirements.txt # If you create a requirements.txt
    # OR manually install if you don't have requirements.txt
    pip install Flask tensorflow tensorflow-keras numpy opencv-python werkzeug Pillow matplotlib seaborn scikit-learn
    ```
    *(You can generate a `requirements.txt` file after installing dependencies using `pip freeze > requirements.txt`)*

4.  **Obtain the dataset:** Download the `Chessman-image-dataset` and place it in the project's root directory. Ensure the path in the notebook and `app.py` matches the location of the `/Chess` subfolder (e.g., `Chessman-image-dataset/Chess`).
5.  **Run the Notebook:** Execute all cells in `computer-vision copy.ipynb`. This notebook loads and preprocesses the data, trains a simple neural network, evaluates its performance, and saves the trained model as `model22.h5`. **Ensure the `model22.h5` file is saved in the same directory as `app.py`**.

## How it Works

1.  **Data Preparation & Preprocessing (Notebook):** The Jupyter Notebook (`computer-vision copy.ipynb`) loads the dataset, converts images to grayscale, resizes them, and applies a sequence of preprocessing steps: blurring, histogram equalization, adaptive thresholding, and Sobel edge detection. It then flattens the processed images into 1D vectors and normalizes them for input into the neural network.
2.  **Model Training (Notebook):** A simple Sequential neural network model with Dense layers is defined, compiled, and trained on the preprocessed image data. The training aims to minimize sparse categorical crossentropy loss. The trained model object is saved to `model22.h5`.
3.  **Model Loading (Flask App):** The `app.py` script loads the pre-trained Keras model (`model22.h5`) into memory when the Flask application starts. It includes basic error handling for model loading failure.
4.  **Preprocessing Function (`preprocess_image`):** This function in `app.py` replicates the exact sequence of preprocessing steps performed in the notebook's training pipeline. It takes the path to an uploaded image, reads it, converts to grayscale, resizes, applies the blur, equalization, adaptive thresholding, and Sobel edge detection, then flattens and normalizes the image data. This ensures that images presented to the loaded model have the same format and features as the images it was trained on.
5.  **Web Interface (HTML + JavaScript):**
    *   `index.html` provides the user interface with a drag-and-drop area or file input, an image preview area, a loading spinner, and a display area for the prediction result.
    *   JavaScript embedded in `index.html` handles frontend interactions:
        *   Implementing drag-and-drop functionality for image files.
        *   Handling file selection via the hidden input.
        *   Displaying a preview of the selected image.
        *   Showing a loading spinner while the image is being processed.
        *   Sending the image file to the Flask backend using `fetch` (AJAX).
        *   Receiving the JSON response containing the predicted piece name.
        *   Updating the `resultContainer` to display the prediction, including an appropriate chess piece icon and some basic information.
        *   Showing a "New Upload" button after a result is shown to clear the interface and allow a new upload.
        *   Handling basic frontend validation (file type, size).
6.  **Prediction Endpoint (`/` with POST):**
    *   The Flask `index_post` function handles POST requests to the root URL (`/`).
    *   It checks if a file is included in the request and if a file was actually selected.
    *   The uploaded file is securely saved to the `uploads` directory (created by `app.py` if it doesn't exist).
    *   The `preprocess_image` function is called to prepare the image data.
    *   The loaded Keras model's `predict` method is called with the preprocessed image data.
    *   The `np.argmax` function is used to find the index of the class with the highest predicted probability.
    *   The index is mapped to the corresponding category name from the `categ` list defined in `app.py`.
    *   A JSON response is returned to the frontend containing the predicted chess piece name.
    *   Robust error handling is included throughout the process, including cleanup of the temporarily uploaded file.

## Model Performance (from Notebook)

The notebook `computer-vision copy.ipynb` includes evaluation steps for the trained model. The `print(classification_report(y , pred))` output provides metrics like Precision, Recall, F1-score, and Support for each class, as well as overall accuracy, macro average, and weighted average. This report indicates how well the model performed on the data it was trained on. *Note: It's common practice to evaluate on a separate test set derived from the original data split to get a better estimate of performance on unseen data.*

## Web Application Usage

1.  **Start the Flask server:** Follow the installation steps and run `python app.py` in your terminal. Ensure your virtual environment is activated and you are in the project's root directory.
2.  **Open in browser:** Navigate to `http://127.0.0.1:5000/` (or the address specified by Flask) in your web browser.
3.  **Upload Image:** Drag and drop an image of a single chess piece onto the designated area, or click the area to select a file using your file browser.
4.  **Processing:** The application will show a preview of your image and a loading spinner while the image is sent to the server, processed, and the model makes a prediction.
5.  **View Results:** Once the analysis is complete, the loading spinner will disappear, and the predicted chess piece will be displayed, along with an icon and basic information about the piece.
6.  **New Upload:** Click the "New Upload" button that appears after a result is shown to clear the interface and upload another image.
