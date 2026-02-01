# Disease CNN Classifier Web Application

A Flask-based web application that serves a Convolutional Neural Network (CNN) for classifying images related to various diseases (e.g., lung X-rays, brain scans). Users can upload an image, select a model, and receive a real-time classification result. The backend leverages PyTorch for high-performance inference.

## Features

  * **Web Interface (Flask):** Simple GUI implemented with Flask to handle image uploads and display predictions.
  * **Multi-Model Support:** Capable of loading and switching between different pre-trained CNN models (e.g., `lung_cnn`, `brain_cnn`) based on user selection.
  * **PyTorch Integration:** Utilizes PyTorch for defining, loading, and running the deep learning models.
  * **Image Preprocessing:** Handles image loading, conversion to grayscale, resizing to $256 \times 256$, normalization, and tensor conversion for consistent model inference.
  * **Error Handling:** Includes basic error handling for file uploads and prediction exceptions.

## Project Structure

The project is structured into three main Python components and relies on specific directories for model assets:

```
.
├── GUI.py          # Main Flask application file. Handles routing, file uploads, and calls the predicter.
├── Model.py        # Defines the architecture of the diseaseCNN class (PyTorch nn.Module).
├── Predict.py      # The core prediction logic. Manages model loading, class mapping, and inference.
├── fileUtils.py    # (Assumed) Helper for reading class names from text files.
├── templates/
│   └── gui.html    # The frontend HTML template for image upload and result display.
├── upload/         # Directory created at startup to temporarily store uploaded images.
├── Weights/        # Directory to store trained PyTorch model state dictionaries (`.pth` files).
│   ├── lung_cnn_model.pth
│   └── brain_cnn_model.pth
└── classes/        # Directory to store text files mapping class indices to human-readable names.
    ├── lung_cnn_model.txt
    └── brain_cnn_model.txt
```

## Prerequisites

To run this application, you need Python $3.x$ and the following packages.

```bash
pip install Flask torch torchvision pillow numpy
```

## Installation and Setup

1.  **Clone the repository:**

    ```bash
    git clone [Your Repository URL]
    cd [Your Repository Name]
    ```

2.  **Install dependencies:**
    You can install the required packages using the command in the Prerequisites section.

3.  **Prepare Model Assets:**

      * Create the necessary assets directories: `mkdir Weights classes`
      * Place your trained PyTorch model files (e.g., `lung_cnn_model.pth`) into the **`Weights/`** directory.
      * Place the corresponding text files defining the class names (one class name per line) into the **`classes/`** directory.

## Running the Application

Execute the main Flask application file. The necessary `upload` folder will be created automatically if it doesn't exist.

```bash
python GUI.py
```

The application will start in debug mode on `http://127.0.0.1:5000/`.

## Data Flow

The prediction process, orchestrated by `GUI.py` and handled by the `predicter` class in `Predict.py`, follows these steps:

1.  **Upload:** User submits an image file and selects a model ID via the web form.
2.  **File Handling:** `GUI.py` saves the incoming file to the local `./upload` directory.
3.  **Inference Call:** The `predicter.predict()` method is called with the saved image path and the model ID.
4.  **Preprocessing:** The image is opened, converted to grayscale, resized to $256 \times 256$, normalized (division by $255.0$), and converted into a PyTorch tensor with the shape $[1, 1, 256, 256]$ (Batch, Channel, Height, Width).
5.  **Prediction:** The selected `diseaseCNN` model runs a forward pass on the tensor.
6.  **Result Mapping:** The output tensor's highest value index is found ($\text{torch.max}$) and mapped back to the human-readable class name using the class lists loaded during initialization.
7.  **Response:** The final prediction string is returned to the client.

