# ♻️ EcoScan: Edge-Optimized Waste Classification AI

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Backend-green?style=for-the-badge&logo=flask)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-lightgrey?style=for-the-badge&logo=onnx)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-red?style=for-the-badge&logo=opencv)

**EcoScan** is a full-stack web application that uses a custom-trained Deep Learning model to classify waste into **Recyclable** or **Organic** categories. It is engineered with a focus on **edge-deployment optimization**, utilizing ONNX runtime to bypass strict cloud storage and memory constraints.

### 🚀 [Live Demo (PythonAnywhere)](http://atharvakesare.pythonanywhere.com/)

---

## ✨ Key Features

- **Edge-Optimized Inference:** Converted a heavy TensorFlow/Keras `.hdf5` model to **ONNX**, reducing the model and dependency footprint by over 90% (allowing deployment on a 512MB free-tier cloud host).
- **Out-of-Distribution (OOD) Rejection:** Implemented OpenCV Haar Cascades as a preprocessing stage to detect and reject human faces before they hit the classification model, preventing embarrassing false positives.
- **Continuous Learning Feedback Loop:** Features a built-in SQLite dashboard where users can flag incorrect predictions ("Thumbs Up/Down"). This data is automatically stored in `database.db` to assist in future model retraining and evaluation.
- **Responsive UI/UX:** A clean, modern, and mobile-friendly web interface built with HTML/CSS and Jinja2 templating.

## 🧠 System Architecture

The classification pipeline consists of two stages to ensure robust predictions:
1. **Stage 1 (Anomaly Detection):** The image is passed through an OpenCV Haar Cascade. If a human face is detected, the pipeline immediately halts and returns a "Non-Waste" flag.
2. **Stage 2 (Inference):** The image is preprocessed (resized to 224x224, normalized) and passed to the ONNX Inference Session. The model outputs a probability score, classifying the item as either `Recyclable` or `Organic` along with a confidence percentage.

## 🛠️ Local Installation

Want to run EcoScan on your own machine? Follow these steps:

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   ```

2. **Create a virtual environment (Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: .\venv\Scripts\activate
   ```

3. **Install the lightweight dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask application**
   ```bash
   python index.py
   ```
   The app will be running at `http://127.0.0.1:5000/`.

## 📂 Project Structure
```text
📦 waste_project
 ┣ 📂 Resources/Model       # Contains the edge-optimized waste_model.onnx
 ┣ 📂 static                # CSS, uploaded images, and assets
 ┣ 📂 templates             # HTML Jinja2 templates (index.html, dashboard.html)
 ┣ 📜 index.py              # Main Flask application routing
 ┣ 📜 main.py               # ONNX Inference Pipeline and OpenCV logic
 ┣ 📜 database.py           # SQLite initialization and feedback loop logic
 ┣ 📜 requirements.txt      # Dependency list (TensorFlow removed for ONNX)
 ┗ 📜 README.md             # You are here!
```

## 📈 Future Improvements
- [ ] Expand the dataset to include "Hazardous" and "E-Waste" categories.
- [ ] Implement an admin portal to easily view and download flagged feedback images for retraining.
- [ ] Containerize the application using Docker for easier CI/CD pipelines.
