import numpy as np
import logging
import cv2
import os
import onnxruntime as ort

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WasteClassifierPipeline:
    """
    A robust ML pipeline optimized for edge deployments (PythonAnywhere free tier).
    Stage 1: Explicit Face Detection (Haar Cascades)
    Stage 2: Custom trained binary classifier for Organic vs Recycle (ONNX Runtime).
    """
    def __init__(self):
        logger.info("Initializing Edge-Optimized WasteClassifierPipeline...")
        
        # Load Stage 1: Face Detector
        try:
            logger.info("Loading Stage 1 Model (Haar Cascade)...")
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        except Exception as e:
            logger.error(f"Failed to load Stage 1 Model: {e}")
            self.face_cascade = None

        # Load Stage 2: Waste Classifier (ONNX)
        try:
            logger.info("Loading Stage 2 Model (Custom Waste Classifier via ONNX)...")
            self.waste_model = ort.InferenceSession("Resources/Model/waste_model.onnx")
            self.input_name = self.waste_model.get_inputs()[0].name
        except Exception as e:
            logger.error(f"Failed to load Stage 2 Model: {e}")
            self.waste_model = None
            
        logger.info("WasteClassifierPipeline initialized successfully.")

    def predict(self, filename):
        filepath = os.path.join('static', filename)
        
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return "Error", "File not found", filename

        try:
            # --- STAGE 1: Face Detection (OpenCV) ---
            if self.face_cascade is not None:
                logger.info("Running Stage 1: Face Detection...")
                cv_img = cv2.imread(filepath)
                if cv_img is not None:
                    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                    # Use absolute minimum neighbors to catch any possible face
                    faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(30, 30))
                    if len(faces) > 0:
                        logger.warning("Rejected by Stage 1: Human Face Detected")
                        return "Non-Waste", "Human Face", filename

            # --- STAGE 2: Waste Classification ---
            if self.waste_model:
                logger.info("Running Stage 2: Waste Classification (ONNX)...")
                # Custom image loading without keras
                img = cv2.imread(filepath)
                if img is None:
                    return "Error", "Invalid image", filename
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                img = img.astype('float32') / 255.0
                img = np.expand_dims(img, axis=0)

                prediction = self.waste_model.run(None, {self.input_name: img})
                prob = float(prediction[0][0][0])

                if prob > 0.5:
                    label = "Recycle"
                    confidence = prob
                else:
                    label = "Organic"
                    confidence = 1 - prob

                confidence_percent = round(confidence * 100, 2)
                
                if confidence_percent < 65.0:
                    logger.info(f"Stage 2 uncertain. Confidence: {confidence_percent}%")
                    return "Uncertain", str(confidence_percent), filename

                logger.info(f"Stage 2 success: {label} ({confidence_percent}%)")
                return str(label), str(confidence_percent), filename
            else:
                return "Error", "Model not loaded", filename
            
        except Exception as e:
            logger.error(f"Prediction error pipeline: {e}")
            return "Error", str(e), filename

# Instantiate pipeline globally so models load only once at startup
pipeline = WasteClassifierPipeline()

def getPrediction(filename):
    return pipeline.predict(filename)
