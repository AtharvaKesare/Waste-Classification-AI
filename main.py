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
    Uses a custom trained binary classifier for Organic vs Recycle (ONNX Runtime).
    """
    def __init__(self):
        logger.info("Initializing WasteClassifierPipeline...")
        
        # Load Waste Classifier (ONNX)
        try:
            logger.info("Loading Waste Classifier (ONNX)...")
            self.waste_model = ort.InferenceSession("Resources/Model/waste_model.onnx")
            self.input_name = self.waste_model.get_inputs()[0].name
        except Exception as e:
            logger.error(f"Failed to load Waste Classifier: {e}")
            self.waste_model = None
            
        logger.info("WasteClassifierPipeline initialized successfully.")

    def predict(self, filename):
        filepath = os.path.join('static', filename)
        
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return "Error", "File not found", filename

        try:
            # --- Waste Classification (ONNX) ---
            if self.waste_model:
                logger.info("Running Waste Classification (ONNX)...")
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
                    logger.info(f"Model uncertain. Confidence: {confidence_percent}%")
                    return "Uncertain", str(confidence_percent), filename

                logger.info(f"Prediction: {label} ({confidence_percent}%)")
                return str(label), str(confidence_percent), filename
            else:
                return "Error", "Model not loaded", filename
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "Error", str(e), filename

# Instantiate pipeline globally so model loads only once at startup
pipeline = WasteClassifierPipeline()

def getPrediction(filename):
    return pipeline.predict(filename)
