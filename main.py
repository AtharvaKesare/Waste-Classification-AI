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
        
        # Load Stage 2: Waste Classifier (ONNX)
        try:
            logger.info("Loading Stage 2 Model (Custom Waste Classifier via ONNX)...")
            self.waste_model = ort.InferenceSession("Resources/Model/waste_model.onnx")
            self.input_name = self.waste_model.get_inputs()[0].name
        except Exception as e:
            logger.error(f"Failed to load Stage 2 Model: {e}")
            self.waste_model = None
            
        logger.info("WasteClassifierPipeline initialized successfully.")

    def _has_human_skin(self, filepath):
        """
        Detects human presence using HSV skin-tone color analysis.
        Works regardless of angle, lighting, or number of faces.
        Returns True if significant skin tone pixels are found.
        """
        try:
            img = cv2.imread(filepath)
            if img is None:
                return False
            
            # Resize for speed
            img = cv2.resize(img, (300, 300))
            
            # Convert to HSV for robust color matching
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # HSV range for human skin tones (covers all ethnicities)
            # KEY: cap saturation at 150 to EXCLUDE vivid orange objects (carrots, fruit, etc.)
            # Human skin is moderately saturated; carrots/oranges are highly saturated (S > 170)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 150, 255], dtype=np.uint8)
            
            # Catch slightly darker/redder skin tones (also capped at S=150)
            lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
            upper_skin2 = np.array([180, 150, 255], dtype=np.uint8)
            
            mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
            mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
            mask = cv2.bitwise_or(mask1, mask2)
            
            # Apply morphological operations to remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.dilate(mask, kernel, iterations=2)
            
            # Calculate what % of the image is skin-tone
            total_pixels = 300 * 300
            skin_pixels = cv2.countNonZero(mask)
            skin_ratio = skin_pixels / total_pixels
            
            logger.info(f"Stage 1: Skin ratio = {skin_ratio:.3f}")
            
            # If more than 18% of image is skin-tone, flag as human
            # (high enough to avoid false positives from wood/food, low enough to catch selfies)
            return skin_ratio > 0.18
            
        except Exception as e:
            logger.error(f"Stage 1 skin detection error: {e}")
            return False

    def predict(self, filename):
        filepath = os.path.join('static', filename)
        
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return "Error", "File not found", filename

        try:
            # --- STAGE 1: Skin-Tone Human Detection ---
            logger.info("Running Stage 1: Skin-Tone Detection...")
            if self._has_human_skin(filepath):
                logger.warning("Rejected by Stage 1: Human/Skin Detected")
                return "Non-Waste", "Human/Face", filename

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
