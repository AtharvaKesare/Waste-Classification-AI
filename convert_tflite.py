import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import os

def convert_models():
    print("Converting MobileNetV2 to TFLite...")
    # Load MobileNetV2
    mobilenet = MobileNetV2(weights='imagenet')
    converter = tf.lite.TFLiteConverter.from_keras_model(mobilenet)
    tflite_mobilenet = converter.convert()
    
    with open('Resources/Model/mobilenet_v2.tflite', 'wb') as f:
        f.write(tflite_mobilenet)
    print("MobileNetV2 converted successfully.")

    print("Converting Custom Waste Model to TFLite...")
    custom_model = tf.keras.models.load_model('Resources/Model/final_model_weights.hdf5')
    converter2 = tf.lite.TFLiteConverter.from_keras_model(custom_model)
    tflite_custom = converter2.convert()
    
    with open('Resources/Model/waste_model.tflite', 'wb') as f:
        f.write(tflite_custom)
    print("Custom model converted successfully.")

if __name__ == '__main__':
    os.makedirs('Resources/Model', exist_ok=True)
    convert_models()
