import tensorflow as tf
import tf2onnx
import os

def convert_to_onnx():
    print("Loading Custom Waste Model...")
    model = tf.keras.models.load_model('Resources/Model/final_model_weights.hdf5')
    
    print("Converting Custom Waste Model to ONNX...")
    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input_image"),)
    output_path = "Resources/Model/waste_model.onnx"
    
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
    print(f"ONNX model saved successfully to {output_path}")

if __name__ == '__main__':
    convert_to_onnx()
