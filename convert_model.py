import tensorflow as tf
import tf2onnx

# Load the TensorFlow model
model_path = "face_recognition_nn_softmax_model.h5"
model = tf.keras.models.load_model(model_path)

# Ensure the model's output names are unique
model.output_names = ['output']

# Convert the TensorFlow model to ONNX format
onnx_model_path = "face_recognition_nn_softmax_model.onnx"
spec = (tf.TensorSpec((None, 512), tf.float32, name="input"),)  # Assuming embeddings are 512-dimensional
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

# Save the ONNX model
with open(onnx_model_path, "wb") as f:
    f.write(model_proto.SerializeToString())

print(f"Model converted to ONNX format and saved to {onnx_model_path}")
