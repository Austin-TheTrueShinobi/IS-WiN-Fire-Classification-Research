import torch
import tensorflow as tf
from tensorflow.python.framework import tensor_util

# Load the TensorFlow model
tf_model_path = 'path/to/your/model.pb'
loaded = tf.saved_model.load(tf_model_path)

# Convert TensorFlow model to PyTorch
def tf_to_torch(tf_tensor):
    return torch.tensor(tensor_util.MakeNdarray(tf_tensor))

torch_model = torch.jit.trace(tf_to_torch, loaded.signatures["serving_default"].inputs[0])

# Save the PyTorch model
torch.save(torch_model, 'path/to/your/model.pt')
