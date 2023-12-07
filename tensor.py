from tensorflow.python.compiler.tensorrt import trt_convert as trt
import tensorflow as tf
print(tf.__version__)
# Conversion Parameters 
conversion_params = trt.TrtConversionParams(
    precision_mode=trt.TrtPrecisionMode.FP32) # or FP16

converter = trt.TrtGraphConverterV2(
    input_saved_model_dir='saved_model.pb',
    conversion_params=conversion_params)

# Converter method used to partition and optimize TensorRT compatible segments
converter.convert()

# Optionally, build TensorRT engines before deployment to save time at runtime
#converter.build(input_fn=my_input_fn)

# Save the model to the disk 
converter.save('saved_model_tensor.pb')