import tensorrt as trt
TRT_LOGGER = trt.Logger()
def get_engine1(engine_path):
    # If a serialized engine exists, use it instead of building an engine.
    print("Reading engine from file {}".format(engine_path))
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

if __name__ == '__main__':
    engine_file_path = "F_Let_fp16.trt"
    #onnx engine setter
    engines = get_engine1(engine_file_path)
    for binding in engines:
        size = trt.volume(engines.get_binding_shape(binding)) * 1
        dims = engines.get_binding_shape(binding)
        print('size=', size)
        print('dims=', dims)
        print('binding=', binding)
        print("input =", engines.binding_is_input(binding))
        dtype = trt.nptype(engines.get_binding_dtype(binding))
        print("dtype =", dtype)