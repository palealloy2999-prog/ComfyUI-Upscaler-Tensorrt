import torch
import time
from trt_utilities import Engine
from .. import IMAGE_DIM_MIN, IMAGE_DIM_OPT, IMAGE_DIM_MAX

def export_trt(trt_path=None, onnx_path=None, use_fp16=True):
    if trt_path is None:
        trt_path = input("Enter the path to save the TensorRT engine (e.g ./realesrgan.engine): ")
    if onnx_path is None:
        onnx_path = input("Enter the path to the ONNX model (e.g ./realesrgan.onnx): ")

    engine = Engine(trt_path)

    torch.cuda.empty_cache()

    s = time.time()
    ret = engine.build(
        onnx_path,
        use_fp16,
        enable_preview=True,
        input_profile = [
            {"input": [
                (1, 3, IMAGE_DIM_MIN, IMAGE_DIM_MIN),
                (1, 3, IMAGE_DIM_OPT, IMAGE_DIM_OPT),
                (1, 3, IMAGE_DIM_MAX, IMAGE_DIM_MAX),
            ]}
        ],
    )
    e = time.time()
    print(f"Time taken to build: {(e-s)} seconds")

    return ret

export_trt()
