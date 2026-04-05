import os
from setuptools import find_packages, setup

# 1. Set CUDA_HOME dynamically FIRST
conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    os.environ["CUDA_HOME"] = conda_prefix
    print(f"CUDA_HOME automatically set to: {os.environ['CUDA_HOME']}")
else:
    print(
        "Warning: CONDA_PREFIX not found. Are you sure your conda environment is activated?"
    )

# 2. NOW import PyTorch cpp_extension so it sees the variable we just set
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="FasterOps",
    packages=find_packages(),
    version="0.0.0",
    ext_modules=[
        CUDAExtension(
            "faster_ops",  # operator name
            [
                "./cpp/faster.cpp",
                "./cpp/faster_cuda.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
