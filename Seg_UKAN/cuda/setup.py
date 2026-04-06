import os
from setuptools import find_packages, setup

# 1. Grab the Conda environment path
conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    os.environ["CUDA_HOME"] = conda_prefix
    print(f"CUDA_HOME environment variable set to: {os.environ['CUDA_HOME']}")

# 2. Import PyTorch tools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch.utils.cpp_extension

# 3. NUCLEAR OPTION: Forcibly overwrite PyTorch's cached internal variable!
if conda_prefix:
    torch.utils.cpp_extension.CUDA_HOME = conda_prefix
    print(
        f"PyTorch internal CUDA_HOME forcibly overridden to: {torch.utils.cpp_extension.CUDA_HOME}"
    )

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
