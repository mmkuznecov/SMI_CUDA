from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="mi_cuda",
    ext_modules=[
        CUDAExtension(
            name="mi_cuda",
            sources=[
                "binding.cpp",
                "knn_kernel.cu",
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            },
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)